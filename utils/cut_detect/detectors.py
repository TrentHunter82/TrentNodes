"""
Shot-boundary detection backends for TrentNodes.

Three detectors behind one result type, strongest first:

- omnishotcut  OmniShotCut (UVA Computer Vision Lab, 2026; MIT). A
               shot-query video Transformer that predicts shot ranges
               together with the *kind* of each boundary, so a dissolve,
               wipe, fade or whip-pan comes back labelled instead of
               being guessed at from a difference spike. Range F1 0.883
               vs 0.814 for both TransNetV2 and AutoShot on the paper's
               benchmark. Needs the `omnishotcut` package and CUDA.
- transnetv2   TransNetV2 (Soucek & Lokoc). The long-standing baseline;
               weights ship inside the transnetv2-pytorch wheel, so it
               works offline. Hard cuts only - every boundary comes back
               as "hard cut" because the model has no transition classes.
- classic      The frame-difference detector in utils/scene_detect.py.
               No model, no download, no GPU. A safety net, not a peer.

`auto` walks that list and takes the first backend that loads, recording
why it skipped the others in ShotList.notes.

Result model
------------
A transition effect (dissolve, wipe, fade...) is a *boundary*, not a
shot: OmniShotCut emits it as its own span, and this module folds that
span into the following Shot as its `entry` kind plus the frame range in
`transition_frames`. So len(shots) is the number of real shots, and
every shot after the first begins at a cut.

OmniShotCut's inter-shot label New_Start means "first clip of the
sequence". It runs on overlapping 100-frame windows, so each window's
first span carries it; mid-video it means "no relation to the previous
shot could be read", which is still an instantaneous boundary. Those map
to entry "hard cut", and the untouched model labels stay in
Shot.raw_labels so nothing is lost: `intra`/`inter` are the shot's own,
and a shot entered through a folded transition also carries that span's
labels as `transition_intra`/`transition_inter`.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DETECTOR_CHOICES = ("auto", "omnishotcut", "transnetv2", "classic")

# OmniShotCut resizes every input to this before inference; matching it
# up front keeps a long clip from ever being materialized at full size.
OSC_PROCESS_SIZE = (128, 96)  # width, height
OSC_HF_REPO = "uva-cv-lab/OmniShotCut"
# engine.py asserts overlap < the 100-frame inference window. Stay well
# under it: past roughly half the window the extra passes cost time
# without changing the boundaries.
OSC_MAX_OVERLAP = 90

# TransNetV2's fixed input geometry.
TNV2_SIZE = (48, 27)  # width, height

# OmniShotCut intra-shot class -> the plain word used for a boundary of
# that kind. "General" is a normal shot span, never a transition.
TRANSITION_NAMES = {
    "Dissolve": "dissolve",
    "Wipes": "wipe",
    "Push": "push",
    "Slide": "slide",
    "Zoom": "zoom",
    "Fade": "fade",
    "Doorway": "doorway",
}

_osc_model = None
_tnv2_model = None


@dataclass
class Shot:
    """One real shot. `entry` is how it begins, not how it ends."""

    index: int              # 1-based
    start_frame: int        # inclusive
    end_frame: int          # exclusive
    fps: float
    entry: str = "start"    # start | hard cut | sudden jump | dissolve | ...
    # Frame span of the transition effect that opens this shot, when the
    # detector resolved one. None for hard cuts and for shot 1.
    transition_frames: Optional[Tuple[int, int]] = None
    raw_labels: Dict[str, str] = field(default_factory=dict)

    @property
    def start_time(self) -> float:
        return round(self.start_frame / self.fps, 3)

    @property
    def end_time(self) -> float:
        return round(self.end_frame / self.fps, 3)

    @property
    def duration(self) -> float:
        return round((self.end_frame - self.start_frame) / self.fps, 3)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def is_hard_cut(self) -> bool:
        """A cut with no transition effect across it."""
        return self.entry in ("hard cut", "sudden jump")


@dataclass
class ShotList:
    shots: List[Shot] = field(default_factory=list)
    fps: float = 24.0
    total_frames: int = 0
    detector: str = "classic"
    notes: List[str] = field(default_factory=list)
    # What the caller asked for ("auto" when the cascade chose), and
    # whether a stronger detector was skipped to get here. A note string
    # only reaches the report; a field reaches the node's outputs, which
    # is what a workflow can act on.
    requested: str = "auto"
    fallback: bool = False

    @property
    def duration(self) -> float:
        return round(self.total_frames / self.fps, 3) if self.fps else 0.0

    @property
    def cut_times(self) -> List[float]:
        """Start time of every shot, including 0.0 for the first."""
        return [s.start_time for s in self.shots]

    @property
    def hard_cut_times(self) -> List[float]:
        """Start times of shots entered through an instantaneous cut."""
        return [s.start_time for s in self.shots if s.is_hard_cut]

    @property
    def num_cuts(self) -> int:
        """Boundaries, i.e. one fewer than the number of shots."""
        return max(0, len(self.shots) - 1)


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------

def _to_uint8_frames(
    images: torch.Tensor, width: int, height: int, chunk: int = 256
) -> np.ndarray:
    """
    (B, H, W, C) float [0,1] -> (B, height, width, 3) uint8 numpy.

    Resizes on whatever device the tensor is already on, in chunks, so a
    long clip never needs a full-resolution uint8 copy. Area resampling
    matches the cv2.INTER_AREA both detectors apply internally.
    """
    if images.dim() != 4:
        raise ValueError(
            f"expected an (B, H, W, C) image batch, got {tuple(images.shape)}"
        )

    out = np.empty((images.shape[0], height, width, 3), dtype=np.uint8)
    for i in range(0, images.shape[0], chunk):
        block = images[i:i + chunk, :, :, :3].float()
        block = block.permute(0, 3, 1, 2)
        if block.shape[-2:] != (height, width):
            block = F.interpolate(block, size=(height, width), mode="area")
        block = block.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        out[i:i + block.shape[0]] = block.permute(0, 2, 3, 1).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# OmniShotCut
# ---------------------------------------------------------------------------

def _load_omnishotcut():
    """Load and cache the OmniShotCut model. Raises with an install hint."""
    global _osc_model
    if _osc_model is not None:
        return _osc_model

    try:
        import omnishotcut
    except ImportError as exc:
        raise RuntimeError(
            "the 'omnishotcut' package is not installed. Install it with:\n"
            "  pip install --no-deps "
            "git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git\n"
            "Use --no-deps: its requirements.txt pins transformers==4.57.3, "
            "which would downgrade a modern ComfyUI environment. The "
            "package itself never imports transformers."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "OmniShotCut hardcodes CUDA and cannot run on CPU. Use the "
            "transnetv2 detector instead."
        )

    _osc_model = omnishotcut.load(OSC_HF_REPO)
    return _osc_model


def _clamp_overlap(overlap: int) -> int:
    """
    Hold overlap inside the model's inference window.

    omnishotcut/engine.py:72 asserts 0 <= overlap < chunk_size, where
    chunk_size is the model's max_process_window_length (100 frames).
    A typo in a widget should not surface as an AssertionError raised
    from inside a third-party package.
    """
    return int(min(max(int(overlap), 0), OSC_MAX_OVERLAP))


def _detect_omnishotcut(
    frames: np.ndarray, fps: float, overlap: int, min_shot_frames: int
) -> ShotList:
    model = _load_omnishotcut()
    ranges, intra_labels, inter_labels = model.inference(
        frames, mode="default", overlap=_clamp_overlap(overlap)
    )

    spans = [
        (int(a), int(b), str(intra), str(inter))
        for (a, b), intra, inter in zip(ranges, intra_labels, inter_labels)
    ]
    shots = _spans_to_shots(spans, fps, len(frames))
    shots = _merge_runts(shots, min_shot_frames)
    return ShotList(
        shots=shots, fps=fps, total_frames=len(frames), detector="omnishotcut"
    )


def _spans_to_shots(
    spans: List[Tuple[int, int, str, str]], fps: float, total_frames: int
) -> List[Shot]:
    """
    Fold OmniShotCut spans into real shots.

    A span whose intra label is a transition effect is a boundary: it
    sets the next shot's entry kind and is not a shot of its own. A
    leading transition (a fade up from black) has no previous shot to
    bound, so it stays part of shot 1.
    """
    shots: List[Shot] = []
    pending_entry = "start"
    pending_transition: Optional[Tuple[int, int]] = None
    pending_raw: Dict[str, str] = {}

    for start, end, intra, inter in spans:
        if end <= start:
            continue

        transition = TRANSITION_NAMES.get(intra)
        if transition and shots:
            # Between two shots: remember it, then hand it to the next one.
            pending_entry = transition
            pending_transition = (start, end)
            pending_raw = {"intra": intra, "inter": inter}
            continue

        if transition and not shots:
            # Opens the clip; it belongs to shot 1 rather than bounding it.
            pending_entry = "start"
            pending_transition = (start, end)
            pending_raw = {"intra": intra, "inter": inter}
            continue

        if not shots:
            entry = "start"
        elif pending_transition is not None:
            entry = pending_entry
        elif inter == "Sudden_Jump":
            entry = "sudden jump"
        else:
            # Hard_Cut, or a mid-video New_Start, which is a window-edge
            # artifact over an instantaneous boundary. See module docs.
            entry = "hard cut"

        # Both label sets matter and they answer different questions:
        # the shot's own say what the model made of the shot, the
        # transition span's say what kind of boundary opened it.
        # "pending_raw or {...}" kept only the transition's and dropped
        # the shot's, which made this module's "nothing is lost" untrue.
        raw = {"intra": intra, "inter": inter}
        if pending_raw:
            raw["transition_intra"] = pending_raw["intra"]
            raw["transition_inter"] = pending_raw["inter"]
        # A transition's frames sit before the new shot's clean start, so
        # the shot begins where the effect ends.
        shot_start = (
            pending_transition[0] if (pending_transition and not shots)
            else start
        )
        shots.append(Shot(
            index=len(shots) + 1,
            start_frame=shot_start,
            end_frame=end,
            fps=fps,
            entry=entry,
            transition_frames=pending_transition if shots else None,
            raw_labels=raw,
        ))
        pending_entry = "start"
        pending_transition = None
        pending_raw = {}

    if not shots:
        return [Shot(index=1, start_frame=0, end_frame=total_frames, fps=fps)]

    # A trailing transition span leaves the last shot short of the end.
    shots[-1].end_frame = max(shots[-1].end_frame, total_frames)
    shots[0].start_frame = 0
    return shots


# ---------------------------------------------------------------------------
# TransNetV2
# ---------------------------------------------------------------------------

def _load_transnetv2():
    """
    Load and cache TransNetV2 on CPU.

    CPU is deliberate, not a fallback: on torch 2.11 + cu130 the model's
    dilated 3D convolutions dispatch to aten::slow_conv3d_forward, which
    has no CUDA kernel and raises NotImplementedError. On CPU it runs at
    roughly 300 frames/second, fast enough that the GPU is not missed.
    """
    global _tnv2_model
    if _tnv2_model is not None:
        return _tnv2_model

    try:
        from transnetv2_pytorch import TransNetV2
    except ImportError as exc:
        raise RuntimeError(
            "the 'transnetv2-pytorch' package is not installed. Install it "
            "with: pip install transnetv2-pytorch"
        ) from exc

    model = TransNetV2(device="cpu")
    model.eval()
    _tnv2_model = model
    return _tnv2_model


def _detect_transnetv2(
    frames: np.ndarray, fps: float, threshold: float, min_shot_frames: int
) -> ShotList:
    model = _load_transnetv2()
    with torch.no_grad():
        single_frame_pred, _ = model.predict_frames(
            torch.from_numpy(frames), quiet=True
        )
    predictions = np.asarray(
        single_frame_pred.cpu() if torch.is_tensor(single_frame_pred)
        else single_frame_pred
    ).reshape(-1)

    # predictions_to_scenes returns inclusive [start, end] pairs. It is a
    # staticmethod, so reaching it through the loaded instance keeps the
    # only import of the package inside _load_transnetv2, whose
    # ImportError handler carries the pip hint. A module-level import
    # here ran first and made that hint unreachable.
    scenes = model.predictions_to_scenes(predictions, threshold=threshold)

    shots = []
    for start, end in np.asarray(scenes).reshape(-1, 2).tolist():
        shots.append(Shot(
            index=len(shots) + 1,
            start_frame=int(start),
            end_frame=int(end) + 1,
            fps=fps,
            entry="start" if not shots else "hard cut",
            raw_labels={
                "cut_confidence": f"{predictions[int(start)]:.3f}"
            } if shots else {},
        ))

    if not shots:
        shots = [Shot(index=1, start_frame=0, end_frame=len(frames), fps=fps)]
    shots[-1].end_frame = max(shots[-1].end_frame, len(frames))
    shots = _merge_runts(shots, min_shot_frames)
    return ShotList(
        shots=shots, fps=fps, total_frames=len(frames), detector="transnetv2",
        notes=[
            "TransNetV2 reports hard cuts only; dissolves, wipes and fades "
            "are not labelled. Use omnishotcut to type them."
        ],
    )


# ---------------------------------------------------------------------------
# Classic frame-difference fallback
# ---------------------------------------------------------------------------

def _detect_classic(
    images: torch.Tensor, fps: float, sensitivity: float, min_shot_frames: int
) -> ShotList:
    from ..scene_detect import (
        compute_hybrid_differences,
        detect_boundaries,
        downsample_for_detection,
    )

    total = int(images.shape[0])
    if total < 2:
        return ShotList(
            shots=[Shot(index=1, start_frame=0, end_frame=total, fps=fps)],
            fps=fps, total_frames=total, detector="classic",
        )

    detection = downsample_for_detection(images, max_side=384)
    differences = compute_hybrid_differences(detection, threshold=sensitivity)
    boundaries = detect_boundaries(
        differences, sensitivity, max(2, min_shot_frames), adaptive=True
    )

    shots = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else total
        if end <= start:
            continue
        shots.append(Shot(
            index=len(shots) + 1, start_frame=start, end_frame=end, fps=fps,
            entry="start" if not shots else "hard cut",
        ))

    if not shots:
        shots = [Shot(index=1, start_frame=0, end_frame=total, fps=fps)]
    # detect_boundaries only suppresses cuts that land closer together
    # than min_shot_frames; it never length-checks the shot that runs to
    # the end of the clip. Folding runts here is what makes the widget
    # mean the same thing on this path as on the other two.
    shots = _merge_runts(shots, min_shot_frames)
    return ShotList(
        shots=shots, fps=fps, total_frames=total, detector="classic",
        notes=[
            "The classic detector thresholds frame differences. It fires "
            "on fast camera moves and misses slow dissolves; treat its "
            "cut list as a hint, not a fact."
        ],
    )


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _merge_runts(shots: List[Shot], min_shot_frames: int) -> List[Shot]:
    """
    Absorb shots shorter than `min_shot_frames` into the previous one.

    A runt is nearly always a detector wobble around one real cut, and a
    spurious [Shot N] costs more in an H3 prompt than a missed one. Shot
    1 absorbs forward instead, since it has nothing to its left.
    """
    if min_shot_frames <= 1 or len(shots) < 2:
        return shots

    kept: List[Shot] = []
    for shot in shots:
        if shot.frame_count >= min_shot_frames or not kept:
            kept.append(shot)
        else:
            kept[-1].end_frame = shot.end_frame

    # Shot 1 being a runt means the real opening cut is the one after it.
    while len(kept) > 1 and kept[0].frame_count < min_shot_frames:
        kept[1].start_frame = kept[0].start_frame
        kept[1].entry = "start"
        kept[1].transition_frames = None
        kept.pop(0)

    for i, shot in enumerate(kept, start=1):
        shot.index = i
    return kept


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def detect_shots(
    images: torch.Tensor,
    fps: float,
    detector: str = "auto",
    sensitivity: float = 0.5,
    min_shot_frames: int = 4,
    overlap: int = 20,
    fallback_policy: str = "cascade",
) -> ShotList:
    """
    Detect shots in a (B, H, W, C) [0,1] image batch.

    Args:
        images: frame batch in playback order.
        fps: real frame rate, used for every timestamp.
        detector: one of DETECTOR_CHOICES.
        sensitivity: 0-1. Feeds TransNetV2's boundary threshold
            (inverted, so higher = more cuts) and scales the classic
            detector's difference threshold. OmniShotCut has no
            threshold to turn - it predicts shot ranges directly.
        min_shot_frames: shots shorter than this fold into the previous.
        overlap: OmniShotCut inference window overlap, in frames.
        fallback_policy: how far 'auto' may fall. 'cascade' tries all
            three, 'neural_only' refuses the classic detector, 'strict'
            demands OmniShotCut. Ignored when a detector is named.

    Returns:
        ShotList, with .requested and .fallback recording whether a
        stronger detector was skipped. Never raises for an unavailable
        detector under 'auto' unless the policy runs out of backends;
        it falls through and records why in .notes.
    """
    fps = float(fps) if fps and fps > 0 else 24.0
    total = int(images.shape[0])
    if total == 0:
        return ShotList(fps=fps, total_frames=0, detector=detector,
                        requested=detector, notes=["no frames to analyze"])
    if total == 1:
        return ShotList(
            shots=[Shot(index=1, start_frame=0, end_frame=1, fps=fps)],
            fps=fps, total_frames=1, detector=detector, requested=detector,
        )

    sensitivity = min(max(float(sensitivity), 0.0), 1.0)
    if detector == "auto":
        # The classic detector is a safety net, not a peer: on the demo
        # clip it produced false 0.13s shots and missed two real cuts.
        # A policy lets a workflow refuse a silent drop that far.
        cascade = ["omnishotcut", "transnetv2", "classic"]
        order = {
            "cascade": cascade,
            "neural_only": cascade[:2],
            "strict": cascade[:1],
        }.get(fallback_policy, cascade)
    else:
        order = [detector]

    notes: List[str] = []
    for name in order:
        try:
            if name == "omnishotcut":
                frames = _to_uint8_frames(images, *OSC_PROCESS_SIZE)
                result = _detect_omnishotcut(
                    frames, fps, overlap, min_shot_frames
                )
            elif name == "transnetv2":
                frames = _to_uint8_frames(images, *TNV2_SIZE)
                # TransNetV2's threshold is a cut probability: a lower bar
                # yields more cuts, so higher sensitivity must lower it.
                result = _detect_transnetv2(
                    frames, fps, 0.9 - 0.8 * sensitivity, min_shot_frames
                )
            elif name == "classic":
                # scene_detect works in 0-255 mean-difference units.
                result = _detect_classic(
                    images, fps, 20.0 - 18.0 * sensitivity, min_shot_frames
                )
            else:
                raise RuntimeError(
                    f"unknown detector '{name}'; choose one of "
                    f"{', '.join(DETECTOR_CHOICES)}"
                )
        except Exception as exc:  # noqa: BLE001
            if detector != "auto":
                raise
            notes.append(f"{name} unavailable: {exc}")
            continue

        result.notes = notes + result.notes
        result.requested = detector
        result.fallback = bool(notes)
        return result

    raise RuntimeError(
        "every detector failed:\n" + "\n".join(f"  - {n}" for n in notes)
    )
