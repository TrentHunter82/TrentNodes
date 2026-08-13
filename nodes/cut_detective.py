"""
Cut Detective - neural shot-boundary detection with a film-strip preview.

Finds every cut in a clip, says what kind of cut each one is, draws a
labelled film strip of the result, and writes the cut list as a string
the H3 Auto Prompt Generator can take verbatim.

Default detector is OmniShotCut (UVA Computer Vision Lab, 2026), a
shot-query video Transformer that types its boundaries - hard cut,
dissolve, wipe, fade, whip-pan, sudden jump - instead of thresholding a
frame-difference curve. TransNetV2 and a no-model frame-difference
detector stand behind it. See utils/cut_detect/detectors.py.
"""

from fractions import Fraction
from typing import Optional, Tuple

import torch

from ..utils.cut_detect import (
    DETECTOR_CHOICES,
    detect_shots,
    format_cut_times,
    format_report,
    format_shot_table,
    render_film_strip,
    shots_to_json,
)

LOG_PREFIX = "[CutDetective]"


class CutDetective:
    """Detect shot boundaries and preview them as a film strip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detector": (list(DETECTOR_CHOICES), {
                    "default": "auto",
                    "tooltip": (
                        "auto: OmniShotCut, then TransNetV2, then the "
                        "classic frame-difference detector - first one "
                        "that loads wins. omnishotcut (2026 SOTA, needs "
                        "CUDA + the omnishotcut package) is the only one "
                        "that labels dissolves, wipes and fades. "
                        "transnetv2 runs on CPU with bundled weights but "
                        "reports hard cuts only. classic needs nothing "
                        "and is a safety net, not a peer."
                    ),
                }),
                "sensitivity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Higher finds more cuts. Applies to transnetv2 "
                        "and classic; omnishotcut predicts shot ranges "
                        "directly and ignores it."
                    ),
                }),
                "min_shot_frames": ("INT", {
                    "default": 4, "min": 1, "max": 240,
                    "tooltip": (
                        "Shots shorter than this fold into the previous "
                        "one. Kills detector wobble around a single cut."
                    ),
                }),
                "thumb_width": ("INT", {
                    "default": 240, "min": 64, "max": 640, "step": 8,
                    "tooltip": "Width of each film-strip thumbnail, in pixels",
                }),
                "columns": ("INT", {
                    "default": 0, "min": 0, "max": 32,
                    "tooltip": "Thumbnails per row. 0 fits as many as it can.",
                }),
                "show_timeline": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Draw the proportional timeline ribbon under the "
                        "strip, with a tick at every cut"
                    ),
                }),
                "include_first_shot": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Keep 0.000 at the head of cut_times. On means "
                        "'every shot start' (what the H3 node wants); off "
                        "means 'boundaries only'."
                    ),
                }),
            },
            "optional": {
                "video": ("VIDEO", {
                    "tooltip": "Source clip. Preferred input; carries its fps."
                }),
                "frames": ("IMAGE", {
                    "tooltip": (
                        "Alternative to video: an IMAGE batch (e.g. from "
                        "VHS Load Video). Set fps to match."
                    ),
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": (
                        "Frame rate of the frames input. Ignored when a "
                        "video is connected."
                    ),
                }),
                "title": ("STRING", {
                    "default": "",
                    "tooltip": "Film-strip header. Blank uses a default.",
                }),
                "thumbs_per_shot": ("INT", {
                    "default": 1, "min": 1, "max": 8,
                    "tooltip": (
                        "Thumbnails sampled from each shot. 1 shows "
                        "only the frame after the cut. Raise it to see "
                        "what happens inside a long shot. Shots are "
                        "never dropped to fit; the count degrades "
                        "instead, and the sheet says so."
                    ),
                }),
                "omnishotcut_overlap": ("INT", {
                    "default": 20, "min": 0, "max": 90,
                    "tooltip": (
                        "Frames of overlap between OmniShotCut's "
                        "100-frame inference windows. This is the knob "
                        "that actually changes its results, since it "
                        "predicts shot ranges directly and ignores "
                        "sensitivity. Raise it if boundaries near a "
                        "window edge look wrong; it costs proportional "
                        "time. No effect on the other detectors."
                    ),
                }),
                "fallback_policy": (["cascade", "neural_only", "strict"], {
                    "default": "cascade",
                    "tooltip": (
                        "How far 'auto' may fall when a detector is "
                        "unavailable. cascade tries all three. "
                        "neural_only refuses the classic detector, which "
                        "misses gradual boundaries and invents short "
                        "shots. strict demands omnishotcut and errors "
                        "instead of substituting. Ignored when you name "
                        "a detector."
                    ),
                }),
            },
        }

    RETURN_TYPES = (
        "STRING", "STRING", "IMAGE", "STRING", "STRING", "INT", "STRING",
    )
    RETURN_NAMES = (
        "cut_times", "shot_table", "film_strip", "report", "cuts_json",
        "num_shots", "detector_used",
    )
    FUNCTION = "detect"
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Detects every shot boundary in a clip with OmniShotCut (2026 "
        "shot-query Transformer), TransNetV2, or a frame-difference "
        "fallback. Labels each cut as a hard cut, dissolve, wipe, fade, "
        "or sudden jump, renders a film-strip contact sheet with the cut "
        "times marked, and emits the cut list as a string for the H3 "
        "Auto Prompt Generator's cut_times input."
    )

    def detect(
        self,
        detector: str,
        sensitivity: float,
        min_shot_frames: int,
        thumb_width: int,
        columns: int,
        show_timeline: bool,
        include_first_shot: bool,
        video=None,
        frames: Optional[torch.Tensor] = None,
        fps: float = 24.0,
        title: str = "",
        thumbs_per_shot: int = 1,
        omnishotcut_overlap: int = 20,
        fallback_policy: str = "cascade",
    ) -> Tuple[str, str, torch.Tensor, str, str, int, str]:
        images, real_fps = self._resolve_frames(video, frames, fps)
        print(
            f"{LOG_PREFIX} analyzing {images.shape[0]} frames @ "
            f"{real_fps:.3f} fps with detector '{detector}'"
        )

        shots = detect_shots(
            images,
            fps=real_fps,
            detector=detector,
            sensitivity=sensitivity,
            min_shot_frames=min_shot_frames,
            overlap=omnishotcut_overlap,
            fallback_policy=fallback_policy,
        )

        # Turning a knob that does nothing is a reasonable complaint.
        # Say so where the other findings are, rather than hiding the
        # widget from the canvas.
        if shots.detector == "omnishotcut" and abs(sensitivity - 0.5) > 1e-6:
            shots.notes.append(
                "sensitivity was moved but omnishotcut predicts shot "
                "ranges directly and ignores it; use min_shot_frames or "
                "omnishotcut_overlap on this path"
            )

        cut_times = format_cut_times(shots, include_first=include_first_shot)

        # A clip with no cuts and no leading 0.000 leaves nothing to
        # emit. Downstream that blank string is indistinguishable from
        # an unconnected widget, so the H3 node throws away this
        # measured "one shot" answer and guesses its own cuts instead.
        if shots.shots and not cut_times:
            shots.notes.append(
                "no cuts were found and include_first_shot is off, so "
                "cut_times is empty; a downstream node cannot tell that "
                "apart from nothing being connected. Turn "
                "include_first_shot on to send the single shot at 0.000"
            )

        # A fallback changes what every other output means, so it is a
        # warning, not a note.
        level = "WARNING:" if shots.fallback else "note:"
        for note in shots.notes:
            print(f"{LOG_PREFIX} {level} {note}")
        if shots.fallback:
            print(
                f"{LOG_PREFIX} WARNING: asked for '{shots.requested}', "
                f"ran '{shots.detector}'"
            )
        print(
            f"{LOG_PREFIX} {shots.detector} found {len(shots.shots)} shots "
            f"({shots.num_cuts} cuts) in {shots.duration:.3f}s"
        )

        strip = render_film_strip(
            images, shots,
            thumb_width=thumb_width,
            columns=columns,
            show_timeline=show_timeline,
            title=title.strip() or None,
            thumbs_per_shot=thumbs_per_shot,
        )

        return (
            cut_times,
            format_shot_table(shots),
            strip,
            format_report(shots),
            shots_to_json(shots),
            len(shots.shots),
            shots.detector,
        )

    def _resolve_frames(
        self, video, frames, fps: float
    ) -> Tuple[torch.Tensor, float]:
        """Resolve (frames, fps) from the two input paths. Video wins."""
        if video is not None:
            try:
                components = video.get_components()
            except AttributeError as exc:
                raise RuntimeError(
                    "The video input is not a ComfyUI VIDEO object. "
                    "Connect a Load Video node, or use the frames input."
                ) from exc
            rate = components.frame_rate
            real_fps = (
                float(rate) if isinstance(rate, Fraction)
                else float(rate or 24.0)
            )
            return components.images, real_fps

        if frames is not None:
            if frames.dim() != 4 or frames.shape[0] < 1:
                raise RuntimeError(
                    "frames must be a non-empty IMAGE batch (B, H, W, C)."
                )
            return frames, float(fps)

        raise RuntimeError(
            "Connect either a VIDEO (video input) or an IMAGE batch "
            "(frames input, with fps set)."
        )


NODE_CLASS_MAPPINGS = {
    "TrentCutDetective": CutDetective,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentCutDetective": "Cut Detective",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
