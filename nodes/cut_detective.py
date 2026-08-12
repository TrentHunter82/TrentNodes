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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "cut_times", "shot_table", "film_strip", "report", "cuts_json",
        "num_shots",
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
    ) -> Tuple[str, str, torch.Tensor, str, str, int]:
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
        )

        for note in shots.notes:
            print(f"{LOG_PREFIX} note: {note}")
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
        )

        return (
            format_cut_times(shots, include_first=include_first_shot),
            format_shot_table(shots),
            strip,
            format_report(shots),
            shots_to_json(shots),
            len(shots.shots),
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
