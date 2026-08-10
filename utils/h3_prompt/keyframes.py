"""
Keyframe selection for the H3 Auto Prompt Generator.

Picks the frames a VLM should see: scene-cut boundaries first, then
motion peaks, then structural anchors (first/mid/last), capped at a
budget. Timing comes from the caller's fps so every selected frame
carries a real timestamp.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from ..scene_detect import (
    compute_hybrid_differences,
    compute_intensity_differences,
    detect_boundaries,
    downsample_for_detection,
    scenes_from_boundaries,
)

# Above this batch size the per-frame Sobel loop in the hybrid method
# costs more than it helps; plain intensity diffs are vectorized.
HYBRID_MAX_FRAMES = 2000

DEFAULT_CUT_THRESHOLD = 8.0


@dataclass
class KeyframeResult:
    indices: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    scene_boundaries: List[int] = field(default_factory=list)
    diff_stats: Dict = field(default_factory=dict)
    method: str = "hybrid"


def _gaussian_smooth(curve: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth a 1-D curve with a reflected gaussian kernel."""
    if len(curve) < 3:
        return curve.astype(np.float64)
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(curve.astype(np.float64), radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _motion_peaks(
    smoothed: np.ndarray,
    boundaries: List[int],
    exclusion: int = 2,
) -> List[int]:
    """
    Local maxima of the smoothed diff curve, excluding +/-`exclusion`
    frames around each cut boundary, strongest first.

    smoothed[i] is the motion between frames i and i+1; the peak is
    attributed to frame i+1 (where the motion lands).
    """
    if len(smoothed) < 3:
        return []

    excluded = set()
    for b in boundaries:
        for off in range(-exclusion, exclusion + 1):
            excluded.add(b + off)

    peaks = []
    for i in range(1, len(smoothed) - 1):
        frame_idx = i + 1
        if frame_idx in excluded:
            continue
        if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            peaks.append((smoothed[i], frame_idx))

    peaks.sort(key=lambda p: -p[0])
    return [idx for _, idx in peaks]


def select_keyframes(
    images: torch.Tensor,
    fps: float,
    max_frames: int = 8,
    cut_threshold: float = DEFAULT_CUT_THRESHOLD,
) -> KeyframeResult:
    """
    Select up to `max_frames` frame indices from a (B, H, W, C) [0,1]
    batch for VLM analysis.

    Priority: first + last frame > scene starts (longest scenes first)
    > scene midpoints > motion peaks > uniform fill. Selected indices
    keep a minimum spacing except the always-kept first/last anchors.
    """
    batch_size = images.shape[0]
    fps = float(fps) if fps and fps > 0 else 24.0
    max_frames = max(2, int(max_frames))

    if batch_size == 0:
        return KeyframeResult(method="empty")
    if batch_size == 1:
        return KeyframeResult(
            indices=[0], timestamps=[0.0], scene_boundaries=[0],
            diff_stats={"frames": 1}, method="single",
        )

    detection = downsample_for_detection(images, max_side=384)

    if batch_size > HYBRID_MAX_FRAMES:
        method = "intensity"
        differences = compute_intensity_differences(detection)
    else:
        method = "hybrid"
        differences = compute_hybrid_differences(
            detection, threshold=cut_threshold
        )

    min_scene_frames = max(2, int(round(0.15 * fps)))
    boundaries = detect_boundaries(
        differences, cut_threshold, min_scene_frames, adaptive=True
    )
    scenes = scenes_from_boundaries(boundaries, batch_size)

    smoothed = _gaussian_smooth(differences, sigma=2.0)
    min_spacing = max(3, batch_size // (2 * max_frames))

    # Candidates in priority order; first/last are unconditional anchors.
    scenes_by_length = sorted(scenes, key=lambda s: -s["length"])
    candidates: List[int] = []
    candidates.extend(s["start"] for s in scenes_by_length)
    candidates.extend(
        (s["start"] + s["end"]) // 2 for s in scenes_by_length
    )
    candidates.extend(_motion_peaks(smoothed, boundaries))
    if batch_size > max_frames:
        candidates.extend(
            int(round(i * (batch_size - 1) / (max_frames - 1)))
            for i in range(max_frames)
        )

    selected = [0, batch_size - 1]
    for idx in candidates:
        if len(selected) >= max_frames:
            break
        idx = int(min(max(idx, 0), batch_size - 1))
        if all(abs(idx - s) >= min_spacing for s in selected):
            selected.append(idx)

    selected = sorted(set(selected))[:max_frames]
    if (batch_size - 1) not in selected:
        selected[-1] = batch_size - 1

    return KeyframeResult(
        indices=selected,
        timestamps=[round(i / fps, 3) for i in selected],
        scene_boundaries=boundaries,
        diff_stats={
            "frames": batch_size,
            "diff_mean": float(np.mean(differences)),
            "diff_max": float(np.max(differences)),
            "num_scenes": len(scenes),
            "min_spacing": min_spacing,
        },
        method=method,
    )


def frame_label(position: int, total: int, timestamp: float, index: int) -> str:
    """Text label sent to the VLM ahead of each frame image."""
    return (
        f"Frame {position} of {total} - t={timestamp:.3f}s "
        f"(video frame index {index})"
    )
