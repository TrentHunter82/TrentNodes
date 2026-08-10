"""
Scene-cut detection utilities for TrentNodes.

Module-level port of the detection core from nodes/chop_cuts.py so other
nodes can reuse it without instantiating the ChopCuts node class. Pure
torch/numpy (cv2 only for the histogram method, imported lazily), no
ComfyUI imports, so this module is unit-testable outside the server.

Differences from the chop_cuts originals:
- common_upscale -> F.interpolate (area mode; detection does not need
  lanczos fidelity)
- No ProgressBar (callers that want progress wrap these themselves)
- detect_scenes operates on a precomputed difference curve so callers
  can reuse the curve for motion-peak analysis
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .image_ops import get_sobel_kernels, to_grayscale


def downsample_for_detection(
    images: torch.Tensor,
    max_side: int = 384,
    chunk_size: int = 64,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Downsample a (B, H, W, C) [0,1] batch so max(H, W) <= max_side.

    Returns a detection copy on `device` (defaults to images.device).
    Chunked to bound peak memory on long batches.
    """
    batch_size, height, width, _ = images.shape
    if device is None:
        device = images.device.type
    if max(height, width) <= max_side:
        return images.to(device)

    scale = max_side / max(width, height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    result_chunks = []
    for i in range(0, batch_size, chunk_size):
        chunk = images[i:i + chunk_size].to(device)
        chunk_bchw = chunk.permute(0, 3, 1, 2)
        resized = F.interpolate(
            chunk_bchw, size=(new_height, new_width), mode="area"
        )
        result_chunks.append(resized.permute(0, 2, 3, 1))

    return torch.cat(result_chunks, dim=0)


def compute_intensity_differences(images: torch.Tensor) -> np.ndarray:
    """
    Mean absolute grayscale difference between consecutive frames,
    scaled to the OpenCV absdiff range (0-255).

    Args:
        images: (B, H, W, C) [0,1] tensor, B >= 2

    Returns:
        np.ndarray of length B-1; element i is the difference between
        frames i and i+1.
    """
    gray = to_grayscale(images)
    diffs = torch.abs(gray[1:] - gray[:-1])
    mean_diffs = diffs.mean(dim=(1, 2)) * 255.0
    return mean_diffs.cpu().numpy()


def compute_hybrid_differences(
    images: torch.Tensor, threshold: float = 8.0
) -> np.ndarray:
    """
    Intensity + Sobel-edge difference between consecutive frames with
    early accept/reject around `threshold` (same scheme as chop_cuts).

    Returns np.ndarray of length B-1 in the 0-255 intensity scale.
    """
    batch_size = images.shape[0]
    gray = to_grayscale(images) * 255.0
    sobel_x, sobel_y = get_sobel_kernels(images.device)

    differences = []
    for i in range(1, batch_size):
        curr_gray = gray[i]
        prev_gray = gray[i - 1]

        intensity_diff = torch.abs(curr_gray - prev_gray).mean().item()

        # Early rejection / acceptance skips the Sobel pass
        if intensity_diff < threshold * 0.3 or intensity_diff > threshold * 3.0:
            differences.append(intensity_diff)
            continue

        curr_4d = curr_gray.unsqueeze(0).unsqueeze(0)
        prev_4d = prev_gray.unsqueeze(0).unsqueeze(0)

        curr_gx = F.conv2d(curr_4d, sobel_x, padding=1)
        curr_gy = F.conv2d(curr_4d, sobel_y, padding=1)
        prev_gx = F.conv2d(prev_4d, sobel_x, padding=1)
        prev_gy = F.conv2d(prev_4d, sobel_y, padding=1)

        curr_edges = torch.sqrt(curr_gx ** 2 + curr_gy ** 2)
        prev_edges = torch.sqrt(prev_gx ** 2 + prev_gy ** 2)
        edge_diff = torch.abs(curr_edges - prev_edges).mean().item()

        differences.append((intensity_diff * 0.6) + (edge_diff * 0.4))

    return np.array(differences)


def compute_histogram_differences(images: torch.Tensor) -> np.ndarray:
    """
    HSV histogram chi-square difference between consecutive frames.
    CPU/OpenCV path; cv2 imported lazily.
    """
    import cv2

    np_images = (images * 255).cpu().numpy().astype(np.uint8)
    batch_size = np_images.shape[0]

    hist_size = [16, 16]
    ranges = [0, 180, 0, 256]

    differences = []
    for i in range(1, batch_size):
        curr_hsv = cv2.cvtColor(np_images[i], cv2.COLOR_RGB2HSV)
        prev_hsv = cv2.cvtColor(np_images[i - 1], cv2.COLOR_RGB2HSV)

        curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, hist_size, ranges)
        prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, hist_size, ranges)

        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)

        differences.append(
            cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_CHISQR)
        )

    return np.array(differences)


def detect_boundaries(
    differences: np.ndarray,
    threshold: float,
    min_scene_frames: int = 2,
    adaptive: bool = False,
) -> List[int]:
    """
    Find cut boundaries on a difference curve.

    A boundary at frame index f means frame f starts a new scene
    (the cut is between frames f-1 and f). Index 0 is always included.

    adaptive=True uses the chop_cuts rolling-window scheme: a cut fires
    when the local diff exceeds max(2x the windowed average, 0.3x the
    base threshold).
    """
    boundaries = [0]

    if adaptive:
        window_size = 10
        min_content_val = threshold * 0.3
        for i, diff in enumerate(differences):
            start_idx = max(0, i - window_size)
            end_idx = min(len(differences), i + window_size + 1)
            local_avg = np.mean(differences[start_idx:end_idx])
            adaptive_thresh = max(local_avg * 2.0, min_content_val)
            if diff > adaptive_thresh:
                frame_idx = i + 1
                if (frame_idx - boundaries[-1]) >= min_scene_frames:
                    boundaries.append(frame_idx)
    else:
        for i, diff in enumerate(differences):
            if diff > threshold:
                frame_idx = i + 1
                if (frame_idx - boundaries[-1]) >= min_scene_frames:
                    boundaries.append(frame_idx)

    return boundaries


def scenes_from_boundaries(
    boundaries: List[int], total_frames: int
) -> List[Dict]:
    """
    Convert boundary indices into scene dicts:
    [{'start': int, 'end': int (exclusive), 'length': int}, ...]

    Unlike chop_cuts (which drops runt scenes because it exports them
    as videos), every frame stays covered here - keyframe selection
    needs full coverage.
    """
    scenes = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else total_frames
        if end > start:
            scenes.append({"start": start, "end": end, "length": end - start})
    return scenes
