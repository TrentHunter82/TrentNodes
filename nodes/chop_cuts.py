"""
Chop Cuts - GPU-accelerated scene detection and video splitting for ComfyUI.

Detects scene cuts in video frames and exports each scene as a separate MP4 file
with a detailed report of cut locations and timestamps.

Based on FL_VideoCut's proven detection algorithms with enhancements:
- Full GPU acceleration for intensity/hybrid detection
- Clean sequential filenames (no random UUIDs)
- Adaptive detection method inspired by PySceneDetect
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from comfy.utils import ProgressBar, common_upscale

from ..utils.image_ops import to_grayscale, get_sobel_kernels


class ChopCuts:
    """
    GPU-accelerated scene detection and video splitting node.
    Accurately detects cuts, fades, and transitions, then exports each scene
    as a separate MP4 file with a detailed report.
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("video_paths", "output_folder", "report", "num_scenes",)
    FUNCTION = "process"
    CATEGORY = "Trent/Video"

    DESCRIPTION = """Chop Cuts - GPU-Accelerated Scene Detection & Video Splitting

Automatically detects scene cuts in video frames and exports each scene
as a separate MP4 file. Generates a report with cut locations and timestamps.

Features:
- GPU-accelerated detection (intensity, edge, hybrid methods)
- Adaptive detection mode for fast camera motion
- Clean sequential filenames (scene_001.mp4, scene_002.mp4, etc.)
- Fast FFmpeg-based video export
- Downsampling option for faster detection on large frames
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames to scan for cuts and split into scenes."}),
                "output_folder": ("STRING", {"default": "./output/chop_cuts", "tooltip": "Folder for the scene MP4 files. Relative paths resolve from the ComfyUI working directory; the folder is created if missing."}),
                "base_filename": ("STRING", {"default": "scene", "tooltip": "Filename prefix for each scene, saved as <prefix>_001.mp4, <prefix>_002.mp4, and so on."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "tooltip": "Frame rate of the exported videos and of the timestamps in the report."}),
                "threshold": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Frame difference (0-255 scale) needed to count as a cut. Lower finds more cuts."
                }),
                "min_scene_frames": ("INT", {
                    "default": 12, "min": 2, "max": 1000, "step": 1,
                    "tooltip": "Minimum frames between cuts. Scenes shorter than this, including a short final scene, are not exported."
                }),
                "quality": ("INT", {
                    "default": 85, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Export quality from 1 to 100, mapped to an x264 CRF. Higher means better quality and larger files."
                }),
                "detection_method": (
                    ["hybrid", "intensity", "histogram", "adaptive"], {
                        "default": "hybrid",
                        "tooltip": "hybrid mixes intensity and edge changes; intensity uses brightness only; histogram compares color histograms on CPU; adaptive uses a rolling local threshold for fast camera motion."
                    }
                ),
                "max_workers": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Number of scene videos encoded with FFmpeg at the same time."
                }),
                "downsample_detection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Shrink frames larger than 640px to 640px for detection only. Exported videos keep full resolution."
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run detection on CUDA when available. Falls back to CPU otherwise."
                }),
            },
        }

    def process(
        self, images: torch.Tensor, output_folder: str, base_filename: str,
        fps: int, threshold: float, min_scene_frames: int, quality: int,
        detection_method: str, max_workers: int, downsample_detection: bool,
        use_gpu: bool
    ) -> Tuple[str, str, str, int]:
        """Main processing function."""

        print(
            f"[Chop Cuts] Starting scene detection with method: "
            f"{detection_method}"
        )

        # Setup output directory
        output_folder = os.path.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Get frame dimensions
        batch_size, height, width, channels = images.shape
        print(f"[Chop Cuts] Processing {batch_size} frames ({width}x{height})")

        # Determine device
        use_gpu = use_gpu and torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        print(f"[Chop Cuts] Using device: {device}")

        # Prepare detection images (optionally downsampled)
        if downsample_detection and (width > 640 or height > 640):
            detection_images = self._downsample_for_detection(images, device)
        else:
            detection_images = images.to(device) if use_gpu else images

        # Detect scenes
        scenes = self._detect_scenes(
            detection_images, images, threshold, min_scene_frames,
            detection_method, device
        )
        print(f"[Chop Cuts] Detected {len(scenes)} scenes")

        # Convert to numpy for video export
        np_frames = (images * 255).cpu().numpy().astype(np.uint8)

        # Export videos
        video_paths = self._export_videos(
            np_frames, scenes, output_folder, base_filename, fps, quality, max_workers
        )

        # Generate report
        report = self._generate_report(scenes, batch_size, fps, video_paths)

        print(
            f"[Chop Cuts] Complete! {len(scenes)} scenes "
            f"exported to {output_folder}"
        )

        return (
            ",".join(video_paths),
            output_folder,
            report,
            len(scenes)
        )

    def _downsample_for_detection(
        self, images: torch.Tensor, device: str
    ) -> torch.Tensor:
        """Downsample images to max 640px for faster detection."""
        batch_size, height, width, channels = images.shape
        scale = 640 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        print(
            f"[Chop Cuts] Downsampling to {new_width}x{new_height} "
            f"for detection..."
        )

        pbar = ProgressBar(batch_size)

        # Process in chunks to manage memory
        chunk_size = 64 if device == 'cuda' else 16
        result_chunks = []

        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = images[i:end_idx]

            # Convert to BCHW for common_upscale
            chunk_bchw = chunk.permute(0, 3, 1, 2)
            if device == 'cuda':
                chunk_bchw = chunk_bchw.to(device)

            # Downsample
            resized = common_upscale(
                chunk_bchw, new_width, new_height, "lanczos", "disabled"
            )

            # Convert back to BHWC
            resized_bhwc = resized.permute(0, 2, 3, 1)
            if device == 'cuda':
                resized_bhwc = resized_bhwc.cpu()

            result_chunks.append(resized_bhwc)
            pbar.update_absolute(end_idx)

        detection_images = torch.cat(result_chunks, dim=0)
        if device == 'cuda':
            detection_images = detection_images.to(device)

        return detection_images

    def _detect_scenes(
        self, detection_images: torch.Tensor, full_images: torch.Tensor,
        threshold: float, min_scene_frames: int,
        method: str, device: str
    ) -> List[Dict]:
        """
        Detect scene boundaries using the specified method.
        GPU-accelerated for intensity and hybrid methods.
        """
        batch_size = detection_images.shape[0]

        if batch_size < 2:
            return [{'start': 0, 'end': batch_size, 'length': batch_size}]

        print(f"[Chop Cuts] Analyzing frames with {method} method...")

        # Adjust threshold based on method
        if method == "intensity":
            adjusted_threshold = threshold * 0.8
        elif method == "histogram":
            adjusted_threshold = threshold * 3.0
        elif method == "adaptive":
            adjusted_threshold = threshold  # Will be computed dynamically
        else:  # hybrid
            adjusted_threshold = threshold

        # Compute frame differences
        if method == "histogram":
            # Histogram requires CPU/OpenCV
            differences = self._compute_histogram_differences(detection_images)
        elif method == "adaptive":
            # Adaptive uses intensity differences with rolling threshold
            differences = self._compute_intensity_differences_gpu(
                detection_images, device
            )
        else:
            # GPU-accelerated for intensity and hybrid
            if method == "hybrid":
                differences = self._compute_hybrid_differences_gpu(
                    detection_images, device, threshold
                )
            else:
                differences = self._compute_intensity_differences_gpu(
                    detection_images, device
                )

        # Find scene boundaries
        pbar = ProgressBar(len(differences))
        scene_boundaries = [0]

        if method == "adaptive":
            # Adaptive threshold based on rolling window
            window_size = 10
            min_content_val = threshold * 0.3  # Minimum threshold floor

            for i, diff in enumerate(differences):
                # Compute local average threshold
                start_idx = max(0, i - window_size)
                end_idx = min(len(differences), i + window_size + 1)
                local_avg = np.mean(differences[start_idx:end_idx])
                adaptive_thresh = max(local_avg * 2.0, min_content_val)

                if diff > adaptive_thresh:
                    frame_idx = i + 1
                    if (frame_idx - scene_boundaries[-1]) >= min_scene_frames:
                        scene_boundaries.append(frame_idx)
                        print(
                            f"[Chop Cuts] Cut at frame {frame_idx} "
                            f"(diff: {diff:.2f}, thresh: {adaptive_thresh:.2f})"
                        )

                pbar.update_absolute(i)
        else:
            # Fixed threshold methods
            for i, diff in enumerate(differences):
                if diff > adjusted_threshold:
                    frame_idx = i + 1
                    if (frame_idx - scene_boundaries[-1]) >= min_scene_frames:
                        scene_boundaries.append(frame_idx)
                        print(
                            f"[Chop Cuts] Cut at frame {frame_idx} "
                            f"(diff: {diff:.2f})"
                        )

                pbar.update_absolute(i)

        # Build scene list
        scenes = []
        for i in range(len(scene_boundaries)):
            start = scene_boundaries[i]
            if i + 1 < len(scene_boundaries):
                end = scene_boundaries[i + 1]
            else:
                end = batch_size

            if end - start >= min_scene_frames:
                scenes.append({
                    'start': start,
                    'end': end,
                    'length': end - start
                })

        return scenes

    def _compute_intensity_differences_gpu(
        self, images: torch.Tensor, device: str
    ) -> np.ndarray:
        """Compute intensity differences between consecutive frames on GPU."""
        print("[Chop Cuts] Computing intensity differences (GPU)...")

        # Ensure on correct device
        if images.device.type != device:
            images = images.to(device)

        # Convert to grayscale using shared utility
        gray = to_grayscale(images)

        # Compute differences between consecutive frames
        diffs = torch.abs(gray[1:] - gray[:-1])

        # Mean difference per frame pair
        mean_diffs = diffs.mean(dim=(1, 2))

        # Scale to match OpenCV absdiff range (0-255)
        mean_diffs = mean_diffs * 255.0

        return mean_diffs.cpu().numpy()

    def _compute_hybrid_differences_gpu(
        self, images: torch.Tensor, device: str, threshold: float
    ) -> np.ndarray:
        """
        Compute hybrid differences (intensity + edge) on GPU.
        Uses early accept/reject for efficiency.
        """
        print("[Chop Cuts] Computing hybrid differences (GPU)...")

        if images.device.type != device:
            images = images.to(device)

        batch_size = images.shape[0]
        differences = []

        # Convert to grayscale using shared utility and scale to 0-255 range
        gray = to_grayscale(images) * 255.0

        # Get Sobel kernels from shared utility
        sobel_x, sobel_y = get_sobel_kernels(torch.device(device))

        pbar = ProgressBar(batch_size - 1)

        for i in range(1, batch_size):
            curr_gray = gray[i]
            prev_gray = gray[i - 1]

            # Intensity difference
            intensity_diff = torch.abs(curr_gray - prev_gray).mean().item()

            # Early rejection
            if intensity_diff < threshold * 0.3:
                differences.append(intensity_diff)
                pbar.update_absolute(i - 1)
                continue

            # Early acceptance
            if intensity_diff > threshold * 3.0:
                differences.append(intensity_diff)
                pbar.update_absolute(i - 1)
                continue

            # Edge detection for ambiguous cases
            curr_4d = curr_gray.unsqueeze(0).unsqueeze(0)
            prev_4d = prev_gray.unsqueeze(0).unsqueeze(0)

            # Compute Sobel gradients
            curr_gx = F.conv2d(curr_4d, sobel_x, padding=1)
            curr_gy = F.conv2d(curr_4d, sobel_y, padding=1)
            prev_gx = F.conv2d(prev_4d, sobel_x, padding=1)
            prev_gy = F.conv2d(prev_4d, sobel_y, padding=1)

            # Edge magnitude difference
            curr_edges = torch.sqrt(curr_gx ** 2 + curr_gy ** 2)
            prev_edges = torch.sqrt(prev_gx ** 2 + prev_gy ** 2)
            edge_diff = torch.abs(curr_edges - prev_edges).mean().item()

            # Combined score
            combined = (intensity_diff * 0.6) + (edge_diff * 0.4)
            differences.append(combined)

            pbar.update_absolute(i - 1)

        return np.array(differences)

    def _compute_histogram_differences(
        self, images: torch.Tensor
    ) -> np.ndarray:
        """Compute histogram differences using OpenCV (CPU)."""
        print("[Chop Cuts] Computing histogram differences (CPU)...")

        # Convert to numpy
        np_images = (images * 255).cpu().numpy().astype(np.uint8)
        batch_size = np_images.shape[0]

        differences = []
        pbar = ProgressBar(batch_size - 1)

        for i in range(1, batch_size):
            curr_frame = np_images[i]
            prev_frame = np_images[i - 1]

            # Convert to HSV
            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)
            prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)

            # Calculate histograms
            hist_size = [16, 16]
            ranges = [0, 180, 0, 256]

            curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, hist_size, ranges)
            prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, hist_size, ranges)

            # Normalize
            cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)

            # Chi-Square comparison
            diff = cv2.compareHist(
                curr_hist, prev_hist, cv2.HISTCMP_CHISQR
            )
            differences.append(diff)

            pbar.update_absolute(i - 1)

        return np.array(differences)

    def _export_videos(
        self, frames: np.ndarray, scenes: List[Dict],
        output_folder: str, base_filename: str,
        fps: int, quality: int, max_workers: int
    ) -> List[str]:
        """Export each scene as an MP4 video using FFmpeg."""

        if not scenes:
            return []

        print(f"[Chop Cuts] Exporting {len(scenes)} videos...")
        pbar = ProgressBar(len(scenes))

        # Calculate CRF from quality
        crf = str(int((100 - quality) / 4) + 1)

        height, width = frames[0].shape[:2]

        def export_scene(idx: int, scene: Dict) -> Tuple[int, str, bool]:
            """Export a single scene to MP4."""
            # Clean sequential filename - no UUID!
            filename = f"{base_filename}_{idx + 1:03d}.mp4"
            filepath = os.path.join(output_folder, filename)

            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "rgb24",
                    "-r", str(fps),
                    "-i", "pipe:",
                    "-c:v", "libx264",
                    "-crf", crf,
                    "-preset", "veryfast",
                    "-tune", "film",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-loglevel", "error",
                    filepath
                ]

                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Write frames
                for i in range(scene['start'], scene['end']):
                    process.stdin.write(frames[i].tobytes())

                process.stdin.close()
                process.wait()

                if process.returncode != 0:
                    stderr = process.stderr.read().decode('utf-8')
                    print(
                        f"[Chop Cuts] FFmpeg error for {filename}: {stderr}"
                    )
                    return (idx, filepath, False)

                return (idx, filepath, True)

            except Exception as e:
                print(f"[Chop Cuts] Error exporting {filename}: {e}")
                return (idx, filepath, False)

        # Export videos in parallel
        actual_workers = min(max_workers, len(scenes))
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = []
            for idx, scene in enumerate(scenes):
                futures.append(executor.submit(export_scene, idx, scene))

            results = []
            for future in as_completed(futures):
                idx, filepath, success = future.result()
                if success:
                    results.append((idx, filepath))
                pbar.update_absolute(len(results))

        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        video_paths = [path for _, path in results]

        return video_paths

    def _generate_report(
        self, scenes: List[Dict], total_frames: int,
        fps: int, video_paths: List[str]
    ) -> str:
        """Generate a human-readable report of detected scenes."""

        total_duration = total_frames / fps

        report = f"""Chop Cuts - Scene Detection Report
===================================
Total frames: {total_frames} | Duration: {total_duration:.1f}s | FPS: {fps}
Scenes detected: {len(scenes)}

"""  # noqa: E501

        if not scenes:
            report += "No scene cuts detected.\n"
            return report

        for i, scene in enumerate(scenes):
            start_time = scene['start'] / fps
            end_time = scene['end'] / fps
            duration = scene['length'] / fps

            # Format timestamps as MM:SS.ms
            start_min = int(start_time // 60)
            start_sec = start_time % 60
            end_min = int(end_time // 60)
            end_sec = end_time % 60
            start_str = f"{start_min:02d}:{start_sec:05.2f}"
            end_str = f"{end_min:02d}:{end_sec:05.2f}"

            if i < len(video_paths):
                filename = os.path.basename(video_paths[i])
            else:
                filename = "N/A"

            report += (
                f"Scene {i + 1}: frames {scene['start']}-{scene['end']} "
                f"({start_str} - {end_str}, {duration:.1f}s) -> {filename}\n"
            )

        if video_paths:
            folder = os.path.dirname(video_paths[0])
            report += f"\nOutput folder: {folder}\n"

        return report


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "ChopCuts": ChopCuts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChopCuts": "Chop Cuts"
}
