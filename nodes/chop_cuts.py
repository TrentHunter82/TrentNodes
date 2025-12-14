"""
Chop Cuts - Accurate scene detection and video splitting for ComfyUI.

Detects scene cuts in video frames and exports each scene as a separate MP4 file
with a detailed report of cut locations and timestamps.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from comfy.utils import ProgressBar


class ChopCuts:
    """
    Scene detection and video splitting node.
    Accurately detects cuts, fades, and transitions, then exports each scene
    as a separate MP4 file with a detailed report.
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT",)
    RETURN_NAMES = ("video_paths", "output_folder", "report", "num_scenes",)
    FUNCTION = "process"
    CATEGORY = "Trent/Video"

    DESCRIPTION = """Chop Cuts - Scene Detection & Video Splitting

Automatically detects scene cuts in video frames and exports each scene
as a separate MP4 file. Generates a report with cut locations and timestamps.

Features:
- Multi-metric detection (intensity, color histogram, structural similarity)
- Catches hard cuts, dissolves, and fade transitions
- Fast FFmpeg-based video export
- Simple sensitivity control
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_folder": ("STRING", {"default": "./output/chop_cuts"}),
                "base_filename": ("STRING", {"default": "scene"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                               "description": "Detection sensitivity (lower = more sensitive)"}),
                "min_scene_frames": ("INT", {"default": 12, "min": 1, "max": 300, "step": 1,
                                    "description": "Minimum frames per scene"}),
                "quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1,
                           "description": "Video quality (1-100)"}),
            },
        }

    def process(self, images: torch.Tensor, output_folder: str, base_filename: str,
                fps: int, sensitivity: float, min_scene_frames: int,
                quality: int) -> Tuple[str, str, str, int]:
        """Main processing function."""

        print(f"[Chop Cuts] Starting scene detection...")

        # Setup output directory
        output_folder = os.path.abspath(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Get frame dimensions
        batch_size, height, width, channels = images.shape
        print(f"[Chop Cuts] Processing {batch_size} frames ({width}x{height})")

        # Convert to numpy for processing
        frames = (images * 255).cpu().numpy().astype(np.uint8)

        # Detect scenes
        scenes = self._detect_scenes(frames, sensitivity, min_scene_frames)
        print(f"[Chop Cuts] Detected {len(scenes)} scenes")

        # Export videos
        video_paths = self._export_videos(frames, scenes, output_folder,
                                          base_filename, fps, quality)

        # Generate report
        report = self._generate_report(scenes, batch_size, fps, video_paths)

        print(f"[Chop Cuts] Complete! {len(scenes)} scenes exported to {output_folder}")

        return (
            ",".join(video_paths),
            output_folder,
            report,
            len(scenes)
        )

    def _detect_scenes(self, frames: np.ndarray, sensitivity: float,
                       min_scene_frames: int) -> List[Dict]:
        """
        Detect scene boundaries using multi-metric analysis.

        Uses a combination of:
        - Intensity difference (fast baseline)
        - Color histogram comparison (catches color/lighting changes)
        - Structural similarity (catches structural changes)
        """
        batch_size = frames.shape[0]

        if batch_size < 2:
            return [{'start': 0, 'end': batch_size, 'length': batch_size}]

        print("[Chop Cuts] Analyzing frames...")
        pbar = ProgressBar(batch_size)

        # Pre-compute grayscale frames
        gray_frames = []
        for i in range(batch_size):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)
            pbar.update_absolute(i)

        # Compute frame differences
        differences = []
        for i in range(1, batch_size):
            # Intensity difference
            diff_intensity = np.mean(cv2.absdiff(gray_frames[i], gray_frames[i-1]))

            # Color histogram difference
            diff_color = self._compute_histogram_diff(frames[i], frames[i-1])

            # Structural difference (simplified SSIM-like metric)
            diff_struct = self._compute_structural_diff(gray_frames[i], gray_frames[i-1])

            # Weighted combination
            combined = (diff_intensity * 0.4) + (diff_color * 30 * 0.3) + (diff_struct * 100 * 0.3)
            differences.append(combined)

        # Calculate adaptive threshold based on content
        if differences:
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            # Lower sensitivity = lower threshold = more cuts detected
            threshold = mean_diff + (sensitivity * 2.0) * std_diff
            print(f"[Chop Cuts] Threshold: {threshold:.2f} (mean: {mean_diff:.2f}, std: {std_diff:.2f})")
        else:
            threshold = 25.0

        # Detect scene boundaries
        scenes = []
        current_start = 0

        for i, diff in enumerate(differences):
            frame_idx = i + 1  # differences[i] is between frame i and i+1

            if diff > threshold:
                scene_length = frame_idx - current_start
                if scene_length >= min_scene_frames:
                    scenes.append({
                        'start': current_start,
                        'end': frame_idx,
                        'length': scene_length,
                        'cut_strength': float(diff / threshold)
                    })
                    current_start = frame_idx

        # Add final scene
        final_length = batch_size - current_start
        if final_length >= min_scene_frames:
            scenes.append({
                'start': current_start,
                'end': batch_size,
                'length': final_length,
                'cut_strength': 1.0
            })

        # Validate and merge nearby cuts
        scenes = self._validate_scenes(frames, gray_frames, scenes, min_scene_frames)

        return scenes

    def _compute_histogram_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute color histogram difference between two frames."""
        # Convert to HSV for better color comparison
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)

        # Calculate 2D histogram (hue + saturation)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [16, 16], [0, 180, 0, 256])

        # Normalize
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

        # Bhattacharyya distance (0 = identical, 1 = completely different)
        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))

    def _compute_structural_diff(self, gray1: np.ndarray, gray2: np.ndarray) -> float:
        """
        Compute structural difference between two grayscale frames.
        Simplified SSIM-like metric for speed.
        """
        # Compute local means using box filter (faster than Gaussian)
        kernel_size = 11
        mu1 = cv2.blur(gray1.astype(np.float32), (kernel_size, kernel_size))
        mu2 = cv2.blur(gray2.astype(np.float32), (kernel_size, kernel_size))

        # Compute local variances
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.blur(gray1.astype(np.float32) ** 2, (kernel_size, kernel_size)) - mu1_sq
        sigma2_sq = cv2.blur(gray2.astype(np.float32) ** 2, (kernel_size, kernel_size)) - mu2_sq
        sigma12 = cv2.blur(gray1.astype(np.float32) * gray2.astype(np.float32),
                          (kernel_size, kernel_size)) - mu1_mu2

        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        # Compute SSIM
        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        # Return dissimilarity (1 - SSIM)
        return float(1 - np.mean(ssim))

    def _validate_scenes(self, frames: np.ndarray, gray_frames: List[np.ndarray],
                        scenes: List[Dict], min_scene_frames: int) -> List[Dict]:
        """Validate scene cuts and merge very short gaps."""
        if not scenes:
            return scenes

        validated = []

        for scene in scenes:
            # Skip scenes that are too short
            if scene['length'] < min_scene_frames:
                continue

            # Refine boundary if needed
            if scene['start'] > 0:
                # Check if boundary is at the sharpest transition
                best_start = scene['start']
                best_diff = 0

                for offset in range(-2, 3):
                    test_idx = scene['start'] + offset
                    if 0 < test_idx < len(gray_frames):
                        diff = np.mean(cv2.absdiff(gray_frames[test_idx],
                                                   gray_frames[test_idx - 1]))
                        if diff > best_diff:
                            best_diff = diff
                            best_start = test_idx

                scene['start'] = best_start
                scene['length'] = scene['end'] - scene['start']

            if scene['length'] >= min_scene_frames:
                validated.append(scene)

        # Merge scenes with very small gaps (< 3 frames)
        if len(validated) < 2:
            return validated

        merged = [validated[0]]
        for scene in validated[1:]:
            gap = scene['start'] - merged[-1]['end']
            if gap < 3:
                # Merge with previous scene
                merged[-1]['end'] = scene['end']
                merged[-1]['length'] = merged[-1]['end'] - merged[-1]['start']
            else:
                merged.append(scene)

        return merged

    def _export_videos(self, frames: np.ndarray, scenes: List[Dict],
                       output_folder: str, base_filename: str,
                       fps: int, quality: int) -> List[str]:
        """Export each scene as an MP4 video using FFmpeg."""

        if not scenes:
            return []

        print(f"[Chop Cuts] Exporting {len(scenes)} videos...")
        pbar = ProgressBar(len(scenes))

        # Calculate CRF from quality (quality 100 = CRF 1, quality 1 = CRF 51)
        crf = str(int((100 - quality) / 2) + 1)

        video_paths = []
        height, width = frames[0].shape[:2]

        def export_scene(idx: int, scene: Dict) -> Tuple[int, str, bool]:
            """Export a single scene to MP4."""
            filename = f"{base_filename}_{idx + 1:03d}.mp4"
            filepath = os.path.join(output_folder, filename)

            try:
                # FFmpeg command for direct piping
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
                    "-preset", "fast",
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
                    print(f"[Chop Cuts] FFmpeg error for {filename}: {stderr}")
                    return (idx, filepath, False)

                return (idx, filepath, True)

            except Exception as e:
                print(f"[Chop Cuts] Error exporting {filename}: {e}")
                return (idx, filepath, False)

        # Export videos in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
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

    def _generate_report(self, scenes: List[Dict], total_frames: int,
                         fps: int, video_paths: List[str]) -> str:
        """Generate a human-readable report of detected scenes."""

        total_duration = total_frames / fps

        report = f"""Chop Cuts - Scene Detection Report
===================================
Total frames: {total_frames} | Duration: {total_duration:.1f}s | FPS: {fps}
Scenes detected: {len(scenes)}

"""

        if not scenes:
            report += "No scene cuts detected.\n"
            return report

        for i, scene in enumerate(scenes):
            start_time = scene['start'] / fps
            end_time = scene['end'] / fps
            duration = scene['length'] / fps

            # Format timestamps as MM:SS.ms
            start_str = f"{int(start_time // 60):02d}:{start_time % 60:05.2f}"
            end_str = f"{int(end_time // 60):02d}:{end_time % 60:05.2f}"

            filename = os.path.basename(video_paths[i]) if i < len(video_paths) else "N/A"

            report += f"Scene {i + 1}: frames {scene['start']}-{scene['end']} "
            report += f"({start_str} - {end_str}, {duration:.1f}s) -> {filename}\n"

        if video_paths:
            report += f"\nOutput folder: {os.path.dirname(video_paths[0])}\n"

        return report


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "ChopCuts": ChopCuts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChopCuts": "Chop Cuts"
}
