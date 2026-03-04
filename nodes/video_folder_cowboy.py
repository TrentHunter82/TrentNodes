"""
Video Folder Cowboy - A directory iterator for video files.

Based on ImageFolderCowboy, adapted for video files:
1. Natural sorting for filenames (vid1 < vid2 < vid10)
2. Sorted subdirectory processing
3. Configurable index overflow handling (wrap, clamp, error)
4. Loads video frames via OpenCV
5. Optional frame sampling (every Nth frame, max frames)
"""

import glob
import os
import random
import traceback
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from .image_folder_cowboy import natural_sort_key, get_sort_key


# Valid video extensions
VALID_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.flv', '.wmv', '.m4v', '.mpg', '.mpeg',
}


class VideoFolderCowboy:
    """
    Load video files from a directory with proper natural sorting
    and flexible index handling.

    Iterates through video files in a folder, loading frames as
    IMAGE batches for use in ComfyUI workflows.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/videos",
                    "tooltip": (
                        "Root directory to search for video files"
                    ),
                }),
                "patterns": ("STRING", {
                    "default": (
                        "**/*.mp4, **/*.avi, **/*.mov, "
                        "**/*.mkv, **/*.webm"
                    ),
                    "tooltip": (
                        "Comma-separated glob patterns for "
                        "matching files"
                    ),
                }),
                "video_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Index of the video to load",
                }),
                "sort_by": (
                    [
                        "name", "date_modified", "size",
                        "random", "none",
                    ],
                    {
                        "default": "name",
                        "tooltip": (
                            "Sorting method: name uses natural sort"
                        ),
                    },
                ),
                "sort_order": (
                    ["ascending", "descending"],
                    {
                        "default": "ascending",
                        "tooltip": "Sort direction",
                    },
                ),
            },
            "optional": {
                "frame_skip": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 120,
                    "step": 1,
                    "tooltip": (
                        "Skip N frames between each loaded frame "
                        "(0 = load every frame)"
                    ),
                }),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Maximum frames to load (0 = all frames)"
                    ),
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": (
                        "First frame index to start loading from"
                    ),
                }),
                "index_mode": (
                    ["wrap", "clamp", "error"],
                    {
                        "default": "wrap",
                        "tooltip": (
                            "How to handle index overflow"
                        ),
                    },
                ),
                "sort_subdirs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Sort subdirectories alphabetically"
                    ),
                }),
                "randomize_final": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Shuffle entire list after sorting"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "INT", "FLOAT")
    RETURN_NAMES = (
        "frames", "filename", "total_videos",
        "file_path", "frame_count", "fps",
    )
    OUTPUT_TOOLTIPS = (
        "Video frames as image batch",
        "Filename without extension",
        "Total number of videos found",
        "Full file path of loaded video",
        "Number of frames loaded",
        "Video FPS (frames per second)",
    )

    FUNCTION = "load_video"
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Load video files from a directory with natural sorting. "
        "Iterates through videos by index, loading frames as "
        "an IMAGE batch. Supports frame skipping and max frame "
        "limits for large videos."
    )

    def discover_files(
        self,
        directory: str,
        patterns: str,
    ) -> List[str]:
        """
        Discover video files matching the given patterns.

        Args:
            directory: Root directory to search
            patterns: Comma-separated glob patterns

        Returns:
            List of unique file paths
        """
        if not directory or not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        pattern_list = [
            p.strip() for p in patterns.split(',') if p.strip()
        ]
        if not pattern_list:
            pattern_list = ["**/*.mp4", "**/*.avi", "**/*.mov"]

        found_files = set()
        for pattern in pattern_list:
            full_pattern = os.path.join(directory, pattern)
            matches = glob.glob(full_pattern, recursive=True)
            for match in matches:
                ext = os.path.splitext(match)[1].lower()
                if (
                    ext in VALID_VIDEO_EXTENSIONS
                    and os.path.isfile(match)
                ):
                    found_files.add(match)

        if not found_files:
            raise FileNotFoundError(
                f"No videos found in '{directory}' "
                f"matching '{patterns}'"
            )

        return list(found_files)

    def sort_files(
        self,
        files: List[str],
        sort_by: str,
        sort_order: str,
        sort_subdirs: bool,
        randomize_final: bool,
    ) -> List[str]:
        """
        Sort files with subdirectory grouping.

        Args:
            files: List of file paths
            sort_by: Sorting method
            sort_order: ascending or descending
            sort_subdirs: Whether to sort subdirectories
            randomize_final: Shuffle entire list after sorting

        Returns:
            Sorted list of file paths
        """
        subdirs: Dict[str, List[str]] = {}
        for f in files:
            subdir = os.path.dirname(f)
            if subdir not in subdirs:
                subdirs[subdir] = []
            subdirs[subdir].append(f)

        if sort_subdirs:
            subdir_order = sorted(subdirs.keys())
        else:
            subdir_order = list(subdirs.keys())

        reverse = (sort_order == "descending")
        sort_key = get_sort_key(sort_by)

        sorted_files = []
        for subdir in subdir_order:
            subdir_files = subdirs[subdir]

            if sort_by == "random":
                random.shuffle(subdir_files)
            elif sort_by == "none":
                pass
            else:
                subdir_files.sort(key=sort_key, reverse=reverse)

            sorted_files.extend(subdir_files)

        if randomize_final:
            random.shuffle(sorted_files)

        return sorted_files

    def resolve_index(
        self,
        total_count: int,
        video_index: int,
        index_mode: str,
    ) -> int:
        """
        Resolve video index with overflow handling.

        Args:
            total_count: Total number of video files
            video_index: User-specified index
            index_mode: wrap, clamp, or error

        Returns:
            Resolved index

        Raises:
            IndexError: If mode is 'error' and out of bounds
        """
        if index_mode == "wrap":
            return video_index % total_count
        elif index_mode == "clamp":
            return max(0, min(video_index, total_count - 1))
        elif index_mode == "error":
            if video_index < 0 or video_index >= total_count:
                raise IndexError(
                    f"Video index {video_index} out of range "
                    f"[0, {total_count})"
                )
            return video_index
        return video_index % total_count

    def load_video_frames(
        self,
        file_path: str,
        frame_skip: int,
        max_frames: int,
        start_frame: int,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Load frames from a video file using OpenCV.

        Args:
            file_path: Path to the video file
            frame_skip: Frames to skip between loads
            max_frames: Max frames to load (0 = all)
            start_frame: First frame index to load

        Returns:
            Tuple of (frames_tensor, frame_count, fps)
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open video: {file_path}"
            )

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            step = frame_skip + 1
            frames = []
            frame_idx = start_frame

            while True:
                if max_frames > 0 and len(frames) >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % step == 0:
                    # BGR to RGB
                    frame_rgb = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )
                    frame_np = (
                        frame_rgb.astype(np.float32) / 255.0
                    )
                    frames.append(frame_np)

                frame_idx += 1

            if not frames:
                raise RuntimeError(
                    f"No frames loaded from video: {file_path}\n"
                    f"  Total frames: {total_frames}\n"
                    f"  Start frame: {start_frame}\n"
                    f"  Frame skip: {frame_skip}"
                )

            # Stack frames into batch tensor [N, H, W, C]
            frames_tensor = torch.from_numpy(
                np.stack(frames, axis=0)
            )

            return frames_tensor, len(frames), fps

        finally:
            cap.release()

    def load_video(
        self,
        directory: str,
        patterns: str,
        video_index: int,
        sort_by: str,
        sort_order: str,
        frame_skip: int = 0,
        max_frames: int = 0,
        start_frame: int = 0,
        index_mode: str = "wrap",
        sort_subdirs: bool = True,
        randomize_final: bool = False,
    ) -> Tuple[
        torch.Tensor, str, int, str, int, float
    ]:
        """
        Main execution function for the node.

        Args:
            directory: Root directory path
            patterns: Comma-separated glob patterns
            video_index: Index of video to load
            sort_by: Sorting method
            sort_order: Sort direction
            frame_skip: Frames to skip between loads
            max_frames: Max frames to load (0 = all)
            start_frame: First frame to load
            index_mode: Overflow handling mode
            sort_subdirs: Sort subdirectories alphabetically
            randomize_final: Shuffle after sorting

        Returns:
            Tuple of (frames, filename, total_videos,
                       file_path, frame_count, fps)
        """
        print(
            f"[VideoFolderCowboy] "
            f"Searching directory: {directory}"
        )
        print(
            f"[VideoFolderCowboy] "
            f"Using patterns: {patterns}"
        )
        files = self.discover_files(directory, patterns)
        print(
            f"[VideoFolderCowboy] Found {len(files)} videos"
        )

        sorted_files = self.sort_files(
            files, sort_by, sort_order,
            sort_subdirs, randomize_final,
        )

        total_videos = len(sorted_files)

        idx = self.resolve_index(
            total_videos, video_index, index_mode,
        )

        file_path = sorted_files[idx]
        basename = os.path.basename(file_path)
        filename = os.path.splitext(basename)[0]

        print(
            f"[VideoFolderCowboy] Loading video [{idx}]: "
            f"{basename}"
        )

        frames, frame_count, fps = self.load_video_frames(
            file_path, frame_skip, max_frames, start_frame,
        )

        print(
            f"[VideoFolderCowboy] Loaded {frame_count} frames "
            f"@ {fps:.1f} fps"
        )

        return (
            frames, filename, total_videos,
            file_path, frame_count, fps,
        )


NODE_CLASS_MAPPINGS = {
    "VideoFolderCowboy": VideoFolderCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFolderCowboy": "Video Folder Cowboy",
}
