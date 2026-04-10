"""
Remotion Load Video - imports a rendered video file back into
ComfyUI as an IMAGE tensor batch.

Uses ffmpeg for frame extraction and supports frame slicing,
stepping, and resizing.
"""

import shutil
from typing import Dict, Any, Tuple

import torch

from ..utils.remotion_utils import (
    extract_video_frames,
    load_images_from_directory,
    probe_video,
)


class RemotionLoadVideo:
    """
    Load a rendered video file and convert it to a ComfyUI
    IMAGE tensor (batch of frames).

    Uses ffmpeg to extract frames, then stacks them into a
    (B, H, W, 3) float32 tensor.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to the video file to load. "
                        "Connect from Remotion Render output."
                    ),
                }),
            },
            "optional": {
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": (
                        "Maximum frames to extract. "
                        "0 = all frames."
                    ),
                }),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": "First frame number to extract.",
                }),
                "frame_step": ("INT", {
                    "default": 1, "min": 1, "max": 100,
                    "tooltip": (
                        "Extract every Nth frame. "
                        "1 = every frame, 2 = every other."
                    ),
                }),
                "resize_width": ("INT", {
                    "default": 0, "min": 0, "max": 7680,
                    "tooltip": (
                        "Resize frames during extraction. "
                        "0 = original size."
                    ),
                }),
                "resize_height": ("INT", {
                    "default": 0, "min": 0, "max": 4320,
                    "tooltip": (
                        "Resize frames during extraction. "
                        "0 = original size."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "load_video"
    CATEGORY = "Trent/Remotion Get Down"
    DESCRIPTION = (
        "Loads a rendered video file and converts it to a "
        "ComfyUI IMAGE tensor batch. Supports frame slicing, "
        "stepping, and resizing."
    )

    def load_video(
        self,
        video_path: str,
        max_frames: int = 0,
        start_frame: int = 0,
        frame_step: int = 1,
        resize_width: int = 0,
        resize_height: int = 0,
    ) -> Tuple:
        video_path = video_path.strip()
        if not video_path:
            raise ValueError("video_path is empty")

        info = probe_video(video_path)
        fps = info["fps"]

        frames_dir, frame_count, _ = extract_video_frames(
            video_path,
            max_frames=max_frames,
            start_frame=start_frame,
            frame_step=frame_step,
            resize_width=resize_width,
            resize_height=resize_height,
        )

        try:
            images = load_images_from_directory(
                frames_dir, pattern="frame_*.png"
            )
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

        actual_count = images.shape[0]

        print(
            f"[Remotion Get Down] Loaded {actual_count} frames "
            f"from {video_path} ({fps:.1f} fps)"
        )

        return (images, actual_count, fps)


NODE_CLASS_MAPPINGS = {
    "RemotionLoadVideo": RemotionLoadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionLoadVideo": "Remotion Load Video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
