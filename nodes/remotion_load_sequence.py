"""
Remotion Load Sequence - imports a directory of numbered image
files (from Remotion --sequence output) as a ComfyUI IMAGE tensor.
"""

from typing import Dict, Any, Tuple

from ..utils.remotion_utils import load_images_from_directory


class RemotionLoadSequence:
    """
    Load a directory of sequentially numbered images as a
    ComfyUI IMAGE tensor batch.

    Works with Remotion's --sequence output or any directory
    of numbered image files.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to the directory containing "
                        "numbered image files."
                    ),
                }),
            },
            "optional": {
                "pattern": ("STRING", {
                    "default": "*.png",
                    "tooltip": (
                        "Glob pattern for matching frame files. "
                        "E.g., '*.png', 'frame_*.jpg'"
                    ),
                }),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": (
                        "Maximum frames to load. 0 = all frames."
                    ),
                }),
                "start_index": ("INT", {
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": (
                        "Skip this many files from the sorted "
                        "file list."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "load_sequence"
    CATEGORY = "Trent/Remotion Get Down"
    DESCRIPTION = (
        "Loads a directory of sequentially numbered images as a "
        "ComfyUI IMAGE tensor batch. Works with Remotion "
        "--sequence output."
    )

    def load_sequence(
        self,
        directory: str,
        pattern: str = "*.png",
        max_frames: int = 0,
        start_index: int = 0,
    ) -> Tuple:
        directory = directory.strip()
        if not directory:
            raise ValueError("directory is empty")

        images = load_images_from_directory(
            directory,
            pattern=pattern,
            max_frames=max_frames,
            start_index=start_index,
        )

        frame_count = images.shape[0]

        print(
            f"[Remotion Get Down] Loaded {frame_count} frames "
            f"from {directory}"
        )

        return (images, frame_count)


NODE_CLASS_MAPPINGS = {
    "RemotionLoadSequence": RemotionLoadSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionLoadSequence": "Remotion Load Sequence",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
