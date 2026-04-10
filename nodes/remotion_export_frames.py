"""
Remotion Export Frames - pushes ComfyUI IMAGE tensor frames into
a Remotion project's public/ directory as numbered image files.

Remotion compositions access these via staticFile().
"""

import os
from typing import Dict, Any, Tuple

import torch
from PIL import Image


class RemotionExportFrames:
    """
    Export ComfyUI IMAGE tensor frames to a Remotion project's
    public/ directory as numbered PNG or JPEG files.

    The exported frames can be loaded inside Remotion compositions
    using staticFile('subdirectory/frame_0000.png').
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "project": ("REMOTION_PROJECT",),
                "images": ("IMAGE",),
            },
            "optional": {
                "subdirectory": ("STRING", {
                    "default": "comfyui_frames",
                    "tooltip": (
                        "Subdirectory inside public/. Remotion "
                        "accesses via "
                        "staticFile('comfyui_frames/frame_0000.png')"
                    ),
                }),
                "format": (["png", "jpeg"], {"default": "png"}),
                "jpeg_quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "tooltip": (
                        "JPEG compression quality (only used when "
                        "format is jpeg)"
                    ),
                }),
                "filename_prefix": ("STRING", {
                    "default": "frame_",
                }),
                "zero_pad": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "tooltip": (
                        "Number of digits for zero-padded frame "
                        "numbers. 4 = frame_0000.png"
                    ),
                }),
                "clear_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Delete existing files in the subdirectory "
                        "before export"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("frame_dir", "frame_count")
    FUNCTION = "export_frames"
    CATEGORY = "Trent/Remotion Get Down"
    DESCRIPTION = (
        "Exports ComfyUI image batch frames to a Remotion "
        "project's public/ directory so compositions can "
        "reference them via staticFile()."
    )

    def export_frames(
        self,
        project: dict,
        images: torch.Tensor,
        subdirectory: str = "comfyui_frames",
        format: str = "png",
        jpeg_quality: int = 95,
        filename_prefix: str = "frame_",
        zero_pad: int = 4,
        clear_existing: bool = True,
    ) -> Tuple:
        out_path = os.path.join(
            project["public_dir"], subdirectory
        )
        os.makedirs(out_path, exist_ok=True)

        if clear_existing:
            for f in os.listdir(out_path):
                fp = os.path.join(out_path, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        frame_count = images.shape[0]
        ext = "jpg" if format == "jpeg" else "png"

        for i in range(frame_count):
            frame = images[i]  # (H, W, C)
            frame_np = (
                (frame * 255).clamp(0, 255).byte().cpu().numpy()
            )
            img = Image.fromarray(frame_np)

            num_str = str(i).zfill(zero_pad)
            filename = f"{filename_prefix}{num_str}.{ext}"
            filepath = os.path.join(out_path, filename)

            if format == "jpeg":
                img.save(filepath, quality=jpeg_quality)
            else:
                img.save(filepath)

        print(
            f"[Remotion Get Down] Exported {frame_count} frames "
            f"to public/{subdirectory}/"
        )

        return (subdirectory, frame_count)


NODE_CLASS_MAPPINGS = {
    "RemotionExportFrames": RemotionExportFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionExportFrames": "Remotion Export Frames",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
