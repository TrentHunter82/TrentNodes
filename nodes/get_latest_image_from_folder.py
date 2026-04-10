"""
Get Latest Image From Folder

Reads the most recent image from a folder and reports the
total image count. Designed for auto-queue iterative video
workflows where each run reads its own previous output
(e.g. MegaWarp stylization loop).

Critical: uses IS_CHANGED returning NaN so ComfyUI re-runs
this node on every queue execution instead of caching.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image, ImageOps


class GetLatestImageFromFolder:
    """
    Return the most recently modified image in a folder,
    plus the total count of matching files. Intended for
    iterative video workflows driven by ComfyUI's auto-
    queue: each queue run reads the previous frame's
    saved output, processes the next frame, and saves
    back to the same folder.
    """

    CATEGORY = "Trent/Video"
    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "image",
        "frame_count",
        "found",
        "filename",
    )
    OUTPUT_TOOLTIPS = (
        "The most recently modified image (or fallback)",
        "Total number of matching files in the folder "
        "(use as next frame index)",
        "True if a real image was loaded, False if the "
        "fallback was returned",
        "Basename of the loaded file (empty if fallback)",
    )
    FUNCTION = "get_latest"
    DESCRIPTION = (
        "Read the most recent image from a folder. "
        "Designed for iterative auto-queue workflows: "
        "wire the output folder of SaveImage back to "
        "this node to feed the previous frame into the "
        "next iteration. Handles empty folders via a "
        "fallback mode. Returns frame_count to drive "
        "the next video frame index."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Absolute or relative path "
                        "to the folder containing "
                        "images. Created if missing."
                    ),
                }),
            },
            "optional": {
                "extensions": ("STRING", {
                    "default": "png,jpg,jpeg,webp,"
                               "tiff,bmp",
                    "multiline": False,
                    "tooltip": (
                        "Comma-separated image "
                        "extensions to match"
                    ),
                }),
                "fallback_mode": (
                    ["passthrough", "black", "error"],
                    {
                        "default": "passthrough",
                        "tooltip": (
                            "What to return when "
                            "folder is empty. "
                            "passthrough = use the "
                            "fallback_image input. "
                            "black = 512x512 black. "
                            "error = raise exception"
                        ),
                    },
                ),
                "sort_by": (
                    ["mtime", "name"],
                    {
                        "default": "mtime",
                        "tooltip": (
                            "mtime = newest modified "
                            "file. name = highest "
                            "natural-sorted filename"
                        ),
                    },
                ),
                "filename_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Optional substring filter "
                        "on filenames (e.g. "
                        "\"megawarp_\"). Empty = all"
                    ),
                }),
                "fallback_image": ("IMAGE", {
                    "tooltip": (
                        "Image to return when folder "
                        "is empty and fallback_mode "
                        "is 'passthrough'"
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        folder_path,
        extensions="png,jpg,jpeg,webp,tiff,bmp",
        fallback_mode="passthrough",
        sort_by="mtime",
        filename_filter="",
        **kwargs,
    ):
        # Always re-execute so auto-queue loops see new
        # files as they appear. NaN != NaN in IEEE so
        # ComfyUI always treats this as dirty.
        return float("NaN")

    def get_latest(
        self,
        folder_path,
        extensions="png,jpg,jpeg,webp,tiff,bmp",
        fallback_mode="passthrough",
        sort_by="mtime",
        filename_filter="",
        fallback_image=None,
    ):
        folder_path = folder_path.strip()
        if not folder_path:
            raise ValueError(
                "folder_path is required"
            )

        # Create the folder if it doesn't exist so
        # first-run auto-queue works without manual
        # setup.
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        exts = [
            e.strip().lower().lstrip(".")
            for e in extensions.split(",")
            if e.strip()
        ]

        files = self._find_images(
            folder_path, exts, filename_filter
        )

        if not files:
            return self._handle_empty(
                fallback_mode, fallback_image
            )

        if sort_by == "mtime":
            latest_path = max(files, key=os.path.getmtime)
        else:
            files_sorted = sorted(
                files, key=self._natural_key
            )
            latest_path = files_sorted[-1]

        image_tensor = self._load_image(latest_path)
        filename = os.path.basename(latest_path)

        return (
            image_tensor,
            len(files),
            True,
            filename,
        )

    # ---- helpers ----

    @staticmethod
    def _find_images(folder, exts, name_filter):
        results = []
        try:
            entries = os.listdir(folder)
        except OSError:
            return results

        for name in entries:
            if name_filter and name_filter not in name:
                continue
            ext = name.rsplit(".", 1)[-1].lower() \
                if "." in name else ""
            if ext not in exts:
                continue
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                results.append(full)
        return results

    @staticmethod
    def _natural_key(path):
        import re
        name = os.path.basename(path)
        parts = re.split(r"(\d+)", name)
        return [
            int(p) if p.isdigit() else p.lower()
            for p in parts
        ]

    @staticmethod
    def _load_image(path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]
        return tensor

    @staticmethod
    def _handle_empty(mode, fallback_image):
        if mode == "error":
            raise RuntimeError(
                "No matching images found in folder "
                "and fallback_mode is 'error'"
            )
        if mode == "passthrough":
            if fallback_image is None:
                raise RuntimeError(
                    "fallback_mode is 'passthrough' "
                    "but no fallback_image was "
                    "connected. Wire your current "
                    "video frame to fallback_image."
                )
            return (fallback_image, 0, False, "")
        # black
        black = torch.zeros(
            1, 512, 512, 3, dtype=torch.float32
        )
        return (black, 0, False, "")


NODE_CLASS_MAPPINGS = {
    "GetLatestImageFromFolder": GetLatestImageFromFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetLatestImageFromFolder":
        "Get Latest Image From Folder",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
