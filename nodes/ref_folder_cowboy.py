"""
Ref Folder Cowboy - Load a folder of reference images onto fixed slots.

Built for MiniMax H3 ref_image wiring:
1. Loads up to six images from directory (or directory/<filename_key>)
   in natural-sort order onto six separate IMAGE outputs
2. Empty slots output None - MiniMaxH3ReferenceToVideo skips None
   ref images, so all six outputs stay wired and nothing needs muting
3. An empty or missing folder is NOT an error: all slots pass None
   (scenery chunks with no characters just run ref-less)
4. mtime-based IS_CHANGED so replaced refs bust the cache
"""

import glob
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from .image_folder_cowboy import natural_sort_key

VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
NUM_SLOTS = 6


class RefFolderCowboy:
    """
    Load a folder of reference images onto six fixed IMAGE outputs.

    Missing slots emit None, which optional inputs (like the MiniMax
    H3 ref_image sockets) treat as not connected.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/refs",
                    "tooltip": (
                        "Folder of reference images, or the base "
                        "folder when filename_key is connected"
                    ),
                }),
            },
            "optional": {
                "filename_key": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "If connected, load from "
                        "directory/<filename_key>/ - wire "
                        "VideoFolderCowboy.filename here so clip "
                        "000.mp4 loads refs_by_clip/000/"
                    ),
                }),
            },
        }

    RETURN_TYPES = tuple(["IMAGE"] * NUM_SLOTS + ["INT", "STRING"])
    RETURN_NAMES = tuple(
        [f"image_{i + 1}" for i in range(NUM_SLOTS)]
        + ["count", "names"]
    )
    OUTPUT_TOOLTIPS = tuple(
        [f"Reference image {i + 1} (None when the folder has fewer)"
         for i in range(NUM_SLOTS)]
        + ["Number of images found",
           "Newline-separated filenames in slot order"]
    )

    FUNCTION = "load_refs"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Load up to six reference images from a folder onto six "
        "fixed IMAGE outputs in natural-sort order. Empty slots "
        "output None, which the MiniMax H3 ref_image inputs "
        "silently skip - wire all six once and never mute. An "
        "empty or missing folder is not an error: everything "
        "passes None."
    )

    @classmethod
    def _resolve_dir(
        cls,
        directory: str,
        filename_key: str,
    ) -> str:
        if filename_key:
            return os.path.join(directory, filename_key)
        return directory

    @classmethod
    def _list_images(cls, folder: str) -> list:
        if not folder or not os.path.isdir(folder):
            return []
        files = [
            f for f in glob.glob(os.path.join(folder, "*"))
            if os.path.isfile(f)
            and os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
        ]
        files.sort(key=natural_sort_key)
        return files[:NUM_SLOTS]

    @classmethod
    def IS_CHANGED(
        cls,
        directory: str,
        filename_key: str = "",
    ) -> str:
        folder = cls._resolve_dir(directory, filename_key)
        files = cls._list_images(folder)
        try:
            stamp = ";".join(
                f"{f}:{os.path.getmtime(f)}" for f in files
            )
        except OSError:
            stamp = "unstat-able"
        return stamp or f"empty:{folder}"

    def _load_one(self, path: str) -> Optional[torch.Tensor]:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[RefFolderCowboy] Failed to read: {path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def load_refs(
        self,
        directory: str,
        filename_key: str = "",
    ) -> Tuple:
        folder = self._resolve_dir(directory, filename_key)
        files = self._list_images(folder)

        images = []
        names = []
        for f in files:
            t = self._load_one(f)
            if t is not None:
                images.append(t)
                names.append(os.path.basename(f))

        slots = images + [None] * (NUM_SLOTS - len(images))

        print(
            f"[RefFolderCowboy] {len(images)}/{NUM_SLOTS} refs "
            f"from {folder or '(unset)'}"
        )

        return tuple(
            slots + [len(images), "\n".join(names)]
        )


NODE_CLASS_MAPPINGS = {
    "RefFolderCowboy": RefFolderCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RefFolderCowboy": "Ref Folder Cowboy",
}
