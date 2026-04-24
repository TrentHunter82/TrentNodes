"""
PSD Layer Flatten

Flatten a contiguous range of PSD layers into a single
rasterized image. Reads layer PNGs and metadata from a
PSDLayerSplitter output folder.
"""

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from .psd_utils import parse_background_color


class PSDLayerFlatten:
    """
    Flatten a range of PSD layers into one composited image.
    Reads the _manifest.json from PSDLayerSplitter for
    positions, opacity, and visibility. Wire start/end from
    PSDBackgroundDetect or set manually.
    """

    CATEGORY = "Trent/PSD"
    FUNCTION = "flatten"

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = (
        "Flattened RGBA composite of the selected layers",
        "Alpha mask of the flattened result (inverted: "
        "white = transparent)",
    )
    DESCRIPTION = (
        "Flatten a contiguous index range of PSD layers "
        "into a single rasterized image. Reads layer PNGs "
        "and positioning from a PSDLayerSplitter folder. "
        "Useful for merging multi-layer backgrounds into "
        "one image before further processing."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/layer/folder",
                    "tooltip": (
                        "Folder from PSDLayerSplitter "
                        "(must contain _manifest.json)"
                    ),
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "First layer index to include "
                        "(0 = bottom). Wire from "
                        "PSDBackgroundDetect bg_start."
                    ),
                }),
                "end_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Last layer index to include "
                        "(inclusive). Wire from "
                        "PSDBackgroundDetect bg_end."
                    ),
                }),
                "background_color": ("STRING", {
                    "default": "transparent",
                    "tooltip": (
                        "Canvas backdrop: 'transparent', "
                        "'white', 'black', or hex like "
                        "'#FFFFFF'"
                    ),
                }),
                "respect_visibility": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If True, skip layers marked "
                        "hidden in the original PSD"
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def flatten(
        self,
        folder_path,
        start_index,
        end_index,
        background_color,
        respect_visibility,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        folder_path = str(folder_path or "").strip()
        try:
            start_index = int(start_index)
        except (TypeError, ValueError):
            start_index = 0
        try:
            end_index = int(end_index)
        except (TypeError, ValueError):
            end_index = 0
        if end_index < start_index:
            start_index, end_index = end_index, start_index

        background_color = str(
            background_color or "transparent"
        )
        respect_visibility = bool(respect_visibility)

        if not folder_path:
            raise ValueError("folder_path is required")
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        manifest_path = os.path.join(
            folder_path, "_manifest.json"
        )
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"_manifest.json not found in {folder_path}"
            )

        with open(
            manifest_path, "r", encoding="utf-8"
        ) as f:
            manifest = json.load(f)

        canvas_w = int(manifest["canvas_width"])
        canvas_h = int(manifest["canvas_height"])
        layers = manifest.get("layers", [])
        if not layers:
            raise RuntimeError(
                "Manifest contains no layers"
            )

        sizing = (
            manifest.get("extraction_settings", {})
            .get("layer_sizing", "cropped")
        )

        bg_rgba = parse_background_color(background_color)
        canvas = Image.new(
            "RGBA", (canvas_w, canvas_h), bg_rgba
        )

        layers_sorted = sorted(
            layers, key=lambda lyr: lyr["index"]
        )

        composited = 0
        for lyr in layers_sorted:
            idx = int(lyr["index"])
            if idx < start_index or idx > end_index:
                continue

            if (respect_visibility
                    and not bool(lyr.get("visible", True))):
                continue

            filename = lyr["filename"]
            layer_path = os.path.join(folder_path, filename)
            if not os.path.isfile(layer_path):
                print(
                    f"[PSDLayerFlatten] Missing file, "
                    f"skipping: {filename}"
                )
                continue

            layer_img = Image.open(layer_path)
            layer_img = ImageOps.exif_transpose(layer_img)
            layer_img = layer_img.convert("RGBA")

            if sizing == "canvas":
                paste_x, paste_y = 0, 0
            else:
                paste_x = int(lyr["position"]["left"])
                paste_y = int(lyr["position"]["top"])

            opacity = int(lyr.get("opacity", 255))
            if opacity < 255:
                alpha = layer_img.getchannel("A")
                scale = opacity / 255.0
                alpha = alpha.point(
                    lambda v, s=scale: int(v * s)
                )
                layer_img.putalpha(alpha)

            canvas.paste(
                layer_img, (paste_x, paste_y), layer_img
            )
            composited += 1

        if composited == 0:
            print(
                f"[PSDLayerFlatten] No layers in range "
                f"{start_index}..{end_index} (all skipped "
                f"or missing)"
            )

        print(
            f"[PSDLayerFlatten] Flattened {composited} "
            f"layer(s) from range {start_index}..{end_index} "
            f"into {canvas_w}x{canvas_h}"
        )

        rgb = canvas.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr)[None, ...]

        alpha = np.array(
            canvas.getchannel("A")
        ).astype(np.float32) / 255.0
        mask_tensor = 1.0 - torch.from_numpy(alpha)

        return (image_tensor, mask_tensor)


NODE_CLASS_MAPPINGS = {
    "PSDLayerFlatten": PSDLayerFlatten,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDLayerFlatten": "PSD Layer Flatten",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

