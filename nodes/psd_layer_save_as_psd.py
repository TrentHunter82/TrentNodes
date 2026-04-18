"""
PSD Layer Save As PSD

Open an existing .psd, replace one layer's pixels with a
provided image, and write the result to a new .psd file.

Unlike PSDLayerCompositor (which outputs a flat IMAGE),
this node preserves the entire original layer tree -
text layers, smart objects, adjustment layers, layer
effects, masks, blend modes, and group structure - and
only swaps the pixels of one named layer.

The replaced layer becomes a flat PixelLayer (rasterized).
That's the right tradeoff for swapping a background image.
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image

from psd_tools import PSDImage

from .psd_utils import (
    replace_psd_layer_pixels,
    tensor_to_pil_rgba,
)


def composite_to_tensor(psd: PSDImage) -> torch.Tensor:
    """Render PSD to a (1,H,W,3) float tensor for preview."""
    pil = psd.composite()
    if pil is None:
        # Fall back to a black canvas matching PSD size
        pil = Image.new(
            "RGB", (psd.width, psd.height), (0, 0, 0)
        )
    pil = pil.convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


class PSDLayerSaveAsPSD:
    """
    Save a new .psd with one layer's pixels replaced.

    Reads the original PSD, locates the target layer by
    name (recursively walks groups), swaps it for a new
    PixelLayer built from the provided image, and writes
    the modified PSD to disk. All other layers, including
    text, smart objects, effects, and groups, are
    preserved.
    """

    CATEGORY = "Trent/PSD"
    FUNCTION = "save_psd"
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_path", "preview")
    OUTPUT_TOOLTIPS = (
        "Absolute path of the written .psd file",
        "Composite preview of the modified PSD",
    )
    DESCRIPTION = (
        "Open an original .psd, replace one layer's "
        "pixels with the provided image, and save as a "
        "new .psd. Preserves every other layer (text, "
        "smart objects, adjustment layers, effects, "
        "masks, blend modes, group structure). The "
        "replaced layer is rasterized to a PixelLayer."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "original_psd_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/original.psd",
                    "tooltip": (
                        "Source .psd to read structure "
                        "from. Not modified."
                    ),
                }),
                "replacement_image": ("IMAGE", {
                    "tooltip": (
                        "New pixels for the target layer"
                    ),
                }),
                "target_layer_name": ("STRING", {
                    "default": "",
                    "placeholder": "Background",
                    "tooltip": (
                        "Name of the layer to replace "
                        "(matches PSD layer names; walks "
                        "into groups). First match wins."
                    ),
                }),
                "output_psd_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/output.psd",
                    "tooltip": (
                        "Destination .psd path. Parent "
                        "directory is created if missing."
                    ),
                }),
                "resize_mode": (
                    ["stretch", "fit", "cover", "center", "native"],
                    {
                        "default": "stretch",
                        "tooltip": (
                            "How to fit the replacement "
                            "into the original layer's "
                            "bounding box. 'native' keeps "
                            "the replacement's own size "
                            "(positioned at the original "
                            "layer's top-left)."
                        ),
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def save_psd(
        self,
        original_psd_path,
        replacement_image,
        target_layer_name,
        output_psd_path,
        resize_mode,
        **kwargs,
    ) -> Tuple[str, torch.Tensor]:
        original_psd_path = str(original_psd_path or "").strip()
        target_layer_name = str(target_layer_name or "").strip()
        output_psd_path = str(output_psd_path or "").strip()
        if resize_mode not in (
            "stretch", "fit", "cover", "center", "native"
        ):
            resize_mode = "stretch"

        if not original_psd_path:
            raise ValueError("original_psd_path is required")
        if not os.path.isfile(original_psd_path):
            raise FileNotFoundError(
                f"PSD not found: {original_psd_path}"
            )
        if not target_layer_name:
            raise ValueError("target_layer_name is required")
        if not output_psd_path:
            raise ValueError("output_psd_path is required")

        # Make sure the destination directory exists.
        out_dir = os.path.dirname(
            os.path.abspath(output_psd_path)
        )
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        psd = PSDImage.open(original_psd_path)

        replacement_pil = tensor_to_pil_rgba(replacement_image)
        old_name = replace_psd_layer_pixels(
            psd,
            target_layer_name,
            replacement_pil,
            resize_mode,
        )

        preview_tensor = composite_to_tensor(psd)

        with open(output_psd_path, "wb") as fp:
            psd.save(fp)

        print(
            f"[PSDLayerSaveAsPSD] Replaced layer "
            f"'{old_name}' and wrote "
            f"{output_psd_path}"
        )

        return (os.path.abspath(output_psd_path), preview_tensor)


NODE_CLASS_MAPPINGS = {
    "PSDLayerSaveAsPSD": PSDLayerSaveAsPSD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDLayerSaveAsPSD": "PSD Layer Save As PSD",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
