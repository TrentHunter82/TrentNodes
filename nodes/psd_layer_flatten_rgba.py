"""
PSD Layer Flatten (RGBA)

Same as PSDLayerFlatten, but returns the composite as a
4-channel RGBA IMAGE tensor with the alpha channel
preserved, plus a MASK using standard ComfyUI convention
(white = opaque/selected, black = transparent).
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch

from .psd_utils import composite_layer_range


class PSDLayerFlattenRGBA:
    """
    Flatten a range of PSD layers into one RGBA composite.
    Identical inputs to PSDLayerFlatten but returns a
    4-channel image tensor and a standard-convention mask.
    """

    CATEGORY = "Trent/PSD"
    FUNCTION = "flatten"

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = (
        "Flattened RGBA composite of the selected layers "
        "(4-channel, alpha preserved)",
        "Alpha mask using standard ComfyUI convention: "
        "white = opaque/selected, black = transparent",
    )
    DESCRIPTION = (
        "Flatten a contiguous index range of PSD layers "
        "into a single RGBA composite. Returns a 4-channel "
        "image tensor with alpha preserved and a mask "
        "where white = opaque. Use this variant when "
        "downstream nodes need the alpha channel."
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
        canvas, _composited = composite_layer_range(
            folder_path=folder_path,
            start_index=start_index,
            end_index=end_index,
            background_color=background_color,
            respect_visibility=respect_visibility,
            log_prefix="[PSDLayerFlattenRGBA]",
        )

        rgba = np.array(canvas).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(rgba)[None, ...]

        # Standard ComfyUI mask convention:
        # white (1.0) = opaque/selected, black (0.0) = transparent.
        # NOTE: PSDLayerFlatten (the non-RGBA variant) inverts
        # this mask (1.0 - alpha). That may be a latent bug —
        # investigate whether downstream graphs rely on the
        # inversion before changing the original node.
        alpha = np.array(
            canvas.getchannel("A")
        ).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(alpha)

        return (image_tensor, mask_tensor)


NODE_CLASS_MAPPINGS = {
    "PSDLayerFlattenRGBA": PSDLayerFlattenRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDLayerFlattenRGBA": "PSD Layer Flatten (RGBA)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
