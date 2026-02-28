"""
Just Pad or Crop It
Resize an image to match a reference by padding with gray or cropping.
"""

import torch
from typing import Dict, Any, Tuple


class JustPadOrCropIt:
    """
    Pad or crop an image to match a reference image's dimensions.

    Each axis is handled independently: axes smaller than the
    target are padded with a solid gray fill, axes larger are
    center-cropped.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "reference": ("IMAGE", {
                    "tooltip": (
                        "Reference image whose height and width "
                        "define the target size"
                    ),
                }),
                "image": ("IMAGE", {
                    "tooltip": (
                        "Image to pad or crop to match the "
                        "reference dimensions"
                    ),
                }),
            },
            "optional": {
                "gray_value": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Brightness of the padding fill",
                }),
                "alignment": (["center", "top-left"], {
                    "default": "center",
                    "tooltip": (
                        "Where to anchor the source image "
                        "within the target frame"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = (
        "Padded or cropped image matching reference dimensions",
        "Binary mask: 1.0 = real pixel, 0.0 = padded region",
    )

    FUNCTION = "execute"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Pad or crop an image to match a reference image's "
        "dimensions. Smaller axes get solid gray padding, "
        "larger axes get center-cropped."
    )

    def execute(
        self,
        reference: torch.Tensor,
        image: torch.Tensor,
        gray_value: float = 0.5,
        alignment: str = "center",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad or crop image to match reference dimensions."""
        if image.dim() != 4:
            raise ValueError(
                f"Expected 4D image (B,H,W,C), got {image.dim()}D"
            )

        _, h_target, w_target, _ = reference.shape
        batch, h_src, w_src, channels = image.shape
        device = image.device
        dtype = image.dtype
        center = alignment == "center"

        # -- Compute source region to copy (crop if oversized) --
        copy_h = min(h_src, h_target)
        copy_w = min(w_src, w_target)

        # Offsets into the source tensor (crop origin)
        src_y = (h_src - copy_h) // 2 if center else 0
        src_x = (w_src - copy_w) // 2 if center else 0

        # Offsets into the output tensor (pad origin)
        dst_y = (h_target - copy_h) // 2 if center else 0
        dst_x = (w_target - copy_w) // 2 if center else 0

        # -- Fast path: dimensions already match --
        if h_src == h_target and w_src == w_target:
            mask = torch.ones(
                (batch, h_target, w_target),
                dtype=dtype, device=device,
            )
            return (image, mask)

        # -- Build output filled with gray --
        output = torch.full(
            (batch, h_target, w_target, channels),
            gray_value, dtype=dtype, device=device,
        )

        # Slice-assign the copied region
        output[
            :,
            dst_y:dst_y + copy_h,
            dst_x:dst_x + copy_w,
            :,
        ] = image[
            :,
            src_y:src_y + copy_h,
            src_x:src_x + copy_w,
            :,
        ]

        # -- Build binary mask (1.0 = real, 0.0 = padded) --
        mask = torch.zeros(
            (batch, h_target, w_target),
            dtype=dtype, device=device,
        )
        mask[
            :,
            dst_y:dst_y + copy_h,
            dst_x:dst_x + copy_w,
        ] = 1.0

        return (output, mask)


NODE_CLASS_MAPPINGS = {
    "JustPadOrCropIt": JustPadOrCropIt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JustPadOrCropIt": "Just Pad or Crop It",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
