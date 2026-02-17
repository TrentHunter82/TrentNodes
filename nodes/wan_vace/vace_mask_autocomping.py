"""
Vace Mask AutoComping node.

Composites solid gray over masked areas of input images,
outputting both the composited image and a matching clean
mask for use with Wan VACE inpainting workflows.
"""

import torch
from typing import Any, Dict, Tuple

from ...utils.mask_ops import dilate_mask


class VaceMaskAutoComping:
    """
    Composites gray over masked image regions for VACE.

    Takes images and masks, optionally expands the mask,
    then overlays solid gray on the masked areas. Outputs
    the composited images and a matching binary mask.
    """

    GRAY_VALUE = 0.5

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Input image or image batch to composite"
                        " gray over masked regions"
                    ),
                }),
                "masks": ("MASK", {
                    "tooltip": (
                        "Mask or mask batch defining regions"
                        " to cover with gray (white = cover)"
                    ),
                }),
            },
            "optional": {
                "expand_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": (
                        "Expand mask outward by this many"
                        " pixels before compositing"
                        " (hard edge, no feather)"
                    ),
                }),
                "gray_value": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Gray level for masked areas"
                        " (0.5 = mid gray, matches VACE"
                        " default filler)"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "Original images with solid gray composited"
        " over masked areas",
        "Clean binary mask: white (1) where gray was"
        " applied, black (0) elsewhere",
    )

    FUNCTION = "composite"
    CATEGORY = "Trent/Keyframes"
    DESCRIPTION = (
        "Composites solid gray over masked areas of input"
        " images for Wan VACE workflows. Saves manual"
        " compositing of gray-over-original for inpaint"
        " region prep. Optionally expands the mask."
    )

    def composite(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        expand_pixels: int = 0,
        gray_value: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Composite gray over masked image regions.

        Args:
            images: (B, H, W, C) image batch
            masks: (B, H, W) or (H, W) mask batch
            expand_pixels: Dilation radius in pixels
            gray_value: Gray level for fill

        Returns:
            Tuple of (composited images, binary masks)
        """
        b, h, w, c = images.shape
        device = images.device
        dtype = images.dtype

        # Normalize mask dimensions to (B, H, W)
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)

        # Broadcast single mask to match batch size
        if masks.shape[0] == 1 and b > 1:
            masks = masks.expand(b, -1, -1)

        # Resize mask if spatial dims don't match images
        if masks.shape[1] != h or masks.shape[2] != w:
            m = masks.unsqueeze(1)  # (B, 1, H, W)
            m = torch.nn.functional.interpolate(
                m, size=(h, w), mode='nearest'
            )
            masks = m.squeeze(1)  # (B, H, W)

        # Clamp batch dimension to match
        mask_b = min(masks.shape[0], b)
        if masks.shape[0] > b:
            masks = masks[:b]
        elif masks.shape[0] < b:
            # Repeat last mask for remaining frames
            pad = masks[-1:].expand(b - masks.shape[0], -1, -1)
            masks = torch.cat([masks, pad], dim=0)

        # Ensure float32 for operations
        masks = masks.to(dtype=torch.float32, device=device)

        # Expand mask if requested (hard edge dilation)
        if expand_pixels > 0:
            masks = dilate_mask(masks, expand_pixels)

        # Binarize mask (hard threshold, no feather)
        masks = (masks > 0.5).float()

        # Composite: original where mask=0, gray where mask=1
        # mask_4d shape: (B, H, W, 1) for broadcasting with
        # images (B, H, W, C)
        mask_4d = masks.unsqueeze(-1)
        gray = torch.full_like(images, gray_value)
        composited = images * (1.0 - mask_4d) + gray * mask_4d

        # Ensure output dtype matches input
        composited = composited.to(dtype=dtype)

        count = int(masks.sum(dim=(1, 2)).mean().item())
        total = h * w
        pct = count / total * 100.0 if total > 0 else 0.0
        print(
            f"[VaceMaskAutoComping] {b} frames, "
            f"expand={expand_pixels}px, "
            f"gray={gray_value:.2f}, "
            f"avg mask coverage={pct:.1f}%"
        )

        return (composited, masks)


NODE_CLASS_MAPPINGS = {
    "VaceMaskAutoComping": VaceMaskAutoComping,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VaceMaskAutoComping": "Vace Mask AutoComping",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
