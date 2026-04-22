"""
White To Whatever node.

Finds white (or near-white) pixels in an image or batch
of images and replaces them with a user-chosen color.

All operations are GPU-accelerated via torch tensor ops;
no per-pixel loops and no CPU-GPU round trips.
"""

import torch
from typing import Any, Dict, Tuple

import comfy.model_management as mm


def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    """
    Convert a hex color string to an (R, G, B) tuple in
    the range [0.0, 1.0]. Accepts '#RRGGBB', 'RRGGBB',
    '#RGB', or 'RGB'. Falls back to red on parse errors.
    """
    if not isinstance(hex_str, str):
        return (1.0, 0.0, 0.0)

    s = hex_str.strip().lstrip("#")

    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)

    if len(s) != 6:
        return (1.0, 0.0, 0.0)

    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
    except ValueError:
        return (1.0, 0.0, 0.0)

    return (r, g, b)


class WhiteToWhatever:
    """
    Replace white pixels in an image batch with any color.

    A pixel's "whiteness" is the minimum of its RGB
    channels: pure white = 1.0, any colored pixel has a
    lower value. Pixels at or above `threshold` are fully
    replaced; `softness` provides a smooth ramp below the
    threshold to avoid hard aliased edges.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Input image or batch; white"
                        " pixels will be recolored"
                    ),
                }),
                "color": ("STRING", {
                    "default": "#ff3366",
                    "tooltip": (
                        "Replacement color as a hex"
                        " string. Use the color swatch"
                        " to open a native picker"
                    ),
                }),
                "threshold": ("FLOAT", {
                    "default": 0.90,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "How white a pixel must be to"
                        " count. 1.0 = pure white only,"
                        " 0.9 = near-white, 0.5 = any"
                        " light pixel"
                    ),
                }),
                "softness": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Soft transition width below"
                        " the threshold for smooth"
                        " edges; 0 = hard cut"
                    ),
                }),
                "preserve_luminance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Scale the replacement color by"
                        " each pixel's original brightness"
                        " so shading is preserved instead"
                        " of a flat color fill"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "white_mask")
    OUTPUT_TOOLTIPS = (
        "Image with white pixels recolored",
        "Mask showing which pixels were treated"
        " as white (1.0 = fully replaced)",
    )

    FUNCTION = "recolor"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Find white pixels in an image or batch and"
        " replace them with any color. Includes a"
        " built-in color picker popup."
    )

    def recolor(
        self,
        images: torch.Tensor,
        color: str = "#ff3366",
        threshold: float = 0.90,
        softness: float = 0.05,
        preserve_luminance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replace near-white pixels with the chosen color.

        Args:
            images: (B, H, W, C) tensor in [0, 1]
            color: Hex color string, e.g. '#ff3366'
            threshold: Minimum whiteness for full replace
            softness: Soft ramp width below threshold
            preserve_luminance: Modulate color by original
                pixel brightness

        Returns:
            (recolored image, white mask)
        """
        device = mm.get_torch_device()
        images = images.to(device)

        b, h, w, c = images.shape
        dtype = images.dtype

        # Take only RGB; drop alpha if present
        rgb = images[..., :3]

        # Whiteness = min(R, G, B). Pure white -> 1.0,
        # any saturated pixel -> lower value
        whiteness = rgb.min(dim=-1).values  # (B, H, W)

        # Soft ramp: 0 below (threshold - softness),
        # 1 at threshold, linear in between
        if softness <= 0.0:
            mask = (whiteness >= threshold).to(dtype)
        else:
            lower = max(0.0, threshold - softness)
            mask = (
                (whiteness - lower) / (threshold - lower)
            ).clamp_(0.0, 1.0).to(dtype)

        # Build replacement color tensor, broadcast to
        # full batch shape with zero-copy expand()
        r, g, bl = hex_to_rgb(color)
        color_rgb = torch.tensor(
            (r, g, bl), device=device, dtype=dtype
        ).view(1, 1, 1, 3).expand(b, h, w, -1)

        if preserve_luminance:
            # Scale color by per-pixel brightness so
            # shaded areas of the original white region
            # remain shaded in the replacement
            lum = rgb.mean(dim=-1, keepdim=True)
            color_rgb = color_rgb * lum

        mask_4d = mask.unsqueeze(-1)  # (B, H, W, 1)

        # Fused lerp: one CUDA kernel for the composite
        result = torch.lerp(rgb, color_rgb, mask_4d)

        # If the input had alpha, preserve it
        if c == 4:
            alpha = images[..., 3:4]
            result = torch.cat([result, alpha], dim=-1)

        print(
            f"[WhiteToWhatever] {b} frame(s),"
            f" {h}x{w}, color={color},"
            f" threshold={threshold:.2f},"
            f" softness={softness:.2f},"
            f" preserve_lum={preserve_luminance}"
        )

        return (result, mask)


NODE_CLASS_MAPPINGS = {
    "WhiteToWhatever": WhiteToWhatever,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WhiteToWhatever": "White To Whatever",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
