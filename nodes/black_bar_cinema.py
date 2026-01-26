"""
Black Bar Cinema Scope
Adds cinematic black bars (letterbox) to images for widescreen aspect ratios.
"""

import torch
from typing import Dict, Any, Tuple, Optional


class BlackBarCinemaScope:
    """
    Adds cinematic black bars to achieve widescreen aspect ratios.

    Supports letterboxing (top/bottom bars) and pillarboxing (side bars)
    to fit any source image into common cinematic aspect ratios.
    """

    ASPECT_RATIOS = {
        "16:9 (HD/UHD)": (16, 9),
        "1.85:1 (Flat Widescreen)": (185, 100),
        "2.00:1 (Univisium)": (2, 1),
        "2.20:1 (Todd-AO)": (220, 100),
        "2.35:1 (Scope/Cinemascope)": (235, 100),
        "2.39:1 (Anamorphic)": (239, 100),
        "2.76:1 (Ultra Panavision)": (276, 100),
        "21:9 (Ultrawide)": (21, 9),
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(cls.ASPECT_RATIOS.keys()), {
                    "default": "16:9 (HD/UHD)",
                    "tooltip": "Target cinematic aspect ratio"
                }),
            },
            "optional": {
                "custom_ratio_width": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": (
                        "Custom aspect width (set both width and height > 0 "
                        "to override preset)"
                    )
                }),
                "custom_ratio_height": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Custom aspect height"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "bar_mask")

    FUNCTION = "apply_cinema_bars"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Adds cinematic black bars to achieve widescreen aspect ratios. "
        "Outputs the letterboxed image and a mask of the bar regions."
    )

    def _calculate_target_ratio(
        self,
        aspect_ratio: str,
        custom_width: float,
        custom_height: float
    ) -> float:
        """Calculate the target aspect ratio from inputs."""
        if custom_width > 0 and custom_height > 0:
            return custom_width / custom_height

        ratio_tuple = self.ASPECT_RATIOS[aspect_ratio]
        return ratio_tuple[0] / ratio_tuple[1]

    def _calculate_padding(
        self,
        src_h: int,
        src_w: int,
        target_ratio: float
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Calculate padding needed to achieve target aspect ratio.

        Returns:
            (new_h, new_w, pad_top, pad_bottom, pad_left, pad_right)
        """
        current_ratio = src_w / src_h

        # Check if already at target (within tolerance)
        if abs(current_ratio - target_ratio) < 0.001:
            return src_h, src_w, 0, 0, 0, 0

        if current_ratio > target_ratio:
            # Source is wider - add letterbox (top/bottom)
            new_w = src_w
            new_h = round(src_w / target_ratio)
            # Ensure even dimensions for video compatibility
            new_h = new_h + (new_h % 2)
            pad_top = (new_h - src_h) // 2
            pad_bottom = new_h - src_h - pad_top
            pad_left = 0
            pad_right = 0
        else:
            # Source is narrower - add pillarbox (left/right)
            new_h = src_h
            new_w = round(src_h * target_ratio)
            new_w = new_w + (new_w % 2)
            pad_left = (new_w - src_w) // 2
            pad_right = new_w - src_w - pad_left
            pad_top = 0
            pad_bottom = 0

        return new_h, new_w, pad_top, pad_bottom, pad_left, pad_right

    def apply_cinema_bars(
        self,
        image: torch.Tensor,
        aspect_ratio: str,
        custom_ratio_width: Optional[float] = 0.0,
        custom_ratio_height: Optional[float] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cinematic black bars to the image batch."""
        if image.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B,H,W,C), got {image.dim()}D"
            )

        batch_size, src_h, src_w, channels = image.shape
        device = image.device
        dtype = image.dtype

        # Handle None values from optional inputs
        custom_w = custom_ratio_width if custom_ratio_width else 0.0
        custom_h = custom_ratio_height if custom_ratio_height else 0.0

        target_ratio = self._calculate_target_ratio(
            aspect_ratio, custom_w, custom_h
        )
        new_h, new_w, pad_top, pad_bottom, pad_left, pad_right = \
            self._calculate_padding(src_h, src_w, target_ratio)

        # Fast path: no padding needed
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            # Create empty mask (no bars)
            mask = torch.zeros(
                (batch_size, src_h, src_w),
                dtype=dtype,
                device=device
            )
            return (image, mask)

        # Create output tensor (black fill)
        output = torch.zeros(
            (batch_size, new_h, new_w, channels),
            dtype=dtype,
            device=device
        )

        # Place source image in the center
        output[
            :,
            pad_top:pad_top + src_h,
            pad_left:pad_left + src_w,
            :
        ] = image

        # Create bar mask (1.0 where bars are, 0.0 where image is)
        mask = torch.ones(
            (batch_size, new_h, new_w),
            dtype=dtype,
            device=device
        )
        mask[
            :,
            pad_top:pad_top + src_h,
            pad_left:pad_left + src_w
        ] = 0.0

        return (output, mask)


NODE_CLASS_MAPPINGS = {
    "BlackBarCinemaScope": BlackBarCinemaScope,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlackBarCinemaScope": "Black Bar Cinema Scope",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
