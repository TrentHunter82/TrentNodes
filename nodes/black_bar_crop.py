"""
Black Bar Crop
Detects and removes black bars (letterbox/pillarbox) from images or video
frames, optionally center-cropping the result to an exact aspect ratio.
"""

import torch
from typing import Dict, Any, Tuple, Optional

from .black_bar_cinema import BlackBarCinemaScope

AUTO_RATIO = "Auto (remove bars only)"


class BlackBarCrop:
    """
    Removes black bars from an image batch by scanning inward from each
    edge, then optionally center-crops to an exact target aspect ratio.

    Detection runs across the whole batch at once so every frame of a
    video gets the same crop (no size jitter between frames).
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        ratio_choices = [AUTO_RATIO] + list(
            BlackBarCinemaScope.ASPECT_RATIOS.keys()
        )
        return {
            "required": {
                "image": ("IMAGE",),
                "target_ratio": (ratio_choices, {
                    "default": "16:9 (HD/UHD)",
                    "tooltip": (
                        "After removing bars, center-crop to this exact "
                        "aspect ratio. Auto only removes detected bars."
                    )
                }),
                "threshold": ("FLOAT", {
                    "default": 0.06,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.005,
                    "tooltip": (
                        "Brightness below this counts as black. Raise it "
                        "if compressed bars leave a thin gray edge."
                    )
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

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")

    FUNCTION = "crop_bars"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Detects letterbox/pillarbox black bars, crops them off, and "
        "optionally center-crops to an exact aspect ratio such as 16:9."
    )

    # A row/column counts as a bar if fewer than this fraction of its
    # pixels rise above the threshold (tolerates codec noise and logos).
    STRAY_PIXEL_FRACTION = 0.02

    # Refuse a detection that would discard more than this fraction of a
    # dimension — a mostly-dark scene, not a bar.
    MAX_BAR_FRACTION = 0.45

    # Refuse a detection that keeps less than this fraction of the frame
    # area — bars shrink one axis, they never leave a small island.
    MIN_KEPT_AREA = 0.25

    def _detect_content_box(
        self,
        image: torch.Tensor,
        threshold: float
    ) -> Tuple[int, int, int, int]:
        """
        Find the non-bar content region across the whole batch.

        Returns (top, bottom, left, right) as slice bounds.
        """
        _, src_h, src_w, _ = image.shape

        # Max brightness per pixel across all frames and channels: a
        # pixel is bar only if it stays black in every frame.
        brightness = image.amax(dim=(0, 3)).float()  # (H, W)
        lit = (brightness > threshold).float()

        row_lit = lit.mean(dim=1)  # fraction of lit pixels per row
        col_lit = lit.mean(dim=0)

        row_is_content = row_lit > self.STRAY_PIXEL_FRACTION
        col_is_content = col_lit > self.STRAY_PIXEL_FRACTION

        def bounds(is_content: torch.Tensor, size: int) -> Tuple[int, int]:
            idx = torch.nonzero(is_content, as_tuple=False)
            if idx.numel() == 0:
                return 0, size  # fully black input: leave untouched
            first = int(idx[0])
            last = int(idx[-1]) + 1
            max_bars = int(size * self.MAX_BAR_FRACTION)
            if first > max_bars or (size - last) > max_bars:
                return 0, size  # implausible bar: bail out
            return first, last

        top, bottom = bounds(row_is_content, src_h)
        left, right = bounds(col_is_content, src_w)

        kept_area = (bottom - top) * (right - left)
        if kept_area < self.MIN_KEPT_AREA * src_h * src_w:
            return 0, src_h, 0, src_w  # dark scene, not bars: bail out

        return top, bottom, left, right

    def _apply_target_ratio(
        self,
        top: int,
        bottom: int,
        left: int,
        right: int,
        target_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Shrink the box symmetrically to hit the target ratio exactly."""
        h = bottom - top
        w = right - left
        if h <= 0 or w <= 0:
            return top, bottom, left, right

        if w / h > target_ratio:
            # Too wide: trim the sides.
            new_w = max(2, round(h * target_ratio))
            new_w = new_w - (new_w % 2)
            trim = (w - new_w) // 2
            left += trim
            right = left + new_w
        else:
            # Too tall: trim top and bottom.
            new_h = max(2, round(w / target_ratio))
            new_h = new_h - (new_h % 2)
            trim = (h - new_h) // 2
            top += trim
            bottom = top + new_h

        return top, bottom, left, right

    def crop_bars(
        self,
        image: torch.Tensor,
        target_ratio: str,
        threshold: float,
        custom_ratio_width: Optional[float] = 0.0,
        custom_ratio_height: Optional[float] = 0.0,
    ) -> Tuple[torch.Tensor, int, int]:
        """Detect and crop black bars from the image batch."""
        if image.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B,H,W,C), got {image.dim()}D"
            )

        top, bottom, left, right = self._detect_content_box(image, threshold)

        custom_w = custom_ratio_width if custom_ratio_width else 0.0
        custom_h = custom_ratio_height if custom_ratio_height else 0.0

        if custom_w > 0 and custom_h > 0:
            ratio = custom_w / custom_h
            top, bottom, left, right = self._apply_target_ratio(
                top, bottom, left, right, ratio
            )
        elif target_ratio != AUTO_RATIO:
            rw, rh = BlackBarCinemaScope.ASPECT_RATIOS[target_ratio]
            top, bottom, left, right = self._apply_target_ratio(
                top, bottom, left, right, rw / rh
            )

        # Even dimensions for video encoders.
        if (bottom - top) % 2:
            bottom -= 1
        if (right - left) % 2:
            right -= 1

        cropped = image[:, top:bottom, left:right, :]
        return (cropped, right - left, bottom - top)


NODE_CLASS_MAPPINGS = {
    "BlackBarCrop": BlackBarCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlackBarCrop": "Black Bar Crop",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
