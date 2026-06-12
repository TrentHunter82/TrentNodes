"""
Wan Vace Gray Gap Joiner
Joins two image batches with a run of gray filler frames in between,
plus a matching mask batch for Wan Vace (0 = preserve, 1 = generate).
"""

import torch
from typing import Any, Dict, Tuple


class WanVaceGrayGapJoiner:
    """
    Concatenates two image batches with gray filler frames between them.

    Useful for VACE temporal inpainting: feed two clips and let the
    model generate the transition across the gray gap frames.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images_a": ("IMAGE", {
                    "tooltip": "First image batch (defines output resolution)"
                }),
                "images_b": ("IMAGE", {
                    "tooltip": "Second image batch (resized to match images_a if needed)"
                }),
                "gap_frames": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Number of gray filler frames inserted between the two batches"
                }),
            },
            "optional": {
                "filler_color": (["gray", "green"], {
                    "default": "gray",
                    "tooltip": "Color for filler frames: gray (0.5, Wan VACE default) or pure green (chroma-key style)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "images_a + gap filler frames + images_b as one batch",
        "Matching masks: 0 for real frames (preserve), 1 for filler frames (generate)"
    )

    FUNCTION = "join"
    CATEGORY = "Trent/Keyframes"
    DESCRIPTION = "Joins two image batches with gray filler frames (and VACE masks) in between."

    FILLER_COLORS = {
        "gray": (0.5, 0.5, 0.5),
        "green": (0.0, 1.0, 0.0),
    }

    def _match_to(
        self,
        images: torch.Tensor,
        h: int,
        w: int,
        c: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resize/convert a batch to (N, h, w, c) float32 on device."""
        images = images.to(dtype=torch.float32, device=device)

        if images.shape[1] != h or images.shape[2] != w:
            images = images.permute(0, 3, 1, 2)  # BHWC -> BCHW
            images = torch.nn.functional.interpolate(
                images,
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )
            images = images.permute(0, 2, 3, 1)  # BCHW -> BHWC

        # Harmonize channel count (e.g. RGBA vs RGB)
        if images.shape[3] > c:
            images = images[..., :c]
        elif images.shape[3] < c:
            pad = torch.ones(
                (*images.shape[:3], c - images.shape[3]),
                dtype=images.dtype, device=device,
            )
            images = torch.cat([images, pad], dim=3)

        return images

    def join(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
        gap_frames: int,
        filler_color: str = "gray",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = images_a.shape[1]
        w = images_a.shape[2]
        c = images_a.shape[3]
        device = images_a.device

        images_a = images_a.to(dtype=torch.float32)
        images_b = self._match_to(images_b, h, w, c, device)

        # .get() so a stale value from an old saved workflow falls back to gray
        filler_rgb = self.FILLER_COLORS.get(filler_color, self.FILLER_COLORS["gray"])
        filler = torch.zeros((gap_frames, h, w, c), dtype=torch.float32, device=device)
        for ch in range(c):
            filler[..., ch] = filler_rgb[ch] if ch < 3 else 1.0

        images_out = torch.cat([images_a, filler, images_b], dim=0)

        # VACE mask convention: 0 = preserve real frames, 1 = generate fillers
        masks_out = torch.cat([
            torch.zeros((images_a.shape[0], h, w), dtype=torch.float32, device=device),
            torch.ones((gap_frames, h, w), dtype=torch.float32, device=device),
            torch.zeros((images_b.shape[0], h, w), dtype=torch.float32, device=device),
        ], dim=0)

        print(
            f"[WanVaceGrayGapJoiner] {images_a.shape[0]} + {gap_frames} {filler_color}"
            f" + {images_b.shape[0]} = {images_out.shape[0]} frames"
            f" ({w}x{h}, {c}ch)"
        )

        return (images_out, masks_out)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "WanVaceGrayGapJoiner": WanVaceGrayGapJoiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVaceGrayGapJoiner": "Wan Vace Gray Gap Joiner",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
