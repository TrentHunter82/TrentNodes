"""
Easiest Green Screen node.

One-click background removal and chroma key replacement.
Uses BiRefNet AI to segment the foreground, then composites
it over a solid color background. Works with image batches.

All operations are GPU-accelerated: BiRefNet inference,
mask morphology (max_pool2d / conv2d), and compositing
(torch.lerp fused kernel). No CPU-GPU transfers during
processing.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

import comfy.model_management as mm

from ..utils.birefnet_wrapper import birefnet_segment
from ..utils.mask_ops import (
    dilate_mask,
    erode_mask,
    feather_mask,
    temporal_smooth as temporal_smooth_fn,
)


# Chroma key color RGB values (float32, 0-1 range)
CHROMA_COLORS = {
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "aqua": (0.0, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0),
}

# Resolution dropdown -> integer mapping
RESOLUTION_MAP = {
    "fast (512)": 512,
    "balanced (768)": 768,
    "quality (1024)": 1024,
}


class EasiestGreenScreen:
    """
    Remove background and replace with chroma key color.

    Uses BiRefNet AI segmentation with optional edge refinement
    for clean, production-ready green screen composites.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Input image or image batch;"
                        " backgrounds will be replaced with"
                        " the selected chroma key color"
                    ),
                }),
            },
            "optional": {
                "bg_color": (
                    list(CHROMA_COLORS.keys()),
                    {
                        "default": "green",
                        "tooltip": (
                            "Background replacement color:"
                            " green and blue are standard"
                            " chroma key colors"
                        ),
                    },
                ),
                "model_variant": (
                    ["lite", "standard"],
                    {
                        "default": "lite",
                        "tooltip": (
                            "BiRefNet model quality: lite"
                            " is faster, standard gives"
                            " finer edge detail"
                        ),
                    },
                ),
                "resolution": (
                    list(RESOLUTION_MAP.keys()),
                    {
                        "default": "balanced (768)",
                        "tooltip": (
                            "Processing resolution for"
                            " segmentation; higher gives"
                            " sharper mask edges but uses"
                            " more VRAM"
                        ),
                    },
                ),
                "edge_refine": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": (
                        "Gaussian feather radius in pixels"
                        " for soft mask edges; 0 for hard"
                        " edges, 2-5 for natural blending"
                    ),
                }),
                "mask_expand": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "tooltip": (
                        "Grow or shrink the foreground mask"
                        " boundary in pixels; positive"
                        " expands foreground (keeps more),"
                        " negative shrinks it (removes"
                        " edge fringe)"
                    ),
                }),
                "temporal_smooth": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 15,
                    "step": 1,
                    "tooltip": (
                        "Temporal smoothing window for"
                        " video batches; reduces mask"
                        " flickering between frames."
                        " 0 = off, 3-7 = subtle,"
                        " 9-15 = heavy. Has no effect"
                        " on single images"
                    ),
                }),
                "background_images": ("IMAGE", {
                    "tooltip": (
                        "Optional custom background image"
                        " or batch; overrides bg_color"
                        " when connected. Auto-resized"
                        " to match input dimensions. A"
                        " single image is broadcast to"
                        " all frames"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = (
        "Image with background replaced by the selected"
        " chroma key color",
        "Foreground mask from BiRefNet segmentation"
        " (white = subject, black = background)",
    )

    FUNCTION = "green_screen"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "One-click background removal and chroma key"
        " replacement. Uses BiRefNet AI to segment the"
        " foreground, then composites it over a solid"
        " color background. Works with image batches."
    )

    def green_screen(
        self,
        images: torch.Tensor,
        bg_color: str = "green",
        model_variant: str = "lite",
        resolution: str = "balanced (768)",
        edge_refine: int = 2,
        mask_expand: int = 0,
        temporal_smooth: int = 0,
        background_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remove background and composite over chroma key.

        All ops run on GPU with no CPU-GPU transfers.
        Uses torch.lerp (single fused CUDA kernel) for
        compositing instead of separate multiply/add ops.

        Args:
            images: (B, H, W, C) image batch in [0, 1]
            bg_color: Chroma key color name
            model_variant: BiRefNet quality tier
            resolution: Processing resolution string
            edge_refine: Feather radius in pixels
            mask_expand: Mask boundary adjustment
            temporal_smooth: Temporal blur window (0=off)
            background_images: Optional (B, H, W, C) or
                (1, H, W, C) custom background

        Returns:
            Tuple of (composited image, foreground mask)
        """
        # Ensure GPU execution
        device = mm.get_torch_device()
        images = images.to(device)

        b, h, w, c = images.shape
        dtype = images.dtype

        # Parse resolution from dropdown string
        res_int = RESOLUTION_MAP.get(resolution, 768)

        # Free VRAM before inference: ask ComfyUI to unload
        # cached models (diffusion, VAE, etc.) so BiRefNet
        # has room for activations. ~1 GiB per frame at 768
        # with sub-batch of 4 = ~4 GiB headroom needed,
        # plus ~500 MiB for model weights.
        needed = 4 * (res_int ** 2) * 3 * 4 * 6  # conservative
        mm.free_memory(needed, device)
        mm.soft_empty_cache()

        # Run BiRefNet segmentation -> (B, H, W) mask
        # Sub-batched (max 4 frames per forward pass) to
        # cap peak VRAM regardless of input batch size
        mask = birefnet_segment(
            images, device,
            resolution=res_int,
            model_variant=model_variant,
        )

        if mask is None:
            print(
                "[EasiestGreenScreen] BiRefNet not available."
                " Install: pip install transformers"
            )
            empty_mask = torch.zeros(
                b, h, w, dtype=dtype, device=device
            )
            return (images, empty_mask)

        # Adjust mask boundary (dilate or erode)
        # GPU-accelerated via F.max_pool2d
        if mask_expand > 0:
            mask = dilate_mask(mask, mask_expand)
        elif mask_expand < 0:
            mask = erode_mask(mask, abs(mask_expand))

        # Temporal smoothing: 1D Gaussian blur along time
        # axis to reduce per-frame mask flickering in video
        if temporal_smooth >= 3 and b >= 3:
            kernel_size = temporal_smooth
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask = temporal_smooth_fn(
                mask,
                kernel_size=kernel_size,
                sigma=kernel_size / 3.0,
            )

        # Feather edges for smooth compositing
        # GPU-accelerated via separable F.conv2d
        if edge_refine > 0:
            mask = feather_mask(mask, edge_refine, device)

        # Clamp after morphological ops
        mask = mask.clamp_(0.0, 1.0)

        # Build background tensor (B, H, W, 3)
        if background_images is not None:
            # Custom background: move to GPU, resize to match
            bg = background_images.to(device=device, dtype=dtype)
            bg = bg[..., :3]
            bg_b, bg_h, bg_w, _ = bg.shape

            # Resize if spatial dims don't match
            if bg_h != h or bg_w != w:
                bg = F.interpolate(
                    bg.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1)

            # Broadcast single bg image to all frames
            if bg_b == 1 and b > 1:
                bg = bg.expand(b, -1, -1, -1)
            elif bg_b < b:
                # Repeat last frame for remaining
                pad = bg[-1:].expand(b - bg_b, -1, -1, -1)
                bg = torch.cat([bg, pad], dim=0)
            elif bg_b > b:
                bg = bg[:b]

            bg_label = "custom"
        else:
            # Solid color: expand() is a zero-copy view
            color_rgb = CHROMA_COLORS.get(
                bg_color, (0.0, 1.0, 0.0)
            )
            bg = torch.tensor(
                color_rgb, device=device, dtype=dtype
            ).view(1, 1, 1, 3).expand(b, h, w, -1)
            bg_label = bg_color

        # Composite using torch.lerp: single fused CUDA
        # kernel instead of separate mul/add ops
        # lerp(bg, fg, weight) = bg + weight * (fg - bg)
        mask_4d = mask.unsqueeze(-1)  # (B, H, W, 1)
        result = torch.lerp(bg, images[..., :3], mask_4d)

        print(
            f"[EasiestGreenScreen] {b} frame(s),"
            f" {h}x{w}, bg={bg_label},"
            f" model={model_variant},"
            f" res={res_int},"
            f" expand={mask_expand}px,"
            f" feather={edge_refine}px,"
            f" temporal={temporal_smooth}"
        )

        return (result, mask)


NODE_CLASS_MAPPINGS = {
    "EasiestGreenScreen": EasiestGreenScreen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasiestGreenScreen": "Easiest Green Screen",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
