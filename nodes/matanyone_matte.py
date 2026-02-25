"""
MatAnyone Video Matte node.

Temporally-consistent video matting using MatAnyone (CVPR 2025).
Given a single initial mask, propagates it across all video
frames with memory-based temporal consistency. Produces
flicker-free alpha mattes for green screen compositing.

Uses BiRefNet for automatic initial mask generation when no
mask is provided. All compositing is GPU-accelerated via
torch.lerp.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

import comfy.model_management as mm

from ..utils.birefnet_wrapper import (
    birefnet_segment,
    clear_birefnet_cache,
)
from ..utils.matanyone_wrapper import (
    run_matanyone,
    clear_matanyone_cache,
)
from ..utils.mask_ops import (
    dilate_mask,
    erode_mask,
    feather_mask,
)


# Chroma key color RGB values (float32, 0-1 range)
CHROMA_COLORS = {
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "aqua": (0.0, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
}


class MatAnyoneMatte:
    """
    Temporally-consistent video matting with compositing.

    Uses MatAnyone to propagate a single-frame mask across
    all video frames, then composites the foreground over
    a chroma key or custom background.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Input video frames (image batch);"
                        " MatAnyone propagates the mask"
                        " across all frames with temporal"
                        " consistency"
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Initial foreground mask for the"
                        " reference frame; if not provided,"
                        " BiRefNet auto-segments frame 0"
                    ),
                }),
                "mask_frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Which frame the mask applies to;"
                        " 0 = first frame. MatAnyone"
                        " propagates forward from this"
                        " frame"
                    ),
                }),
                "n_warmup": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": (
                        "Warmup iterations on the reference"
                        " frame; builds initial memory."
                        " 10 is the paper default; more"
                        " = slightly better but slower"
                    ),
                }),
                "bg_color": (
                    list(CHROMA_COLORS.keys()) + ["none"],
                    {
                        "default": "green",
                        "tooltip": (
                            "Background color for composite;"
                            " 'none' skips compositing and"
                            " returns original images"
                        ),
                    },
                ),
                "background_images": ("IMAGE", {
                    "tooltip": (
                        "Optional custom background images;"
                        " overrides bg_color when connected."
                        " Auto-resized to match input dims."
                        " A single image is broadcast to"
                        " all frames"
                    ),
                }),
                "edge_refine": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": (
                        "Gaussian feather radius for soft"
                        " matte edges; 0 = hard edges"
                    ),
                }),
                "mask_expand": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "tooltip": (
                        "Grow or shrink the matte boundary;"
                        " positive expands foreground,"
                        " negative shrinks it"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite", "alpha_matte")
    OUTPUT_TOOLTIPS = (
        "Foreground composited over the selected"
        " background color or custom background",
        "Temporally-consistent alpha matte from"
        " MatAnyone (white = foreground)",
    )

    FUNCTION = "matte"
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Temporally-consistent video matting using"
        " MatAnyone (CVPR 2025). Propagates a single"
        " mask across all frames with memory-based"
        " temporal consistency for flicker-free results."
    )

    def matte(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_frame_index: int = 0,
        n_warmup: int = 10,
        bg_color: str = "green",
        background_images: Optional[torch.Tensor] = None,
        edge_refine: int = 1,
        mask_expand: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MatAnyone video matting and composite.

        Args:
            images: (B, H, W, C) video frames in [0, 1]
            mask: Optional (B, H, W) or (H, W) initial mask
            mask_frame_index: Reference frame for the mask
            n_warmup: Warmup iterations
            bg_color: Chroma key color or 'none'
            background_images: Optional custom background
            edge_refine: Feather radius in pixels
            mask_expand: Mask boundary adjustment

        Returns:
            Tuple of (composited image, alpha matte)
        """
        device = mm.get_torch_device()
        images = images.to(device)
        b, h, w, c = images.shape
        dtype = images.dtype

        # Phase 1: Get initial mask
        if mask is None:
            # Auto-segment with BiRefNet
            needed = 4 * (768 ** 2) * 3 * 4 * 6
            mm.free_memory(needed, device)
            mm.soft_empty_cache()

            frame_idx = min(mask_frame_index, b - 1)
            init_mask = birefnet_segment(
                images[frame_idx:frame_idx + 1],
                device,
                resolution=768,
                model_variant="lite",
            )

            if init_mask is None:
                print(
                    "[MatAnyoneMatte] BiRefNet not"
                    " available for auto-mask."
                    " Provide a mask input or install:"
                    " pip install transformers"
                )
                empty = torch.zeros(
                    b, h, w,
                    dtype=dtype, device=device,
                )
                return (images, empty)

            # Single frame mask: (1, H, W) -> (H, W)
            init_mask = init_mask.squeeze(0)

            # Free BiRefNet before loading MatAnyone
            clear_birefnet_cache()
        else:
            # Use provided mask
            init_mask = mask.to(
                device=device, dtype=torch.float32
            )
            if init_mask.dim() == 3:
                # Take the mask_frame_index slice or first
                idx = min(
                    mask_frame_index, init_mask.shape[0] - 1
                )
                init_mask = init_mask[idx]

        # Phase 2: Free VRAM for MatAnyone (~9 GB)
        mm.unload_all_models()
        mm.soft_empty_cache()

        # Phase 3: Run MatAnyone inference
        alpha = run_matanyone(
            images, init_mask, device,
            n_warmup=n_warmup,
            mask_frame_index=mask_frame_index,
        )

        if alpha is None:
            print(
                "[MatAnyoneMatte] MatAnyone not available."
                " Install: pip install omegaconf"
            )
            empty = torch.zeros(
                b, h, w, dtype=dtype, device=device,
            )
            return (images, empty)

        # Free MatAnyone VRAM
        clear_matanyone_cache()

        # Phase 4: Post-process matte
        if mask_expand > 0:
            alpha = dilate_mask(alpha, mask_expand)
        elif mask_expand < 0:
            alpha = erode_mask(alpha, abs(mask_expand))

        if edge_refine > 0:
            alpha = feather_mask(alpha, edge_refine, device)

        alpha = alpha.clamp_(0.0, 1.0)

        # Phase 5: Composite
        bg_label = bg_color
        if bg_color == "none":
            result = images[..., :3]
        else:
            alpha_4d = alpha.unsqueeze(-1)  # (B, H, W, 1)

            if background_images is not None:
                # Custom background
                bg = background_images.to(
                    device=device, dtype=dtype
                )
                bg = bg[..., :3]
                bg_b, bg_h, bg_w, _ = bg.shape

                if bg_h != h or bg_w != w:
                    bg = F.interpolate(
                        bg.permute(0, 3, 1, 2),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

                if bg_b == 1 and b > 1:
                    bg = bg.expand(b, -1, -1, -1)
                elif bg_b < b:
                    pad = bg[-1:].expand(
                        b - bg_b, -1, -1, -1
                    )
                    bg = torch.cat([bg, pad], dim=0)
                elif bg_b > b:
                    bg = bg[:b]

                bg_label = "custom"
            else:
                color_rgb = CHROMA_COLORS.get(
                    bg_color, (0.0, 1.0, 0.0)
                )
                bg = torch.tensor(
                    color_rgb, device=device, dtype=dtype
                ).view(1, 1, 1, 3).expand(b, h, w, -1)
                bg_label = bg_color

            result = torch.lerp(
                bg, images[..., :3], alpha_4d
            )

        print(
            f"[MatAnyoneMatte] {b} frame(s),"
            f" {h}x{w}, bg={bg_label},"
            f" warmup={n_warmup},"
            f" mask_frame={mask_frame_index},"
            f" expand={mask_expand}px,"
            f" feather={edge_refine}px"
        )

        return (result, alpha)


NODE_CLASS_MAPPINGS = {
    "MatAnyoneMatte": MatAnyoneMatte,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyoneMatte": "MatAnyone Video Matte",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
