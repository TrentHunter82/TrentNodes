"""
MatAnyone Video Matte node.

Temporally-consistent video matting using MatAnyone 2
(CVPR 2026) or MatAnyone (CVPR 2025). Given a single
initial mask, propagates it across all video frames with
memory-based temporal consistency. Produces flicker-free
alpha mattes for green screen compositing.

Features:
- Bidirectional propagation from reference frame
- Auto best-frame selection for clearest subject
- Person-aware auto-masking (BiRefNet standard +
  largest connected component filtering)
- Guided filter edge refinement (preserves hair detail)

Uses BiRefNet for automatic initial mask generation when
no mask is provided. All compositing is GPU-accelerated
via torch.lerp.
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
from ..utils.matanyone2_wrapper import (
    run_matanyone2,
    clear_matanyone2_cache,
)
from ..utils.mask_ops import (
    dilate_mask,
    erode_mask,
    feather_mask,
    guided_filter,
    largest_connected_component,
)


# Chroma key color RGB values (float32, 0-1 range)
CHROMA_COLORS = {
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "aqua": (0.0, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
}


def _auto_select_ref_frame(
    images: torch.Tensor,
    device: torch.device,
    n_samples: int = 16,
) -> int:
    """
    Auto-select the best reference frame for matting.

    Scores sampled frames by sharpness (Laplacian
    variance) to find the frame with the clearest
    subject. Sharp frames have well-defined edges,
    minimal motion blur, and clear foreground detail.

    Args:
        images: (B, H, W, C) video frames in [0, 1]
        device: torch device
        n_samples: Max frames to evaluate

    Returns:
        Index of the best reference frame
    """
    b = images.shape[0]
    if b <= 1:
        return 0

    # Sample up to n_samples frames evenly
    if b <= n_samples:
        indices = list(range(b))
    else:
        indices = torch.linspace(
            0, b - 1, n_samples
        ).long().tolist()

    # Laplacian kernel for edge/sharpness detection
    laplacian = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3, 3)

    best_idx = 0
    best_score = -1.0

    for idx in indices:
        frame = images[idx]  # (H, W, C)
        # Grayscale luminance
        gray = (
            frame[..., 0] * 0.299
            + frame[..., 1] * 0.587
            + frame[..., 2] * 0.114
        )
        gray_4d = gray.unsqueeze(0).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        edges = F.conv2d(gray_4d, laplacian, padding=1)
        score = edges.var().item()

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


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
                "model_version": (
                    ["v2 (CVPR 2026)", "v1 (CVPR 2025)"],
                    {
                        "default": "v2 (CVPR 2026)",
                        "tooltip": (
                            "MatAnyone 2 (CVPR 2026) adds"
                            " a learned quality evaluator"
                            " for better fine detail and"
                            " robustness; v1 is the"
                            " original stable model"
                        ),
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Initial foreground mask for the"
                        " reference frame; if not provided,"
                        " BiRefNet auto-segments the best"
                        " frame with person-aware filtering"
                    ),
                }),
                "mask_frame_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Reference frame for the mask;"
                        " -1 = auto-select sharpest"
                        " frame (recommended); 0+ = manual"
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
                    list(CHROMA_COLORS.keys()) + [
                        "transparent", "none"
                    ],
                    {
                        "default": "green",
                        "tooltip": (
                            "Background color for composite;"
                            " 'transparent' outputs RGBA with"
                            " alpha channel embedded;"
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
                "edge_mode": (
                    ["guided", "feather"],
                    {
                        "default": "guided",
                        "tooltip": (
                            "'guided' uses the RGB image to"
                            " preserve hair and fine edges;"
                            " 'feather' applies Gaussian blur"
                            " (softer but loses detail)"
                        ),
                    },
                ),
                "edge_refine": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": (
                        "Edge refinement radius; for guided"
                        " mode preserves detail at edges;"
                        " for feather mode blurs edges."
                        " 0 = no refinement"
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
        " background color or custom background;"
        " RGBA (4-channel) when bg_color=transparent",
        "Temporally-consistent alpha matte from"
        " MatAnyone (white = foreground)",
    )

    FUNCTION = "matte"
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Temporally-consistent video matting using"
        " MatAnyone 2 (CVPR 2026) or MatAnyone"
        " (CVPR 2025). Auto-selects the best reference"
        " frame, generates a person-aware mask, and"
        " propagates bidirectionally with guided-filter"
        " edge refinement for clean hair and fine detail."
    )

    def matte(
        self,
        images: torch.Tensor,
        model_version: str = "v2 (CVPR 2026)",
        mask: Optional[torch.Tensor] = None,
        mask_frame_index: int = -1,
        n_warmup: int = 10,
        bg_color: str = "green",
        background_images: Optional[torch.Tensor] = None,
        edge_mode: str = "guided",
        edge_refine: int = 4,
        mask_expand: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MatAnyone video matting and composite.

        Args:
            images: (B, H, W, C) video frames in [0, 1]
            model_version: "v2 (CVPR 2026)" or
                "v1 (CVPR 2025)"
            mask: Optional (B, H, W) or (H, W) initial
                mask
            mask_frame_index: Reference frame (-1 = auto)
            n_warmup: Warmup iterations
            bg_color: Chroma key color or 'none'
            background_images: Optional custom background
            edge_mode: "guided" or "feather"
            edge_refine: Refinement radius in pixels
            mask_expand: Mask boundary adjustment

        Returns:
            Tuple of (composited image, alpha matte)
        """
        device = mm.get_torch_device()
        images = images.to(device)
        b, h, w, c = images.shape
        dtype = images.dtype

        # Phase 1: Auto-select reference frame
        if mask_frame_index < 0:
            if mask is not None:
                # Mask provided but auto frame: use 0
                mask_frame_index = 0
            else:
                mask_frame_index = _auto_select_ref_frame(
                    images, device,
                )
                print(
                    f"[MatAnyoneMatte] Auto-selected"
                    f" ref frame: {mask_frame_index}"
                )

        # Phase 2: Get initial mask
        if mask is None:
            # Person-aware auto-segment with BiRefNet
            # standard at 1024px for maximum quality
            needed = 4 * (1024 ** 2) * 3 * 4 * 6
            mm.free_memory(needed, device)
            mm.soft_empty_cache()

            frame_idx = min(mask_frame_index, b - 1)
            init_mask = birefnet_segment(
                images[frame_idx:frame_idx + 1],
                device,
                resolution=1024,
                model_variant="standard",
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

            # Keep only the largest connected component
            # to isolate the primary person and discard
            # small spurious segments
            init_mask = largest_connected_component(
                init_mask
            )

            print(
                "[MatAnyoneMatte] Auto-mask generated"
                " (BiRefNet standard 1024px + largest"
                " component)"
            )

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
                    mask_frame_index,
                    init_mask.shape[0] - 1,
                )
                init_mask = init_mask[idx]

        # Phase 3: Free VRAM for MatAnyone (~9 GB)
        mm.unload_all_models()
        mm.soft_empty_cache()

        # Phase 4: Run inference (bidirectional)
        use_v2 = "v2" in model_version
        if use_v2:
            alpha = run_matanyone2(
                images, init_mask, device,
                n_warmup=n_warmup,
                mask_frame_index=mask_frame_index,
                bidirectional=True,
            )
            clear_fn = clear_matanyone2_cache
            model_label = "MatAnyone2"
        else:
            alpha = run_matanyone(
                images, init_mask, device,
                n_warmup=n_warmup,
                mask_frame_index=mask_frame_index,
                bidirectional=True,
            )
            clear_fn = clear_matanyone_cache
            model_label = "MatAnyone"

        if alpha is None:
            print(
                f"[MatAnyoneMatte] {model_label} not"
                " available."
                " Install: pip install omegaconf"
            )
            empty = torch.zeros(
                b, h, w, dtype=dtype, device=device,
            )
            return (images, empty)

        # Free MatAnyone VRAM
        clear_fn()

        # Phase 5: Post-process matte
        if mask_expand > 0:
            alpha = dilate_mask(alpha, mask_expand)
        elif mask_expand < 0:
            alpha = erode_mask(alpha, abs(mask_expand))

        if edge_refine > 0:
            if edge_mode == "guided":
                alpha = guided_filter(
                    images, alpha,
                    radius=edge_refine,
                )
            else:
                alpha = feather_mask(
                    alpha, edge_refine, device,
                )

        alpha = alpha.clamp_(0.0, 1.0)

        # Phase 6: Composite
        bg_label = bg_color
        if bg_color == "none":
            result = images[..., :3]
        elif bg_color == "transparent":
            # RGBA output: RGB foreground + alpha channel
            result = torch.cat(
                [images[..., :3], alpha.unsqueeze(-1)],
                dim=-1,
            )
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
            f"[MatAnyoneMatte] {model_label},"
            f" {b} frame(s), {h}x{w},"
            f" bg={bg_label},"
            f" warmup={n_warmup},"
            f" ref_frame={mask_frame_index},"
            f" edge={edge_mode}({edge_refine}px),"
            f" expand={mask_expand}px,"
            f" bidirectional=True"
        )

        return (result, alpha)


NODE_CLASS_MAPPINGS = {
    "MatAnyoneMatte": MatAnyoneMatte,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyoneMatte": "MatAnyone Video Matte",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
