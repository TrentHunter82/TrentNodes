"""
CorridorKey Green Screen Keyer node.

Neural green screen keying using CorridorKey (Corridor
Digital). Instead of producing binary masks, it unmixes
true foreground color from the green screen background,
preserving semi-transparent details like hair, motion
blur, and out-of-focus edges.

Outputs clean straight foreground color, a linear alpha
matte, and a composited preview. Uses BiRefNet for
automatic alpha hint generation when no mask is provided.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

import comfy.model_management as mm

from ..utils.birefnet_wrapper import (
    birefnet_segment,
    clear_birefnet_cache,
)
from ..utils.corridorkey_wrapper import (
    run_corridorkey,
    clear_corridorkey_cache,
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


class CorridorKeyKeyer:
    """
    Neural green screen keyer with color unmixing.

    Uses CorridorKey to separate foreground from green
    screen background, producing clean foreground color
    and alpha matte rather than just a binary mask.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Green screen footage (image"
                        " batch); CorridorKey unmixes"
                        " foreground color from the"
                        " green background per frame"
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Alpha hint / garbage matte;"
                        " rough black-and-white mask"
                        " isolating the subject. If not"
                        " provided, BiRefNet auto-"
                        "generates one per frame"
                    ),
                }),
                "bg_color": (
                    list(CHROMA_COLORS.keys()) + [
                        "transparent", "none"
                    ],
                    {
                        "default": "green",
                        "tooltip": (
                            "Background color for"
                            " composite output;"
                            " 'transparent' outputs"
                            " RGBA with alpha embedded;"
                            " 'none' returns original"
                            " images unchanged"
                        ),
                    },
                ),
                "background_images": ("IMAGE", {
                    "tooltip": (
                        "Optional custom background;"
                        " overrides bg_color when"
                        " connected. Auto-resized to"
                        " match input. Single image is"
                        " broadcast to all frames"
                    ),
                }),
                "refiner_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": (
                        "CNN refiner strength; 1.0 is"
                        " standard. Higher pushes edge"
                        " detail but may introduce"
                        " artifacts"
                    ),
                }),
                "despill_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": (
                        "Green spill removal; 1.0 is"
                        " full despill, 0.0 disables."
                        " Removes green fringing on"
                        " edges and translucent areas"
                    ),
                }),
                "input_is_linear": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Enable if input is linear"
                        " color space (EXR). Default"
                        " assumes sRGB/REC709 camera"
                        " footage"
                    ),
                }),
                "auto_despeckle": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Auto-remove small disconnected"
                        " alpha regions like tracking"
                        " markers and background noise"
                    ),
                }),
                "despeckle_size": ("INT", {
                    "default": 400,
                    "min": 0,
                    "max": 2000,
                    "step": 50,
                    "tooltip": (
                        "Minimum pixel area to keep;"
                        " alpha regions smaller than"
                        " this are removed. Higher ="
                        " more aggressive cleanup"
                    ),
                }),
                "edge_refine": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": (
                        "Gaussian feather radius for"
                        " soft matte edges; 0 = sharp"
                        " edges from CorridorKey"
                    ),
                }),
                "mask_expand": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "tooltip": (
                        "Grow or shrink the matte"
                        " boundary; positive expands"
                        " foreground, negative shrinks"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("composite", "alpha_matte", "foreground")
    OUTPUT_TOOLTIPS = (
        "Foreground composited over the selected"
        " background color or custom background;"
        " RGBA (4-channel) when bg_color=transparent",
        "Clean linear alpha matte from CorridorKey"
        " (white = foreground)",
        "Straight (un-premultiplied) foreground color"
        " with green screen removed; useful for"
        " re-compositing in other tools",
    )

    FUNCTION = "key"
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Neural green screen keying using CorridorKey"
        " (Corridor Digital). Unmixes true foreground"
        " color from the green background, preserving"
        " semi-transparent details like hair and motion"
        " blur. Outputs clean foreground, alpha matte,"
        " and composite."
    )

    def key(
        self,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bg_color: str = "green",
        background_images: Optional[torch.Tensor] = None,
        refiner_scale: float = 1.0,
        despill_strength: float = 1.0,
        input_is_linear: bool = False,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        edge_refine: int = 0,
        mask_expand: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run CorridorKey green screen keying.

        Args:
            images: (B, H, W, C) video frames [0, 1]
            mask: Optional (B, H, W) alpha hints
            bg_color: Composite background selection
            background_images: Custom background
            refiner_scale: CNN refiner strength
            despill_strength: Green spill removal
            input_is_linear: Linear color input
            auto_despeckle: Remove small artifacts
            despeckle_size: Min area to keep
            edge_refine: Feather radius in pixels
            mask_expand: Mask boundary adjustment

        Returns:
            Tuple of (composite, alpha_matte, foreground)
        """
        device = mm.get_torch_device()
        images = images.to(device)
        b, h, w, c = images.shape
        dtype = images.dtype

        # Phase 1: Get alpha hint masks
        if mask is None:
            # Auto-segment with BiRefNet
            needed = 4 * (768 ** 2) * 3 * 4 * 6
            mm.free_memory(needed, device)
            mm.soft_empty_cache()

            alpha_hints = birefnet_segment(
                images,
                device,
                resolution=768,
                model_variant="lite",
            )

            if alpha_hints is None:
                print(
                    "[CorridorKey] BiRefNet not"
                    " available for auto-mask."
                    " Provide a mask input or"
                    " install: pip install"
                    " transformers"
                )
                empty_mask = torch.zeros(
                    b, h, w,
                    dtype=dtype, device=device,
                )
                return (images, empty_mask, images)

            # Free BiRefNet before loading CorridorKey
            clear_birefnet_cache()
        else:
            # Use provided mask(s)
            alpha_hints = mask.to(
                device=device, dtype=torch.float32
            )
            if alpha_hints.dim() == 2:
                # Single (H, W) mask -> broadcast
                alpha_hints = alpha_hints.unsqueeze(
                    0
                ).expand(b, -1, -1)
            elif alpha_hints.shape[0] == 1 and b > 1:
                alpha_hints = alpha_hints.expand(
                    b, -1, -1
                )

        # Ensure masks match spatial dims
        if (
            alpha_hints.shape[1] != h
            or alpha_hints.shape[2] != w
        ):
            alpha_hints = F.interpolate(
                alpha_hints.unsqueeze(1),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

        # Phase 2: Free VRAM for CorridorKey (~3-5 GB)
        mm.unload_all_models()
        mm.soft_empty_cache()

        # Phase 3: Run CorridorKey inference
        result = run_corridorkey(
            images, alpha_hints, device,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
        )

        if result is None:
            print(
                "[CorridorKey] Model not available."
                " Install: pip install timm"
            )
            empty_mask = torch.zeros(
                b, h, w, dtype=dtype, device=device,
            )
            return (images, empty_mask, images)

        foreground, alpha, processed = result

        # Free CorridorKey VRAM
        clear_corridorkey_cache()

        # Phase 4: Post-process matte
        if mask_expand > 0:
            alpha = dilate_mask(alpha, mask_expand)
        elif mask_expand < 0:
            alpha = erode_mask(alpha, abs(mask_expand))

        if edge_refine > 0:
            alpha = feather_mask(
                alpha, edge_refine, device
            )

        alpha = alpha.clamp_(0.0, 1.0)

        # Phase 5: Composite foreground over background
        bg_label = bg_color
        if bg_color == "none":
            composite = images[..., :3]
        elif bg_color == "transparent":
            # RGBA: foreground + alpha channel
            composite = torch.cat(
                [foreground, alpha.unsqueeze(-1)],
                dim=-1,
            )
        else:
            alpha_4d = alpha.unsqueeze(-1)  # (B,H,W,1)

            if background_images is not None:
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
                    color_rgb, device=device,
                    dtype=dtype,
                ).view(1, 1, 1, 3).expand(
                    b, h, w, -1
                )

            composite = torch.lerp(
                bg, foreground, alpha_4d
            )

        print(
            f"[CorridorKey] {b} frame(s), {h}x{w},"
            f" bg={bg_label},"
            f" refiner={refiner_scale},"
            f" despill={despill_strength},"
            f" expand={mask_expand}px,"
            f" feather={edge_refine}px"
        )

        return (composite, alpha, foreground)


NODE_CLASS_MAPPINGS = {
    "CorridorKeyKeyer": CorridorKeyKeyer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CorridorKeyKeyer": "CorridorKey Green Screen Keyer",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
