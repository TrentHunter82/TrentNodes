"""
Video Layer Ho Down

Multi-layer compositing node with interactive canvas
positioning. Place up to 5 transparent layers on a
background with per-layer scale, opacity, blend mode,
and drag-to-position support. GPU-accelerated.
"""
import base64
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MAX_LAYERS = 5


class VideoLayerHoDown:
    """
    Composite up to 5 foreground layers onto a background
    with interactive drag positioning per layer.
    """

    CATEGORY = "Trent/Compositing"
    DISPLAY_NAME = "Video Layer Ho Down"
    DESCRIPTION = (
        "Composite up to 5 transparent layers onto a"
        " background. Drag layers on the canvas to"
        " position them. Supports RGBA, per-layer"
        " scale, opacity, and blend modes."
    )
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        blend_modes = [
            "normal", "multiply", "screen",
            "overlay", "add",
        ]
        return {
            "required": {
                "background": ("IMAGE", {
                    "tooltip": (
                        "Base layer; all foreground"
                        " layers composite on top"
                    ),
                }),
            },
            "optional": {
                "anchor_point": (
                    ["top_left", "center"],
                    {
                        "default": "top_left",
                        "tooltip": (
                            "Whether x,y refers to"
                            " the layer top-left"
                            " corner or center"
                        ),
                    },
                ),
                "layer_1": ("IMAGE", {
                    "tooltip": (
                        "Foreground layer 1;"
                        " supports RGBA for"
                        " transparency"
                    ),
                }),
                "layer_1_x": ("INT", {
                    "default": 0,
                    "min": -8192,
                    "max": 8192,
                    "step": 1,
                    "tooltip": (
                        "Layer 1 horizontal"
                        " position; set by"
                        " dragging on canvas"
                    ),
                }),
                "layer_1_y": ("INT", {
                    "default": 0,
                    "min": -8192,
                    "max": 8192,
                    "step": 1,
                    "tooltip": (
                        "Layer 1 vertical"
                        " position; set by"
                        " dragging on canvas"
                    ),
                }),
                "layer_1_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Layer 1 scale factor",
                }),
                "layer_1_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Layer 1 opacity",
                }),
                "layer_1_blend": (
                    blend_modes,
                    {
                        "default": "normal",
                        "tooltip": (
                            "Layer 1 blend mode"
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited",)
    OUTPUT_TOOLTIPS = (
        "Composited image batch with all layers"
        " blended onto background",
    )
    FUNCTION = "composite"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    @staticmethod
    def _encode_preview_jpeg(
        frame: torch.Tensor,
    ) -> str:
        """Encode a single frame to base64 JPEG."""
        img_np = (
            frame.cpu().numpy() * 255
        ).astype(np.uint8)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
        pil_img = Image.fromarray(img_np)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")

    @staticmethod
    def _encode_preview_png(
        frame: torch.Tensor,
    ) -> str:
        """Encode a single frame to base64 PNG
        (preserves alpha)."""
        img_np = (
            frame.cpu().numpy() * 255
        ).astype(np.uint8)
        if img_np.shape[-1] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
        pil_img = Image.fromarray(img_np, mode=mode)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")

    @staticmethod
    def _scale_layer(
        layer: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scale a layer batch via GPU interpolation."""
        if abs(scale - 1.0) < 1e-6:
            return layer
        bchw = layer.permute(0, 3, 1, 2)
        new_h = max(1, int(bchw.shape[2] * scale))
        new_w = max(1, int(bchw.shape[3] * scale))
        scaled = F.interpolate(
            bchw,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        return scaled.permute(0, 2, 3, 1)

    @staticmethod
    def _blend(
        background: torch.Tensor,
        foreground: torch.Tensor,
        alpha: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Blend foreground onto background."""
        alpha = alpha.clamp(0.0, 1.0)
        if mode == "normal":
            return torch.lerp(
                background, foreground, alpha
            )
        if mode == "multiply":
            blended = background * foreground
        elif mode == "screen":
            blended = (
                1.0
                - (1.0 - background)
                * (1.0 - foreground)
            )
        elif mode == "overlay":
            mask = background < 0.5
            blended = torch.where(
                mask,
                2.0 * background * foreground,
                1.0
                - 2.0
                * (1.0 - background)
                * (1.0 - foreground),
            )
        elif mode == "add":
            blended = (
                background + foreground
            ).clamp(0.0, 1.0)
        else:
            return torch.lerp(
                background, foreground, alpha
            )
        return torch.lerp(background, blended, alpha)

    @staticmethod
    def _match_batch(
        tensor: torch.Tensor,
        target_b: int,
    ) -> torch.Tensor:
        """Expand or repeat tensor to match batch."""
        b = tensor.shape[0]
        if b == target_b:
            return tensor
        if b == 1:
            return tensor.expand(
                target_b, *[-1] * (tensor.dim() - 1)
            )
        reps = (target_b + b - 1) // b
        return tensor.repeat(
            reps, *[1] * (tensor.dim() - 1)
        )[:target_b]

    # --------------------------------------------------
    # Main composite
    # --------------------------------------------------

    def composite(
        self,
        background: torch.Tensor,
        anchor_point: str = "top_left",
        unique_id: str = None,
        prompt: dict = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Composite all connected layers onto bg."""
        device = background.device
        bg = background.clone()
        bg_b, bg_h, bg_w = bg.shape[:3]

        # Get widget values from prompt data
        widget_vals = {}
        if prompt and unique_id:
            node_data = prompt.get(unique_id, {})
            widget_vals = node_data.get("inputs", {})

        # Collect connected layers in order
        layers = []
        for i in range(1, MAX_LAYERS + 1):
            key = f"layer_{i}"
            layer_img = kwargs.get(key)
            if layer_img is None:
                continue

            x = widget_vals.get(
                f"layer_{i}_x",
                kwargs.get(f"layer_{i}_x", 0),
            )
            y = widget_vals.get(
                f"layer_{i}_y",
                kwargs.get(f"layer_{i}_y", 0),
            )
            scale = widget_vals.get(
                f"layer_{i}_scale",
                kwargs.get(f"layer_{i}_scale", 1.0),
            )
            opacity = widget_vals.get(
                f"layer_{i}_opacity",
                kwargs.get(f"layer_{i}_opacity", 1.0),
            )
            blend = widget_vals.get(
                f"layer_{i}_blend",
                kwargs.get(
                    f"layer_{i}_blend", "normal"
                ),
            )

            if isinstance(x, float):
                x = int(x)
            if isinstance(y, float):
                y = int(y)
            if isinstance(scale, str):
                scale = float(scale)
            if isinstance(opacity, str):
                opacity = float(opacity)

            layers.append({
                "index": i,
                "image": layer_img.to(device),
                "x": x,
                "y": y,
                "scale": scale,
                "opacity": opacity,
                "blend": blend,
            })

        # Determine max batch size
        max_b = bg_b
        for lyr in layers:
            max_b = max(max_b, lyr["image"].shape[0])

        # Match background batch
        result = self._match_batch(bg, max_b)
        if not result.is_contiguous():
            result = result.contiguous()
        result = result.clone()

        # Encode previews
        bg_preview = self._encode_preview_jpeg(
            background[0]
        )
        layer_previews = []
        layer_sizes = []
        layer_positions = []

        # Composite each layer in order
        for lyr in layers:
            fg = lyr["image"]
            fg = self._match_batch(fg, max_b)
            if not fg.is_contiguous():
                fg = fg.contiguous()

            # Store original size for preview
            orig_h, orig_w = fg.shape[1], fg.shape[2]

            # Scale
            fg = self._scale_layer(fg, lyr["scale"])
            fg_h, fg_w = fg.shape[1], fg.shape[2]

            # Extract alpha
            if fg.shape[-1] == 4:
                alpha = fg[..., 3:4]
                fg = fg[..., :3]
            else:
                alpha = torch.ones(
                    max_b, fg_h, fg_w, 1,
                    device=device, dtype=fg.dtype,
                )

            alpha = alpha * lyr["opacity"]

            # Compute placement
            px = lyr["x"]
            py = lyr["y"]
            if anchor_point == "center":
                px = px - fg_w // 2
                py = py - fg_h // 2

            # Clipped bounds
            src_x1 = max(0, -px)
            src_y1 = max(0, -py)
            src_x2 = fg_w - max(
                0, px + fg_w - bg_w
            )
            src_y2 = fg_h - max(
                0, py + fg_h - bg_h
            )
            dst_x1 = max(0, px)
            dst_y1 = max(0, py)
            dst_x2 = min(bg_w, px + fg_w)
            dst_y2 = min(bg_h, py + fg_h)

            if dst_x1 < dst_x2 and dst_y1 < dst_y2:
                fg_region = fg[
                    :, src_y1:src_y2,
                    src_x1:src_x2, :
                ]
                a_region = alpha[
                    :, src_y1:src_y2,
                    src_x1:src_x2, :
                ]
                bg_region = result[
                    :, dst_y1:dst_y2,
                    dst_x1:dst_x2, :
                ]
                blended = self._blend(
                    bg_region, fg_region,
                    a_region, lyr["blend"],
                )
                result[
                    :, dst_y1:dst_y2,
                    dst_x1:dst_x2, :
                ] = blended

            # Preview data
            preview_frame = lyr["image"][0]
            layer_previews.append(
                self._encode_preview_png(
                    preview_frame
                )
            )
            layer_sizes.append([orig_w, orig_h])
            layer_positions.append(
                [lyr["x"], lyr["y"]]
            )

        layer_count = len(layers)
        print(
            f"[VideoLayerHoDown] {layer_count}"
            f" layer(s), {max_b} frame(s),"
            f" bg={bg_h}x{bg_w}"
        )

        return {
            "ui": {
                "bg_preview": [bg_preview],
                "layer_previews": layer_previews,
                "bg_size": [bg_w, bg_h],
                "layer_sizes": layer_sizes,
                "layer_positions": layer_positions,
            },
            "result": (result,),
        }


NODE_CLASS_MAPPINGS = {
    "VideoLayerHoDown": VideoLayerHoDown,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLayerHoDown": "Video Layer Ho Down",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
