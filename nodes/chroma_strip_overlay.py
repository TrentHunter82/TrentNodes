"""
Chroma Strip Overlay
Composites an image onto a chroma key green strip placed on the
side of a video frame, for easy keying in NLEs like Premiere Pro.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple


# Chroma key green in 0-1 float RGB
CHROMA_GREEN = (0.0, 1.0, 0.0)


class ChromaStripOverlay:
    """
    Places a chroma key green strip on one side of a video frame
    and composites an overlay image within that strip, scaled to
    fit with configurable padding and vertical position.
    """

    SIDES = ["left", "right"]
    MODES = ["overlay", "expand"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "side": (cls.SIDES, {
                    "default": "left",
                    "tooltip": "Which side of the frame to place "
                               "the green strip"
                }),
                "mode": (cls.MODES, {
                    "default": "overlay",
                    "tooltip": "overlay: strip covers the video "
                               "edge. expand: strip adds extra "
                               "width to the frame"
                }),
                "strip_width_pct": ("FLOAT", {
                    "default": 25.0,
                    "min": 5.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Width of the green strip as a "
                               "percentage of total frame width"
                }),
                "image_padding_pct": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 40.0,
                    "step": 0.5,
                    "tooltip": "Padding around the overlay image "
                               "inside the strip, as percentage "
                               "of strip width"
                }),
                "vertical_position": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Vertical position of overlay "
                               "image: 0.0=top, 0.5=center, "
                               "1.0=bottom"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_chroma_strip"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Places a chroma key green strip on the side of a video "
        "frame with an overlay image composited on top. Designed "
        "for keying in NLEs like Premiere Pro."
    )

    def _resize_overlay(
        self,
        overlay: torch.Tensor,
        target_w: int,
        target_h: int,
    ) -> torch.Tensor:
        """
        Resize overlay to fit within target dimensions while
        preserving aspect ratio. Returns (1, H, W, C) tensor.
        """
        # overlay shape: (1, H, W, C) - take first frame
        img = overlay[0:1]  # (1, H, W, C)
        oh, ow = img.shape[1], img.shape[2]

        scale = min(target_w / ow, target_h / oh)
        new_w = max(1, round(ow * scale))
        new_h = max(1, round(oh * scale))

        # F.interpolate expects (B, C, H, W)
        img_bchw = img.permute(0, 3, 1, 2)
        resized = F.interpolate(
            img_bchw,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1)  # back to (1, H, W, C)

    def apply_chroma_strip(
        self,
        video_frames: torch.Tensor,
        overlay_image: torch.Tensor,
        side: str,
        mode: str,
        strip_width_pct: float,
        image_padding_pct: float,
        vertical_position: float,
    ) -> Tuple[torch.Tensor]:
        batch, frame_h, frame_w, channels = video_frames.shape
        device = video_frames.device
        dtype = video_frames.dtype

        strip_w = max(1, round(frame_w * strip_width_pct / 100.0))

        green = torch.tensor(
            CHROMA_GREEN[:channels],
            dtype=dtype,
            device=device,
        )

        if mode == "expand":
            total_w = frame_w + strip_w
            output = torch.zeros(
                (batch, frame_h, total_w, channels),
                dtype=dtype,
                device=device,
            )
            if side == "left":
                strip_x = 0
                output[:, :, strip_w:, :] = video_frames
            else:
                strip_x = frame_w
                output[:, :, :frame_w, :] = video_frames
            output[:, :, strip_x:strip_x + strip_w, :] = green
        else:
            # overlay mode - strip covers the video edge
            output = video_frames.clone()
            if side == "left":
                strip_x = 0
            else:
                strip_x = frame_w - strip_w
            output[:, :, strip_x:strip_x + strip_w, :] = green

        # Resize and place overlay image within the strip
        pad_px = max(0, round(strip_w * image_padding_pct / 100.0))
        avail_w = max(1, strip_w - 2 * pad_px)
        avail_h = max(1, frame_h - 2 * pad_px)

        resized = self._resize_overlay(
            overlay_image, avail_w, avail_h
        )
        ov_h, ov_w = resized.shape[1], resized.shape[2]

        # Horizontal: center within strip
        ov_x = strip_x + pad_px + (avail_w - ov_w) // 2

        # Vertical: position based on slider
        max_y_offset = avail_h - ov_h
        ov_y = pad_px + round(max_y_offset * vertical_position)

        # Composite overlay onto every frame
        # Move resized to same device/dtype
        resized = resized.to(device=device, dtype=dtype)
        output[
            :,
            ov_y:ov_y + ov_h,
            ov_x:ov_x + ov_w,
            :
        ] = resized[0]

        return (output,)


NODE_CLASS_MAPPINGS = {
    "ChromaStripOverlay": ChromaStripOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChromaStripOverlay": "Chroma Strip Overlay",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
