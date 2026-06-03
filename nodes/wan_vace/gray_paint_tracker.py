"""
Gray Paint Tracker node.

Paint a gray region on the first frame of a video, then have that
region follow a tracked point across the whole clip. Outputs the
gray-composited video and a matching white binary mask for use with
Wan VACE inpaint workflows.

Combines three existing TrentNodes building blocks:
  - PointTracker (lipsync): pyramidal Lucas-Kanade point tracking
  - VaceMaskAutoComping: gray-over-region compositing convention
  - mask_ops: centroid + dilation utilities

The interactive painting happens in the companion JS widget
(js/gray_paint.js), which sends the painted mask back as a base64
PNG in the hidden ``mask_data`` widget. The node runs in two passes:
pass 1 (empty mask) just sends the first frame to the canvas to paint
on; pass 2 (mask painted) tracks + composites.
"""

import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.mask_ops import dilate_mask, get_mask_centroid
from ..lipsync.point_tracker import PointTracker


class GrayPaintTracker:
    """
    Interactive paint-a-mask node with single-point motion tracking.

    Paint a region on the first frame; the painted mask rigidly
    follows a tracked point (centroid of the paint by default, or a
    manually clicked anchor) across all frames. Solid gray is
    composited over the moving region and a matching binary mask is
    emitted for Wan VACE inpaint.
    """

    GRAY_VALUE = 0.5
    CATEGORY = "Trent/Keyframes"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Paint a gray region on the first frame of a video and have"
        " it follow a tracked point across the whole clip. Outputs"
        " the gray-composited video plus a matching white binary"
        " mask for Wan VACE inpaint. Run once to load the first"
        " frame onto the node, paint, then run again."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Video frames batch (B, H, W, C)"
                }),
                "mask_data": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Painted mask (base64 PNG). Set automatically"
                        " by the paint canvas - hidden in the UI."
                    ),
                }),
            },
            "optional": {
                "track_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Track a point so the mask follows motion."
                        " Off = static mask held across all frames."
                    ),
                }),
                "anchor_mode": (["centroid", "manual"], {
                    "default": "centroid",
                    "tooltip": (
                        "centroid = track the center of the painted"
                        " region. manual = track the clicked anchor"
                        " (use 'Set Anchor' on the canvas)."
                    ),
                }),
                "anchor_x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Manual anchor X (click image to set)",
                }),
                "anchor_y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Manual anchor Y (click image to set)",
                }),
                "gray_value": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": (
                        "Gray level for masked areas (0.5 = mid gray,"
                        " matches VACE default filler)"
                    ),
                }),
                "expand_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": (
                        "Expand mask outward by this many pixels"
                        " before compositing (hard edge, no feather)"
                    ),
                }),
                "draw_preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Output a tracking_preview with crosshairs at"
                        " the tracked point (slower; debug aid)."
                    ),
                }),
                "window_size": ("INT", {
                    "default": 31,
                    "min": 11,
                    "max": 1025,
                    "step": 2,
                    "tooltip": (
                        "Tracking search window in pixels. Use large"
                        " values (201+) for fast-moving objects."
                    ),
                }),
                "pyramid_levels": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": (
                        "Pyramid levels (more = handles larger"
                        " motion). 6-8 for very large motion."
                    ),
                }),
                "iterations": ("INT", {
                    "default": 10,
                    "min": 3,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Iterations per level (more = accurate)",
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Temporal smoothing (0=none, higher=smoother)",
                }),
                "search_radius_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 5.0,
                    "tooltip": (
                        "Template-match search as % of frame size."
                        " 0=use window_size, 100=search whole frame."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("images", "masks", "tracking_preview")
    OUTPUT_TOOLTIPS = (
        "Original video with solid gray composited over the"
        " (moving) masked region",
        "White (1) where gray was applied per frame, black (0)"
        " elsewhere - binary, VACE-ready",
        "Optional debug preview with tracked-point crosshairs"
        " (1x1 placeholder unless draw_preview is on)",
    )
    FUNCTION = "run"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_preview(self, frame: torch.Tensor) -> str:
        """Encode a single (H, W, C) frame to base64 JPEG for the canvas."""
        img_np = (frame.detach().cpu().numpy() * 255.0)
        img_np = img_np.clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
        pil_img = Image.fromarray(img_np)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=75)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_mask(
        self, mask_data: str, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        """Decode the painted base64 PNG into a binary (H, W) mask."""
        raw = mask_data
        if "," in raw:  # strip any data-URL prefix defensively
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        pil = Image.open(BytesIO(img_bytes))

        # Painted strokes carry alpha (erase clears alpha), so the
        # alpha channel is the cleanest mask signal. Fall back to
        # luminance for non-alpha PNGs.
        if pil.mode in ("RGBA", "LA"):
            band = np.array(pil.split()[-1], dtype=np.float32) / 255.0
        else:
            band = np.array(pil.convert("L"), dtype=np.float32) / 255.0

        mask = torch.from_numpy(band).to(device=device, dtype=torch.float32)

        # Guard against any resolution mismatch (paint canvas authors
        # at native frame res, so this is normally a no-op).
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = F.interpolate(
                mask.view(1, 1, *mask.shape),
                size=(H, W),
                mode="nearest",
            ).view(H, W)

        return (mask > 0.5).float()

    def _translate_masks(
        self,
        mask0: torch.Tensor,
        points: List[Tuple[int, int]],
        anchor: Tuple[int, int],
        B: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Translate the frame-0 mask to follow the tracked point.

        Subpixel translation via grid_sample with zero padding so the
        mask correctly vanishes off-frame (torch.roll would wrap).
        """
        ax, ay = anchor

        # Per-frame displacement (dx, dy); frame 0 is always (0, 0)
        # because points[0] == anchor by construction.
        disp = torch.tensor(
            [[px - ax, py - ay] for (px, py) in points[:B]],
            device=device,
            dtype=torch.float32,
        )  # (B, 2) as (dx, dy)
        if disp.shape[0] < B:  # pad with last displacement if short
            pad = disp[-1:].expand(B - disp.shape[0], 2)
            disp = torch.cat([disp, pad], dim=0)

        # Identity sampling grid in (x, y) pixel coordinates.
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
        base = torch.stack([gx, gy], dim=-1)  # (H, W, 2) -> (x, y)

        # To move content by +disp, sample from coord - disp.
        src = base.view(1, H, W, 2) - disp.view(B, 1, 1, 2)  # (B, H, W, 2)

        # Normalize to [-1, 1] (align_corners=False convention).
        norm = torch.empty_like(src)
        norm[..., 0] = (src[..., 0] + 0.5) * 2.0 / W - 1.0
        norm[..., 1] = (src[..., 1] + 0.5) * 2.0 / H - 1.0

        m0 = mask0.view(1, 1, H, W).expand(B, 1, H, W).contiguous()
        warped = F.grid_sample(
            m0, norm, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )  # (B, 1, H, W)

        # Re-binarize so translated edges stay crisp (no gray fringe).
        return (warped.squeeze(1) > 0.5).float()  # (B, H, W)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(
        self,
        video: torch.Tensor,
        mask_data: str,
        track_enabled: bool = True,
        anchor_mode: str = "centroid",
        anchor_x: int = 0,
        anchor_y: int = 0,
        gray_value: float = 0.5,
        expand_pixels: int = 0,
        draw_preview: bool = False,
        window_size: int = 31,
        pyramid_levels: int = 4,
        iterations: int = 10,
        smoothing: float = 0.3,
        search_radius_percent: float = 0.0,
    ) -> Dict[str, Any]:
        if video.dim() == 3:
            video = video.unsqueeze(0)

        B, H, W, C = video.shape
        device = video.device
        dtype = video.dtype

        frames_rgb = video[..., :3].contiguous()
        preview_b64 = self._encode_preview(video[0])
        placeholder = torch.zeros((1, 1, 1, 3), device=device, dtype=dtype)

        # --- Pass 1 / nothing painted: pass-through ---------------------
        mask_str = (mask_data or "").strip()
        if not mask_str:
            print(
                "[GrayPaintTracker] No mask painted yet - pass-through."
                " Paint on the node, then run again."
            )
            zeros_mask = torch.zeros((B, H, W), device=device,
                                     dtype=torch.float32)
            return {
                "ui": {"preview_image": [preview_b64]},
                "result": (video, zeros_mask, placeholder),
            }

        mask0 = self._decode_mask(mask_str, H, W, device)  # (H, W) binary

        if mask0.sum() < 1:
            print("[GrayPaintTracker] Painted mask is empty - pass-through.")
            zeros_mask = torch.zeros((B, H, W), device=device,
                                     dtype=torch.float32)
            return {
                "ui": {"preview_image": [preview_b64]},
                "result": (video, zeros_mask, placeholder),
            }

        # --- Anchor selection ------------------------------------------
        if anchor_mode == "manual":
            ax, ay = int(anchor_x), int(anchor_y)
        else:
            cy, cx = get_mask_centroid(mask0)  # returns (cy, cx)
            ax, ay = int(round(cx)), int(round(cy))
        ax = max(0, min(W - 1, ax))
        ay = max(0, min(H - 1, ay))

        # --- Track the anchor ------------------------------------------
        tracker_preview = None
        if track_enabled and B > 1:
            points, _tmask, tracker_preview = PointTracker().track_point(
                frames_rgb, ax, ay,
                window_size, pyramid_levels, iterations,
                smoothing, search_radius_percent,
            )
        else:
            points = [(ax, ay)] * B

        # --- Translate mask to follow the point ------------------------
        masks = self._translate_masks(mask0, points, (ax, ay), B, H, W, device)

        if expand_pixels > 0:
            masks = dilate_mask(masks, expand_pixels)
            masks = (masks > 0.5).float()

        # --- Composite gray over RGB -----------------------------------
        video_rgb = video[..., :3].float()
        mask_4d = masks.unsqueeze(-1)  # (B, H, W, 1)
        composited = video_rgb * (1.0 - mask_4d) + gray_value * mask_4d
        composited = composited.to(dtype=dtype)

        # --- Optional tracking preview ---------------------------------
        if draw_preview:
            if tracker_preview is not None:
                preview_out = tracker_preview.to(dtype=dtype)
            else:
                preview_out = PointTracker()._draw_tracking_preview(
                    frames_rgb, points
                ).to(dtype=dtype)
        else:
            preview_out = placeholder

        cov = masks.mean().item() * 100.0
        print(
            f"[GrayPaintTracker] {B} frames, "
            f"anchor=({ax},{ay}) [{anchor_mode}], "
            f"track={'on' if (track_enabled and B > 1) else 'off'}, "
            f"gray={gray_value:.2f}, expand={expand_pixels}px, "
            f"avg mask coverage={cov:.1f}%"
        )

        return {
            "ui": {"preview_image": [preview_b64]},
            "result": (composited, masks, preview_out),
        }


NODE_CLASS_MAPPINGS = {
    "GrayPaintTracker": GrayPaintTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrayPaintTracker": "Gray Paint Tracker",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
