"""
Save Transparent Video node.

Exports image batches as video files with alpha channel transparency.
Supports three output formats:
- Animated WebP with alpha (web-ready, good compression)
- ProRes 4444 in MOV (professional compositing)
- PNG image sequence (universal lossless fallback)

Alpha is sourced from an optional MASK input, the 4th channel of
RGBA images, or defaults to fully opaque.
"""

import logging
import os
from fractions import Fraction
from typing import Any, Dict, Tuple

import av
import torch
import torch.nn.functional as F

import folder_paths
from comfy.utils import ProgressBar

log = logging.getLogger(__name__)

# Format metadata
FORMAT_INFO = {
    "webp": {
        "ext": "webp",
        "codec": "libwebp_anim",
        "label": "Animated WebP (web-ready alpha)",
    },
    "mov_prores4444": {
        "ext": "mov",
        "codec": "prores_ks",
        "label": "ProRes 4444 (compositing)",
    },
    "png_sequence": {
        "ext": "png",
        "codec": None,
        "label": "PNG sequence (lossless)",
    },
}


class SaveTransparentVideo:
    """
    Save video with transparency (alpha channel).

    Encodes an image batch as a transparent video file.
    Alpha is taken from the optional MASK input, the 4th
    channel of RGBA images, or defaults to fully opaque.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Image batch [N, H, W, C]. If C=4 the"
                        " 4th channel is used as alpha unless a"
                        " MASK is connected."
                    ),
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_alpha",
                    "tooltip": (
                        "Output filename prefix. Supports"
                        " ComfyUI variables like"
                        " %date:yyyy-MM-dd%."
                    ),
                }),
                "format": (
                    list(FORMAT_INFO.keys()), {
                        "default": "webp",
                        "tooltip": (
                            "webp: animated WebP, good"
                            " compression, browser-ready."
                            " mov_prores4444: lossless for"
                            " compositing. png_sequence:"
                            " individual RGBA PNGs."
                        ),
                    },
                ),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.01,
                    "tooltip": "Frames per second.",
                }),
                "quality": ("INT", {
                    "default": 80,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": (
                        "Quality 1-100. For WebP controls"
                        " lossy quality. Ignored for PNG"
                        " and ProRes."
                    ),
                }),
            },
            "optional": {
                "masks": ("MASK", {
                    "tooltip": (
                        "Optional alpha mask [N, H, W]."
                        " Overrides any alpha already in"
                        " the images. White = opaque,"
                        " black = transparent."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Trent/Video"

    DESCRIPTION = (
        "Save Transparent Video\n\n"
        "Export image batches as video with alpha"
        " transparency.\n\n"
        "Formats:\n"
        "- Animated WebP: good compression, plays in"
        " browsers\n"
        "- ProRes 4444 MOV: lossless, for DaVinci /"
        " After Effects\n"
        "- PNG sequence: individual RGBA frames\n\n"
        "Alpha source priority:\n"
        "1. Connected MASK input\n"
        "2. 4th channel of RGBA images\n"
        "3. Fully opaque (white) if neither"
    )

    # ----------------------------------------------------------
    # Alpha extraction
    # ----------------------------------------------------------

    @staticmethod
    def _build_rgba(
        images: torch.Tensor,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine images + alpha into [N, H, W, 4] float32.

        All resizing stays on-device via F.interpolate.
        """
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        rgb = images[..., :3]

        if masks is not None:
            alpha = masks
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            # Resize mask to match image dims if needed
            if alpha.shape[-2:] != (h, w):
                alpha = F.interpolate(
                    alpha.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            # Match frame count
            if alpha.shape[0] == 1 and n > 1:
                alpha = alpha.expand(n, -1, -1)
            elif alpha.shape[0] != n:
                raise ValueError(
                    f"Mask batch size ({alpha.shape[0]})"
                    f" != image batch size ({n})."
                    " Provide one mask or one per frame."
                )
            alpha = alpha.unsqueeze(-1)  # [N, H, W, 1]
        elif images.shape[-1] >= 4:
            alpha = images[..., 3:4]
        else:
            log.warning(
                "SaveTransparentVideo: no alpha source;"
                " saving fully opaque."
            )
            alpha = torch.ones(
                n, h, w, 1,
                dtype=images.dtype,
                device=images.device,
            )

        rgba = torch.cat([rgb, alpha], dim=-1)
        return rgba.clamp(0.0, 1.0)

    # ----------------------------------------------------------
    # Format encoders
    # ----------------------------------------------------------

    def _save_webp(
        self,
        rgba: torch.Tensor,
        path: str,
        fps: float,
        quality: int,
    ) -> None:
        """Encode RGBA frames to animated WebP."""
        n = rgba.shape[0]
        pbar = ProgressBar(n)

        container = av.open(path, mode="w")
        stream = container.add_stream(
            "libwebp_anim",
            rate=Fraction(round(fps * 1000), 1000),
        )
        stream.width = rgba.shape[2]
        stream.height = rgba.shape[1]
        stream.pix_fmt = "bgra"
        stream.options = {
            "lossless": "0",
            "quality": str(quality),
            "loop": "0",
        }

        # Batch convert to uint8 on GPU, then iterate
        frames_u8 = (rgba * 255).to(torch.uint8).cpu().numpy()

        for i in range(n):
            frame = av.VideoFrame.from_ndarray(
                frames_u8[i], format="rgba",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
            pbar.update(1)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def _save_prores4444(
        self,
        rgba: torch.Tensor,
        path: str,
        fps: float,
        quality: int,
    ) -> None:
        """Encode RGBA frames to ProRes 4444 MOV."""
        n, h, w = rgba.shape[0], rgba.shape[1], rgba.shape[2]
        pbar = ProgressBar(n)

        container = av.open(path, mode="w")
        stream = container.add_stream(
            "prores_ks",
            rate=Fraction(round(fps * 1000), 1000),
        )
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuva444p10le"
        # Profile 4 = ProRes 4444
        stream.options = {"profile": "4"}

        # Batch convert to uint8 on GPU, then iterate
        frames_u8 = (rgba * 255).to(torch.uint8).cpu().numpy()

        for i in range(n):
            # Create RGBA frame then reformat to
            # yuva444p10le for ProRes 4444 encoding.
            rgba_frame = av.VideoFrame.from_ndarray(
                frames_u8[i], format="rgba",
            )
            frame = rgba_frame.reformat(
                format="yuva444p10le",
            )

            for packet in stream.encode(frame):
                container.mux(packet)
            pbar.update(1)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

    def _save_png_sequence(
        self,
        rgba: torch.Tensor,
        folder: str,
    ) -> None:
        """Save each frame as an individual RGBA PNG."""
        os.makedirs(folder, exist_ok=True)
        n = rgba.shape[0]
        pbar = ProgressBar(n)

        frames_u8 = (rgba * 255).to(torch.uint8).cpu().numpy()

        for i in range(n):
            out_path = os.path.join(
                folder, f"frame_{i:05d}.png",
            )
            container = av.open(out_path, mode="w")
            stream = container.add_stream("png")
            stream.width = frames_u8.shape[2]
            stream.height = frames_u8.shape[1]
            stream.pix_fmt = "rgba"

            frame = av.VideoFrame.from_ndarray(
                frames_u8[i], format="rgba",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            pbar.update(1)

    # ----------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------

    def save(
        self,
        images: torch.Tensor,
        filename_prefix: str,
        format: str,
        fps: float,
        quality: int,
        masks: torch.Tensor | None = None,
    ) -> Tuple[str]:
        rgba = self._build_rgba(images, masks)

        h, w = rgba.shape[1], rgba.shape[2]
        out_dir = folder_paths.get_output_directory()
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(
            filename_prefix, out_dir, w, h,
        )

        if format == "png_sequence":
            seq_folder = os.path.join(
                full_output_folder,
                f"{filename}_{counter:05d}_frames",
            )
            self._save_png_sequence(rgba, seq_folder)
            result_path = seq_folder
            log.info(
                "Saved %d RGBA PNGs to %s",
                rgba.shape[0], seq_folder,
            )
        else:
            ext = FORMAT_INFO[format]["ext"]
            file_name = f"{filename}_{counter:05d}_.{ext}"
            file_path = os.path.join(
                full_output_folder, file_name,
            )

            if format == "webp":
                self._save_webp(
                    rgba, file_path, fps, quality,
                )
            elif format == "mov_prores4444":
                self._save_prores4444(
                    rgba, file_path, fps, quality,
                )

            result_path = file_path
            log.info(
                "Saved %d frames to %s", rgba.shape[0],
                file_path,
            )

        return (result_path,)


NODE_CLASS_MAPPINGS = {
    "SaveTransparentVideo": SaveTransparentVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTransparentVideo": "Save Transparent Video",
}
