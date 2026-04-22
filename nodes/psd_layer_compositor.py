"""
PSD Layer Compositor

Composite PSD layers from a PSD Layer Splitter output
folder back into a single image. Reads layer positions
and opacity from _manifest.json. Optionally replaces one
layer with a provided image (e.g. swap a background).
"""

import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from psd_tools import PSDImage

from .psd_utils import (
    parse_background_color,
    replace_psd_layer_pixels,
    resize_to_bounds,
    tensor_to_pil_rgba,
)


class PSDLayerCompositor:
    """
    Composite PSD layers back into a single image using
    the manifest from PSDLayerSplitter. Optionally replace
    one layer with a new image (e.g. swap a background).
    """

    CATEGORY = "Trent/PSD"
    FUNCTION = "composite"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (
        "Composited image at the original PSD canvas size",
    )
    DESCRIPTION = (
        "Composite layers from a PSD Layer Splitter folder "
        "back into a single image. Uses _manifest.json for "
        "positions, sizes, opacity, and visibility. "
        "Optionally replaces one layer with a provided "
        "image - perfect for swapping a background while "
        "keeping all other layers intact."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/layer/folder",
                    "tooltip": (
                        "Folder from PSD Layer Splitter "
                        "(must contain _manifest.json)"
                    ),
                }),
                "replacement_index": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Layer index to replace (or start "
                        "of range). 0 = bottom (usually "
                        "background). -1 = no replacement. "
                        "Ignored when replacement_mode = "
                        "underlay."
                    ),
                }),
                "replacement_end_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Inclusive end of layer range to "
                        "replace. -1 = single-layer swap "
                        "(use replacement_index alone). "
                        "Only used when replacement_mode = "
                        "replace_range."
                    ),
                }),
                "replacement_mode": (
                    ["single", "replace_range", "underlay"],
                    {
                        "default": "single",
                        "tooltip": (
                            "single = swap one layer at "
                            "replacement_index. "
                            "replace_range = swap layers "
                            "[start..end] with one image. "
                            "underlay = paste replacement "
                            "under all original layers "
                            "(escape hatch for messy PSDs)."
                        ),
                    },
                ),
                "replacement_fit": (
                    ["stretch", "fit", "cover", "center"],
                    {
                        "default": "stretch",
                        "tooltip": (
                            "How to size the replacement "
                            "image. stretch = exact fit. "
                            "fit = preserve AR, letterbox. "
                            "cover = preserve AR, crop. "
                            "center = no resize, center"
                        ),
                    },
                ),
                "range_fit": (
                    ["canvas", "union_bbox"],
                    {
                        "default": "canvas",
                        "tooltip": (
                            "Target area for replace_range "
                            "mode. canvas = full PSD canvas "
                            "(usually what you want). "
                            "union_bbox = combined bbox of "
                            "the replaced layers."
                        ),
                    },
                ),
                "background_color": ("STRING", {
                    "default": "transparent",
                    "tooltip": (
                        "Canvas backdrop: 'transparent', "
                        "'white', 'black', or hex like "
                        "'#FFFFFF'"
                    ),
                }),
                "respect_visibility": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If True, skip layers marked "
                        "hidden in the original PSD"
                    ),
                }),
            },
            "optional": {
                "replacement_image": ("IMAGE", {
                    "tooltip": (
                        "New image to insert at "
                        "replacement_index"
                    ),
                }),
                "output_psd_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/new.psd",
                    "tooltip": (
                        "Optional. If set, also writes a "
                        "new .psd by opening the original "
                        "PSD (from the manifest) and "
                        "swapping the layer at "
                        "replacement_index with "
                        "replacement_image. Original PSD is "
                        "never overwritten - must be a "
                        "different path. Requires "
                        "replacement_image and "
                        "replacement_index >= 0."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Bypass strict validation - inputs are coerced
        # inside composite() to handle ComfyUI's widget-
        # input conversion that can shift widgets_values.
        return True

    def composite(
        self,
        folder_path,
        replacement_index,
        replacement_end_index,
        replacement_mode,
        replacement_fit,
        range_fit,
        background_color,
        respect_visibility,
        replacement_image=None,
        output_psd_path="",
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        # Coerce inputs (defensive against widget shifting)
        folder_path = str(folder_path or "").strip()
        try:
            replacement_index = int(replacement_index)
        except (TypeError, ValueError):
            replacement_index = -1
        try:
            replacement_end_index = int(replacement_end_index)
        except (TypeError, ValueError):
            replacement_end_index = -1

        # Back-compat: older workflows put a fit value
        # (stretch/fit/cover/center) in replacement_mode.
        # Treat those as single-mode + that fit.
        legacy_fits = ("stretch", "fit", "cover", "center")
        if replacement_mode in legacy_fits:
            replacement_fit = replacement_mode
            replacement_mode = "single"
        if replacement_mode not in (
            "single", "replace_range", "underlay"
        ):
            replacement_mode = "single"
        if replacement_fit not in legacy_fits:
            replacement_fit = "stretch"
        if range_fit not in ("canvas", "union_bbox"):
            range_fit = "canvas"

        background_color = str(background_color or "transparent")
        respect_visibility = bool(respect_visibility)
        output_psd_path = str(output_psd_path or "").strip()

        if not folder_path:
            raise ValueError("folder_path is required")
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        manifest_path = os.path.join(
            folder_path, "_manifest.json"
        )
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"_manifest.json not found in "
                f"{folder_path}. PSDLayerCompositor "
                f"requires the manifest saved by "
                f"PSDLayerSplitter."
            )

        with open(
            manifest_path, "r", encoding="utf-8"
        ) as f:
            manifest = json.load(f)

        canvas_w = int(manifest["canvas_width"])
        canvas_h = int(manifest["canvas_height"])
        layers = manifest.get("layers", [])
        if not layers:
            raise RuntimeError(
                "Manifest contains no layers"
            )

        # Detect cropped vs canvas-sized layer extraction
        sizing = (
            manifest.get("extraction_settings", {})
            .get("layer_sizing", "cropped")
        )

        bg_rgba = parse_background_color(background_color)
        canvas = Image.new(
            "RGBA", (canvas_w, canvas_h), bg_rgba
        )

        replacement_pil = None
        if replacement_image is not None:
            replacement_pil = tensor_to_pil_rgba(
                replacement_image
            )

        # Sort by index, bottom to top
        layers_sorted = sorted(
            layers, key=lambda lyr: lyr["index"]
        )

        # Resolve range bounds for replace_range mode
        range_start, range_end = -1, -1
        if (replacement_mode == "replace_range"
                and replacement_pil is not None
                and replacement_index >= 0):
            range_start = replacement_index
            range_end = (
                replacement_end_index
                if replacement_end_index >= 0
                else replacement_index
            )
            if range_end < range_start:
                range_start, range_end = range_end, range_start

        # Underlay: paste replacement on canvas first, then
        # all original layers composite on top.
        if (replacement_mode == "underlay"
                and replacement_pil is not None):
            underlay_img = resize_to_bounds(
                replacement_pil,
                canvas_w, canvas_h,
                replacement_fit,
            )
            canvas.paste(
                underlay_img, (0, 0), underlay_img
            )

        # Range: drop in the replacement once, sized to
        # canvas or union bbox of the replaced layers.
        if range_start >= 0:
            if range_fit == "union_bbox":
                bx0, by0, bx1, by1 = self._union_bbox(
                    layers_sorted, range_start, range_end,
                    canvas_w, canvas_h,
                )
            else:
                bx0, by0 = 0, 0
                bx1, by1 = canvas_w, canvas_h
            tgt_w = max(1, bx1 - bx0)
            tgt_h = max(1, by1 - by0)
            range_img = resize_to_bounds(
                replacement_pil,
                tgt_w, tgt_h,
                replacement_fit,
            )
            canvas.paste(range_img, (bx0, by0), range_img)

        composited = 0
        for lyr in layers_sorted:
            idx = int(lyr["index"])
            filename = lyr["filename"]
            left = int(lyr["position"]["left"])
            top = int(lyr["position"]["top"])
            width = int(lyr["size"]["width"])
            height = int(lyr["size"]["height"])
            opacity = int(lyr.get("opacity", 255))
            visible = bool(lyr.get("visible", True))

            if respect_visibility and not visible:
                continue

            # Skip layers being replaced by range mode
            if (range_start >= 0
                    and range_start <= idx <= range_end):
                continue

            # Single-layer swap
            if (replacement_mode == "single"
                    and replacement_pil is not None
                    and idx == replacement_index):
                if width <= 0 or height <= 0:
                    width, height = canvas_w, canvas_h
                    left, top = 0, 0
                layer_img = resize_to_bounds(
                    replacement_pil,
                    width, height,
                    replacement_fit,
                )
            else:
                layer_path = os.path.join(
                    folder_path, filename
                )
                if not os.path.isfile(layer_path):
                    print(
                        f"[PSDLayerCompositor] Missing "
                        f"file, skipping: {filename}"
                    )
                    continue
                layer_img = Image.open(layer_path)
                layer_img = ImageOps.exif_transpose(
                    layer_img
                )
                layer_img = layer_img.convert("RGBA")

            # If layers were saved canvas-sized, paste at
            # (0,0). If cropped, paste at the layer offset.
            if sizing == "canvas":
                paste_x, paste_y = 0, 0
            else:
                paste_x, paste_y = left, top

            # Apply layer opacity to alpha channel
            if opacity < 255:
                alpha = layer_img.getchannel("A")
                scale = opacity / 255.0
                alpha = alpha.point(
                    lambda v, s=scale: int(v * s)
                )
                layer_img.putalpha(alpha)

            # Alpha-composite onto canvas
            canvas.paste(
                layer_img, (paste_x, paste_y), layer_img
            )
            composited += 1

        if composited == 0 and replacement_mode == "single":
            raise RuntimeError(
                "No layers were composited (all skipped "
                "or missing)"
            )

        print(
            f"[PSDLayerCompositor] Composited "
            f"{composited} layer(s) into "
            f"{canvas_w}x{canvas_h} canvas"
        )

        # Optional: also write a new .psd by swapping the
        # target layer in the original PSD and saving to a
        # different path. The original is never modified.
        # v1: only single-layer mode supports PSD re-export.
        if output_psd_path:
            if replacement_mode != "single":
                print(
                    f"[PSDLayerCompositor] output_psd_path "
                    f"is set but replacement_mode is "
                    f"'{replacement_mode}'. PSD re-export "
                    f"only supports 'single' mode in v1. "
                    f"Skipping PSD write."
                )
            else:
                self._save_modified_psd(
                    manifest=manifest,
                    layers=layers,
                    replacement_index=replacement_index,
                    replacement_pil=replacement_pil,
                    replacement_mode=replacement_fit,
                    output_psd_path=output_psd_path,
                )

        # Convert to ComfyUI image tensor
        rgb = canvas.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]
        return (tensor,)

    @staticmethod
    def _union_bbox(
        layers_sorted,
        start_idx,
        end_idx,
        canvas_w,
        canvas_h,
    ):
        """Union bbox of layers in [start_idx..end_idx],
        clamped to canvas. Falls back to full canvas if no
        layers in range have positive size."""
        x0, y0 = canvas_w, canvas_h
        x1, y1 = 0, 0
        found = False
        for lyr in layers_sorted:
            idx = int(lyr["index"])
            if not (start_idx <= idx <= end_idx):
                continue
            w = int(lyr["size"]["width"])
            h = int(lyr["size"]["height"])
            if w <= 0 or h <= 0:
                continue
            l = int(lyr["position"]["left"])
            t = int(lyr["position"]["top"])
            x0 = min(x0, l)
            y0 = min(y0, t)
            x1 = max(x1, l + w)
            y1 = max(y1, t + h)
            found = True
        if not found:
            return (0, 0, canvas_w, canvas_h)
        # Clamp to canvas
        x0 = max(0, min(x0, canvas_w))
        y0 = max(0, min(y0, canvas_h))
        x1 = max(0, min(x1, canvas_w))
        y1 = max(0, min(y1, canvas_h))
        return (x0, y0, x1, y1)

    def _save_modified_psd(
        self,
        manifest,
        layers,
        replacement_index,
        replacement_pil,
        replacement_mode,
        output_psd_path,
    ):
        """Open the source PSD, swap one layer, save anew.

        Source PSD path comes from the manifest. The target
        layer name is resolved from the manifest using
        replacement_index. The original PSD is never
        overwritten - output_psd_path must differ.
        """
        if replacement_pil is None:
            raise ValueError(
                "output_psd_path is set but no "
                "replacement_image was provided - "
                "nothing to swap"
            )
        if replacement_index < 0:
            raise ValueError(
                "output_psd_path is set but "
                "replacement_index is -1 (no replacement)"
            )

        source_psd_path = manifest.get("source_psd", "")
        if not source_psd_path:
            raise RuntimeError(
                "Manifest has no source_psd - cannot save "
                "modified PSD"
            )
        if not os.path.isfile(source_psd_path):
            raise FileNotFoundError(
                f"Original PSD not found at "
                f"{source_psd_path}"
            )

        if (
            os.path.abspath(output_psd_path)
            == os.path.abspath(source_psd_path)
        ):
            raise ValueError(
                "output_psd_path must differ from the "
                "original PSD path - refusing to overwrite "
                f"the source: {source_psd_path}"
            )

        target = next(
            (
                lyr for lyr in layers
                if int(lyr["index"]) == replacement_index
            ),
            None,
        )
        if target is None:
            raise ValueError(
                f"replacement_index {replacement_index} "
                f"not found in manifest"
            )
        target_name = target.get("original_name", "")
        if not target_name:
            raise ValueError(
                f"Manifest layer at index "
                f"{replacement_index} has no original_name"
            )

        out_dir = os.path.dirname(
            os.path.abspath(output_psd_path)
        )
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        psd = PSDImage.open(source_psd_path)
        old_name = replace_psd_layer_pixels(
            psd,
            target_name,
            replacement_pil,
            replacement_mode,
        )

        with open(output_psd_path, "wb") as fp:
            psd.save(fp)

        print(
            f"[PSDLayerCompositor] Wrote modified PSD to "
            f"{output_psd_path} (replaced layer "
            f"'{old_name}' at index {replacement_index})"
        )


NODE_CLASS_MAPPINGS = {
    "PSDLayerCompositor": PSDLayerCompositor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDLayerCompositor": "PSD Layer Compositor",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
