"""
PSD Layer Compositor

Composite PSD layers from a PSD Layer Splitter output
folder back into a single image. Reads layer positions
and opacity from _manifest.json. Optionally replaces one
layer with a provided image (e.g. swap a background).
"""

import colorsys
import json
import os
import re
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer, TypeLayer
from psd_tools.constants import BlendMode, Compression

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
                "text_recolor_mode": (
                    ["off", "manual", "auto_contrast", "complement"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Recolor PSD text layers "
                            "(kind=='type') in the output. "
                            "off = leave colors alone. "
                            "manual = use text_color for "
                            "every text layer. "
                            "auto_contrast = pick black or "
                            "white per layer based on the "
                            "luminance of pixels behind it. "
                            "complement = sample bg under "
                            "each text layer and pick a "
                            "complementary color (hue +180) "
                            "with luminance pushed for "
                            "legibility. When output_psd_path "
                            "is set, recolor is also applied "
                            "to the saved PSD as clipping "
                            "PixelLayers above each text "
                            "layer (text stays editable)."
                        ),
                    },
                ),
                "text_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": (
                        "Color for text layers when "
                        "text_recolor_mode=manual. Hex "
                        "(#RRGGBB), CSS name, or rgb()."
                    ),
                }),
                "auto_contrast_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Used when "
                        "text_recolor_mode=auto_contrast. "
                        "If avg luminance under the text "
                        "(0..1) is below this, recolor to "
                        "white; otherwise to black."
                    ),
                }),
                "text_layer_pattern": ("STRING", {
                    "default": "(?i)text",
                    "placeholder": "(?i)text",
                    "tooltip": (
                        "Regex matched against layer names. "
                        "Any layer whose name matches is "
                        "treated as text and recolored, in "
                        "addition to real TypeLayers. The "
                        "default '(?i)text' matches any "
                        "layer whose name contains 'text' "
                        "(case-insensitive) anywhere - "
                        "'Text 1', 'Layer 4 - text', "
                        "'header_text' all match. Clear "
                        "the field to recolor TypeLayers "
                        "only."
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
        text_recolor_mode="off",
        text_color="#FFFFFF",
        auto_contrast_threshold=0.5,
        text_layer_pattern="",
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

        text_recolor_mode = str(text_recolor_mode or "off")
        if text_recolor_mode not in (
            "off", "manual", "auto_contrast", "complement"
        ):
            text_recolor_mode = "off"
        text_color = str(text_color or "#FFFFFF").strip()
        try:
            auto_contrast_threshold = float(
                auto_contrast_threshold
            )
        except (TypeError, ValueError):
            auto_contrast_threshold = 0.5
        auto_contrast_threshold = max(
            0.0, min(1.0, auto_contrast_threshold)
        )

        # Pre-parse the manual color so a typo errors fast
        # rather than silently per layer.
        manual_text_rgb = None
        if text_recolor_mode == "manual":
            rgba = parse_background_color(text_color)
            manual_text_rgb = (rgba[0], rgba[1], rgba[2])

        # Compile the optional name pattern that promotes
        # non-TypeLayer layers (e.g. rasterized text named
        # consistently) into the text-recolor pass.
        text_layer_pattern = str(text_layer_pattern or "").strip()
        text_pattern_re = None
        if text_layer_pattern:
            try:
                text_pattern_re = re.compile(text_layer_pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid text_layer_pattern regex: {e}"
                )

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

        if text_recolor_mode != "off":
            kinds_seen = sorted(
                {str(l.get("kind", "")).lower()
                 for l in layers_sorted}
            )
            type_count = sum(
                1 for l in layers_sorted
                if str(l.get("kind", "")).lower() == "type"
            )
            pattern_matches = 0
            if text_pattern_re is not None:
                pattern_matches = sum(
                    1 for l in layers_sorted
                    if text_pattern_re.search(
                        l.get("original_name", "") or ""
                    )
                )
            print(
                f"[PSDLayerCompositor] text_recolor_mode="
                f"{text_recolor_mode} | text_color="
                f"{text_color} | threshold="
                f"{auto_contrast_threshold} | "
                f"text_layer_pattern="
                f"{text_layer_pattern!r} | "
                f"layer kinds in manifest: {kinds_seen} | "
                f"text layers (kind=='type'): {type_count}"
                f" | name-pattern matches: "
                f"{pattern_matches}"
            )
            if type_count == 0 and pattern_matches == 0:
                print(
                    "[PSDLayerCompositor] WARNING: no "
                    "kind=='type' layers and no name "
                    "pattern matches. Either re-split the "
                    "PSD so TypeLayers are tagged kind="
                    "'type', or set text_layer_pattern to "
                    "a regex that matches your text layer "
                    "names (e.g. '(?i)\\btext\\b')."
                )

        composited = 0
        recolored_count = 0
        # Maps text-layer original_name -> (r, g, b) chosen
        # during the flat composite. Reused by the PSD save
        # path to apply matching clipping overlays.
        text_color_cache: Dict[str, Tuple[int, int, int]] = {}
        for lyr in layers_sorted:
            idx = int(lyr["index"])
            filename = lyr["filename"]
            left = int(lyr["position"]["left"])
            top = int(lyr["position"]["top"])
            width = int(lyr["size"]["width"])
            height = int(lyr["size"]["height"])
            opacity = int(lyr.get("opacity", 255))
            visible = bool(lyr.get("visible", True))
            kind = str(lyr.get("kind", "")).lower()

            if respect_visibility and not visible:
                continue

            # Skip layers being replaced by range mode
            if (range_start >= 0
                    and range_start <= idx <= range_end):
                continue

            # Single-layer swap
            is_replacement = False
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
                is_replacement = True
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

            # Recolor PSD text layers (kind=='type'). Done
            # before the composite so auto_contrast can
            # sample the canvas as it currently stands -
            # which is exactly what will sit behind the
            # text after this paste. Skip when this slot
            # is being replaced by an external image -
            # nobody wants their replacement turned into
            # a flat color.
            name_for_match = lyr.get("original_name", "") or ""
            is_text_layer = (
                kind == "type"
                or (
                    text_pattern_re is not None
                    and bool(text_pattern_re.search(name_for_match))
                )
            )
            if (text_recolor_mode != "off"
                    and is_text_layer
                    and not is_replacement):
                layer_img, chosen_rgb = self._recolor_text_layer(
                    layer_img=layer_img,
                    canvas=canvas,
                    paste_x=paste_x,
                    paste_y=paste_y,
                    mode=text_recolor_mode,
                    manual_rgb=manual_text_rgb,
                    threshold=auto_contrast_threshold,
                )
                cache_key = lyr.get("original_name") or ""
                if cache_key:
                    text_color_cache[cache_key] = chosen_rgb
                recolored_count += 1

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

        recolor_note = ""
        if text_recolor_mode != "off":
            recolor_note = (
                f" (recolored {recolored_count} text "
                f"layer(s) via {text_recolor_mode})"
            )
        print(
            f"[PSDLayerCompositor] Composited "
            f"{composited} layer(s) into "
            f"{canvas_w}x{canvas_h} canvas"
            f"{recolor_note}"
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
                    text_recolor_mode=text_recolor_mode,
                    text_color_cache=text_color_cache,
                    text_pattern_re=text_pattern_re,
                )

        # Convert to ComfyUI image tensor
        rgb = canvas.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]
        return (tensor,)

    @staticmethod
    def _recolor_text_layer(
        layer_img,
        canvas,
        paste_x,
        paste_y,
        mode,
        manual_rgb,
        threshold,
    ):
        """Return (recolored_rgba_image, chosen_rgb).

        Alpha is preserved so the text's anti-aliased edges
        stay intact. The chosen RGB is also returned so the
        caller can cache it for the PSD-side recolor pass.
        """
        alpha = layer_img.getchannel("A")

        if mode == "manual" and manual_rgb is not None:
            new_rgb = manual_rgb
        elif mode == "complement":
            new_rgb = PSDLayerCompositor._complement_rgb(
                alpha=alpha,
                canvas=canvas,
                paste_x=paste_x,
                paste_y=paste_y,
            )
        else:  # auto_contrast
            new_rgb = PSDLayerCompositor._auto_contrast_rgb(
                alpha=alpha,
                canvas=canvas,
                paste_x=paste_x,
                paste_y=paste_y,
                threshold=threshold,
            )

        recolored = Image.new(
            "RGB", layer_img.size, new_rgb
        )
        recolored.putalpha(alpha)
        return recolored, new_rgb

    @staticmethod
    def _sample_bg_rgb(alpha, canvas, paste_x, paste_y):
        """Alpha-weighted mean RGB of the canvas region
        beneath the layer. Returns (mean_rgb, weight_sum).
        weight_sum == 0 when the layer is entirely off-
        canvas or has no opaque pixels in the overlap."""
        layer_w, layer_h = alpha.size
        canvas_w, canvas_h = canvas.size

        x0 = max(0, paste_x)
        y0 = max(0, paste_y)
        x1 = min(canvas_w, paste_x + layer_w)
        y1 = min(canvas_h, paste_y + layer_h)
        if x1 <= x0 or y1 <= y0:
            return (0.0, 0.0, 0.0), 0.0

        canvas_crop = canvas.crop((x0, y0, x1, y1))
        layer_x0 = x0 - paste_x
        layer_y0 = y0 - paste_y
        layer_x1 = layer_x0 + (x1 - x0)
        layer_y1 = layer_y0 + (y1 - y0)
        alpha_crop = alpha.crop(
            (layer_x0, layer_y0, layer_x1, layer_y1)
        )

        canvas_arr = np.asarray(
            canvas_crop.convert("RGB"),
            dtype=np.float32,
        )
        alpha_arr = np.asarray(
            alpha_crop, dtype=np.float32
        )
        weight_sum = float(alpha_arr.sum())
        if weight_sum <= 0.0:
            return (0.0, 0.0, 0.0), 0.0

        weighted = canvas_arr * alpha_arr[..., None]
        mean_rgb = (
            weighted.reshape(-1, 3).sum(axis=0) / weight_sum
        )
        return (
            float(mean_rgb[0]),
            float(mean_rgb[1]),
            float(mean_rgb[2]),
        ), weight_sum

    @staticmethod
    def _auto_contrast_rgb(
        alpha,
        canvas,
        paste_x,
        paste_y,
        threshold,
    ):
        """Pick (255,255,255) or (0,0,0) based on the
        alpha-weighted luminance of the canvas region
        beneath the layer."""
        mean_rgb, weight_sum = PSDLayerCompositor._sample_bg_rgb(
            alpha, canvas, paste_x, paste_y
        )
        if weight_sum <= 0.0:
            return (255, 255, 255)

        # ITU-R BT.601 luminance, normalized to 0..1
        avg_lum = (
            0.299 * mean_rgb[0]
            + 0.587 * mean_rgb[1]
            + 0.114 * mean_rgb[2]
        ) / 255.0
        if avg_lum < threshold:
            return (255, 255, 255)
        return (0, 0, 0)

    @staticmethod
    def _complement_rgb(
        alpha,
        canvas,
        paste_x,
        paste_y,
    ):
        """Sample bg under the layer and return a pure
        complementary color: hue shifted 180 degrees,
        saturation boosted so the complement reads colorful,
        value inverted (1 - v) for natural luminance
        contrast. No thresholds, no branching - bypasses
        auto_contrast_threshold entirely.
        """
        mean_rgb, weight_sum = PSDLayerCompositor._sample_bg_rgb(
            alpha, canvas, paste_x, paste_y
        )
        if weight_sum <= 0.0:
            return (255, 255, 255)

        r = mean_rgb[0] / 255.0
        g = mean_rgb[1] / 255.0
        b = mean_rgb[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        new_h = (h + 0.5) % 1.0
        new_s = max(0.55, s)
        new_v = 1.0 - v

        nr, ng, nb = colorsys.hsv_to_rgb(new_h, new_s, new_v)
        return (
            int(round(nr * 255)),
            int(round(ng * 255)),
            int(round(nb * 255)),
        )

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
        text_recolor_mode="off",
        text_color_cache=None,
        text_pattern_re=None,
    ):
        """Open the source PSD, swap one layer, save anew.

        Source PSD path comes from the manifest. The target
        layer name is resolved from the manifest using
        replacement_index. The original PSD is never
        overwritten - output_psd_path must differ.

        When text_recolor_mode is set and text_color_cache
        is non-empty, also insert a clipping PixelLayer
        above each TypeLayer whose name appears in the
        cache, filled with that layer's chosen color. The
        TypeLayers themselves are untouched - text stays
        editable in Photoshop.
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

        clip_count = 0
        if text_recolor_mode != "off" and text_color_cache:
            clip_count = self._apply_text_clipping_overlays(
                psd, text_color_cache, text_pattern_re,
            )

        with open(output_psd_path, "wb") as fp:
            psd.save(fp)

        clip_note = ""
        if clip_count:
            clip_note = (
                f", added {clip_count} text recolor "
                f"clipping layer(s)"
            )
        print(
            f"[PSDLayerCompositor] Wrote modified PSD to "
            f"{output_psd_path} (replaced layer "
            f"'{old_name}' at index {replacement_index}"
            f"{clip_note})"
        )

    @staticmethod
    def _apply_text_clipping_overlays(
        psd, color_cache, text_pattern_re=None,
    ):
        """For each layer in psd that is either a TypeLayer
        OR has a name matching text_pattern_re, and whose
        name is in color_cache, insert a same-sized solid-
        color PixelLayer directly above it with
        clipping=True. The original layer is left
        untouched (TypeLayers stay editable as text;
        rasterized pixel layers stay as their original
        pixels with the clipping color baked over).

        Returns the number of clipping layers added.
        """
        # Snapshot first - we mutate parent layer lists
        # below.
        text_layers = [
            layer for layer in psd.descendants()
            if (
                isinstance(layer, TypeLayer)
                or (
                    text_pattern_re is not None
                    and bool(
                        text_pattern_re.search(
                            layer.name or ""
                        )
                    )
                )
            )
        ]

        added = 0
        for text_layer in text_layers:
            color = color_cache.get(text_layer.name)
            if color is None:
                continue

            left = int(text_layer.left)
            top = int(text_layer.top)
            width = int(text_layer.width)
            height = int(text_layer.height)
            if width <= 0 or height <= 0:
                continue

            parent = text_layer.parent
            if parent is None:
                continue
            # Find current idx by identity. We may have
            # inserted earlier siblings already, so the
            # idx must be re-resolved here.
            idx = None
            for i, child in enumerate(parent):
                if child is text_layer:
                    idx = i
                    break
            if idx is None:
                continue

            fill_pil = Image.new(
                "RGBA",
                (width, height),
                (color[0], color[1], color[2], 255),
            )
            clip_layer = PixelLayer.frompil(
                pil_im=fill_pil,
                psd_file=psd,
                layer_name="recolor",
                top=top,
                left=left,
                compression=Compression.RLE,
            )
            # Set the human-readable name through the
            # property setter so non-ASCII glyphs route
            # through the tagged-block path.
            clip_layer.name = f"{text_layer.name} - recolor"
            clip_layer.opacity = 255
            clip_layer.visible = True
            clip_layer.blend_mode = BlendMode.NORMAL
            clip_layer.clipping = True

            # Insert at idx + 1 so the clip layer sits
            # visually above the text layer (psd_tools
            # internal lists are bottom-to-top).
            parent.insert(idx + 1, clip_layer)
            added += 1

        return added


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
