"""
PSD Layer Splitter

Opens a .psd file, extracts each renderable layer as a PNG,
and saves them to a folder with a naming convention that
encodes the index, total layer count, layer kind, and name.

Filename format: 001_of_015__pixel__Layer_Name.png

Also saves a _manifest.json with full per-layer metadata.
"""

import json
import os
import re
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from psd_tools import PSDImage
from psd_tools.api.layers import (
    AdjustmentLayer,
    FillLayer,
    Group,
    PixelLayer,
    ShapeLayer,
    SmartObjectLayer,
    TypeLayer,
)


def sanitize_name(name: str) -> str:
    """
    Make a layer name filesystem-safe.

    Replaces non-alphanumeric characters with underscores,
    collapses runs, and truncates to 60 chars.
    """
    safe = re.sub(r'[^\w\-]', '_', name)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:60] if safe else "unnamed"


def get_group_path(layer) -> str:
    """Walk up the parent chain to build a slash-separated
    group path like 'GroupA/SubGroup'."""
    parts: List[str] = []
    parent = layer.parent
    while parent is not None:
        parent_name = getattr(parent, 'name', None)
        if parent_name:
            parts.append(parent_name)
        parent = getattr(parent, 'parent', None)
    return '/'.join(reversed(parts))


def layer_kind_string(layer) -> str:
    """Return a short kind string for a layer."""
    if isinstance(layer, Group):
        return "group"
    if isinstance(layer, TypeLayer):
        return "type"
    if isinstance(layer, ShapeLayer):
        return "shape"
    if isinstance(layer, SmartObjectLayer):
        return "smartobject"
    if isinstance(layer, AdjustmentLayer):
        return "adjustment"
    if isinstance(layer, FillLayer):
        return "fill"
    if isinstance(layer, PixelLayer):
        return "pixel"
    return getattr(layer, 'kind', 'unknown')


def rasterize_layer(layer, psd_width, psd_height, sizing):
    """
    Rasterize a single layer to an RGBA PIL Image.

    Returns None if the layer cannot be rasterized.
    For 'canvas' sizing the image is placed at the
    correct offset on a canvas-sized transparent image.
    """
    img = None
    try:
        img = layer.composite()
    except Exception:
        pass

    if img is None:
        try:
            img = layer.topil()
        except Exception:
            pass

    if img is None:
        return None

    img = img.convert("RGBA")

    if img.size[0] == 0 or img.size[1] == 0:
        return None

    if sizing == "canvas":
        canvas = Image.new(
            "RGBA", (psd_width, psd_height), (0, 0, 0, 0)
        )
        left = max(0, layer.left)
        top = max(0, layer.top)
        canvas.paste(img, (left, top))
        return canvas

    return img


def pil_to_tensor(img):
    """Convert an RGBA PIL Image to an RGB torch tensor
    (B,H,W,3) and an alpha mask tensor (H,W)."""
    rgb = img.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(arr)[None, ...]

    if 'A' in img.getbands():
        alpha = np.array(
            img.getchannel('A')
        ).astype(np.float32) / 255.0
        mask_tensor = 1.0 - torch.from_numpy(alpha)
    else:
        h, w = arr.shape[:2]
        mask_tensor = torch.zeros(
            (h, w), dtype=torch.float32
        )

    return image_tensor, mask_tensor


class PSDLayerSplitter:
    """
    Extract every renderable layer from a PSD file.

    Saves numbered PNGs and a _manifest.json to the
    output folder. Also passes the images and masks
    through as list outputs for immediate downstream use.
    """

    CATEGORY = "Trent/PSD"
    FUNCTION = "split_psd"
    OUTPUT_NODE = True

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "images", "masks", "layer_names",
        "layer_count", "output_folder",
    )
    OUTPUT_IS_LIST = (True, True, True, False, False)
    OUTPUT_TOOLTIPS = (
        "Each extracted layer as an RGB image",
        "Alpha mask per layer (inverted: white = transparent)",
        "Original layer names",
        "Number of extracted layers",
        "Path to the output folder (chain to PSD Layer Loader)",
    )
    DESCRIPTION = (
        "Open a .psd file and save every renderable layer "
        "as a PNG. Filenames encode the index, total count, "
        "layer kind, and name so any single file tells you "
        "how many layers exist. A _manifest.json stores full "
        "metadata (opacity, blend mode, position, group path, "
        "text content, etc.)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "psd_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/file.psd",
                    "tooltip": (
                        "Absolute path to the .psd file"
                    ),
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/output/folder",
                    "tooltip": (
                        "Directory to save extracted "
                        "layer PNGs and manifest"
                    ),
                }),
                "layer_sizing": (
                    ["cropped", "canvas"],
                    {
                        "default": "cropped",
                        "tooltip": (
                            "cropped = layer bounds only. "
                            "canvas = placed on full PSD "
                            "canvas at correct position"
                        ),
                    },
                ),
                "layer_types": ("STRING", {
                    "default": (
                        "pixel,type,shape,"
                        "smartobject,fill"
                    ),
                    "tooltip": (
                        "Comma-separated layer kinds to "
                        "extract: pixel, type, shape, "
                        "smartobject, fill, adjustment. "
                        "Note: 'group' is handled by "
                        "extract_groups (off by default) "
                        "because exporting both a group "
                        "and its children produces "
                        "duplicate layers and shifts "
                        "indices."
                    ),
                }),
            },
            "optional": {
                "include_hidden": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Include layers marked as hidden "
                        "in the PSD"
                    ),
                }),
                "extract_groups": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "If True, also rasterize Group "
                        "nodes as their own layer. Off "
                        "by default because the group's "
                        "children are already exported "
                        "individually, so including the "
                        "group produces a duplicate that "
                        "covers the children when "
                        "recomposited."
                    ),
                }),
                "save_manifest": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Save _manifest.json alongside "
                        "the PNGs with full metadata"
                    ),
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Overwrite existing files in "
                        "the output directory"
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, layer_sizing=None, **kwargs):
        # Bypass strict COMBO validation. We coerce the
        # value inside split_psd() to handle ComfyUI's
        # widget-to-input conversion quirks that can shift
        # widgets_values entries.
        return True

    def split_psd(
        self,
        psd_path,
        output_dir,
        layer_sizing,
        layer_types,
        include_hidden=False,
        extract_groups=False,
        save_manifest=True,
        overwrite=True,
        **kwargs,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[str],
        int,
        str,
    ]:
        # Coerce inputs - ComfyUI's widget-to-input
        # conversion can pass unexpected types when
        # widgets_values get shifted during serialization.
        psd_path = str(psd_path or "").strip()
        output_dir = str(output_dir or "").strip()
        layer_types = str(layer_types or "")

        # layer_sizing must be a valid choice
        if layer_sizing not in ("cropped", "canvas"):
            print(
                f"[PSDLayerSplitter] Invalid layer_sizing "
                f"'{layer_sizing}', defaulting to 'cropped'"
            )
            layer_sizing = "cropped"

        include_hidden = bool(include_hidden)
        extract_groups = bool(extract_groups)
        save_manifest = bool(save_manifest)
        overwrite = bool(overwrite)

        if not psd_path:
            raise ValueError("psd_path is required")
        if not os.path.isfile(psd_path):
            raise FileNotFoundError(
                f"PSD file not found: {psd_path}"
            )
        if not output_dir:
            raise ValueError("output_dir is required")

        os.makedirs(output_dir, exist_ok=True)

        # Clean old layer files so previous PSDs don't
        # contaminate the loader when auto-queuing.
        _layer_pat = re.compile(
            r'^\d+_of_\d+__\w+__.+\.png$'
        )
        for old_file in os.listdir(output_dir):
            if _layer_pat.match(old_file) or old_file == "_manifest.json":
                os.remove(os.path.join(output_dir, old_file))

        wanted_kinds = {
            k.strip().lower()
            for k in layer_types.split(",")
            if k.strip()
        }

        psd = PSDImage.open(psd_path)
        canvas_w, canvas_h = psd.width, psd.height

        # Collect layers that pass the filters
        candidates = []
        for layer in psd.descendants():
            kind = layer_kind_string(layer)
            # Groups are gated by extract_groups (see
            # widget tooltip): exporting a group AND its
            # children produces duplicate renders, since
            # the children are already extracted.
            if kind == "group" and not extract_groups:
                continue
            if kind not in wanted_kinds and not (
                kind == "group" and extract_groups
            ):
                continue
            if not include_hidden and not layer.visible:
                continue
            candidates.append((layer, kind))

        total = len(candidates)
        if total == 0:
            print(
                "[PSDLayerSplitter] No renderable layers "
                "matched the filters"
            )
            black = torch.zeros(
                1, 64, 64, 3, dtype=torch.float32
            )
            mask = torch.ones(64, 64, dtype=torch.float32)
            return ([black], [mask], ["(empty)"], 0, output_dir)

        # Build unique filenames
        seen_names: Dict[str, int] = {}
        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        layer_names: List[str] = []
        manifest_layers: List[dict] = []
        skipped: List[dict] = []

        pad = max(3, len(str(total)))
        idx = 0

        for layer, kind in candidates:
            raw_name = layer.name or "unnamed"
            safe = sanitize_name(raw_name)

            # Handle duplicate names
            if safe in seen_names:
                seen_names[safe] += 1
                safe = f"{safe}_{seen_names[safe]}"
            else:
                seen_names[safe] = 1

            filename = (
                f"{str(idx).zfill(pad)}_of_"
                f"{str(total).zfill(pad)}__{kind}__{safe}.png"
            )

            # Rasterize
            img = rasterize_layer(
                layer, canvas_w, canvas_h, layer_sizing
            )

            if img is None:
                skipped.append({
                    "original_name": raw_name,
                    "kind": kind,
                    "reason": "no_pixel_data",
                })
                print(
                    f"[PSDLayerSplitter] Skipped layer "
                    f"'{raw_name}' ({kind}): no pixel data"
                )
                continue

            # Save PNG
            out_path = os.path.join(output_dir, filename)
            if overwrite or not os.path.exists(out_path):
                img.save(out_path, "PNG")

            # Convert to tensors
            image_tensor, mask_tensor = pil_to_tensor(img)
            images.append(image_tensor)
            masks.append(mask_tensor)
            layer_names.append(raw_name)

            # Manifest entry
            text_content = None
            if isinstance(layer, TypeLayer):
                try:
                    text_content = layer.text
                except Exception:
                    pass

            manifest_layers.append({
                "index": idx,
                "filename": filename,
                "original_name": raw_name,
                "kind": kind,
                "visible": layer.visible,
                "opacity": layer.opacity,
                "blend_mode": str(layer.blend_mode),
                "position": {
                    "left": layer.left,
                    "top": layer.top,
                },
                "size": {
                    "width": layer.width,
                    "height": layer.height,
                },
                "canvas_size": {
                    "width": canvas_w,
                    "height": canvas_h,
                },
                "clipping": getattr(
                    layer, 'clipping', False
                ),
                "group_path": get_group_path(layer),
                "text": text_content,
            })

            idx += 1

        # If every candidate was skipped
        if not images:
            print(
                "[PSDLayerSplitter] All layers skipped "
                "(no pixel data)"
            )
            black = torch.zeros(
                1, 64, 64, 3, dtype=torch.float32
            )
            mask = torch.ones(64, 64, dtype=torch.float32)
            return ([black], [mask], ["(empty)"], 0, output_dir)

        # Save manifest
        if save_manifest:
            manifest = {
                "source_psd": os.path.abspath(psd_path),
                "canvas_width": canvas_w,
                "canvas_height": canvas_h,
                "extraction_settings": {
                    "layer_sizing": layer_sizing,
                    "include_hidden": include_hidden,
                    "extract_groups": extract_groups,
                    "layer_types": sorted(wanted_kinds),
                },
                "layers": manifest_layers,
                "skipped_layers": skipped,
            }
            manifest_path = os.path.join(
                output_dir, "_manifest.json"
            )
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        extracted = len(images)
        print(
            f"[PSDLayerSplitter] Extracted {extracted} "
            f"layer(s) to {output_dir}"
        )

        return (
            images,
            masks,
            layer_names,
            extracted,
            output_dir,
        )


NODE_CLASS_MAPPINGS = {
    "PSDLayerSplitter": PSDLayerSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDLayerSplitter": "PSD Layer Splitter",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
