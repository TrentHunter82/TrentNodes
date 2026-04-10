"""
PSD Layer Loader

Loads PNG layers previously extracted by PSD Layer Splitter.
Parses the naming convention to recover index, total count,
layer kind, and name. Optionally loads _manifest.json for
full metadata.

Expected filename format: 001_of_015__pixel__Layer_Name.png
"""

import json
import os
import re
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps


# Matches filenames produced by PSDLayerSplitter
LAYER_RE = re.compile(
    r'^(\d+)_of_(\d+)__(\w+)__(.+)\.png$'
)


def natural_sort_key(path: str) -> List:
    """Sort key that orders numeric parts numerically."""
    name = os.path.basename(path)
    parts = re.split(r'(\d+)', name)
    return [
        int(p) if p.isdigit() else p.lower()
        for p in parts
    ]


class PSDLayerLoader:
    """
    Load layer PNGs from a folder produced by PSD Layer
    Splitter. Parses filenames to recover layer kind and
    name. Supports filtering by kind and index range.
    """

    CATEGORY = "Trent/PSD"
    FUNCTION = "load_layers"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "images", "masks", "layer_names",
        "layer_count", "manifest_json",
    )
    OUTPUT_IS_LIST = (True, True, True, False, False)
    OUTPUT_TOOLTIPS = (
        "Each layer as an RGB image",
        "Alpha mask per layer (inverted: white = transparent)",
        "Layer names parsed from filenames or manifest",
        "Number of loaded layers",
        "Raw JSON from _manifest.json (empty string if absent)",
    )
    DESCRIPTION = (
        "Load layer PNGs from a PSD Layer Splitter output "
        "folder. Filenames encode the index, total count, "
        "layer kind, and name. Supports filtering by kind "
        "and index range."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/layer/folder",
                    "tooltip": (
                        "Folder containing PNGs from "
                        "PSD Layer Splitter"
                    ),
                }),
            },
            "optional": {
                "filter_kind": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Comma-separated layer kinds to "
                        "load (empty = all). Parsed from "
                        "filenames: pixel, type, shape, "
                        "smartobject, group, fill"
                    ),
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": "First layer index to load",
                }),
                "end_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Last layer index to load "
                        "(0 = load all)"
                    ),
                }),
                "load_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Extract alpha channel as a "
                        "separate mask output"
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def load_layers(
        self,
        folder_path: str,
        filter_kind: str = "",
        start_index: int = 0,
        end_index: int = 0,
        load_alpha: bool = True,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[str],
        int,
        str,
    ]:
        folder_path = folder_path.strip()
        if not folder_path:
            raise ValueError("folder_path is required")
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        # Parse kind filter
        kind_filter = set()
        if filter_kind:
            kind_filter = {
                k.strip().lower()
                for k in filter_kind.split(",")
                if k.strip()
            }

        # Scan for matching PNGs
        entries = []
        for fname in os.listdir(folder_path):
            m = LAYER_RE.match(fname)
            if not m:
                continue
            idx = int(m.group(1))
            kind = m.group(3)
            name = m.group(4).replace('_', ' ')
            entries.append({
                "index": idx,
                "kind": kind,
                "name": name,
                "filename": fname,
                "path": os.path.join(folder_path, fname),
            })

        if not entries:
            raise FileNotFoundError(
                f"No PSD Layer Splitter PNGs found in "
                f"'{folder_path}'. Expected filenames like "
                f"001_of_015__pixel__Layer_Name.png"
            )

        # Sort by index
        entries.sort(key=lambda e: e["index"])

        # Apply kind filter
        if kind_filter:
            entries = [
                e for e in entries
                if e["kind"] in kind_filter
            ]

        # Apply index range (end_index 0 = load all)
        if end_index > 0:
            entries = [
                e for e in entries
                if start_index <= e["index"] <= end_index
            ]
        else:
            entries = [
                e for e in entries
                if e["index"] >= start_index
            ]

        if not entries:
            raise RuntimeError(
                "No layers matched the filter/index criteria"
            )

        # Load manifest if present
        manifest_path = os.path.join(
            folder_path, "_manifest.json"
        )
        manifest_json = ""
        manifest_names: Dict[str, str] = {}
        if os.path.isfile(manifest_path):
            try:
                with open(
                    manifest_path, "r", encoding="utf-8"
                ) as f:
                    manifest_data = json.load(f)
                manifest_json = json.dumps(
                    manifest_data, indent=2, ensure_ascii=False
                )
                # Build filename -> original_name lookup
                for lyr in manifest_data.get("layers", []):
                    fn = lyr.get("filename", "")
                    orig = lyr.get("original_name", "")
                    if fn and orig:
                        manifest_names[fn] = orig
            except Exception as exc:
                print(
                    f"[PSDLayerLoader] Could not read "
                    f"manifest: {exc}"
                )

        # Load images
        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        layer_names: List[str] = []

        for entry in entries:
            try:
                img = Image.open(entry["path"])
                img = ImageOps.exif_transpose(img)

                rgb = img.convert("RGB")
                arr = np.array(rgb).astype(
                    np.float32
                ) / 255.0
                image_tensor = torch.from_numpy(arr)[None, ...]
                images.append(image_tensor)

                if load_alpha and 'A' in img.getbands():
                    alpha = np.array(
                        img.getchannel('A')
                    ).astype(np.float32) / 255.0
                    mask_tensor = (
                        1.0 - torch.from_numpy(alpha)
                    )
                else:
                    h, w = arr.shape[:2]
                    mask_tensor = torch.zeros(
                        (h, w), dtype=torch.float32
                    )
                masks.append(mask_tensor)

                # Prefer manifest name, fall back to parsed
                name = manifest_names.get(
                    entry["filename"], entry["name"]
                )
                layer_names.append(name)

            except Exception as exc:
                print(
                    f"[PSDLayerLoader] Error loading "
                    f"{entry['filename']}: {exc}"
                )
                traceback.print_exc()
                continue

        if not images:
            raise RuntimeError(
                "Failed to load any layer images"
            )

        count = len(images)
        print(
            f"[PSDLayerLoader] Loaded {count} layer(s) "
            f"from {folder_path}"
        )

        return (
            images,
            masks,
            layer_names,
            count,
            manifest_json,
        )


NODE_CLASS_MAPPINGS = {
    "PSDLayerLoader": PSDLayerLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDLayerLoader": "PSD Layer Loader",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
