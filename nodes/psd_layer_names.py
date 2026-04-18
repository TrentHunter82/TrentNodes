"""
PSD Layer Names

List every layer name in a .psd file so you can pick the
one to hand to PSDLayerSaveAsPSD's `target_layer_name`.
No rasterization, no manifest side-effects - just walks
the layer tree.
"""

import os
from typing import Any, Dict, List, Tuple

from psd_tools import PSDImage
from psd_tools.api.layers import Group


def layer_kind_string(layer) -> str:
    """Short kind string matching PSDLayerSplitter."""
    from psd_tools.api.layers import (
        AdjustmentLayer,
        FillLayer,
        PixelLayer,
        ShapeLayer,
        SmartObjectLayer,
        TypeLayer,
    )
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
    return getattr(layer, "kind", "unknown")


def walk_descendants(psd: PSDImage, include_groups: bool):
    """
    Yield (index, kind, name, group_path) for every layer.
    Index numbering matches PSDLayerSplitter:
      - iterates psd.descendants() in the same order
      - skips groups unless include_groups is True
    """
    idx = 0
    for layer in psd.descendants():
        kind = layer_kind_string(layer)
        if kind == "group" and not include_groups:
            continue
        # Reconstruct the group path
        parts: List[str] = []
        parent = layer.parent
        while parent is not None:
            pname = getattr(parent, "name", None)
            if pname:
                parts.append(pname)
            parent = getattr(parent, "parent", None)
        group_path = "/".join(reversed(parts))
        yield idx, kind, layer.name or "unnamed", group_path
        idx += 1


class PSDLayerNames:
    """List all layer names in a .psd, pick one by index."""

    CATEGORY = "Trent/PSD"
    FUNCTION = "list_names"

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("all_names", "name_at_index", "count")
    OUTPUT_TOOLTIPS = (
        "Multiline list of every layer - feed into a "
        "Show Text node to see what's in the PSD",
        "The layer name at `index` (after the same "
        "filtering PSDLayerSplitter uses). Feed into "
        "PSDLayerSaveAsPSD's target_layer_name.",
        "How many layers the PSD has (after filtering)",
    )
    DESCRIPTION = (
        "Open a .psd and list all its layer names. Pairs "
        "with PSDLayerSaveAsPSD: use all_names to discover "
        "what's in the file, then name_at_index to wire "
        "the chosen name straight in."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "psd_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/file.psd",
                    "tooltip": "Absolute path to the .psd",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "tooltip": (
                        "Which layer to return in "
                        "name_at_index (0 = bottom)"
                    ),
                }),
            },
            "optional": {
                "include_groups": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Match the splitter default: "
                        "groups hidden. Turn on if you "
                        "want to target a group by name."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def list_names(
        self,
        psd_path,
        index,
        include_groups=False,
        **kwargs,
    ) -> Tuple[str, str, int]:
        psd_path = str(psd_path or "").strip()
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = 0
        include_groups = bool(include_groups)

        if not psd_path:
            raise ValueError("psd_path is required")
        if not os.path.isfile(psd_path):
            raise FileNotFoundError(
                f"PSD not found: {psd_path}"
            )

        psd = PSDImage.open(psd_path)

        entries = list(
            walk_descendants(psd, include_groups=include_groups)
        )

        lines = []
        for i, kind, name, group_path in entries:
            loc = f" [{group_path}]" if group_path else ""
            lines.append(f"{i:>3}  {kind:<12} {name}{loc}")
        all_names = "\n".join(lines) if lines else "(no layers)"

        name_at_index = ""
        if 0 <= index < len(entries):
            name_at_index = entries[index][2]
        else:
            print(
                f"[PSDLayerNames] index {index} out of "
                f"range (0..{len(entries) - 1})"
            )

        print(
            f"[PSDLayerNames] {len(entries)} layer(s) in "
            f"{os.path.basename(psd_path)}; "
            f"picked '{name_at_index}' at index {index}"
        )

        return (all_names, name_at_index, len(entries))


NODE_CLASS_MAPPINGS = {
    "PSDLayerNames": PSDLayerNames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDLayerNames": "PSD Layer Names",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
