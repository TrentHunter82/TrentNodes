"""
Shared helpers for PSD-related nodes.

Tensor/PIL conversion, target-bound resizing, and
background color parsing used by both PSDLayerCompositor
and PSDLayerSaveAsPSD.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageColor

from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import Compression


def tensor_to_pil_rgba(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI image tensor (B,H,W,C) to a PIL
    RGBA image. Takes the first frame of the batch.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 3:
        return Image.fromarray(arr, "RGB").convert("RGBA")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, "RGBA")
    raise ValueError(
        f"Unexpected channel count: {arr.shape[-1]}"
    )


def resize_to_bounds(
    img: Image.Image,
    target_w: int,
    target_h: int,
    mode: str,
) -> Image.Image:
    """Resize a PIL image to target dimensions."""
    if target_w <= 0 or target_h <= 0:
        return img

    if mode == "stretch":
        return img.resize(
            (target_w, target_h), Image.LANCZOS
        )

    if mode == "fit":
        src_w, src_h = img.size
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = img.resize(
            (new_w, new_h), Image.LANCZOS
        )
        canvas = Image.new(
            "RGBA", (target_w, target_h), (0, 0, 0, 0)
        )
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y), resized)
        return canvas

    if mode == "cover":
        src_w, src_h = img.size
        scale = max(target_w / src_w, target_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = img.resize(
            (new_w, new_h), Image.LANCZOS
        )
        crop_x = (new_w - target_w) // 2
        crop_y = (new_h - target_h) // 2
        return resized.crop((
            crop_x, crop_y,
            crop_x + target_w, crop_y + target_h,
        ))

    # center
    canvas = Image.new(
        "RGBA", (target_w, target_h), (0, 0, 0, 0)
    )
    src_w, src_h = img.size
    offset_x = (target_w - src_w) // 2
    offset_y = (target_h - src_h) // 2
    canvas.paste(img, (offset_x, offset_y), img)
    return canvas


def parse_background_color(
    spec: str,
) -> Tuple[int, int, int, int]:
    """
    Parse a background color spec into an RGBA tuple.

    Accepts:
      - 'transparent' (fully transparent)
      - Any CSS color name: 'white', 'black', 'cyan',
        'magenta', 'red', 'navy', 'silver', etc.
        (see PIL.ImageColor.colormap for the full list)
      - Hex: '#RGB', '#RRGGBB', '#RRGGBBAA'
      - Functional: 'rgb(r,g,b)', 'rgba(...)',
        'hsl(h,s%,l%)', 'hsv(...)'
    """
    spec = (spec or "").strip()
    # Normalize to alphanumerics only for name matching
    # (tolerates stray whitespace, punctuation, nbsp, etc.)
    alnum = "".join(c for c in spec.lower() if c.isalnum())
    if not alnum or alnum == "transparent":
        return (0, 0, 0, 0)

    # Try ImageColor first (supports ~140 CSS names +
    # every hex/functional form Pillow knows about).
    candidates = [spec, spec.lower(), alnum]
    for cand in candidates:
        try:
            rgba = ImageColor.getrgb(cand)
        except ValueError:
            continue
        if len(rgba) == 3:
            return (rgba[0], rgba[1], rgba[2], 255)
        return (rgba[0], rgba[1], rgba[2], rgba[3])

    raise ValueError(
        f"Invalid background_color: {spec!r}. Use "
        f"'transparent', a CSS color name like 'white', "
        f"'cyan', 'magenta', or hex like '#FFFFFF'"
    )


def find_layer_by_name(
    container, target_name: str
) -> Optional[Tuple[Any, int, Any]]:
    """
    Recursively walk the layer tree looking for the first
    layer whose name matches target_name. Returns
    (parent_container, index_in_parent, layer) or None.
    """
    for idx, layer in enumerate(container):
        if layer.name == target_name:
            return (container, idx, layer)
        if layer.is_group():
            hit = find_layer_by_name(layer, target_name)
            if hit is not None:
                return hit
    return None


def list_all_layer_names(
    container, prefix: str = ""
) -> List[str]:
    """Flat list of every layer name with group path."""
    names: List[str] = []
    for layer in container:
        path = (
            f"{prefix}/{layer.name}"
            if prefix else layer.name
        )
        names.append(path)
        if layer.is_group():
            names.extend(list_all_layer_names(layer, path))
    return names


def replace_psd_layer_pixels(
    psd: PSDImage,
    target_layer_name: str,
    replacement_pil: Image.Image,
    resize_mode: str,
) -> str:
    """
    Replace one PSD layer's pixels with replacement_pil.

    Locates the layer by name (recursively walks groups),
    rasterizes the replacement to a PixelLayer fitted to
    the original layer's bounding box (or kept native if
    resize_mode == 'native'), and swaps it in place.
    Preserves the original layer's name, opacity, blend
    mode, and visibility. Mutates psd in place.

    Returns the original layer name on success.
    """
    hit = find_layer_by_name(psd, target_layer_name)
    if hit is None:
        available = list_all_layer_names(psd)
        raise ValueError(
            f"Layer '{target_layer_name}' not found. "
            f"Available layers:\n  - "
            + "\n  - ".join(available)
        )
    parent, idx, old_layer = hit

    old_left = int(old_layer.left)
    old_top = int(old_layer.top)
    old_w = int(old_layer.width)
    old_h = int(old_layer.height)
    if old_w <= 0 or old_h <= 0:
        old_left, old_top = 0, 0
        old_w, old_h = psd.width, psd.height

    if resize_mode == "native":
        new_pil = replacement_pil
    else:
        new_pil = resize_to_bounds(
            replacement_pil, old_w, old_h, resize_mode
        )

    old_name = old_layer.name
    old_opacity = int(old_layer.opacity)
    old_blend_mode = old_layer.blend_mode
    old_visible = bool(old_layer.visible)

    # frompil writes layer_name straight to the legacy
    # Pascal record (mac_roman encoded), which crashes
    # on non-Latin names. Use a safe ASCII placeholder,
    # then assign the real name through the Layer.name
    # setter, which routes Unicode into the proper
    # tagged block.
    new_layer = PixelLayer.frompil(
        pil_im=new_pil,
        psd_file=psd,
        layer_name="Layer",
        top=old_top,
        left=old_left,
        compression=Compression.RLE,
    )
    new_layer.name = old_name
    new_layer.opacity = old_opacity
    new_layer.blend_mode = old_blend_mode
    new_layer.visible = old_visible

    parent[idx] = new_layer
    return old_name
