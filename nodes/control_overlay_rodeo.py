"""
Control Overlay Rodeo - interactive control-map overlay player.

Feed it an image (or a whole batch) plus any of the usual ControlNet
preprocessor outputs - depth, HED/lineart, canny, pose - and it plays
them back inside the node with live per-layer toggles, opacity and
blend modes. The browser previews the mix live; the next run outputs
the same mix at full resolution.
"""

import json
import os
import uuid
from typing import Any, Dict

import torch
from PIL import Image

import comfy.utils
import folder_paths

CONTROL_LAYERS = ("depth", "hed", "canny", "pose")

DEFAULT_LAYERS = {
    "original": {"on": True, "opacity": 1.0, "blend": "normal"},
    "depth": {"on": False, "opacity": 1.0, "blend": "normal"},
    "hed": {"on": True, "opacity": 1.0, "blend": "screen"},
    "canny": {"on": True, "opacity": 1.0, "blend": "screen"},
    "pose": {"on": True, "opacity": 1.0, "blend": "screen"},
}

# Separable blend functions, matching the canvas modes in
# js/control_overlay_rodeo.js. "add" is canvas "lighter", which is
# Porter-Duff plus rather than a blend, so composite() handles it.
BLENDS = {
    "normal": lambda base, top: top,
    "screen": lambda base, top: 1 - (1 - base) * (1 - top),
    "lighten": torch.maximum,
    "multiply": torch.mul,
    "difference": lambda base, top: (base - top).abs(),
}


def composite(batches, layers, height, width, frames):
    """Mix the enabled layers in order, as the preview canvas does."""
    first = batches[0][1]
    out = torch.zeros((frames, height, width, 3), dtype=first.dtype, device=first.device)
    for name, batch in batches:
        cfg = layers[name]
        if not cfg["on"] or cfg["opacity"] <= 0:
            continue
        for i in range(frames):
            top = batch[min(i, len(batch) - 1)][..., :3]
            if top.shape[:2] != (height, width):
                top = comfy.utils.common_upscale(
                    top.movedim(-1, 0).unsqueeze(0), width, height,
                    "bilinear", "disabled")[0].movedim(0, -1)
            if cfg["blend"] == "add":
                out[i].add_(top, alpha=cfg["opacity"]).clamp_(max=1)
            else:
                out[i].lerp_(BLENDS[cfg["blend"]](out[i], top), cfg["opacity"])
    return out


class ControlOverlayRodeo:
    """
    Preview a batch with control maps overlaid on top of it, and output
    the mix.

    Each connected layer is written to the temp folder as JPEG frames
    and played back by js/control_overlay_rodeo.js, which stores the
    layer toggles, opacity and blend in the hidden layer_settings
    widget. Layers with fewer frames than the longest one hold their
    last frame, so a single image pairs fine with a full control batch
    and vice versa.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Source image or frame batch "
                               "(the Original layer)"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.001,
                    "tooltip": "Playback rate for the preview"
                }),
                "max_preview_size": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Longest side of the preview frames "
                               "(0 = full resolution). Lower is "
                               "faster to write and load. The output "
                               "is always full resolution."
                }),
                "layer_settings": ("STRING", {
                    "default": json.dumps(DEFAULT_LAYERS),
                    "hidden": True,
                    "tooltip": "Layer toggles, opacity and blend, "
                               "set by the in-node controls"
                }),
            },
            "optional": {
                "depth": ("IMAGE", {
                    "tooltip": "Depth map batch"
                }),
                "hed": ("IMAGE", {
                    "tooltip": "HED / soft edge / lineart batch"
                }),
                "canny": ("IMAGE", {
                    "tooltip": "Canny edge batch"
                }),
                "pose": ("IMAGE", {
                    "tooltip": "Rendered pose batch "
                               "(DWPose, OpenPose, ...)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Play an image batch with control maps overlaid on top. "
        "Connect any of depth, hed, canny and pose from your "
        "preprocessor nodes, then toggle each layer, set its opacity "
        "and blend mode, scrub, step frames and expand to full "
        "screen - all live, without re-running the workflow. "
        "Outputs the mix at the source resolution; queue again "
        "after changing layers to update the output."
    )

    def preview(self, images, fps, max_preview_size, layer_settings, **controls):
        subfolder = f"control_overlay_rodeo/{uuid.uuid4().hex[:12]}"
        out_dir = os.path.join(folder_paths.get_temp_directory(), subfolder)
        os.makedirs(out_dir, exist_ok=True)

        height, width = images.shape[1:3]
        scale = 1.0
        if max_preview_size > 0:
            scale = min(1.0, max_preview_size / max(width, height))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))

        batches = [("original", images)]
        batches += [(name, controls[name]) for name in CONTROL_LAYERS
                    if controls.get(name) is not None]
        for name, batch in batches:
            for i, frame in enumerate(batch):
                pixels = (frame[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
                img = Image.fromarray(pixels)
                if img.size != size:
                    img = img.resize(size, Image.Resampling.BILINEAR)
                img.save(os.path.join(out_dir, f"{name}_{i:05d}.jpg"), quality=92)

        frames = max(len(batch) for _, batch in batches)
        mixed = composite(batches, json.loads(layer_settings), height, width, frames)

        return {"ui": {"rodeo": [{
            "subfolder": subfolder,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": frames,
            "layers": [{"name": name, "count": len(batch)} for name, batch in batches],
        }]}, "result": (mixed,)}


NODE_CLASS_MAPPINGS = {
    "ControlOverlayRodeo": ControlOverlayRodeo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlOverlayRodeo": "Control Overlay Rodeo",
}
