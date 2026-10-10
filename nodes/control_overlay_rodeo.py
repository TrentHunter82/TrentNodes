"""
Control Overlay Rodeo - interactive control-map overlay player.

Feed it an image (or a whole batch) plus any of the usual ControlNet
preprocessor outputs - depth, HED/lineart, canny, pose - and it plays
them back inside the node with live per-layer toggles, opacity and
blend modes. All compositing happens in the browser, so flipping
layers never re-runs the workflow.
"""

import os
import uuid
from typing import Any, Dict

import numpy as np
from PIL import Image

import folder_paths

CONTROL_LAYERS = ("depth", "hed", "canny", "pose")


class ControlOverlayRodeo:
    """
    Preview a batch with control maps overlaid on top of it.

    Each connected layer is written to the temp folder as JPEG frames
    and played back by js/control_overlay_rodeo.js. Layers with fewer
    frames than the longest one hold their last frame, so a single
    image pairs fine with a full control batch and vice versa.
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
                               "faster to write and load."
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

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Play an image batch with control maps overlaid on top. "
        "Connect any of depth, hed, canny and pose from your "
        "preprocessor nodes, then toggle each layer, set its opacity "
        "and blend mode, scrub, step frames and expand to full "
        "screen - all live, without re-running the workflow."
    )

    def preview(self, images, fps, max_preview_size, **controls):
        subfolder = f"control_overlay_rodeo/{uuid.uuid4().hex[:12]}"
        out_dir = os.path.join(folder_paths.get_temp_directory(), subfolder)
        os.makedirs(out_dir, exist_ok=True)

        height, width = images.shape[1:3]
        scale = 1.0
        if max_preview_size > 0:
            scale = min(1.0, max_preview_size / max(width, height))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))

        layers = []
        batches = [("original", images)]
        batches += [(name, controls.get(name)) for name in CONTROL_LAYERS]
        for name, batch in batches:
            if batch is None:
                continue
            for i, frame in enumerate(batch):
                pixels = (frame[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
                img = Image.fromarray(pixels)
                if img.size != size:
                    img = img.resize(size, Image.Resampling.BILINEAR)
                img.save(os.path.join(out_dir, f"{name}_{i:05d}.jpg"), quality=92)
            layers.append({"name": name, "count": len(batch)})

        return {"ui": {"rodeo": [{
            "subfolder": subfolder,
            "fps": fps,
            "width": width,
            "height": height,
            "frames": max(layer["count"] for layer in layers),
            "layers": layers,
        }]}}


NODE_CLASS_MAPPINGS = {
    "ControlOverlayRodeo": ControlOverlayRodeo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ControlOverlayRodeo": "Control Overlay Rodeo",
}
