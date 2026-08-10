"""
Tensor -> JPEG base64 helpers for the H3 Auto Prompt Generator.

Vendored rather than imported from comfy_api_nodes.util.conversions:
that module is internal to core's paid API nodes and can drift. The
needed code is small and this keeps the package importable offline.
"""

import base64
import io

import numpy as np
import torch
from PIL import Image

FRAME_MAX_SIDE = 1024
# Identity/wardrobe detail matters most on the reference image, so it
# gets a higher resolution budget than the sampled frames.
REFERENCE_MAX_SIDE = 1344
JPEG_QUALITY = 90


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """(H, W, C) or (1, H, W, C) [0,1] float tensor -> RGB PIL image."""
    if image.dim() == 4:
        image = image[0]
    arr = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr)


def pil_to_jpeg_b64(pil: Image.Image, max_side: int = FRAME_MAX_SIDE) -> str:
    """Resize to fit max_side, encode JPEG, return raw base64 string."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    width, height = pil.size
    scale = max_side / max(width, height)
    if scale < 1.0:
        pil = pil.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def tensor_to_jpeg_b64(
    image: torch.Tensor, max_side: int = FRAME_MAX_SIDE
) -> str:
    return pil_to_jpeg_b64(tensor_to_pil(image), max_side=max_side)
