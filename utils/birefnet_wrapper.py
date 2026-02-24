"""
BiRefNet wrapper for TrentNodes.

Provides background removal using BiRefNet from Hugging Face transformers,
independent of other ComfyUI custom nodes.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import folder_paths


# Model cache - separate caches for different model variants
_birefnet_models = {}  # {model_name: model}
_birefnet_device = None

# Available model variants (fastest to highest quality)
BIREFNET_MODELS = {
    "lite": "ZhengPeng7/BiRefNet_lite",
    "standard": "ZhengPeng7/BiRefNet",
}

# Resolution presets
RESOLUTION_PRESETS = {
    "fast": 512,
    "balanced": 768,
    "quality": 1024,
}


def is_birefnet_available() -> bool:
    """
    Check if BiRefNet dependencies are available.

    Returns:
        True if transformers is installed
    """
    try:
        from transformers import AutoModelForImageSegmentation
        return True
    except ImportError:
        return False


def load_birefnet(
    device: torch.device,
    model_variant: str = "lite"
) -> Tuple[any, any]:
    """
    Load BiRefNet model from Hugging Face.

    Args:
        device: torch device to load model on
        model_variant: "lite" (faster) or "standard" (better quality)

    Returns:
        Tuple of (model, None) or (None, None) if not available
    """
    global _birefnet_models, _birefnet_device

    # Validate variant
    if model_variant not in BIREFNET_MODELS:
        model_variant = "lite"

    # Return cached model if available and on same device
    cache_key = model_variant
    if cache_key in _birefnet_models and _birefnet_device == device:
        return _birefnet_models[cache_key], None

    try:
        from transformers import AutoModelForImageSegmentation

        model_name = BIREFNET_MODELS[model_variant]
        print(f"[TrentNodes] Loading BiRefNet ({model_variant}) from HF...")

        # Use cache directory in ComfyUI models folder
        cache_dir = os.path.join(folder_paths.models_dir, "birefnet")
        os.makedirs(cache_dir, exist_ok=True)

        # Load model
        model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        # Force float32 to avoid dtype issues
        model = model.to(device=device, dtype=torch.float32).eval()

        # Note: torch.compile disabled for BiRefNet - the model uses
        # deform_conv2d which doesn't benefit from CUDA graphs and
        # produces many warnings without performance improvement.

        _birefnet_models[cache_key] = model
        _birefnet_device = device

        print(f"[TrentNodes] BiRefNet ({model_variant}) loaded.")
        return model, None

    except Exception as e:
        print(f"[TrentNodes] Failed to load BiRefNet: {e}")
        print("[TrentNodes] Install with: pip install transformers")
        return None, None


def birefnet_segment(
    image: torch.Tensor,
    device: torch.device,
    resolution: int = 512,
    model_variant: str = "lite",
    max_batch: int = 4
) -> Optional[torch.Tensor]:
    """
    Segment foreground using BiRefNet with GPU-accelerated batch processing.

    Processes frames in sub-batches to cap peak VRAM usage. Each
    sub-batch runs through the model independently, then results
    are concatenated. This prevents OOM on large video batches.

    Args:
        image: (B, H, W, C) tensor in [0, 1] range
        device: torch device
        resolution: Processing resolution (512=fast, 768=balanced, 1024=quality)
        model_variant: "lite" (faster) or "standard" (better quality)
        max_batch: Max frames per forward pass (caps peak VRAM)

    Returns:
        mask: (B, H, W) tensor, 1 = foreground, 0 = background
        Returns None if BiRefNet is not available
    """
    model, _ = load_birefnet(device, model_variant)
    if model is None:
        return None

    B, H, W, C = image.shape

    # Clamp resolution to valid range
    resolution = max(256, min(1024, resolution))

    # GPU-accelerated batch preprocessing
    # Convert from (B, H, W, C) [0,1] to (B, C, H, W) normalized
    batch_tensor = image[..., :3].permute(0, 3, 1, 2)  # (B, 3, H, W)

    # Resize to processing resolution on GPU
    batch_tensor = F.interpolate(
        batch_tensor,
        size=(resolution, resolution),
        mode='bilinear',
        align_corners=False
    )

    # Normalize with ImageNet stats (on GPU) - create once, reuse
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device, dtype=batch_tensor.dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device, dtype=batch_tensor.dtype
    ).view(1, 3, 1, 1)
    batch_tensor = (batch_tensor - mean) / std

    # Ensure float32 for model
    batch_tensor = batch_tensor.float()

    # Sub-batch inference to cap peak VRAM
    if B <= max_batch:
        # Small batch: single forward pass (fastest)
        with torch.no_grad(), torch.amp.autocast(
            device.type, enabled=False
        ):
            preds = model(batch_tensor)[-1].sigmoid()
    else:
        # Large batch: chunk to avoid OOM
        pred_chunks = []
        for start in range(0, B, max_batch):
            chunk = batch_tensor[start:start + max_batch]
            with torch.no_grad(), torch.amp.autocast(
                device.type, enabled=False
            ):
                pred = model(chunk)[-1].sigmoid()
            pred_chunks.append(pred)
        preds = torch.cat(pred_chunks, dim=0)

    # Resize all masks back to original size at once
    masks = F.interpolate(
        preds,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)

    return masks


def clear_birefnet_cache():
    """Clear BiRefNet model from memory."""
    global _birefnet_models, _birefnet_device

    _birefnet_models.clear()
    _birefnet_device = None

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[TrentNodes] BiRefNet cache cleared.")
