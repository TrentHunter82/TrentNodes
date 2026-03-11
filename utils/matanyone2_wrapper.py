"""
MatAnyone 2 wrapper for TrentNodes.

Provides temporally-consistent video matting using
MatAnyone 2 (CVPR 2026). Handles model download,
caching, VRAM management, and frame-by-frame inference
with progress reporting.

Model weights (~141 MB) are auto-downloaded from GitHub
on first use and cached in ComfyUI/models/matanyone2/.
"""

import gc
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf

import folder_paths

from .matanyone2.inference.inference_core import (
    InferenceCore,
)
from .matanyone2.model.matanyone2 import MatAnyone2


# Model cache
_matanyone2_model: Optional[MatAnyone2] = None
_matanyone2_device: Optional[torch.device] = None

# Config path (vendored base.yaml)
_CONFIG_PATH = (
    Path(__file__).parent / "matanyone2" / "base.yaml"
)

# Download URL (GitHub release v1.0.0)
_DOWNLOAD_URL = (
    "https://github.com/pq-yang/MatAnyone2/releases/"
    "download/v1.0.0/matanyone2.pth"
)
_HF_FILENAME = "matanyone2.pth"


def is_matanyone2_available() -> bool:
    """Check if MatAnyone 2 dependencies are available."""
    try:
        from omegaconf import OmegaConf  # noqa: F811
        return True
    except ImportError:
        return False


def _get_weight_path() -> str:
    """
    Download MatAnyone 2 weights if needed, return path.

    Downloads matanyone2.pth from GitHub releases into
    ComfyUI/models/matanyone2/ on first call.
    """
    cache_dir = os.path.join(
        folder_paths.models_dir, "matanyone2"
    )
    os.makedirs(cache_dir, exist_ok=True)

    weight_path = os.path.join(cache_dir, _HF_FILENAME)

    if os.path.exists(weight_path):
        return weight_path

    print(
        "[TrentNodes] Downloading MatAnyone 2 weights"
        " from GitHub..."
    )
    import urllib.request

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        if pct % 10 == 0:
            print(
                f"[TrentNodes] Downloading MatAnyone 2:"
                f" {pct}%"
            )

    urllib.request.urlretrieve(
        _DOWNLOAD_URL, weight_path, _progress
    )
    print("[TrentNodes] MatAnyone 2 weights downloaded.")
    return weight_path


def _load_weights(
    path: str, device: torch.device
) -> dict:
    """Load model weights from .pth or .safetensors."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(device))
    else:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )


def load_matanyone2(
    device: torch.device,
) -> Tuple[MatAnyone2, object]:
    """
    Load MatAnyone 2 model, returning (model, config).

    Uses global cache: returns cached model if device
    matches. Otherwise loads fresh from disk.
    """
    global _matanyone2_model, _matanyone2_device

    if (
        _matanyone2_model is not None
        and _matanyone2_device == device
    ):
        return _matanyone2_model, _matanyone2_model.cfg

    try:
        cfg = OmegaConf.load(str(_CONFIG_PATH))
        weight_path = _get_weight_path()

        print("[TrentNodes] Loading MatAnyone 2 model...")

        model = (
            MatAnyone2(cfg, single_object=True)
            .to(device)
            .eval()
        )

        weights = _load_weights(weight_path, device)
        model.load_weights(weights)

        _matanyone2_model = model
        _matanyone2_device = device

        print("[TrentNodes] MatAnyone 2 loaded.")
        return model, cfg

    except Exception as e:
        print(
            f"[TrentNodes] Failed to load"
            f" MatAnyone 2: {e}"
        )
        return None, None


def clear_matanyone2_cache() -> None:
    """Free MatAnyone 2 model from VRAM and clear cache."""
    global _matanyone2_model, _matanyone2_device

    _matanyone2_model = None
    _matanyone2_device = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[TrentNodes] MatAnyone 2 cache cleared.")


def run_matanyone2(
    images: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    n_warmup: int = 10,
    mask_frame_index: int = 0,
) -> Optional[torch.Tensor]:
    """
    Run MatAnyone 2 video matting inference.

    Processes video frames sequentially with temporal
    memory propagation for flicker-free alpha mattes.

    Args:
        images: (B, H, W, C) image batch in [0, 1]
        mask: (H, W) single-frame mask in [0, 1]
        device: torch device
        n_warmup: Warmup iterations on reference frame
        mask_frame_index: Which frame the mask applies to

    Returns:
        (B, H, W) alpha matte tensor in [0, 1],
        or None if model unavailable
    """
    from comfy.utils import ProgressBar

    model, cfg = load_matanyone2(device)
    if model is None:
        return None

    b, h, w, c = images.shape

    # (B, H, W, C) [0,1] -> (B, C, H, W) [0,1]
    vframes = (
        images[..., :3]
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
    )

    # Mask: (H, W) [0,1] -> (H, W) [0,255]
    mask_ma = mask.to(
        device=device, dtype=torch.float32
    ) * 255.0
    if mask_ma.dim() == 3:
        mask_ma = mask_ma[0]

    # Fresh processor for each run (resets memory state)
    processor = InferenceCore(model, cfg=model.cfg)

    mask_frame_index = min(mask_frame_index, b - 1)

    repeated = (
        vframes[mask_frame_index]
        .unsqueeze(0)
        .repeat(n_warmup, 1, 1, 1)
    )

    total_steps = n_warmup + b
    pbar = ProgressBar(total_steps)

    # Warmup phase
    for ti in range(n_warmup):
        image = repeated[ti]
        if ti == 0:
            output_prob = processor.step(
                image, mask_ma, objects=[1]
            )
            output_prob = processor.step(
                image, first_frame_pred=True
            )
        else:
            output_prob = processor.step(
                image, first_frame_pred=True
            )
        pbar.update(1)

    # Allocate output
    alphas: List[torch.Tensor] = [None] * b

    # Store warmup frame result
    alphas[mask_frame_index] = (
        processor.output_prob_to_mask(output_prob)
        .unsqueeze(0)
    )
    pbar.update(1)

    # Forward propagation
    for ti in range(mask_frame_index + 1, b):
        image = vframes[ti]
        output_prob = processor.step(image)
        alphas[ti] = (
            processor.output_prob_to_mask(output_prob)
            .unsqueeze(0)
        )
        pbar.update(1)

    # Fill frames before mask_frame_index
    if mask_frame_index > 0:
        fill = alphas[mask_frame_index]
        for ti in range(mask_frame_index):
            if alphas[ti] is None:
                alphas[ti] = fill
            pbar.update(1)

    result = torch.cat(alphas, dim=0)

    if result.shape[1] != h or result.shape[2] != w:
        import torch.nn.functional as F

        result = F.interpolate(
            result.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

    return result.clamp(0.0, 1.0)
