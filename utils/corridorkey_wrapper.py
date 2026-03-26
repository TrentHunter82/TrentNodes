"""
CorridorKey wrapper for TrentNodes.

Provides neural green screen keying using CorridorKey
(Corridor Digital). Handles model download, caching,
VRAM management, and per-frame inference with progress
reporting.

Model weights (~300 MB) are auto-downloaded from
HuggingFace on first use and cached in
ComfyUI/models/corridorkey/.
"""

import gc
import os
from typing import Optional, Tuple

import numpy as np
import torch

import folder_paths

from .corridorkey import CorridorKeyEngine


# Model cache
_corridorkey_engine: Optional[CorridorKeyEngine] = None
_corridorkey_device: Optional[torch.device] = None

# HuggingFace model info
_HF_REPO = "nikopueringer/CorridorKey"
_HF_FILENAME = "CorridorKey_v1.0.pth"


def is_corridorkey_available() -> bool:
    """Check if CorridorKey dependencies are available."""
    try:
        import timm  # noqa: F401
        return True
    except ImportError:
        return False


def _get_weight_path() -> str:
    """
    Download CorridorKey weights if needed, return path.

    Downloads CorridorKey_v1.0.pth from HuggingFace Hub
    into ComfyUI/models/corridorkey/ on first call.
    Subsequent calls return the cached path immediately.
    """
    cache_dir = os.path.join(
        folder_paths.models_dir, "corridorkey"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Check for existing weight file
    weight_path = os.path.join(
        cache_dir, "CorridorKey.pth"
    )
    if os.path.exists(weight_path):
        return weight_path

    # Also check for the versioned filename
    versioned_path = os.path.join(
        cache_dir, _HF_FILENAME
    )
    if os.path.exists(versioned_path):
        return versioned_path

    # Download from HuggingFace
    print(
        "[TrentNodes] Downloading CorridorKey weights"
        " (~300 MB) from HuggingFace..."
    )
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        _HF_REPO,
        _HF_FILENAME,
        local_dir=cache_dir,
    )
    print("[TrentNodes] CorridorKey weights downloaded.")
    return downloaded


def load_corridorkey(
    device: torch.device,
) -> Optional[CorridorKeyEngine]:
    """
    Load CorridorKey engine, returning cached if available.

    Uses global cache: returns cached engine if device
    matches. Otherwise loads fresh from disk.

    Args:
        device: Target torch device

    Returns:
        CorridorKeyEngine instance, or None on failure
    """
    global _corridorkey_engine, _corridorkey_device

    if (
        _corridorkey_engine is not None
        and _corridorkey_device == device
    ):
        return _corridorkey_engine

    if not is_corridorkey_available():
        print(
            "[TrentNodes] CorridorKey requires timm."
            " Install: pip install timm"
        )
        return None

    try:
        weight_path = _get_weight_path()

        print(
            "[TrentNodes] Loading CorridorKey model..."
        )

        engine = CorridorKeyEngine(
            checkpoint_path=weight_path,
            device=str(device),
            img_size=2048,
            use_refiner=True,
            mixed_precision=True,
            model_precision=torch.float32,
        )

        _corridorkey_engine = engine
        _corridorkey_device = device

        print("[TrentNodes] CorridorKey loaded.")
        return engine

    except Exception as e:
        print(
            f"[TrentNodes] Failed to load"
            f" CorridorKey: {e}"
        )
        return None


def clear_corridorkey_cache() -> None:
    """Free CorridorKey model from VRAM and clear cache."""
    global _corridorkey_engine, _corridorkey_device

    _corridorkey_engine = None
    _corridorkey_device = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[TrentNodes] CorridorKey cache cleared.")


def run_corridorkey(
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    refiner_scale: float = 1.0,
    input_is_linear: bool = False,
    despill_strength: float = 1.0,
    auto_despeckle: bool = True,
    despeckle_size: int = 400,
) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Run CorridorKey green screen keying inference.

    Processes frames sequentially through the neural
    keyer. Each frame produces a clean foreground color
    and alpha matte via physical color unmixing.

    Args:
        images: (B, H, W, C) image batch in [0, 1]
        masks: (B, H, W) alpha hint masks in [0, 1]
        device: torch device
        refiner_scale: CNN refiner strength multiplier
        input_is_linear: True if input is linear color
        despill_strength: Green spill removal (0-2)
        auto_despeckle: Remove small alpha artifacts
        despeckle_size: Min pixel area to keep

    Returns:
        Tuple of (foreground, alpha, processed) tensors,
        or None if model unavailable.
        - foreground: (B, H, W, 3) straight FG in sRGB
        - alpha: (B, H, W) linear alpha matte
        - processed: (B, H, W, 4) premul linear RGBA
    """
    from comfy.utils import ProgressBar

    engine = load_corridorkey(device)
    if engine is None:
        return None

    b, h, w, c = images.shape

    # Allocate output lists
    fg_list = []
    alpha_list = []
    processed_list = []

    pbar = ProgressBar(b)

    for i in range(b):
        # Convert torch (H, W, C) -> numpy (H, W, 3)
        img_np = (
            images[i, :, :, :3]
            .cpu().float().numpy()
        )
        # Convert torch (H, W) -> numpy (H, W)
        mask_np = (
            masks[i].cpu().float().numpy()
        )

        # Run inference
        result = engine.process_frame(
            image=img_np,
            mask_linear=mask_np,
            refiner_scale=refiner_scale,
            input_is_linear=input_is_linear,
            fg_is_straight=True,
            despill_strength=despill_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
        )

        # Convert numpy results back to torch
        # fg: (H, W, 3) sRGB straight color
        fg_t = torch.from_numpy(
            result["fg"].copy()
        ).to(device=device, dtype=torch.float32)

        # alpha: (H, W, 1) -> (H, W)
        alpha_np = result["alpha"]
        if alpha_np.ndim == 3:
            alpha_np = alpha_np[:, :, 0]
        alpha_t = torch.from_numpy(
            alpha_np.copy()
        ).to(device=device, dtype=torch.float32)

        # processed: (H, W, 4) premul linear RGBA
        proc_t = torch.from_numpy(
            result["processed"].copy()
        ).to(device=device, dtype=torch.float32)

        fg_list.append(fg_t)
        alpha_list.append(alpha_t)
        processed_list.append(proc_t)

        pbar.update(1)

    # Stack into batch tensors
    foreground = torch.stack(fg_list, dim=0)  # (B,H,W,3)
    alpha = torch.stack(alpha_list, dim=0)  # (B, H, W)
    processed = torch.stack(
        processed_list, dim=0
    )  # (B, H, W, 4)

    return (foreground, alpha, processed)
