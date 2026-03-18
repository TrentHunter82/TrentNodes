"""
MatAnyone wrapper for TrentNodes.

Provides temporally-consistent video matting using MatAnyone
(CVPR 2025). Handles model download, caching, VRAM management,
and frame-by-frame inference with progress reporting.

Model weights (~141 MB) are auto-downloaded from HuggingFace
on first use and cached in ComfyUI/models/matanyone/.
"""

import gc
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf

import folder_paths

from .matanyone.core.inference_core import InferenceCore
from .matanyone.model.matanyone import MatAnyone


# Model cache
_matanyone_model: Optional[MatAnyone] = None
_matanyone_device: Optional[torch.device] = None

# Config path (vendored base.yaml)
_CONFIG_PATH = Path(__file__).parent / "matanyone" / "base.yaml"

# HuggingFace model info
_HF_REPO = "PeiqingYang/MatAnyone"
_HF_FILENAME = "model.safetensors"


def is_matanyone_available() -> bool:
    """Check if MatAnyone dependencies are available."""
    try:
        from omegaconf import OmegaConf  # noqa: F811
        return True
    except ImportError:
        return False


def _get_weight_path() -> str:
    """
    Download MatAnyone weights if needed, return path.

    Downloads model.safetensors from HuggingFace Hub into
    ComfyUI/models/matanyone/ on first call. Subsequent
    calls return the cached path immediately.
    """
    cache_dir = os.path.join(
        folder_paths.models_dir, "matanyone"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Check for existing weight files
    safetensors_path = os.path.join(
        cache_dir, _HF_FILENAME
    )
    pth_path = os.path.join(cache_dir, "matanyone.pth")

    if os.path.exists(safetensors_path):
        return safetensors_path
    if os.path.exists(pth_path):
        return pth_path

    # Download from HuggingFace
    print(
        "[TrentNodes] Downloading MatAnyone weights"
        " from HuggingFace..."
    )
    from huggingface_hub import hf_hub_download
    weight_path = hf_hub_download(
        _HF_REPO,
        _HF_FILENAME,
        local_dir=cache_dir,
    )
    print("[TrentNodes] MatAnyone weights downloaded.")
    return weight_path


def _load_weights(path: str, device: torch.device) -> dict:
    """Load model weights from .safetensors or .pth file."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device=str(device))
    else:
        return torch.load(
            path, map_location=device, weights_only=True
        )


def load_matanyone(
    device: torch.device,
) -> Tuple[MatAnyone, object]:
    """
    Load MatAnyone model, returning (model, config).

    Uses global cache: returns cached model if device
    matches. Otherwise loads fresh from disk.

    Args:
        device: Target torch device

    Returns:
        Tuple of (MatAnyone model, OmegaConf config)
    """
    global _matanyone_model, _matanyone_device

    if (
        _matanyone_model is not None
        and _matanyone_device == device
    ):
        return _matanyone_model, _matanyone_model.cfg

    try:
        cfg = OmegaConf.load(str(_CONFIG_PATH))
        weight_path = _get_weight_path()

        print(
            "[TrentNodes] Loading MatAnyone model..."
        )

        model = MatAnyone(
            model_cfg=cfg, single_object=True
        ).to(device).eval()

        weights = _load_weights(weight_path, device)
        model.load_weights(weights)

        _matanyone_model = model
        _matanyone_device = device

        print("[TrentNodes] MatAnyone loaded.")
        return model, cfg

    except Exception as e:
        print(f"[TrentNodes] Failed to load MatAnyone: {e}")
        return None, None


def clear_matanyone_cache() -> None:
    """Free MatAnyone model from VRAM and clear cache."""
    global _matanyone_model, _matanyone_device

    _matanyone_model = None
    _matanyone_device = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[TrentNodes] MatAnyone cache cleared.")


def _warmup_and_propagate(
    processor: InferenceCore,
    vframes: torch.Tensor,
    mask_ma: torch.Tensor,
    mask_frame_index: int,
    frame_indices: List[int],
    n_warmup: int,
    pbar,
) -> List[Optional[torch.Tensor]]:
    """
    Warmup on reference frame, then propagate through
    the given frame indices sequentially.

    Args:
        processor: Fresh InferenceCore instance
        vframes: (B, C, H, W) video frames
        mask_ma: (H, W) mask in [0, 255]
        mask_frame_index: Reference frame index
        frame_indices: Ordered list of frame indices
            to propagate through (excluding ref frame)
        n_warmup: Warmup iterations
        pbar: ComfyUI progress bar

    Returns:
        List of alpha tensors indexed by frame number,
        None for frames not in this pass
    """
    b = vframes.shape[0]
    alphas: List[Optional[torch.Tensor]] = [None] * b

    ref_frame = vframes[mask_frame_index]

    # Warmup phase on reference frame
    for ti in range(n_warmup):
        if ti == 0:
            output_prob = processor.step(
                ref_frame, mask_ma, objects=[1]
            )
            output_prob = processor.step(
                ref_frame, first_frame_pred=True
            )
        else:
            output_prob = processor.step(
                ref_frame, first_frame_pred=True
            )
        pbar.update(1)

    # Store reference frame result
    alphas[mask_frame_index] = (
        processor.output_prob_to_mask(output_prob)
        .unsqueeze(0)
    )
    pbar.update(1)

    # Propagate through ordered frame indices
    for ti in frame_indices:
        image = vframes[ti]
        output_prob = processor.step(image)
        alphas[ti] = (
            processor.output_prob_to_mask(output_prob)
            .unsqueeze(0)
        )
        pbar.update(1)

    return alphas


def run_matanyone(
    images: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    n_warmup: int = 10,
    mask_frame_index: int = 0,
    bidirectional: bool = True,
) -> Optional[torch.Tensor]:
    """
    Run MatAnyone video matting inference.

    Processes video frames with temporal memory
    propagation. When bidirectional=True and the
    reference frame is not frame 0, runs a separate
    backward pass for frames before the reference
    to maintain temporal consistency in both
    directions.

    Args:
        images: (B, H, W, C) image batch in [0, 1]
        mask: (H, W) single-frame mask in [0, 1]
        device: torch device
        n_warmup: Warmup iterations on reference frame
        mask_frame_index: Which frame the mask applies to
        bidirectional: If True, propagate both forward
            and backward from the reference frame

    Returns:
        (B, H, W) alpha matte tensor in [0, 1],
        or None if model unavailable
    """
    from comfy.utils import ProgressBar

    model, cfg = load_matanyone(device)
    if model is None:
        return None

    b, h, w, c = images.shape

    # (B, H, W, C) [0,1] -> (B, C, H, W) [0,1]
    vframes = images[..., :3].permute(0, 3, 1, 2).to(
        device=device, dtype=torch.float32
    )

    # Mask: (H, W) [0,1] -> (H, W) [0,255]
    mask_ma = (mask.to(
        device=device, dtype=torch.float32
    ) * 255.0)
    if mask_ma.dim() == 3:
        mask_ma = mask_ma[0]

    mask_frame_index = min(mask_frame_index, b - 1)

    need_backward = (
        bidirectional and mask_frame_index > 0
    )

    # Calculate total steps for progress bar
    fwd_frames = b - mask_frame_index - 1
    fwd_steps = n_warmup + 1 + fwd_frames
    if need_backward:
        bwd_frames = mask_frame_index
        bwd_steps = n_warmup + 1 + bwd_frames
    else:
        bwd_steps = 0
    pbar = ProgressBar(fwd_steps + bwd_steps)

    # Forward pass: ref frame -> end
    fwd_processor = InferenceCore(
        model, cfg=model.cfg
    )
    fwd_indices = list(
        range(mask_frame_index + 1, b)
    )
    fwd_alphas = _warmup_and_propagate(
        fwd_processor, vframes, mask_ma,
        mask_frame_index, fwd_indices,
        n_warmup, pbar,
    )
    del fwd_processor

    if need_backward:
        # Backward pass: ref frame -> start
        bwd_processor = InferenceCore(
            model, cfg=model.cfg
        )
        bwd_indices = list(
            range(mask_frame_index - 1, -1, -1)
        )
        bwd_alphas = _warmup_and_propagate(
            bwd_processor, vframes, mask_ma,
            mask_frame_index, bwd_indices,
            n_warmup, pbar,
        )
        del bwd_processor

        # Merge: backward fills pre-ref frames
        for ti in range(mask_frame_index):
            fwd_alphas[ti] = bwd_alphas[ti]
    else:
        # No backward: fill pre-ref with ref result
        if mask_frame_index > 0:
            fill = fwd_alphas[mask_frame_index]
            for ti in range(mask_frame_index):
                if fwd_alphas[ti] is None:
                    fwd_alphas[ti] = fill

    result = torch.cat(fwd_alphas, dim=0)

    if result.shape[1] != h or result.shape[2] != w:
        import torch.nn.functional as F
        result = F.interpolate(
            result.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

    return result.clamp(0.0, 1.0)
