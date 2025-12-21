"""
Model loading and caching utilities for TrentNodes.

Provides cached model loading for commonly used models:
- SD 1.5 Inpainting model

Note: BiRefNet is now handled by birefnet_wrapper.py using
the official Hugging Face transformers implementation.
"""

import os

import torch

import folder_paths


# SD Inpainting model cache
_inpaint_model = None
_inpaint_model_name = None


def load_sd_inpaint_model(
    checkpoint: str = "sd-v1-5-inpainting.safetensors",
    device: torch.device = None
):
    """
    Load SD 1.5 inpainting model for clean plate generation.

    Model is cached for reuse across calls.

    Args:
        checkpoint: Checkpoint filename
        device: torch device (unused, ComfyUI handles device placement)

    Returns:
        Tuple of (model, clip, vae)
    """
    global _inpaint_model, _inpaint_model_name

    import comfy.sd

    # Return cached model if available
    if _inpaint_model is not None and _inpaint_model_name == checkpoint:
        return _inpaint_model

    # Find or download checkpoint
    ckpt_path = os.path.join(
        folder_paths.models_dir, "checkpoints", checkpoint
    )

    if not os.path.exists(ckpt_path):
        alt_path = folder_paths.get_full_path("checkpoints", checkpoint)
        if alt_path and os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            print("[TrentNodes] Downloading SD 1.5 inpainting (~4GB)...")
            print("[TrentNodes] This is a one-time download.")
            from torch.hub import download_url_to_file
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            url = (
                "https://huggingface.co/webui/stable-diffusion-inpainting/"
                "resolve/main/sd-v1-5-inpainting.safetensors"
            )
            download_url_to_file(url, ckpt_path)
            print("[TrentNodes] SD inpainting model downloaded.")

    print("[TrentNodes] Loading SD inpainting model...")
    model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    _inpaint_model = (model, clip, vae)
    _inpaint_model_name = checkpoint
    print("[TrentNodes] SD inpainting model loaded.")

    return model, clip, vae


def clear_model_cache():
    """Clear all cached models to free memory."""
    global _inpaint_model, _inpaint_model_name

    _inpaint_model = None
    _inpaint_model_name = None

    # Also clear BiRefNet cache
    from .birefnet_wrapper import clear_birefnet_cache
    clear_birefnet_cache()

    # Also clear MiniCPM cache
    try:
        from .minicpm_wrapper import clear_minicpm_cache
        clear_minicpm_cache()
    except ImportError:
        pass  # MiniCPM wrapper not available

    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[TrentNodes] Model cache cleared.")
