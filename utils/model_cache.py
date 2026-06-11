"""
Model loading and caching utilities for TrentNodes.

Provides cached model loading for commonly used models:
- big-lama (fast unprompted inpainting / object removal)
- VOID (Netflix video inpainting, HQ tier; loaded via ComfyUI core)

Note: BiRefNet is now handled by birefnet_wrapper.py using
the official Hugging Face transformers implementation.
"""

import os

import torch

import folder_paths


# big-lama: purpose-built removal model, TorchScript JIT, Apache-2.0.
# Hosted on GitHub releases (NOT Hugging Face - HF/Xet downloads can fail
# on this kind of WSL2 setup due to CDN DNS issues).
LAMA_URL = (
    "https://github.com/Sanster/models/releases/download/"
    "add_big_lama/big-lama.pt"
)
LAMA_SIZE = 205669692

_lama_model = None
_lama_device = None

# VOID model bundle cache: (model_pass1, model_pass2, clip, vae, flow)
_void_models = None


def _robust_download(url: str, dest: str, expected_size: int = None):
    """
    Download with resume support and size verification.

    Downloads to dest + '.part' and renames on success so an interrupted
    download never leaves a truncated file behind. On failure, prints
    manual instructions instead of leaving the caller guessing.
    """
    import urllib.error
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    resume_from = os.path.getsize(part) if os.path.exists(part) else 0

    request = urllib.request.Request(url)
    if resume_from > 0:
        request.add_header("Range", f"bytes={resume_from}-")

    try:
        mode = "ab" if resume_from > 0 else "wb"
        with urllib.request.urlopen(request, timeout=60) as resp:
            if resume_from > 0 and resp.status != 206:
                # Server ignored the Range header; restart from scratch
                resume_from = 0
                mode = "wb"
            with open(part, mode) as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"Download failed ({e}). Download manually:\n"
            f"  curl -L -o '{dest}' '{url}'\n"
            f"then re-run."
        ) from e

    size = os.path.getsize(part)
    if expected_size is not None and size != expected_size:
        raise RuntimeError(
            f"Downloaded size {size} != expected {expected_size} for {url}. "
            f"Delete '{part}' and retry, or download manually:\n"
            f"  curl -L -o '{dest}' '{url}'"
        )
    os.replace(part, dest)


def load_lama_model(device: torch.device):
    """
    Load the big-lama TorchScript inpainting model (cached).

    Looks for models/inpaint/big-lama.pt (also via folder_paths' 'inpaint'
    folder if registered by another node pack), downloading from GitHub
    releases if missing (~196MB, one time).
    """
    global _lama_model, _lama_device

    if _lama_model is not None and _lama_device == device:
        return _lama_model

    path = os.path.join(folder_paths.models_dir, "inpaint", "big-lama.pt")
    if not os.path.exists(path):
        try:
            alt = folder_paths.get_full_path("inpaint", "big-lama.pt")
            if alt and os.path.exists(alt):
                path = alt
        except Exception:
            pass
    if not os.path.exists(path):
        print(f"[TrentNodes] Downloading big-lama (~196MB) to {path} ...")
        _robust_download(LAMA_URL, path, LAMA_SIZE)
        print("[TrentNodes] big-lama downloaded.")

    print("[TrentNodes] Loading big-lama inpainting model...")
    _lama_model = torch.jit.load(path, map_location=device).eval()
    _lama_device = device
    return _lama_model


# VOID model filenames as shipped by Comfy-Org (the official ComfyUI
# template's documented locations). All resolved via folder_paths so the
# extra_model_paths.yaml roots work.
VOID_FILES = {
    "pass1": ("diffusion_models", "void_pass1.safetensors"),
    "pass2": ("diffusion_models", "void_pass2.safetensors"),
    "vae": ("vae", "cogvideox_vae.safetensors"),
    "t5": ("text_encoders", "t5xxl_fp16.safetensors"),
    "flow": ("optical_flow", "raft_large_C_T_SKHT_V2-ff5fadd5.safetensors"),
}
VOID_URLS = {
    "pass1": "https://huggingface.co/Comfy-Org/void-model/resolve/main/diffusion_models/void_pass1.safetensors",
    "pass2": "https://huggingface.co/Comfy-Org/void-model/resolve/main/diffusion_models/void_pass2.safetensors",
    "vae": "https://huggingface.co/Comfy-Org/void-model/resolve/main/vae/cogvideox_vae.safetensors",
    "t5": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
    "flow": "https://huggingface.co/Comfy-Org/void-model/resolve/main/optical_flow/raft_large_C_T_SKHT_V2-ff5fadd5.safetensors",
}


def load_void_models():
    """
    Load the VOID inpainting bundle through ComfyUI core (cached).

    Returns (model_pass1, model_pass2, clip, vae, optical_flow). Raises
    with manual download instructions if any file is missing - VOID
    weights are NOT auto-downloaded (22GB+, and HF downloads are
    unreliable on this WSL2 setup).
    """
    global _void_models

    if _void_models is not None:
        return _void_models

    import comfy.sd
    import comfy.utils
    from comfy.sd import CLIPType
    from comfy_extras.nodes_void import OpticalFlowLoader

    missing = []
    paths = {}
    for key, (folder, name) in VOID_FILES.items():
        p = folder_paths.get_full_path(folder, name)
        if p is None or not os.path.exists(p):
            missing.append(f"  models/{folder}/{name}\n    {VOID_URLS[key]}")
        else:
            paths[key] = p
    if missing:
        raise RuntimeError(
            "[TrentNodes] VOID models missing. Download manually into the "
            "ComfyUI models folders:\n" + "\n".join(missing)
        )

    print("[TrentNodes] Loading VOID models (one-time per session)...")
    clip = comfy.sd.load_clip(
        ckpt_paths=[paths["t5"]],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=CLIPType.COGVIDEOX,
    )
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(paths["vae"]))
    model1 = comfy.sd.load_diffusion_model(paths["pass1"])
    model2 = comfy.sd.load_diffusion_model(paths["pass2"])
    flow = OpticalFlowLoader.execute(
        os.path.basename(paths["flow"])
    ).args[0]
    print("[TrentNodes] VOID models loaded.")

    _void_models = (model1, model2, clip, vae, flow)
    return _void_models


def clear_model_cache():
    """Clear all cached models to free memory."""
    global _lama_model, _lama_device, _void_models

    _lama_model = None
    _lama_device = None
    _void_models = None

    # Also clear BiRefNet cache
    from .birefnet_wrapper import clear_birefnet_cache
    clear_birefnet_cache()

    # Also clear MiniCPM cache
    try:
        from .minicpm_wrapper import clear_minicpm_cache
        clear_minicpm_cache()
    except ImportError:
        pass  # MiniCPM wrapper not available

    # Also clear CorridorKey cache
    try:
        from .corridorkey_wrapper import (
            clear_corridorkey_cache,
        )
        clear_corridorkey_cache()
    except ImportError:
        pass  # CorridorKey wrapper not available

    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[TrentNodes] Model cache cleared.")
