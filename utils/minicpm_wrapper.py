"""
MiniCPM-V 4.5 wrapper for TrentNodes.

Provides GPU-accelerated vision-language inference with:
- int4 quantization for reduced VRAM (~6-8GB)
- Auto-unload after idle timeout
- Smart frame sampling for video
- Thread-safe model caching
"""

import gc
import os
import threading
import time
from typing import List, Optional, Tuple

import torch
from PIL import Image

import folder_paths
import comfy.model_management


# Model cache
_minicpm_model = None
_minicpm_tokenizer = None
_minicpm_lock = threading.Lock()

# Auto-unload state
_last_use_time = 0.0
_unload_timeout = 60.0  # seconds
_unload_thread = None
_unload_lock = threading.Lock()

# Model configuration
MODEL_ID = "openbmb/MiniCPM-V-4_5"
MODEL_ID_INT4 = "openbmb/MiniCPM-V-4_5-int4"
CACHE_DIR_NAME = "minicpm"

# System prompt presets
SYSTEM_PROMPTS = {
    "default": (
        "You are a helpful vision assistant. Describe what you see accurately "
        "and thoroughly. Focus on the main subjects, actions, and important "
        "details. Be objective and descriptive."
    ),
    "detailed": (
        "You are an expert visual analyst. Provide comprehensive, detailed "
        "descriptions of everything you observe. Include: subjects and their "
        "appearance, actions and movements, setting and environment, lighting "
        "and atmosphere, composition and framing, colors and textures, any "
        "text or symbols visible, and temporal changes if viewing video."
    ),
    "concise": (
        "You are a concise visual assistant. Provide brief, to-the-point "
        "descriptions. Focus only on the most important elements. Use short "
        "sentences. Avoid unnecessary detail."
    ),
    "narrator": (
        "You are a cinematic narrator describing scenes for an audience. "
        "Use vivid, engaging language that brings the visuals to life. "
        "Describe the mood, atmosphere, and story unfolding. Write as if "
        "narrating a documentary or film."
    ),
    "technical": (
        "You are a technical visual analyst. Focus on objective, measurable "
        "aspects: resolution quality, lighting conditions, camera angles, "
        "motion blur, color grading, composition rules, and technical "
        "attributes. Use precise terminology."
    ),
    "accessible": (
        "You are an accessibility assistant providing audio descriptions for "
        "visually impaired users. Describe visual content clearly and "
        "comprehensively so someone who cannot see can understand what is "
        "happening. Prioritize: who/what is present, what they are doing, "
        "the setting, and any important visual information."
    ),
    "creative": (
        "You are a creative writer inspired by visuals. Respond with "
        "imaginative, artistic interpretations. You may write poetry, "
        "short prose, or evocative descriptions that capture the emotional "
        "essence of what you see."
    ),
    "none": "",
    "custom": "",  # Placeholder, actual value comes from custom_system_prompt
}

SYSTEM_PROMPT_CHOICES = [
    "default",
    "detailed",
    "concise",
    "narrator",
    "technical",
    "accessible",
    "creative",
    "none",
    "custom",
]


def is_minicpm_available() -> bool:
    """
    Check if MiniCPM-V dependencies are available.

    Returns:
        True if transformers and accelerate are installed
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        import accelerate
        return True
    except ImportError:
        return False


def _get_cache_dir() -> str:
    """Get the cache directory for MiniCPM model files."""
    cache_dir = os.path.join(folder_paths.models_dir, CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _auto_unload_worker():
    """Background thread that unloads model after idle timeout."""
    global _minicpm_model, _minicpm_tokenizer

    while True:
        time.sleep(10)  # Check every 10 seconds

        with _unload_lock:
            if _minicpm_model is None:
                # Model already unloaded
                break

            elapsed = time.time() - _last_use_time
            if elapsed >= _unload_timeout:
                print(
                    f"[TrentNodes] MiniCPM idle for {elapsed:.0f}s, unloading..."
                )
                clear_minicpm_cache()
                break


def _touch_model():
    """Update last use time and start unload timer if needed."""
    global _last_use_time, _unload_thread

    with _unload_lock:
        _last_use_time = time.time()

        # Start unload thread if not running
        if _unload_thread is None or not _unload_thread.is_alive():
            _unload_thread = threading.Thread(
                target=_auto_unload_worker,
                daemon=True,
                name="minicpm-unload"
            )
            _unload_thread.start()


def load_minicpm_model(
    device: Optional[torch.device] = None
) -> Tuple[any, any]:
    """
    Load MiniCPM-V 4.5 model (pre-quantized int4 version).

    Model is cached for reuse. Auto-downloads on first use.

    Args:
        device: torch device (uses auto device mapping)

    Returns:
        Tuple of (model, tokenizer) or (None, None) if not available
    """
    global _minicpm_model, _minicpm_tokenizer

    with _minicpm_lock:
        # Return cached model if available
        if _minicpm_model is not None and _minicpm_tokenizer is not None:
            _touch_model()
            return _minicpm_model, _minicpm_tokenizer

        # Free ComfyUI models before loading MiniCPM
        prepare_vram_for_minicpm()

        if not is_minicpm_available():
            print("[TrentNodes] MiniCPM dependencies not available.")
            print("[TrentNodes] Install: pip install transformers accelerate")
            return None, None

        try:
            from transformers import AutoModel, AutoTokenizer

            cache_dir = _get_cache_dir()

            # Use pre-quantized int4 model (smaller, faster, no dtype issues)
            model_id = MODEL_ID_INT4
            print("[TrentNodes] Loading MiniCPM-V 4.5-int4...")
            print("[TrentNodes] First run will download ~5GB model.")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            # Load pre-quantized int4 model
            load_kwargs = {
                "trust_remote_code": True,
                "cache_dir": cache_dir,
                "device_map": "auto",
            }

            # Try SDPA first, fall back to eager if not supported
            try:
                load_kwargs["attn_implementation"] = "sdpa"
                model = AutoModel.from_pretrained(model_id, **load_kwargs)
            except Exception as e:
                if "sdpa" in str(e).lower():
                    print("[TrentNodes] SDPA not supported, using eager attn")
                    load_kwargs["attn_implementation"] = "eager"
                    model = AutoModel.from_pretrained(model_id, **load_kwargs)
                else:
                    raise

            model = model.eval()

            _minicpm_model = model
            _minicpm_tokenizer = tokenizer
            _touch_model()

            print("[TrentNodes] MiniCPM-V 4.5-int4 loaded successfully.")
            return model, tokenizer

        except Exception as e:
            print(f"[TrentNodes] Failed to load MiniCPM: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def clear_minicpm_cache():
    """Clear MiniCPM model completely from GPU and CPU memory."""
    global _minicpm_model, _minicpm_tokenizer

    with _minicpm_lock:
        if _minicpm_model is not None:
            try:
                _minicpm_model.to("cpu")
            except Exception:
                pass
            del _minicpm_model
            _minicpm_model = None

        if _minicpm_tokenizer is not None:
            del _minicpm_tokenizer
            _minicpm_tokenizer = None

    # Multiple GC passes for thorough cleanup
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("[TrentNodes] MiniCPM cache cleared.")


def prepare_vram_for_minicpm():
    """Free all ComfyUI-managed models before loading MiniCPM."""
    try:
        device = comfy.model_management.get_torch_device()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info(device)[0] / (1024**3)
            print(f"[TrentNodes] VRAM cleared: {free_vram:.1f}GB free")
    except Exception as e:
        print(f"[TrentNodes] VRAM prep warning: {e}")


def complete_minicpm_inference():
    """
    Aggressively unload MiniCPM from GPU and CPU memory.

    Performs multiple cleanup passes to ensure complete memory release.
    """
    global _minicpm_model, _minicpm_tokenizer

    with _minicpm_lock:
        if _minicpm_model is not None:
            try:
                # Move model to CPU first (helps CUDA release memory)
                _minicpm_model.to("cpu")
            except Exception:
                pass

            # Clear any internal caches
            if hasattr(_minicpm_model, "clear_cache"):
                try:
                    _minicpm_model.clear_cache()
                except Exception:
                    pass

            del _minicpm_model
            _minicpm_model = None

        if _minicpm_tokenizer is not None:
            del _minicpm_tokenizer
            _minicpm_tokenizer = None

    # Aggressive garbage collection (multiple passes)
    for _ in range(3):
        gc.collect()

    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Clear ComfyUI's cache too
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass

    print("[TrentNodes] MiniCPM fully unloaded from GPU and CPU memory.")
    return "VRAM_CLEARED"


def sample_frames(
    images: torch.Tensor,
    use_all_frames: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Smart frame sampling to optimize quality vs VRAM.

    Sampling strategy:
    - 1-32 frames: use all
    - 33-64 frames: every 2nd
    - 65-128 frames: every 4th
    - 129-256 frames: every 8th
    - 257+ frames: dynamic to get ~32

    Args:
        images: (B, H, W, C) tensor of frames
        use_all_frames: If True, skip sampling and use all frames

    Returns:
        Tuple of (sampled_images, original_indices)
    """
    n_frames = images.shape[0]

    if use_all_frames or n_frames <= 32:
        return images, list(range(n_frames))

    # Determine stride based on frame count
    if n_frames <= 64:
        stride = 2
    elif n_frames <= 128:
        stride = 4
    elif n_frames <= 256:
        stride = 8
    else:
        # Dynamic stride to get approximately 32 frames
        stride = max(1, n_frames // 32)

    indices = list(range(0, n_frames, stride))[:32]
    sampled = images[indices]

    return sampled, indices


def tensors_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert ComfyUI image tensors to PIL Images.

    Args:
        images: (B, H, W, C) tensor in [0, 1] range, RGB format

    Returns:
        List of PIL Images
    """
    # Ensure on CPU and in correct format
    if images.device.type != "cpu":
        images = images.cpu()

    # Clamp and convert to uint8
    images = (images.clamp(0, 1) * 255).to(torch.uint8)

    pil_images = []
    for i in range(images.shape[0]):
        # Convert tensor to numpy and create PIL Image
        img_np = images[i].numpy()
        pil_img = Image.fromarray(img_np, mode="RGB")
        pil_images.append(pil_img)

    return pil_images


def run_inference(
    images: List[Image.Image],
    prompt: str,
    mode: str = "video_frames",
    system_prompt: str = "",
    thinking_mode: str = "fast",
    max_tokens: int = 512,
    temperature: float = 0.7,
    seed: int = 0
) -> str:
    """
    Run MiniCPM-V inference on images.

    Args:
        images: List of PIL Images
        prompt: Text prompt/question
        mode: "single_image", "multi_image", or "video_frames"
        system_prompt: System prompt to set model behavior
        thinking_mode: "fast" or "deep_thinking"
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        seed: Random seed for reproducibility

    Returns:
        Generated text response
    """
    model, tokenizer = load_minicpm_model()
    if model is None or tokenizer is None:
        return "[Error] MiniCPM model not available"

    _touch_model()

    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Combine system prompt with user prompt if provided
    # MiniCPM-V doesn't support system role, so prepend to user message
    if system_prompt and system_prompt.strip():
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt

    # Build message content based on mode
    if mode == "single_image":
        # Use only first image
        content = [images[0], full_prompt]
    elif mode == "multi_image":
        # All images as separate images for comparison
        content = images + [full_prompt]
    else:  # video_frames
        # All images as video frames
        content = images + [full_prompt]

    # Build messages list (only user role supported)
    msgs = [{"role": "user", "content": content}]

    # Determine thinking mode
    enable_thinking = thinking_mode == "deep_thinking"

    try:
        # Run inference
        # For video mode, we need temporal IDs
        if mode == "video_frames" and len(images) > 1:
            # Create temporal ID groups (each frame is its own group)
            # This tells the model the temporal order
            temporal_ids = [[i] for i in range(len(images))]

            response = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                use_image_id=False,
                max_slice_nums=1,  # Reduce slicing for video frames
                temporal_ids=temporal_ids
            )
        else:
            response = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0
            )

        return response

    except Exception as e:
        import traceback
        print(f"[TrentNodes] Inference error: {e}")
        traceback.print_exc()
        return f"[Error] Inference failed: {e}"
