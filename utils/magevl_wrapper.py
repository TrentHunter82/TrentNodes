"""
Mage-VL wrapper for TrentNodes.

Microsoft Mage-VL 4B (Jul 2026): compact vision-language model with
strong video understanding (beats Phi-4-reasoning-vision 15B on video
benchmarks at 4.7B params). Apache-2.0, transformers >= 5.7.

Note: Mage-VL's headline feature is codec-native input (reading the
compressed stream directly). Inside ComfyUI we only have decoded frame
batches, so this wrapper uses the frames path — same quality, without
the codec-path token savings.

Mirrors the minicpm_wrapper API: cached load, idle auto-unload,
run_magevl_inference(), clear/complete cleanup.
"""

import contextlib
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
_magevl_model = None
_magevl_processor = None
_magevl_lock = threading.Lock()

# Auto-unload state
_last_use_time = 0.0
_unload_timeout = 60.0  # seconds
_unload_thread = None
_unload_lock = threading.Lock()

MODEL_ID = "microsoft/Mage-VL"
CACHE_DIR_NAME = "magevl"

# Remembered across calls so the fallback probe runs only once per session
_video_content_supported = None


def is_magevl_available() -> bool:
    """Check if Mage-VL dependencies are available (transformers >= 5.7)."""
    try:
        import transformers
        import accelerate  # noqa: F401
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        return (major, minor) >= (5, 7)
    except (ImportError, ValueError):
        return False


def _get_cache_dir() -> str:
    """Get the cache directory for Mage-VL model files."""
    cache_dir = os.path.join(folder_paths.models_dir, CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _auto_unload_worker():
    """Background thread that unloads model after idle timeout."""
    while True:
        time.sleep(10)
        with _unload_lock:
            if _magevl_model is None:
                break
            elapsed = time.time() - _last_use_time
            if elapsed >= _unload_timeout:
                print(f"[TrentNodes] Mage-VL idle for {elapsed:.0f}s, unloading...")
                clear_magevl_cache()
                break


def _touch_model():
    """Update last use time and start unload timer if needed."""
    global _last_use_time, _unload_thread

    with _unload_lock:
        _last_use_time = time.time()
        if _unload_thread is None or not _unload_thread.is_alive():
            _unload_thread = threading.Thread(
                target=_auto_unload_worker,
                daemon=True,
                name="magevl-unload"
            )
            _unload_thread.start()


def _prepare_vram():
    """Free all ComfyUI-managed models before loading Mage-VL."""
    try:
        device = comfy.model_management.get_torch_device()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info(device)[0] / (1024**3)
            print(f"[TrentNodes] VRAM cleared: {free_vram:.1f}GB free")
    except Exception as e:
        print(f"[TrentNodes] VRAM prep warning: {e}")


@contextlib.contextmanager
def _allow_missing_mamba():
    """
    Let Mage-VL load without mamba_ssm.

    mamba_ssm is only imported by streammind_gate.py, which the model
    loads lazily and only for streaming mode (never used here). But
    transformers' static check_imports scans the whole remote-code repo
    and hard-fails on it. Temporarily filter that one package; anything
    else missing still raises.
    """
    import transformers.dynamic_module_utils as dmu
    orig = dmu.check_imports

    def patched(filename):
        try:
            return orig(filename)
        except ImportError as e:
            if "mamba_ssm" in str(e) and "," not in str(e):
                return dmu.get_relative_imports(filename)
            raise

    dmu.check_imports = patched
    try:
        yield
    finally:
        dmu.check_imports = orig


def load_magevl_model(
    device: Optional[torch.device] = None
) -> Tuple[any, any]:
    """
    Load Mage-VL 4B (bf16). Cached for reuse; auto-downloads (~10GB) on
    first use into models/magevl.

    Returns:
        Tuple of (model, processor) or (None, None) if not available
    """
    global _magevl_model, _magevl_processor

    with _magevl_lock:
        if _magevl_model is not None and _magevl_processor is not None:
            _touch_model()
            return _magevl_model, _magevl_processor

        _prepare_vram()

        if not is_magevl_available():
            print("[TrentNodes] Mage-VL needs transformers >= 5.7 and accelerate.")
            return None, None

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            cache_dir = _get_cache_dir()
            print("[TrentNodes] Loading Mage-VL 4B...")
            print("[TrentNodes] First run will download ~10GB model.")

            # Some AutoProcessor internals re-resolve trust_remote_code
            # and would prompt on a headless server; the env var answers.
            os.environ.setdefault("TRUST_REMOTE_CODE", "1")

            with _allow_missing_mamba():
                processor = AutoProcessor.from_pretrained(
                    MODEL_ID,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )

                load_kwargs = {
                    "trust_remote_code": True,
                    "cache_dir": cache_dir,
                    "device_map": "auto",
                    "torch_dtype": "auto",
                }
                try:
                    load_kwargs["attn_implementation"] = "sdpa"
                    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs)
                except Exception as e:
                    if "sdpa" in str(e).lower():
                        print("[TrentNodes] SDPA not supported, using eager attn")
                        load_kwargs["attn_implementation"] = "eager"
                        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs)
                    else:
                        raise

            model = model.eval()

            _magevl_model = model
            _magevl_processor = processor
            _touch_model()

            print("[TrentNodes] Mage-VL 4B loaded successfully.")
            return model, processor

        except Exception as e:
            print(f"[TrentNodes] Failed to load Mage-VL: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def clear_magevl_cache():
    """Clear Mage-VL model completely from GPU and CPU memory."""
    global _magevl_model, _magevl_processor

    with _magevl_lock:
        if _magevl_model is not None:
            try:
                _magevl_model.to("cpu")
            except Exception:
                pass
            del _magevl_model
            _magevl_model = None

        if _magevl_processor is not None:
            del _magevl_processor
            _magevl_processor = None

    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("[TrentNodes] Mage-VL cache cleared.")


def complete_magevl_inference():
    """Aggressively unload Mage-VL and clear ComfyUI's cache too."""
    clear_magevl_cache()
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    print("[TrentNodes] Mage-VL fully unloaded from GPU and CPU memory.")
    return "VRAM_CLEARED"


def _build_user_content(images: List[Image.Image], prompt: str, mode: str,
                        as_video: bool):
    """Message content for one user turn, video or multi-image form."""
    if mode == "single_image" or len(images) == 1:
        return [{"type": "image", "image": images[0]},
                {"type": "text", "text": prompt}]
    if mode == "video_frames" and as_video:
        return [{"type": "video", "video": images},
                {"type": "text", "text": prompt}]
    # multi_image, or video fallback: frames as individual images
    return ([{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": prompt}])


def run_magevl_inference(
    images: List[Image.Image],
    prompt: str,
    mode: str = "video_frames",
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    seed: int = 0
) -> str:
    """
    Run Mage-VL inference on images.

    Args:
        images: List of PIL Images
        prompt: Text prompt/question
        mode: "single_image", "multi_image", or "video_frames"
        system_prompt: System prompt to set model behavior
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        seed: Random seed for reproducibility

    Returns:
        Generated text response
    """
    global _video_content_supported

    model, processor = load_magevl_model()
    if model is None or processor is None:
        return "[Error] Mage-VL model not available"

    _touch_model()

    if seed > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _generate(as_video: bool) -> str:
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        messages.append({
            "role": "user",
            "content": _build_user_content(images, prompt, mode, as_video)
        })

        # Mage-VL's apply_chat_template only renders text (placeholders);
        # pixels must go through processor.__call__ separately.
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        proc_kwargs = {"text": prompt_text, "return_tensors": "pt"}
        if mode == "single_image" or len(images) == 1:
            proc_kwargs["images"] = [images[0]]
        elif as_video:
            proc_kwargs["videos"] = [images]  # one video = list of frames
        else:
            proc_kwargs["images"] = list(images)

        inputs = processor(**proc_kwargs).to(model.device)

        gen_kwargs = {"max_new_tokens": max_tokens}
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        return processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()

    try:
        # Prefer the native video content type; probe once, remember result.
        if mode == "video_frames" and len(images) > 1 and _video_content_supported is not False:
            try:
                response = _generate(as_video=True)
                _video_content_supported = True
                return response
            except Exception as e:
                if _video_content_supported is None:
                    print(f"[TrentNodes] Mage-VL video content type failed "
                          f"({e}); falling back to per-frame images.")
                    _video_content_supported = False
                else:
                    raise
        return _generate(as_video=False)

    except Exception as e:
        import traceback
        print(f"[TrentNodes] Mage-VL inference error: {e}")
        traceback.print_exc()
        return f"[Error] Inference failed: {e}"
