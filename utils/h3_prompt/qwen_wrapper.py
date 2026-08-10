"""
Local Qwen3-VL wrapper for the H3 Auto Prompt Generator.

Same lifecycle pattern as utils/minicpm_wrapper.py: thread-safe module
cache, 60s idle auto-unload daemon, lazy transformers import. Supports
any Qwen VL checkpoint loadable via AutoModelForImageTextToText
(transformers >= 5.x ships qwen3_vl and qwen2_5_vl).
"""

import threading
import time
from typing import List, Tuple

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
UNLOAD_TIMEOUT_S = 60

_qwen_model = None
_qwen_processor = None
_qwen_model_id = None
_qwen_lock = threading.Lock()

_unload_lock = threading.Lock()
_last_use_time = 0.0
_unload_thread = None


def _auto_unload_worker():
    global _qwen_model
    while True:
        time.sleep(10)
        with _unload_lock:
            if _qwen_model is None:
                break
            elapsed = time.time() - _last_use_time
            if elapsed >= UNLOAD_TIMEOUT_S:
                print(
                    f"[TrentNodes] Qwen-VL idle for {elapsed:.0f}s, unloading..."
                )
                clear_qwen_cache()
                break


def _touch_model():
    global _last_use_time, _unload_thread
    with _unload_lock:
        _last_use_time = time.time()
        if _unload_thread is None or not _unload_thread.is_alive():
            _unload_thread = threading.Thread(
                target=_auto_unload_worker, daemon=True, name="qwenvl-unload"
            )
            _unload_thread.start()


def clear_qwen_cache():
    """Unload the cached model and free VRAM."""
    global _qwen_model, _qwen_processor, _qwen_model_id
    with _qwen_lock:
        if _qwen_model is not None:
            try:
                _qwen_model.to("cpu")
            except Exception:
                pass
            del _qwen_model
            _qwen_model = None
        _qwen_processor = None
        _qwen_model_id = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_qwen_model(model_id: str = DEFAULT_MODEL_ID):
    """Load (or return cached) model + processor for `model_id`."""
    global _qwen_model, _qwen_processor, _qwen_model_id

    with _qwen_lock:
        if _qwen_model is not None and _qwen_model_id == model_id:
            _touch_model()
            return _qwen_model, _qwen_processor

    # Different checkpoint requested: drop the old one first
    if _qwen_model_id is not None and _qwen_model_id != model_id:
        clear_qwen_cache()

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Qwen-VL local backend needs transformers. Install with: "
            "pip install 'transformers>=5.0' accelerate"
        ) from exc

    with _qwen_lock:
        if _qwen_model is not None and _qwen_model_id == model_id:
            _touch_model()
            return _qwen_model, _qwen_processor

        print(f"[TrentNodes] Loading Qwen-VL model: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        globals()["_qwen_model"] = model
        globals()["_qwen_processor"] = processor
        globals()["_qwen_model_id"] = model_id
        _touch_model()
        return model, processor


def run_qwen_inference(
    labeled_images: List[Tuple[str, "object"]],
    system_prompt: str,
    user_text: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 4096,
) -> str:
    """
    Run one chat turn: labeled PIL images interleaved with their text
    labels, followed by the task text. Greedy decoding (deterministic).

    Args:
        labeled_images: list of (label, PIL.Image) in send order
    """
    import torch

    model, processor = load_qwen_model(model_id)
    _touch_model()

    content = []
    for label, pil in labeled_images:
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": pil})
    content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    _touch_model()
    return text.strip()
