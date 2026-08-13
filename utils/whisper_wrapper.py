"""
Whisper transcription wrapper for TrentNodes.

Loads openai/whisper-* through transformers and runs speech/lyric
transcription with correct dtype handling.

Why this exists: transformers 5.x changed `from_pretrained` to default to
the *checkpoint* dtype instead of float32. whisper-large-v3 ships as fp16,
so a model loaded with no explicit dtype is fp16 while the feature
extractor still emits fp32 mel features. The first conv1d then dies with
"Input type (float) and bias type (c10::Half) should be the same".
The fix is to always cast the features to `model.dtype` before generate().
"""

import gc
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30.0

# Hugging Face repo ids offered in the node dropdown.
HF_MODEL_CHOICES = [
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
]

LANGUAGE_CHOICES = [
    "auto", "english", "chinese", "german", "spanish", "russian", "korean",
    "french", "japanese", "portuguese", "turkish", "polish", "catalan",
    "dutch", "arabic", "swedish", "italian", "indonesian", "hindi", "finnish",
    "vietnamese", "hebrew", "ukrainian", "greek", "malay", "czech",
    "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu",
    "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam",
    "welsh", "slovak", "telugu", "persian", "latvian", "bengali", "serbian",
    "azerbaijani", "slovenian", "kannada", "estonian", "macedonian",
    "breton", "basque", "icelandic", "armenian", "nepali", "mongolian",
    "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi",
    "punjabi", "sinhala", "khmer", "shona", "yoruba", "somali", "afrikaans",
    "occitan", "georgian", "belarusian", "tajik", "sindhi", "gujarati",
    "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole",
    "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish",
    "myanmar", "tibetan", "tagalog", "malagasy", "assamese", "tatar",
    "hawaiian", "lingala", "hausa", "bashkir", "javanese", "sundanese",
    "cantonese",
]

PRECISION_CHOICES = ["auto", "fp16", "bf16", "fp32"]

# Cache: (model_id, dtype, device) -> (processor, model)
_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


def _import_whisper():
    """Import the transformers Whisper classes, or explain why we cannot."""
    try:
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        return WhisperProcessor, WhisperForConditionalGeneration
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Whisper transcription needs the transformers Whisper classes, "
            f"which failed to import ({type(e).__name__}: {e}). "
            "Install/repair transformers in the ComfyUI venv."
        ) from e


def local_whisper_dir() -> Optional[str]:
    """Return the ComfyUI models/whisper directory, or None outside ComfyUI."""
    try:
        import folder_paths
    except Exception:  # noqa: BLE001
        return None
    return os.path.join(folder_paths.models_dir, "whisper")


def list_local_whisper_models() -> List[str]:
    """
    List local Whisper folders under ComfyUI/models/whisper.

    A folder counts only when it holds a config.json, so half-finished
    downloads never reach the dropdown.
    """
    root = local_whisper_dir()
    if not root or not os.path.isdir(root):
        return []

    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "config.json")):
            found.append(f"local:{name}")
    return found


def model_choices() -> List[str]:
    """Dropdown entries: local folders first, then the Hugging Face ids."""
    return list_local_whisper_models() + HF_MODEL_CHOICES


def _resolve_source(model_choice: str) -> str:
    """Turn a dropdown entry into a path or repo id."""
    if model_choice.startswith("local:"):
        root = local_whisper_dir()
        if not root:
            raise RuntimeError(
                "Local Whisper models need ComfyUI's folder_paths module."
            )
        return os.path.join(root, model_choice[len("local:"):])
    return model_choice


def resolve_dtype(precision: str, device: torch.device) -> torch.dtype:
    """
    Pick the compute dtype.

    CPU conv1d has no fp16 kernel, so the CPU always gets fp32.
    """
    if device.type != "cuda":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    return torch.float16  # auto


def load_whisper(
    model_choice: str,
    precision: str = "auto",
    device: Optional[torch.device] = None,
):
    """
    Load (and cache) a Whisper processor + model pair.

    Returns:
        (processor, model) with the model on `device` in the chosen dtype.
    """
    WhisperProcessor, WhisperForConditionalGeneration = _import_whisper()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = resolve_dtype(precision, device)
    source = _resolve_source(model_choice)
    key = (source, str(dtype), str(device))

    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    logger.info(
        "[TrentNodes] Loading Whisper %s (%s, %s)", source, dtype, device
    )
    processor = WhisperProcessor.from_pretrained(source)
    model = WhisperForConditionalGeneration.from_pretrained(
        source,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device).eval()

    _CACHE[key] = (processor, model)
    return processor, model


def unload_whisper():
    """Drop every cached Whisper model and free the VRAM."""
    _CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_waveform(audio: Dict[str, Any]) -> torch.Tensor:
    """
    Turn a ComfyUI AUDIO dict into a mono 16 kHz float32 1-D tensor.

    Args:
        audio: {"waveform": (B, C, S) tensor, "sample_rate": int}
    """
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("Expected a ComfyUI AUDIO dict with a 'waveform'.")

    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", SAMPLE_RATE))

    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)

    waveform = waveform.detach().cpu().float()

    # (B, C, S) -> (C, S): keep the first item of the batch.
    while waveform.dim() > 2:
        waveform = waveform[0]

    # (C, S) -> (S,): average the channels down to mono.
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    waveform = waveform.reshape(-1)

    if waveform.numel() == 0:
        raise ValueError("The audio input is empty.")

    if sample_rate != SAMPLE_RATE:
        from .audio_utils import resample_audio
        waveform = resample_audio(
            waveform, sample_rate, SAMPLE_RATE, use_gpu=False
        )

    return waveform.contiguous()


def _decode_segments(processor, segments) -> List[Dict[str, Any]]:
    """Decode transformers' long-form segment dicts into plain records."""
    out = []
    for seg in segments:
        tokens = seg["tokens"]
        text = processor.batch_decode(
            tokens.unsqueeze(0) if tokens.dim() == 1 else tokens,
            skip_special_tokens=True,
        )[0].strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": text,
        })
    return out


def _segments_from_timestamp_tokens(
    processor, sequence, offset: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Rebuild segments from a sequence that still carries timestamp tokens.

    Used for the short-form path, where transformers returns one flat
    sequence instead of segment dicts.
    """
    import re

    text = processor.tokenizer.decode(
        sequence, decode_with_timestamps=True, skip_special_tokens=False
    )
    # Drop the task/language/notimestamps preamble, keep <|1.23|> markers.
    text = re.sub(r"<\|(?!\d+\.\d+\|)[^|]*\|>", "", text)

    parts = re.split(r"<\|(\d+\.\d+)\|>", text)
    # parts = [pre, time, chunk, time, chunk, ...]
    segments = []
    times = parts[1::2]
    chunks = parts[2::2]
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        start = float(times[i]) + offset
        end = (
            float(times[i + 1]) + offset
            if i + 1 < len(times)
            else start
        )
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": chunk,
        })
    return segments


def transcribe_waveform(
    waveform: torch.Tensor,
    processor,
    model,
    language: str = "auto",
    task: str = "transcribe",
    with_timestamps: bool = True,
    hint_prompt: str = "",
    beam_size: int = 1,
    condition_on_previous: bool = False,
) -> Dict[str, Any]:
    """
    Transcribe a mono 16 kHz waveform.

    Audio longer than 30 s uses transformers' sequential long-form decoding,
    which carries timestamps across chunks instead of cutting words at a
    hard 30 s boundary.

    Returns:
        {"text": str, "segments": [{"start", "end", "text"}, ...]}
    """
    device = model.device
    duration = waveform.numel() / SAMPLE_RATE
    long_form = duration > CHUNK_SECONDS
    use_prompt = bool(hint_prompt.strip())

    # transformers 5.x returns garbage ("¶¶") when a short-form call gets
    # both `prompt_ids` and `return_timestamps=True`. Long-form is fine.
    # Keep the prompt and drop the timestamps for those clips.
    if use_prompt and with_timestamps and not long_form:
        logger.warning(
            "[TrentNodes] Whisper: a hint prompt on audio under 30 s "
            "disables timestamps (transformers limitation). "
            "The text output is unaffected."
        )
        with_timestamps = False

    inputs = processor(
        waveform.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        truncation=not long_form,
        padding="longest" if long_form else "max_length",
        return_attention_mask=long_form,
    )

    # The actual fix for the fp32-features / fp16-weights crash.
    features = inputs.input_features.to(device=device, dtype=model.dtype)

    gen_kwargs: Dict[str, Any] = {"task": task}
    if language and language != "auto":
        gen_kwargs["language"] = language
    if long_form:
        gen_kwargs["attention_mask"] = inputs.attention_mask.to(device)
        # Long-form decoding always needs timestamps to stitch chunks.
        gen_kwargs["return_timestamps"] = True
        gen_kwargs["return_segments"] = True
        gen_kwargs["condition_on_prev_tokens"] = bool(condition_on_previous)
    elif with_timestamps:
        gen_kwargs["return_timestamps"] = True

    if beam_size > 1:
        gen_kwargs["num_beams"] = int(beam_size)

    if use_prompt:
        prompt_ids = processor.get_prompt_ids(
            hint_prompt.strip(), return_tensors="pt"
        ).to(device)
        gen_kwargs["prompt_ids"] = prompt_ids

    with torch.inference_mode():
        outputs = model.generate(features, **gen_kwargs)

    # Long-form returns a dict with "sequences" and "segments".
    if isinstance(outputs, dict):
        sequences = outputs["sequences"]
        segments = []
        if outputs.get("segments"):
            segments = _decode_segments(processor, outputs["segments"][0])
    else:
        sequences = outputs
        segments = []
        if with_timestamps:
            segments = _segments_from_timestamp_tokens(
                processor, sequences[0]
            )

    text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    text = text.strip()

    if not segments and text:
        segments = [{
            "start": 0.0,
            "end": round(duration, 3),
            "text": text,
        }]

    return {"text": text, "segments": segments, "duration": duration}


def segments_to_lrc(segments: List[Dict[str, Any]]) -> str:
    """Format segments as LRC lyric lines: [mm:ss.xx]text."""
    lines = []
    for seg in segments:
        total = max(0.0, float(seg["start"]))
        minutes = int(total // 60)
        seconds = total - minutes * 60
        lines.append(f"[{minutes:02d}:{seconds:05.2f}]{seg['text']}")
    return "\n".join(lines)


def segments_to_json(segments: List[Dict[str, Any]]) -> str:
    """Serialize segments as pretty JSON."""
    return json.dumps(segments, ensure_ascii=False, indent=2)
