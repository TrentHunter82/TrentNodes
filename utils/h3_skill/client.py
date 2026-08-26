"""
OpenAI-protocol chat transport for the H3 Skill Promptor.

Talks to any OpenAI-compatible endpoint - normally the managed local
llama-server - with interleaved text labels and base64 JPEG image_url
parts (the exact wire shape llama-server's multimodal endpoint and the
cloud providers both accept). Keyless by design; llama-server ignores
the placeholder key.
"""

import time
from typing import List, Optional, Sequence, Tuple

from ..h3_prompt.backends import normalize_seed

PLACEHOLDER_KEY = "sk-no-key"
DEFAULT_TIMEOUT_S = 600.0


def build_user_message(
    text: str,
    images: Sequence[Tuple[str, str]] = (),
    audio_parts: Sequence[str] = (),
) -> dict:
    """
    images: (label, jpeg_b64) pairs, interleaved after the text so every
    picture arrives immediately after the line naming it.
    audio_parts: base64 wav strings, appended as OpenAI input_audio
    parts (llama-server accepts them when the mmproj has an audio
    encoder).
    """
    if not images and not audio_parts:
        return {"role": "user", "content": text}
    content: List[dict] = [{"type": "text", "text": text}]
    for label, jpeg_b64 in images:
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"},
        })
    for wav_b64 in audio_parts:
        content.append({
            "type": "input_audio",
            "input_audio": {"data": wav_b64, "format": "wav"},
        })
    return {"role": "user", "content": content}


def chat(
    base_url: str,
    messages: List[dict],
    *,
    seed: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.80,
    max_tokens: int = 2048,
    reasoning_effort: Optional[str] = "low",
    model: str = "default",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    log_path: Optional[str] = None,
) -> Tuple[str, dict]:
    """One chat completion; returns (text, usage-info dict).

    reasoning_effort=None omits chat_template_kwargs entirely - for
    model families (Qwen3-Omni Instruct) whose template has no such
    variable."""
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is missing. Install with: pip install openai"
        ) from exc

    client = openai.OpenAI(
        base_url=base_url, api_key=PLACEHOLDER_KEY, timeout=timeout_s
    )
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if reasoning_effort is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"reasoning_effort": reasoning_effort}
        }
    # Always sent - dropping seed 0 (the widget default) would make the
    # default configuration non-reproducible on a local server.
    kwargs["seed"] = normalize_seed(seed)

    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
    except openai.APIConnectionError as exc:
        hint = f" Server log: {log_path}" if log_path else ""
        raise RuntimeError(
            f"Could not reach the LLM server at {base_url}. It may have "
            f"crashed or been stopped.{hint}"
        ) from exc

    text = (response.choices[0].message.content or "").strip()
    usage = {
        "model": getattr(response, "model", model),
        "latency_s": round(time.time() - start, 2),
    }
    if getattr(response, "usage", None) is not None:
        usage["prompt_tokens"] = response.usage.prompt_tokens
        usage["completion_tokens"] = response.usage.completion_tokens
    return text, usage
