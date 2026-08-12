"""
VLM provider backends for the H3 Auto Prompt Generator.

One small interface (VLMBackend.generate) with hosted-API and local
implementations. All SDK imports are lazy inside constructors so the
node registers even when a provider's package is missing; selecting
that provider then raises an actionable RuntimeError (same UX as the
fal/midjourney nodes).

API keys resolve widget-first, then environment variable, mirroring
utils/midjourney_client.resolve_credentials().
"""

import base64
import io
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-4o",
    "kimi": "kimi-k3",
    "glm": "glm-4.6v",
    "qwen_api": "qwen3-vl-plus",
    "qwen_local": "Qwen/Qwen3-VL-8B-Instruct",
    "minicpm_local": "minicpm-v-4.5",
    "magevl_local": "microsoft/Mage-VL",
    "ollama": "qwen3-vl",
}

# OpenAI-protocol providers: base_url None = the real OpenAI endpoint.
# All of these take images as data-URI base64 image_url parts.
# supports_seed: only pass the seed kwarg where it is documented;
# unknown kwargs can 400 on some compatible endpoints.
OPENAI_COMPAT_PROVIDERS = {
    "openai": {
        "base_url": None,
        "env": "OPENAI_API_KEY",
        "supports_seed": True,
    },
    # Moonshot Kimi K3 (native-vision MoE, 2026-07 release)
    "kimi": {
        "base_url": "https://api.moonshot.ai/v1",
        "env": "MOONSHOT_API_KEY",
        "supports_seed": False,
    },
    # Z.ai (Zhipu) GLM vision line
    "glm": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "env": "ZAI_API_KEY",
        "supports_seed": False,
    },
    # Alibaba DashScope international, OpenAI compatible-mode
    "qwen_api": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "env": "DASHSCOPE_API_KEY",
        "supports_seed": True,
    },
}

PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    # Google publishes both names; accept either.
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    **{name: cfg["env"] for name, cfg in OPENAI_COMPAT_PROVIDERS.items()},
}


@dataclass
class VLMImage:
    label: str
    jpeg_b64: str
    media_type: str = "image/jpeg"

    def to_pil(self):
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(self.jpeg_b64)))


@dataclass
class VLMAudio:
    """Source-clip audio for soundscape description. 16 kHz mono WAV."""
    wav_b64: str
    duration_seconds: float
    media_type: str = "audio/wav"


@dataclass
class VLMResult:
    text: str
    usage: Dict = field(default_factory=dict)


class VLMBackend(ABC):
    name = "base"
    # Only backends that actually accept an audio track set this. The
    # node warns and drops the audio for the rest rather than failing.
    supports_audio = False

    @abstractmethod
    def generate(
        self,
        system: str,
        images: List[VLMImage],
        user_text: str,
        max_tokens: int = 4096,
        seed: int = 0,
        audio: Optional[VLMAudio] = None,
    ) -> VLMResult:
        ...


def resolve_api_key(widget_key: str, provider: str) -> str:
    """Widget value first, then the provider's env var(s)."""
    env_vars = PROVIDER_ENV_VARS.get(provider, ())
    if isinstance(env_vars, str):
        env_vars = (env_vars,)

    api_key = (widget_key or "").strip()
    for env_var in env_vars:
        if api_key:
            break
        api_key = (os.environ.get(env_var) or "").strip()

    if not api_key:
        raise RuntimeError(
            f"{provider} API key missing. Paste it into the node's "
            f"api_key widget or export {' or '.join(env_vars)} in the "
            "shell that starts ComfyUI."
        )
    return api_key


def _legend_prompt(images: List[VLMImage], user_text: str) -> str:
    """
    For backends without true text/image interleaving: prepend a legend
    mapping image order to labels.
    """
    legend = "\n".join(
        f"Image {i + 1} = {img.label}" for i, img in enumerate(images)
    )
    return f"IMAGE ORDER\n{legend}\n\n{user_text}"


class AnthropicBackend(VLMBackend):
    name = "anthropic"

    def __init__(self, model: str, api_key: str = ""):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is missing. Install with: "
                "pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(
            api_key=resolve_api_key(api_key, "anthropic")
        )
        self.model = model

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        content = []
        for img in images:
            content.append({"type": "text", "text": img.label})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.jpeg_b64,
                },
            })
        content.append({"type": "text", "text": user_text})

        start = time.time()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(
                "The Anthropic model declined to process this request."
            )
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = {
            "model": self.model,
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
            "latency_s": round(time.time() - start, 2),
        }
        return VLMResult(text=text, usage=usage)


class GeminiBackend(VLMBackend):
    """
    Google Gemini via the native google-genai SDK.

    Native rather than Gemini's OpenAI-compatibility shim because the
    SDK path carries an audio track reliably - Gemini is currently the
    only provider here that can listen to the source clip and describe
    the real soundscape instead of inferring it from pixels.
    """

    name = "gemini"
    supports_audio = True

    def __init__(self, model: str, api_key: str = ""):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The google-genai package is missing. Install with: "
                "pip install google-genai"
            ) from exc
        self._types = types
        self._client = genai.Client(
            api_key=resolve_api_key(api_key, "gemini")
        )
        self.model = model

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        types = self._types
        parts = []
        for img in images:
            parts.append(types.Part.from_text(text=img.label))
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(img.jpeg_b64),
                mime_type=img.media_type,
            ))
        if audio is not None:
            parts.append(types.Part.from_text(
                text=(
                    "Audio track of <Video 1> "
                    f"({audio.duration_seconds:.2f}s). Describe the "
                    "soundscape from what you actually hear here."
                )
            ))
            parts.append(types.Part.from_bytes(
                data=base64.b64decode(audio.wav_b64),
                mime_type=audio.media_type,
            ))
        parts.append(types.Part.from_text(text=user_text))

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.0,
        )
        if seed:
            config.seed = seed

        start = time.time()
        response = self._client.models.generate_content(
            model=self.model, contents=parts, config=config,
        )

        text = getattr(response, "text", None)
        if not text:
            reason = ""
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                reason = str(getattr(candidates[0], "finish_reason", "") or "")
            feedback = getattr(response, "prompt_feedback", None)
            if feedback is not None and not reason:
                reason = str(getattr(feedback, "block_reason", "") or "")
            raise RuntimeError(
                "Gemini returned no text"
                + (f" (finish reason: {reason})" if reason else "")
                + ". A safety filter or token limit is the usual cause."
            )

        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "model": self.model,
            "input_tokens": getattr(usage_meta, "prompt_token_count", None),
            "output_tokens": getattr(
                usage_meta, "candidates_token_count", None
            ),
            "latency_s": round(time.time() - start, 2),
            "audio_sent": audio is not None,
        }
        return VLMResult(text=text, usage=usage)


class OpenAICompatibleBackend(VLMBackend):
    """
    Any provider speaking the OpenAI chat-completions protocol with
    image_url data-URI parts: OpenAI itself, Moonshot Kimi, Z.ai GLM,
    DashScope Qwen. Configured via OPENAI_COMPAT_PROVIDERS.
    """

    def __init__(self, provider: str, model: str, api_key: str = ""):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is missing (used for all "
                "OpenAI-compatible providers). Install with: "
                "pip install openai"
            ) from exc
        config = OPENAI_COMPAT_PROVIDERS[provider]
        self.name = provider
        self._supports_seed = config["supports_seed"]
        client_kwargs = {"api_key": resolve_api_key(api_key, provider)}
        if config["base_url"]:
            client_kwargs["base_url"] = config["base_url"]
        self._client = openai.OpenAI(**client_kwargs)
        self.model = model

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        content = []
        for img in images:
            content.append({"type": "text", "text": img.label})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.media_type};base64,{img.jpeg_b64}"
                },
            })
        content.append({"type": "text", "text": user_text})

        kwargs = {}
        if seed and self._supports_seed:
            kwargs["seed"] = seed
        start = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        usage = {
            "model": self.model,
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(
                response.usage, "completion_tokens", None
            ),
            "latency_s": round(time.time() - start, 2),
        }
        return VLMResult(text=text, usage=usage)


class QwenVLBackend(VLMBackend):
    name = "qwen_local"

    def __init__(self, model: str, api_key: str = ""):
        from . import qwen_wrapper
        self._wrapper = qwen_wrapper
        self.model = model

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        labeled = [(img.label, img.to_pil()) for img in images]
        start = time.time()
        text = self._wrapper.run_qwen_inference(
            labeled, system, user_text,
            model_id=self.model, max_tokens=max_tokens,
        )
        return VLMResult(
            text=text,
            usage={
                "model": self.model,
                "latency_s": round(time.time() - start, 2),
            },
        )


class MiniCPMBackend(VLMBackend):
    name = "minicpm_local"

    def __init__(self, model: str, api_key: str = ""):
        try:
            from .. import minicpm_wrapper
        except ImportError as exc:
            raise RuntimeError(
                "The MiniCPM wrapper is unavailable (transformers "
                "missing?). Install TrentNodes requirements."
            ) from exc
        self._wrapper = minicpm_wrapper
        self.model = model

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        pils = [img.to_pil() for img in images]
        prompt = _legend_prompt(images, user_text)
        start = time.time()
        text = self._wrapper.run_inference(
            pils,
            prompt,
            mode="multi_image",
            system_prompt=system,
            thinking_mode="fast",
            max_tokens=min(max_tokens, 4096),
            temperature=0.1,
            seed=seed,
        )
        return VLMResult(
            text=text,
            usage={
                "model": self.model,
                "latency_s": round(time.time() - start, 2),
            },
        )


class MageVLBackend(VLMBackend):
    """
    Microsoft Mage-VL 4B via utils/magevl_wrapper.py (shared with the
    VidScribe node, including its cache and idle auto-unload). The
    wrapper imports folder_paths/comfy, so this backend only works
    inside the ComfyUI environment - not from the standalone dev CLI.
    """

    name = "magevl_local"

    def __init__(self, model: str, api_key: str = ""):
        try:
            from .. import magevl_wrapper
        except ImportError as exc:
            raise RuntimeError(
                "The Mage-VL wrapper is unavailable. It needs the "
                "ComfyUI environment and transformers >= 5.7."
            ) from exc
        if not magevl_wrapper.is_magevl_available():
            raise RuntimeError(
                "Mage-VL needs transformers >= 5.7 and accelerate. "
                "Update the ComfyUI venv requirements."
            )
        self._wrapper = magevl_wrapper
        self.model = model

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        pils = [img.to_pil() for img in images]
        prompt = _legend_prompt(images, user_text)
        start = time.time()
        text = self._wrapper.run_magevl_inference(
            pils,
            prompt,
            mode="multi_image",
            system_prompt=system,
            max_tokens=min(max_tokens, 4096),
            temperature=0.0,
            seed=seed,
        )
        if text.startswith("[Error]"):
            raise RuntimeError(f"Mage-VL backend failed: {text}")
        return VLMResult(
            text=text,
            usage={
                "model": self.model,
                "latency_s": round(time.time() - start, 2),
            },
        )


class OllamaBackend(VLMBackend):
    name = "ollama"

    def __init__(self, model: str, api_key: str = ""):
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "The ollama package is missing. Install with: "
                "pip install ollama (and run an ollama server)."
            ) from exc
        self._ollama = ollama
        self.model = model

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        start = time.time()
        response = self._ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": _legend_prompt(images, user_text),
                    "images": [img.jpeg_b64 for img in images],
                },
            ],
            options={"num_predict": max_tokens, "temperature": 0.0},
        )
        return VLMResult(
            text=response["message"]["content"],
            usage={
                "model": self.model,
                "latency_s": round(time.time() - start, 2),
            },
        )


def _compat_factory(provider: str):
    def factory(model: str, api_key: str = ""):
        return OpenAICompatibleBackend(provider, model, api_key=api_key)
    return factory


_BACKEND_CLASSES = {
    "anthropic": AnthropicBackend,
    "gemini": GeminiBackend,
    **{name: _compat_factory(name) for name in OPENAI_COMPAT_PROVIDERS},
    "qwen_local": QwenVLBackend,
    "minicpm_local": MiniCPMBackend,
    "magevl_local": MageVLBackend,
    "ollama": OllamaBackend,
}


def get_backend(provider: str, model: str = "auto", api_key: str = "") -> VLMBackend:
    """Instantiate the backend for `provider`; 'auto' picks its default model."""
    if provider not in _BACKEND_CLASSES:
        raise RuntimeError(
            f"Unknown VLM provider '{provider}'. "
            f"Options: {', '.join(sorted(_BACKEND_CLASSES))}"
        )
    resolved_model = model.strip()
    if not resolved_model or resolved_model == "auto":
        resolved_model = DEFAULT_MODELS[provider]
    return _BACKEND_CLASSES[provider](resolved_model, api_key=api_key)
