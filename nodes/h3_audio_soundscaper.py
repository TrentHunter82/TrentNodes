"""
H3 Audio Soundscaper - hear a clip's audio track with a local omni GGUF
(Qwen3-Omni under llama-server) and emit the audio parts of a MiniMax H3
prompt: overall_soundscape, non_diegetic_music, verbatim dialogue, and a
timestamped sound-design log.

Companion to the H3 Skill Promptor: wire `overall_soundscape` /
`non_diegetic_music` prose into your brief (or downstream prompt
surgery) and `dialogue` into the promptor's dialogue input. Runs on its
OWN llama-server port (default 8736) so the text-VLM server on 8735
stays resident - the manager is per-port.
"""

import os

from ..utils import llamacpp_server
from ..utils.h3_skill.audio_io import audio_to_wav_b64
from ..utils.h3_skill.audio_prompts import (
    SYSTEM_PROMPT,
    build_retry_message,
    build_user_context,
    parse_response,
)
from ..utils.h3_skill.client import build_user_message, chat

try:
    import folder_paths
except ImportError:  # dev CLI / pytest outside ComfyUI
    folder_paths = None

DEFAULT_PORT = 8736  # separate slot from the H3 Skill Promptor's 8735


def _gguf_files():
    if folder_paths is None:
        return []
    return folder_paths.get_filename_list("llm_gguf")


def _model_choices():
    files = [f for f in _gguf_files() if "mmproj" not in f.lower()]
    # An omni model is the point of this node - float those to the top.
    files.sort(key=lambda name: ("omni" not in name.lower(), name.lower()))
    return files or ["(put an omni .gguf in models/LLM)"]


def _mmproj_choices():
    files = [f for f in _gguf_files() if "mmproj" in f.lower()]
    return ["auto"] + files + ["none"]


def _resolve_gguf(name: str) -> str:
    if os.path.isabs(name):
        return name
    if folder_paths is not None:
        path = folder_paths.get_full_path("llm_gguf", name)
        if path:
            return path
    raise RuntimeError(
        f"Could not resolve '{name}' to a .gguf file. Put it in "
        "ComfyUI/models/LLM or give an absolute path."
    )


class H3AudioSoundscaper:
    """Describe a clip's audio as H3-ready soundscape/music/dialogue text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "The clip's audio track (LoadAudio / VHS).",
                }),
                "gguf_model": (_model_choices(), {
                    "tooltip": (
                        "An AUDIO-capable (omni) .gguf from models/LLM. "
                        "Text-only models will reject the audio input."
                    ),
                }),
                "mmproj": (_mmproj_choices(), {
                    "default": "auto",
                    "tooltip": (
                        "The model's mmproj (must contain an AUDIO "
                        "encoder). auto = pair by filename."
                    ),
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
            },
            "optional": {
                "scene_context": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "What happens on screen (optional). Used only to "
                        "sort diegetic vs non-diegetic; the model still "
                        "describes only what it hears."
                    ),
                }),
                "base_url": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Attach to a running audio-capable server instead "
                        "of spawning one. Empty = managed server."
                    ),
                }),
                "ctx_size": ("INT", {
                    "default": 16384, "min": 4096, "max": 131072,
                    "step": 4096,
                }),
                "port": ("INT", {
                    "default": DEFAULT_PORT, "min": 1024, "max": 65535,
                }),
                "free_vram_first": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {
                    "default": 1500, "min": 256, "max": 4096,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("overall_soundscape", "non_diegetic_music",
                    "dialogue", "sound_log", "report")
    FUNCTION = "analyze"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "Hears the clip's audio with a local omni GGUF (Qwen3-Omni via "
        "llama-server, port 8736) and writes the audio parts of an H3 "
        "prompt: skill-budgeted overall_soundscape and non_diegetic_music, "
        "verbatim dialogue for <d> tags, and a timestamped sound log. "
        "One corrective retry, no silent rewrites."
    )

    def analyze(
        self,
        audio,
        gguf_model,
        mmproj,
        temperature,
        seed,
        scene_context="",
        base_url="",
        ctx_size=16384,
        port=DEFAULT_PORT,
        free_vram_first=False,
        max_tokens=1500,
    ):
        report = []
        wav_b64, duration, truncated = audio_to_wav_b64(audio)
        report.append(f"audio: {duration:.2f}s @16kHz mono"
                      + (" (truncated)" if truncated else ""))

        if base_url.strip():
            handle = llamacpp_server.attach(base_url.strip())
            model_name = handle.alias or "default"
            if handle.vision is False:
                raise RuntimeError(
                    f"The server at {handle.base_url} reports no "
                    "multimodal capability; it cannot hear audio."
                )
            report.append(f"server: attached {handle.base_url} ({model_name})")
        else:
            model_path = _resolve_gguf(gguf_model)
            if mmproj == "auto":
                mmproj_path = llamacpp_server.find_mmproj_for(model_path)
            elif mmproj == "none":
                mmproj_path = None
            else:
                mmproj_path = _resolve_gguf(mmproj)
            if mmproj_path is None:
                raise RuntimeError(
                    "Audio input needs the model's mmproj (it carries the "
                    "audio encoder). Put it next to the model in "
                    "models/LLM; auto pairs it by filename."
                )
            spec = llamacpp_server.ServerSpec(
                model_path=model_path,
                mmproj_path=mmproj_path,
                ctx_size=int(ctx_size),
                port=int(port),
                reasoning_effort="",  # omni template has no such kwarg
            )
            handle = llamacpp_server.ensure_server(
                spec, free_vram_first=bool(free_vram_first)
            )
            model_name = llamacpp_server._stem(model_path)
            report.append(f"server: {handle.base_url} ({model_name})")

        context = build_user_context(
            duration, scene_context=scene_context, truncated=truncated
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            build_user_message(context, audio_parts=[wav_b64]),
        ]
        chat_kwargs = {
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": None,  # per-request kwarg omitted too
            "model": model_name,
            "log_path": getattr(handle, "log_path", None),
        }

        raw, usage = chat(handle.base_url, messages, **chat_kwargs)
        sections, errors = parse_response(raw)
        report.append(f"latency: {usage.get('latency_s')} s")
        retried = False
        if errors:
            retried = True
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": build_retry_message(errors)})
            raw, usage2 = chat(handle.base_url, messages, **chat_kwargs)
            sections, errors = parse_response(raw)
            report.append(f"retry latency: {usage2.get('latency_s')} s")
            usage = usage2

        report.append(f"corrective retry used: {'yes' if retried else 'no'}")
        # The omni Instruct model does not think, so max_tokens is pure
        # reply budget - hitting it means the sound log got cut.
        if usage.get("finish_reason") == "length":
            report.append(
                f"WARNING: reply truncated at max_tokens ({max_tokens}) "
                "- raise it."
            )
        if errors:
            report.append("CONTRACT VIOLATIONS REMAIN:")
            report.extend(f"  - {error}" for error in errors)
        else:
            report.append("output contract: PASS")

        if not sections:  # both passes malformed: hand back the raw text
            sections = {name: "" for name in
                        ("overall_soundscape", "non_diegetic_music",
                         "dialogue")}
            sections["sound_log"] = raw.strip()
            report.append("reply was unparseable; raw text in sound_log")

        return (
            sections.get("overall_soundscape", ""),
            sections.get("non_diegetic_music", ""),
            sections.get("dialogue", ""),
            sections.get("sound_log", ""),
            "\n".join(report),
        )


NODE_CLASS_MAPPINGS = {
    "TrentH3AudioSoundscaper": H3AudioSoundscaper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentH3AudioSoundscaper": "H3 Audio Soundscaper (Local GGUF)",
}
