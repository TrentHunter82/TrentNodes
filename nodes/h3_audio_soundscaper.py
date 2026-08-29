"""
H3 Audio Soundscaper - two modes, one output contract:

- listening: hear a clip's audio track with a local omni GGUF
  (Qwen3-Omni under llama-server) and transcribe the audio parts of a
  MiniMax H3 prompt: overall_soundscape, non_diegetic_music, verbatim
  dialogue, and a timestamped sound-design log.
- design: no audio connected + video_prompt filled -> read the video
  prompt and DESIGN the soundtrack from text alone (same four
  sections; no mmproj needed).

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
    DESIGN_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_design_context,
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
    """Hear a clip (or read a prompt) -> H3 soundscape/music/dialogue."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_model": (_model_choices(), {
                    "tooltip": (
                        "An AUDIO-capable (omni) .gguf from models/LLM. "
                        "Text-only models will reject the audio input "
                        "(design mode works with any chat model)."
                    ),
                }),
                "mmproj": (_mmproj_choices(), {
                    "default": "auto",
                    "tooltip": (
                        "The model's mmproj (must contain an AUDIO "
                        "encoder). auto = pair by filename. Design mode "
                        "(no audio connected) does not need one."
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
                "audio": ("AUDIO", {
                    "tooltip": (
                        "The clip's audio track (LoadAudio / VHS). "
                        "Leave unconnected and fill video_prompt to "
                        "design a soundtrack from text instead."
                    ),
                }),
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
                # New widget stays LAST: widget values save positionally.
                "video_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "An H3 video prompt or scene description. With "
                        "no audio connected, the node designs the "
                        "soundtrack for it from text (design mode). "
                        "With audio connected, it adds to "
                        "scene_context."
                    ),
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
        "No audio + a video_prompt = design mode: the same sections are "
        "invented to fit the described visuals. "
        "One corrective retry, no silent rewrites."
    )

    def analyze(
        self,
        gguf_model,
        mmproj,
        temperature,
        seed,
        audio=None,
        scene_context="",
        base_url="",
        ctx_size=16384,
        port=DEFAULT_PORT,
        free_vram_first=False,
        max_tokens=1500,
        video_prompt="",
    ):
        report = []
        design_mode = audio is None
        if design_mode:
            source_text = (video_prompt or "").strip() or \
                (scene_context or "").strip()
            if not source_text:
                raise RuntimeError(
                    "Nothing to analyze: connect the clip's audio to "
                    "hear it, or fill video_prompt to design a "
                    "soundtrack from text."
                )
            report.append("mode: design (no audio; soundtrack invented "
                          "from the video prompt)")
        else:
            report.append("mode: listening")
            wav_b64, duration, truncated = audio_to_wav_b64(audio)
            report.append(f"audio: {duration:.2f}s @16kHz mono"
                          + (" (truncated)" if truncated else ""))

        if base_url.strip():
            handle = llamacpp_server.attach(base_url.strip())
            model_name = handle.alias or "default"
            if not design_mode and handle.vision is False:
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
            if mmproj_path is None and not design_mode:
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

        if design_mode:
            messages = [
                {"role": "system", "content": DESIGN_SYSTEM_PROMPT},
                build_user_message(build_design_context(source_text)),
            ]
        else:
            # A filled video_prompt still helps in listening mode: it
            # rides along as extra scene context for the diegetic sort.
            context_text = "\n\n".join(
                part for part in
                ((scene_context or "").strip(), (video_prompt or "").strip())
                if part
            )
            context = build_user_context(
                duration, scene_context=context_text, truncated=truncated
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
