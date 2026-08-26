"""
H3 Skill Promptor - standalone MiniMax H3 prompt generation with a
local GGUF vision LLM served by a managed llama-server.

Independent of the H3AutoPromptGenerator / Cowboy pipeline by design:
the system prompt IS the h3-prompting skill text, validation is the
skill's checklist (report + one corrective retry, never a silent
rewrite), and the LLM runs out of process so a llama.cpp crash cannot
take ComfyUI down.
"""

import os

from ..utils import llamacpp_server
from ..utils.any_type import any_typ
from ..utils.h3_prompt import imaging
from ..utils.h3_prompt.keyframes import frame_label, select_keyframes
# Submodule-level imports so the module also loads when a test scaffold
# registers TrentNodes.utils.h3_skill as a hollow namespace package.
from ..utils.h3_skill.skill_loader import (
    CHECKPOINT_FOR_MODE,
    MODES,
    build_system_prompt,
    build_user_context,
)
from ..utils.h3_skill.checklist import assemble_final, validate
from ..utils.h3_skill.client import build_user_message, chat

try:
    import folder_paths
except ImportError:  # dev CLI / pytest outside ComfyUI
    folder_paths = None

if folder_paths is not None:
    _llm_dir = os.path.join(folder_paths.models_dir, "LLM")
    os.makedirs(_llm_dir, exist_ok=True)
    # Load-order safe: "LLM" may already exist (Florence2 registers it);
    # this only appends the path if missing.
    folder_paths.add_model_folder_path("LLM", _llm_dir)
    # Virtual .gguf-filtered view over the same path list (the
    # ComfyUI-GGUF pattern). Shares the list object so later additions
    # to "LLM" propagate.
    _llm_paths = folder_paths.folder_names_and_paths["LLM"][0]
    folder_paths.folder_names_and_paths["llm_gguf"] = (_llm_paths, {".gguf"})


def _gguf_files():
    if folder_paths is None:
        return []
    return folder_paths.get_filename_list("llm_gguf")


def _model_choices():
    files = [f for f in _gguf_files() if "mmproj" not in f.lower()]
    return files or ["(put .gguf files in models/LLM)"]


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


def _strip_transport_wrapper(text: str):
    """
    Remove transport artifacts only - a leaked <think> block or a
    markdown fence around the whole reply. Content is never edited;
    what was removed is reported.
    """
    notes = []
    cleaned = text.strip()
    if cleaned.startswith("<think>") and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
        notes.append("stripped a leaked <think> block")
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        notes.append("stripped a markdown code fence")
    return cleaned, notes


class H3SkillPromptor:
    """Generate an official-format MiniMax H3 prompt with a local GGUF VLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(MODES), {
                    "default": "ref2va",
                    "tooltip": (
                        "ref2va = six-section reference prompt "
                        "(MiniMax-H3-Base-Ref2VA checkpoint). t2va/i2va/"
                        "fl2va/l2va = three-field base prompts "
                        "(MiniMax-H3-Base-FL2VA checkpoint)."
                    ),
                }),
                "creative_brief": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "What the clip should show. Plain language.",
                }),
                "gguf_model": (_model_choices(), {
                    "tooltip": "LLM .gguf from ComfyUI/models/LLM.",
                }),
                "mmproj": (_mmproj_choices(), {
                    "default": "auto",
                    "tooltip": (
                        "Vision projector. auto = pair by filename prefix; "
                        "none = text-only (image inputs then error)."
                    ),
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 6.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": (
                        "Timestamp budget for the shot list. Never written "
                        "into the prompt itself."
                    ),
                }),
                "max_frames_to_analyze": ("INT", {
                    "default": 8, "min": 2, "max": 16,
                    "tooltip": "Keyframes sampled from video_frames.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "0.7 = Unsloth instruct default for Qwen3.8.",
                }),
                "reasoning_effort": (["low", "medium", "xhigh"], {
                    "default": "low",
                    "tooltip": (
                        "Qwen3.8 thinking budget (chat_template_kwargs). "
                        "The template accepts ONLY low/medium/xhigh - "
                        "there is no off switch; low is the minimum."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Folded into int32 for the server.",
                }),
            },
            "optional": {
                "reference_images": ("IMAGE", {
                    "tooltip": (
                        "Picture 1..N in order. ref2va: identity/wardrobe/"
                        "scene references. i2va: first frame. l2va: last "
                        "frame. fl2va: first then last frame."
                    ),
                }),
                "video_frames": ("IMAGE", {
                    "tooltip": "A clip as an image batch; keyframes are sampled.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0,
                    "tooltip": "fps of video_frames, for keyframe timestamps.",
                }),
                "dialogue": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Exact words for <d> tags, kept verbatim.",
                }),
                "base_url": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Attach to a running OpenAI-compatible server "
                        "instead of spawning one (e.g. "
                        "http://127.0.0.1:8735). Empty = managed server."
                    ),
                }),
                "ctx_size": ("INT", {
                    "default": llamacpp_server.DEFAULT_CTX,
                    "min": 4096, "max": 262144, "step": 4096,
                }),
                "port": ("INT", {
                    "default": llamacpp_server.DEFAULT_PORT,
                    "min": 1024, "max": 65535,
                }),
                "free_vram_first": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Unload ComfyUI models before spawning the server. "
                        "Custom-node module caches stay resident either way."
                    ),
                }),
                "max_tokens": ("INT", {
                    "default": 3072, "min": 256, "max": 8192,
                    "tooltip": (
                        "Thinking tokens count against this too; low "
                        "reasoning spends ~1.5k before the prompt starts."
                    ),
                }),
                "source_soundscape": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "Wire the H3 Audio Soundscaper's overall_soundscape "
                        "here: measured diegetic sound of the source clip, "
                        "used to anchor the prompt's soundscape section."
                    ),
                }),
                "source_music": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "Wire the Soundscaper's non_diegetic_music here: "
                        "the measured score anchors the prompt's music "
                        "section."
                    ),
                }),
                "sound_log": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": (
                        "Wire the Soundscaper's sound_log here: timestamped "
                        "events help the model time sound to shots."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "checkpoint_hint", "validation_report")
    FUNCTION = "generate"
    CATEGORY = "Trent/VLM"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The skill file is an input the widgets cannot see: bust the
        # execution cache when it changes on disk.
        from ..utils.h3_skill.skill_loader import (
            LIVE_SKILL_PATH,
            VENDORED_SKILL_PATH,
        )
        for path in (LIVE_SKILL_PATH, VENDORED_SKILL_PATH):
            try:
                return str(os.path.getmtime(path))
            except OSError:
                continue
        return ""
    DESCRIPTION = (
        "Writes an official MiniMax H3 prompt (Ref2VA six-section or Base "
        "three-field) with a local GGUF vision LLM under llama-server. The "
        "system prompt is the h3-prompting skill itself; output is checked "
        "against the skill checklist with one corrective retry and no "
        "silent rewrites."
    )

    def generate(
        self,
        mode,
        creative_brief,
        gguf_model,
        mmproj,
        duration_seconds,
        max_frames_to_analyze,
        temperature,
        reasoning_effort,
        seed,
        reference_images=None,
        video_frames=None,
        fps=24.0,
        dialogue="",
        base_url="",
        ctx_size=llamacpp_server.DEFAULT_CTX,
        port=llamacpp_server.DEFAULT_PORT,
        free_vram_first=False,
        max_tokens=3072,
        source_soundscape="",
        source_music="",
        sound_log="",
    ):
        report = [f"mode: {mode}"]

        # ---- images ----------------------------------------------------
        bracket = mode in ("ref2va", "i2va", "l2va")
        image_pairs = []
        image_lines = []
        if reference_images is not None and reference_images.shape[0] > 8:
            # A batch this size is a clip, not a picture set. Route it
            # to the keyframe sampler instead of failing the run.
            if video_frames is None:
                video_frames = reference_images
                reference_images = None
                report.append(
                    f"reference_images got {video_frames.shape[0]} frames "
                    "- treated as video_frames (keyframes sampled). Use "
                    "reference_images for up to ~6 identity stills."
                )
            else:
                raise RuntimeError(
                    f"reference_images got a batch of "
                    f"{reference_images.shape[0]} frames while "
                    "video_frames is also connected. Each reference image "
                    "becomes a separate <Picture N>; keep reference_images "
                    "to ~6 stills and put clips on video_frames."
                )
        if reference_images is not None:
            for index in range(reference_images.shape[0]):
                number = index + 1
                if mode == "t2va":
                    # t2va prompts contain no picture references at all -
                    # attached images are inspiration only, and a
                    # "Picture N" label would push the model into
                    # writing a reference the validator must then reject.
                    label = (
                        f"(style/content inspiration image {number} - "
                        "describe what you see, never cite it as a "
                        "reference picture)"
                    )
                else:
                    tag = f"<Picture {number}>" if bracket else f"Picture {number}"
                    label = f"{tag} (reference picture {number})"
                image_pairs.append((label, imaging.tensor_to_jpeg_b64(
                    reference_images[index],
                    max_side=imaging.REFERENCE_MAX_SIDE,
                )))
                image_lines.append(label)
        if video_frames is not None and video_frames.shape[0] > 0:
            keyframes = select_keyframes(
                video_frames, fps, max_frames=int(max_frames_to_analyze)
            )
            total = len(keyframes.indices)
            for position, (index, timestamp) in enumerate(
                zip(keyframes.indices, keyframes.timestamps), start=1
            ):
                label = frame_label(position, total, timestamp, index)
                image_pairs.append((label, imaging.tensor_to_jpeg_b64(
                    video_frames[index],
                    max_side=imaging.FRAME_MAX_SIDE,
                )))
                image_lines.append(label)
            report.append(
                f"video frames analyzed: {total} of {video_frames.shape[0]} "
                f"({keyframes.method})"
            )

        # ---- server ----------------------------------------------------
        if base_url.strip():
            handle = llamacpp_server.attach(base_url.strip())
            model_name = handle.alias or "default"
            report.append(
                f"server: attached {handle.base_url} ({model_name})"
            )
            if image_pairs and handle.vision is False:
                raise RuntimeError(
                    "Images are attached, but the server at "
                    f"{handle.base_url} reports no multimodal capability. "
                    "Start it with --mmproj, or remove the image inputs."
                )
        else:
            model_path = _resolve_gguf(gguf_model)
            if mmproj == "auto":
                mmproj_path = llamacpp_server.find_mmproj_for(model_path)
            elif mmproj == "none":
                mmproj_path = None
            else:
                mmproj_path = _resolve_gguf(mmproj)
            if image_pairs and mmproj_path is None:
                raise RuntimeError(
                    "Images are attached but no mmproj vision projector is "
                    "selected. Put the model's mmproj .gguf next to it in "
                    "models/LLM (auto pairs it by filename), or remove the "
                    "image inputs."
                )
            spec = llamacpp_server.ServerSpec(
                model_path=model_path,
                mmproj_path=mmproj_path,
                ctx_size=int(ctx_size),
                port=int(port),
                reasoning_effort=reasoning_effort,
            )
            handle = llamacpp_server.ensure_server(
                spec, free_vram_first=bool(free_vram_first)
            )
            model_name = llamacpp_server._stem(model_path)
            report.append(f"server: {handle.base_url} ({model_name})")
            if mmproj_path:
                report.append(f"mmproj: {os.path.basename(mmproj_path)}")

        # ---- prompt ----------------------------------------------------
        system, skill_source = build_system_prompt(mode)
        report.append(f"skill source: {skill_source}")
        context = build_user_context(
            mode, creative_brief, float(duration_seconds),
            dialogue=dialogue, image_lines=image_lines,
            source_soundscape=source_soundscape,
            source_music=source_music,
            sound_log=sound_log,
        )
        if source_soundscape.strip() or source_music.strip():
            report.append("audio sections anchored to measured source audio")
        messages = [
            {"role": "system", "content": system},
            build_user_message(context, image_pairs),
        ]
        chat_kwargs = {
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "model": model_name,
            "log_path": getattr(handle, "log_path", None),
        }

        raw, usage = chat(handle.base_url, messages, **chat_kwargs)
        body, notes = _strip_transport_wrapper(raw)
        report.extend(notes)
        report.append(
            f"latency: {usage.get('latency_s')} s, tokens: "
            f"{usage.get('prompt_tokens', '?')} in / "
            f"{usage.get('completion_tokens', '?')} out"
        )

        errors = validate(body, mode, float(duration_seconds))
        retried = False
        if errors:
            retried = True
            numbered = "\n".join(
                f"{index}. {error}" for index, error in enumerate(errors, 1)
            )
            messages.append({"role": "assistant", "content": body})
            messages.append({"role": "user", "content": (
                "Your prompt violates the skill checklist:\n"
                f"{numbered}\n"
                "Fix ONLY these violations and return the full corrected "
                "prompt. Same output contract: prompt text only."
            )})
            raw, usage2 = chat(handle.base_url, messages, **chat_kwargs)
            body, notes = _strip_transport_wrapper(raw)
            report.extend(notes)
            report.append(f"corrective retry latency: {usage2.get('latency_s')} s")
            errors = validate(body, mode, float(duration_seconds))

        final_prompt = assemble_final(body, mode, float(duration_seconds))

        report.append(f"corrective retry used: {'yes' if retried else 'no'}")
        if errors:
            report.append("VALIDATION FAILED - remaining violations:")
            report.extend(f"  - {error}" for error in errors)
        else:
            report.append("validation: PASS (skill checklist)")

        return (final_prompt, CHECKPOINT_FOR_MODE[mode], "\n".join(report))


class H3LocalLLMStop:
    """Stop the managed llama-server and release its VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "after": (any_typ, {
                    "tooltip": (
                        "Connect any output here so the stop runs AFTER "
                        "that node. IMPORTANT: left unconnected, ComfyUI "
                        "schedules this output node FIRST in the queue - "
                        "it would kill the server before the promptor "
                        "runs, forcing a cold reload."
                    ),
                }),
                "port": ("INT", {
                    "default": llamacpp_server.DEFAULT_PORT,
                    "min": 1024, "max": 65535,
                    "tooltip": (
                        "Also reaps an orphaned llama-server on this port "
                        "(leftover of a hard ComfyUI crash)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "stop"
    CATEGORY = "Trent/VLM"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Stops the llama-server the H3 Skill Promptor spawned (or an "
        "orphaned one from a crashed session). Process death releases "
        "all of its VRAM immediately. Wire something into `after` to "
        "control WHEN it runs - unconnected output nodes execute first."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def stop(self, after=None, port=llamacpp_server.DEFAULT_PORT):
        if llamacpp_server.stop_server():  # every managed slot (8735, 8736, ...)
            return ("stopped the managed llama-server(s)",)
        if llamacpp_server.stop_orphan(int(port)):
            return (f"stopped an orphaned llama-server on port {port}",)
        return ("no llama-server was running",)


NODE_CLASS_MAPPINGS = {
    "TrentH3SkillPromptor": H3SkillPromptor,
    "TrentH3LocalLLMStop": H3LocalLLMStop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentH3SkillPromptor": "H3 Skill Promptor (Local GGUF)",
    "TrentH3LocalLLMStop": "H3 Local LLM Stop (free VRAM)",
}
