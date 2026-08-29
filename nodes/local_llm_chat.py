"""
Ask Local LLM - a general-purpose chat node over the managed llama-server.

The H3 Skill Promptor's infrastructure (utils/llamacpp_server.py +
utils/h3_skill/client.py), minus the H3 skill contract: free system
prompt, free question, optional images, and a JSON chat history that
chains node-to-node for multi-turn conversations. Defaults mirror the
promptor's ServerSpec exactly (port 8735, ctx 32768, mmproj auto), so
when the promptor's Qwen3.8 server is already resident this node reuses
it with zero reload - and vice versa.
"""

import json
import os

from ..utils import llamacpp_server
from ..utils.h3_prompt import imaging
from ..utils.h3_skill.client import (
    BUDGET_MESSAGE,
    THINKING_ALLOWANCE,
    build_user_message,
    chat,
)

try:
    import folder_paths
except ImportError:  # dev CLI / pytest outside ComfyUI
    folder_paths = None

# The h3_skill_promptor module normally registers "llm_gguf" at import;
# guard so this node also works if that module is ever removed.
if folder_paths is not None and "llm_gguf" not in folder_paths.folder_names_and_paths:
    _llm_dir = os.path.join(folder_paths.models_dir, "LLM")
    os.makedirs(_llm_dir, exist_ok=True)
    folder_paths.add_model_folder_path("LLM", _llm_dir)
    _llm_paths = folder_paths.folder_names_and_paths["LLM"][0]
    folder_paths.folder_names_and_paths["llm_gguf"] = (_llm_paths, {".gguf"})

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, direct assistant. Answer plainly and concisely. "
    "When asked to write a prompt for an image, video, or audio model, "
    "return only the prompt text itself unless asked otherwise."
)


def _gguf_files():
    if folder_paths is None:
        return []
    return folder_paths.get_filename_list("llm_gguf")


def _model_choices():
    files = [f for f in _gguf_files() if "mmproj" not in f.lower()]
    # Text-first node: sink omni audio models below the text VLMs.
    files.sort(key=lambda name: ("omni" in name.lower(), name.lower()))
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


def _parse_history(history_json: str):
    """JSON list of {"role": user|assistant, "content": str} -> messages.
    Empty/blank input is a fresh conversation; malformed input raises
    (a silently dropped history would look like the model forgot)."""
    text = (history_json or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"history_json is not valid JSON: {exc}. Wire the history "
            "output of another Ask Local LLM node here, or leave it empty."
        ) from exc
    if not isinstance(data, list):
        raise RuntimeError(
            "history_json must be a JSON list of "
            '{"role": ..., "content": ...} objects.'
        )
    messages = []
    for index, entry in enumerate(data):
        role = entry.get("role") if isinstance(entry, dict) else None
        content = entry.get("content") if isinstance(entry, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise RuntimeError(
                f"history_json entry {index} is not "
                '{"role": "user"|"assistant", "content": str}. System '
                "prompts live in the system_prompt widget, not the history."
            )
        messages.append({"role": role, "content": content})
    return messages


def _strip_think(text: str):
    """Remove a leaked <think> block; content is never edited otherwise.
    Markdown fences are kept - in a general chat answer they are usually
    intentional (code), unlike in the H3 promptor's prompt-only output."""
    cleaned = text.strip()
    if cleaned.startswith("<think>") and "</think>" in cleaned:
        return cleaned.split("</think>", 1)[1].strip(), True
    return cleaned, False


def _check_finish(body: str, usage: dict, max_tokens: int,
                  reasoning_effort: str):
    """Turn finish_reason == "length" into an actionable signal."""
    if usage.get("finish_reason") != "length":
        return []
    if not body:
        raise RuntimeError(
            "The model hit the token limit while still thinking and "
            "returned no reply text (finish_reason length, "
            f"{usage.get('completion_tokens', '?')} completion tokens at "
            f"reasoning_effort {reasoning_effort}). Raise max_tokens "
            f"(now {max_tokens}) or lower reasoning_effort."
        )
    return [
        "WARNING: the reply hit the token limit and may be cut short. "
        "Raise max_tokens."
    ]


class LocalLLMChat:
    """Ask the local GGUF LLM anything; chain history for follow-ups."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "What to ask. Plain language.",
                }),
                "system_prompt": ("STRING", {
                    "default": DEFAULT_SYSTEM_PROMPT, "multiline": True,
                    "tooltip": (
                        "Who the model is for this call. Ignored widgets "
                        "aside, this is the only steering the model gets - "
                        "put format rules and role here."
                    ),
                }),
                "gguf_model": (_model_choices(), {
                    "tooltip": "LLM .gguf from ComfyUI/models/LLM.",
                }),
                "mmproj": (_mmproj_choices(), {
                    "default": "auto",
                    "tooltip": (
                        "Vision projector. auto = pair by filename prefix "
                        "(keeps the server spec identical to the H3 Skill "
                        "Promptor's, so a resident server is reused with "
                        "no reload). none = text-only."
                    ),
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "0.7 = Unsloth instruct default for Qwen3.8.",
                }),
                "reasoning_effort": (["low", "medium", "xhigh"], {
                    "default": "low",
                    "tooltip": (
                        "Qwen3.8 thinking budget (chat_template_kwargs). "
                        "The template accepts ONLY low/medium/xhigh."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": (
                        "Bump for a different answer to the same prompt "
                        "(the node caches on identical inputs)."
                    ),
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 64, "max": 16384,
                    "tooltip": (
                        "Budget for the visible reply. Thinking gets its "
                        "own capped allowance on top (low +2048, medium "
                        "+3072, xhigh +7168), so it cannot starve the reply."
                    ),
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Optional pictures to ask about (Image 1..N in "
                        "batch order). Needs an mmproj vision projector."
                    ),
                }),
                "input_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Wired-in text (a prompt to improve, a caption, "
                        "a file). Appended to your prompt under an "
                        "INPUT TEXT header."
                    ),
                }),
                "history_json": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Wire another Ask Local LLM node's history output "
                        "here to continue that conversation."
                    ),
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
                    "tooltip": (
                        "Keep at the default to share the resident server "
                        "with the H3 nodes - a different value forces a "
                        "cold respawn."
                    ),
                }),
                "port": ("INT", {
                    "default": llamacpp_server.DEFAULT_PORT,
                    "min": 1024, "max": 65535,
                }),
                "free_vram_first": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Unload ComfyUI models before spawning the server."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "history_json", "info")
    FUNCTION = "ask"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "General-purpose chat with a local GGUF LLM under the same "
        "managed llama-server as the H3 Skill Promptor (defaults share "
        "its resident Qwen3.8 on port 8735 with no reload). Ask anything: "
        "write or refine prompts for other models, describe attached "
        "images, brainstorm. Chain history_json between nodes for "
        "multi-turn follow-ups."
    )

    def ask(
        self,
        prompt,
        system_prompt,
        gguf_model,
        mmproj,
        temperature,
        reasoning_effort,
        seed,
        max_tokens,
        images=None,
        input_text="",
        history_json="",
        base_url="",
        ctx_size=llamacpp_server.DEFAULT_CTX,
        port=llamacpp_server.DEFAULT_PORT,
        free_vram_first=False,
    ):
        info = []

        # ---- user turn -------------------------------------------------
        user_text = prompt.strip()
        if input_text.strip():
            user_text = (
                f"{user_text}\n\n--- INPUT TEXT ---\n{input_text.strip()}"
                if user_text else input_text.strip()
            )
        if not user_text and images is None:
            raise RuntimeError(
                "The prompt is empty. Type a question, wire input_text, "
                "or attach images."
            )
        if not user_text:
            user_text = "Describe the attached image(s)."

        image_pairs = []
        if images is not None:
            for index in range(images.shape[0]):
                label = f"Image {index + 1}:"
                image_pairs.append((label, imaging.tensor_to_jpeg_b64(
                    images[index], max_side=imaging.REFERENCE_MAX_SIDE,
                )))
            info.append(f"images attached: {len(image_pairs)}")

        # ---- server ----------------------------------------------------
        if base_url.strip():
            handle = llamacpp_server.attach(base_url.strip())
            model_name = handle.alias or "default"
            info.append(f"server: attached {handle.base_url} ({model_name})")
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
                    "Images are attached but no mmproj vision projector "
                    "is selected. Put the model's mmproj .gguf next to it "
                    "in models/LLM (auto pairs it by filename), or remove "
                    "the image inputs."
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
            info.append(f"server: {handle.base_url} ({model_name})")

        # ---- messages --------------------------------------------------
        history = _parse_history(history_json)
        if history:
            info.append(f"history turns carried in: {len(history)}")
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append(build_user_message(user_text, image_pairs))

        allowance = THINKING_ALLOWANCE.get(reasoning_effort, 0)
        raw, usage = chat(
            handle.base_url,
            messages,
            seed=seed,
            temperature=temperature,
            max_tokens=int(max_tokens) + allowance,
            reasoning_effort=reasoning_effort,
            reasoning_budget=allowance or None,
            reasoning_budget_message=BUDGET_MESSAGE,
            model=model_name,
            log_path=getattr(handle, "log_path", None),
        )
        body, stripped = _strip_think(raw)
        if stripped:
            info.append("stripped a leaked <think> block")
        info.append(
            f"latency: {usage.get('latency_s')} s, tokens: "
            f"{usage.get('prompt_tokens', '?')} in / "
            f"{usage.get('completion_tokens', '?')} out, "
            f"finish: {usage.get('finish_reason', '?')}"
        )
        info.extend(_check_finish(body, usage, int(max_tokens),
                                  reasoning_effort))

        # History stays text-only: re-sending base64 images every turn
        # would balloon the context. A note marks where they were.
        stored_user = user_text
        if image_pairs:
            stored_user += f"\n[{len(image_pairs)} image(s) were attached]"
        new_history = history + [
            {"role": "user", "content": stored_user},
            {"role": "assistant", "content": body},
        ]

        return (body, json.dumps(new_history, indent=2), "\n".join(info))


NODE_CLASS_MAPPINGS = {
    "TrentLocalLLMChat": LocalLLMChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentLocalLLMChat": "Ask Local LLM (GGUF)",
}
