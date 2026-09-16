"""Official-skill H3 prompting through a locally signed-in Codex CLI."""

from pathlib import Path

import comfy.model_management
import folder_paths

from ..utils.h3_codex import prompt
from ..utils.h3_codex.skill import REVISION


def text_input(tooltip, multiline=True):
    return ("STRING", {"default": "", "multiline": multiline, "tooltip": tooltip})


class H3CodexPromptor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "creative_brief": text_input("Describe what should happen. Codex uses your ChatGPT subscription to write an H3 prompt."),
                "duration_seconds": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 600.0, "step": 0.1,
                    "tooltip": "Match the actual H3 clip duration. The official guide targets 4–15 seconds."}),
                "mode": (list(prompt.MODES), {"default": "auto",
                    "tooltip": "Auto: first/last frame inputs choose base modes; general references choose Ref2VA. Override to match your checkpoint."}),
                "action": (["generate", "refine", "locked"], {"default": "generate",
                    "tooltip": "Generate from your brief, refine the editor using your request, or use the locked editor with no Codex call."}),
                "prompt_text": text_input("Editable result. Lock uses this exact text. Saved inside the workflow."),
                "refinement": text_input("What to change in the current prompt, e.g. slower camera movement; preserve dialogue exactly."),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "Opening keyframe; each batch image receives a Picture label."}),
                "last_frame": ("IMAGE", {"tooltip": "Ending keyframe. With first_frame, auto chooses FL2VA."}),
                "reference_images": ("IMAGE", {"tooltip": "Reference batch in the same order you wire into H3. Auto selects Ref2VA."}),
                "video_frames": ("IMAGE", {"tooltip": "Video as a frame batch (VHS IMAGE output). Codex sees sampled frames, not the full video or its audio."}),
                "audio": ("AUDIO", {"tooltip": "Marks Audio 1 as available to your H3 workflow. Metadata only; connect Soundscaper analysis below to describe its content."}),
                "reference_roles": text_input("Explain what each reference supplies, e.g. Picture 1 identity; Video 1 motion; Audio 1 voice."),
                "context": text_input("Extra scene, character, story, style, or continuity context."),
                "dialogue": text_input("Exact spoken words, lyrics, and speaker information to preserve."),
                "source_soundscape": text_input("Connect H3 Audio Soundscaper overall_soundscape here."),
                "source_music": text_input("Connect H3 Audio Soundscaper non_diegetic_music here."),
                "sound_log": text_input("Connect H3 Audio Soundscaper sound_log or a transcript here."),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0,
                    "tooltip": "Source video fps, used for sampled-frame timestamps."}),
                "max_frames": ("INT", {"default": 8, "min": 2, "max": 32}),
                "model": text_input("Empty uses your Codex configured model. Otherwise enter a model available to your subscription.", False),
                "reasoning_effort": (["default", "low", "medium", "high", "xhigh"], {"default": "default"}),
                "revision": ("INT", {"default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "Increase to request another draft with the same inputs. Keep fixed when rendering H3 seed variations."}),
                "timeout_seconds": ("INT", {"default": 600, "min": 30, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "checkpoint_hint", "validation_report", "overall_soundscape", "non_diegetic_music")
    FUNCTION = "generate"
    CATEGORY = "Trent/VLM"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Writes and refines MiniMax H3 prompts using the official MiniMax skill and your Codex ChatGPT login. "
        "H3 renders locally; the brief and attached images are sent to Codex. No API key or local LLM needed. "
        "First use downloads the pinned skill. Successful requests are cached; locked mode is offline."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return REVISION

    def generate(self, creative_brief, duration_seconds, mode, action, prompt_text,
                 refinement, first_frame=None, last_frame=None, reference_images=None,
                 video_frames=None, audio=None, reference_roles="", context="", dialogue="",
                 source_soundscape="", source_music="", sound_log="", fps=24.0,
                 max_frames=8, model="", reasoning_effort="default", revision=0,
                 timeout_seconds=600):
        result = prompt.generate(
            Path(folder_paths.get_user_directory()) / "default" / "trentnodes" / "h3_codex",
            brief=creative_brief, duration=float(duration_seconds), mode=mode,
            action=action, prompt_text=prompt_text, refinement=refinement,
            first_frame=first_frame, last_frame=last_frame, reference_images=reference_images,
            video_frames=video_frames, audio=audio, reference_roles=reference_roles,
            context=context, dialogue=dialogue, source_soundscape=source_soundscape,
            source_music=source_music, sound_log=sound_log, fps=float(fps),
            max_frames=int(max_frames), model=model.strip(),
            effort="" if reasoning_effort == "default" else reasoning_effort,
            revision=int(revision), timeout=int(timeout_seconds),
            interrupt=comfy.model_management.throw_exception_if_processing_interrupted,
        )
        return {
            "ui": {"h3_prompt": [result["prompt"]], "h3_report": [result["report"]]},
            "result": tuple(result[key] for key in ("prompt", "checkpoint", "report", "soundscape", "music")),
        }


NODE_CLASS_MAPPINGS = {"TrentH3CodexPromptor": H3CodexPromptor}
NODE_DISPLAY_NAME_MAPPINGS = {"TrentH3CodexPromptor": "H3 Codex Promptor (Trent)"}
