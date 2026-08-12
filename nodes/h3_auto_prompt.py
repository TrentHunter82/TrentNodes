"""
H3 Auto Prompt Generator.

Analyzes a source video plus an identity reference image with a VLM and
emits a production-ready Minimax H3 REF2VA prompt in the official
six-section format (subject_definitions / summary / retention_analysis /
detailed_description / overall_soundscape / non_diegetic_music, plus a
trailing block of "No ..." exclusion sentences).

Pipeline: keyframe selection (scene cuts > motion peaks > anchors) ->
one VLM call with timestamp-labeled frames -> deterministic assembler
that repairs format drift -> optional single corrective retry -> final
prompt + debug JSON.
"""

import json
from fractions import Fraction
from typing import Optional, Tuple

import torch

from ..utils.h3_prompt import assembler, audio_io, prompts, video_io
from ..utils.h3_prompt.backends import (
    DEFAULT_MODELS,
    VLMAudio,
    VLMImage,
    get_backend,
)
from ..utils.h3_prompt.video_io import VLMVideo
from ..utils.h3_prompt.imaging import (
    FRAME_MAX_SIDE,
    REFERENCE_MAX_SIDE,
    tensor_to_jpeg_b64,
)
from ..utils.h3_prompt.keyframes import frame_label, select_keyframes

MAX_RETRIES = 1
LOG_PREFIX = "[H3AutoPrompt]"


class H3AutoPromptGenerator:
    """Generate a Minimax H3 character-replacement prompt from a video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Identity + wardrobe reference (<Picture 1>)"
                }),
                "subject_name": ("STRING", {
                    "default": "Aria Voss",
                    "tooltip": "Name used for <Subject 1> throughout the prompt"
                }),
                "subject_wardrobe": ("STRING", {
                    "multiline": True,
                    "default": (
                        "charcoal utility jacket over a slate-gray tee, "
                        "black cargo pants, scuffed black combat boots"
                    ),
                    "tooltip": (
                        "Comma-separated wardrobe items exactly as seen in "
                        "the reference image. Named 3+ times in the prompt."
                    )
                }),
                "scene_style": ("STRING", {
                    "multiline": True,
                    "default": (
                        "gritty handheld action thriller, overcast daylight, "
                        "desaturated teal-and-rust color grade"
                    ),
                    "tooltip": "One-line cinematic style for detailed_description"
                }),
                "soundscape_type": (["fight", "ambient", "dialogue"], {
                    "default": "ambient",
                    "tooltip": "Drives the overall_soundscape guidance"
                }),
                "vlm_provider": (
                    ["anthropic", "gemini", "openai", "kimi", "glm",
                     "qwen_api", "qwen_local", "minicpm_local",
                     "magevl_local", "ollama"],
                    {
                        "default": "anthropic",
                        "tooltip": (
                            "Hosted APIs: anthropic, gemini (Google; the "
                            "only provider that can hear the audio "
                            "input), openai, kimi (Moonshot Kimi K3), "
                            "glm (Z.ai GLM vision), qwen_api (DashScope "
                            "intl). Local: qwen_local/minicpm_local/"
                            "magevl_local (Microsoft Mage-VL 4B, shared "
                            "with VidScribe) run on this GPU; ollama "
                            "needs an ollama server."
                        )
                    }
                ),
                "model": ("STRING", {
                    "default": "auto",
                    "tooltip": (
                        "auto = provider default ("
                        + ", ".join(
                            f"{k}: {v}" for k, v in DEFAULT_MODELS.items()
                        )
                        + "). Or type an explicit model id."
                    )
                }),
                "max_frames_to_analyze": ("INT", {
                    "default": 8, "min": 2, "max": 16,
                    "tooltip": "Keyframes sent to the VLM (plus the reference)"
                }),
                "enable_audio_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Off: forces minimal soundscape, no dialogue, "
                        "music N/A"
                    )
                }),
                "video_mode": (["keyframes", "full_clip"], {
                    "default": "keyframes",
                    "tooltip": (
                        "keyframes: send sampled stills (works "
                        "everywhere). full_clip: send the whole clip so "
                        "the model reads real motion, cut timing and "
                        "camera movement instead of inferring them from "
                        "stills. Video-capable providers only (gemini, "
                        "kimi); others warn and fall back to keyframes."
                    )
                }),
                "listen_to_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Send the clip's audio to the VLM so the "
                        "soundscape describes what is actually heard "
                        "instead of what the frames imply. Uses the "
                        "audio input, or the VIDEO's own track when "
                        "nothing is connected. Audio-capable providers "
                        "only (gemini today); ignored elsewhere."
                    )
                }),
                "prompt_profile": (["official", "upgraded", "both_ab"], {
                    "default": "official",
                    "tooltip": (
                        "official: HF-guide format with a trailing 'No...' "
                        "exclusion block. upgraded: same sections + merged "
                        "battle-tested practices (positive assertions, "
                        "~3000-char budget, camera speed words, cut "
                        "re-anchoring, resolved ending). both_ab: two "
                        "separate VLM calls; official -> h3_prompt, "
                        "upgraded -> h3_prompt_b, for A/B testing."
                    )
                }),
            },
            "optional": {
                "video": ("VIDEO", {
                    "tooltip": "Source clip (<Video 1>). Preferred input."
                }),
                "frames": ("IMAGE", {
                    "tooltip": (
                        "Alternative to video: an IMAGE batch (e.g. from "
                        "VHS Load Video). Set fps to match."
                    )
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Frame rate of the frames input (ignored for video)"
                }),
                "audio": ("AUDIO", {
                    "tooltip": (
                        "Soundtrack to describe. Overrides the VIDEO's "
                        "own audio. Needs listen_to_audio on and an "
                        "audio-capable provider."
                    )
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Blank = read the provider's env var: "
                        "ANTHROPIC_API_KEY / GEMINI_API_KEY (or "
                        "GOOGLE_API_KEY) / OPENAI_API_KEY / "
                        "MOONSHOT_API_KEY (kimi) / ZAI_API_KEY (glm) / "
                        "DASHSCOPE_API_KEY (qwen_api)"
                    )
                }),
                "dialogue": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Exact spoken words, if any. Blank = no dialogue "
                        "lines are written."
                    )
                }),
                "duration_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.001,
                    "tooltip": (
                        "0 = use the measured clip duration for shot-time "
                        "validation"
                    )
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Passed to providers that support seeding"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("h3_prompt", "h3_prompt_b", "duration_seconds", "fps",
                    "frame_analysis_json")
    FUNCTION = "generate"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "Analyzes a video + identity reference image with a VLM and "
        "writes a production-ready Minimax H3 REF2VA prompt in the "
        "official six-section format. Selects keyframes on scene cuts "
        "and motion peaks, validates and repairs the VLM output "
        "(shot times, <Subject 1> tagging, wardrobe mentions, "
        "exclusions, 7000-char cap), and retries once with the "
        "validator's error list when needed."
    )

    def generate(
        self,
        reference_image: torch.Tensor,
        subject_name: str,
        subject_wardrobe: str,
        scene_style: str,
        soundscape_type: str,
        vlm_provider: str,
        model: str,
        max_frames_to_analyze: int,
        enable_audio_prompt: bool,
        video_mode: str = "keyframes",
        listen_to_audio: bool = True,
        prompt_profile: str = "official",
        video=None,
        frames: Optional[torch.Tensor] = None,
        fps: float = 24.0,
        audio: Optional[dict] = None,
        api_key: str = "",
        dialogue: str = "",
        duration_override: float = 0.0,
        seed: int = 0,
    ) -> Tuple[str, str, float, int, str]:
        warnings = []

        images, real_fps, duration_source = self._resolve_frames(
            video, frames, fps, warnings
        )
        duration = images.shape[0] / real_fps
        if duration_override > 0:
            duration = float(duration_override)
            duration_source = "override"
            warnings.append(
                f"duration overridden to {duration:.3f}s by widget"
            )

        print(
            f"{LOG_PREFIX} {images.shape[0]} frames @ {real_fps:.3f} fps "
            f"({duration:.3f}s, source: {duration_source})"
        )

        keyframes = select_keyframes(
            images, real_fps, max_frames=max_frames_to_analyze
        )
        print(
            f"{LOG_PREFIX} selected frames {keyframes.indices} "
            f"({keyframes.diff_stats.get('num_scenes', 1)} scene(s))"
        )

        cut_timestamps = [
            round(b / real_fps, 3) for b in keyframes.scene_boundaries if b > 0
        ]

        # Backend first: what it can actually take - a whole clip, an
        # audio track, or only stills - shapes both the payload and the
        # task context built below.
        backend = get_backend(vlm_provider, model, api_key)
        vlm_video = self._resolve_video(
            backend, video_mode, images, real_fps, duration, video, warnings
        )
        vlm_audio = self._resolve_audio(
            backend, audio, video, listen_to_audio, enable_audio_prompt,
            warnings,
        )

        vlm_images = [VLMImage(
            label=(
                "Reference image <Picture 1> - identity and wardrobe "
                f"lock for {subject_name}"
            ),
            jpeg_b64=tensor_to_jpeg_b64(
                reference_image, max_side=REFERENCE_MAX_SIDE
            ),
        )]
        # With the whole clip attached, sampled stills are redundant
        # payload - the model reads the motion straight from the video.
        if vlm_video is None:
            total = len(keyframes.indices)
            for pos, (idx, ts) in enumerate(
                zip(keyframes.indices, keyframes.timestamps), start=1
            ):
                vlm_images.append(VLMImage(
                    label=frame_label(pos, total, ts, idx),
                    jpeg_b64=tensor_to_jpeg_b64(
                        images[idx], max_side=FRAME_MAX_SIDE
                    ),
                ))

        user_context = prompts.build_user_context(
            subject_name=subject_name,
            subject_wardrobe=subject_wardrobe,
            scene_style=scene_style,
            soundscape_type=soundscape_type,
            duration_seconds=duration,
            fps=real_fps,
            frame_timestamps=keyframes.timestamps,
            cut_timestamps=cut_timestamps,
            enable_audio_prompt=enable_audio_prompt,
            dialogue_text=dialogue,
            audio_available=vlm_audio is not None,
            full_clip=vlm_video is not None,
        )

        profiles = (
            ["official", "upgraded"] if prompt_profile == "both_ab"
            else [prompt_profile]
        )
        variants = {}
        prompts_out = []
        last_usage = {}
        for profile in profiles:
            print(
                f"{LOG_PREFIX} generating '{profile}' variant via "
                f"{vlm_provider}"
            )
            ctx = assembler.AssemblyContext(
                subject_name=subject_name,
                subject_wardrobe=subject_wardrobe,
                duration_seconds=duration,
                enable_audio_prompt=enable_audio_prompt,
                profile=profile,
            )
            result, attempts, usage = self._run_variant(
                backend, profile, vlm_images, user_context, ctx, seed,
                vlm_audio, vlm_video,
            )
            last_usage = usage
            prompts_out.append(result.prompt)
            variants[profile] = {
                "attempts": attempts,
                "usage": usage,
                "applied_fixes": result.applied_fixes,
                "warnings": result.warnings,
                "unresolved_errors": result.retry_errors,
                "char_count": result.char_count,
                "detailed_word_count": result.detailed_word_count,
            }
            warnings.extend(f"[{profile}] {w}" for w in result.warnings)
            warnings.extend(
                f"[{profile}] unresolved after retry: {e}"
                for e in result.retry_errors
            )

        for w in warnings:
            print(f"{LOG_PREFIX} warning: {w}")

        analysis = {
            "selected_frame_indices": keyframes.indices,
            "timestamps": keyframes.timestamps,
            "scene_boundaries": keyframes.scene_boundaries,
            "cut_timestamps": cut_timestamps,
            "diff_stats": keyframes.diff_stats,
            "detection_method": keyframes.method,
            "provider": vlm_provider,
            "model": last_usage.get("model", model),
            "profile_mode": prompt_profile,
            "video_mode": "full_clip" if vlm_video else "keyframes",
            "video_mb": round(vlm_video.size_mb, 3) if vlm_video else 0.0,
            "video_source_reused": (
                vlm_video.reused_source if vlm_video else False
            ),
            "audio_sent": vlm_audio is not None,
            "audio_seconds": (
                round(vlm_audio.duration_seconds, 3) if vlm_audio else 0.0
            ),
            "variants": variants,
            "warnings": warnings,
            "duration_source": duration_source,
        }

        return (
            prompts_out[0],
            prompts_out[1] if len(prompts_out) > 1 else "",
            round(duration, 3),
            int(round(real_fps)),
            json.dumps(analysis, indent=2),
        )

    def _run_variant(
        self, backend, profile: str, vlm_images, user_context: str,
        ctx: "assembler.AssemblyContext", seed: int,
        vlm_audio: Optional[VLMAudio] = None,
        vlm_video: Optional[VLMVideo] = None,
    ):
        """One profile's generate -> assemble -> retry loop."""
        system = prompts.get_system_prompt(profile)
        attempts = []
        result = None
        usage = {}
        raw_text = ""
        for attempt in range(MAX_RETRIES + 1):
            if attempt == 0:
                prompt_text = user_context
            else:
                print(
                    f"{LOG_PREFIX} [{profile}] retrying with "
                    f"{len(result.retry_errors)} validation error(s)"
                )
                prompt_text = (
                    user_context
                    + "\n\nYOUR PREVIOUS ATTEMPT:\n" + raw_text
                    + "\n\n" + prompts.build_retry_message(result.retry_errors)
                )
            vlm_result = backend.generate(
                system, vlm_images, prompt_text, seed=seed,
                audio=vlm_audio, video=vlm_video,
            )
            raw_text = vlm_result.text
            usage = vlm_result.usage
            result = assembler.process(raw_text, ctx)
            attempts.append({
                "errors": list(result.retry_errors),
                "warnings": list(result.warnings),
                "chars": result.char_count,
            })
            if not result.retry_errors:
                break
        return result, attempts, usage

    def _resolve_video(
        self, backend, video_mode: str, images: torch.Tensor,
        real_fps: float, duration: float, video, warnings: list,
    ) -> Optional[VLMVideo]:
        """
        Encode the whole clip when the user asked for it and the
        provider can read video. Any obstacle degrades to keyframes
        with a warning rather than failing the run.
        """
        if video_mode != "full_clip":
            return None

        if not getattr(backend, "supports_video", False):
            warnings.append(
                f"provider '{backend.name}' cannot read video; falling "
                "back to keyframes. Use gemini or kimi to send the "
                "whole clip."
            )
            return None

        max_bytes = (
            video_io.UPLOAD_MAX_BYTES if getattr(backend, "_video_upload", None)
            else video_io.INLINE_MAX_BYTES
        )
        try:
            clip = video_io.prepare_video(
                images, real_fps, duration, video=video, max_bytes=max_bytes
            )
        except RuntimeError as exc:
            warnings.append(f"full clip not sent, using keyframes: {exc}")
            return None

        print(
            f"{LOG_PREFIX} sending full clip to {backend.name}: "
            f"{clip.size_mb:.2f} MB, {clip.duration_seconds:.2f}s"
            + (" (source file reused)" if clip.reused_source else " (encoded)")
        )
        return clip

    def _resolve_audio(
        self, backend, audio, video, listen_to_audio: bool,
        enable_audio_prompt: bool, warnings: list,
    ) -> Optional[VLMAudio]:
        """
        Pick the audio track to send: the audio input first, else the
        VIDEO's own. Returns None (with a warning when the user clearly
        asked for audio) if it cannot or should not be sent.
        """
        if not listen_to_audio:
            return None
        if not enable_audio_prompt:
            if audio is not None:
                warnings.append(
                    "audio connected but enable_audio_prompt is off; "
                    "audio not sent"
                )
            return None

        source = "audio input"
        track = audio
        if track is None:
            track = audio_io.audio_from_video(video) if video is not None else None
            source = "video track"
        if track is None:
            if audio is not None:
                warnings.append("the audio input carried no usable waveform")
            return None

        if not getattr(backend, "supports_audio", False):
            warnings.append(
                f"provider '{backend.name}' cannot accept audio; the "
                "soundscape will be inferred from the frames. Use "
                "gemini to describe the real audio."
            )
            return None

        try:
            wav_b64, duration = audio_io.audio_to_wav_b64(track)
        except RuntimeError as exc:
            warnings.append(f"audio not sent: {exc}")
            return None

        print(
            f"{LOG_PREFIX} sending {duration:.2f}s of audio "
            f"({source}) to {backend.name}"
        )
        return VLMAudio(wav_b64=wav_b64, duration_seconds=duration)

    def _resolve_frames(
        self, video, frames, fps: float, warnings: list
    ) -> Tuple[torch.Tensor, float, str]:
        """Resolve (frames_tensor, fps, source) from the two input paths."""
        if video is not None:
            if frames is not None:
                warnings.append(
                    "both video and frames connected; using video"
                )
            try:
                components = video.get_components()
            except AttributeError as exc:
                raise RuntimeError(
                    "The video input is not a ComfyUI VIDEO object. "
                    "Connect a Load Video node, or use the frames input."
                ) from exc
            images = components.images
            frame_rate = components.frame_rate
            real_fps = (
                float(frame_rate) if isinstance(frame_rate, Fraction)
                else float(frame_rate or 24.0)
            )
            return images, real_fps, "video"

        if frames is not None:
            if frames.dim() != 4 or frames.shape[0] < 1:
                raise RuntimeError(
                    "frames must be a non-empty IMAGE batch (B, H, W, C)."
                )
            return frames, float(fps), "frames+fps"

        raise RuntimeError(
            "Connect either a VIDEO (video input) or an IMAGE batch "
            "(frames input, with fps set)."
        )


NODE_CLASS_MAPPINGS = {
    "TrentH3AutoPromptGenerator": H3AutoPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentH3AutoPromptGenerator": "H3 Auto Prompt Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
