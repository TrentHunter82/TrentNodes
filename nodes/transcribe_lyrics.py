"""
Transcribe Lyrics - Whisper speech/lyric transcription for ComfyUI audio.

A drop-in replacement for VRGDG_TranscribeText that survives the
transformers 5.x dtype change, keeps the model cached between runs, and
also returns timestamps (LRC + JSON) for lyric work.
"""

from typing import Any, Dict, Tuple

import torch

from ..utils.whisper_wrapper import (
    LANGUAGE_CHOICES,
    PRECISION_CHOICES,
    load_whisper,
    model_choices,
    prepare_waveform,
    segments_to_json,
    segments_to_lrc,
    transcribe_waveform,
    unload_whisper,
)


class TranscribeLyrics:
    """
    Transcribes an AUDIO input with OpenAI Whisper.

    Audio longer than 30 seconds uses sequential long-form decoding, so
    words are not cut at a hard chunk boundary.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to transcribe"
                }),
                "model": (model_choices(), {
                    "default": "openai/whisper-large-v3",
                    "tooltip": (
                        "Whisper model. 'local:<name>' reads "
                        "ComfyUI/models/whisper/<name>. The rest download "
                        "from Hugging Face on first use. "
                        "large-v3-turbo is much faster, large-v3 is the "
                        "most accurate."
                    )
                }),
                "language": (LANGUAGE_CHOICES, {
                    "default": "auto",
                    "tooltip": (
                        "Language of the audio. 'auto' lets Whisper detect "
                        "it, which is less reliable on sung vocals."
                    )
                }),
                "task": (["transcribe", "translate"], {
                    "default": "transcribe",
                    "tooltip": (
                        "transcribe keeps the source language. "
                        "translate outputs English."
                    )
                }),
                "precision": (PRECISION_CHOICES, {
                    "default": "auto",
                    "tooltip": (
                        "Compute dtype. auto = fp16 on CUDA, fp32 on CPU. "
                        "Use fp32 if you see NaN or empty output."
                    )
                }),
                "beam_size": ("INT", {
                    "default": 1, "min": 1, "max": 10,
                    "tooltip": (
                        "1 = greedy and fast. 5 is more accurate and "
                        "slower."
                    )
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Keep Whisper in VRAM for the next run. Turn off to "
                        "free the VRAM after each transcription."
                    )
                }),
            },
            "optional": {
                "hint_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional context prompt: names, spellings, or a "
                        "style hint. Whisper biases the output toward it. "
                        "On audio under 30 s a prompt disables the "
                        "timestamps."
                    )
                }),
                "condition_on_previous": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Long audio only. Feeds each chunk the previous "
                        "text. Better flow, but it can repeat itself on "
                        "music."
                    )
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("text", "lrc", "segments_json", "duration")

    FUNCTION = "transcribe"
    CATEGORY = "Trent/Audio"
    DESCRIPTION = (
        "Transcribes audio with Whisper. Returns plain text, LRC-timed "
        "lyrics, segment JSON, and the duration in seconds."
    )

    def transcribe(
        self,
        audio: Any,
        model: str,
        language: str,
        task: str,
        precision: str,
        beam_size: int,
        keep_model_loaded: bool,
        hint_prompt: str = "",
        condition_on_previous: bool = False,
    ) -> Tuple[str, str, str, float]:
        waveform = prepare_waveform(audio)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        processor, whisper_model = load_whisper(model, precision, device)

        try:
            result = transcribe_waveform(
                waveform,
                processor,
                whisper_model,
                language=language,
                task=task,
                with_timestamps=True,
                hint_prompt=hint_prompt or "",
                beam_size=beam_size,
                condition_on_previous=condition_on_previous,
            )
        finally:
            if not keep_model_loaded:
                unload_whisper()

        segments = result["segments"]
        return (
            result["text"],
            segments_to_lrc(segments),
            segments_to_json(segments),
            float(result["duration"]),
        )


NODE_CLASS_MAPPINGS = {
    "TranscribeLyrics": TranscribeLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TranscribeLyrics": "Transcribe Lyrics (Whisper)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
