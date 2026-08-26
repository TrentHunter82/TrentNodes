"""
ComfyUI AUDIO -> base64 WAV for OpenAI input_audio parts.

ComfyUI's AUDIO type is {"waveform": float tensor (B, C, T) in [-1, 1],
"sample_rate": int}. The audio-understanding models want mono 16 kHz;
llama-server decodes the wav itself, but resampling here keeps payloads
small and behavior identical to the A/B harness this pipeline was
validated with.
"""

import base64
import io
import wave
from typing import Tuple

TARGET_RATE = 16000
# Two 30-s analysis windows; beyond that the models mostly narrate
# padding anyway. Long clips get truncated with a report note.
MAX_SECONDS = 60.0


def audio_to_wav_b64(audio: dict, max_seconds: float = MAX_SECONDS) -> Tuple[str, float, bool]:
    """Return (wav_b64, duration_seconds, truncated)."""
    import torch

    waveform = audio["waveform"]
    rate = int(audio["sample_rate"])
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 2:
        mono = waveform.float().mean(dim=0)
    else:
        mono = waveform.float()

    if rate != TARGET_RATE:
        try:
            import torchaudio
            mono = torchaudio.functional.resample(mono, rate, TARGET_RATE)
        except ImportError:
            # linear fallback; fine for captioning purposes
            length = int(round(mono.shape[-1] * TARGET_RATE / rate))
            mono = torch.nn.functional.interpolate(
                mono.view(1, 1, -1), size=length,
                mode="linear", align_corners=False,
            ).view(-1)

    truncated = False
    max_samples = int(max_seconds * TARGET_RATE)
    if mono.shape[-1] > max_samples:
        mono = mono[:max_samples]
        truncated = True

    duration = mono.shape[-1] / TARGET_RATE
    pcm = (mono.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(TARGET_RATE)
        writer.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), duration, truncated
