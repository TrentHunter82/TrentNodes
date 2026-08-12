"""
ComfyUI AUDIO -> base64 WAV for VLM audio input.

ComfyUI AUDIO is {"waveform": [B, C, T] float tensor, "sample_rate": int}
- the same shape VideoComponents.audio carries, so video-embedded audio
and a separately loaded track go through one path.

Encodes 16 kHz mono PCM16 WAV via the stdlib `wave` module: no torchaudio
I/O backend, no ffmpeg, nothing to fail at runtime. 16 kHz mono is what
the audio-capable APIs downsample to anyway, so this costs no fidelity
and keeps the base64 payload small.
"""

import base64
import io
import wave
from typing import Optional, Tuple

import torch

TARGET_SAMPLE_RATE = 16000
# 10 min of 16 kHz mono PCM16 is ~19 MB raw / ~26 MB base64. Well past
# any sane clip for this node; the cap stops a mis-wired input from
# building a payload the API will reject anyway.
MAX_AUDIO_SECONDS = 600.0


def _resample(waveform: torch.Tensor, src_rate: int, dst_rate: int) -> torch.Tensor:
    """Resample a [C, T] float waveform. Prefers torchaudio's sinc filter."""
    if src_rate == dst_rate:
        return waveform
    try:
        import torchaudio
        return torchaudio.functional.resample(waveform, src_rate, dst_rate)
    except Exception:
        # Pure-torch linear fallback; adequate for description tasks
        import torch.nn.functional as F
        new_len = max(1, int(round(waveform.shape[-1] * dst_rate / src_rate)))
        return F.interpolate(
            waveform.unsqueeze(0), size=new_len, mode="linear",
            align_corners=False,
        ).squeeze(0)


def audio_to_wav_b64(
    audio: dict, max_seconds: float = MAX_AUDIO_SECONDS
) -> Tuple[str, float]:
    """
    Encode a ComfyUI AUDIO dict as base64 16 kHz mono WAV.

    Returns (base64_string, duration_seconds).
    Raises RuntimeError when the dict is malformed or silent-empty.
    """
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise RuntimeError(
            "The audio input is not a ComfyUI AUDIO object "
            "({'waveform', 'sample_rate'})."
        )

    waveform = audio["waveform"]
    src_rate = int(audio.get("sample_rate") or TARGET_SAMPLE_RATE)
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        raise RuntimeError("The audio input carries an empty waveform.")

    # [B, C, T] -> [C, T] (first batch item), then mix down to mono
    wf = waveform.detach().cpu().float()
    if wf.dim() == 3:
        wf = wf[0]
    elif wf.dim() == 1:
        wf = wf.unsqueeze(0)
    if wf.dim() != 2:
        raise RuntimeError(
            f"Expected an audio waveform of [B, C, T] or [C, T]; got "
            f"shape {tuple(waveform.shape)}."
        )
    if wf.shape[0] > 1:
        wf = wf.mean(dim=0, keepdim=True)

    wf = _resample(wf, src_rate, TARGET_SAMPLE_RATE)

    max_samples = int(max_seconds * TARGET_SAMPLE_RATE)
    if wf.shape[-1] > max_samples:
        wf = wf[..., :max_samples]

    duration = wf.shape[-1] / TARGET_SAMPLE_RATE
    pcm = (wf.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(TARGET_SAMPLE_RATE)
        handle.writeframes(pcm)

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), duration


def audio_from_video(video) -> Optional[dict]:
    """
    Pull the embedded AUDIO dict off a ComfyUI VIDEO object, or None when
    the clip is silent or the components cannot be read.
    """
    try:
        components = video.get_components()
    except Exception:
        return None
    audio = getattr(components, "audio", None)
    if not isinstance(audio, dict) or "waveform" not in audio:
        return None
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        return None
    return audio
