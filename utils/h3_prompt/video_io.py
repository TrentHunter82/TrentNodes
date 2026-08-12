"""
Clip -> MP4 bytes for providers that read video natively.

Most VLMs only take stills, so the node samples keyframes. Gemini and
Kimi K3 read video directly, which gives them real motion, real cut
timing, and real camera movement instead of inferences drawn from a
handful of frames. This module produces the MP4 those paths need.

Two sources:
- a ComfyUI VIDEO object, whose original file is reused untouched when
  it is already a reasonably sized MP4 (no re-encode, no quality loss)
- an IMAGE batch, encoded with PyAV

PyAV rather than comfy_api's VideoFromComponents so the module carries
no ComfyUI import and stays testable outside the server, matching
audio_io.py.
"""

import io
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
import torch

# Gemini allows larger inline payloads now but still asks for the Files
# API once the whole request passes 20 MB. Staying under that keeps the
# simple inline path correct for every clip this node realistically sees.
INLINE_MAX_BYTES = 18 * 1024 * 1024
UPLOAD_MAX_BYTES = 100 * 1024 * 1024

# Encode ladder: (crf, max_long_side). Later entries are used only when
# an earlier one overshoots the byte budget.
ENCODE_LADDER = ((23, 1280), (28, 960), (32, 640))


@dataclass
class VLMVideo:
    """An encoded clip plus the timing a provider needs to sample it."""
    mp4_bytes: bytes
    duration_seconds: float
    fps: float
    reused_source: bool = False

    @property
    def size_mb(self) -> float:
        return len(self.mp4_bytes) / (1024 * 1024)


def _even(value: int) -> int:
    """H.264 needs even dimensions."""
    return max(2, value - (value % 2))


def encode_frames_to_mp4(
    images: torch.Tensor,
    fps: float,
    crf: int = 23,
    max_long_side: int = 1280,
) -> bytes:
    """
    Encode a (B, H, W, C) [0,1] batch as H.264 MP4 bytes.

    Raises RuntimeError when PyAV is unavailable.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to send a full clip from an IMAGE batch. "
            "Connect a VIDEO input instead, or install av."
        ) from exc

    if images.dim() != 4 or images.shape[0] == 0:
        raise RuntimeError("Expected a non-empty (B, H, W, C) image batch.")

    height, width = int(images.shape[1]), int(images.shape[2])
    scale = min(1.0, max_long_side / max(height, width))
    out_w = _even(int(width * scale))
    out_h = _even(int(height * scale))

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        rate = Fraction(fps).limit_denominator(1000)
        stream = container.add_stream("libx264", rate=rate)
        stream.width = out_w
        stream.height = out_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}

        for frame_tensor in images:
            arr = (
                frame_tensor.detach().cpu().float().clamp(0, 1) * 255.0
            ).to(torch.uint8).numpy()
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr),
                                               format="rgb24")
            if (out_w, out_h) != (width, height):
                frame = frame.reformat(width=out_w, height=out_h)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    return buffer.getvalue()


def _source_mp4_bytes(video, max_bytes: int) -> Optional[bytes]:
    """
    Reuse the VIDEO's own file when it is already an MP4 within budget.
    Returns None when that is not possible, so the caller re-encodes.
    """
    try:
        source = video.get_stream_source()
    except Exception:
        return None

    if isinstance(source, io.BytesIO):
        data = source.getvalue()
        return data if 0 < len(data) <= max_bytes else None

    if not isinstance(source, str) or not os.path.isfile(source):
        return None
    if os.path.splitext(source)[1].lower() not in (".mp4", ".m4v", ".mov"):
        return None
    if os.path.getsize(source) > max_bytes:
        return None
    try:
        with open(source, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def prepare_video(
    images: torch.Tensor,
    fps: float,
    duration: float,
    video=None,
    max_bytes: int = INLINE_MAX_BYTES,
) -> VLMVideo:
    """
    Produce an MP4 for the clip, within `max_bytes`.

    Prefers the VIDEO input's original file; otherwise encodes the frame
    batch, stepping down the quality ladder until it fits.

    Raises RuntimeError when even the smallest rung overshoots.
    """
    if video is not None:
        data = _source_mp4_bytes(video, max_bytes)
        if data:
            return VLMVideo(
                mp4_bytes=data, duration_seconds=duration, fps=fps,
                reused_source=True,
            )

    last_size = 0
    for crf, max_long_side in ENCODE_LADDER:
        data = encode_frames_to_mp4(
            images, fps, crf=crf, max_long_side=max_long_side
        )
        last_size = len(data)
        if last_size <= max_bytes:
            return VLMVideo(
                mp4_bytes=data, duration_seconds=duration, fps=fps
            )

    raise RuntimeError(
        f"The clip is still {last_size / 1024 / 1024:.1f} MB at the "
        f"lowest quality, over the {max_bytes / 1024 / 1024:.0f} MB "
        "budget. Shorten the clip or use keyframes mode."
    )


def sampling_fps(duration: float, source_fps: float,
                 target_frames: int = 60) -> float:
    """
    Frame rate a provider should sample the clip at.

    Gemini defaults to 1 fps, which is far too coarse to place cuts in a
    short action clip. Aim for ~`target_frames` samples over the whole
    clip, never above the source rate and never below 1 fps.
    """
    if duration <= 0:
        return 1.0
    rate = target_frames / duration
    rate = min(rate, float(source_fps), 10.0)
    return round(max(rate, 1.0), 3)
