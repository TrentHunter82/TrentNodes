"""
Unit tests for the Whisper transcription wrapper.
CPU-only, no ComfyUI server, no network, no model download. Run from the
ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_transcribe_lyrics.py
"""

import json
import os
import sys
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402

from TrentNodes.utils import whisper_wrapper as ww  # noqa: E402


class _FakeTokenizer:
    """Stands in for WhisperTokenizer.decode in the short-form path."""

    def __init__(self, text):
        self._text = text

    def decode(self, sequence, **kwargs):
        return self._text


class _FakeProcessor:
    def __init__(self, text):
        self.tokenizer = _FakeTokenizer(text)


def test_prepare_waveform_stereo_resample():
    # 2 s of stereo 44.1 kHz in ComfyUI's (B, C, S) layout
    wave = torch.rand(1, 2, 88200) * 2 - 1
    out = ww.prepare_waveform({"waveform": wave, "sample_rate": 44100})
    assert out.dim() == 1, out.shape
    assert out.dtype == torch.float32
    # 2 s at 16 kHz, allow a few samples of resampler slack
    assert abs(out.numel() - 32000) < 50, out.numel()
    print("ok  prepare_waveform stereo 44.1k -> mono 16k")


def test_prepare_waveform_passthrough_and_errors():
    wave = torch.zeros(1, 1, 16000)
    out = ww.prepare_waveform({"waveform": wave, "sample_rate": 16000})
    assert out.numel() == 16000

    for bad in ({"sample_rate": 16000}, "not audio"):
        try:
            ww.prepare_waveform(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")

    try:
        ww.prepare_waveform({"waveform": torch.zeros(1, 1, 0),
                             "sample_rate": 16000})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty audio")
    print("ok  prepare_waveform passthrough + error cases")


def test_segments_from_timestamp_tokens():
    raw = (
        "<|startoftranscript|><|en|><|transcribe|>"
        "<|0.00|> Hello there<|2.50|>"
        "<|3.00|> second line<|6.20|><|endoftext|>"
    )
    segs = ww._segments_from_timestamp_tokens(_FakeProcessor(raw), [0])
    assert len(segs) == 2, segs
    assert segs[0] == {"start": 0.0, "end": 2.5, "text": "Hello there"}
    assert segs[1] == {"start": 3.0, "end": 6.2, "text": "second line"}
    print("ok  short-form timestamp parsing")


def test_segments_from_timestamp_tokens_offset():
    raw = "<|0.00|> tail<|1.00|>"
    segs = ww._segments_from_timestamp_tokens(
        _FakeProcessor(raw), [0], offset=30.0
    )
    assert segs[0]["start"] == 30.0 and segs[0]["end"] == 31.0, segs
    print("ok  timestamp offset")


def test_lrc_and_json():
    segs = [
        {"start": 0.0, "end": 3.5, "text": "first"},
        {"start": 65.25, "end": 70.0, "text": "second"},
    ]
    lrc = ww.segments_to_lrc(segs).splitlines()
    assert lrc[0] == "[00:00.00]first", lrc
    assert lrc[1] == "[01:05.25]second", lrc

    parsed = json.loads(ww.segments_to_json(segs))
    assert parsed == segs
    print("ok  LRC + JSON formatting")


def test_dtype_resolution():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    # CPU has no fp16 conv1d kernel, so every request lands on fp32
    for precision in ("auto", "fp16", "bf16", "fp32"):
        assert ww.resolve_dtype(precision, cpu) == torch.float32
    assert ww.resolve_dtype("auto", cuda) == torch.float16
    assert ww.resolve_dtype("bf16", cuda) == torch.bfloat16
    assert ww.resolve_dtype("fp32", cuda) == torch.float32
    print("ok  dtype resolution")


def test_model_choices():
    choices = ww.model_choices()
    assert "openai/whisper-large-v3" in choices
    assert all(isinstance(c, str) for c in choices)
    print(f"ok  model choices ({len(choices)} entries)")


if __name__ == "__main__":
    test_prepare_waveform_stereo_resample()
    test_prepare_waveform_passthrough_and_errors()
    test_segments_from_timestamp_tokens()
    test_segments_from_timestamp_tokens_offset()
    test_lrc_and_json()
    test_dtype_resolution()
    test_model_choices()
    print("\nAll transcribe-lyrics tests passed.")
