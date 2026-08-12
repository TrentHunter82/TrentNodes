"""
Unit tests for H3 audio encoding and the node's audio routing.
CPU-only, no ComfyUI server, no network. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_audio.py
"""

import base64
import io
import json
import os
import sys
import types
import wave

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils", "utils.h3_prompt"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402

from TrentNodes.nodes import h3_auto_prompt  # noqa: E402
from TrentNodes.utils.h3_prompt import audio_io  # noqa: E402
from TrentNodes.utils.h3_prompt.backends import VLMResult  # noqa: E402

# Reuse the canned prompt body from the node test
sys.path.insert(0, os.path.join(PKG, "tests"))
from test_h3_node import CANNED  # noqa: E402


def _tone(seconds=1.0, sample_rate=44100, channels=2):
    t = torch.linspace(0, seconds, int(seconds * sample_rate))
    wave_data = 0.5 * torch.sin(2 * torch.pi * 440.0 * t)
    return {
        "waveform": wave_data.expand(channels, -1).unsqueeze(0).clone(),
        "sample_rate": sample_rate,
    }


def test_encode_resamples_to_16k_mono():
    b64, duration = audio_io.audio_to_wav_b64(_tone(2.0))
    assert abs(duration - 2.0) < 0.01, duration

    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16000
        assert handle.getsampwidth() == 2
        assert abs(handle.getnframes() - 32000) < 200


def test_encode_accepts_2d_waveform():
    audio = _tone(0.5)
    audio["waveform"] = audio["waveform"][0]  # [C, T]
    _b64, duration = audio_io.audio_to_wav_b64(audio)
    assert abs(duration - 0.5) < 0.02


def test_encode_truncates_to_cap():
    _b64, duration = audio_io.audio_to_wav_b64(_tone(3.0), max_seconds=1.0)
    assert abs(duration - 1.0) < 0.01


def test_encode_rejects_bad_input():
    for bad in ({}, {"waveform": None}, {"waveform": torch.zeros(0)}):
        try:
            audio_io.audio_to_wav_b64(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"expected RuntimeError for {bad}")


def test_audio_from_video_handles_silent_and_broken():
    class Silent:
        def get_components(self):
            return types.SimpleNamespace(audio=None)

    class Broken:
        def get_components(self):
            raise ValueError("no container")

    class WithAudio:
        def get_components(self):
            return types.SimpleNamespace(audio=_tone(0.25))

    assert audio_io.audio_from_video(Silent()) is None
    assert audio_io.audio_from_video(Broken()) is None
    assert audio_io.audio_from_video(WithAudio()) is not None


class AudioFake:
    """Backend that records whether it received an audio track."""

    def __init__(self, name="gemini", supports_audio=True):
        self.name = name
        self.supports_audio = supports_audio
        self.received = []

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None):
        self.received.append(audio)
        if audio is not None:
            assert "AUDIO TRACK IS ATTACHED" in user_text
        return VLMResult(text=CANNED, usage={"model": "fake-model"})


def _run(fake, **overrides):
    original = h3_auto_prompt.get_backend
    h3_auto_prompt.get_backend = lambda *a, **k: fake
    try:
        kwargs = dict(
            reference_image=torch.rand((1, 32, 32, 3)),
            subject_name="Aria Voss",
            subject_wardrobe="charcoal utility jacket, black cargo pants",
            scene_style="gritty thriller",
            soundscape_type="ambient",
            vlm_provider="gemini",
            model="auto",
            max_frames_to_analyze=3,
            enable_audio_prompt=True,
            frames=torch.rand((12, 32, 32, 3)),
            fps=12.0,
        )
        kwargs.update(overrides)
        return h3_auto_prompt.H3AutoPromptGenerator().generate(**kwargs)
    finally:
        h3_auto_prompt.get_backend = original


def test_audio_reaches_capable_backend():
    fake = AudioFake()
    _p, _b, _d, _f, analysis_json = _run(fake, audio=_tone(1.0))
    assert fake.received[0] is not None
    assert abs(fake.received[0].duration_seconds - 1.0) < 0.02
    analysis = json.loads(analysis_json)
    assert analysis["audio_sent"] is True
    assert analysis["audio_seconds"] > 0


def test_audio_dropped_for_incapable_backend():
    fake = AudioFake(name="anthropic", supports_audio=False)
    _p, _b, _d, _f, analysis_json = _run(fake, audio=_tone(1.0))
    assert fake.received[0] is None
    analysis = json.loads(analysis_json)
    assert analysis["audio_sent"] is False
    assert any("cannot accept audio" in w for w in analysis["warnings"])


def test_listen_toggle_off():
    fake = AudioFake()
    _p, _b, _d, _f, analysis_json = _run(
        fake, audio=_tone(1.0), listen_to_audio=False
    )
    assert fake.received[0] is None
    assert json.loads(analysis_json)["audio_sent"] is False


def test_no_audio_connected_is_silent_path():
    fake = AudioFake()
    _p, _b, _d, _f, analysis_json = _run(fake)
    assert fake.received[0] is None
    analysis = json.loads(analysis_json)
    assert analysis["audio_sent"] is False
    assert not any("cannot accept audio" in w for w in analysis["warnings"])


def test_gemini_registered():
    from TrentNodes.utils.h3_prompt import backends

    assert backends.DEFAULT_MODELS["gemini"] == "gemini-3.6-flash"
    assert backends.GeminiBackend.supports_audio is True
    assert backends.AnthropicBackend.supports_audio is False

    saved = {
        k: os.environ.pop(k, None)
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    }
    try:
        backends.resolve_api_key("", "gemini")
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc) and "GOOGLE_API_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing key")
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value

    # GOOGLE_API_KEY alone satisfies the gemini fallback
    os.environ["GOOGLE_API_KEY"] = "from-env"
    try:
        assert backends.resolve_api_key("", "gemini") == "from-env"
    finally:
        if saved.get("GOOGLE_API_KEY") is None:
            os.environ.pop("GOOGLE_API_KEY", None)
        else:
            os.environ["GOOGLE_API_KEY"] = saved["GOOGLE_API_KEY"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All audio tests passed.")
