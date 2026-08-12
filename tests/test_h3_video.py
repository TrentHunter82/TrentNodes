"""
Unit tests for full-clip video input: encoding, source reuse, provider
capability routing, and the Kimi upload/reference flow. CPU-only, no
network. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_video.py
"""

import io
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
    for sub in ("nodes", "utils", "utils.h3_prompt"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402

from TrentNodes.nodes import h3_auto_prompt  # noqa: E402
from TrentNodes.utils.h3_prompt import prompts, video_io  # noqa: E402
from TrentNodes.utils.h3_prompt.backends import VLMResult  # noqa: E402

sys.path.insert(0, os.path.join(PKG, "tests"))
from test_h3_node import CANNED  # noqa: E402

SCRATCH = os.environ.get("TMPDIR", "/tmp")


def _clip(n=24, h=64, w=64):
    return torch.rand((n, h, w, 3))


def test_encode_roundtrip_and_odd_dims():
    import av

    data = video_io.encode_frames_to_mp4(_clip(12, 138, 247), 12.0)
    assert len(data) > 0

    container = av.open(io.BytesIO(data))
    stream = container.streams.video[0]
    assert stream.codec_context.name == "h264"
    # H.264 needs even dimensions
    assert stream.width % 2 == 0 and stream.height % 2 == 0
    assert stream.width == 246 and stream.height == 138


def test_encode_downscales_to_max_side():
    import av

    data = video_io.encode_frames_to_mp4(
        _clip(6, 800, 1600), 12.0, max_long_side=640
    )
    stream = av.open(io.BytesIO(data)).streams.video[0]
    assert max(stream.width, stream.height) <= 640


def test_encode_rejects_empty_batch():
    for bad in (torch.zeros((0, 32, 32, 3)), torch.zeros((32, 32, 3))):
        try:
            video_io.encode_frames_to_mp4(bad, 12.0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")


def test_prepare_video_encodes_from_frames():
    clip = video_io.prepare_video(_clip(12), 12.0, 1.0)
    assert clip.mp4_bytes[:12].find(b"ftyp") > 0, "not an MP4 container"
    assert clip.reused_source is False
    assert abs(clip.duration_seconds - 1.0) < 1e-6
    assert clip.size_mb > 0


def test_prepare_video_reuses_source_file():
    path = os.path.join(SCRATCH, "h3_test_source.mp4")
    with open(path, "wb") as handle:
        handle.write(video_io.encode_frames_to_mp4(_clip(8), 8.0))

    class SourceVideo:
        def get_stream_source(self):
            return path

    try:
        clip = video_io.prepare_video(
            _clip(8), 8.0, 1.0, video=SourceVideo()
        )
        assert clip.reused_source is True
        with open(path, "rb") as handle:
            assert clip.mp4_bytes == handle.read()
    finally:
        os.unlink(path)


def test_prepare_video_ignores_unusable_source():
    class NoSource:
        def get_stream_source(self):
            raise ValueError("no container")

    clip = video_io.prepare_video(_clip(8), 8.0, 1.0, video=NoSource())
    assert clip.reused_source is False
    assert len(clip.mp4_bytes) > 0


def test_prepare_video_raises_when_over_budget():
    try:
        video_io.prepare_video(_clip(24, 256, 256), 12.0, 2.0, max_bytes=500)
    except RuntimeError as exc:
        assert "lowest quality" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a tiny budget")


def test_sampling_fps_bounds():
    # Short clip: dense sampling, capped at 10 fps
    assert video_io.sampling_fps(6.0, 24.0) == 10.0
    # Long clip: floors at 1 fps
    assert video_io.sampling_fps(300.0, 24.0) == 1.0
    # Never above the source rate
    assert video_io.sampling_fps(6.0, 4.0) == 4.0


def test_full_clip_context_replaces_frame_timestamps():
    common = dict(
        subject_name="Aria Voss", subject_wardrobe="charcoal jacket",
        scene_style="thriller", soundscape_type="ambient",
        duration_seconds=6.0, fps=24.0,
        frame_timestamps=[0.0, 3.0, 5.9], cut_timestamps=[3.0],
    )
    sampled = prompts.build_user_context(**common, full_clip=False)
    whole = prompts.build_user_context(**common, full_clip=True)

    assert "sampled frame timestamps" in sampled
    assert "sampled frame timestamps" not in whole
    assert "COMPLETE SOURCE CLIP IS ATTACHED" in whole
    assert "trust the attached video over it" in whole


class VideoFake:
    """Backend recording what payload it received."""

    def __init__(self, name="gemini", supports_video=True):
        self.name = name
        self.supports_video = supports_video
        self.supports_audio = False
        self.seen = []

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None, video=None):
        self.seen.append({"video": video, "images": len(images),
                          "text": user_text})
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
            max_frames_to_analyze=4,
            enable_audio_prompt=True,
            frames=_clip(24),
            fps=12.0,
        )
        kwargs.update(overrides)
        return h3_auto_prompt.H3AutoPromptGenerator().generate(**kwargs)
    finally:
        h3_auto_prompt.get_backend = original


def test_full_clip_sent_and_keyframes_dropped():
    fake = VideoFake()
    _p, _b, _d, _f, analysis_json = _run(fake, video_mode="full_clip")

    call = fake.seen[0]
    assert call["video"] is not None
    # Only <Picture 1> travels; sampled stills would be redundant
    assert call["images"] == 1
    assert "COMPLETE SOURCE CLIP IS ATTACHED" in call["text"]

    analysis = json.loads(analysis_json)
    assert analysis["video_mode"] == "full_clip"
    assert analysis["video_mb"] > 0


def test_keyframes_mode_unchanged():
    fake = VideoFake()
    _p, _b, _d, _f, analysis_json = _run(fake)

    call = fake.seen[0]
    assert call["video"] is None
    assert call["images"] > 1, "keyframe stills should still be sent"
    assert json.loads(analysis_json)["video_mode"] == "keyframes"


def test_incapable_provider_falls_back_to_keyframes():
    fake = VideoFake(name="anthropic", supports_video=False)
    _p, _b, _d, _f, analysis_json = _run(fake, video_mode="full_clip")

    call = fake.seen[0]
    assert call["video"] is None
    assert call["images"] > 1

    analysis = json.loads(analysis_json)
    assert analysis["video_mode"] == "keyframes"
    assert any("cannot read video" in w for w in analysis["warnings"])


def test_capability_flags():
    from TrentNodes.utils.h3_prompt import backends

    assert backends.GeminiBackend.supports_video is True
    assert backends.AnthropicBackend.supports_video is False
    # Kimi reads video, but only via a Files upload
    assert backends.OPENAI_COMPAT_PROVIDERS["kimi"]["video_upload"] == "moonshot"
    for other in ("openai", "glm", "qwen_api"):
        assert not backends.OPENAI_COMPAT_PROVIDERS[other].get("video_upload")

    kimi = backends.get_backend("kimi", "auto", api_key="dummy")
    assert kimi.supports_video is True
    openai_backend = backends.get_backend("openai", "auto", api_key="dummy")
    assert openai_backend.supports_video is False


def test_kimi_uploads_and_references_ms_uri():
    """The Moonshot flow: files.create -> ms://<id> -> files.delete."""
    from TrentNodes.utils.h3_prompt import backends

    backend = backends.get_backend("kimi", "auto", api_key="dummy")
    calls = {"created": [], "deleted": []}

    class FakeFiles:
        def create(self, file, purpose):
            calls["created"].append({"file": file, "purpose": purpose})
            return types.SimpleNamespace(id="file-abc123")

        def delete(self, file_id):
            calls["deleted"].append(file_id)

    class FakeCompletions:
        def create(self, model, max_tokens, messages, **kwargs):
            calls["messages"] = messages
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=CANNED)
                )],
                usage=types.SimpleNamespace(
                    prompt_tokens=10, completion_tokens=20
                ),
            )

    backend._client = types.SimpleNamespace(
        files=FakeFiles(),
        chat=types.SimpleNamespace(completions=FakeCompletions()),
    )

    clip = video_io.VLMVideo(
        mp4_bytes=b"\x00\x01fake-mp4", duration_seconds=2.0, fps=12.0
    )
    result = backend.generate("system", [], "task text", video=clip)

    assert result.usage["video_sent"] is True
    assert calls["created"][0]["purpose"] == "video"
    filename, buffer, mime = calls["created"][0]["file"]
    assert filename == "clip.mp4" and mime == "video/mp4"
    assert buffer.getvalue() == b"\x00\x01fake-mp4"

    content = calls["messages"][1]["content"]
    video_parts = [c for c in content if c.get("type") == "video_url"]
    assert len(video_parts) == 1
    assert video_parts[0]["video_url"]["url"] == "ms://file-abc123"

    # Uploaded clip is cleaned up rather than left in Moonshot storage
    assert calls["deleted"] == ["file-abc123"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All video tests passed.")
