"""Tests for bounded, private H3 Hermes media staging.

CPU-only and network-free. Run from the ComfyUI root with:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_hermes_assets.py
"""

import base64
import hashlib
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
import wave
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("utils", "utils.h3_prompt"):
        module = types.ModuleType(f"TrentNodes.{sub}")
        module.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = module

import torch  # noqa: E402

from TrentNodes.utils.h3_hermes import assets  # noqa: E402
from TrentNodes.utils.h3_prompt import video_io  # noqa: E402


def _request_id() -> str:
    return str(uuid.uuid4())


def _image(value: float = 0.5) -> torch.Tensor:
    """A real ComfyUI IMAGE tensor: [B, H, W, C], float in [0, 1]."""
    image = torch.full((1, 24, 32, 3), value, dtype=torch.float32)
    image[:, 4:20, 8:24, 1] = 1.0 - value
    return image


def _clip(frame_count: int = 8) -> torch.Tensor:
    frames = torch.zeros((frame_count, 32, 48, 3), dtype=torch.float32)
    for index in range(frame_count):
        frames[index, :, :, 0] = index / max(1, frame_count - 1)
        frames[index, 8:24, 8 + index:24 + index, 1] = 1.0
    return frames


def _aliased_non_mp4_clip(container_format: str, expected_brand: bytes) -> bytes:
    """Encode video in a non-MP4 format covered by PyAV's MP4 alias."""
    try:
        av = importlib.import_module("av")
    except ImportError as exc:
        raise unittest.SkipTest(f"PyAV unavailable: {exc}") from exc

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format=container_format)
    try:
        stream = container.add_stream("libx264", rate=4)
        stream.width = 48
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for frame_tensor in _clip(4):
            array = (frame_tensor * 255.0).to(torch.uint8).numpy()
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    payload = buffer.getvalue()
    assert payload[4:8] == b"ftyp"
    assert payload[8:12] == expected_brand
    with av.open(io.BytesIO(payload), mode="r") as readable:
        assert "mp4" in (readable.format.name or "").split(",")
        assert readable.streams.video
    return payload


def _tone(seconds: float = 0.1, sample_rate: int = 16000) -> dict:
    sample_count = int(seconds * sample_rate)
    timeline = torch.arange(sample_count, dtype=torch.float32) / sample_rate
    waveform = 0.25 * torch.sin(2 * torch.pi * 440.0 * timeline)
    return {
        "waveform": waveform.reshape(1, 1, -1),
        "sample_rate": sample_rate,
    }


def _manifest(bundle: assets.StagedAssets) -> dict:
    with bundle.manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_raises(exception_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception_type as exc:
        return exc
    raise AssertionError(f"expected {exception_type.__name__}")


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_real_image_and_audio_stage_with_hashes_and_secret_free_manifest():
    with tempfile.TemporaryDirectory() as temp:
        request_id = _request_id()
        bundle = assets.stage_assets(
            temp,
            request_id=request_id,
            images={2: _image(0.8), 1: _image(0.2)},
            audio=_tone(),
        )
        manifest = _manifest(bundle)

        assert bundle.request_id == request_id
        assert bundle.job_dir == Path(temp).resolve() / "h3_hermes" / request_id
        assert bundle.manifest_path == bundle.job_dir / "manifest.json"
        assert manifest["schema_version"] == "h3_asset_manifest/1.0"
        assert manifest["request_id"] == request_id
        assert [item["asset_id"] for item in manifest["assets"]] == [
            "picture_01",
            "picture_02",
            "reference_audio_01",
        ]
        assert [item["h3_label"] for item in manifest["assets"]] == [
            "<Picture 1>",
            "<Picture 2>",
            "<Audio 1>",
        ]

        assert set(manifest) == {"schema_version", "request_id", "assets"}
        forbidden = {
            "api_key",
            "authorization",
            "base64",
            "credential",
            "password",
            "secret",
            "token",
        }
        assert forbidden.isdisjoint(set(_all_keys(manifest)))

        for item in manifest["assets"]:
            assert set(item) == {
                "asset_id",
                "h3_label",
                "kind",
                "path",
                "intended_jobs",
                "prohibited_transfers",
                "sha256",
                "bytes",
                "mime_type",
            }
            path = Path(item["path"])
            assert path.is_absolute() and path.is_file()
            assert assets.assert_contained(path, bundle.job_dir) == path.resolve()
            payload = path.read_bytes()
            assert item["bytes"] == len(payload)
            assert item["sha256"] == hashlib.sha256(payload).hexdigest()

        pictures = manifest["assets"][:2]
        assert all(item["mime_type"] == "image/jpeg" for item in pictures)
        assert all(Path(item["path"]).read_bytes().startswith(b"\xff\xd8") for item in pictures)
        assert pictures[0]["intended_jobs"] == ["identity", "appearance"]
        assert "motion" in pictures[0]["prohibited_transfers"]

        audio_item = manifest["assets"][2]
        assert audio_item["mime_type"] == "audio/wav"
        with wave.open(audio_item["path"], "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert handle.getsampwidth() == 2

        result = assets.cleanup_assets(bundle, success=True, policy="delete_on_success")
        assert result.retained is False and result.path is None
        assert not bundle.job_dir.exists()


def test_strict_image_gap_rejected_and_non_strict_keeps_physical_slots():
    with tempfile.TemporaryDirectory() as temp:
        rejected_id = _request_id()
        _assert_raises(
            assets.AssetValidationError,
            assets.stage_assets,
            temp,
            request_id=rejected_id,
            images={1: _image(0.1), 3: _image(0.9)},
        )
        assert not (Path(temp) / "h3_hermes" / rejected_id).exists()

        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images={3: _image(0.9), 1: _image(0.1)},
            strict_image_slots=False,
        )
        manifest = _manifest(bundle)
        assert [item["asset_id"] for item in manifest["assets"]] == [
            "picture_01",
            "picture_03",
        ]
        assert [item["h3_label"] for item in manifest["assets"]] == [
            "<Picture 1>",
            "<Picture 3>",
        ]
        assert not (bundle.job_dir / "picture_02.jpg").exists()
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_sequence_image_slots_do_not_compact_none_values():
    with tempfile.TemporaryDirectory() as temp:
        request_id = _request_id()
        _assert_raises(
            assets.AssetValidationError,
            assets.stage_assets,
            temp,
            request_id=request_id,
            images=[_image(0.1), None, _image(0.9)],
        )
        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images=[_image(0.1), None, _image(0.9)],
            strict_image_slots=False,
        )
        assert [item["h3_label"] for item in _manifest(bundle)["assets"]] == [
            "<Picture 1>",
            "<Picture 3>",
        ]
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_keyframe_evidence_is_canonical_ordered_real_jpeg_and_immutable():
    with tempfile.TemporaryDirectory() as temp:
        try:
            bundle = assets.stage_assets(
                temp,
                request_id=_request_id(),
                images={3: _image(0.9), 1: _image(0.1)},
                video_frames=_clip(4),
                video_fps=4.0,
                video_duration=1.0,
                keyframe_images={2: _image(0.75), 1: _image(0.25)},
                audio=_tone(0.05),
                strict_image_slots=False,
            )
        except RuntimeError as exc:
            if "pyav is required" in str(exc).lower():
                raise unittest.SkipTest(f"runtime H.264 encoder unavailable: {exc}")
            raise

        manifest = _manifest(bundle)
        assert [item["asset_id"] for item in manifest["assets"]] == [
            "picture_01",
            "picture_03",
            "reference_video_01",
            "video_01_keyframe_01",
            "video_01_keyframe_02",
            "reference_audio_01",
        ]
        assert [item["h3_label"] for item in manifest["assets"]] == [
            "<Picture 1>",
            "<Picture 3>",
            "<Video 1>",
            "<Video 1 Keyframe 1>",
            "<Video 1 Keyframe 2>",
            "<Audio 1>",
        ]
        assert not (bundle.job_dir / "picture_02.jpg").exists()

        keyframes = manifest["assets"][3:5]
        for index, item in enumerate(keyframes, start=1):
            path = Path(item["path"])
            payload = path.read_bytes()
            assert item == {
                "asset_id": f"video_01_keyframe_{index:02d}",
                "h3_label": f"<Video 1 Keyframe {index}>",
                "kind": "image",
                "path": str(bundle.job_dir / f"video_01_keyframe_{index:02d}.jpg"),
                "intended_jobs": ["visual_evidence", "timestamp_context"],
                "prohibited_transfers": ["sampler", "identity", "audio"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "mime_type": "image/jpeg",
            }
            assert payload.startswith(b"\xff\xd8")
            assert payload.endswith(b"\xff\xd9")

        expected = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        bundle.manifest["assets"][3]["h3_label"] = "<Picture 99>"
        bundle.manifest["assets"][3]["path"] = "/caller/poisoned-keyframe.jpg"
        bundle.manifest["assets"][3]["intended_jobs"].append("sampler")
        assert assets.verify_staged_assets(bundle) is None
        assert assets.verified_manifest_snapshot(bundle) == expected

        result = assets.cleanup_assets(bundle, success=True, policy="delete_on_success")
        assert result.retained is False and result.path is None
        assert not bundle.job_dir.exists()


def test_keyframe_evidence_accepts_an_ordered_sequence():
    with tempfile.TemporaryDirectory() as temp:
        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            keyframe_images=[_image(0.2), _image(0.8)],
        )
        assert [item["h3_label"] for item in _manifest(bundle)["assets"]] == [
            "<Video 1 Keyframe 1>",
            "<Video 1 Keyframe 2>",
        ]
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_keyframe_evidence_slots_and_limits_are_rejected_before_job_writes():
    class IntSlot(int):
        pass

    class DuplicateItems(Mapping):
        def __getitem__(self, key):
            if key != 1:
                raise KeyError(key)
            return _image()

        def __iter__(self):
            return iter((1, 1))

        def __len__(self):
            return 2

    invalid_inputs = (
        _image(),
        (_image(value) for value in (0.2, 0.8)),
        {True: _image()},
        {1.0: _image()},
        {IntSlot(1): _image()},
        {0: _image()},
        {-1: _image()},
        {2: _image()},
        {1: _image(), 3: _image()},
        DuplicateItems(),
    )
    with tempfile.TemporaryDirectory() as temp:
        for keyframe_images in invalid_inputs:
            request_id = _request_id()
            _assert_raises(
                (TypeError, assets.AssetValidationError),
                assets.stage_assets,
                temp,
                request_id=request_id,
                keyframe_images=keyframe_images,
            )
            assert not (Path(temp) / "h3_hermes" / request_id).exists()
            assert not (Path(temp) / "h3_hermes").exists()

        for limits, keyframe_images in (
            (assets.AssetLimits(max_asset_count=1), [_image(0.2), _image(0.8)]),
            (assets.AssetLimits(max_asset_bytes=64), [_image()]),
            (assets.AssetLimits(max_total_bytes=64), [_image()]),
        ):
            request_id = _request_id()
            _assert_raises(
                assets.AssetLimitError,
                assets.stage_assets,
                temp,
                request_id=request_id,
                keyframe_images=keyframe_images,
                limits=limits,
            )
            assert not (Path(temp) / "h3_hermes" / request_id).exists()
            assert not (Path(temp) / "h3_hermes").exists()

        request_id = _request_id()
        _assert_raises(
            assets.AssetValidationError,
            assets.stage_assets,
            temp,
            request_id=request_id,
            keyframe_images=[_image()],
            asset_directives={
                "<Video 1 Keyframe 1>": assets.AssetDirective(("sampler",), ())
            },
        )
        assert not (Path(temp) / "h3_hermes" / request_id).exists()
        assert not (Path(temp) / "h3_hermes").exists()

        original_encoder = assets.imaging.tensor_to_jpeg_b64
        assets.imaging.tensor_to_jpeg_b64 = lambda *_args, **_kwargs: base64.b64encode(
            b"\xff\xd8not-a-real-jpeg\xff\xd9"
        ).decode("ascii")
        try:
            request_id = _request_id()
            _assert_raises(
                assets.AssetValidationError,
                assets.stage_assets,
                temp,
                request_id=request_id,
                keyframe_images=[_image()],
            )
            assert not (Path(temp) / "h3_hermes" / request_id).exists()
            assert not (Path(temp) / "h3_hermes").exists()
        finally:
            assets.imaging.tensor_to_jpeg_b64 = original_encoder


def test_keyframe_evidence_tamper_missing_symlink_and_extra_are_rejected():
    with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
        def stage():
            return assets.stage_assets(
                temp,
                request_id=_request_id(),
                keyframe_images=[_image(0.4)],
            )

        def rejected_after(tamper):
            bundle = stage()
            tamper(bundle)
            _assert_raises(
                assets.AssetIntegrityError,
                assets.verify_staged_assets,
                bundle,
            )
            assets.cleanup_assets(bundle, success=True, policy="delete_on_success")
            assert not bundle.job_dir.exists()

        def tamper(bundle):
            path = bundle.asset_paths[0]
            changed = bytearray(path.read_bytes())
            changed[len(changed) // 2] ^= 1
            path.write_bytes(changed)

        def missing(bundle):
            bundle.asset_paths[0].unlink()

        def symlink(bundle):
            path = bundle.asset_paths[0]
            target = Path(outside) / f"{bundle.request_id}.jpg"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)

        def extra(bundle):
            (bundle.job_dir / "rogue.jpg").write_bytes(b"rogue")

        for mutation in (tamper, missing, symlink, extra):
            rejected_after(mutation)


def test_video_stages_mp4_through_existing_encoder():
    with tempfile.TemporaryDirectory() as temp:
        try:
            bundle = assets.stage_assets(
                temp,
                request_id=_request_id(),
                video_frames=_clip(),
                video_fps=8.0,
                video_duration=1.0,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "pyav is required" in message or (
                "encoder" in message and ("unavailable" in message or "not found" in message)
            ):
                raise unittest.SkipTest(f"runtime H.264 encoder unavailable: {exc}")
            raise

        item = _manifest(bundle)["assets"][0]
        payload = Path(item["path"]).read_bytes()
        assert item["asset_id"] == "reference_video_01"
        assert item["h3_label"] == "<Video 1>"
        assert item["mime_type"] == "video/mp4"
        assert item["kind"] == "video"
        assert payload[:16].find(b"ftyp") > 0
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
        assert item["bytes"] == len(payload)
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_allowlisted_non_symlink_mp4_source_can_be_copy_staged():
    with tempfile.TemporaryDirectory() as temp:
        source_root = Path(temp) / "sources"
        source_root.mkdir()
        source = source_root / "input.mp4"
        try:
            source.write_bytes(video_io.encode_frames_to_mp4(_clip(4), 4.0))
        except RuntimeError as exc:
            if "pyav is required" in str(exc).lower():
                raise unittest.SkipTest(f"runtime H.264 encoder unavailable: {exc}")
            raise

        class SourceVideo:
            def get_stream_source(self):
                return str(source)

        bundle = assets.stage_assets(
            Path(temp) / "stage",
            request_id=_request_id(),
            video_frames=_clip(4),
            video_fps=4.0,
            video_duration=1.0,
            video=SourceVideo(),
            allow_video_source_reuse=True,
            allowed_video_source_roots=(source_root,),
        )
        staged = bundle.job_dir / "reference_video_01.mp4"
        assert staged.read_bytes() == source.read_bytes()
        assert staged.resolve() != source.resolve()
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_quicktime_mov_renamed_mp4_source_is_rejected():
    with tempfile.TemporaryDirectory() as temp:
        source_root = Path(temp) / "sources"
        source_root.mkdir()
        source = source_root / "renamed-quicktime.mp4"
        source.write_bytes(_aliased_non_mp4_clip("mov", b"qt  "))

        class SourceVideo:
            def get_stream_source(self):
                return str(source)

        request_id = _request_id()
        _assert_raises(
            assets.AssetValidationError,
            assets.stage_assets,
            Path(temp) / "stage",
            request_id=request_id,
            video_fps=4.0,
            video_duration=1.0,
            video=SourceVideo(),
            allow_video_source_reuse=True,
            allowed_video_source_roots=(source_root,),
        )
        assert not (Path(temp) / "stage" / "h3_hermes" / request_id).exists()


def test_3gp_renamed_mp4_source_is_rejected_even_with_isom_compatibility():
    with tempfile.TemporaryDirectory() as temp:
        source_root = Path(temp) / "sources"
        source_root.mkdir()
        source = source_root / "renamed-3gp.mp4"
        payload = _aliased_non_mp4_clip("3gp", b"3gp6")
        assert b"isom" in payload[:32]
        source.write_bytes(payload)

        class SourceVideo:
            def get_stream_source(self):
                return str(source)

        request_id = _request_id()
        _assert_raises(
            assets.AssetValidationError,
            assets.stage_assets,
            Path(temp) / "stage",
            request_id=request_id,
            video_fps=4.0,
            video_duration=1.0,
            video=SourceVideo(),
            allow_video_source_reuse=True,
            allowed_video_source_roots=(source_root,),
        )
        assert not (Path(temp) / "stage" / "h3_hermes" / request_id).exists()


def test_malformed_and_mime_spoofed_mp4_sources_are_rejected_without_frames():
    with tempfile.TemporaryDirectory() as temp:
        source_root = Path(temp) / "sources"
        source_root.mkdir()

        class SourceVideo:
            def __init__(self, source):
                self.source = source

            def get_stream_source(self):
                return str(self.source)

        payloads = {
            "malformed": b"not an mp4 container",
            "fake_ftyp": b"Xftypnot-an-mp4-payload",
        }
        for name, payload in payloads.items():
            source = source_root / f"{name}.mp4"
            source.write_bytes(payload)
            request_id = _request_id()
            _assert_raises(
                assets.AssetValidationError,
                assets.stage_assets,
                Path(temp) / "stage",
                request_id=request_id,
                video_fps=4.0,
                video_duration=1.0,
                video=SourceVideo(source),
                allow_video_source_reuse=True,
                allowed_video_source_roots=(source_root,),
            )
            assert not (
                Path(temp) / "stage" / "h3_hermes" / request_id
            ).exists()


def test_mime_spoofed_mp4_source_falls_back_to_frames_instead_of_copying():
    with tempfile.TemporaryDirectory() as temp:
        source_root = Path(temp) / "sources"
        source_root.mkdir()
        source = source_root / "spoofed.mp4"
        source.write_bytes(b"Xftypnot-an-mp4-payload")

        class SourceVideo:
            def get_stream_source(self):
                return str(source)

        try:
            bundle = assets.stage_assets(
                Path(temp) / "stage",
                request_id=_request_id(),
                video_frames=_clip(4),
                video_fps=4.0,
                video_duration=1.0,
                video=SourceVideo(),
                allow_video_source_reuse=True,
                allowed_video_source_roots=(source_root,),
            )
        except RuntimeError as exc:
            if "pyav is required" in str(exc).lower():
                raise unittest.SkipTest(f"runtime H.264 encoder unavailable: {exc}")
            raise

        staged = bundle.job_dir / "reference_video_01.mp4"
        assert staged.read_bytes() != source.read_bytes()
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_video_encoder_output_with_fake_ftyp_is_rejected():
    original_prepare_video = video_io.prepare_video
    video_io.prepare_video = lambda *_args, **_kwargs: types.SimpleNamespace(
        mp4_bytes=b"Xftypnot-an-mp4-payload"
    )
    try:
        with tempfile.TemporaryDirectory() as temp:
            request_id = _request_id()
            _assert_raises(
                assets.AssetValidationError,
                assets.stage_assets,
                temp,
                request_id=request_id,
                video_frames=_clip(4),
                video_fps=4.0,
                video_duration=1.0,
            )
            assert not (Path(temp) / "h3_hermes" / request_id).exists()
    finally:
        video_io.prepare_video = original_prepare_video


def test_asset_count_per_asset_total_and_media_type_limits_leave_no_job():
    with tempfile.TemporaryDirectory() as temp:
        attempts = [
            (
                assets.AssetLimits(max_asset_count=1),
                {"images": {1: _image(0.1), 2: _image(0.9)}},
                assets.AssetLimitError,
            ),
            (
                assets.AssetLimits(max_asset_bytes=64),
                {"images": {1: _image()}},
                assets.AssetLimitError,
            ),
            (
                assets.AssetLimits(max_total_bytes=64),
                {"images": {1: _image()}},
                assets.AssetLimitError,
            ),
            (
                replace(assets.AssetLimits(), valid_image_extensions=(".png",)),
                {"images": {1: _image()}},
                assets.AssetValidationError,
            ),
            (
                replace(assets.AssetLimits(), valid_audio_mime_types=("audio/mpeg",)),
                {"audio": _tone()},
                assets.AssetValidationError,
            ),
            (
                replace(assets.AssetLimits(), valid_video_extensions=(".mov",)),
                {"video_frames": _clip(), "video_fps": 8.0, "video_duration": 1.0},
                assets.AssetValidationError,
            ),
        ]

        for limits, media, expected_error in attempts:
            request_id = _request_id()
            _assert_raises(
                expected_error,
                assets.stage_assets,
                temp,
                request_id=request_id,
                limits=limits,
                **media,
            )
            job_dir = Path(temp) / "h3_hermes" / request_id
            assert not job_dir.exists()
            assert not (job_dir / "manifest.json").exists()

        _assert_raises(ValueError, assets.AssetLimits, max_asset_count=0)
        _assert_raises(ValueError, assets.AssetLimits, max_asset_bytes=-1)
        _assert_raises(ValueError, assets.AssetLimits, max_total_bytes=0)


def test_staging_rollback_preserves_original_base_exception_when_cleanup_fails():
    class ManifestFinalizationFailure(BaseException):
        pass

    class CleanupFailure(BaseException):
        pass

    original_error = ManifestFinalizationFailure("manifest finalization probe")
    original_replace = assets.os.replace
    original_cleanup = assets.cleanup_assets

    def fail_manifest_finalization(*_args, **_kwargs):
        raise original_error

    def fail_cleanup(*_args, **_kwargs):
        raise CleanupFailure("cleanup probe")

    with tempfile.TemporaryDirectory() as temp:
        request_id = _request_id()
        job_dir = Path(temp) / "h3_hermes" / request_id
        assets.os.replace = fail_manifest_finalization
        assets.cleanup_assets = fail_cleanup
        try:
            try:
                assets.stage_assets(
                    temp,
                    request_id=request_id,
                    images={1: _image()},
                )
            except BaseException as caught:
                assert caught is original_error
            else:
                raise AssertionError("expected manifest finalization failure")
        finally:
            assets.cleanup_assets = original_cleanup
            assets.os.replace = original_replace

        assert not job_dir.exists()


def test_containment_and_request_id_traversal_and_symlink_escape_rejected():
    with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
        root = Path(temp).resolve()
        normal = root / "child" / "file.jpg"
        assert assets.assert_contained(normal, root) == normal.resolve()

        escape = root / "escape"
        escape.symlink_to(Path(outside), target_is_directory=True)
        _assert_raises(
            assets.AssetSecurityError,
            assets.assert_contained,
            escape / "stolen.jpg",
            root,
        )
        _assert_raises(
            assets.AssetSecurityError,
            assets.assert_contained,
            root.parent / "outside.jpg",
            root,
        )

        for bad_id in ("../escape", "not-a-uuid", str(uuid.uuid4()) + "/x", ""):
            _assert_raises(
                assets.AssetSecurityError,
                assets.stage_assets,
                root,
                request_id=bad_id,
                images={1: _image()},
            )

        foreign_temp = Path(outside) / "foreign-temp"
        foreign_job = foreign_temp / "h3_hermes" / _request_id()
        foreign_job.mkdir(parents=True)
        _assert_raises(
            assets.AssetSecurityError,
            assets.cleanup_assets,
            foreign_job,
            temp_root=root,
            success=True,
            policy="delete_on_success",
        )
        assert foreign_job.exists()


def test_cleanup_policies_and_expired_retention_pruning():
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as temp:
        successful = assets.stage_assets(
            temp, request_id=_request_id(), images={1: _image(0.1)}
        )
        deleted = assets.cleanup_assets(
            successful,
            success=True,
            policy=assets.CleanupPolicy.DELETE_ON_SUCCESS,
            now=now,
        )
        assert not deleted.retained and deleted.path is None
        assert not successful.job_dir.exists()

        failed = assets.stage_assets(
            temp, request_id=_request_id(), images={1: _image(0.2)}
        )
        failed_result = assets.cleanup_assets(
            failed, success=False, policy="delete_on_success", now=now
        )
        assert failed_result.retained and failed_result.path == failed.job_dir
        assert failed_result.retention_expires_at == "2026-01-03T03:04:05Z"
        assert failed.job_dir.exists()
        assert assets.prune_expired_assets(temp, now=now + timedelta(hours=23)) == ()
        assert failed.job_dir.exists()
        assert assets.prune_expired_assets(temp, now=now + timedelta(hours=25)) == (
            failed.job_dir,
        )
        assert not failed.job_dir.exists()

        bounded = assets.stage_assets(
            temp, request_id=_request_id(), images={1: _image(0.3)}
        )
        bounded_result = assets.cleanup_assets(
            bounded, success=True, policy="retain_24h", now=now
        )
        assert bounded_result.retained and bounded_result.path == bounded.job_dir
        assert bounded_result.retention_expires_at == "2026-01-03T03:04:05Z"

        forever = assets.stage_assets(
            temp, request_id=_request_id(), images={1: _image(0.4)}
        )
        forever_result = assets.cleanup_assets(
            forever, success=True, policy="retain", now=now
        )
        assert forever_result.retained and forever_result.path == forever.job_dir
        assert forever_result.retention_expires_at is None

        pruned = assets.prune_expired_assets(temp, now=now + timedelta(hours=25))
        assert pruned == (bounded.job_dir,)
        assert not bounded.job_dir.exists()
        assert forever.job_dir.exists()
        assets.cleanup_assets(forever, success=True, policy="delete_on_success")


def test_asset_directives_override_real_image_video_and_audio_metadata_only():
    directives = {
        "<Picture 1>": assets.AssetDirective(
            intended_jobs=["first_frame", "appearance", "identity"],
            prohibited_transfers=(),
        ),
        "<Picture 2>": assets.AssetDirective(
            intended_jobs=("last_frame", "continuity"),
            prohibited_transfers=("audio",),
        ),
        "<Video 1>": assets.AssetDirective(
            intended_jobs=("driving_motion", "camera", "timing"),
            prohibited_transfers=("identity", "appearance", "audio"),
        ),
        "<Audio 1>": assets.AssetDirective(
            intended_jobs=("dialogue", "lip_sync", "timing"),
            prohibited_transfers=(),
        ),
    }
    with tempfile.TemporaryDirectory() as temp:
        request_id = _request_id()
        media = {
            "images": {2: _image(0.8), 1: _image(0.2)},
            "video_frames": _clip(4),
            "video_fps": 4.0,
            "video_duration": 1.0,
            "audio": _tone(0.05),
        }
        original_prepare_video = assets.video_io.prepare_video
        try:
            # Encode one real H.264 payload, then feed that exact immutable
            # converter result through both staging paths. Independent x264
            # invocations are not guaranteed to be byte-deterministic, so
            # comparing two fresh encodes made this metadata-only test flaky.
            prepared_video = original_prepare_video(
                media["video_frames"],
                media["video_fps"],
                media["video_duration"],
            )
            assets.video_io.prepare_video = lambda *args, **kwargs: prepared_video
            baseline = assets.stage_assets(
                Path(temp) / "baseline",
                request_id=request_id,
                **media,
            )
            directed = assets.stage_assets(
                Path(temp) / "directed",
                request_id=request_id,
                asset_directives=directives,
                **media,
            )
        except RuntimeError as exc:
            if "pyav is required" in str(exc).lower():
                raise unittest.SkipTest(f"runtime H.264 encoder unavailable: {exc}")
            raise
        finally:
            assets.video_io.prepare_video = original_prepare_video

        baseline_by_label = {
            item["h3_label"]: item for item in _manifest(baseline)["assets"]
        }
        directed_manifest = _manifest(directed)
        directed_by_label = {
            item["h3_label"]: item for item in directed_manifest["assets"]
        }
        assert list(directed_by_label) == [
            "<Picture 1>",
            "<Picture 2>",
            "<Video 1>",
            "<Audio 1>",
        ]
        assert baseline_by_label["<Picture 1>"]["intended_jobs"] == [
            "identity",
            "appearance",
        ]
        assert baseline_by_label["<Video 1>"]["intended_jobs"] == [
            "pose",
            "motion",
            "camera",
            "timing",
        ]
        assert baseline_by_label["<Audio 1>"]["prohibited_transfers"] == [
            "identity",
            "appearance",
            "pose",
            "motion",
        ]

        for label, directive in directives.items():
            baseline_item = baseline_by_label[label]
            directed_item = directed_by_label[label]
            assert directed_item["intended_jobs"] == list(directive.intended_jobs)
            assert directed_item["prohibited_transfers"] == list(
                directive.prohibited_transfers
            )
            assert Path(directed_item["path"]).name == Path(baseline_item["path"]).name
            assert Path(directed_item["path"]) == directed.job_dir / Path(
                baseline_item["path"]
            ).name
            assert Path(directed_item["path"]).read_bytes() == Path(
                baseline_item["path"]
            ).read_bytes()
            assert directed_item["sha256"] == baseline_item["sha256"]
            assert directed_item["bytes"] == baseline_item["bytes"]
            assert directed_item["mime_type"] == baseline_item["mime_type"]

        assets.cleanup_assets(baseline, success=True, policy="delete_on_success")
        assets.cleanup_assets(directed, success=True, policy="delete_on_success")


def test_asset_directive_mapping_types_and_labels_are_rejected_before_writes():
    directive = assets.AssetDirective(("identity",), ())
    with tempfile.TemporaryDirectory() as temp:
        invalid_cases = [
            ([], TypeError),
            ({1: directive}, TypeError),
            ({"<Picture 1>": object()}, TypeError),
            ({"picture_01": directive}, assets.AssetValidationError),
            ({"<Picture 01>": directive}, assets.AssetValidationError),
            ({"<Video 2>": directive}, assets.AssetValidationError),
            ({"<Picture 2>": directive}, assets.AssetValidationError),
            ({"<Picture 1>\n": directive}, assets.AssetValidationError),
            (
                {"<Picture 1>": directive, " <picture 1> ": directive},
                assets.AssetValidationError,
            ),
        ]
        for asset_directives, expected_error in invalid_cases:
            request_id = _request_id()
            _assert_raises(
                expected_error,
                assets.stage_assets,
                temp,
                request_id=request_id,
                images={1: _image()},
                asset_directives=asset_directives,
            )
            assert not (Path(temp) / "h3_hermes" / request_id).exists()

        too_many_directives = {
            f"<Picture {slot}>": directive
            for slot in range(1, assets.MAX_ASSET_DIRECTIVES + 2)
        }
        request_id = _request_id()
        _assert_raises(
            assets.AssetLimitError,
            assets.stage_assets,
            temp,
            request_id=request_id,
            images={1: _image()},
            asset_directives=too_many_directives,
        )
        assert not (Path(temp) / "h3_hermes" / request_id).exists()


def test_asset_directive_values_are_bounded_and_unambiguous():
    valid = assets.AssetDirective(
        intended_jobs=tuple(
            f"job-{index}" for index in range(assets.MAX_ASSET_DIRECTIVE_ITEMS)
        ),
        prohibited_transfers=("x" * assets.MAX_ASSET_DIRECTIVE_VALUE_LENGTH,),
    )
    with tempfile.TemporaryDirectory() as temp:
        accepted = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images={1: _image()},
            asset_directives={"<Picture 1>": valid},
        )
        item = _manifest(accepted)["assets"][0]
        assert item["intended_jobs"] == list(valid.intended_jobs)
        assert item["prohibited_transfers"] == list(valid.prohibited_transfers)
        assets.cleanup_assets(accepted, success=True, policy="delete_on_success")

        invalid_directives = [
            assets.AssetDirective((), ()),
            assets.AssetDirective("identity", ()),
            assets.AssetDirective(("identity",), "pose"),
            assets.AssetDirective((None,), ()),
            assets.AssetDirective(("identity",), (None,)),
            assets.AssetDirective(("   ",), ()),
            assets.AssetDirective(("identity",), ("   ",)),
            assets.AssetDirective(("identity\nadmin",), ()),
            assets.AssetDirective(("identity",), ("motion\u0085override",)),
            assets.AssetDirective(("Identity", " identity "), ()),
            assets.AssetDirective(("identity",), ("Motion", " motion ")),
            assets.AssetDirective(("Pose",), (" pose ",)),
            assets.AssetDirective(
                tuple(
                    f"job-{index}"
                    for index in range(assets.MAX_ASSET_DIRECTIVE_ITEMS + 1)
                ),
                (),
            ),
            assets.AssetDirective(
                ("identity",),
                tuple(
                    f"transfer-{index}"
                    for index in range(assets.MAX_ASSET_DIRECTIVE_ITEMS + 1)
                ),
            ),
            assets.AssetDirective(
                ("x" * (assets.MAX_ASSET_DIRECTIVE_VALUE_LENGTH + 1),),
                (),
            ),
            assets.AssetDirective(
                ("identity",),
                ("x" * (assets.MAX_ASSET_DIRECTIVE_VALUE_LENGTH + 1),),
            ),
        ]
        for directive in invalid_directives:
            request_id = _request_id()
            _assert_raises(
                (TypeError, assets.AssetStagingError),
                assets.stage_assets,
                temp,
                request_id=request_id,
                images={1: _image()},
                asset_directives={"<Picture 1>": directive},
            )
            assert not (Path(temp) / "h3_hermes" / request_id).exists()


def test_manifest_serialization_and_asset_order_are_deterministic():
    with tempfile.TemporaryDirectory() as temp:
        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images={3: _image(0.9), 1: _image(0.1)},
            audio=_tone(0.05),
            strict_image_slots=False,
        )
        raw = bundle.manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
        assert raw.endswith("\n")
        assert raw.index('"schema_version"') < raw.index('"request_id"') < raw.index('"assets"')
        first_asset = raw.index('"asset_id"')
        assert first_asset < raw.index('"h3_label"', first_asset)
        assert [item["asset_id"] for item in manifest["assets"]] == [
            "picture_01",
            "picture_03",
            "reference_audio_01",
        ]
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_verify_staged_assets_uses_detached_immutable_manifest_authority():
    with tempfile.TemporaryDirectory() as temp:
        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images={1: _image(0.2), 2: _image(0.8)},
        )
        trusted_paths = bundle.asset_paths
        assert assets.verify_staged_assets(bundle) is None

        bundle.manifest["schema_version"] = "caller-mutated/9"
        bundle.manifest["request_id"] = _request_id()
        bundle.manifest["assets"].reverse()
        bundle.manifest["assets"][0]["path"] = "/caller/poisoned/path.jpg"
        bundle.manifest["assets"][0]["sha256"] = "0" * 64
        bundle.manifest["assets"][0]["intended_jobs"].append("caller mutation")

        assert bundle.asset_paths == trusted_paths
        assert assets.verify_staged_assets(bundle) is None
        json.dumps(bundle.manifest)
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_verified_manifest_snapshot_is_detached_and_trusted_for_requests():
    with tempfile.TemporaryDirectory() as temp:
        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images={1: _image(0.2)},
        )
        expected = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        bundle.manifest["assets"][0]["path"] = "/caller/poisoned.jpg"
        bundle.manifest["assets"][0]["sha256"] = "0" * 64

        first = assets.verified_manifest_snapshot(bundle)
        assert first == expected
        first["assets"][0]["path"] = "/caller/second-poison.jpg"
        second = assets.verified_manifest_snapshot(bundle)
        assert second == expected
        assert second is not first
        assert second["assets"] is not first["assets"]
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_verify_staged_assets_rejects_manifest_field_and_byte_rewrites():
    with tempfile.TemporaryDirectory() as temp:
        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images={1: _image(0.2), 2: _image(0.8)},
        )
        raw = bundle.manifest_path.read_bytes()

        def mutate_schema(manifest):
            manifest["schema_version"] = "future/9"

        def mutate_request_id(manifest):
            manifest["request_id"] = _request_id()

        def mutate_order(manifest):
            manifest["assets"].reverse()

        def mutate_path(manifest):
            manifest["assets"][0]["path"] = str(bundle.job_dir / "other.jpg")

        def mutate_label(manifest):
            manifest["assets"][0]["h3_label"] = "<Picture 9>"

        def mutate_kind(manifest):
            manifest["assets"][0]["kind"] = "video"

        def mutate_mime(manifest):
            manifest["assets"][0]["mime_type"] = "image/png"

        def mutate_bytes(manifest):
            manifest["assets"][0]["bytes"] += 1

        def mutate_sha(manifest):
            manifest["assets"][0]["sha256"] = "0" * 64

        def mutate_directives(manifest):
            manifest["assets"][0]["intended_jobs"].append("motion")

        for field, mutate in (
            ("schema", mutate_schema),
            ("request_id", mutate_request_id),
            ("order", mutate_order),
            ("path", mutate_path),
            ("label", mutate_label),
            ("kind", mutate_kind),
            ("mime", mutate_mime),
            ("bytes", mutate_bytes),
            ("sha", mutate_sha),
            ("directives", mutate_directives),
        ):
            manifest = json.loads(raw)
            mutate(manifest)
            bundle.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with unittest.TestCase().subTest(field=field):
                error = _assert_raises(
                    assets.AssetIntegrityError,
                    assets.verify_staged_assets,
                    bundle,
                )
                assert str(error) == "Staged asset integrity verification failed."
            bundle.manifest_path.write_bytes(raw)
            assert assets.verify_staged_assets(bundle) is None

        bundle.manifest_path.write_bytes(
            json.dumps(json.loads(raw), ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        _assert_raises(
            assets.AssetIntegrityError,
            assets.verify_staged_assets,
            bundle,
        )
        bundle.manifest_path.write_bytes(raw)
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


def test_verify_staged_assets_rejects_file_tamper_missing_symlinks_and_extras():
    with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
        def stage():
            return assets.stage_assets(
                temp,
                request_id=_request_id(),
                images={1: _image(0.2)},
            )

        def rejected_after(tamper):
            bundle = stage()
            tamper(bundle)
            _assert_raises(
                assets.AssetIntegrityError,
                assets.verify_staged_assets,
                bundle,
            )
            assets.cleanup_assets(bundle, success=True, policy="delete_on_success")

        def change_hash(bundle):
            path = bundle.asset_paths[0]
            changed = bytearray(path.read_bytes())
            changed[len(changed) // 2] ^= 1
            path.write_bytes(changed)

        def change_size(bundle):
            path = bundle.asset_paths[0]
            path.write_bytes(path.read_bytes() + b"oversized")

        def remove_asset(bundle):
            bundle.asset_paths[0].unlink()

        def replace_with_same_bytes(bundle):
            path = bundle.asset_paths[0]
            replacement = bundle.job_dir / "replacement.tmp"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)

        def symlink_asset(bundle):
            path = bundle.asset_paths[0]
            outside_asset = Path(outside) / f"{bundle.request_id}.jpg"
            outside_asset.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside_asset)

        def add_extra(bundle):
            (bundle.job_dir / "rogue.jpg").write_bytes(b"rogue")

        def symlink_manifest(bundle):
            outside_manifest = Path(outside) / f"{bundle.request_id}.json"
            outside_manifest.write_bytes(bundle.manifest_path.read_bytes())
            bundle.manifest_path.unlink()
            bundle.manifest_path.symlink_to(outside_manifest)

        for tamper in (
            change_hash,
            change_size,
            remove_asset,
            replace_with_same_bytes,
            symlink_asset,
            add_extra,
            symlink_manifest,
        ):
            rejected_after(tamper)


def test_verify_staged_assets_rejects_mutated_bundle_containment_authority():
    with tempfile.TemporaryDirectory() as temp:
        bundle = assets.stage_assets(
            temp,
            request_id=_request_id(),
            images={1: _image()},
        )
        mutations = (
            replace(bundle, request_id=_request_id()),
            replace(bundle, temp_root=Path(temp).parent),
            replace(bundle, job_dir=bundle.job_dir.parent / _request_id()),
            replace(bundle, manifest_path=bundle.job_dir / "other-manifest.json"),
        )
        for mutated in mutations:
            _assert_raises(
                assets.AssetIntegrityError,
                assets.verify_staged_assets,
                mutated,
            )
        assert assets.verify_staged_assets(bundle) is None
        assets.cleanup_assets(bundle, success=True, policy="delete_on_success")


if __name__ == "__main__":
    passed = 0
    skipped = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"H3 Hermes asset tests passed: {passed}; skipped: {skipped}.")
