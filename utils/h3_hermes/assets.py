"""Bounded per-request media staging for the H3 Hermes bridge.

This module deliberately delegates conversion to the established H3 prompt
helpers. It only validates their output, writes a private UUID-scoped job
directory, records hashes in one manifest, and applies one cleanup policy.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import stat
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, Optional

from ..h3_prompt import audio_io, imaging, video_io

MANIFEST_SCHEMA_VERSION = "h3_asset_manifest/1.0"
STAGING_DIRECTORY_NAME = "h3_hermes"
MANIFEST_FILENAME = "manifest.json"
RETENTION_MARKER_FILENAME = ".retention_until"
FAILED_RETENTION_HOURS = 24
MAX_ASSET_DIRECTIVES = 12
MAX_ASSET_DIRECTIVE_ITEMS = 16
MAX_ASSET_DIRECTIVE_VALUE_LENGTH = 128

_PICTURE_LABEL_PATTERN = re.compile(r"<Picture ([1-9][0-9]{0,3})>")
_KEYFRAME_LABEL_PATTERN = re.compile(r"<Video 1 Keyframe ([1-9][0-9]{0,3})>")

# Brands emitted by, or explicitly compatible with, normal FFmpeg MP4 output.
# PyAV/libavformat reports several distinct ISO-BMFF formats through the broad
# ``mov,mp4,m4a,3gp,3g2,mj2`` alias, so that alias alone is not accepted.
_MP4_FILE_TYPE_BRANDS = frozenset(
    {
        b"isom",
        b"iso2",
        b"iso3",
        b"iso4",
        b"iso5",
        b"iso6",
        b"iso7",
        b"iso8",
        b"iso9",
        b"mp41",
        b"mp42",
        b"avc1",
        b"dash",
        b"M4V ",
        b"MSNV",
    }
)


class AssetStagingError(RuntimeError):
    """Base error for media staging and lifecycle failures."""


class AssetValidationError(AssetStagingError):
    """An input or converted media asset violates the staging contract."""


class AssetLimitError(AssetStagingError):
    """An asset count or byte budget was exceeded."""


class AssetSecurityError(AssetStagingError):
    """A path, UUID, or symlink failed the private-media boundary checks."""


class AssetIntegrityError(AssetSecurityError):
    """A finalized staged bundle no longer matches its trusted snapshot."""


_INTEGRITY_ERROR_MESSAGE = "Staged asset integrity verification failed."


class CleanupPolicy(str, Enum):
    DELETE_ON_SUCCESS = "delete_on_success"
    RETAIN_24H = "retain_24h"
    RETAIN = "retain"


@dataclass(frozen=True)
class AssetLimits:
    """Hard local limits and allowlists for one staged request.

    Defaults cover normal H3 reference jobs while bounding a job to twelve
    assets, 32 MiB per asset, and 128 MiB total. Output formats are fixed to
    the formats emitted by the existing TrentNodes conversion helpers.
    """

    max_asset_count: int = 12
    max_asset_bytes: int = 32 * 1024 * 1024
    max_total_bytes: int = 128 * 1024 * 1024
    valid_image_extensions: tuple[str, ...] = (".jpg", ".jpeg")
    valid_video_extensions: tuple[str, ...] = (".mp4",)
    valid_audio_extensions: tuple[str, ...] = (".wav",)
    valid_image_mime_types: tuple[str, ...] = ("image/jpeg",)
    valid_video_mime_types: tuple[str, ...] = ("video/mp4",)
    valid_audio_mime_types: tuple[str, ...] = ("audio/wav",)

    def __post_init__(self) -> None:
        for field_name in (
            "max_asset_count",
            "max_asset_bytes",
            "max_total_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        for field_name in (
            "valid_image_extensions",
            "valid_video_extensions",
            "valid_audio_extensions",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_extensions(getattr(self, field_name), field_name),
            )

        for field_name, prefix in (
            ("valid_image_mime_types", "image/"),
            ("valid_video_mime_types", "video/"),
            ("valid_audio_mime_types", "audio/"),
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_mime_types(
                    getattr(self, field_name), field_name, prefix
                ),
            )


@dataclass(frozen=True)
class AssetDirective:
    """Caller-supplied intent and transfer boundaries for one H3 asset label.

    Values are validated and copied to immutable tuples by :func:`stage_assets`
    after the complete set of staged labels is known. This keeps construction
    lightweight while preventing mutable caller collections from reaching the
    manifest.
    """

    intended_jobs: Sequence[str]
    prohibited_transfers: Sequence[str] = ()


@dataclass(frozen=True)
class _TrustedAssetRecord:
    asset_id: str
    h3_label: str
    kind: str
    path: Path
    intended_jobs: tuple[str, ...]
    prohibited_transfers: tuple[str, ...]
    sha256: str
    bytes: int
    mime_type: str
    device: int
    inode: int


@dataclass(frozen=True)
class _TrustedStagingSnapshot:
    request_id: str
    temp_root: Path
    staging_root: Path
    job_dir: Path
    manifest_path: Path
    manifest_bytes: bytes
    assets: tuple[_TrustedAssetRecord, ...]
    max_asset_count: int
    max_asset_bytes: int
    max_total_bytes: int
    temp_device: int
    temp_inode: int
    staging_device: int
    staging_inode: int
    job_device: int
    job_inode: int
    manifest_device: int
    manifest_inode: int


@dataclass(frozen=True)
class StagedAssets:
    """A finalized manifest and the private directory that contains it."""

    request_id: str
    temp_root: Path
    job_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    _trusted_snapshot: _TrustedStagingSnapshot

    @property
    def asset_paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self._trusted_snapshot.assets)


@dataclass(frozen=True)
class CleanupResult:
    """Observable result of applying a cleanup policy."""

    retained: bool
    path: Optional[Path]
    retention_expires_at: Optional[str]
    policy: CleanupPolicy


@dataclass(frozen=True)
class _PreparedAsset:
    asset_id: str
    h3_label: str
    kind: str
    filename: str
    intended_jobs: tuple[str, ...]
    prohibited_transfers: tuple[str, ...]
    payload: bytes
    mime_type: str


def _is_known_h3_asset_label(label: str) -> bool:
    return (
        label in {"<Video 1>", "<Audio 1>"}
        or _PICTURE_LABEL_PATTERN.fullmatch(label) is not None
    )


def _is_known_staged_asset_label(label: str) -> bool:
    """Accept physical H3 labels plus canonical analysis-only keyframes."""

    return (
        _is_known_h3_asset_label(label)
        or _KEYFRAME_LABEL_PATTERN.fullmatch(label) is not None
    )


def _validate_directive_values(
    values: Any,
    *,
    field_name: str,
    h3_label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(
            f"{field_name} for {h3_label} must be an ordered collection of strings"
        )
    if len(values) > MAX_ASSET_DIRECTIVE_ITEMS:
        raise AssetLimitError(
            f"{field_name} for {h3_label} exceeds the "
            f"{MAX_ASSET_DIRECTIVE_ITEMS}-item limit."
        )
    if not values and not allow_empty:
        raise AssetValidationError(f"{field_name} for {h3_label} cannot be empty.")

    checked: list[str] = []
    equivalent_values: set[str] = set()
    for index, value in enumerate(values, start=1):
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} item {index} for {h3_label} must be a string"
            )
        if not value.strip():
            raise AssetValidationError(
                f"{field_name} item {index} for {h3_label} cannot be blank."
            )
        if len(value) > MAX_ASSET_DIRECTIVE_VALUE_LENGTH:
            raise AssetLimitError(
                f"{field_name} item {index} for {h3_label} exceeds the "
                f"{MAX_ASSET_DIRECTIVE_VALUE_LENGTH}-character limit."
            )
        if any(unicodedata.category(character) == "Cc" for character in value):
            raise AssetValidationError(
                f"{field_name} item {index} for {h3_label} contains a control character."
            )

        equivalent = value.strip().casefold()
        if equivalent in equivalent_values:
            raise AssetValidationError(
                f"{field_name} for {h3_label} contains duplicate-equivalent values."
            )
        equivalent_values.add(equivalent)
        checked.append(value)
    return tuple(checked)


def _apply_asset_directives(
    prepared: Sequence[_PreparedAsset],
    asset_directives: Any,
) -> list[_PreparedAsset]:
    """Validate and apply metadata-only overrides without reordering assets."""

    if asset_directives is None:
        return list(prepared)
    if not isinstance(asset_directives, Mapping):
        raise TypeError("asset_directives must be a mapping keyed by H3 asset label")
    if len(asset_directives) > MAX_ASSET_DIRECTIVES:
        raise AssetLimitError(
            f"asset_directives exceeds the {MAX_ASSET_DIRECTIVES}-directive limit."
        )

    prepared_labels = {asset.h3_label for asset in prepared}
    checked_directives: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    equivalent_labels: set[str] = set()
    for h3_label, directive in asset_directives.items():
        if not isinstance(h3_label, str):
            raise TypeError("asset_directives keys must be H3 label strings")
        if any(unicodedata.category(character) == "Cc" for character in h3_label):
            raise AssetValidationError(
                f"Asset directive label {h3_label!r} contains a control character."
            )

        equivalent_label = h3_label.strip().casefold()
        if equivalent_label in equivalent_labels:
            raise AssetValidationError(
                f"Duplicate-equivalent asset directive label: {h3_label!r}."
            )
        equivalent_labels.add(equivalent_label)
        if not _is_known_h3_asset_label(h3_label):
            raise AssetValidationError(f"Unknown H3 asset directive label: {h3_label!r}.")
        if h3_label not in prepared_labels:
            raise AssetValidationError(
                f"Asset directive label was not staged for this request: {h3_label}."
            )
        if not isinstance(directive, AssetDirective):
            raise TypeError(
                f"asset_directives[{h3_label!r}] must be an AssetDirective"
            )

        intended_jobs = _validate_directive_values(
            directive.intended_jobs,
            field_name="intended_jobs",
            h3_label=h3_label,
            allow_empty=False,
        )
        prohibited_transfers = _validate_directive_values(
            directive.prohibited_transfers,
            field_name="prohibited_transfers",
            h3_label=h3_label,
            allow_empty=True,
        )
        intended_equivalents = {value.strip().casefold() for value in intended_jobs}
        prohibited_equivalents = {
            value.strip().casefold() for value in prohibited_transfers
        }
        if intended_equivalents & prohibited_equivalents:
            raise AssetValidationError(
                f"Asset directive for {h3_label} cannot both intend and prohibit "
                "duplicate-equivalent values."
            )
        checked_directives[h3_label] = (intended_jobs, prohibited_transfers)

    return [
        replace(
            asset,
            intended_jobs=checked_directives[asset.h3_label][0],
            prohibited_transfers=checked_directives[asset.h3_label][1],
        )
        if asset.h3_label in checked_directives
        else asset
        for asset in prepared
    ]


@dataclass(frozen=True)
class _BufferedVideoSource:
    """VIDEO-compatible wrapper around bytes read after source validation."""

    payload: bytes

    def get_stream_source(self) -> io.BytesIO:
        return io.BytesIO(self.payload)


def _normalize_extensions(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a collection, not a string")
    try:
        normalized = {
            str(value).strip().lower()
            for value in values
        }
    except TypeError as exc:
        raise ValueError(f"{field_name} must be an iterable of extensions") from exc
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    for extension in normalized:
        if (
            len(extension) < 2
            or not extension.startswith(".")
            or "/" in extension
            or "\\" in extension
            or "\x00" in extension
        ):
            raise ValueError(f"invalid extension {extension!r} in {field_name}")
    return tuple(sorted(normalized))


def _normalize_mime_types(
    values: Any, field_name: str, expected_prefix: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a collection, not a string")
    try:
        normalized = {
            str(value).strip().lower()
            for value in values
        }
    except TypeError as exc:
        raise ValueError(f"{field_name} must be an iterable of MIME types") from exc
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    for mime_type in normalized:
        if (
            not mime_type.startswith(expected_prefix)
            or mime_type.count("/") != 1
            or any(character.isspace() for character in mime_type)
            or "\x00" in mime_type
        ):
            raise ValueError(f"invalid MIME type {mime_type!r} in {field_name}")
    return tuple(sorted(normalized))


def assert_contained(path: os.PathLike[str] | str, root: os.PathLike[str] | str) -> Path:
    """Resolve ``path`` and assert it remains at or below resolved ``root``.

    Existing symlinks are followed by :meth:`Path.resolve`, including a
    symlink in an intermediate component. A link that escapes the allowlisted
    root is therefore rejected even when the final path does not exist yet.
    Relative paths are interpreted relative to ``root``.
    """

    root_path = Path(root).expanduser()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved_root = root_path.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise AssetSecurityError(
            f"Path escapes the allowlisted directory: {candidate}"
        ) from exc
    return resolved_candidate


def _canonical_request_id(request_id: Any) -> str:
    try:
        parsed = uuid.UUID(str(request_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssetSecurityError("request_id must be a canonical UUID") from exc
    canonical = str(parsed)
    if str(request_id) != canonical:
        raise AssetSecurityError("request_id must be a canonical UUID")
    return canonical


def _normalize_image_slots(images: Any, strict: bool) -> list[tuple[int, Any]]:
    if images is None:
        return []

    if isinstance(images, Mapping):
        raw_items = list(images.items())
    elif isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
        raw_items = list(enumerate(images, start=1))
    else:
        raw_items = [(1, images)]

    normalized: list[tuple[int, Any]] = []
    for slot, image in raw_items:
        if isinstance(slot, bool) or not isinstance(slot, int) or slot <= 0:
            raise AssetValidationError(
                f"Image slots must be positive integers; got {slot!r}."
            )
        if slot > 9999:
            raise AssetValidationError("Image slot numbers cannot exceed 9999.")
        if image is not None:
            normalized.append((slot, image))

    normalized.sort(key=lambda item: item[0])
    if strict and normalized:
        actual = [slot for slot, _image in normalized]
        expected = list(range(1, actual[-1] + 1))
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            raise AssetValidationError(
                "Strict image slots cannot contain gaps; missing physical "
                f"slot(s): {', '.join(str(slot) for slot in missing)}."
            )
    return normalized


def _normalize_keyframe_slots(keyframe_images: Any) -> list[tuple[int, Any]]:
    """Return sorted, contiguous analysis-evidence slots starting at one."""

    if keyframe_images is None:
        return []
    if isinstance(keyframe_images, Mapping):
        raw_items = list(keyframe_images.items())
    elif isinstance(keyframe_images, Sequence) and not isinstance(
        keyframe_images, (str, bytes)
    ):
        raw_items = list(enumerate(keyframe_images, start=1))
    else:
        raise TypeError("keyframe_images must be a mapping or ordered sequence")

    normalized: list[tuple[int, Any]] = []
    seen_slots: set[int] = set()
    for slot, image in raw_items:
        if type(slot) is not int or slot <= 0:
            raise AssetValidationError(
                "Keyframe evidence slots must be built-in positive integers; "
                f"got {slot!r}."
            )
        if slot > 9999:
            raise AssetValidationError(
                "Keyframe evidence slot numbers cannot exceed 9999."
            )
        if slot in seen_slots:
            raise AssetValidationError(
                f"Keyframe evidence slots cannot contain duplicates; got slot {slot}."
            )
        if image is None:
            raise AssetValidationError(
                f"Keyframe evidence slot {slot} cannot be empty."
            )
        seen_slots.add(slot)
        normalized.append((slot, image))

    normalized.sort(key=lambda item: item[0])
    actual = [slot for slot, _image in normalized]
    expected = list(range(1, len(normalized) + 1))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise AssetValidationError(
            "Keyframe evidence slots must be contiguous from 1; missing slot(s): "
            f"{', '.join(str(slot) for slot in missing)}."
        )
    return normalized


def _allowed_type(
    kind: str,
    extension: str,
    mime_type: str,
    limits: AssetLimits,
) -> None:
    extensions = getattr(limits, f"valid_{kind}_extensions")
    mime_types = getattr(limits, f"valid_{kind}_mime_types")
    if extension.lower() not in extensions:
        raise AssetValidationError(
            f"The {kind} extension {extension!r} is not allowlisted."
        )
    if mime_type.lower() not in mime_types:
        raise AssetValidationError(
            f"The {kind} MIME type {mime_type!r} is not allowlisted."
        )


def _validate_filename(filename: str, expected_extension: str) -> None:
    path = Path(filename)
    if (
        not filename
        or path.name != filename
        or path.is_absolute()
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or path.suffix.lower() != expected_extension
    ):
        raise AssetSecurityError(f"Unsafe staged asset filename: {filename!r}")


def _decode_base64(encoded: str, description: str) -> bytes:
    if not isinstance(encoded, str):
        raise AssetValidationError(f"{description} encoder did not return base64 text.")
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AssetValidationError(
            f"{description} encoder returned invalid base64."
        ) from exc


def _bounded_jpeg_payload(
    image: Any,
    *,
    description: str,
    max_side: int,
) -> bytes:
    try:
        encoded = imaging.tensor_to_jpeg_b64(image, max_side=max_side)
    except Exception as exc:
        raise AssetValidationError(
            f"Could not convert {description} to JPEG: {exc}"
        ) from exc
    payload = _decode_base64(encoded, description)
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise AssetValidationError(
            f"{description} did not produce a complete JPEG payload."
        )
    try:
        with imaging.Image.open(io.BytesIO(payload)) as decoded:
            if decoded.format != "JPEG" or decoded.width <= 0 or decoded.height <= 0:
                raise AssetValidationError(
                    f"{description} did not produce a readable JPEG image."
                )
            decoded.verify()
    except AssetValidationError:
        raise
    except Exception as exc:
        raise AssetValidationError(
            f"{description} did not produce a readable JPEG image: {exc}"
        ) from exc
    return payload


def _prepare_image(slot: int, image: Any, limits: AssetLimits) -> _PreparedAsset:
    filename = f"picture_{slot:02d}.jpg"
    _validate_filename(filename, ".jpg")
    _allowed_type("image", ".jpg", "image/jpeg", limits)
    payload = _bounded_jpeg_payload(
        image,
        description=f"Physical image slot {slot}",
        max_side=imaging.REFERENCE_MAX_SIDE,
    )
    return _PreparedAsset(
        asset_id=f"picture_{slot:02d}",
        h3_label=f"<Picture {slot}>",
        kind="image",
        filename=filename,
        intended_jobs=("identity", "appearance"),
        prohibited_transfers=("pose", "motion", "audio"),
        payload=payload,
        mime_type="image/jpeg",
    )


def _prepare_keyframe(slot: int, image: Any, limits: AssetLimits) -> _PreparedAsset:
    filename = f"video_01_keyframe_{slot:02d}.jpg"
    _validate_filename(filename, ".jpg")
    _allowed_type("image", ".jpg", "image/jpeg", limits)
    payload = _bounded_jpeg_payload(
        image,
        description=f"Video 1 keyframe evidence slot {slot}",
        max_side=imaging.FRAME_MAX_SIDE,
    )
    return _PreparedAsset(
        asset_id=f"video_01_keyframe_{slot:02d}",
        h3_label=f"<Video 1 Keyframe {slot}>",
        kind="image",
        filename=filename,
        intended_jobs=("visual_evidence", "timestamp_context"),
        prohibited_transfers=("sampler", "identity", "audio"),
        payload=payload,
        mime_type="image/jpeg",
    )


def _is_contained_in_any(path: Path, roots: Sequence[os.PathLike[str] | str]) -> bool:
    for root in roots:
        root_path = Path(root).expanduser()
        try:
            resolved_root = root_path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not resolved_root.is_dir():
            continue
        try:
            path.relative_to(resolved_root)
        except ValueError:
            continue
        return True
    return False


def _validate_mp4_file_type_box(payload: bytes, description: str) -> None:
    """Require a structurally bounded ``ftyp`` box declaring MP4 branding."""

    payload_size = len(payload)
    offset = 0
    while offset < payload_size:
        remaining = payload_size - offset
        if remaining < 8:
            raise AssetValidationError(
                f"{description} has a truncated ISO-BMFF box header."
            )

        box_size = int.from_bytes(payload[offset:offset + 4], "big")
        box_type = payload[offset + 4:offset + 8]
        header_size = 8
        if box_size == 1:
            if remaining < 16:
                raise AssetValidationError(
                    f"{description} has a truncated extended ISO-BMFF box header."
                )
            box_size = int.from_bytes(payload[offset + 8:offset + 16], "big")
            header_size = 16
        elif box_size == 0:
            box_size = remaining

        if box_size < header_size or box_size > remaining:
            raise AssetValidationError(
                f"{description} has an invalid ISO-BMFF box boundary."
            )

        if box_type == b"ftyp":
            body_start = offset + header_size
            body_size = box_size - header_size
            if body_size < 8 or (body_size - 8) % 4:
                raise AssetValidationError(
                    f"{description} has a malformed ISO-BMFF file-type box."
                )

            major_brand = payload[body_start:body_start + 4]
            compatible_start = body_start + 8
            compatible_brands = {
                payload[index:index + 4]
                for index in range(compatible_start, offset + box_size, 4)
            }
            if major_brand == b"qt  ":
                raise AssetValidationError(
                    f"{description} is a QuickTime/MOV file, not an MP4 file."
                )
            if major_brand.lower().startswith(b"3g"):
                raise AssetValidationError(
                    f"{description} is a 3GPP/3GPP2 file, not an MP4 file."
                )
            if not ({major_brand} | compatible_brands) & _MP4_FILE_TYPE_BRANDS:
                raise AssetValidationError(
                    f"{description} does not declare an allowlisted MP4 file-type brand."
                )
            return

        offset += box_size

    raise AssetValidationError(
        f"{description} does not contain an ISO-BMFF file-type box."
    )


def _validate_mp4_video_payload(payload: Any, description: str) -> None:
    """Require MP4 file-type branding and a readable container video stream."""

    if not isinstance(payload, bytes) or not payload:
        raise AssetValidationError(f"{description} is not a non-empty byte payload.")
    _validate_mp4_file_type_box(payload, description)
    try:
        av = importlib.import_module("av")
    except ImportError as exc:
        raise AssetValidationError(
            f"PyAV is required to validate {description} as an MP4 container."
        ) from exc

    try:
        with av.open(io.BytesIO(payload), mode="r") as container:
            format_names = {
                name.strip().lower()
                for name in (container.format.name or "").split(",")
                if name.strip()
            }
            if "mp4" not in format_names:
                raise AssetValidationError(
                    f"{description} is not an MP4/ISO-BMFF container."
                )
            if not container.streams.video:
                raise AssetValidationError(
                    f"{description} does not contain a video stream."
                )
    except AssetValidationError:
        raise
    except Exception as exc:
        raise AssetValidationError(
            f"{description} is not a readable MP4/ISO-BMFF video container: {exc}"
        ) from exc


def _buffered_allowlisted_video_source(
    video: Any,
    *,
    allow_reuse: bool,
    allowed_roots: Sequence[os.PathLike[str] | str],
    limits: AssetLimits,
) -> Optional[_BufferedVideoSource]:
    """Return validated source bytes, or ``None`` to select re-encoding."""

    if not allow_reuse or video is None or not allowed_roots:
        return None
    try:
        source_value = video.get_stream_source()
    except Exception:
        return None
    if not isinstance(source_value, (str, os.PathLike)):
        return None

    source = Path(source_value).expanduser()
    if source.is_symlink():
        return None
    try:
        resolved = source.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file() or not _is_contained_in_any(resolved, allowed_roots):
        return None
    if resolved.suffix.lower() != ".mp4":
        return None
    _allowed_type("video", ".mp4", "video/mp4", limits)
    try:
        size = resolved.stat().st_size
        if size <= 0 or size > limits.max_asset_bytes:
            return None
        with resolved.open("rb") as handle:
            payload = handle.read(limits.max_asset_bytes + 1)
    except OSError:
        return None
    if len(payload) != size or len(payload) > limits.max_asset_bytes:
        return None
    _validate_mp4_video_payload(payload, f"Video copy source {resolved}")
    return _BufferedVideoSource(payload)


def _positive_finite(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AssetValidationError(f"{field_name} must be a positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise AssetValidationError(f"{field_name} must be a positive number.")
    return number


def _prepare_video(
    frames: Any,
    fps: Any,
    duration: Any,
    video: Any,
    *,
    allow_source_reuse: bool,
    allowed_source_roots: Sequence[os.PathLike[str] | str],
    limits: AssetLimits,
) -> _PreparedAsset:
    filename = "reference_video_01.mp4"
    _validate_filename(filename, ".mp4")
    _allowed_type("video", ".mp4", "video/mp4", limits)
    checked_fps = _positive_finite(fps, "video_fps")
    checked_duration = _positive_finite(duration, "video_duration")

    try:
        reusable = _buffered_allowlisted_video_source(
            video,
            allow_reuse=allow_source_reuse,
            allowed_roots=allowed_source_roots,
            limits=limits,
        )
    except AssetValidationError:
        if frames is None:
            raise
        reusable = None
    if frames is None and reusable is None:
        raise AssetValidationError(
            "video_frames are required when no allowlisted MP4 source can be reused."
        )
    try:
        prepared = video_io.prepare_video(
            frames,
            checked_fps,
            checked_duration,
            video=reusable,
            max_bytes=limits.max_asset_bytes,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "budget" in message or "over the" in message or "lowest quality" in message:
            raise AssetLimitError(str(exc)) from exc
        raise
    payload = prepared.mp4_bytes
    _validate_mp4_video_payload(payload, "The video encoder output")
    return _PreparedAsset(
        asset_id="reference_video_01",
        h3_label="<Video 1>",
        kind="video",
        filename=filename,
        intended_jobs=("pose", "motion", "camera", "timing"),
        prohibited_transfers=("identity", "appearance", "audio"),
        payload=payload,
        mime_type="video/mp4",
    )


def _prepare_audio(audio: Any, limits: AssetLimits) -> _PreparedAsset:
    filename = "reference_audio_01.wav"
    _validate_filename(filename, ".wav")
    _allowed_type("audio", ".wav", "audio/wav", limits)
    try:
        encoded, _duration = audio_io.audio_to_wav_b64(audio)
    except RuntimeError as exc:
        raise AssetValidationError(f"Could not convert audio to WAV: {exc}") from exc
    payload = _decode_base64(encoded, "Audio")
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise AssetValidationError("The audio encoder did not produce a WAV container.")
    return _PreparedAsset(
        asset_id="reference_audio_01",
        h3_label="<Audio 1>",
        kind="audio",
        filename=filename,
        intended_jobs=("audio", "timing"),
        prohibited_transfers=("identity", "appearance", "pose", "motion"),
        payload=payload,
        mime_type="audio/wav",
    )


def _append_with_limits(
    prepared: list[_PreparedAsset],
    asset: _PreparedAsset,
    limits: AssetLimits,
    current_total: int,
) -> int:
    asset_bytes = len(asset.payload)
    if asset_bytes <= 0:
        raise AssetValidationError(f"{asset.asset_id} is empty.")
    if len(prepared) + 1 > limits.max_asset_count:
        raise AssetLimitError(
            f"Asset count exceeds the {limits.max_asset_count}-asset limit."
        )
    if asset_bytes > limits.max_asset_bytes:
        raise AssetLimitError(
            f"{asset.asset_id} is {asset_bytes} bytes, over the "
            f"{limits.max_asset_bytes}-byte per-asset limit."
        )
    total = current_total + asset_bytes
    if total > limits.max_total_bytes:
        raise AssetLimitError(
            f"Staged media totals {total} bytes, over the "
            f"{limits.max_total_bytes}-byte job limit."
        )
    prepared.append(asset)
    return total


def _prepare_staging_roots(
    temp_root: os.PathLike[str] | str, request_id: str
) -> tuple[Path, Path, Path]:
    configured_root = Path(temp_root).expanduser()
    try:
        configured_root.mkdir(parents=True, exist_ok=True)
        resolved_root = configured_root.resolve(strict=True)
    except OSError as exc:
        raise AssetStagingError(f"Could not prepare temp root: {exc}") from exc
    if not resolved_root.is_dir():
        raise AssetStagingError(f"Temp root is not a directory: {resolved_root}")

    h3_root_candidate = resolved_root / STAGING_DIRECTORY_NAME
    if h3_root_candidate.is_symlink():
        raise AssetSecurityError("The H3 Hermes staging root cannot be a symlink.")
    try:
        h3_root_candidate.mkdir(mode=0o700, exist_ok=True)
        h3_root = h3_root_candidate.resolve(strict=True)
    except OSError as exc:
        raise AssetStagingError(f"Could not prepare H3 Hermes staging root: {exc}") from exc
    assert_contained(h3_root, resolved_root)

    job_candidate = h3_root / request_id
    if os.path.lexists(job_candidate):
        raise AssetSecurityError(
            f"A staging path already exists for request_id {request_id}."
        )
    job_dir = assert_contained(job_candidate, h3_root)
    return resolved_root, h3_root, job_dir


def stage_assets(
    temp_root: os.PathLike[str] | str,
    *,
    request_id: Optional[str] = None,
    images: Any = None,
    video_frames: Any = None,
    video_fps: Optional[float] = None,
    video_duration: Optional[float] = None,
    video: Any = None,
    keyframe_images: Any = None,
    audio: Any = None,
    asset_directives: Optional[Mapping[str, AssetDirective]] = None,
    strict_image_slots: bool = True,
    limits: Optional[AssetLimits] = None,
    allow_video_source_reuse: bool = False,
    allowed_video_source_roots: Sequence[os.PathLike[str] | str] = (),
) -> StagedAssets:
    """Convert, bound, and stage media below one UUID request directory.

    Conversion and all byte-limit checks happen before the job directory is
    created. A write/finalization failure invokes :func:`cleanup_assets` with
    ``incomplete=True``, so a failed call can never leave a successful-looking
    partial manifest.
    """

    checked_limits = limits if limits is not None else AssetLimits()
    if not isinstance(checked_limits, AssetLimits):
        raise TypeError("limits must be an AssetLimits instance")
    checked_request_id = (
        str(uuid.uuid4())
        if request_id is None
        else _canonical_request_id(request_id)
    )
    slots = _normalize_image_slots(images, strict_image_slots)
    keyframe_slots = _normalize_keyframe_slots(keyframe_images)
    has_video = video_frames is not None or video is not None
    requested_count = (
        len(slots)
        + int(has_video)
        + len(keyframe_slots)
        + int(audio is not None)
    )
    if requested_count > checked_limits.max_asset_count:
        raise AssetLimitError(
            f"Asset count {requested_count} exceeds the "
            f"{checked_limits.max_asset_count}-asset limit."
        )

    prepared: list[_PreparedAsset] = []
    total_bytes = 0
    for slot, image in slots:
        total_bytes = _append_with_limits(
            prepared,
            _prepare_image(slot, image, checked_limits),
            checked_limits,
            total_bytes,
        )
    if has_video:
        total_bytes = _append_with_limits(
            prepared,
            _prepare_video(
                video_frames,
                video_fps,
                video_duration,
                video,
                allow_source_reuse=allow_video_source_reuse,
                allowed_source_roots=allowed_video_source_roots,
                limits=checked_limits,
            ),
            checked_limits,
            total_bytes,
        )
    for slot, image in keyframe_slots:
        total_bytes = _append_with_limits(
            prepared,
            _prepare_keyframe(slot, image, checked_limits),
            checked_limits,
            total_bytes,
        )
    if audio is not None:
        total_bytes = _append_with_limits(
            prepared,
            _prepare_audio(audio, checked_limits),
            checked_limits,
            total_bytes,
        )

    prepared = _apply_asset_directives(prepared, asset_directives)
    resolved_temp_root, h3_root, job_dir = _prepare_staging_roots(
        temp_root, checked_request_id
    )

    created = False
    try:
        job_dir.mkdir(mode=0o700, exist_ok=False)
        created = True
        manifest_assets: list[dict[str, Any]] = []
        trusted_assets: list[_TrustedAssetRecord] = []
        for asset in prepared:
            destination = assert_contained(job_dir / asset.filename, job_dir)
            with destination.open("xb") as handle:
                handle.write(asset.payload)
            resolved_destination = assert_contained(destination, job_dir)
            if resolved_destination.is_symlink() or not resolved_destination.is_file():
                raise AssetSecurityError(
                    f"Staged asset did not resolve to a regular file: {destination}"
                )
            destination_stat = resolved_destination.stat()
            digest = hashlib.sha256(asset.payload).hexdigest()
            manifest_item = {
                "asset_id": asset.asset_id,
                "h3_label": asset.h3_label,
                "kind": asset.kind,
                "path": str(resolved_destination),
                "intended_jobs": list(asset.intended_jobs),
                "prohibited_transfers": list(asset.prohibited_transfers),
                "sha256": digest,
                "bytes": len(asset.payload),
                "mime_type": asset.mime_type,
            }
            manifest_assets.append(manifest_item)
            trusted_assets.append(
                _TrustedAssetRecord(
                    asset_id=asset.asset_id,
                    h3_label=asset.h3_label,
                    kind=asset.kind,
                    path=resolved_destination,
                    intended_jobs=asset.intended_jobs,
                    prohibited_transfers=asset.prohibited_transfers,
                    sha256=digest,
                    bytes=len(asset.payload),
                    mime_type=asset.mime_type,
                    device=destination_stat.st_dev,
                    inode=destination_stat.st_ino,
                )
            )

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "request_id": checked_request_id,
            "assets": manifest_assets,
        }
        manifest_path = assert_contained(job_dir / MANIFEST_FILENAME, job_dir)
        temporary_manifest = assert_contained(job_dir / ".manifest.json.tmp", job_dir)
        serialized = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            separators=(",", ": "),
        ) + "\n"
        with temporary_manifest.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, manifest_path)
        manifest_path = assert_contained(manifest_path, job_dir)
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise AssetSecurityError("Final manifest is not a regular contained file.")
        temp_stat = resolved_temp_root.stat()
        staging_stat = h3_root.stat()
        job_stat = job_dir.stat()
        manifest_stat = manifest_path.stat()
        trusted_snapshot = _TrustedStagingSnapshot(
            request_id=checked_request_id,
            temp_root=resolved_temp_root,
            staging_root=h3_root,
            job_dir=job_dir,
            manifest_path=manifest_path,
            manifest_bytes=serialized.encode("utf-8"),
            assets=tuple(trusted_assets),
            max_asset_count=checked_limits.max_asset_count,
            max_asset_bytes=checked_limits.max_asset_bytes,
            max_total_bytes=checked_limits.max_total_bytes,
            temp_device=temp_stat.st_dev,
            temp_inode=temp_stat.st_ino,
            staging_device=staging_stat.st_dev,
            staging_inode=staging_stat.st_ino,
            job_device=job_stat.st_dev,
            job_inode=job_stat.st_ino,
            manifest_device=manifest_stat.st_dev,
            manifest_inode=manifest_stat.st_ino,
        )
        return StagedAssets(
            request_id=checked_request_id,
            temp_root=resolved_temp_root,
            job_dir=job_dir,
            manifest_path=manifest_path,
            manifest=json.loads(serialized),
            _trusted_snapshot=trusted_snapshot,
        )
    except BaseException:
        if created:
            try:
                cleanup_assets(
                    job_dir,
                    temp_root=resolved_temp_root,
                    success=False,
                    policy=CleanupPolicy.DELETE_ON_SUCCESS,
                    incomplete=True,
                )
            except BaseException:
                try:
                    if job_dir.is_symlink():
                        job_dir.unlink()
                    elif os.path.lexists(job_dir):
                        shutil.rmtree(job_dir)
                except BaseException:
                    pass
        raise


def _integrity_fail() -> NoReturn:
    raise AssetIntegrityError(_INTEGRITY_ERROR_MESSAGE)


def _trusted_manifest(snapshot: _TrustedStagingSnapshot) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "request_id": snapshot.request_id,
        "assets": [
            {
                "asset_id": asset.asset_id,
                "h3_label": asset.h3_label,
                "kind": asset.kind,
                "path": str(asset.path),
                "intended_jobs": list(asset.intended_jobs),
                "prohibited_transfers": list(asset.prohibited_transfers),
                "sha256": asset.sha256,
                "bytes": asset.bytes,
                "mime_type": asset.mime_type,
            }
            for asset in snapshot.assets
        ],
    }


def _verify_directory_identity(path: Path, device: int, inode: int) -> None:
    if path.is_symlink():
        _integrity_fail()
    try:
        path_stat = path.stat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        _integrity_fail()
    if (
        resolved != path
        or not stat.S_ISDIR(path_stat.st_mode)
        or path_stat.st_dev != device
        or path_stat.st_ino != inode
    ):
        _integrity_fail()


def _read_verified_file(
    path: Path,
    *,
    expected_bytes: int,
    device: int,
    inode: int,
) -> bytes:
    if expected_bytes < 0 or path.is_symlink():
        _integrity_fail()
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        _integrity_fail()

    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size != expected_bytes
            or file_stat.st_dev != device
            or file_stat.st_ino != inode
            or file_stat.st_nlink != 1
        ):
            _integrity_fail()
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            payload = handle.read(expected_bytes + 1)
        if len(payload) != expected_bytes:
            _integrity_fail()
        return payload
    except AssetIntegrityError:
        raise
    except (OSError, ValueError):
        _integrity_fail()
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    _integrity_fail()


def _verify_staged_assets(bundle: StagedAssets) -> None:
    snapshot = bundle._trusted_snapshot
    if (
        bundle.request_id != snapshot.request_id
        or bundle.temp_root != snapshot.temp_root
        or bundle.job_dir != snapshot.job_dir
        or bundle.manifest_path != snapshot.manifest_path
    ):
        _integrity_fail()
    _canonical_request_id(snapshot.request_id)
    if (
        snapshot.staging_root != snapshot.temp_root / STAGING_DIRECTORY_NAME
        or snapshot.job_dir != snapshot.staging_root / snapshot.request_id
        or snapshot.manifest_path != snapshot.job_dir / MANIFEST_FILENAME
    ):
        _integrity_fail()

    _verify_directory_identity(
        snapshot.temp_root,
        snapshot.temp_device,
        snapshot.temp_inode,
    )
    _verify_directory_identity(
        snapshot.staging_root,
        snapshot.staging_device,
        snapshot.staging_inode,
    )
    _verify_directory_identity(
        snapshot.job_dir,
        snapshot.job_device,
        snapshot.job_inode,
    )

    if (
        len(snapshot.assets) > snapshot.max_asset_count
        or any(
            asset.bytes <= 0 or asset.bytes > snapshot.max_asset_bytes
            for asset in snapshot.assets
        )
        or sum(asset.bytes for asset in snapshot.assets) > snapshot.max_total_bytes
    ):
        _integrity_fail()

    expected_names = {MANIFEST_FILENAME}
    for asset in snapshot.assets:
        if (
            asset.path.parent != snapshot.job_dir
            or asset.path.name in expected_names
            or not _is_known_staged_asset_label(asset.h3_label)
        ):
            _integrity_fail()
        expected_names.add(asset.path.name)
    if len(expected_names) != len(snapshot.assets) + 1:
        _integrity_fail()

    seen_names: set[str] = set()
    try:
        entries = snapshot.job_dir.iterdir()
        for entry in entries:
            if entry.name not in expected_names or entry.is_symlink():
                _integrity_fail()
            seen_names.add(entry.name)
    except AssetIntegrityError:
        raise
    except OSError:
        _integrity_fail()
    if seen_names != expected_names:
        _integrity_fail()

    manifest_payload = _read_verified_file(
        snapshot.manifest_path,
        expected_bytes=len(snapshot.manifest_bytes),
        device=snapshot.manifest_device,
        inode=snapshot.manifest_inode,
    )
    if manifest_payload != snapshot.manifest_bytes:
        _integrity_fail()
    try:
        disk_manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        _integrity_fail()
    if disk_manifest != _trusted_manifest(snapshot):
        _integrity_fail()

    for asset in snapshot.assets:
        if asset.path.is_symlink():
            _integrity_fail()
        try:
            if asset.path.resolve(strict=True) != asset.path:
                _integrity_fail()
        except (OSError, RuntimeError):
            _integrity_fail()
        payload = _read_verified_file(
            asset.path,
            expected_bytes=asset.bytes,
            device=asset.device,
            inode=asset.inode,
        )
        if hashlib.sha256(payload).hexdigest() != asset.sha256:
            _integrity_fail()


def verify_staged_assets(bundle: StagedAssets) -> None:
    """Fail closed unless a staged bundle matches its creation-time snapshot.

    Public ``bundle.manifest`` is intentionally ignored: it remains a mutable,
    JSON-compatible convenience view, while verification trusts only the
    separately frozen snapshot and bounded reads of the on-disk files.
    """

    if not isinstance(bundle, StagedAssets):
        _integrity_fail()
    try:
        _verify_staged_assets(bundle)
    except AssetIntegrityError:
        raise
    except Exception:
        raise AssetIntegrityError(_INTEGRITY_ERROR_MESSAGE) from None


def verified_manifest_snapshot(bundle: StagedAssets) -> dict[str, Any]:
    """Return a detached request manifest from verified immutable authority."""

    verify_staged_assets(bundle)
    return _trusted_manifest(bundle._trusted_snapshot)


def _coerce_policy(policy: CleanupPolicy | str) -> CleanupPolicy:
    try:
        return CleanupPolicy(policy)
    except ValueError as exc:
        choices = ", ".join(item.value for item in CleanupPolicy)
        raise AssetValidationError(
            f"Unknown cleanup policy {policy!r}; expected one of: {choices}."
        ) from exc


def _utc_now(now: Optional[datetime]) -> datetime:
    value = datetime.now(timezone.utc) if now is None else now
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _cleanup_target(
    staged_or_path: StagedAssets | os.PathLike[str] | str,
    temp_root: Optional[os.PathLike[str] | str],
) -> tuple[Path, Path]:
    if isinstance(staged_or_path, StagedAssets):
        if temp_root is not None:
            supplied = Path(temp_root).expanduser().resolve(strict=False)
            if supplied != staged_or_path.temp_root:
                raise AssetSecurityError(
                    "Supplied temp_root does not match the staged asset bundle."
                )
        resolved_temp = staged_or_path.temp_root
        job_candidate = staged_or_path.job_dir
    else:
        if temp_root is None:
            raise AssetSecurityError(
                "temp_root is required when cleaning a raw job directory path."
            )
        resolved_temp = Path(temp_root).expanduser().resolve(strict=False)
        job_candidate = Path(staged_or_path).expanduser()

    h3_root_candidate = resolved_temp / STAGING_DIRECTORY_NAME
    if h3_root_candidate.is_symlink():
        raise AssetSecurityError("The H3 Hermes staging root cannot be a symlink.")
    h3_root = h3_root_candidate.resolve(strict=False)
    if job_candidate.is_symlink():
        raise AssetSecurityError("A UUID job directory cannot be a symlink.")
    job_dir = job_candidate.resolve(strict=False)
    if job_dir.parent != h3_root:
        raise AssetSecurityError("Cleanup target is not a direct UUID job directory.")
    _canonical_request_id(job_dir.name)
    assert_contained(job_dir, h3_root)
    if job_dir.exists() and not job_dir.is_dir():
        raise AssetSecurityError("Cleanup target is not a directory.")
    return resolved_temp, job_dir


def cleanup_assets(
    staged_or_path: StagedAssets | os.PathLike[str] | str,
    *,
    success: bool,
    policy: CleanupPolicy | str,
    temp_root: Optional[os.PathLike[str] | str] = None,
    now: Optional[datetime] = None,
    incomplete: bool = False,
) -> CleanupResult:
    """Apply the only staged-job deletion/retention policy.

    ``delete_on_success`` removes successful jobs and retains failed jobs for
    24 hours. ``retain_24h`` retains either outcome for 24 hours. ``retain``
    keeps the directory indefinitely. ``incomplete`` is reserved for staging
    rollback and always removes the partial job directory.
    """

    checked_policy = _coerce_policy(policy)
    _resolved_temp, job_dir = _cleanup_target(staged_or_path, temp_root)

    if incomplete or (success and checked_policy is CleanupPolicy.DELETE_ON_SUCCESS):
        if job_dir.exists():
            shutil.rmtree(job_dir)
        return CleanupResult(
            retained=False,
            path=None,
            retention_expires_at=None,
            policy=checked_policy,
        )

    if not job_dir.exists():
        return CleanupResult(
            retained=False,
            path=None,
            retention_expires_at=None,
            policy=checked_policy,
        )

    marker = assert_contained(job_dir / RETENTION_MARKER_FILENAME, job_dir)
    marker_temp = assert_contained(job_dir / ".retention_until.tmp", job_dir)
    if checked_policy is CleanupPolicy.RETAIN:
        marker.unlink(missing_ok=True)
        marker_temp.unlink(missing_ok=True)
        return CleanupResult(
            retained=True,
            path=job_dir,
            retention_expires_at=None,
            policy=checked_policy,
        )

    checked_now = _utc_now(now)
    expires = checked_now + timedelta(hours=FAILED_RETENTION_HOURS)
    expires_text = _format_utc(expires)
    with marker_temp.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(expires_text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(marker_temp, marker)
    os.utime(job_dir, (checked_now.timestamp(), checked_now.timestamp()))
    return CleanupResult(
        retained=True,
        path=job_dir,
        retention_expires_at=expires_text,
        policy=checked_policy,
    )


def _parse_retention_marker(marker: Path) -> Optional[datetime]:
    if marker.is_symlink() or not marker.is_file():
        return None
    try:
        with marker.open("r", encoding="ascii") as handle:
            raw = handle.read(128).strip()
            if handle.read(1):
                return None
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (OSError, UnicodeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def prune_expired_assets(
    temp_root: os.PathLike[str] | str,
    *,
    now: Optional[datetime] = None,
) -> tuple[Path, ...]:
    """Delete only UUID jobs carrying an expired bounded-retention marker."""

    checked_now = _utc_now(now)
    resolved_temp = Path(temp_root).expanduser().resolve(strict=False)
    h3_root_candidate = resolved_temp / STAGING_DIRECTORY_NAME
    if h3_root_candidate.is_symlink():
        raise AssetSecurityError("The H3 Hermes staging root cannot be a symlink.")
    h3_root = h3_root_candidate.resolve(strict=False)
    if not h3_root.exists():
        return ()
    if not h3_root.is_dir():
        raise AssetSecurityError("The H3 Hermes staging root is not a directory.")

    removed: list[Path] = []
    for child in sorted(h3_root.iterdir(), key=lambda path: path.name):
        if child.is_symlink() or not child.is_dir():
            continue
        try:
            _canonical_request_id(child.name)
        except AssetSecurityError:
            continue
        marker = assert_contained(child / RETENTION_MARKER_FILENAME, child)
        expires = _parse_retention_marker(marker)
        if expires is None or expires > checked_now:
            continue
        cleanup_assets(
            child,
            temp_root=resolved_temp,
            success=True,
            policy=CleanupPolicy.DELETE_ON_SUCCESS,
            now=checked_now,
        )
        removed.append(child)
    return tuple(removed)


# Descriptive compatibility names for callers integrating the bridge.
AssetStagingLimits = AssetLimits
stage_media_assets = stage_assets
stage_media = stage_assets
cleanup_staged_assets = cleanup_assets
prune_expired_jobs = prune_expired_assets

__all__ = [
    "AssetDirective",
    "AssetIntegrityError",
    "AssetLimitError",
    "AssetLimits",
    "AssetSecurityError",
    "AssetStagingError",
    "AssetStagingLimits",
    "AssetValidationError",
    "CleanupPolicy",
    "CleanupResult",
    "MANIFEST_SCHEMA_VERSION",
    "MAX_ASSET_DIRECTIVES",
    "MAX_ASSET_DIRECTIVE_ITEMS",
    "MAX_ASSET_DIRECTIVE_VALUE_LENGTH",
    "StagedAssets",
    "assert_contained",
    "cleanup_assets",
    "cleanup_staged_assets",
    "prune_expired_assets",
    "prune_expired_jobs",
    "stage_assets",
    "stage_media",
    "stage_media_assets",
    "verified_manifest_snapshot",
    "verify_staged_assets",
]
