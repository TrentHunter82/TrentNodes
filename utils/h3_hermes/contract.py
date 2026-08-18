"""Build and strictly parse the versioned H3 Hermes JSON contract.

This module is intentionally dependency-free.  It treats Hermes as an
untrusted text-producing boundary: JSON shape, selection, identifiers, byte
caps, and physical H3 asset labels are all checked locally.
"""

import copy
import json
import re
import unicodedata
from dataclasses import asdict, is_dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, NoReturn, Optional, Set, cast
from uuid import UUID, uuid4

from .schema import (
    H3_MODES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    QUALITY_BUDGETS,
    QUALITY_MODES,
    REQUEST_SCHEMA_VERSION,
    RESEARCH_POLICIES,
    RESPONSE_SCHEMA_VERSION,
    TARGET_MODEL,
    HermesCandidate,
    ParsedHermesResult,
)


class ContractError(ValueError):
    """A request or result violates the local H3 Hermes contract."""


HermesContractError = ContractError


STABLE_INSTRUCTIONS = """You are the H3 Hermes prompt director. Follow every rule below.
1. The private-media scope is only submitted asset paths in this request; do not inspect any unrelated path.
2. Use actual tool inspection for relevant submitted assets when available. Never claim inspection from filenames or model reports alone.
3. Separate direct observations from assumptions explicitly.
4. Use current/official MiniMax H3 evidence first. Label community evidence separately and never present it as official.
5. Build a canonical intent plan before drafting candidate prompts.
6. Preserve exact literals (dialogue, lyrics, and visible text), reference bindings, and all user-locked constraints exactly.
7. In hero mode, make candidates deliberately distinct policies rather than paraphrases.
8. Critique every candidate against the typed intent and its required atoms, not generic prose beauty.
9. All budget limits are maxima and must not be exceeded.
10. Return JSON only: exactly one object conforming to h3_hermes_result/1.0, with no prose or Markdown fence.
11. Use exactly this object shape and field names; do not rename, add, or omit fields:
{"schema_version":"h3_hermes_result/1.0","request_id":"<request.request_id>","status":"ok","evidence":{"observations":[],"assumptions":[],"uninspected_assets":[]},"intent_ir":{"required_atoms":[],"preferred_atoms":[],"optional_atoms":[],"reference_jobs":[]},"candidates":[{"candidate_id":"candidate_1","policy":"...","prompt":"...","score_vector":{},"critic_findings":[]}],"selected_candidate_id":"candidate_1","h3_prompt":"...","repairs":[],"quality_report":{"hard_errors":[],"warnings":[],"unresolved_ambiguities":[],"reported_tools":[],"reported_sources":[]}}
12. Labels of the form <Video 1 Keyframe N> are JPEG inspection evidence only. They are never physical H3 labels or sampler/identity/audio references and MUST NOT occur anywhere in candidates, h3_prompt, intent_ir, evidence, repairs, or quality_report. Bind any observation derived from them only to physical <Video 1>.
13. NEVER write/delete files (never write or delete files), change configuration (change config), manage skills, send messages, schedule jobs, or modify repositories.
"""

# Public aliases make the stable Runs-API instruction block easy to discover.
H3_HERMES_INSTRUCTIONS = STABLE_INSTRUCTIONS
HERMES_INSTRUCTIONS = STABLE_INSTRUCTIONS

_H3_LABEL_RE = re.compile(r"<(?:Picture|Video|Audio) [1-9][0-9]*>")
_PICTURE_ASSET_LABEL_RE = re.compile(r"<Picture ([1-9][0-9]{0,3})>")
_KEYFRAME_ASSET_LABEL_RE = re.compile(
    r"<Video 1 Keyframe ([1-9][0-9]{0,3})>"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_FL2VA_BARE_PICTURE_RE = re.compile(
    r"(?<![\w<])Picture ([1-9][0-9]*)(?![\w>]|,\d)"
)
_JSON_FENCE_RE = re.compile(
    r"```json[ \t]*\r?\n(?P<body>.*?)\r?\n```",
    re.DOTALL | re.IGNORECASE,
)

_GENERATION_FIELDS = (
    "requested_duration_seconds",
    "snapped_duration_seconds",
    "fps",
    "width",
    "height",
    "length",
)
_TASK_FIELDS = (
    "task_types",
    "video_role",
    "audio_role",
    "constraints",
    "cut_timestamps",
)
_RESPONSE_FIELDS = (
    "schema_version",
    "request_id",
    "status",
    "evidence",
    "intent_ir",
    "candidates",
    "selected_candidate_id",
    "h3_prompt",
    "repairs",
    "quality_report",
)
_EVIDENCE_FIELDS = ("observations", "assumptions", "uninspected_assets")
_INTENT_IR_FIELDS = (
    "required_atoms",
    "preferred_atoms",
    "optional_atoms",
    "reference_jobs",
)
_CANDIDATE_FIELDS = (
    "candidate_id",
    "policy",
    "prompt",
    "score_vector",
    "critic_findings",
)
_QUALITY_REPORT_FIELDS = (
    "hard_errors",
    "warnings",
    "unresolved_ambiguities",
    "reported_tools",
    "reported_sources",
)

_STAGED_ASSET_FIELDS = (
    "asset_id",
    "h3_label",
    "kind",
    "path",
    "intended_jobs",
    "prohibited_transfers",
    "sha256",
    "bytes",
    "mime_type",
)
_EXTERNAL_ASSET_FIELDS = ("h3_label", "authority", "inspection_status")
_EXTERNAL_AUDIO_AUTHORITY = {
    "h3_label": "<Audio 1>",
    "authority": "downstream_required_external",
    "inspection_status": "uninspected",
}

# These mirror the staging boundary's hard media limits without importing its
# conversion/filesystem module into this dependency-free wire contract.
_MAX_STAGED_ASSETS = 12
_MAX_PHYSICAL_ASSETS = 12
_MAX_PICTURE_SLOT = 6
_MAX_KEYFRAME_ASSETS = 16
_MAX_REQUEST_ASSETS = _MAX_STAGED_ASSETS + 1
_MAX_ASSET_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_ASSET_BYTES = 128 * 1024 * 1024
_MAX_ASSET_LIST_ITEMS = 16
_MAX_ASSET_LIST_STRING_LENGTH = 128
_MAX_ASSET_PATH_LENGTH = 4096

# Thawing is used on both ordinary dict/list requests and deeply immutable
# request authority. Bound it independently of the eventual JSON byte cap so
# pathological custom Mappings cannot force unbounded recursive traversal.
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 20_000
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


def _fail(message: str) -> NoReturn:
    raise ContractError(message)


def _validate_cap(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{name} must be a positive integer byte cap")
    return value


def _integer(value: Any, field: str, *, positive: bool) -> int:
    requirement = "positive" if positive else "nonnegative"
    if type(value) is not int:
        _fail(f"{field} must be a {requirement} integer")
    if (positive and value <= 0) or (not positive and value < 0):
        _fail(f"{field} must be a {requirement} integer")
    return value


def _uuid_text(value: Any, field: str) -> str:
    if type(value) is UUID:
        return str(value)
    if type(value) is not str or not value:
        _fail(f"{field} must be a UUID string")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ContractError(f"{field} must be a valid UUID") from exc
    # Normalize requests to one deterministic spelling.  Result request IDs
    # remain subject to an exact match against this normalized value.
    return str(parsed)


def _canonical_uuid_text(value: Any, field: str) -> str:
    canonical = _uuid_text(value, field)
    if type(value) is not str or value != canonical:
        _fail(f"{field} must use exact canonical UUID spelling")
    return canonical


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if type(value) not in (dict, _MAPPING_PROXY_TYPE):
        _fail(f"{field} must be a JSON object")
    return value


def _thaw_json(value: Any, field: str) -> Any:
    """Detach a bounded JSON tree, converting read-only containers to plain ones."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    active: Set[int] = set()
    remaining = [_MAX_JSON_NODES]

    def walk(item: Any, depth: int) -> Any:
        if depth > _MAX_JSON_DEPTH:
            _fail(f"{field} exceeds the maximum JSON nesting depth")
        remaining[0] -= 1
        if remaining[0] < 0:
            _fail(f"{field} exceeds the maximum JSON value count")

        if isinstance(item, Mapping):
            if type(item) not in (dict, _MAPPING_PROXY_TYPE):
                _fail(f"{field} contains a non-plain JSON object")
            identity = id(item)
            if identity in active:
                _fail(f"{field} contains a recursive mapping")
            active.add(identity)
            try:
                result: Dict[str, Any] = {}
                for key, nested in item.items():
                    if type(key) is not str:
                        _fail(f"{field} JSON object keys must be strings")
                    result[key] = walk(nested, depth + 1)
                return result
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple)):
            if type(item) not in (list, tuple):
                _fail(f"{field} contains a non-plain JSON array")
            identity = id(item)
            if identity in active:
                _fail(f"{field} contains a recursive array")
            active.add(identity)
            try:
                return [walk(nested, depth + 1) for nested in item]
            finally:
                active.remove(identity)
        if item is None or type(item) in (str, bool, int, float):
            return item
        _fail(f"{field} contains a non-JSON value of type {type(item).__name__}")

    return walk(value, 0)


def _deep_freeze_json(value: Any) -> Any:
    """Recursively freeze a validated plain JSON tree without dict subclasses."""

    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _deep_freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _list(value: Any, field: str) -> List[Any]:
    if not isinstance(value, list):
        _fail(f"{field} must be a JSON array")
    return value


def _plain_mapping(value: Any, field: str) -> Dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if type(value) is not dict:
        _fail(f"{field} must be a plain JSON object")
    result = cast(Dict[str, Any], value)
    if any(type(key) is not str for key in result):
        _fail(f"{field} JSON object keys must be exact strings")
    return copy.deepcopy(result)


def _plain_list(value: Any, field: str) -> List[Any]:
    if type(value) not in (list, tuple):
        _fail(f"{field} must be a list")
    result: List[Any] = []
    for item in value:
        if is_dataclass(item) and not isinstance(item, type):
            result.append(asdict(item))
        else:
            result.append(copy.deepcopy(item))
    return result


def _string_list(value: Any, field: str) -> List[str]:
    items = _plain_list(value, field)
    if any(type(item) is not str for item in items):
        _fail(f"{field} must contain only strings")
    return items


def _number(value: Any, field: str, *, positive: bool = False) -> Any:
    if type(value) not in (int, float):
        _fail(f"{field} must be a number")
    if positive and value <= 0:
        _fail(f"{field} must be greater than zero")
    return value


def _validate_exact_literals(value: Any) -> Dict[str, Any]:
    result = _plain_mapping(value, "exact_literals")
    fields = ("dialogue", "lyrics", "visible_text")
    _validate_exact_keys(result, fields, "exact_literals")
    for field in fields:
        if field not in result:
            _fail(f"exact_literals.{field} is required")
    if type(result["dialogue"]) is not str:
        _fail("exact_literals.dialogue must be a string")
    if type(result["lyrics"]) is not str:
        _fail("exact_literals.lyrics must be a string")
    result["visible_text"] = _string_list(
        result["visible_text"], "exact_literals.visible_text"
    )
    return result


def _validate_generation(value: Any) -> Dict[str, Any]:
    result = _plain_mapping(value, "generation")
    _validate_exact_keys(result, _GENERATION_FIELDS, "generation")
    for field in _GENERATION_FIELDS:
        if field not in result:
            _fail(f"generation.{field} is required")
    for field in ("requested_duration_seconds", "snapped_duration_seconds",
                  "fps"):
        _number(result[field], f"generation.{field}", positive=True)
    for field in ("width", "height", "length"):
        item = result[field]
        if type(item) is not int or item <= 0:
            _fail(f"generation.{field} must be a positive integer")
    return result


def _validate_task(value: Any) -> Dict[str, Any]:
    result = _plain_mapping(value, "task")
    _validate_exact_keys(result, _TASK_FIELDS, "task")
    for field in _TASK_FIELDS:
        if field not in result:
            _fail(f"task.{field} is required")
    result["task_types"] = _string_list(result["task_types"], "task.task_types")
    for field in ("video_role", "audio_role"):
        if type(result[field]) is not str:
            _fail(f"task.{field} must be a string")
    result["constraints"] = _string_list(
        result["constraints"], "task.constraints"
    )
    cuts = _plain_list(result["cut_timestamps"], "task.cut_timestamps")
    for index, cut in enumerate(cuts):
        _number(cut, f"task.cut_timestamps[{index}]")
        if cut < 0:
            _fail(f"task.cut_timestamps[{index}] cannot be negative")
    result["cut_timestamps"] = cuts
    return result


def _asset_text(value: Any, field: str, *, max_length: int) -> str:
    if type(value) is not str or not value or len(value) > max_length:
        _fail(f"{field} must be a non-empty string of at most {max_length} characters")
    if any(unicodedata.category(character) == "Cc" for character in value):
        _fail(f"{field} cannot contain control characters")
    return value


def _asset_string_list(
    value: Any, field: str, *, allow_empty: bool
) -> List[str]:
    items = _plain_list(value, field)
    if len(items) > _MAX_ASSET_LIST_ITEMS:
        _fail(f"{field} exceeds the {_MAX_ASSET_LIST_ITEMS}-item limit")
    if not items and not allow_empty:
        _fail(f"{field} cannot be empty")
    result: List[str] = []
    equivalents: Set[str] = set()
    for index, item in enumerate(items):
        checked = _asset_text(
            item,
            f"{field}[{index}]",
            max_length=_MAX_ASSET_LIST_STRING_LENGTH,
        )
        if not checked.strip():
            _fail(f"{field}[{index}] cannot be blank")
        equivalent = checked.strip().casefold()
        if equivalent in equivalents:
            _fail(f"{field} contains duplicate-equivalent strings")
        equivalents.add(equivalent)
        result.append(checked)
    return result


def _staged_path_identity(
    value: Any, *, request_id: str, filename: str, field: str
) -> tuple[str, bool]:
    path = _asset_text(value, field, max_length=_MAX_ASSET_PATH_LENGTH)
    normalized = path.replace("\\", "/")
    windows_path = (
        _WINDOWS_ABSOLUTE_PATH_RE.match(path) is not None
        or path.startswith("\\\\")
    )
    if not (path.startswith("/") or windows_path):
        _fail(f"{field} must be an absolute staged asset path")
    parts = [part for part in normalized.split("/") if part]
    if any(part in (".", "..") for part in parts):
        _fail(f"{field} cannot contain traversal components")
    expected_suffix = ["h3_hermes", request_id, filename]
    if len(parts) < len(expected_suffix) or parts[-3:] != expected_suffix:
        _fail(
            f"{field} must end in h3_hermes/{request_id}/{filename}"
        )
    identity = "/".join(parts)
    if windows_path:
        identity = identity.casefold()
    return identity, windows_path


def _staged_asset_identity(
    record: Mapping[str, Any], index: int
) -> tuple[str, str, str, str, bool]:
    prefix = f"request.assets[{index}]"
    label = _asset_text(record.get("h3_label"), f"{prefix}.h3_label", max_length=64)
    keyframe = False
    picture_match = _PICTURE_ASSET_LABEL_RE.fullmatch(label)
    keyframe_match = _KEYFRAME_ASSET_LABEL_RE.fullmatch(label)
    if picture_match is not None:
        slot = int(picture_match.group(1))
        if slot > _MAX_PICTURE_SLOT:
            _fail(
                f"{prefix}.h3_label exceeds the {_MAX_PICTURE_SLOT}-picture "
                "node interface"
            )
        asset_id = f"picture_{slot:02d}"
        filename = f"{asset_id}.jpg"
        kind = "image"
        mime_type = "image/jpeg"
    elif label == "<Video 1>":
        asset_id = "reference_video_01"
        filename = "reference_video_01.mp4"
        kind = "video"
        mime_type = "video/mp4"
    elif label == "<Audio 1>":
        asset_id = "reference_audio_01"
        filename = "reference_audio_01.wav"
        kind = "audio"
        mime_type = "audio/wav"
    elif keyframe_match is not None:
        keyframe_index = int(keyframe_match.group(1))
        asset_id = f"video_01_keyframe_{keyframe_index:02d}"
        filename = f"{asset_id}.jpg"
        kind = "image"
        mime_type = "image/jpeg"
        keyframe = True
    else:
        _fail(f"{prefix}.h3_label is not a canonical staged asset label")

    for field, expected in (
        ("asset_id", asset_id),
        ("kind", kind),
        ("mime_type", mime_type),
    ):
        if record.get(field) != expected:
            _fail(f"{prefix}.{field} must be exactly {expected!r}")
    return label, asset_id, filename, mime_type, keyframe


def _expected_asset_directives(
    label: str, *, h3_mode: str, keyframe: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Bind every staged record to one node-produced semantic authority."""

    if keyframe:
        if h3_mode != "ref":
            _fail("keyframe evidence assets are valid only in ref mode")
        return (
            ("visual_evidence", "timestamp_context"),
            ("sampler", "identity", "audio"),
        )
    if label == "<Video 1>":
        if h3_mode != "ref":
            _fail("staged Video 1 is valid only in ref mode")
        return (
            ("pose", "motion", "camera", "timing"),
            ("identity", "appearance", "audio"),
        )
    if label == "<Audio 1>":
        if h3_mode != "ref":
            _fail("staged Audio 1 is valid only in ref mode")
        return (
            ("audio", "timing"),
            ("identity", "appearance", "pose", "motion"),
        )

    picture_match = _PICTURE_ASSET_LABEL_RE.fullmatch(label)
    if picture_match is None:
        _fail("staged asset has no canonical semantic directive authority")
    slot = int(picture_match.group(1))
    if h3_mode == "ref":
        return (
            ("identity", "appearance"),
            ("pose", "motion", "audio"),
        )
    if h3_mode == "base_I2VA" and slot == 1:
        return (("first_frame", "appearance", "identity"), ())
    if h3_mode == "base_FL2VA" and slot == 1:
        return (("first_frame", "appearance", "identity"), ())
    if h3_mode == "base_FL2VA" and slot == 2:
        return (("last_frame", "continuity"), ("audio",))
    if h3_mode == "base_L2VA" and slot == 1:
        return (("last_frame", "continuity"), ("audio",))
    _fail(f"{label} is not a staged authority for mode {h3_mode}")


def _validate_assets(
    value: Any,
    *,
    request_id: str,
    h3_mode: str,
    task: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    assets = _plain_list(value, "request.assets")
    if len(assets) > _MAX_REQUEST_ASSETS:
        _fail(f"request.assets exceeds the {_MAX_REQUEST_ASSETS}-record limit")

    result: List[Dict[str, Any]] = []
    seen_labels: Set[str] = set()
    seen_asset_ids: Set[str] = set()
    seen_paths: Set[str] = set()
    physical_count = 0
    staged_count = 0
    keyframe_count = 0
    total_bytes = 0
    staged_audio = False
    staged_video = False
    external_audio = False
    keyframe_indices: List[int] = []

    for index, item in enumerate(assets):
        record = _plain_mapping(item, f"request.assets[{index}]")
        label_value = record.get("h3_label")
        label = _asset_text(
            label_value,
            f"request.assets[{index}].h3_label",
            max_length=64,
        )
        equivalent_label = label.strip().casefold()
        if equivalent_label in seen_labels:
            _fail(f"request.assets contains duplicate-equivalent label {label!r}")
        seen_labels.add(equivalent_label)

        if "authority" in record or "inspection_status" in record:
            _validate_exact_keys(
                record, _EXTERNAL_ASSET_FIELDS, f"request.assets[{index}]"
            )
            if any(
                type(record.get(field)) is not str
                or record.get(field) != expected
                for field, expected in _EXTERNAL_AUDIO_AUTHORITY.items()
            ):
                _fail("external asset authority must use the exact Audio 1 record")
            external_audio = True
            result.append(record)
            continue

        staged_count += 1
        if staged_count > _MAX_STAGED_ASSETS:
            _fail(
                f"request.assets exceeds the {_MAX_STAGED_ASSETS}-staged-asset limit"
            )
        _validate_exact_keys(record, _STAGED_ASSET_FIELDS, f"request.assets[{index}]")
        label, asset_id, filename, _mime_type, keyframe = _staged_asset_identity(
            record, index
        )
        if asset_id in seen_asset_ids:
            _fail(f"request.assets contains duplicate asset_id {asset_id!r}")
        seen_asset_ids.add(asset_id)
        path_identity, _windows_path = _staged_path_identity(
            record.get("path"),
            request_id=request_id,
            filename=filename,
            field=f"request.assets[{index}].path",
        )
        if path_identity in seen_paths:
            _fail("request.assets contains duplicate-equivalent staged paths")
        seen_paths.add(path_identity)

        sha256 = record.get("sha256")
        if type(sha256) is not str or _SHA256_RE.fullmatch(sha256) is None:
            _fail(f"request.assets[{index}].sha256 must be lowercase 64-hex")
        byte_count = record.get("bytes")
        if (
            type(byte_count) is not int
            or byte_count <= 0
            or byte_count > _MAX_ASSET_BYTES
        ):
            _fail(
                f"request.assets[{index}].bytes must be a positive integer "
                f"no greater than {_MAX_ASSET_BYTES}"
            )
        total_bytes += byte_count
        if total_bytes > _MAX_TOTAL_ASSET_BYTES:
            _fail("request.assets exceeds the staged total-byte limit")
        record["intended_jobs"] = _asset_string_list(
            record.get("intended_jobs"),
            f"request.assets[{index}].intended_jobs",
            allow_empty=False,
        )
        record["prohibited_transfers"] = _asset_string_list(
            record.get("prohibited_transfers"),
            f"request.assets[{index}].prohibited_transfers",
            allow_empty=True,
        )
        expected_jobs, expected_prohibited = _expected_asset_directives(
            label, h3_mode=h3_mode, keyframe=keyframe
        )
        if tuple(record["intended_jobs"]) != expected_jobs:
            _fail(
                f"request.assets[{index}].intended_jobs does not match the "
                "canonical semantic authority"
            )
        if tuple(record["prohibited_transfers"]) != expected_prohibited:
            _fail(
                f"request.assets[{index}].prohibited_transfers does not match "
                "the canonical semantic authority"
            )

        if keyframe:
            keyframe_count += 1
            if keyframe_count > _MAX_KEYFRAME_ASSETS:
                _fail("request.assets exceeds the staged keyframe evidence limit")
            keyframe_match = _KEYFRAME_ASSET_LABEL_RE.fullmatch(label)
            if keyframe_match is None:
                _fail("keyframe evidence label is not canonical")
            keyframe_indices.append(int(keyframe_match.group(1)))
        else:
            physical_count += 1
            if physical_count > _MAX_PHYSICAL_ASSETS:
                _fail("request.assets exceeds the staged physical asset limit")
            if label == "<Audio 1>":
                staged_audio = True
            elif label == "<Video 1>":
                staged_video = True
        result.append(record)

    if keyframe_indices:
        if not staged_video:
            _fail("keyframe evidence requires staged Video 1 authority")
        if keyframe_indices != list(range(1, len(keyframe_indices) + 1)):
            _fail("keyframe evidence labels must be ordered and contiguous from 1")
    if external_audio:
        if h3_mode != "ref" or task.get("audio_role") != "reuse":
            _fail(
                "external Audio 1 authority requires ref mode and task.audio_role='reuse'"
            )
        if staged_audio:
            _fail("external Audio 1 authority conflicts with staged Audio 1")
    return result


def _quality_budget(
    quality_mode: str,
    wall_clock_timeout_seconds: Any,
    supplied_budgets: Optional[Mapping[str, Any]],
) -> Dict[str, int]:
    if supplied_budgets is not None and wall_clock_timeout_seconds is None:
        supplied_budgets = _mapping(supplied_budgets, "budgets")
        wall_clock_timeout_seconds = supplied_budgets.get(
            "wall_clock_timeout_seconds", 900
        )
    if wall_clock_timeout_seconds is None:
        wall_clock_timeout_seconds = 900
    wall_clock_timeout_seconds = _integer(
        wall_clock_timeout_seconds, "wall_clock_timeout_seconds", positive=True
    )

    # Quality controls are authoritative schema values, not caller-overridable
    # knobs.  Only the node timeout is carried in from runtime configuration.
    result = dict(QUALITY_BUDGETS[quality_mode])
    result["wall_clock_timeout_seconds"] = wall_clock_timeout_seconds
    return result


def build_request(
    *,
    request_id: Optional[Any] = None,
    h3_mode: str,
    quality_mode: str,
    research_policy: str,
    creative_brief: str,
    exact_literals: Mapping[str, Any],
    generation: Mapping[str, Any],
    task: Mapping[str, Any],
    subjects: Iterable[Any],
    assets: Iterable[Any],
    local_h3_format_guide: Optional[str] = None,
    wall_clock_timeout_seconds: Optional[int] = None,
    schema_version: str = REQUEST_SCHEMA_VERSION,
    target_model: str = TARGET_MODEL,
    budgets: Optional[Mapping[str, Any]] = None,
    required_response_schema: str = RESPONSE_SCHEMA_VERSION,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> Dict[str, Any]:
    """Build and validate one JSON-compatible `h3_hermes_request/1.0`.

    Exact literal strings are copied but never stripped, normalized, or
    re-encoded.  Candidate/delegation/repair budgets come only from the chosen
    quality mode so a malformed caller cannot silently widen them.
    """

    if type(schema_version) is not str or schema_version != REQUEST_SCHEMA_VERSION:
        _fail(
            f"incompatible request schema_version {schema_version!r}; "
            f"expected {REQUEST_SCHEMA_VERSION!r}"
        )
    if type(target_model) is not str or target_model != TARGET_MODEL:
        _fail(f"target_model must be {TARGET_MODEL!r}")
    if (
        type(required_response_schema) is not str
        or required_response_schema != RESPONSE_SCHEMA_VERSION
    ):
        _fail(
            "required_response_schema must be "
            f"{RESPONSE_SCHEMA_VERSION!r}"
        )
    if type(h3_mode) is not str or h3_mode not in H3_MODES:
        _fail(f"h3_mode must be one of {', '.join(H3_MODES)}")
    if type(quality_mode) is not str or quality_mode not in QUALITY_MODES:
        _fail(f"quality_mode must be one of {', '.join(QUALITY_MODES)}")
    if type(research_policy) is not str or research_policy not in RESEARCH_POLICIES:
        _fail(
            "research_policy must be one of " + ", ".join(RESEARCH_POLICIES)
        )
    if type(creative_brief) is not str:
        _fail("creative_brief must be a string")

    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_id": _uuid_text(
            uuid4() if request_id is None else request_id, "request_id"
        ),
        "target_model": TARGET_MODEL,
        "h3_mode": h3_mode,
        "quality_mode": quality_mode,
        "research_policy": research_policy,
        "creative_brief": creative_brief,
        "exact_literals": _validate_exact_literals(exact_literals),
        "generation": _validate_generation(generation),
        "task": _validate_task(task),
        "subjects": _plain_list(subjects, "subjects"),
        "assets": _plain_list(assets, "assets"),
        "budgets": _quality_budget(
            quality_mode, wall_clock_timeout_seconds, budgets
        ),
        "required_response_schema": RESPONSE_SCHEMA_VERSION,
    }
    request["assets"] = _validate_assets(
        request["assets"],
        request_id=cast(str, request["request_id"]),
        h3_mode=h3_mode,
        task=cast(Mapping[str, Any], request["task"]),
    )
    if local_h3_format_guide is not None:
        request["local_h3_format_guide"] = _nonempty_string(
            local_h3_format_guide, "local_h3_format_guide"
        )
    # Validate JSON compatibility and enforce the configured text cap now, not
    # only when the client eventually serializes it.
    serialize_request(request, max_bytes=max_request_bytes)
    return request


def _validate_request_shape(request: Mapping[str, Any]) -> None:
    required = (
        "schema_version", "request_id", "target_model", "h3_mode",
        "quality_mode", "research_policy", "creative_brief", "exact_literals",
        "generation", "task", "subjects", "assets", "budgets",
        "required_response_schema",
    )
    allowed_fields = required + (
        ("local_h3_format_guide",)
        if "local_h3_format_guide" in request
        else ()
    )
    _validate_exact_keys(request, allowed_fields, "request")
    for field in required:
        if field not in request:
            _fail(f"request.{field} is required")
    if (
        type(request["schema_version"]) is not str
        or request["schema_version"] != REQUEST_SCHEMA_VERSION
    ):
        _fail("incompatible request schema_version")
    if (
        type(request["required_response_schema"]) is not str
        or request["required_response_schema"] != RESPONSE_SCHEMA_VERSION
    ):
        _fail("incompatible required_response_schema")
    if type(request["target_model"]) is not str or request["target_model"] != TARGET_MODEL:
        _fail(f"target_model must be {TARGET_MODEL!r}")
    _canonical_uuid_text(request["request_id"], "request.request_id")
    if type(request["h3_mode"]) is not str or request["h3_mode"] not in H3_MODES:
        _fail("invalid h3_mode")
    if (
        type(request["quality_mode"]) is not str
        or request["quality_mode"] not in QUALITY_MODES
    ):
        _fail("invalid quality_mode")
    if (
        type(request["research_policy"]) is not str
        or request["research_policy"] not in RESEARCH_POLICIES
    ):
        _fail("invalid research_policy")
    if type(request["creative_brief"]) is not str:
        _fail("creative_brief must be a string")
    if "local_h3_format_guide" in request:
        _nonempty_string(
            request["local_h3_format_guide"], "local_h3_format_guide"
        )
    _validate_exact_literals(request["exact_literals"])
    _validate_generation(request["generation"])
    validated_task = _validate_task(request["task"])
    _plain_list(request["subjects"], "subjects")
    _validate_assets(
        request["assets"],
        request_id=cast(str, request["request_id"]),
        h3_mode=cast(str, request["h3_mode"]),
        task=validated_task,
    )
    budget = _mapping(request["budgets"], "budgets")
    expected = QUALITY_BUDGETS[request["quality_mode"]]
    _validate_exact_keys(
        budget,
        tuple(expected) + ("wall_clock_timeout_seconds",),
        "budgets",
    )
    for field, value in expected.items():
        if field not in budget:
            _fail(f"budgets.{field} is required")
        actual = _integer(
            budget[field],
            f"budgets.{field}",
            positive=field in ("candidate_count", "tool_call_target"),
        )
        if actual != value:
            _fail(f"budgets.{field} is incompatible with quality_mode")
    if "wall_clock_timeout_seconds" not in budget:
        _fail("budgets.wall_clock_timeout_seconds is required")
    _integer(
        budget["wall_clock_timeout_seconds"],
        "budgets.wall_clock_timeout_seconds",
        positive=True,
    )


def serialize_request(
    request: Mapping[str, Any],
    max_bytes: int = MAX_REQUEST_BYTES,
    *,
    max_request_bytes: Optional[int] = None,
) -> str:
    """Serialize a request deterministically and enforce its UTF-8 byte cap."""

    if max_request_bytes is not None:
        max_bytes = max_request_bytes
    cap = _validate_cap(max_bytes, "max request")
    if is_dataclass(request) and not isinstance(request, type):
        request = asdict(request)
    request_map = _mapping(request, "request")
    request_snapshot = cast(Dict[str, Any], _thaw_json(request_map, "request"))
    _validate_request_shape(request_snapshot)
    try:
        encoded = json.dumps(
            request_snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"request is not valid JSON data: {exc}") from exc
    size = len(encoded.encode("utf-8"))
    if size > cap:
        _fail(f"request text exceeds byte cap ({size} > {cap} bytes)")
    return encoded


def freeze_request_authority(
    request: Mapping[str, Any],
    max_bytes: int = MAX_REQUEST_BYTES,
    *,
    max_request_bytes: Optional[int] = None,
) -> Mapping[str, Any]:
    """Return a validated, canonical, deeply immutable request authority.

    The returned mapping shares no nested containers with ``request``. Nested
    mappings are read-only proxies and JSON arrays are tuples, so the exact
    serialized submission remains authoritative during later result parsing.
    """

    encoded = serialize_request(
        request,
        max_bytes=max_bytes,
        max_request_bytes=max_request_bytes,
    )
    frozen = _deep_freeze_json(json.loads(encoded))
    return cast(Mapping[str, Any], frozen)


def _unique_object(pairs: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    _fail(f"non-standard JSON constant {value!r} is not allowed")


def extract_json_object(
    raw_text: str,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    *,
    max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract exactly one JSON object, optionally inside one `json` fence.

    Apart from surrounding whitespace, prose, multiple objects, unlabeled
    fences, malformed/truncated JSON, duplicate keys, and NaN/Infinity are all
    rejected.
    """

    if max_bytes is not None:
        max_response_bytes = max_bytes
    cap = _validate_cap(max_response_bytes, "max response")
    if not isinstance(raw_text, str):
        _fail("Hermes response must be text")
    size = len(raw_text.encode("utf-8"))
    if size > cap:
        _fail(f"response text exceeds byte cap ({size} > {cap} bytes)")

    stripped = raw_text.strip()
    if not stripped:
        _fail("Hermes response is empty")
    if stripped.startswith("```") or stripped.endswith("```"):
        match = _JSON_FENCE_RE.fullmatch(stripped)
        if match is None:
            _fail("response may contain at most one surrounding Markdown json fence")
        body = match.group("body")
    else:
        body = stripped

    try:
        value = json.loads(
            body,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except ContractError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ContractError(
            "response must contain exactly one complete JSON object"
        ) from exc
    if type(value) is not dict:
        _fail("response JSON root must be an object")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"{field} must be a non-empty string")
    return value


def _required_list(container: Mapping[str, Any], field: str, prefix: str) -> List[Any]:
    if field not in container:
        _fail(f"{prefix}.{field} is required")
    return copy.deepcopy(_list(container[field], f"{prefix}.{field}"))


def _validate_exact_keys(
    container: Mapping[str, Any], fields: Iterable[str], prefix: str
) -> None:
    if any(type(key) is not str for key in container):
        _fail(f"{prefix} JSON object keys must be exact strings")
    expected = set(fields)
    actual = set(container)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: List[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        _fail(f"{prefix} must use exact object keys ({'; '.join(details)})")


def _manifest_label_set(manifest_labels: Any) -> Set[str]:
    if manifest_labels is None:
        return set()
    source: Any = manifest_labels
    if is_dataclass(source) and not isinstance(source, type):
        source = asdict(source)
    if isinstance(source, Mapping):
        if "assets" in source:
            source = source["assets"]
        elif "labels" in source:
            source = source["labels"]
        elif "h3_label" in source:
            source = [source]
        else:
            source = list(source.keys())
    elif hasattr(source, "assets"):
        source = getattr(source, "assets")
    if isinstance(source, str):
        source = [source]
    try:
        items = list(cast(Iterable[Any], source))
    except TypeError as exc:
        raise ContractError("manifest_labels must be a manifest or iterable") from exc

    labels: Set[str] = set()
    for index, item in enumerate(items):
        if is_dataclass(item) and not isinstance(item, type):
            item = asdict(item)
        if isinstance(item, str):
            label = item
        elif isinstance(item, Mapping):
            label = item.get("h3_label")
            if (
                isinstance(label, str)
                and _KEYFRAME_ASSET_LABEL_RE.fullmatch(label) is not None
            ):
                # Analysis-only JPEG evidence is inspectable by tools but is
                # never a physical H3 sampler/reference label.
                continue
        else:
            label = getattr(item, "h3_label", None)
        if not isinstance(label, str) or _H3_LABEL_RE.fullmatch(label) is None:
            _fail(f"manifest label at index {index} is not a real H3 asset label")
        labels.add(label)
    return labels


def _strings_in(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings_in(item)


def _validate_response_labels(
    value: Mapping[str, Any],
    allowed: Set[str],
    *,
    h3_mode: Optional[str],
) -> None:
    used: Set[str] = set()
    for text in _strings_in(value):
        if _KEYFRAME_ASSET_LABEL_RE.search(text):
            _fail(
                "response contains a keyframe evidence label, which is never a "
                "legal H3 response label"
            )
        used.update(match.group(0) for match in _H3_LABEL_RE.finditer(text))
        if h3_mode == "base_FL2VA":
            used.update(
                f"<Picture {match.group(1)}>"
                for match in _FL2VA_BARE_PICTURE_RE.finditer(text)
            )
    unknown = sorted(used - allowed)
    if unknown:
        _fail("response uses H3 asset labels absent from the submitted manifest: "
              + ", ".join(unknown))


def parse_result(
    raw_text: str,
    expected_request_id: Optional[Any] = None,
    manifest_labels: Any = None,
    max_candidates: Optional[int] = None,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    *,
    request_id: Optional[Any] = None,
    submitted_manifest: Any = None,
    allowed_asset_labels: Any = None,
    candidate_limit: Optional[int] = None,
    response_byte_cap: Optional[int] = None,
    request: Optional[Mapping[str, Any]] = None,
) -> ParsedHermesResult:
    """Strictly validate and type one `h3_hermes_result/1.0` object.

    `score_vector` values are intentionally copied without numeric/range
    validation: they are model-reported critique, never local authority.
    Selection is strict: one unique candidate ID must exist and its prompt must
    equal `h3_prompt` byte-for-byte. When `request` is supplied, its ID, asset
    label set, and candidate/repair budgets are authoritative; duplicate legacy
    context is accepted only when it is exactly equivalent.
    """

    request_repair_limit: Optional[int] = None
    authoritative_request_id: Optional[str] = None
    authoritative_labels: Optional[Set[str]] = None
    authoritative_candidate_limit: Optional[int] = None
    authoritative_h3_mode: Optional[str] = None
    if request is not None:
        if is_dataclass(request) and not isinstance(request, type):
            request = asdict(request)
        request_map = cast(
            Dict[str, Any],
            _thaw_json(_mapping(request, "request"), "request"),
        )
        _validate_request_shape(request_map)
        authoritative_request_id = _canonical_uuid_text(
            request_map["request_id"], "request.request_id"
        )
        authoritative_labels = _manifest_label_set(request_map["assets"])
        authoritative_h3_mode = cast(str, request_map["h3_mode"])
        budget = cast(Mapping[str, int], request_map["budgets"])
        authoritative_candidate_limit = budget["candidate_count"]
        request_repair_limit = budget["max_repairs"]

    request_id_inputs: List[str] = []
    for field, supplied in (
        ("expected request_id", expected_request_id),
        ("request_id", request_id),
    ):
        if supplied is None:
            continue
        canonical = _canonical_uuid_text(supplied, field)
        request_id_inputs.append(canonical)
        if (authoritative_request_id is not None
                and canonical != authoritative_request_id):
            _fail(f"conflicting {field}: does not match request.request_id")
    if len(set(request_id_inputs)) > 1:
        _fail("conflicting expected request_id values")
    if authoritative_request_id is not None:
        expected = authoritative_request_id
    elif request_id_inputs:
        expected = request_id_inputs[0]
    else:
        _fail("expected request_id must be a UUID string")

    label_inputs: List[Set[str]] = []
    for supplied in (manifest_labels, submitted_manifest, allowed_asset_labels):
        if supplied is None:
            continue
        labels = _manifest_label_set(supplied)
        label_inputs.append(labels)
        if authoritative_labels is not None and labels != authoritative_labels:
            _fail("conflicting manifest label context does not match request.assets")
    if label_inputs and any(labels != label_inputs[0]
                            for labels in label_inputs[1:]):
        _fail("conflicting manifest label inputs")
    if authoritative_labels is not None:
        allowed_labels = authoritative_labels
    elif label_inputs:
        allowed_labels = label_inputs[0]
    else:
        allowed_labels = set()

    candidate_limits: List[int] = []
    for field, supplied in (
        ("max_candidates", max_candidates),
        ("candidate_limit", candidate_limit),
    ):
        if supplied is None:
            continue
        limit = _integer(supplied, field, positive=True)
        candidate_limits.append(limit)
        if (authoritative_candidate_limit is not None
                and limit != authoritative_candidate_limit):
            _fail(f"conflicting candidate limit in {field}")
    if len(set(candidate_limits)) > 1:
        _fail("conflicting candidate limit inputs")
    if authoritative_candidate_limit is not None:
        resolved_candidate_limit = authoritative_candidate_limit
    elif candidate_limits:
        resolved_candidate_limit = candidate_limits[0]
    else:
        resolved_candidate_limit = 3
    if response_byte_cap is not None:
        max_response_bytes = response_byte_cap

    value = extract_json_object(raw_text, max_response_bytes=max_response_bytes)
    _validate_exact_keys(value, _RESPONSE_FIELDS, "response")
    if value.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        _fail(
            "missing or incompatible response schema_version; expected "
            f"{RESPONSE_SCHEMA_VERSION!r}"
        )
    wire_request_id = value.get("request_id")
    actual_request_id = _uuid_text(wire_request_id, "response request_id")
    if wire_request_id != actual_request_id:
        _fail("response request_id must use exact canonical UUID spelling")
    if wire_request_id != expected:
        _fail(
            f"response request_id {wire_request_id!r} does not match "
            f"request_id {expected!r}"
        )
    if value.get("status") != "ok":
        _fail("response status must be 'ok'")

    h3_prompt = _nonempty_string(value.get("h3_prompt"), "h3_prompt")

    evidence = _plain_mapping(value.get("evidence"), "evidence")
    _validate_exact_keys(evidence, _EVIDENCE_FIELDS, "evidence")
    for field in _EVIDENCE_FIELDS:
        _required_list(evidence, field, "evidence")

    intent_ir = _plain_mapping(value.get("intent_ir"), "intent_ir")
    _validate_exact_keys(intent_ir, _INTENT_IR_FIELDS, "intent_ir")
    for field in _INTENT_IR_FIELDS:
        _required_list(intent_ir, field, "intent_ir")

    candidate_values = _list(value.get("candidates"), "candidates")
    if not candidate_values:
        _fail("candidates must contain at least one candidate")
    if len(candidate_values) > resolved_candidate_limit:
        _fail(
            f"candidate count {len(candidate_values)} exceeds limit "
            f"{resolved_candidate_limit}"
        )

    candidates: List[HermesCandidate] = []
    seen_ids: Set[str] = set()
    for index, item in enumerate(candidate_values):
        candidate = _mapping(item, f"candidates[{index}]")
        _validate_exact_keys(
            candidate, _CANDIDATE_FIELDS, f"candidates[{index}]"
        )
        candidate_id = _nonempty_string(
            candidate.get("candidate_id"), f"candidates[{index}].candidate_id"
        )
        if candidate_id in seen_ids:
            _fail(f"duplicate candidate_id {candidate_id!r}")
        seen_ids.add(candidate_id)
        policy = _nonempty_string(
            candidate.get("policy"), f"candidates[{index}].policy"
        )
        prompt = _nonempty_string(
            candidate.get("prompt"), f"candidates[{index}].prompt"
        )
        if "score_vector" not in candidate:
            _fail(f"candidates[{index}].score_vector is required")
        score_vector = _plain_mapping(
            candidate["score_vector"], f"candidates[{index}].score_vector"
        )
        if "critic_findings" not in candidate:
            _fail(f"candidates[{index}].critic_findings is required")
        critic_findings = copy.deepcopy(
            _list(candidate["critic_findings"],
                  f"candidates[{index}].critic_findings")
        )
        candidates.append(HermesCandidate(
            candidate_id=candidate_id,
            policy=policy,
            prompt=prompt,
            score_vector=score_vector,
            critic_findings=critic_findings,
        ))

    selected_id = _nonempty_string(
        value.get("selected_candidate_id"), "selected_candidate_id"
    )
    selected = [item for item in candidates if item.candidate_id == selected_id]
    if len(selected) != 1:
        _fail("selected_candidate_id must identify exactly one candidate")
    if selected[0].prompt != h3_prompt:
        _fail("selected candidate prompt must exactly match h3_prompt")

    if "repairs" not in value:
        _fail("repairs is required")
    repairs = copy.deepcopy(_list(value["repairs"], "repairs"))
    if request_repair_limit is not None and len(repairs) > request_repair_limit:
        _fail(
            f"repair count {len(repairs)} exceeds request limit "
            f"{request_repair_limit}"
        )
    quality_report = _plain_mapping(value.get("quality_report"), "quality_report")
    _validate_exact_keys(
        quality_report, _QUALITY_REPORT_FIELDS, "quality_report"
    )
    for field in _QUALITY_REPORT_FIELDS:
        _required_list(quality_report, field, "quality_report")
    reported_tools = copy.deepcopy(quality_report["reported_tools"])
    reported_sources = copy.deepcopy(quality_report["reported_sources"])

    _validate_response_labels(
        value,
        allowed_labels,
        h3_mode=authoritative_h3_mode,
    )

    return ParsedHermesResult(
        schema_version=RESPONSE_SCHEMA_VERSION,
        request_id=actual_request_id,
        status="ok",
        evidence=evidence,
        intent_ir=intent_ir,
        candidates=candidates,
        selected_candidate_id=selected_id,
        h3_prompt=h3_prompt,
        repairs=repairs,
        quality_report=quality_report,
        reported_tools=reported_tools,
        reported_sources=reported_sources,
    )


__all__ = [
    "ContractError",
    "HermesContractError",
    "STABLE_INSTRUCTIONS",
    "H3_HERMES_INSTRUCTIONS",
    "HERMES_INSTRUCTIONS",
    "build_request",
    "freeze_request_authority",
    "serialize_request",
    "extract_json_object",
    "parse_result",
]
