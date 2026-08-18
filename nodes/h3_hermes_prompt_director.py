"""Hermes-backed MiniMax H3 prompt director for secure Base-mode jobs.

The node deliberately keeps deterministic H3 formatting, timing, and output
wiring local. Hermes supplies candidate prompt prose through the asynchronous
Runs API; it never uses the legacy node's provider backend.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import math
from numbers import Real
import os
import re
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from urllib.parse import urlsplit
from uuid import uuid4

import comfy.model_management
import folder_paths

from .ultimate_h3_cowboy_promptor import (
    DEFAULT_BASE_DURATION,
    MUSIC_SOURCES,
    NUM_SUBJECT_SLOTS,
    UltimateH3CowboyPromptor,
)
from ..utils.h3_cowboy import prompts_base, prompts_ref, spec
from ..utils.h3_cowboy.assembler import PICTURES_FOR_MODE, CowboyContext, process
from ..utils.h3_cowboy.subjects import (
    bind_images,
    merge_text_subjects,
    subjects_from_rows,
)
from ..utils.h3_cowboy.wiring import H3_FPS, build_label_map
from ..utils.h3_prompt.keyframes import select_keyframes
from ..utils.h3_hermes.assets import (
    AssetDirective,
    AssetLimits,
    cleanup_assets,
    stage_assets,
    verified_manifest_snapshot,
    verify_staged_assets,
)
from ..utils.h3_hermes.client import HermesRunsClient
from ..utils.h3_hermes.contract import (
    ContractError,
    STABLE_INSTRUCTIONS,
    build_request,
    freeze_request_authority,
    parse_result,
    serialize_request,
)
from ..utils.h3_prompt.prompts import MAX_PROMPT_CHARS


DEFAULT_HERMES_BASE_URL = "http://127.0.0.1:8642"
TIMEOUT_MIN_SECONDS = 30
TIMEOUT_MAX_SECONDS = 3600
POLL_MIN_SECONDS = 0.05
POLL_MAX_SECONDS = 10.0
CLEANUP_POLICIES = ("delete_on_success", "retain_24h", "retain")
MAX_ANALYSIS_STRING_CHARS = 1024
MAX_ANALYSIS_LIST_ITEMS = 32
MAX_ANALYSIS_REPORTED_TOOLS = 16
MAX_HASHED_IDENTIFIER_UTF8_BYTES = 4096
ROUTE_MAX_CHARS = 256
ROUTE_MAX_UTF8_BYTES = 1024
LANGUAGE_PREFIX_MAX_CHARS = 64
DEFAULT_TEXT_CANVAS = (768, 768)
H3_MIN_DURATION_SECONDS = 4.0
H3_MAX_DURATION_SECONDS = 15.0

_SUCCESS_RUN_STATUSES = frozenset({"completed"})
_ALLOWED_REPORTED_TOOLS = frozenset({
    "web_search", "web_extract", "vision", "vision_analyze", "video_analyze",
})
_ALLOWED_SCORE_FIELDS = (
    "required_intent_coverage",
    "contradictions",
    "h3_format_compliance",
    "literal_preservation",
    "temporal_coherence",
    "asset_grounding",
    "creative_quality",
    "research_quality",
    "economy",
)
_ALLOWED_BASE_PICTURE_ROLES = frozenset({"auto", "first_frame", "last_frame"})
_KEYFRAME_EVIDENCE_LABEL_RE = re.compile(
    r"<Video 1 Keyframe ([1-9][0-9]{0,3})>"
)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:[\\/]|\\\\|(?<![\w/])/(?:[^\s/]+/)*[^\s/]*)"
)
_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://\S+")
_BEARER_RE = re.compile(r"(?i)\bbearer(?:\s+|\s*[:=]\s*)\S+")
_ASSIGNMENT_RE = re.compile(
    r"(?i)(?<![\w.-])[a-z_][a-z0-9_.-]{0,63}\s*[:=]\s*\S+"
)
_OPAQUE_KEY_RE = re.compile(
    r"(?i)(?<![a-z0-9_-])(?:sk|pk|rk|ghp|github_pat|xox[baprs]|akia)"
    r"[-_][a-z0-9_-]{12,}(?![a-z0-9_-])"
)
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _loopback_base_url(value: str) -> str:
    """Return a canonical, root-only plain-HTTP loopback URL."""

    if type(value) is not str or not value:
        raise RuntimeError("Hermes requires a non-blank loopback HTTP URL.")
    if any(
        ord(char) < 32
        or 0x7F <= ord(char) <= 0x9F
        or char.isspace()
        for char in value
    ):
        raise RuntimeError(
            "Hermes loopback URL must not contain whitespace or controls."
        )
    candidate = value
    try:
        parsed = urlsplit(candidate)
        # Accessing port is what validates malformed/out-of-range port text.
        port = parsed.port
        hostname = parsed.hostname
    except ValueError:
        raise RuntimeError("Hermes requires a valid loopback HTTP URL.") from None

    if parsed.scheme.lower() != "http":
        raise RuntimeError("Hermes loopback URL must use plain http.")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("Hermes loopback URL must not contain credentials.")
    if parsed.query or "?" in candidate:
        raise RuntimeError("Hermes loopback URL must not contain a query.")
    if parsed.fragment or "#" in candidate:
        raise RuntimeError("Hermes loopback URL must not contain a fragment.")
    if parsed.path not in ("", "/"):
        raise RuntimeError("Hermes loopback URL must point at the server root.")

    if not hostname or "%" in hostname or port != 8642:
        raise RuntimeError("Hermes requires a valid loopback HTTP URL.")
    if hostname.lower() == "localhost":
        canonical_host = "localhost"
    else:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            raise RuntimeError(
                "Hermes base URL must use a loopback host."
            ) from None
        if not address.is_loopback:
            raise RuntimeError("Hermes base URL must use a loopback host.")
        canonical_host = address.compressed
        if address.version == 6:
            canonical_host = f"[{canonical_host}]"

    return f"http://{canonical_host}:8642"


def _optional_route(value: Any, name: str) -> str:
    """Validate one optional route selector before trimming its outer spaces."""

    if type(value) is not str:
        raise RuntimeError(f"Hermes {name} route must be text.")
    if len(value) > ROUTE_MAX_CHARS:
        raise RuntimeError(f"Hermes {name} route is too large.")
    if any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in value):
        raise RuntimeError(f"Hermes {name} route contains control characters.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"Hermes {name} route is not valid UTF-8 text."
        ) from exc
    if len(encoded) > ROUTE_MAX_UTF8_BYTES:
        raise RuntimeError(f"Hermes {name} route is too large.")
    return value.strip()


def _bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise RuntimeError(
            f"{name} must be between {minimum:g} and {maximum:g}."
        )
    return number


def _exact_boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"{name} must be a boolean.")
    return value


def _exact_enum(value: Any, name: str, allowed: Sequence[str]) -> str:
    if type(value) is not str or value not in allowed:
        if len(allowed) > 1:
            choices = ", ".join(allowed[:-1]) + f", or {allowed[-1]}"
        else:
            choices = allowed[0]
        raise RuntimeError(f"{name} must be {choices}.")
    return value


def _apply_duration_policy(
    duration_seconds: float,
    *,
    strict: bool,
    warnings: list[str],
) -> bool:
    supported = H3_MIN_DURATION_SECONDS <= duration_seconds <= H3_MAX_DURATION_SECONDS
    if supported:
        return True
    message = (
        "target duration is outside H3's official 4 to 15 second range; "
        "generation may be unsupported"
    )
    if strict:
        raise RuntimeError(
            "Target duration is outside H3's official 4 to 15 second range."
        )
    warnings.append(message)
    return False


def _safe_hard_error_categories(values: Iterable[Any]) -> list[str]:
    """Return only fixed local validator categories, never error payload text."""

    allowed = (
        "R1 FORMAT",
        "R2 LABELS",
        "R3 VERBATIM",
        "R4 STRUCTURE",
        "R5 EMPTY",
        "R6 MUSIC",
    )
    result: list[str] = []
    for value in values:
        text = value if type(value) is str else ""
        category = next(
            (item for item in allowed if text.startswith(item + ":")),
            None,
        )
        if category is None and text.startswith("The prompt exceeds the hard cap"):
            category = (
                f"PROMPT_LENGTH_LIMIT: maximum {MAX_PROMPT_CHARS} characters"
            )
        if category is None:
            category = "LOCAL_VALIDATION_ERROR"
        if category not in result:
            result.append(category)
    return result


def _secret_values() -> Tuple[str, ...]:
    value = os.environ.get("HERMES_AGENT_API_KEY", "")
    return (value,) if value else ()


def _text_digest(value: str) -> Tuple[str, int, int]:
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RuntimeError("Hermes returned invalid UTF-8 metadata.") from exc
    return hashlib.sha256(encoded).hexdigest(), len(value), len(encoded)


def _remote_identifier_fields(prefix: str, value: Any) -> Dict[str, Any]:
    """Expose an untrusted remote identifier only as digest and size metadata."""

    if type(value) is not str:
        raise RuntimeError("Hermes returned invalid identifier metadata.")
    digest, chars, byte_count = _text_digest(value)
    if not byte_count or byte_count > MAX_HASHED_IDENTIFIER_UTF8_BYTES:
        raise RuntimeError("Hermes returned invalid identifier metadata.")
    return {
        f"{prefix}_sha256": digest,
        f"{prefix}_char_count": chars,
        f"{prefix}_utf8_byte_count": byte_count,
    }


def _safe_run_id(value: Any) -> str:
    if type(value) is not str or _RUN_ID_RE.fullmatch(value) is None:
        raise RuntimeError("Hermes returned invalid run ID metadata.")
    return value


def _safe_run_status(value: Any) -> str:
    if type(value) is not str or value not in _SUCCESS_RUN_STATUSES:
        raise RuntimeError("Hermes run returned invalid status metadata.")
    return value


def _safe_elapsed_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Hermes run returned invalid elapsed metadata.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RuntimeError("Hermes run returned invalid elapsed metadata.")
    return round(number, 3)


def _diagnostic_digest(text: str) -> str:
    encoded = text.encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(encoded).hexdigest()
    return (
        f"diagnostic omitted; sha256 {digest}; chars {len(text)}; "
        f"bytes {len(encoded)}"
    )


def _safe_local_diagnostic(value: Any) -> str:
    """Copy only bounded local diagnostics with no credential/path hazards."""

    text = value if type(value) is str else str(value)
    unsafe = (
        len(text) > MAX_ANALYSIS_STRING_CHARS
        or any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in text)
        or any(secret in text for secret in _secret_values())
        or _ABSOLUTE_PATH_RE.search(text) is not None
        or _URL_RE.search(text) is not None
        or _BEARER_RE.search(text) is not None
        or _ASSIGNMENT_RE.search(text) is not None
        or _OPAQUE_KEY_RE.search(text) is not None
    )
    if unsafe:
        return _diagnostic_digest(text)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return _diagnostic_digest(text)
    return text


def _safe_local_diagnostics(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [
        _safe_local_diagnostic(value)
        for value in values[:MAX_ANALYSIS_LIST_ITEMS]
    ]


def _safe_usage(usage: Mapping[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if type(value) is int and value >= 0:
            safe[key] = value
    return safe


def _safe_numeric_scores(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    safe: Dict[str, Any] = {}
    for key in _ALLOWED_SCORE_FIELDS:
        score = value.get(key)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            if not isinstance(score, float) or math.isfinite(score):
                safe[key] = score
    return safe


def _safe_reported_tools(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values[:MAX_ANALYSIS_REPORTED_TOOLS]:
        if (
            type(value) is str
            and value in _ALLOWED_REPORTED_TOOLS
            and value not in seen
        ):
            result.append(value)
            seen.add(value)
    return result


def _candidate_handoff(candidate: Any) -> Dict[str, Any]:
    summary = _remote_identifier_fields("candidate_id", candidate.candidate_id)
    prompt_digest, prompt_chars, prompt_bytes = _text_digest(candidate.prompt)
    summary.update({
        "prompt_sha256": prompt_digest,
        "prompt_char_count": prompt_chars,
        "prompt_utf8_byte_count": prompt_bytes,
        "score_vector": _safe_numeric_scores(candidate.score_vector),
        "critic_finding_count": (
            len(candidate.critic_findings)
            if isinstance(candidate.critic_findings, (list, tuple))
            else 0
        ),
    })
    return summary


def _visible_literals(value: str) -> list[str]:
    if not isinstance(value, str):
        raise RuntimeError("visible_text must be text.")
    # Blank rows are display spacing, not visible text requirements. Nonblank
    # rows retain their whitespace exactly because this is an exact literal.
    return [line for line in value.splitlines() if line.strip()]


def _spoken_blocks(prompt: str) -> Tuple[str, ...]:
    """Extract well-formed ``<d>`` contents without normalizing their bytes."""

    malformed_probe = prompt.replace("<d>", "").replace("</d>", "")
    if re.search(r"<\s*/?\s*d(?:\s|/|>)", malformed_probe, re.IGNORECASE):
        raise ValueError("malformed <d> tag")

    blocks = []
    cursor = 0
    while True:
        opening = prompt.find("<d>", cursor)
        closing = prompt.find("</d>", cursor)
        if opening < 0:
            if closing >= 0:
                raise ValueError("closing <d> tag without an opening tag")
            return tuple(blocks)
        if 0 <= closing < opening:
            raise ValueError("closing <d> tag before an opening tag")

        content_start = opening + len("<d>")
        closing = prompt.find("</d>", content_start)
        if closing < 0:
            raise ValueError("unclosed <d> tag")
        if prompt.find("<d>", content_start, closing) >= 0:
            raise ValueError("nested <d> tag")

        blocks.append(prompt[content_start:closing])
        cursor = closing + len("</d>")


def _spoken_payloads(block: str) -> Tuple[str, ...]:
    """Return the exact block and its one optional language-prefixed payload."""

    payloads = [block]
    if block.startswith("["):
        closing = block.find("]")
        if (
            1 <= closing <= LANGUAGE_PREFIX_MAX_CHARS
            and closing + 1 < len(block)
            and block[closing + 1] == " "
        ):
            label = block[1:closing]
            if (
                "[" not in label
                and "]" not in label
                and not any(
                    ord(char) < 32 or 0x7F <= ord(char) <= 0x9F
                    for char in label
                )
            ):
                payloads.append(block[closing + 2:])
    return tuple(payloads)


def _missing_ordered_spoken_literals(
    blocks: Tuple[str, ...], dialogue: Any, lyrics: Any
) -> list[str]:
    """Bind exact dialogue then lyrics to distinct, ordered ``<d>`` blocks."""

    required = [
        (label, literal)
        for label, literal in (("dialogue", dialogue), ("lyrics", lyrics))
        if isinstance(literal, str) and literal.strip()
    ]
    missing: list[str] = []
    next_block = 0
    for label, literal in required:
        match = next(
            (
                index for index in range(next_block, len(blocks))
                if literal in _spoken_payloads(blocks[index])
            ),
            None,
        )
        if match is None:
            missing.append(label)
        else:
            next_block = match + 1
    return missing


def _integrated_description_section(prompt: str) -> str:
    """Extract only the Base visible-action section before soundscape/music."""

    marker = "integrated_multimodal_description:"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    boundaries = [
        position for section in ("overall_soundscape:", "non_diegetic_music:")
        if (position := prompt.find(section, start)) >= 0
    ]
    end = min(boundaries) if boundaries else len(prompt)
    return prompt[start:end]


def _ref_description_section(prompt: str) -> str:
    """Extract only Ref2VA's visible-action section for literal checks."""

    marker = "detailed_description:"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    boundaries = [
        position for section in ("overall_soundscape:", "non_diegetic_music:")
        if (position := prompt.find(section, start)) >= 0
    ]
    end = min(boundaries) if boundaries else len(prompt)
    return prompt[start:end]


def _validated_uninspected_labels(
    values: Any, submitted_labels: Iterable[str]
) -> list[str]:
    """Keep only deduplicated, contract-validated submitted physical labels."""

    if not isinstance(values, (list, tuple)):
        return []
    allowed = frozenset(submitted_labels)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if (
            type(value) is str
            and value in allowed
            and value not in seen
        ):
            result.append(value)
            seen.add(value)
            if len(result) >= MAX_ANALYSIS_LIST_ITEMS:
                break
    return result


def _count_list(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 0


def _base_asset_directives(h3_mode: str) -> Dict[str, AssetDirective]:
    """Return ordered, mode-authoritative transfer instructions for anchors."""

    first = AssetDirective(
        intended_jobs=("first_frame", "appearance", "identity"),
        prohibited_transfers=(),
    )
    last = AssetDirective(
        intended_jobs=("last_frame", "continuity"),
        prohibited_transfers=("audio",),
    )
    if h3_mode == "base_I2VA":
        return {"<Picture 1>": first}
    if h3_mode == "base_FL2VA":
        return {"<Picture 1>": first, "<Picture 2>": last}
    if h3_mode == "base_L2VA":
        return {"<Picture 1>": last}
    return {}


def _ref_asset_directives(
    picture_slots: Iterable[int], *, has_audio: bool
) -> Dict[str, AssetDirective]:
    """Freeze the ref asset roles without inferring transfer from content."""

    directives: Dict[str, AssetDirective] = {}
    for slot in picture_slots:
        directives[f"<Picture {slot}>"] = AssetDirective(
            intended_jobs=("identity", "appearance"),
            prohibited_transfers=("pose", "motion", "audio"),
        )
    directives["<Video 1>"] = AssetDirective(
        intended_jobs=("pose", "motion", "camera", "timing"),
        prohibited_transfers=("identity", "appearance", "audio"),
    )
    if has_audio:
        directives["<Audio 1>"] = AssetDirective(
            intended_jobs=("audio", "timing"),
            prohibited_transfers=("identity", "appearance", "pose", "motion"),
        )
    return directives


def _trusted_video_source_roots() -> Tuple[str, ...]:
    """Return only roots declared by ComfyUI itself, when those APIs exist."""

    roots: list[str] = []
    for name in (
        "get_input_directory", "get_output_directory", "get_temp_directory",
    ):
        getter = getattr(folder_paths, name, None)
        if not callable(getter):
            continue
        try:
            value = getter()
        except Exception:
            continue
        if isinstance(value, (str, os.PathLike)) and os.fspath(value):
            text = os.fspath(value)
            if text not in roots:
                roots.append(text)
    return tuple(roots)


def _ref_subject_payload(subjects: Iterable[Any]) -> list[Dict[str, Any]]:
    """Detach the stable V1 subject identity into bounded request data."""

    return [
        {
            "index": int(subject.index),
            "kind": str(subject.kind),
            "name": str(subject.name),
            "sources": [str(source) for source in subject.sources],
            "features": str(subject.features),
            "slot": None if subject.slot is None else int(subject.slot),
        }
        for subject in subjects
    ]


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _external_required_asset_authority(
    *, music_video: bool, music_source: str, audio: Any
) -> list[Dict[str, str]]:
    """Authorize one explicit downstream-only label without claiming staging."""

    if music_video and music_source == "reuse_audio_1" and audio is None:
        return [{
            "h3_label": "<Audio 1>",
            "authority": "downstream_required_external",
            "inspection_status": "uninspected",
        }]
    return []


class H3HermesPromptDirector(UltimateH3CowboyPromptor):
    """Direct one Base H3 prompt job through Hermes Agent."""

    @classmethod
    def INPUT_TYPES(cls):
        v1 = UltimateH3CowboyPromptor.INPUT_TYPES()
        v1_required = v1["required"]
        h3_mode = copy.deepcopy(v1_required["h3_mode"])
        h3_mode[1]["default"] = "base_T2VA"
        h3_mode[1]["tooltip"] = (
            "Supports secure staged Ref2VA video/frames with optional audio, "
            "plus all four Base modes: text-only T2VA, first-frame I2VA, "
            "first/last-frame FL2VA, and last-frame L2VA."
        )

        required = {
            "h3_mode": h3_mode,
            "subjects": copy.deepcopy(v1_required["subjects"]),
            "target_description": copy.deepcopy(
                v1_required["target_description"]
            ),
            "quality_mode": (["fast", "balanced", "hero"], {
                "default": "balanced",
                "tooltip": "Bounded Hermes candidate/review budget.",
            }),
            "research_policy": (["never", "when_uncertain", "always"], {
                "default": "when_uncertain",
                "tooltip": "When Hermes may research current H3 guidance.",
            }),
        }

        optional = copy.deepcopy(v1["optional"])
        optional.pop("api_key", None)
        optional["visible_text"] = ("STRING", {
            "multiline": True,
            "default": "",
            "tooltip": (
                "One exact visible-text literal per line. Whitespace on "
                "nonblank lines is preserved."
            ),
        })
        optional["hermes_base_url"] = ("STRING", {
            "default": DEFAULT_HERMES_BASE_URL,
            "advanced": True,
            "tooltip": "Root-only plain-HTTP loopback Hermes API URL.",
        })
        optional["timeout_seconds"] = ("INT", {
            "default": 900,
            "min": TIMEOUT_MIN_SECONDS,
            "max": TIMEOUT_MAX_SECONDS,
            "step": 1,
            "advanced": True,
            "tooltip": "Hard wall-clock limit; timeout requests run stop.",
        })
        optional["poll_interval_seconds"] = ("FLOAT", {
            "default": 1.0,
            "min": POLL_MIN_SECONDS,
            "max": POLL_MAX_SECONDS,
            "step": 0.05,
            "advanced": True,
            "tooltip": "Runs API polling interval.",
        })
        optional["strict_duration"] = ("BOOLEAN", {
            "default": False,
            "advanced": True,
            "tooltip": (
                "Reject durations outside H3's official 4 to 15 second range. "
                "When off, warn and preserve V1-compatible output."
            ),
        })
        optional["cleanup_policy"] = (list(CLEANUP_POLICIES), {
            "default": "delete_on_success",
            "advanced": True,
            "tooltip": "Cleanup/retention policy for this request's staged images.",
        })
        optional["hermes_provider"] = ("STRING", {
            "default": "",
            "advanced": True,
            "tooltip": "Optional Hermes route provider; blank uses gateway default.",
        })
        optional["hermes_model"] = ("STRING", {
            "default": "",
            "advanced": True,
            "tooltip": "Optional Hermes route model; blank uses gateway default.",
        })
        return {"required": required, "optional": optional}

    RETURN_TYPES = UltimateH3CowboyPromptor.RETURN_TYPES
    RETURN_NAMES = UltimateH3CowboyPromptor.RETURN_NAMES
    OUTPUT_TOOLTIPS = UltimateH3CowboyPromptor.OUTPUT_TOOLTIPS
    FUNCTION = "generate"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "Uses Hermes Agent to research, draft, critique, and select a MiniMax "
        "H3 prompt while TrentNodes keeps timing and format validation local. "
        "Supports secure Ref2VA video/frames with optional audio and all Base "
        "modes with untouched reference pass-throughs."
    )

    def generate(
        self,
        h3_mode: str,
        subjects: str,
        target_description: str,
        quality_mode: str,
        research_policy: str,
        video=None,
        frames=None,
        fps: float = 24.0,
        audio=None,
        video_role: str = "subject_source",
        audio_role: str = "none",
        cut_times: str = "",
        dialogue: str = "",
        constraint_notes: str = "",
        duration_override: float = 0.0,
        max_frames_to_analyze: int = 8,
        seed: int = 0,
        base_picture_role: str = "first_frame",
        fl2va_normalize_picture_tags: bool = False,
        snap_duration_to_h3_grid: bool = True,
        subject_rows: int = NUM_SUBJECT_SLOTS,
        music_video: bool = False,
        music_source: str = "auto",
        lyrics: str = "",
        music_description: str = "",
        visible_text: str = "",
        hermes_base_url: str = DEFAULT_HERMES_BASE_URL,
        timeout_seconds: int = 900,
        poll_interval_seconds: float = 1.0,
        strict_duration: bool = False,
        cleanup_policy: str = "delete_on_success",
        hermes_provider: str = "",
        hermes_model: str = "",
        **subject_fields,
    ) -> Tuple:
        strict_duration = _exact_boolean(strict_duration, "strict_duration")
        h3_mode = _exact_enum(
            h3_mode, "h3_mode", ("ref", *tuple(PICTURES_FOR_MODE))
        )
        quality_mode = _exact_enum(
            quality_mode, "quality_mode", ("fast", "balanced", "hero")
        )
        research_policy = _exact_enum(
            research_policy,
            "research_policy",
            ("never", "when_uncertain", "always"),
        )
        if h3_mode == "ref":
            video_role = _exact_enum(
                video_role, "video_role", tuple(spec.VIDEO_ROLES)
            )
            audio_role = _exact_enum(
                audio_role, "audio_role", tuple(spec.AUDIO_ROLES)
            )
            music_source = _exact_enum(
                music_source, "music_source", MUSIC_SOURCES
            )
        elif h3_mode != "base_T2VA":
            base_picture_role = _exact_enum(
                base_picture_role,
                "base_picture_role",
                tuple(sorted(_ALLOWED_BASE_PICTURE_ROLES)),
            )
        cleanup_policy = _exact_enum(
            cleanup_policy, "cleanup_policy", CLEANUP_POLICIES
        )

        if h3_mode == "ref":
            return self._generate_ref(
                subjects=subjects,
                target_description=target_description,
                quality_mode=quality_mode,
                research_policy=research_policy,
                video=video,
                frames=frames,
                fps=fps,
                audio=audio,
                video_role=video_role,
                audio_role=audio_role,
                cut_times=cut_times,
                dialogue=dialogue,
                constraint_notes=constraint_notes,
                duration_override=duration_override,
                max_frames_to_analyze=max_frames_to_analyze,
                base_picture_role=base_picture_role,
                fl2va_normalize_picture_tags=fl2va_normalize_picture_tags,
                snap_duration_to_h3_grid=snap_duration_to_h3_grid,
                subject_rows=subject_rows,
                music_video=music_video,
                music_source=music_source,
                lyrics=lyrics,
                music_description=music_description,
                visible_text=visible_text,
                hermes_base_url=hermes_base_url,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                strict_duration=strict_duration,
                cleanup_policy=cleanup_policy,
                hermes_provider=hermes_provider,
                hermes_model=hermes_model,
                subject_fields=subject_fields,
            )

        del fps, max_frames_to_analyze, seed, subject_rows, music_source

        if h3_mode not in PICTURES_FOR_MODE:
            raise RuntimeError(
                "h3_mode must be ref or one of the supported Base modes."
            )
        if video is not None or frames is not None or audio is not None:
            raise RuntimeError(
                "H3 Hermes Prompt Director does not support video, frames, or "
                "audio inputs in this Base image slice."
            )
        fl2va_normalize_picture_tags = _exact_boolean(
            fl2va_normalize_picture_tags,
            "fl2va_normalize_picture_tags",
        )
        snap_duration_to_h3_grid = _exact_boolean(
            snap_duration_to_h3_grid,
            "snap_duration_to_h3_grid",
        )
        if quality_mode not in ("fast", "balanced", "hero"):
            raise RuntimeError("quality_mode must be fast, balanced, or hero.")
        if research_policy not in ("never", "when_uncertain", "always"):
            raise RuntimeError(
                "research_policy must be never, when_uncertain, or always."
            )

        wired = [
            slot for slot in range(1, NUM_SUBJECT_SLOTS + 1)
            if subject_fields.get(f"subject_{slot}_image") is not None
        ]
        if wired and wired != list(range(1, wired[-1] + 1)):
            missing = sorted(set(range(1, wired[-1] + 1)) - set(wired))
            raise RuntimeError(
                "Wired physical picture slots contain a gap; fill physical slot "
                + ", ".join(str(slot) for slot in missing)
                + " before using a later image slot."
            )

        base_url = _loopback_base_url(hermes_base_url)
        provider = _optional_route(hermes_provider, "provider")
        model = _optional_route(hermes_model, "model")
        timeout = _bounded_number(
            timeout_seconds, "timeout_seconds",
            TIMEOUT_MIN_SECONDS, TIMEOUT_MAX_SECONDS,
        )
        if not timeout.is_integer():
            raise RuntimeError("timeout_seconds must be a whole number of seconds.")
        poll_interval = _bounded_number(
            poll_interval_seconds, "poll_interval_seconds",
            POLL_MIN_SECONDS, POLL_MAX_SECONDS,
        )
        if cleanup_policy not in CLEANUP_POLICIES:
            raise RuntimeError(
                "cleanup_policy must be one of " + ", ".join(CLEANUP_POLICIES)
            )
        if isinstance(duration_override, bool) or not isinstance(
            duration_override, (int, float)
        ):
            raise RuntimeError("duration_override must be a number.")
        requested_duration = float(duration_override)
        if not math.isfinite(requested_duration) or requested_duration < 0:
            raise RuntimeError("duration_override must be finite and nonnegative.")
        warnings: list[str] = []
        anchors = self._resolve_anchors(h3_mode, subject_fields, warnings)
        self._warn_ref_only_widgets(
            h3_mode,
            subjects,
            video,
            frames,
            audio,
            video_role,
            audio_role,
            base_picture_role,
            fl2va_normalize_picture_tags,
            warnings,
            rows=self._read_rows(subject_fields),
            music_video=music_video,
            music_text=(lyrics + music_description),
        )
        # Base modes have no music-video semantics. Preserve the V1 warning
        # above, but never promote stale hidden widget text to request or
        # exact-literal authority.
        lyrics = ""
        music_description = ""
        if requested_duration == 0:
            requested_duration = DEFAULT_BASE_DURATION
            duration_source = "default"
            warnings.append(
                f"no video, no frames and no duration_override, so the "
                f"target duration defaulted to {DEFAULT_BASE_DURATION:.2f}s. "
                "That number is written into the instruction line and the "
                "shot times; set duration_override to the length you "
                "actually want."
            )
        else:
            duration_source = "override"

        duration_supported = _apply_duration_policy(
            requested_duration,
            strict=strict_duration,
            warnings=warnings,
        )
        length, prompt_duration = self._snap(
            requested_duration, snap_duration_to_h3_grid, warnings
        )
        snapped_duration = length / float(H3_FPS)
        requested_cuts, _cut_kinds = self._resolve_cut_list(
            cut_times, prompt_duration, warnings
        )
        exact_visible_text = _visible_literals(visible_text)
        constraints = (
            [constraint_notes.strip()]
            if isinstance(constraint_notes, str) and constraint_notes.strip()
            else []
        )

        if h3_mode == "base_T2VA":
            width, height = DEFAULT_TEXT_CANVAS
        else:
            width, height = self._canvas(None, subject_fields)

        request_id = str(uuid4())
        staged = None
        cleanup_result = None
        if anchors:
            staged = stage_assets(
                folder_paths.get_temp_directory(),
                request_id=request_id,
                images={
                    slot: subject_fields[f"subject_{slot}_image"]
                    for slot in anchors
                },
                asset_directives=_base_asset_directives(h3_mode),
                strict_image_slots=True,
            )

        try:
            assets = (
                verified_manifest_snapshot(staged)["assets"]
                if staged is not None
                else []
            )
            request = freeze_request_authority(build_request(
                request_id=request_id,
                h3_mode=h3_mode,
                quality_mode=quality_mode,
                research_policy=research_policy,
                creative_brief=target_description,
                exact_literals={
                    "dialogue": dialogue,
                    "lyrics": lyrics,
                    "visible_text": exact_visible_text,
                },
                generation={
                    "requested_duration_seconds": requested_duration,
                    "snapped_duration_seconds": snapped_duration,
                    "fps": float(H3_FPS),
                    "width": width,
                    "height": height,
                    "length": length,
                },
                task={
                    "task_types": [],
                    "video_role": "none",
                    "audio_role": "none",
                    "constraints": constraints,
                    "cut_timestamps": requested_cuts,
                },
                subjects=[],
                assets=assets,
                local_h3_format_guide=prompts_base.build_system_prompt(h3_mode),
                wall_clock_timeout_seconds=int(timeout),
            ))
            assets = request["assets"]
            request_text = serialize_request(request)

            client = HermesRunsClient(
                base_url=base_url,
                poll_interval_seconds=poll_interval,
            )
            run_kwargs: Dict[str, Any] = {
                "input": request_text,
                "instructions": STABLE_INSTRUCTIONS,
                "session_id": f"comfyui:h3:{request['request_id']}",
                "timeout_seconds": timeout,
                "interruption_check": (
                    comfy.model_management
                    .throw_exception_if_processing_interrupted
                ),
            }
            if provider:
                run_kwargs["provider"] = provider
            if model:
                run_kwargs["model"] = model
            if staged is not None:
                verify_staged_assets(staged)
            run_result = client.run(**run_kwargs)
            if staged is not None:
                verify_staged_assets(staged)
            run_id = _safe_run_id(getattr(run_result, "run_id", None))
            run_id_fields = {
                "run_id": run_id,
                **_remote_identifier_fields("run_id", run_id),
            }
            run_status = _safe_run_status(
                getattr(run_result, "status", None)
            )
            elapsed_seconds = _safe_elapsed_seconds(
                getattr(run_result, "elapsed_seconds", None)
            )

            try:
                parsed = parse_result(run_result.output, request=request)
            except ContractError as exc:
                raise RuntimeError(
                    f"Hermes response contract validation failed: {exc}"
                ) from exc
            if parsed.quality_report.get("hard_errors"):
                raise RuntimeError(
                    "Hermes response reported hard errors; the prompt was rejected."
                )

            local = process(
                parsed.h3_prompt,
                CowboyContext(
                    subjects=[],
                    duration_seconds=prompt_duration,
                    task_type="",
                    mode=h3_mode,
                    known_shot_times=requested_cuts,
                    is_editing=False,
                    dialogue_text=dialogue,
                    lyrics=lyrics,
                    wired_pictures=len(anchors),
                    has_video=False,
                    has_audio=False,
                    multi_shot_requested=len(requested_cuts) > 1,
                    fl2va_normalize_picture_tags=(
                        fl2va_normalize_picture_tags
                    ),
                ),
            )
            hard_errors = list(local.retry_errors)
            visible_section = _integrated_description_section(local.prompt)
            missing_visible = [
                literal for literal in exact_visible_text
                if literal not in visible_section
            ]
            try:
                spoken_blocks = _spoken_blocks(local.prompt)
            except ValueError:
                spoken_blocks = ()
                malformed_spoken_blocks = True
                hard_errors.append(
                    "R3 VERBATIM: malformed or unclosed <d> block markup in "
                    "the final H3 prompt."
                )
            else:
                malformed_spoken_blocks = False
            missing_spoken = (
                [] if malformed_spoken_blocks
                else _missing_ordered_spoken_literals(
                    spoken_blocks, dialogue, lyrics
                )
            )
            if missing_spoken:
                hard_errors.append(
                    "R3 VERBATIM: exact user "
                    + " and ".join(missing_spoken)
                    + (" are" if len(missing_spoken) > 1 else " is")
                    + " missing from the final H3 prompt."
                )
            if missing_visible:
                hard_errors.append(
                    "R3 VERBATIM: exact visible text is missing from the H3 prompt."
                )
            if local.char_count > MAX_PROMPT_CHARS:
                hard_errors.append(
                    f"The prompt exceeds the hard cap of {MAX_PROMPT_CHARS} "
                    f"characters ({local.char_count} characters after local "
                    "processing)."
                )
            if hard_errors:
                safe_categories = _safe_hard_error_categories(hard_errors)
                raise RuntimeError(
                    "local H3 validation failed: " + " | ".join(safe_categories)
                )

            candidates = [
                _candidate_handoff(item) for item in parsed.candidates
            ]
            selected = parsed.selected_candidate
            validation_warnings = warnings + list(local.warnings)
            staged_labels = [item["h3_label"] for item in assets]
            evidence_uninspected = (
                parsed.evidence.get("uninspected_assets", [])
                if isinstance(parsed.evidence, Mapping)
                else []
            )
            model_uninspected = _validated_uninspected_labels(
                evidence_uninspected,
                staged_labels,
            )
            uninspected_assets = list(staged_labels)
            staged_bytes = sum(item["bytes"] for item in assets)
            analysis = {
                "engine_requested": "hermes_agent",
                "engine_used": "hermes_agent",
                "fallback_used": False,
                "mode": h3_mode,
                "base_picture_role": (
                    base_picture_role
                    if type(base_picture_role) is str
                    and base_picture_role in _ALLOWED_BASE_PICTURE_ROLES
                    else "unrecognized"
                ),
                "fl2va_normalize_picture_tags": (
                    fl2va_normalize_picture_tags
                ),
                "anchor_pictures": [
                    f"Picture {slot}" for slot in anchors
                ],
                "duration_source": duration_source,
                "requested_duration_seconds": round(requested_duration, 3),
                "snapped_duration_seconds": round(snapped_duration, 3),
                "duration_supported": duration_supported,
                "strict_duration": strict_duration,
                "snap_duration_to_h3_grid": snap_duration_to_h3_grid,
                "h3_length_frames": length,
                "hermes": {
                    "base_url": base_url,
                    **run_id_fields,
                    "status": run_status,
                    "elapsed_seconds": elapsed_seconds,
                    "quality_mode": quality_mode,
                    "research_policy": research_policy,
                    "usage": _safe_usage(run_result.usage),
                    "model_reported_tools": _safe_reported_tools(
                        parsed.reported_tools
                    ),
                    "verified_tool_events": [],
                },
                "request": {
                    "schema_version": request["schema_version"],
                    "request_id": request["request_id"],
                    "asset_count": len(assets),
                    "staged_bytes": staged_bytes,
                    "labels": staged_labels,
                },
                "inspection": {
                    "staging_complete": staged is not None,
                    "staged_labels": staged_labels,
                    "model_reported_uninspected_assets": model_uninspected,
                    "verified_inspected_labels": [],
                },
                "selection": {
                    **_remote_identifier_fields(
                        "selected_candidate_id",
                        parsed.selected_candidate_id,
                    ),
                    "candidate_count": len(parsed.candidates),
                    "score_vector": _safe_numeric_scores(selected.score_vector),
                    "candidates": candidates,
                },
                "evidence": {"uninspected_assets": model_uninspected},
                "intent_ir": {
                    "required_atom_count": _count_list(
                        parsed.intent_ir.get("required_atoms")
                    ),
                    "preferred_atom_count": _count_list(
                        parsed.intent_ir.get("preferred_atoms")
                    ),
                    "optional_atom_count": _count_list(
                        parsed.intent_ir.get("optional_atoms")
                    ),
                    "reference_job_count": _count_list(
                        parsed.intent_ir.get("reference_jobs")
                    ),
                },
                "repairs": {"count": _count_list(parsed.repairs)},
                "quality_report": {
                    "hard_error_count": _count_list(
                        parsed.quality_report.get("hard_errors")
                    ),
                    "warning_count": _count_list(
                        parsed.quality_report.get("warnings")
                    ),
                    "unresolved_ambiguity_count": _count_list(
                        parsed.quality_report.get("unresolved_ambiguities")
                    ),
                    "reported_source_count": _count_list(
                        parsed.quality_report.get("reported_sources")
                    ),
                },
                "validation": {
                    "hard_errors": [],
                    "applied_fixes": _safe_local_diagnostics(
                        local.applied_fixes
                    ),
                    "warnings": _safe_local_diagnostics(validation_warnings),
                    "char_count": local.char_count,
                },
                "uninspected_assets": uninspected_assets,
                "cleanup": {
                    "policy": cleanup_policy,
                    "retained_path": None,
                },
            }

            label_map = build_label_map([
                f"subject_{slot}_image" for slot in anchors
            ])
            # Construct the complete, final 18-output tuple before success
            # cleanup. Retention metadata is deterministic from the validated
            # policy and staged job root, so a second post-cleanup assembly is
            # neither needed nor permitted.
            expected_retained_path = (
                str(staged.job_dir)
                if staged is not None
                and cleanup_policy in ("retain", "retain_24h")
                else None
            )
            analysis["cleanup"]["retained_path"] = expected_retained_path
            outputs = self._outputs(
                local.prompt,
                prompt_duration,
                H3_FPS,
                analysis,
                h3_mode,
                images=subject_fields,
                frames=None,
                audio=None,
                width=width,
                height=height,
                length=length,
                label_map=label_map,
            )
            if staged is not None:
                cleanup_result = cleanup_assets(
                    staged,
                    success=True,
                    policy=cleanup_policy,
                )
                actual_retained_path = (
                    str(cleanup_result.path)
                    if cleanup_result.retained and cleanup_result.path is not None
                    else None
                )
                if actual_retained_path != expected_retained_path:
                    raise RuntimeError(
                        "cleanup returned an unexpected retained-path state"
                    )
            return outputs
        except BaseException:
            if staged is not None and cleanup_result is None:
                try:
                    cleanup_assets(
                        staged,
                        success=False,
                        policy=cleanup_policy,
                    )
                except BaseException:
                    pass
            raise

    def _generate_ref(
        self,
        *,
        subjects: str,
        target_description: str,
        quality_mode: str,
        research_policy: str,
        video,
        frames,
        fps: float,
        audio,
        video_role: str,
        audio_role: str,
        cut_times: str,
        dialogue: str,
        constraint_notes: str,
        duration_override: float,
        max_frames_to_analyze: int,
        base_picture_role: str,
        fl2va_normalize_picture_tags: bool,
        snap_duration_to_h3_grid: bool,
        subject_rows: int,
        music_video: bool,
        music_source: str,
        lyrics: str,
        music_description: str,
        visible_text: str,
        hermes_base_url: str,
        timeout_seconds: int,
        poll_interval_seconds: float,
        strict_duration: bool,
        cleanup_policy: str,
        hermes_provider: str,
        hermes_model: str,
        subject_fields: Dict[str, Any],
    ) -> Tuple:
        """Run Ref2VA while keeping media authority and H3 validation local."""

        # Validate every cheap control before touching private media or creating
        # a client. ComfyUI widgets normally enforce these bounds, but API calls
        # and saved workflows must fail closed too.
        base_url = _loopback_base_url(hermes_base_url)
        provider = _optional_route(hermes_provider, "provider")
        model = _optional_route(hermes_model, "model")
        fl2va_normalize_picture_tags = _exact_boolean(
            fl2va_normalize_picture_tags,
            "fl2va_normalize_picture_tags",
        )
        snap_duration_to_h3_grid = _exact_boolean(
            snap_duration_to_h3_grid,
            "snap_duration_to_h3_grid",
        )
        music_video = _exact_boolean(music_video, "music_video")
        timeout = _bounded_number(
            timeout_seconds,
            "timeout_seconds",
            TIMEOUT_MIN_SECONDS,
            TIMEOUT_MAX_SECONDS,
        )
        if not timeout.is_integer():
            raise RuntimeError("timeout_seconds must be a whole number of seconds.")
        poll_interval = _bounded_number(
            poll_interval_seconds,
            "poll_interval_seconds",
            POLL_MIN_SECONDS,
            POLL_MAX_SECONDS,
        )
        if cleanup_policy not in CLEANUP_POLICIES:
            raise RuntimeError(
                "cleanup_policy must be one of " + ", ".join(CLEANUP_POLICIES)
            )
        if quality_mode not in ("fast", "balanced", "hero"):
            raise RuntimeError("quality_mode must be fast, balanced, or hero.")
        if research_policy not in ("never", "when_uncertain", "always"):
            raise RuntimeError(
                "research_policy must be never, when_uncertain, or always."
            )
        if (
            isinstance(subject_rows, bool)
            or not isinstance(subject_rows, int)
            or not 0 <= subject_rows <= NUM_SUBJECT_SLOTS
        ):
            raise RuntimeError(
                f"subject_rows must be a whole number from 0 to {NUM_SUBJECT_SLOTS}."
            )
        if (
            isinstance(max_frames_to_analyze, bool)
            or not isinstance(max_frames_to_analyze, int)
            or not 2 <= max_frames_to_analyze <= 16
        ):
            raise RuntimeError(
                "max_frames_to_analyze must be a whole number from 2 to 16."
            )
        if isinstance(duration_override, bool) or not isinstance(
            duration_override, Real
        ):
            raise RuntimeError("duration_override must be a number.")
        override = float(duration_override)
        if not math.isfinite(override) or override < 0:
            raise RuntimeError("duration_override must be finite and nonnegative.")

        warnings: list[str] = []
        rows = self._read_rows(subject_fields)
        out_of_sight = [
            row.slot for row in rows
            if row.is_filled and row.slot > subject_rows
        ]
        if out_of_sight:
            warnings.append(
                "subject_rows is "
                + f"{subject_rows}, but row "
                + ", ".join(str(slot) for slot in out_of_sight)
                + " holds text and was used anyway. Raise subject_rows to see it."
            )
        parsed_subjects = merge_text_subjects(
            subjects_from_rows(rows, warnings), subjects, warnings
        )
        if not parsed_subjects:
            raise RuntimeError(self._no_subjects_message(subjects))

        wired = [
            slot for slot in range(1, NUM_SUBJECT_SLOTS + 1)
            if subject_fields.get(f"subject_{slot}_image") is not None
        ]
        if wired and wired != list(range(1, wired[-1] + 1)):
            missing = sorted(set(range(1, wired[-1] + 1)) - set(wired))
            raise RuntimeError(
                "Wired physical picture slots contain a gap; fill physical slot "
                + ", ".join(str(slot) for slot in missing)
                + " before using a later image slot."
            )
        image_tags = bind_images(parsed_subjects, wired, warnings)

        # A connected VIDEO wins exactly as in V1, but its component values are
        # now validated rather than accepting a fabricated default rate. Reading
        # components once also preserves identity for the pass-through outputs.
        connected_video = video is not None
        if connected_video:
            if frames is not None:
                warnings.append("both video and frames connected; using video")
            getter = getattr(video, "get_components", None)
            if not callable(getter):
                raise RuntimeError(
                    "The video input is not a ComfyUI VIDEO object. Connect a "
                    "Load Video node, or use the frames input."
                )
            try:
                components = getter()
            except Exception as exc:
                raise RuntimeError(
                    "The video input is not a readable ComfyUI VIDEO object."
                ) from exc
            resolved_frames = getattr(components, "images", None)
            rate_value = getattr(components, "frame_rate", None)
            duration_source = "video"
        else:
            resolved_frames = frames
            rate_value = fps
            duration_source = "frames+fps"

        if resolved_frames is None:
            raise RuntimeError(
                "Connect either a VIDEO (video input) or a non-empty IMAGE "
                "batch (frames input, with fps set)."
            )
        dim = getattr(resolved_frames, "dim", None)
        shape = getattr(resolved_frames, "shape", ())
        if (
            not callable(dim)
            or dim() != 4
            or len(shape) != 4
            or int(shape[0]) < 1
        ):
            raise RuntimeError(
                "frames must be a non-empty IMAGE batch (B, H, W, C)."
            )
        if isinstance(rate_value, bool) or not isinstance(rate_value, Real):
            raise RuntimeError("fps must be a positive finite number.")
        real_fps = float(rate_value)
        if not math.isfinite(real_fps) or real_fps <= 0:
            raise RuntimeError("fps must be a positive finite number.")

        source_duration = int(shape[0]) / real_fps
        requested_duration = source_duration
        if override > 0:
            requested_duration = override
            duration_source = "override"
        duration_supported = _apply_duration_policy(
            requested_duration,
            strict=strict_duration,
            warnings=warnings,
        )
        length, prompt_duration = self._snap(
            requested_duration, snap_duration_to_h3_grid, warnings
        )
        snapped_duration = length / float(H3_FPS)
        self._warn_reference_video(resolved_frames, real_fps, warnings)

        measured, measured_kinds = self._resolve_cut_list(
            cut_times, prompt_duration, warnings
        )

        # V1 role semantics intentionally depend on the VIDEO socket, not the
        # locally sampled frame batch. The frame batch still becomes <Video 1>
        # for staging and output, but does not turn an edit role into an edit.
        if not connected_video and video_role != "subject_source":
            warnings.append(
                f"video_role is '{video_role}' but no video is connected; "
                "it was ignored"
            )
            video_role = "subject_source"
        if audio is None and audio_role != "none":
            warnings.append(
                f"audio_role is '{audio_role}' but no audio is connected; "
                "it was ignored"
            )
            audio_role = "none"
        music_is_reference, audio_role = self._resolve_music(
            music_video,
            music_source,
            audio,
            audio_role,
            lyrics,
            music_description,
            warnings,
        )
        effective_lyrics = lyrics if music_video else ""
        task_types = spec.derive_task_types(video_role, audio_role)
        task_type = spec.format_task_type(task_types)
        is_editing = "video editing" in task_types
        exact_visible_text = _visible_literals(visible_text)
        constraints = (
            [constraint_notes.strip()]
            if isinstance(constraint_notes, str) and constraint_notes.strip()
            else []
        )
        width, height = self._canvas(resolved_frames, subject_fields)

        # Toolset discovery is bounded/authenticated client traffic, not media
        # inspection. Resolve it only after cheap/V1 control validation but
        # before staging, so the no-video-tool path can stage real JPEG pixels
        # for every locally selected fallback frame.
        client = HermesRunsClient(
            base_url=base_url,
            poll_interval_seconds=poll_interval,
        )
        video_tool_available = (
            client.has_enabled_tool("video", "video_analyze") is True
        )
        fallback_slots = (
            AssetLimits().max_asset_count
            - len(wired)
            - 1  # <Video 1>
            - int(audio is not None)
        )
        selection_limit = (
            max_frames_to_analyze
            if video_tool_available
            else min(max_frames_to_analyze, fallback_slots)
        )
        if selection_limit < 1:
            raise RuntimeError(
                "No bounded asset slot remains for keyframe fallback evidence."
            )
        keyframes = select_keyframes(
            resolved_frames,
            real_fps,
            max_frames=selection_limit,
            known_boundaries=(
                [int(round(timestamp * real_fps)) for timestamp in measured]
                if measured else None
            ),
        )
        keyframe_images = (
            None
            if video_tool_available
            else {
                evidence_index: resolved_frames[
                    source_index:source_index + 1
                ]
                for evidence_index, source_index in enumerate(
                    keyframes.indices, start=1
                )
            }
        )

        request_id = str(uuid4())
        external_required_assets = _external_required_asset_authority(
            music_video=music_video,
            music_source=music_source,
            audio=audio,
        )
        staged = None
        cleanup_result = None
        try:
            staged = stage_assets(
                folder_paths.get_temp_directory(),
                request_id=request_id,
                images={
                    slot: subject_fields[f"subject_{slot}_image"]
                    for slot in wired
                },
                video_frames=resolved_frames,
                video_fps=real_fps,
                video_duration=source_duration,
                video=video if connected_video else None,
                keyframe_images=keyframe_images,
                audio=audio,
                asset_directives=_ref_asset_directives(
                    wired, has_audio=audio is not None
                ),
                strict_image_slots=True,
                allow_video_source_reuse=connected_video,
                allowed_video_source_roots=_trusted_video_source_roots(),
            )
            manifest_snapshot = verified_manifest_snapshot(staged)
            staged_assets = manifest_snapshot["assets"]
            staged_asset_count = len(staged_assets)
            request_assets = list(staged_assets) + list(external_required_assets)

            local_cut_times = measured or [
                round(boundary / real_fps, 3)
                for boundary in keyframes.scene_boundaries
                if boundary > 0
            ]
            creative_brief = prompts_ref.build_user_context(
                parsed_subjects,
                target_description,
                prompt_duration,
                real_fps,
                task_type,
                frame_timestamps=keyframes.timestamps,
                cut_timestamps=local_cut_times,
                cut_source="measured" if measured else "local",
                cut_kinds=measured_kinds,
                full_clip=video_tool_available,
                has_video=connected_video,
                audio_available=audio is not None,
                dialogue_text=dialogue,
                constraint_notes=constraint_notes,
                image_tags=image_tags,
                video_role=video_role,
                music_video=music_video,
                lyrics=lyrics,
                music_description=music_description,
                music_is_reference=music_is_reference,
            )
            request = build_request(
                request_id=request_id,
                h3_mode="ref",
                quality_mode=quality_mode,
                research_policy=research_policy,
                creative_brief=creative_brief,
                exact_literals={
                    "dialogue": dialogue,
                    "lyrics": effective_lyrics,
                    "visible_text": exact_visible_text,
                },
                generation={
                    "requested_duration_seconds": requested_duration,
                    "snapped_duration_seconds": snapped_duration,
                    "fps": real_fps,
                    "width": width,
                    "height": height,
                    "length": length,
                },
                task={
                    "task_types": task_types,
                    "video_role": video_role,
                    "audio_role": audio_role,
                    "constraints": constraints,
                    "cut_timestamps": measured,
                },
                subjects=_ref_subject_payload(parsed_subjects),
                assets=request_assets,
                local_h3_format_guide=prompts_ref.build_system_prompt(
                    parsed_subjects, task_types
                ),
                wall_clock_timeout_seconds=int(timeout),
            )
            request = freeze_request_authority(request)
            # Keep immutable request authority inside the existing versioned
            # `assets` field. The downstream-only record has no path, hash,
            # bytes, MIME, or staging claim, while the trusted staged prefix is
            # still used exclusively for integrity and byte diagnostics.
            request_assets = request["assets"]
            staged_assets = request_assets[:staged_asset_count]
            staged_evidence_assets = tuple(
                item for item in staged_assets
                if _KEYFRAME_EVIDENCE_LABEL_RE.fullmatch(
                    item["h3_label"]
                ) is not None
            )
            staged_physical_assets = tuple(
                item for item in staged_assets
                if _KEYFRAME_EVIDENCE_LABEL_RE.fullmatch(
                    item["h3_label"]
                ) is None
            )
            external_required_assets = request_assets[staged_asset_count:]
            request_text = serialize_request(request)
            run_kwargs: Dict[str, Any] = {
                "input": request_text,
                "instructions": STABLE_INSTRUCTIONS,
                "session_id": f"comfyui:h3:{request['request_id']}",
                "timeout_seconds": timeout,
                "interruption_check": (
                    comfy.model_management
                    .throw_exception_if_processing_interrupted
                ),
            }
            if provider:
                run_kwargs["provider"] = provider
            if model:
                run_kwargs["model"] = model

            verify_staged_assets(staged)
            run_result = client.run(**run_kwargs)
            verify_staged_assets(staged)

            run_id = _safe_run_id(getattr(run_result, "run_id", None))
            run_id_fields = {
                "run_id": run_id,
                **_remote_identifier_fields("run_id", run_id),
            }
            run_status = _safe_run_status(
                getattr(run_result, "status", None)
            )
            elapsed_seconds = _safe_elapsed_seconds(
                getattr(run_result, "elapsed_seconds", None)
            )
            try:
                parsed = parse_result(run_result.output, request=request)
            except ContractError as exc:
                raise RuntimeError(
                    f"Hermes response contract validation failed: {exc}"
                ) from exc
            if parsed.quality_report.get("hard_errors"):
                raise RuntimeError(
                    "Hermes response reported hard errors; the prompt was rejected."
                )

            local = process(
                parsed.h3_prompt,
                CowboyContext(
                    subjects=parsed_subjects,
                    duration_seconds=prompt_duration,
                    task_type=task_type,
                    mode="ref",
                    known_shot_times=measured,
                    is_editing=is_editing,
                    dialogue_text=dialogue,
                    lyrics=effective_lyrics,
                    wired_pictures=len(wired),
                    has_video=connected_video,
                    has_audio=audio is not None,
                    music_video=music_video,
                ),
            )
            hard_errors = list(local.retry_errors)
            visible_section = _ref_description_section(local.prompt)
            if any(literal not in visible_section for literal in exact_visible_text):
                hard_errors.append(
                    "R3 VERBATIM: exact visible text is missing from the H3 prompt."
                )
            try:
                spoken_blocks = _spoken_blocks(local.prompt)
            except ValueError:
                spoken_blocks = ()
                malformed_spoken_blocks = True
                hard_errors.append(
                    "R3 VERBATIM: malformed or unclosed <d> block markup in "
                    "the final H3 prompt."
                )
            else:
                malformed_spoken_blocks = False
            missing_spoken = (
                [] if malformed_spoken_blocks
                else _missing_ordered_spoken_literals(
                    spoken_blocks, dialogue, effective_lyrics
                )
            )
            if missing_spoken:
                hard_errors.append(
                    "R3 VERBATIM: exact user "
                    + " and ".join(missing_spoken)
                    + (" are" if len(missing_spoken) > 1 else " is")
                    + " missing from the final H3 prompt."
                )
            if local.char_count > MAX_PROMPT_CHARS:
                hard_errors.append(
                    f"The prompt exceeds the hard cap of {MAX_PROMPT_CHARS} "
                    f"characters ({local.char_count} characters after local "
                    "processing)."
                )
            if hard_errors:
                safe_categories = _safe_hard_error_categories(hard_errors)
                raise RuntimeError(
                    "local H3 validation failed: " + " | ".join(safe_categories)
                )

            # Everything derived from the remote response is reduced to bounded
            # local metadata before success cleanup. A report is never promoted
            # to a verified event, and audio has no inspection tool here.
            candidates = [
                _candidate_handoff(item) for item in parsed.candidates
            ]
            selected = parsed.selected_candidate
            safe_reported_tools = _safe_reported_tools(parsed.reported_tools)
            staged_labels = [
                item["h3_label"] for item in staged_physical_assets
            ]
            staged_evidence_labels = [
                item["h3_label"] for item in staged_evidence_assets
            ]
            keyframe_evidence = [
                {
                    "label": label,
                    "source_frame_index": int(source_index),
                    "timestamp_seconds": float(timestamp),
                }
                for label, source_index, timestamp in zip(
                    staged_evidence_labels,
                    keyframes.indices,
                    keyframes.timestamps,
                )
            ]
            external_required_labels = [
                item["h3_label"] for item in external_required_assets
            ]
            authorized_labels = staged_labels + external_required_labels
            tool_capable_labels = (
                ["<Video 1>"]
                if video_tool_available and "<Video 1>" in staged_labels
                else []
            )
            model_reported_inspected_labels = (
                ["<Video 1>"]
                if (
                    "<Video 1>" in tool_capable_labels
                    and "video_analyze" in safe_reported_tools
                )
                else []
            )
            evidence_uninspected = (
                parsed.evidence.get("uninspected_assets", [])
                if isinstance(parsed.evidence, Mapping)
                else []
            )
            model_uninspected = _validated_uninspected_labels(
                evidence_uninspected, authorized_labels
            )
            # Model reports and tool availability are diagnostics only. No
            # independently attributable asset-tied tool event is available in
            # this Runs response, so physical assets, JPEG evidence, and
            # external downstream authority remain authoritatively uninspected.
            verified_inspected_labels: list[str] = []
            all_evidence_labels = (
                staged_labels
                + staged_evidence_labels
                + external_required_labels
            )
            uninspected_assets = [
                label for label in all_evidence_labels
                if label not in verified_inspected_labels
            ]
            staged_bytes = sum(item["bytes"] for item in staged_assets)
            validation_warnings = warnings + list(local.warnings)

            analysis = {
                "engine_requested": "hermes_agent",
                "engine_used": "hermes_agent",
                "fallback_used": False,
                "mode": "ref",
                "task_type": task_type,
                "task_types": task_types,
                "video_role": (
                    video_role if video_role in spec.VIDEO_ROLES else "unrecognized"
                ),
                "audio_role": (
                    audio_role if audio_role in spec.AUDIO_ROLES else "unrecognized"
                ),
                "music_video": music_video,
                "music_source": _safe_local_diagnostic(music_source),
                "music_is_reference": music_is_reference,
                "subjects": _safe_local_diagnostics(
                    [subject.describe() for subject in parsed_subjects]
                ),
                "subject_kinds": [subject.kind for subject in parsed_subjects],
                "image_tags": image_tags,
                "cut_source": "measured" if measured else "local",
                "cut_timestamps": measured,
                "selected_frame_indices": list(keyframes.indices),
                "duration_source": duration_source,
                "requested_duration_seconds": round(requested_duration, 3),
                "snapped_duration_seconds": round(snapped_duration, 3),
                "duration_supported": duration_supported,
                "strict_duration": strict_duration,
                "snap_duration_to_h3_grid": snap_duration_to_h3_grid,
                "h3_length_frames": length,
                "hermes": {
                    "base_url": base_url,
                    **run_id_fields,
                    "status": run_status,
                    "elapsed_seconds": elapsed_seconds,
                    "quality_mode": quality_mode,
                    "research_policy": research_policy,
                    "usage": _safe_usage(run_result.usage),
                    "model_reported_tools": safe_reported_tools,
                    "verified_tool_events": [],
                },
                "request": {
                    "schema_version": request["schema_version"],
                    "request_id": request["request_id"],
                    "asset_count": len(request_assets),
                    "staged_asset_count": len(staged_assets),
                    "physical_staged_asset_count": len(staged_physical_assets),
                    "evidence_staged_asset_count": len(staged_evidence_assets),
                    "staged_bytes": staged_bytes,
                    "labels": staged_labels,
                    "evidence_labels": staged_evidence_labels,
                    "external_required_labels": external_required_labels,
                },
                "inspection": {
                    "staging_complete": True,
                    "staged_labels": staged_labels,
                    "staged_evidence_labels": staged_evidence_labels,
                    "keyframe_evidence": keyframe_evidence,
                    "external_required_uninspected_labels": (
                        external_required_labels
                    ),
                    "metadata_only_observations": [
                        "fps", "duration", "dimensions", "cuts", "keyframes",
                    ],
                    "tool_capable_labels": tool_capable_labels,
                    "model_reported_tools": safe_reported_tools,
                    "model_reported_inspected_labels": (
                        model_reported_inspected_labels
                    ),
                    "model_reported_uninspected_assets": model_uninspected,
                    "verified_inspected_labels": verified_inspected_labels,
                },
                "selection": {
                    **_remote_identifier_fields(
                        "selected_candidate_id",
                        parsed.selected_candidate_id,
                    ),
                    "candidate_count": len(parsed.candidates),
                    "score_vector": _safe_numeric_scores(selected.score_vector),
                    "candidates": candidates,
                },
                "evidence": {"uninspected_assets": model_uninspected},
                "intent_ir": {
                    "required_atom_count": _count_list(
                        parsed.intent_ir.get("required_atoms")
                    ),
                    "preferred_atom_count": _count_list(
                        parsed.intent_ir.get("preferred_atoms")
                    ),
                    "optional_atom_count": _count_list(
                        parsed.intent_ir.get("optional_atoms")
                    ),
                    "reference_job_count": _count_list(
                        parsed.intent_ir.get("reference_jobs")
                    ),
                },
                "repairs": {"count": _count_list(parsed.repairs)},
                "quality_report": {
                    "hard_error_count": _count_list(
                        parsed.quality_report.get("hard_errors")
                    ),
                    "warning_count": _count_list(
                        parsed.quality_report.get("warnings")
                    ),
                    "unresolved_ambiguity_count": _count_list(
                        parsed.quality_report.get("unresolved_ambiguities")
                    ),
                    "reported_source_count": _count_list(
                        parsed.quality_report.get("reported_sources")
                    ),
                },
                "validation": {
                    "hard_errors": [],
                    "applied_fixes": _safe_local_diagnostics(
                        local.applied_fixes
                    ),
                    "warnings": _safe_local_diagnostics(validation_warnings),
                    "char_count": local.char_count,
                },
                "uninspected_assets": uninspected_assets,
                "cleanup": {
                    "policy": cleanup_policy,
                    "retained_path": None,
                },
            }

            label_map = build_label_map(
                [f"subject_{slot}_image" for slot in wired],
                has_video=True,
                has_audio=audio is not None,
            )
            # Construct the complete, final 18-output tuple before success
            # cleanup. The retained path follows deterministically from the
            # validated policy and staged job root.
            expected_retained_path = (
                str(staged.job_dir)
                if cleanup_policy in ("retain", "retain_24h")
                else None
            )
            analysis["cleanup"]["retained_path"] = expected_retained_path
            outputs = self._outputs(
                local.prompt,
                prompt_duration,
                real_fps,
                analysis,
                "ref",
                images=subject_fields,
                frames=resolved_frames,
                audio=audio,
                width=width,
                height=height,
                length=length,
                label_map=label_map,
            )
            cleanup_result = cleanup_assets(
                staged,
                success=True,
                policy=cleanup_policy,
            )
            actual_retained_path = (
                str(cleanup_result.path)
                if cleanup_result.retained and cleanup_result.path is not None
                else None
            )
            if actual_retained_path != expected_retained_path:
                raise RuntimeError("cleanup returned an unexpected retained-path state")
            return outputs
        except BaseException:
            if staged is not None and cleanup_result is None:
                try:
                    cleanup_assets(
                        staged,
                        success=False,
                        policy=cleanup_policy,
                    )
                except BaseException:
                    pass
            raise


NODE_CLASS_MAPPINGS = {
    "TrentH3HermesPromptDirector": H3HermesPromptDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentH3HermesPromptDirector": "H3 Hermes Prompt Director",
}


__all__ = [
    "H3HermesPromptDirector",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
