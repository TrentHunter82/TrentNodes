#!/usr/bin/env python3
"""Sanitized text-only smoke test for the H3 Hermes Runs API.

The script intentionally imports no ComfyUI node.  It builds the authoritative
``h3_hermes_request/1.0`` contract with the local Base/T2VA guide, asks
:class:`HermesRunsClient` to submit and poll one logical run, validates the
result locally, and prints one compact metadata-only JSON object.

Run from the TrentNodes repository root::

    /home/trent/ComfyUI/venv/bin/python tools/h3_hermes_api_smoke.py \
        --base-url http://127.0.0.1:8642 --mode base_T2VA \
        --brief "A courier crosses a rain-dark loading bay."

The bearer key is accepted only through ``HermesRunsClient``'s existing
``HERMES_AGENT_API_KEY`` environment behavior.  This CLI has no key argument,
does not read credential files or environment variables itself, and never
prints request headers, raw API responses, or the full generated prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import sys
import types
from typing import Any, NoReturn, Sequence, TextIO, Type
from urllib.parse import urlsplit

# Direct execution sets sys.path[0] to tools/.  Load the lightweight utilities
# under a private namespace package rather than the ambiguous top-level name
# ``utils``.  This keeps direct execution install-free and also avoids clashes
# with test runners/applications that already imported an unrelated utils
# package.  The TrentNodes root initializer is deliberately not imported: it
# imports ComfyUI nodes.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRIVATE_PACKAGE = "_trentnodes_h3_smoke"
for _name, _path in (
    (_PRIVATE_PACKAGE, _REPO_ROOT),
    (f"{_PRIVATE_PACKAGE}.utils", os.path.join(_REPO_ROOT, "utils")),
):
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__path__ = [_path]
        sys.modules[_name] = _module

from _trentnodes_h3_smoke.utils.h3_cowboy import prompts_base  # noqa: E402
from _trentnodes_h3_smoke.utils.h3_cowboy.assembler import (  # noqa: E402
    CowboyContext,
    process,
)
from _trentnodes_h3_smoke.utils.h3_cowboy.wiring import (  # noqa: E402
    H3_FPS,
    snap_length,
)
from _trentnodes_h3_smoke.utils.h3_hermes.client import (  # noqa: E402
    MISSING_API_KEY_GUIDANCE,
    HermesAuthenticationError,
    HermesClientError,
    HermesRunsClient,
)
from _trentnodes_h3_smoke.utils.h3_hermes.contract import (  # noqa: E402
    ContractError,
    STABLE_INSTRUCTIONS,
    build_request,
    parse_result,
    serialize_request,
)
from _trentnodes_h3_smoke.utils.h3_hermes.schema import (  # noqa: E402
    QUALITY_MODES,
    RESEARCH_POLICIES,
)
from _trentnodes_h3_smoke.utils.h3_prompt.prompts import (  # noqa: E402
    MAX_PROMPT_CHARS,
)


DEFAULT_BASE_URL = "http://127.0.0.1:8642"
SUPPORTED_MODE = "base_T2VA"
DEFAULT_DURATION_SECONDS = 5.0
DEFAULT_CANVAS_WIDTH = 768
DEFAULT_CANVAS_HEIGHT = 768
TIMEOUT_MIN_SECONDS = 30
TIMEOUT_MAX_SECONDS = 3600
POLL_MIN_SECONDS = 0.05
POLL_MAX_SECONDS = 10.0
DURATION_MAX_SECONDS = 3600.0
MAX_RESULT_TOOLS = 16
MAX_RESULT_DIAGNOSTICS = 32
MAX_RESULT_DIAGNOSTIC_CHARS = 512
ROUTE_MAX_CHARS = 256
ROUTE_MAX_UTF8_BYTES = 1024
MAX_HASHED_IDENTIFIER_UTF8_BYTES = 4096

_AUTH_FAILURE_GUIDANCE = (
    "Hermes API authentication failed. Verify HERMES_AGENT_API_KEY matches "
    "the local Hermes API server key."
)
_GENERIC_CLIENT_FAILURE = (
    "Hermes API request failed safely. Check the local gateway configuration "
    "and logs."
)
_SENSITIVE_RE = re.compile(
    r"(?i)(authorization|bearer|api[ _-]?key|credential|secret|token|private)"
)
_BEARER_VALUE_RE = re.compile(r"(?i)\bbearer\s+\S+")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:api[ _-]?key|authorization|credential|secret|token)\b"
    r"\s*[:=]\s*\S+"
)
_PRIVATE_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:[\\/]|/(?:home|users|mnt|tmp|root|etc|var)/)\S+"
)
_SUCCESS_STATUS_VALUES = frozenset({"completed"})
_ALLOWED_REPORTED_TOOLS = frozenset({"web_search", "web_extract"})


class SmokeInputError(ValueError):
    """A command-line value violates the bounded smoke-test policy."""


class SmokeValidationError(RuntimeError):
    """The remote result failed a local fail-closed validation gate."""


class _SafeArgumentParser(argparse.ArgumentParser):
    """Keep argparse's exit-2 contract without echoing rejected values."""

    def error(self, message: str) -> NoReturn:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments\n")


def validate_base_url(value: str) -> str:
    """Return a canonical root-only plain-HTTP loopback base URL.

    No DNS lookup is used: accepted hosts are the literal name ``localhost``
    or a numeric address for which :mod:`ipaddress` reports ``is_loopback``.
    User information, non-root paths, queries, fragments, whitespace, controls,
    and scoped hosts are rejected before parsing. Port 8642 is required.
    """

    if type(value) is not str or not value:
        raise SmokeInputError("Hermes base URL must be a non-blank loopback URL.")
    if any(
        ord(char) < 32
        or 0x7F <= ord(char) <= 0x9F
        or char.isspace()
        for char in value
    ):
        raise SmokeInputError("Hermes base URL must not contain whitespace or controls.")
    candidate = value
    try:
        parsed = urlsplit(candidate)
        # Reading .port performs malformed and out-of-range port validation.
        port = parsed.port
    except ValueError:
        raise SmokeInputError("Hermes base URL must be a valid loopback URL.") from None

    if parsed.scheme.lower() != "http":
        raise SmokeInputError("Hermes loopback URL must use plain HTTP.")
    if parsed.username is not None or parsed.password is not None:
        raise SmokeInputError("Hermes loopback URL must not contain credentials.")
    if parsed.query or "?" in candidate:
        raise SmokeInputError("Hermes loopback URL must not contain a query.")
    if parsed.fragment or "#" in candidate:
        raise SmokeInputError("Hermes loopback URL must not contain a fragment.")
    if parsed.path not in ("", "/"):
        raise SmokeInputError("Hermes loopback URL must point at the server root.")

    hostname = parsed.hostname
    if not hostname or "%" in hostname or port != 8642:
        raise SmokeInputError("Hermes base URL must be a valid loopback URL.")
    if hostname.lower() == "localhost":
        canonical_host = "localhost"
    else:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            raise SmokeInputError("Hermes base URL must use a loopback host.") from None
        if not address.is_loopback:
            raise SmokeInputError("Hermes base URL must use a loopback host.")
        canonical_host = address.compressed
        if address.version == 6:
            canonical_host = f"[{canonical_host}]"
    return f"http://{canonical_host}:8642"


def _bounded_float_type(
    name: str, minimum: float, maximum: float, *, minimum_inclusive: bool = True
):
    def parse(value: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"{name} must be a number") from None
        lower_ok = number >= minimum if minimum_inclusive else number > minimum
        if not math.isfinite(number) or not lower_ok or number > maximum:
            relation = "at least" if minimum_inclusive else "greater than"
            raise argparse.ArgumentTypeError(
                f"{name} must be {relation} {minimum:g} and at most {maximum:g}"
            )
        return number

    return parse


def _bounded_int_type(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            # int() deliberately rejects decimal spellings such as 30.5.
            number = int(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"{name} must be a whole number") from None
        if number < minimum or number > maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return number

    return parse


def _validate_route_value(value: Any, name: str) -> str:
    """Validate and normalize one optional provider/model route selector."""

    if not isinstance(value, str):
        raise SmokeInputError(f"Hermes {name} route must be text.")
    if len(value) > ROUTE_MAX_CHARS:
        raise SmokeInputError(f"Hermes {name} route is too large.")
    if any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in value):
        raise SmokeInputError(f"Hermes {name} route contains control characters.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise SmokeInputError(f"Hermes {name} route is not valid UTF-8 text.") from None
    if len(encoded) > ROUTE_MAX_UTF8_BYTES:
        raise SmokeInputError(f"Hermes {name} route is too large.")
    return value.strip()


def _route_argument_type(name: str):
    def parse(value: str) -> str:
        try:
            return _validate_route_value(value, name)
        except SmokeInputError:
            raise argparse.ArgumentTypeError(f"invalid {name} route") from None

    return parse


def build_parser() -> argparse.ArgumentParser:
    """Build the strict parser; invalid enums/bounds retain argparse exit 2."""

    parser = _SafeArgumentParser(
        description=(
            "Submit one text-only Base/T2VA H3 prompt job to a loopback Hermes "
            "Runs API and print sanitized validation metadata as compact JSON."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--mode",
        choices=(SUPPORTED_MODE,),
        default=SUPPORTED_MODE,
        help="current smoke slice (only base_T2VA is supported)",
    )
    parser.add_argument("--brief", required=True, help="creative video brief")
    parser.add_argument("--dialogue", default="", help="exact dialogue literal")
    parser.add_argument("--lyrics", default="", help="exact lyrics literal")
    parser.add_argument(
        "--visible-text",
        action="append",
        default=[],
        help="exact visible-text literal; repeat for multiple literals",
    )
    parser.add_argument(
        "--duration",
        type=_bounded_float_type(
            "duration", 0.0, DURATION_MAX_SECONDS, minimum_inclusive=False
        ),
        default=DEFAULT_DURATION_SECONDS,
        help="requested seconds; generation length is snapped to H3's frame grid",
    )
    parser.add_argument(
        "--quality-mode",
        choices=QUALITY_MODES,
        default="balanced",
    )
    parser.add_argument(
        "--research-policy",
        choices=RESEARCH_POLICIES,
        default="when_uncertain",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_bounded_int_type(
            "timeout-seconds", TIMEOUT_MIN_SECONDS, TIMEOUT_MAX_SECONDS
        ),
        default=900,
        help="hard Runs-API wall-clock deadline",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=_bounded_float_type(
            "poll-interval-seconds", POLL_MIN_SECONDS, POLL_MAX_SECONDS
        ),
        default=1.0,
    )
    parser.add_argument(
        "--provider",
        type=_route_argument_type("provider"),
        default="",
        help="optional Hermes route provider; blank uses the gateway default",
    )
    parser.add_argument(
        "--model",
        type=_route_argument_type("model"),
        default="",
        help="optional Hermes route model; blank uses the gateway default",
    )
    return parser


def _remote_identifier_sha256(value: Any) -> str:
    """Represent an untrusted remote identifier only by a bounded digest."""

    if not isinstance(value, str):
        raise SmokeValidationError("Hermes run returned invalid identifier metadata.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise SmokeValidationError(
            "Hermes run returned invalid identifier metadata."
        ) from None
    if not encoded or len(encoded) > MAX_HASHED_IDENTIFIER_UTF8_BYTES:
        raise SmokeValidationError("Hermes run returned invalid identifier metadata.")
    return hashlib.sha256(encoded).hexdigest()


def _safe_diagnostic(value: Any) -> str:
    """Bound and redact a locally produced diagnostic before stdout use."""

    text = str(value).replace("\r", " ").replace("\n", " ")
    text = _BEARER_VALUE_RE.sub("[redacted]", text)
    text = _ASSIGNMENT_RE.sub("[redacted]", text)
    text = _PRIVATE_PATH_RE.sub("[redacted]", text)
    if _SENSITIVE_RE.search(text):
        return "[redacted]"
    return text[:MAX_RESULT_DIAGNOSTIC_CHARS]


def _safe_diagnostics(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [
        _safe_diagnostic(value)
        for value in values[:MAX_RESULT_DIAGNOSTICS]
    ]


def _safe_tools(values: Any) -> list[str]:
    """Copy only fixed, job-relevant tool names; provenance stays unverified."""

    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values[:MAX_RESULT_TOOLS]:
        if (
            isinstance(value, str)
            and value in _ALLOWED_REPORTED_TOOLS
            and value not in seen
        ):
            result.append(value)
            seen.add(value)
    return result


def _safe_status(value: Any) -> str:
    if not isinstance(value, str) or value not in _SUCCESS_STATUS_VALUES:
        raise SmokeValidationError("Hermes run returned invalid status metadata.")
    return value


def _safe_elapsed(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmokeValidationError("Hermes run returned invalid elapsed metadata.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise SmokeValidationError("Hermes run returned invalid elapsed metadata.")
    return round(number, 3)


def _spoken_blocks(prompt: str) -> tuple[str, ...]:
    """Extract exact, well-formed ``<d>`` contents without normalization."""

    blocks: list[str] = []
    cursor = 0
    while True:
        opening = prompt.find("<d>", cursor)
        closing = prompt.find("</d>", cursor)
        if opening < 0:
            if closing >= 0:
                raise SmokeValidationError("exact literal validation failed.")
            return tuple(blocks)
        if 0 <= closing < opening:
            raise SmokeValidationError("exact literal validation failed.")

        content_start = opening + len("<d>")
        closing = prompt.find("</d>", content_start)
        if closing < 0 or prompt.find("<d>", content_start, closing) >= 0:
            raise SmokeValidationError("exact literal validation failed.")
        blocks.append(prompt[content_start:closing])
        cursor = closing + len("</d>")


def _build_authoritative_request(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    if args.mode != SUPPORTED_MODE:
        # Parser choices normally catch this. Keep the runtime boundary strict
        # for callers that construct a Namespace directly.
        raise SmokeInputError("Only text-only base_T2VA mode is supported.")
    if not isinstance(args.brief, str) or not args.brief.strip():
        raise SmokeInputError("The creative brief must not be blank.")
    for field in ("dialogue", "lyrics"):
        if not isinstance(getattr(args, field), str):
            raise SmokeInputError(f"{field} must be text.")
    if not isinstance(args.visible_text, list) or any(
        not isinstance(item, str) for item in args.visible_text
    ):
        raise SmokeInputError("visible-text values must be text.")

    length, snapped_duration = snap_length(args.duration)
    request = build_request(
        h3_mode=SUPPORTED_MODE,
        quality_mode=args.quality_mode,
        research_policy=args.research_policy,
        creative_brief=args.brief,
        exact_literals={
            "dialogue": args.dialogue,
            "lyrics": args.lyrics,
            "visible_text": list(args.visible_text),
        },
        generation={
            "requested_duration_seconds": args.duration,
            "snapped_duration_seconds": snapped_duration,
            "fps": float(H3_FPS),
            "width": DEFAULT_CANVAS_WIDTH,
            "height": DEFAULT_CANVAS_HEIGHT,
            "length": length,
        },
        task={
            "task_types": [],
            "video_role": "none",
            "audio_role": "none",
            "constraints": [],
            "cut_timestamps": [],
        },
        subjects=[],
        assets=[],
        local_h3_format_guide=prompts_base.build_system_prompt(SUPPORTED_MODE),
        wall_clock_timeout_seconds=args.timeout_seconds,
    )
    return request, snapped_duration


def run_smoke(
    args: argparse.Namespace,
    *,
    client_cls: Type[Any] | None = None,
) -> dict[str, Any]:
    """Execute one logical Runs-API smoke and return sanitized metadata only."""

    client_type = HermesRunsClient if client_cls is None else client_cls
    base_url = validate_base_url(args.base_url)
    provider = _validate_route_value(args.provider, "provider")
    model = _validate_route_value(args.model, "model")
    request, snapped_duration = _build_authoritative_request(args)
    request_text = serialize_request(request)

    client = client_type(
        base_url=base_url,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    run_kwargs: dict[str, Any] = {
        "input": request_text,
        "instructions": STABLE_INSTRUCTIONS,
        "session_id": f"h3-smoke:{request['request_id']}",
        "timeout_seconds": float(args.timeout_seconds),
    }
    if provider:
        run_kwargs["provider"] = provider
    if model:
        run_kwargs["model"] = model

    # HermesRunsClient.run performs one logical submit followed by bounded
    # polling/cancellation. No retry path here can instantiate or submit a
    # replacement run.
    run_result = client.run(**run_kwargs)
    run_id_sha256 = _remote_identifier_sha256(
        getattr(run_result, "run_id", None)
    )
    status = _safe_status(getattr(run_result, "status", None))
    elapsed_seconds = _safe_elapsed(getattr(run_result, "elapsed_seconds", None))
    raw_output = getattr(run_result, "output", None)
    if not isinstance(raw_output, str):
        raise SmokeValidationError("Hermes run returned invalid output metadata.")
    parsed = parse_result(raw_output, request=request)

    if parsed.quality_report.get("hard_errors"):
        raise SmokeValidationError(
            "Hermes response reported hard errors; the prompt was rejected."
        )
    if len(parsed.h3_prompt) > MAX_PROMPT_CHARS:
        raise SmokeValidationError(
            "H3 prompt exceeds the local hard character limit."
        )

    local = process(
        parsed.h3_prompt,
        CowboyContext(
            subjects=[],
            duration_seconds=snapped_duration,
            task_type="",
            mode=SUPPORTED_MODE,
            known_shot_times=[],
            is_editing=False,
            dialogue_text=args.dialogue,
            lyrics=args.lyrics,
            wired_pictures=0,
            has_video=False,
            has_audio=False,
            multi_shot_requested=False,
        ),
    )
    if local.retry_errors:
        raise SmokeValidationError("local H3 validation failed.")
    if local.char_count > MAX_PROMPT_CHARS:
        raise SmokeValidationError(
            "H3 prompt exceeds the local hard character limit."
        )
    spoken_blocks = _spoken_blocks(local.prompt)
    spoken_literals = tuple(
        literal for literal in (args.dialogue, args.lyrics) if literal != ""
    )
    if any(
        not any(literal in block for block in spoken_blocks)
        for literal in spoken_literals
    ):
        raise SmokeValidationError("exact literal validation failed.")
    if any(
        literal not in local.prompt
        for literal in args.visible_text
        if literal != ""
    ):
        raise SmokeValidationError("exact literal validation failed.")

    return {
        "request_id": request["request_id"],
        "run_id_sha256": run_id_sha256,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "selected_candidate_id_sha256": _remote_identifier_sha256(
            parsed.selected_candidate_id
        ),
        "prompt_char_count": local.char_count,
        "prompt_sha256": hashlib.sha256(local.prompt.encode("utf-8")).hexdigest(),
        "local_fixes": _safe_diagnostics(local.applied_fixes),
        "local_warnings": _safe_diagnostics(local.warnings),
        "reported_tools": _safe_tools(parsed.reported_tools),
        "verified_tool_events": [],
    }


def _safe_failure_message(exc: BaseException) -> str:
    """Map failures to fixed messages without echoing untrusted exception text."""

    if isinstance(exc, HermesAuthenticationError):
        return _AUTH_FAILURE_GUIDANCE
    if isinstance(exc, HermesClientError):
        if str(exc) == MISSING_API_KEY_GUIDANCE:
            return MISSING_API_KEY_GUIDANCE
        return _GENERIC_CLIENT_FAILURE
    if isinstance(exc, ContractError):
        return "Hermes response contract validation failed."
    if isinstance(exc, SmokeInputError):
        return str(exc)
    if isinstance(exc, SmokeValidationError):
        return str(exc)
    return "H3 Hermes smoke failed safely; inspect local gateway logs."


def main(
    argv: Sequence[str] | None = None,
    *,
    client_cls: Type[Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI entry point. Parser failures exit 2; runtime/validation failures return 1."""

    parser = build_parser()
    args = parser.parse_args(argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    try:
        result = run_smoke(args, client_cls=client_cls)
    except BaseException as exc:
        # KeyboardInterrupt/SystemExit cannot arise from parser here; treating an
        # interruption as a safe nonzero smoke failure avoids a traceback that
        # could include private values from an injected transport.
        print(f"error: {_safe_failure_message(exc)}", file=err)
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
