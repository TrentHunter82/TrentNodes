"""Small, dependency-free client for Hermes Agent's asynchronous Runs API.

The client deliberately resolves its bearer credential from one environment
variable only.  It never logs request headers, response bodies, or transport
exceptions, because any of those surfaces can accidentally disclose secrets.
"""

from __future__ import annotations

import json
import ipaddress
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NoReturn, Optional
from urllib.parse import quote, urlsplit


MISSING_API_KEY_GUIDANCE = (
    "Hermes API key missing. Export HERMES_AGENT_API_KEY in the environment "
    "that starts ComfyUI, then restart ComfyUI. Do not put the key in a "
    "workflow widget."
)
_AUTHENTICATION_GUIDANCE = (
    "Hermes API authentication failed. Verify HERMES_AGENT_API_KEY matches "
    "Hermes API_SERVER_KEY, then restart ComfyUI. Do not put the key in a "
    "workflow widget."
)
_REQUIRED_RUN_FEATURES = ("run_submission", "run_status", "run_stop")
_NORMAL_POLL_STATES = frozenset(
    {"started", "queued", "running", "waiting_for_approval"}
)
_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
_TRANSIENT_POLL_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_STATUS_METADATA_FIELDS = ("object", "created_at", "updated_at", "last_event")
_TOOLSETS_CACHE_TTL_SECONDS = 60.0
_TOOLSETS_OBJECT = "list"
_TOOLSETS_ROOT_KEYS = frozenset({"object", "platform", "data"})
_TOOLSET_ENTRY_KEYS = frozenset(
    {"name", "label", "description", "configured", "enabled", "tools"}
)
_ROUTE_MAX_CHARACTERS = 256
_ROUTE_MAX_UTF8_BYTES = 1024
_IDENTIFIER_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_IDENTIFIER_INITIAL_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def _validate_optional_route(value: Any, name: str) -> Optional[str]:
    """Return one bounded route selector, or ``None`` for an omitted route."""

    def invalid() -> NoReturn:
        raise HermesClientError(f"Hermes run {name} route is invalid.")

    if value is None:
        return None
    if type(value) is not str:
        invalid()
    if len(value) > _ROUTE_MAX_CHARACTERS:
        invalid()
    if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
        invalid()
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        invalid()
    if len(encoded) > _ROUTE_MAX_UTF8_BYTES:
        invalid()

    normalized = value.strip()
    return normalized or None


class HermesClientError(RuntimeError):
    """Base class for safe, user-facing Hermes client failures."""


class HermesAuthenticationError(HermesClientError):
    """The gateway rejected the configured bearer credential."""


class HermesUnsupportedRuntimeError(HermesClientError):
    """The connected Hermes runtime lacks a required Runs API feature."""


class HermesRunFailedError(HermesClientError):
    """A submitted Hermes run reached the ``failed`` terminal state."""


class HermesRunCancelledError(HermesClientError):
    """A submitted Hermes run reached the ``cancelled`` terminal state."""


class HermesRunTimeoutError(HermesClientError):
    """A run exceeded its client-side wall-clock deadline."""


def _canonical_loopback_base_url(value: Any) -> str:
    """Validate the credential authority before any bearer can be attached."""

    def invalid() -> NoReturn:
        raise HermesClientError(
            "Hermes API base URL must use the canonical loopback authority."
        ) from None

    if type(value) is not str or not value or value != value.strip():
        invalid()
    if any(
        ord(char) < 32
        or 0x7F <= ord(char) <= 0x9F
        or char.isspace()
        for char in value
    ):
        invalid()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        invalid()
    if (
        parsed.scheme.lower() != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or "?" in value
        or parsed.fragment
        or "#" in value
        or not hostname
        or "%" in hostname
        or port != 8642
    ):
        invalid()
    if hostname.lower() == "localhost":
        canonical_host = "localhost"
    else:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            invalid()
        if not address.is_loopback:
            invalid()
        canonical_host = address.compressed
        if address.version == 6:
            canonical_host = f"[{canonical_host}]"
    return f"http://{canonical_host}:8642"


@dataclass(frozen=True)
class RunResult:
    """Bounded terminal result returned by :class:`HermesRunsClient`."""

    run_id: str
    status: str
    output: str
    usage: Mapping[str, Any]
    session_id: Optional[str]
    model: Optional[str]
    elapsed_seconds: float
    status_metadata: Mapping[str, Any] = field(default_factory=dict)


class _ResponseLimitExceeded(Exception):
    pass


class _TransportFailure(Exception):
    def __init__(self, transient: bool):
        super().__init__("Hermes API transport failed.")
        self.transient = transient


class _TransientPollFailure(Exception):
    pass


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep every request on its original authority, including its bearer."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


class _UrllibTransport:
    """The default bounded transport; HTTP errors remain ordinary responses."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: Optional[bytes],
        timeout: float,
        max_response_bytes: int,
    ) -> tuple[int, Mapping[str, str], bytes]:
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = self._read_bounded(response, max_response_bytes)
                status = int(response.getcode())
                response_headers = (
                    {} if 300 <= status < 400 else dict(response.headers.items())
                )
                return status, response_headers, raw
        except urllib.error.HTTPError as exc:
            try:
                raw = self._read_bounded(exc, max_response_bytes)
                status = int(exc.code)
                response_headers = (
                    {} if 300 <= status < 400 else dict(exc.headers.items())
                )
                return status, response_headers, raw
            finally:
                exc.close()

    @staticmethod
    def _read_bounded(response: Any, limit: int) -> bytes:
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise _ResponseLimitExceeded()
        return raw


Transport = Callable[
    [str, str, Mapping[str, str], Optional[bytes], float, int],
    tuple[int, Mapping[str, str], bytes],
]


class HermesRunsClient:
    """Authenticated polling client for ``/v1/runs``.

    ``transport``, ``clock``, and ``sleep`` are injectable so normal tests can
    be deterministic and never depend on a live gateway.  There is purposely
    no token constructor argument: the only credential source is
    ``HERMES_AGENT_API_KEY`` in the process that starts ComfyUI.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8642",
        *,
        transport: Optional[Transport] = None,
        request_timeout_seconds: float = 15.0,
        poll_interval_seconds: float = 1.0,
        cancellation_grace_seconds: float = 3.0,
        max_response_bytes: int = 1_048_576,
        max_output_bytes: int = 524_288,
        max_submission_attempts: int = 3,
        submission_retry_base_seconds: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        token = os.environ.get("HERMES_AGENT_API_KEY")
        if not token or not token.strip():
            raise HermesClientError(MISSING_API_KEY_GUIDANCE)

        normalized_url = _canonical_loopback_base_url(base_url)
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if cancellation_grace_seconds < 0:
            raise ValueError("cancellation_grace_seconds must not be negative")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if max_submission_attempts < 1:
            raise ValueError("max_submission_attempts must be at least one")
        if submission_retry_base_seconds < 0:
            raise ValueError("submission_retry_base_seconds must not be negative")

        self._base_url = normalized_url
        self._token = token
        self._transport = transport or _UrllibTransport()
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.cancellation_grace_seconds = float(cancellation_grace_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.max_output_bytes = int(max_output_bytes)
        self.max_submission_attempts = int(max_submission_attempts)
        self.submission_retry_base_seconds = float(submission_retry_base_seconds)
        self._clock = clock
        self._sleep = sleep
        self._terminal_run_ids: set[str] = set()
        self._toolsets_cache: Optional[dict[str, Any]] = None
        self._toolsets_cached_at: Optional[float] = None
        self._toolsets_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        """Canonical read-only authority used for diagnostics."""

        return self._base_url

    def __repr__(self) -> str:
        base_url = self._sanitize(self.base_url)
        return (
            f"{type(self).__name__}(base_url={base_url!r}, "
            f"request_timeout_seconds={self.request_timeout_seconds!r}, "
            f"poll_interval_seconds={self.poll_interval_seconds!r})"
        )

    def preflight_capabilities(self) -> Mapping[str, Any]:
        """Verify that the connected runtime supports submit/status/stop."""
        try:
            status, raw = self._http("GET", "/v1/capabilities", None)
        except _TransportFailure:
            raise HermesClientError(
                "Hermes capabilities preflight transport failed. Verify the "
                "gateway is running and the Hermes API base URL is reachable."
            ) from None
        self._raise_for_auth(status)
        if not 200 <= status < 300:
            raise HermesClientError(
                f"Hermes capabilities preflight failed with HTTP {status}."
            )

        payload = self._decode_object(raw, "capabilities preflight")
        features = payload.get("features")
        if not isinstance(features, Mapping):
            missing = list(_REQUIRED_RUN_FEATURES)
        else:
            missing = [
                name for name in _REQUIRED_RUN_FEATURES
                if features.get(name) is not True
            ]
        if missing:
            joined = ", ".join(missing)
            raise HermesUnsupportedRuntimeError(
                "Hermes runtime does not support required Runs API "
                f"capabilities: {joined}. Update Hermes Agent and restart the "
                "gateway."
            )
        return payload

    def discover_toolsets(
        self,
        *,
        force_refresh: bool = False,
    ) -> Mapping[str, Any]:
        """Return a validated copy of the authenticated toolset inventory."""
        self._validate_force_refresh(force_refresh)

        # Serialize cache misses so concurrent callers still issue exactly one
        # authenticated request for a newly cached inventory.
        with self._toolsets_lock:
            now = self._clock()
            if (
                not force_refresh
                and self._toolsets_cache is not None
                and self._toolsets_cached_at is not None
            ):
                age = now - self._toolsets_cached_at
                if 0.0 <= age < _TOOLSETS_CACHE_TTL_SECONDS:
                    return self._copy_toolsets(self._toolsets_cache)

            # Once refresh begins, the old value is no longer a fallback. A
            # failed refresh therefore cannot be returned silently as stale.
            self._toolsets_cache = None
            self._toolsets_cached_at = None
            try:
                status, raw = self._http("GET", "/v1/toolsets", None)
            except _TransportFailure:
                raise HermesClientError(
                    "Hermes toolset discovery transport failed."
                ) from None

            self._raise_for_auth(status)
            if status != 200:
                raise HermesClientError(
                    f"Hermes toolset discovery failed with HTTP {status}."
                )

            payload = self._decode_object(raw, "toolset discovery")
            validated = self._validate_toolsets_payload(payload)
            self._toolsets_cache = validated
            self._toolsets_cached_at = self._clock()
            return self._copy_toolsets(validated)

    def has_enabled_tool(
        self,
        toolset_name: str,
        tool_name: str,
        *,
        force_refresh: bool = False,
    ) -> bool:
        """Check advertised metadata only; this neither runs nor inspects a tool."""
        self._validate_identifier_argument(
            toolset_name,
            maximum_length=64,
            label="toolset_name",
        )
        self._validate_identifier_argument(
            tool_name,
            maximum_length=128,
            label="tool_name",
        )
        self._validate_force_refresh(force_refresh)

        payload = self.discover_toolsets(force_refresh=force_refresh)
        for entry in payload["data"]:
            if entry["name"] != toolset_name:
                continue
            return (
                entry["configured"] is True
                and entry["enabled"] is True
                and tool_name in entry["tools"]
            )
        return False

    def submit(
        self,
        *,
        input: str,
        instructions: str,
        session_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Submit exactly one logical run, retrying only explicit HTTP 429s."""
        if not isinstance(input, str) or not input.strip():
            raise HermesClientError("Hermes run input must not be blank.")
        if not isinstance(instructions, str):
            raise HermesClientError("Hermes run instructions must be text.")
        if not isinstance(session_id, str) or not session_id.strip():
            raise HermesClientError("Hermes run session_id must not be blank.")

        normalized_provider = _validate_optional_route(provider, "provider")
        normalized_model = _validate_optional_route(model, "model")

        body: dict[str, Any] = {
            "input": input,
            "instructions": instructions,
            "session_id": session_id,
        }
        if normalized_provider is not None:
            body["provider"] = normalized_provider
        if normalized_model is not None:
            body["model"] = normalized_model

        for attempt in range(1, self.max_submission_attempts + 1):
            try:
                status, raw = self._http("POST", "/v1/runs", body)
            except _TransportFailure:
                raise HermesClientError(
                    "Hermes run submission transport failed. The outcome is "
                    "unknown, so the client will not resubmit automatically."
                ) from None

            self._raise_for_auth(status)
            if status == 429:
                if attempt >= self.max_submission_attempts:
                    raise HermesClientError(
                        "Hermes run submission was rate limited after "
                        f"{self.max_submission_attempts} attempts. Retry later "
                        "or increase the Hermes API server concurrent-run capacity."
                    )
                delay = self.submission_retry_base_seconds * (2 ** (attempt - 1))
                self._sleep(delay)
                continue
            if 500 <= status <= 599:
                raise HermesClientError(
                    f"Hermes run submission returned HTTP {status}. The server "
                    "may have accepted the run, so the client will not resubmit "
                    "automatically. Check Hermes run status or server logs before "
                    "retrying."
                )
            if not 200 <= status < 300:
                raise HermesClientError(
                    f"Hermes run submission failed with HTTP {status}."
                )

            payload = self._decode_object(raw, "run submission")
            try:
                run_id = self._validate_run_id(payload.get("run_id"))
            except HermesClientError:
                raise HermesClientError(
                    "Hermes run submission returned an unsafe run_id; the client "
                    "will not resubmit automatically."
                ) from None
            return run_id

        raise AssertionError("submission retry loop exhausted unexpectedly")

    def run(
        self,
        *,
        input: str,
        instructions: str,
        session_id: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 900.0,
        interruption_check: Optional[Callable[[], None]] = None,
    ) -> RunResult:
        """Preflight, submit, and poll one run through a terminal state."""
        normalized_provider = _validate_optional_route(provider, "provider")
        normalized_model = _validate_optional_route(model, "model")
        self.preflight_capabilities()
        run_id = self.submit(
            input=input,
            instructions=instructions,
            session_id=session_id,
            provider=normalized_provider,
            model=normalized_model,
        )
        started_at = self._clock()
        return self.wait(
            run_id,
            timeout_seconds=timeout_seconds,
            interruption_check=interruption_check,
            started_at=started_at,
        )

    def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float = 900.0,
        interruption_check: Optional[Callable[[], None]] = None,
        started_at: Optional[float] = None,
    ) -> RunResult:
        """Poll an existing run without ever submitting a replacement run."""
        self._validate_run_id(run_id)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        began = self._clock() if started_at is None else float(started_at)
        deadline = began + float(timeout_seconds)

        while True:
            now = self._clock()
            if now >= deadline:
                self._cancel_and_grace_poll(run_id)
                raise HermesRunTimeoutError(
                    f"Hermes run timed out (run_id={self._safe_run_id(run_id)})."
                )

            if interruption_check is not None:
                try:
                    interruption_check()
                except BaseException:
                    self._cancel_and_grace_poll(run_id)
                    raise

            remaining = deadline - self._clock()
            if remaining <= 0:
                self._cancel_and_grace_poll(run_id)
                raise HermesRunTimeoutError(
                    f"Hermes run timed out (run_id={self._safe_run_id(run_id)})."
                )

            try:
                payload = self._get_status(
                    run_id,
                    timeout_seconds=min(self.request_timeout_seconds, remaining),
                )
            except _TransientPollFailure:
                self._sleep_until(deadline)
                continue

            if self._clock() >= deadline:
                self._cancel_and_grace_poll(run_id)
                raise HermesRunTimeoutError(
                    f"Hermes run timed out (run_id={self._safe_run_id(run_id)})."
                )

            state = payload.get("status")
            if state in _NORMAL_POLL_STATES:
                self._sleep_until(deadline)
                continue
            if state == "completed":
                self._terminal_run_ids.add(run_id)
                return self._completed_result(run_id, payload, began)
            if state == "failed":
                self._terminal_run_ids.add(run_id)
                raise HermesRunFailedError(
                    f"Hermes run failed (run_id={self._safe_run_id(run_id)})."
                )
            if state == "cancelled":
                self._terminal_run_ids.add(run_id)
                raise HermesRunCancelledError(
                    f"Hermes run was cancelled (run_id={self._safe_run_id(run_id)})."
                )

            safe_state = self._sanitize(str(state))
            raise HermesClientError(
                f"Hermes run {self._safe_run_id(run_id)} returned unknown run "
                f"status {safe_state!r}."
            )

    def stop(self, run_id: str) -> bool:
        """Best-effort, idempotent stop; return whether a request was accepted."""
        self._validate_run_id(run_id)
        if run_id in self._terminal_run_ids:
            return False

        path = f"/v1/runs/{quote(run_id, safe='')}/stop"
        try:
            status, _raw = self._http("POST", path, {})
        except (HermesClientError, _TransportFailure):
            return False

        self._raise_for_auth(status)
        if status in {404, 409, 410, 429} or 500 <= status <= 599:
            return False
        return 200 <= status < 300

    def _get_status(
        self,
        run_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Mapping[str, Any]:
        path = f"/v1/runs/{quote(run_id, safe='')}"
        try:
            status, raw = self._http(
                "GET",
                path,
                None,
                timeout_seconds=timeout_seconds,
            )
        except _TransportFailure as exc:
            if exc.transient:
                raise _TransientPollFailure() from None
            raise HermesClientError("Hermes run status transport failed.") from None

        self._raise_for_auth(status)
        if status in _TRANSIENT_POLL_HTTP_STATUSES:
            raise _TransientPollFailure()
        if not 200 <= status < 300:
            raise HermesClientError(
                f"Hermes run status request failed with HTTP {status} "
                f"(run_id={self._safe_run_id(run_id)})."
            )

        payload = self._decode_object(raw, "run status")
        returned_id = payload.get("run_id")
        if returned_id is not None and returned_id != run_id:
            raise HermesClientError(
                "Hermes run status response identified a different run than the "
                "one requested."
            )
        return payload

    def _completed_result(
        self,
        run_id: str,
        payload: Mapping[str, Any],
        began: float,
    ) -> RunResult:
        output = payload.get("output", "")
        if not isinstance(output, str):
            raise HermesClientError(
                f"Hermes completed run returned non-text output "
                f"(run_id={self._safe_run_id(run_id)})."
            )
        if len(output.encode("utf-8")) > self.max_output_bytes:
            raise HermesClientError(
                f"Hermes completed output exceeded the output size limit "
                f"(run_id={self._safe_run_id(run_id)})."
            )

        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        else:
            usage = dict(usage)

        session_id = payload.get("session_id")
        if not isinstance(session_id, str):
            session_id = None
        model = payload.get("model")
        if not isinstance(model, str):
            model = None

        metadata = {
            name: self._sanitize_metadata_value(payload.get(name))
            for name in _STATUS_METADATA_FIELDS
        }
        elapsed = max(0.0, self._clock() - began)
        return RunResult(
            run_id=run_id,
            status="completed",
            output=output,
            usage=usage,
            session_id=session_id,
            model=model,
            elapsed_seconds=elapsed,
            status_metadata=metadata,
        )

    def _cancel_and_grace_poll(self, run_id: str) -> None:
        try:
            self.stop(run_id)
        except BaseException:
            pass

        if self.cancellation_grace_seconds <= 0:
            return
        try:
            deadline = self._clock() + self.cancellation_grace_seconds
            while self._clock() < deadline:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return
                try:
                    payload = self._get_status(
                        run_id,
                        timeout_seconds=min(self.request_timeout_seconds, remaining),
                    )
                    state = payload.get("status")
                    if state in _TERMINAL_STATES:
                        self._terminal_run_ids.add(run_id)
                        return
                    # ``stopping`` is valid only here, after this client requested
                    # cancellation.  Other active states are tolerated until grace
                    # expires because Hermes stop is cooperative.
                    if state not in _NORMAL_POLL_STATES and state != "stopping":
                        return
                except BaseException:
                    pass
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return
                self._sleep(min(self.poll_interval_seconds, remaining))
        except BaseException:
            # Cancellation cleanup must never replace a timeout or the
            # original ComfyUI interruption exception, including when the
            # injected clock or sleep callbacks fail.
            return

    def _sleep_until(self, deadline: float) -> None:
        remaining = deadline - self._clock()
        if remaining > 0:
            self._sleep(min(self.poll_interval_seconds, remaining))

    def _http(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> tuple[int, bytes]:
        # Revalidate at the credential-owning wire boundary. This remains
        # fail-closed even if a caller deliberately corrupts private state.
        authority = _canonical_loopback_base_url(self._base_url)
        body = None
        if payload is not None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            request_timeout = self.request_timeout_seconds
            if timeout_seconds is not None:
                request_timeout = min(request_timeout, float(timeout_seconds))
            result = self._transport(
                method,
                f"{authority}{path}",
                headers,
                body,
                request_timeout,
                self.max_response_bytes,
            )
        except _ResponseLimitExceeded:
            raise HermesClientError(
                "Hermes API response exceeded the configured response size limit."
            ) from None
        except Exception as exc:
            transient = isinstance(
                exc,
                (TimeoutError, ConnectionError, urllib.error.URLError, OSError),
            )
            raise _TransportFailure(transient=transient) from None

        try:
            status, _response_headers, raw = result
            status = int(status)
        except Exception:
            raise HermesClientError(
                "Hermes API transport returned an invalid response envelope."
            ) from None
        if not isinstance(raw, (bytes, bytearray)):
            raise HermesClientError(
                "Hermes API transport returned a non-byte response body."
            )
        raw_bytes = bytes(raw)
        if len(raw_bytes) > self.max_response_bytes:
            raise HermesClientError(
                "Hermes API response exceeded the configured response size limit."
            )
        return status, raw_bytes

    @classmethod
    def _validate_toolsets_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        def invalid() -> NoReturn:
            raise HermesClientError(
                "Hermes toolset discovery response was invalid."
            )

        if set(payload.keys()) != _TOOLSETS_ROOT_KEYS:
            invalid()
        if payload.get("object") != _TOOLSETS_OBJECT:
            invalid()

        platform = payload.get("platform")
        if not isinstance(platform, str) or not cls._is_bounded_identifier(
            platform,
            maximum_length=64,
        ):
            invalid()

        data = payload.get("data")
        if not isinstance(data, list) or len(data) > 64:
            invalid()

        validated_entries: list[dict[str, Any]] = []
        equivalent_names: set[str] = set()
        for entry in data:
            if not isinstance(entry, Mapping):
                invalid()
            if set(entry.keys()) != _TOOLSET_ENTRY_KEYS:
                invalid()

            name = entry.get("name")
            if not isinstance(name, str) or not cls._is_bounded_identifier(
                name,
                maximum_length=64,
            ):
                invalid()
            equivalent_name = name.casefold()
            if equivalent_name in equivalent_names:
                invalid()
            equivalent_names.add(equivalent_name)

            label = entry.get("label")
            description = entry.get("description")
            if (
                not isinstance(label, str)
                or len(label) > 128
                or not isinstance(description, str)
                or len(description) > 4096
            ):
                invalid()
            try:
                label_bytes = label.encode("utf-8")
                description_bytes = description.encode("utf-8")
            except UnicodeEncodeError:
                invalid()
            if len(label_bytes) > 512 or len(description_bytes) > 16384:
                invalid()

            configured = entry.get("configured")
            enabled = entry.get("enabled")
            if type(configured) is not bool or type(enabled) is not bool:
                invalid()

            tools = entry.get("tools")
            if not isinstance(tools, list) or len(tools) > 128:
                invalid()
            validated_tools: list[str] = []
            equivalent_tools: set[str] = set()
            for tool in tools:
                if not isinstance(tool, str) or not cls._is_bounded_identifier(
                    tool,
                    maximum_length=128,
                ):
                    invalid()
                equivalent_tool = tool.casefold()
                if equivalent_tool in equivalent_tools:
                    invalid()
                equivalent_tools.add(equivalent_tool)
                validated_tools.append(tool)

            validated_entries.append(
                {
                    "name": name,
                    "configured": configured,
                    "enabled": enabled,
                    "tools": validated_tools,
                }
            )

        return {
            "object": _TOOLSETS_OBJECT,
            "platform": platform,
            "data": validated_entries,
        }

    @staticmethod
    def _copy_toolsets(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "object": payload["object"],
            "platform": payload["platform"],
            "data": [
                {
                    "name": entry["name"],
                    "configured": entry["configured"],
                    "enabled": entry["enabled"],
                    "tools": list(entry["tools"]),
                }
                for entry in payload["data"]
            ],
        }

    @staticmethod
    def _is_bounded_identifier(value: Any, *, maximum_length: int) -> bool:
        return (
            isinstance(value, str)
            and 1 <= len(value) <= maximum_length
            and value[0] in _IDENTIFIER_INITIAL_CHARACTERS
            and all(character in _IDENTIFIER_CHARACTERS for character in value)
        )

    @classmethod
    def _validate_identifier_argument(
        cls,
        value: Any,
        *,
        maximum_length: int,
        label: str,
    ) -> None:
        if not cls._is_bounded_identifier(value, maximum_length=maximum_length):
            raise HermesClientError(
                f"Hermes toolset discovery {label} is invalid."
            )

    @staticmethod
    def _validate_force_refresh(force_refresh: Any) -> None:
        if type(force_refresh) is not bool:
            raise HermesClientError(
                "Hermes toolset discovery force_refresh must be a boolean."
            )

    def _decode_object(self, raw: bytes, operation: str) -> Mapping[str, Any]:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON object member")
                result[key] = value
            return result

        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=unique_object,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ):
            raise HermesClientError(
                f"Hermes {operation} returned invalid JSON."
            ) from None
        if not isinstance(payload, Mapping):
            raise HermesClientError(
                f"Hermes {operation} returned a non-object JSON response."
            )
        return payload

    def _raise_for_auth(self, status: int) -> None:
        if status in {401, 403}:
            raise HermesAuthenticationError(_AUTHENTICATION_GUIDANCE)

    def _sanitize(self, text: str) -> str:
        if self._token:
            return text.replace(self._token, "[REDACTED]")
        return text

    def _safe_run_id(self, run_id: str) -> str:
        return self._sanitize(run_id)

    def _sanitize_metadata_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._sanitize(value[:512])
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return None

    @staticmethod
    def _validate_run_id(run_id: Any) -> str:
        if (
            type(run_id) is not str
            or not 1 <= len(run_id) <= 128
            or run_id[0] not in _IDENTIFIER_INITIAL_CHARACTERS
            or any(char not in _IDENTIFIER_CHARACTERS for char in run_id)
        ):
            raise HermesClientError("Hermes run_id is invalid.")
        return run_id


__all__ = [
    "MISSING_API_KEY_GUIDANCE",
    "HermesAuthenticationError",
    "HermesClientError",
    "HermesRunCancelledError",
    "HermesRunFailedError",
    "HermesRunTimeoutError",
    "HermesRunsClient",
    "HermesUnsupportedRuntimeError",
    "RunResult",
]
