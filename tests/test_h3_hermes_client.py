"""Hermes Runs API client tests: local-only, deterministic, and credential-safe.

Run directly from any working directory:

    /home/trent/ComfyUI/venv/bin/python \
        /home/trent/ComfyUI/custom_nodes/TrentNodes/tests/test_h3_hermes_client.py
"""

from __future__ import annotations

import contextlib
import hmac
import io
import json
import logging
import os
import secrets
import sys
import traceback
import types
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from http.client import HTTPMessage
from unittest import mock
from urllib.parse import urlsplit

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("utils", "utils.h3_hermes"):
        module = types.ModuleType(f"TrentNodes.{sub}")
        module.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = module

from TrentNodes.utils.h3_hermes.client import (  # noqa: E402
    MISSING_API_KEY_GUIDANCE,
    HermesAuthenticationError,
    HermesClientError,
    HermesRunCancelledError,
    HermesRunFailedError,
    HermesRunTimeoutError,
    HermesRunsClient,
    HermesUnsupportedRuntimeError,
    RunResult,
)


SUPPORTED_CAPABILITIES = {
    "object": "hermes.api_server.capabilities",
    "features": {
        "run_submission": True,
        "run_status": True,
        "run_stop": True,
    },
}

VALID_TOOLSETS = {
    "object": "list",
    "platform": "api_server",
    "data": [
        {
            "name": "web",
            "label": "Web",
            "description": "Search and extract public web pages.",
            "configured": True,
            "enabled": True,
            "tools": ["web_search", "web_extract"],
        },
        {
            "name": "vision",
            "label": "Vision",
            "description": "Analyze supplied images.",
            "configured": False,
            "enabled": True,
            "tools": ["vision_analyze"],
        },
        {
            "name": "terminal",
            "label": "Terminal",
            "description": "Execute local shell commands.",
            "configured": True,
            "enabled": False,
            "tools": ["terminal"],
        },
    ],
}

DISCOVERED_TOOLSETS = {
    "object": VALID_TOOLSETS["object"],
    "platform": VALID_TOOLSETS["platform"],
    "data": [
        {
            "name": entry["name"],
            "configured": entry["configured"],
            "enabled": entry["enabled"],
            "tools": entry["tools"],
        }
        for entry in VALID_TOOLSETS["data"]
    ],
}


@dataclass
class CapturedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: object
    timeout: float
    max_response_bytes: int


class FakeTransport:
    """Injected transport whose scripted routes never leave this process."""

    def __init__(self):
        self.routes = defaultdict(deque)
        self.requests: list[CapturedRequest] = []

    def add(self, method, path, status=200, data=None):
        self.routes[(method, path)].append((status, data))
        return self

    def fail(self, method, path, error):
        self.routes[(method, path)].append(error)
        return self

    def __call__(self, method, url, headers, body, timeout, max_response_bytes):
        path = urlsplit(url).path
        parsed_body = json.loads(body.decode("utf-8")) if body else None
        request = CapturedRequest(
            method=method,
            path=path,
            headers=dict(headers),
            body=parsed_body,
            timeout=timeout,
            max_response_bytes=max_response_bytes,
        )
        self.requests.append(request)

        queue = self.routes[(method, path)]
        if not queue:
            raise AssertionError(f"unexpected request: {method} {path}")
        scripted = queue.popleft()
        if isinstance(scripted, BaseException):
            raise scripted
        status, data = scripted
        raw = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
        return status, {}, raw

    def count(self, method, path):
        return sum(
            request.method == method and request.path == path
            for request in self.requests
        )


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


@contextlib.contextmanager
def api_key_environment():
    """Install a per-test ephemeral key without persisting or printing it."""
    name = "HERMES_AGENT_API_KEY"
    previous = os.environ.get(name)
    key = secrets.token_urlsafe(32)
    os.environ[name] = key
    try:
        yield key
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@contextlib.contextmanager
def missing_api_key_environment():
    name = "HERMES_AGENT_API_KEY"
    previous = os.environ.pop(name, None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ[name] = previous


def make_client(transport, clock=None, **overrides):
    clock = clock or FakeClock()
    kwargs = dict(
        base_url="http://127.0.0.1:8642/",
        transport=transport,
        clock=clock.monotonic,
        sleep=clock.sleep,
        poll_interval_seconds=1.0,
        cancellation_grace_seconds=2.0,
        request_timeout_seconds=3.0,
    )
    kwargs.update(overrides)
    return HermesRunsClient(**kwargs), clock


def add_supported_preflight(transport):
    transport.add("GET", "/v1/capabilities", 200, SUPPORTED_CAPABILITIES)


def add_submission(transport, run_id):
    transport.add("POST", "/v1/runs", 202, {"run_id": run_id, "status": "started"})


def add_toolsets(transport, data=None):
    transport.add(
        "GET",
        "/v1/toolsets",
        200,
        VALID_TOOLSETS if data is None else data,
    )


def run_once(client, **overrides):
    kwargs = dict(
        input="structured request",
        instructions="return JSON only",
        session_id="comfyui:h3:request",
        timeout_seconds=10.0,
    )
    kwargs.update(overrides)
    return client.run(**kwargs)


def assert_all_requests_are_authenticated_json(transport, key):
    assert transport.requests
    for request in transport.requests:
        authorization = request.headers.get("Authorization")
        assert authorization is not None
        assert authorization.startswith("Bearer ")
        assert hmac.compare_digest(authorization[len("Bearer "):], key)
        assert request.headers.get("Content-Type") == "application/json"
        assert request.max_response_bytes > 0


# ---------------------------------------------------------------------------
# Credential handling and preflight
# ---------------------------------------------------------------------------


def test_missing_key_has_exact_actionable_guidance():
    with missing_api_key_environment():
        try:
            HermesRunsClient(transport=FakeTransport())
            raise AssertionError("expected missing-key failure")
        except HermesClientError as exc:
            assert str(exc) == MISSING_API_KEY_GUIDANCE
            assert str(exc) == (
                "Hermes API key missing. Export HERMES_AGENT_API_KEY in the "
                "environment that starts ComfyUI, then restart ComfyUI. Do not "
                "put the key in a workflow widget."
            )


def test_client_rejects_non_loopback_authority_before_credentialed_transport():
    rejected = [
        "http://198.51.100.23:8642",
        "https://127.0.0.1:8642",
        "http://user:password@127.0.0.1:8642",
        "http://127.0.0.1:8642/path",
        "http://127.0.0.1:8642?query=1",
        "http://127.0.0.1:8642#fragment",
        "http://127.0.0.1:3000",
        " http://127.0.0.1:8642",
        "http://[::1%25lo]:8642",
    ]
    rejected.extend(
        f"http://[::1%25zone{chr(codepoint)}]:8642"
        for codepoint in range(0x7F, 0xA0)
    )
    with api_key_environment():
        for base_url in rejected:
            transport = FakeTransport()
            try:
                HermesRunsClient(base_url=base_url, transport=transport)
                raise AssertionError("expected loopback-authority rejection")
            except HermesClientError as exc:
                assert "loopback" in str(exc).lower()
                assert base_url not in str(exc)
            assert transport.requests == []

        aliases = (
            ("HTTP://LOCALHOST:08642/", "http://localhost:8642"),
            ("http://127.0.0.1:08642", "http://127.0.0.1:8642"),
            ("HTTP://[0:0:0:0:0:0:0:1]:08642/", "http://[::1]:8642"),
        )
        for supplied, canonical in aliases:
            client = HermesRunsClient(base_url=supplied, transport=FakeTransport())
            assert client.base_url == canonical


def test_client_revalidates_private_authority_at_authenticated_wire_boundary():
    with api_key_environment():
        transport = FakeTransport()
        client = HermesRunsClient(
            base_url="http://127.0.0.1:8642",
            transport=transport,
        )
        try:
            client.base_url = "http://198.51.100.23:8642"
            raise AssertionError("expected public authority to be read-only")
        except AttributeError:
            pass

        # Python callers can still reach private attributes. The wire boundary
        # must independently reject even deliberate private-state corruption.
        client._base_url = "http://198.51.100.23:8642"
        try:
            client.preflight_capabilities()
            raise AssertionError("expected wire-boundary authority rejection")
        except HermesClientError as exc:
            assert "loopback" in str(exc).lower()
            assert "198.51.100.23" not in str(exc)
        assert transport.requests == []


def test_malformed_authority_exception_context_is_sanitized():
    secret = "OPAQUE_USERINFO_SECRET_42"
    malformed = f"http://user:{secret}@127.0.0.1：8642"
    with api_key_environment():
        try:
            HermesRunsClient(base_url=malformed, transport=FakeTransport())
            raise AssertionError("expected malformed-authority rejection")
        except HermesClientError as exc:
            rendered = "".join(traceback.format_exception(exc))
            assert secret not in str(exc)
            assert secret not in rendered


def test_auth_header_json_content_type_and_no_secret_in_repr_logs_or_error():
    transport = FakeTransport().add(
        "GET", "/v1/capabilities", 401, {"error": "not authorized"}
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with api_key_environment() as key:
            client, _clock = make_client(transport)
            representation = repr(client)
            try:
                client.preflight_capabilities()
                raise AssertionError("expected authentication failure")
            except HermesAuthenticationError as exc:
                message = str(exc)
                assert "HERMES_AGENT_API_KEY" in message
                assert "API_SERVER_KEY" in message
                assert key not in message
            assert key not in representation
            assert key not in stream.getvalue()
            assert_all_requests_are_authenticated_json(transport, key)
    finally:
        root.removeHandler(handler)


def test_capabilities_preflight_requires_all_runs_features():
    transport = FakeTransport().add(
        "GET",
        "/v1/capabilities",
        200,
        {
            "features": {
                "run_submission": True,
                "run_status": True,
                "run_stop": False,
            }
        },
    )
    with api_key_environment():
        client, _clock = make_client(transport)
        try:
            client.preflight_capabilities()
            raise AssertionError("expected unsupported-runtime failure")
        except HermesUnsupportedRuntimeError as exc:
            assert "run_stop" in str(exc)
            assert "update hermes agent" in str(exc).lower()


def test_every_http_response_is_bounded():
    transport = FakeTransport().add(
        "GET", "/v1/capabilities", 200, b"x" * 65
    )
    with api_key_environment():
        client, _clock = make_client(transport, max_response_bytes=64)
        try:
            client.preflight_capabilities()
            raise AssertionError("expected bounded-response failure")
        except HermesClientError as exc:
            assert "response size limit" in str(exc).lower()


def test_default_transport_ignores_proxy_environment_for_bearer_requests():
    captured = []

    def record_http_open(_handler, request):
        captured.append(request)
        raise urllib.error.URLError("offline interception")

    proxy_environment = {
        "http_proxy": "http://198.51.100.23:8080",
        "HTTP_PROXY": "http://198.51.100.23:8080",
        "no_proxy": "",
        "NO_PROXY": "",
    }
    with mock.patch.dict(os.environ, proxy_environment, clear=False), \
            mock.patch.object(
                urllib.request.HTTPHandler,
                "http_open",
                record_http_open,
            ), api_key_environment() as key:
        client = HermesRunsClient(base_url="http://127.0.0.1:8642")
        try:
            client.preflight_capabilities()
            raise AssertionError("expected offline transport failure")
        except HermesClientError:
            pass

    assert len(captured) == 1
    request = captured[0]
    assert request.host == "127.0.0.1:8642"
    assert request.full_url == "http://127.0.0.1:8642/v1/capabilities"
    headers = dict(request.header_items())
    assert headers.get("Authorization") == f"Bearer {key}"
    assert "198.51.100.23" not in request.full_url


def test_default_transport_never_follows_redirect_or_forwards_authorization():
    opaque_marker = "sk-proj-7f4c9a2b6d1e8f30"
    redirect_url = f"http://198.51.100.23/capture/{opaque_marker}"
    requests = []
    installed_handlers = []

    class FakeResponse:
        def __init__(self, status, raw):
            self._status = status
            self._raw = io.BytesIO(raw)
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return self._status

        def read(self, limit=-1):
            return self._raw.read(limit)

    class RedirectAwareFakeOpener:
        def __init__(self, handlers):
            self._redirect_handler = next(
                handler for handler in handlers
                if isinstance(handler, urllib.request.HTTPRedirectHandler)
            )

        def open(self, request, timeout):
            del timeout
            requests.append(request)
            headers = HTTPMessage()
            headers["Location"] = redirect_url
            redirected = self._redirect_handler.redirect_request(
                request,
                io.BytesIO(b"{}"),
                302,
                "Found",
                headers,
                redirect_url,
            )
            if redirected is not None:
                requests.append(redirected)
                return FakeResponse(200, json.dumps(SUPPORTED_CAPABILITIES).encode())
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "Found",
                headers,
                io.BytesIO(b"{}"),
            )

    def fake_build_opener(*handlers):
        installed_handlers.extend(handlers)
        return RedirectAwareFakeOpener(handlers)

    def unsafe_urlopen(request, timeout):
        # Models the unsafe behavior under review so this test is red while the
        # default transport still delegates to the redirect-following urlopen().
        del timeout
        requests.append(request)
        redirected = urllib.request.Request(
            redirect_url,
            headers=dict(request.header_items()),
            method=request.get_method(),
        )
        requests.append(redirected)
        return FakeResponse(200, json.dumps(SUPPORTED_CAPABILITIES).encode())

    with mock.patch.object(
        urllib.request, "build_opener", side_effect=fake_build_opener
    ), mock.patch.object(urllib.request, "urlopen", side_effect=unsafe_urlopen):
        with api_key_environment() as key:
            client = HermesRunsClient(base_url="http://127.0.0.1:8642")
            try:
                client.preflight_capabilities()
                raise AssertionError("expected redirect to remain an HTTP 302 failure")
            except HermesClientError as exc:
                message = str(exc)
                assert "HTTP 302" in message
                assert redirect_url not in message
                assert opaque_marker not in message
                assert key not in message

            try:
                client.discover_toolsets()
                raise AssertionError("expected toolset redirect to remain HTTP 302")
            except HermesClientError as exc:
                message = str(exc)
                assert "HTTP 302" in message
                assert redirect_url not in message
                assert opaque_marker not in message
                assert key not in message

            assert any(
                isinstance(handler, urllib.request.HTTPRedirectHandler)
                for handler in installed_handlers
            )
            assert any(
                isinstance(handler, urllib.request.ProxyHandler)
                and getattr(handler, "proxies", None) == {}
                for handler in installed_handlers
            )
            assert len(requests) == 2
            assert [request.full_url for request in requests] == [
                "http://127.0.0.1:8642/v1/capabilities",
                "http://127.0.0.1:8642/v1/toolsets",
            ]
            assert all(
                request.get_header("Authorization") == f"Bearer {key}"
                for request in requests
            )
            assert all(request.full_url != redirect_url for request in requests)


# ---------------------------------------------------------------------------
# Authenticated, bounded, cached toolset discovery
# ---------------------------------------------------------------------------


def test_toolset_discovery_and_enabled_lookup_are_exact_and_authenticated():
    transport = FakeTransport()
    add_toolsets(transport)

    with api_key_environment() as key:
        client, _clock = make_client(transport)
        discovered = client.discover_toolsets()

        assert discovered == {
            "object": "list",
            "platform": "api_server",
            "data": [
                {
                    "name": entry["name"],
                    "configured": entry["configured"],
                    "enabled": entry["enabled"],
                    "tools": entry["tools"],
                }
                for entry in VALID_TOOLSETS["data"]
            ],
        }
        assert client.has_enabled_tool("web", "web_search") is True
        assert client.has_enabled_tool("vision", "vision_analyze") is False
        assert client.has_enabled_tool("terminal", "terminal") is False
        assert client.has_enabled_tool("web", "missing_tool") is False
        assert client.has_enabled_tool("missing_set", "web_search") is False
        assert client.has_enabled_tool("WEB", "web_search") is False

        assert transport.count("GET", "/v1/toolsets") == 1
        request = transport.requests[0]
        assert request.body is None
        assert request.timeout == 3.0
        assert_all_requests_are_authenticated_json(transport, key)


def test_toolset_discovery_rejects_unknown_wrong_duplicate_and_oversize_shapes():
    malformed_payloads = [
        {
            **VALID_TOOLSETS,
            "unexpected": "must be rejected",
        },
        {
            **VALID_TOOLSETS,
            "data": [{**VALID_TOOLSETS["data"][0], "unexpected": []}],
        },
        {
            **VALID_TOOLSETS,
            "data": [{**VALID_TOOLSETS["data"][0], "configured": 1}],
        },
        {
            **VALID_TOOLSETS,
            "data": [{**VALID_TOOLSETS["data"][0], "enabled": "true"}],
        },
        {
            **VALID_TOOLSETS,
            "data": [
                VALID_TOOLSETS["data"][0],
                {**VALID_TOOLSETS["data"][0], "name": "WEB"},
            ],
        },
        {
            **VALID_TOOLSETS,
            "data": [
                {**VALID_TOOLSETS["data"][0], "tools": ["Tool_A", "tool_a"]}
            ],
        },
        {
            **VALID_TOOLSETS,
            "data": [VALID_TOOLSETS["data"][0]] * 65,
        },
        {
            **VALID_TOOLSETS,
            "data": [
                {
                    **VALID_TOOLSETS["data"][0],
                    "tools": [f"t{i}" for i in range(129)],
                }
            ],
        },
        {
            **VALID_TOOLSETS,
            "data": [
                {**VALID_TOOLSETS["data"][0], "tools": ["t" * 129]}
            ],
        },
        {
            **VALID_TOOLSETS,
            "data": [{**VALID_TOOLSETS["data"][0], "name": "n" * 65}],
        },
        {
            **VALID_TOOLSETS,
            "platform": "p" * 65,
        },
        {
            **VALID_TOOLSETS,
            "object": "other.object",
        },
        {
            **VALID_TOOLSETS,
            "data": [{**VALID_TOOLSETS["data"][0], "tools": "web_search"}],
        },
    ]

    with api_key_environment():
        for malformed in malformed_payloads:
            transport = FakeTransport()
            add_toolsets(transport, malformed)
            client, _clock = make_client(transport)
            try:
                client.discover_toolsets()
                raise AssertionError("expected invalid toolset response failure")
            except HermesClientError as exc:
                assert str(exc) == "Hermes toolset discovery response was invalid."
            assert transport.count("GET", "/v1/toolsets") == 1


def test_toolset_discovery_rejects_malformed_json_and_excessive_depth_safely():
    nested = (
        b'{"object":"list","platform":"api_server","data":'
        + (b"[" * 1100)
        + (b"]" * 1100)
        + b"}"
    )

    with api_key_environment():
        cases = (
            (b"{not-json", "invalid json"),
            (nested, "response was invalid"),
        )
        for raw, expected_message in cases:
            transport = FakeTransport().add("GET", "/v1/toolsets", 200, raw)
            client, _clock = make_client(transport)
            try:
                client.discover_toolsets()
                raise AssertionError("expected safe JSON/depth failure")
            except HermesClientError as exc:
                assert expected_message in str(exc).lower()
            assert transport.count("GET", "/v1/toolsets") == 1


def test_toolset_discovery_rejects_duplicate_json_object_members():
    duplicate_payloads = (
        (
            b'{"object":"list","object":"list","platform":"api_server",'
            b'"data":[]}'
        ),
        (
            b'{"object":"list","platform":"api_server","data":[{'
            b'"name":"web","name":"web","label":"Web",'
            b'"description":"Web tools.","configured":true,'
            b'"enabled":true,"tools":["web_search"]}]}'
        ),
    )

    with api_key_environment():
        for raw in duplicate_payloads:
            transport = FakeTransport().add("GET", "/v1/toolsets", 200, raw)
            client, _clock = make_client(transport)
            try:
                client.discover_toolsets()
                raise AssertionError("expected duplicate JSON member rejection")
            except HermesClientError as exc:
                assert str(exc) == (
                    "Hermes toolset discovery returned invalid JSON."
                )
            assert transport.count("GET", "/v1/toolsets") == 1


def test_toolset_discovery_uses_ttl_boundary_and_force_refresh():
    def response(tool):
        return {
            **VALID_TOOLSETS,
            "data": [{**VALID_TOOLSETS["data"][0], "tools": [tool]}],
        }

    transport = FakeTransport()
    add_toolsets(transport, response("first"))
    add_toolsets(transport, response("at_ttl"))
    add_toolsets(transport, response("forced"))
    clock = FakeClock()

    with api_key_environment():
        client, _clock = make_client(transport, clock=clock)
        assert client.discover_toolsets()["data"][0]["tools"] == ["first"]
        clock.now = 59.999
        assert client.discover_toolsets()["data"][0]["tools"] == ["first"]
        assert transport.count("GET", "/v1/toolsets") == 1

        clock.now = 60.0
        assert client.discover_toolsets()["data"][0]["tools"] == ["at_ttl"]
        assert transport.count("GET", "/v1/toolsets") == 2

        assert client.discover_toolsets(force_refresh=True)["data"][0]["tools"] == ["forced"]
        assert transport.count("GET", "/v1/toolsets") == 3


def test_toolset_discovery_returns_copies_that_cannot_poison_cache():
    transport = FakeTransport()
    add_toolsets(transport)

    with api_key_environment():
        client, _clock = make_client(transport)
        first = client.discover_toolsets()
        first["object"] = "poisoned"
        first["data"][0]["name"] = "poisoned"
        first["data"][0]["tools"].append("poisoned")

        second = client.discover_toolsets()
        assert second == DISCOVERED_TOOLSETS
        assert second is not first
        assert second["data"] is not first["data"]
        assert transport.count("GET", "/v1/toolsets") == 1


def test_failed_forced_refresh_drops_stale_cache_and_failure_is_not_cached():
    refreshed = {
        **VALID_TOOLSETS,
        "data": [{**VALID_TOOLSETS["data"][0], "tools": ["fresh"]}],
    }
    transport = FakeTransport()
    add_toolsets(transport)
    add_toolsets(transport, {})
    add_toolsets(transport, refreshed)

    with api_key_environment():
        client, _clock = make_client(transport)
        assert client.discover_toolsets() == DISCOVERED_TOOLSETS
        try:
            client.discover_toolsets(force_refresh=True)
            raise AssertionError("expected failed forced refresh")
        except HermesClientError:
            pass

        discovered = client.discover_toolsets()
        assert discovered["data"] == [
            {
                "name": "web",
                "configured": True,
                "enabled": True,
                "tools": ["fresh"],
            }
        ]
        assert transport.count("GET", "/v1/toolsets") == 3


def test_toolset_discovery_failures_are_fixed_sanitized_and_not_cached():
    opaque = "secret-body-location-query-header-token-path"
    cases = (
        (401, HermesAuthenticationError, None),
        (403, HermesAuthenticationError, None),
        (204, HermesClientError, "HTTP 204"),
        (302, HermesClientError, "HTTP 302"),
        (500, HermesClientError, "HTTP 500"),
    )

    with api_key_environment() as key:
        for status, error_type, expected in cases:
            transport = FakeTransport()
            transport.add("GET", "/v1/toolsets", status, {"error": opaque})
            add_toolsets(transport)
            client, _clock = make_client(transport)
            try:
                client.discover_toolsets()
                raise AssertionError("expected discovery HTTP failure")
            except error_type as exc:
                message = str(exc)
                assert opaque not in message
                assert key not in message
                if expected is not None:
                    assert expected in message
            assert client.discover_toolsets() == DISCOVERED_TOOLSETS
            assert transport.count("GET", "/v1/toolsets") == 2

        transport = FakeTransport()
        transport.fail("GET", "/v1/toolsets", RuntimeError(opaque))
        add_toolsets(transport)
        client, _clock = make_client(transport)
        try:
            client.discover_toolsets()
            raise AssertionError("expected discovery transport failure")
        except HermesClientError as exc:
            assert str(exc) == "Hermes toolset discovery transport failed."
            assert opaque not in str(exc)
            assert key not in str(exc)
        assert client.discover_toolsets() == DISCOVERED_TOOLSETS
        assert transport.count("GET", "/v1/toolsets") == 2


def test_toolset_decode_shape_and_oversize_failures_are_not_cached():
    empty_valid = {
        "object": "list",
        "platform": "api_server",
        "data": [],
    }
    cases = (
        (b"{not-json", 1_048_576),
        ({}, 1_048_576),
        (b"x" * 129, 128),
    )

    with api_key_environment():
        for failed_response, max_response_bytes in cases:
            transport = FakeTransport()
            transport.add("GET", "/v1/toolsets", 200, failed_response)
            add_toolsets(transport, empty_valid)
            client, _clock = make_client(
                transport,
                max_response_bytes=max_response_bytes,
            )
            try:
                client.discover_toolsets()
                raise AssertionError("expected uncached discovery failure")
            except HermesClientError:
                pass

            assert client.discover_toolsets() == empty_valid
            assert transport.count("GET", "/v1/toolsets") == 2


def test_toolset_discovery_validates_arguments_before_transport():
    invalid_queries = (
        ("", "web_search"),
        (" ", "web_search"),
        ("web/name", "web_search"),
        ("w" * 65, "web_search"),
        ("web", ""),
        ("web", "bad tool"),
        ("web", "t" * 129),
        (1, "web_search"),
        ("web", None),
    )
    transport = FakeTransport()

    with api_key_environment():
        client, _clock = make_client(transport)
        for toolset_name, tool_name in invalid_queries:
            try:
                client.has_enabled_tool(toolset_name, tool_name)
                raise AssertionError("expected query argument validation failure")
            except HermesClientError:
                pass
        for method in (client.discover_toolsets,):
            try:
                method(force_refresh=1)
                raise AssertionError("expected force_refresh validation failure")
            except HermesClientError:
                pass
        try:
            client.has_enabled_tool("web", "web_search", force_refresh="yes")
            raise AssertionError("expected lookup force_refresh validation failure")
        except HermesClientError:
            pass

    assert transport.requests == []


def test_toolset_discovery_response_bound_is_enforced_by_existing_transport_path():
    transport = FakeTransport().add("GET", "/v1/toolsets", 200, b"x" * 65)
    with api_key_environment():
        client, _clock = make_client(transport, max_response_bytes=64)
        try:
            client.discover_toolsets()
            raise AssertionError("expected bounded-response failure")
        except HermesClientError as exc:
            assert "response size limit" in str(exc).lower()

    assert transport.count("GET", "/v1/toolsets") == 1
    assert transport.requests[0].max_response_bytes == 64


# ---------------------------------------------------------------------------
# Submission and successful polling
# ---------------------------------------------------------------------------


def test_invalid_routes_fail_safely_before_run_or_submit_transport():
    class StrSubclass(str):
        pass

    rejected_sentinel = "credential-like-route-secret-SHOULD-NOT-LEAK"
    invalid_values = [
        b"credential-like-route-secret-SHOULD-NOT-LEAK",
        True,
        7,
        StrSubclass(rejected_sentinel),
        f"{rejected_sentinel}\ud800",
        f"{rejected_sentinel}\udfff",
        " " + ("x" * 255) + " ",
        "\U0001f600" * 257,
    ]
    invalid_values.extend(
        f"{rejected_sentinel}{chr(codepoint)}"
        for codepoint in (*range(0x20), *range(0x7F, 0xA0))
    )

    transport = FakeTransport()
    with api_key_environment():
        client, _clock = make_client(transport)
        for route_name in ("provider", "model"):
            for value in invalid_values:
                route = {route_name: value}
                for operation in ("submit", "run"):
                    kwargs = {
                        "input": "structured request",
                        "instructions": "return JSON only",
                        "session_id": "comfyui:h3:request",
                        **route,
                    }
                    try:
                        getattr(client, operation)(**kwargs)
                        raise AssertionError(
                            f"expected invalid {route_name} failure from {operation}"
                        )
                    except HermesClientError as exc:
                        message = str(exc)
                        assert message == f"Hermes run {route_name} route is invalid."
                        assert rejected_sentinel not in message

        # Valid Unicode scalar values use at most four UTF-8 bytes each, so the
        # production 256-character boundary can reach (but not exceed) 1024
        # bytes. Widen only the character bound by one here to prove the byte
        # guard remains independently enforced for both public entry points.
        byte_oversize = "\U0001f600" * 257
        client_module = sys.modules[HermesRunsClient.__module__]
        with mock.patch.object(
            client_module,
            "_ROUTE_MAX_CHARACTERS",
            len(byte_oversize),
        ):
            for operation in ("submit", "run"):
                try:
                    getattr(client, operation)(
                        input="structured request",
                        instructions="return JSON only",
                        session_id="comfyui:h3:request",
                        model=byte_oversize,
                    )
                    raise AssertionError(
                        f"expected UTF-8 byte limit failure from {operation}"
                    )
                except HermesClientError as exc:
                    assert str(exc) == "Hermes run model route is invalid."

    assert len(byte_oversize.encode("utf-8")) > 1024
    assert transport.requests == []


def test_route_normalization_omission_and_boundaries_are_exact():
    transport = FakeTransport()
    for run_id in ("run_trimmed", "run_omitted", "run_boundary"):
        add_submission(transport, run_id)

    provider_boundary = "p" * 256
    model_boundary = "\U0001f600" * 256
    with api_key_environment():
        client, _clock = make_client(transport)
        assert client.submit(
            input="structured request",
            instructions="return JSON only",
            session_id="comfyui:h3:request",
            provider="\u2003provider/route\u3000",
            model="\u00a0model:exact\u2009",
        ) == "run_trimmed"
        assert client.submit(
            input="structured request",
            instructions="return JSON only",
            session_id="comfyui:h3:request",
            provider=None,
            model="\u2003\u3000",
        ) == "run_omitted"
        assert client.submit(
            input="structured request",
            instructions="return JSON only",
            session_id="comfyui:h3:request",
            provider=provider_boundary,
            model=model_boundary,
        ) == "run_boundary"

    assert [request.body for request in transport.requests] == [
        {
            "input": "structured request",
            "instructions": "return JSON only",
            "session_id": "comfyui:h3:request",
            "provider": "provider/route",
            "model": "model:exact",
        },
        {
            "input": "structured request",
            "instructions": "return JSON only",
            "session_id": "comfyui:h3:request",
        },
        {
            "input": "structured request",
            "instructions": "return JSON only",
            "session_id": "comfyui:h3:request",
            "provider": provider_boundary,
            "model": model_boundary,
        },
    ]
    assert len(model_boundary.encode("utf-8")) == 1024
    assert transport.count("POST", "/v1/runs") == 3


def test_submission_payload_omits_blank_route_fields_and_preserves_run_id():
    run_id = "run_preserved_exactly"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET",
        f"/v1/runs/{run_id}",
        200,
        {
            "run_id": run_id,
            "status": "completed",
            "output": "done",
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            "session_id": "comfyui:h3:request",
            "model": "route-model",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_event": "run.completed",
            "private_field": "must not be copied",
        },
    )

    with api_key_environment() as key:
        client, _clock = make_client(transport)
        result = run_once(
            client,
            provider="\u2003\u3000",
            model="\u2003route-model\u3000",
        )

        assert isinstance(result, RunResult)
        assert result.run_id == run_id
        assert result.status == "completed"
        assert result.output == "done"
        assert result.usage["total_tokens"] == 6
        assert result.session_id == "comfyui:h3:request"
        assert result.model == "route-model"
        assert result.elapsed_seconds >= 0
        assert result.status_metadata == {
            "object": None,
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_event": "run.completed",
        }

        submission = next(
            request for request in transport.requests
            if request.method == "POST" and request.path == "/v1/runs"
        )
        assert submission.body == {
            "input": "structured request",
            "instructions": "return JSON only",
            "session_id": "comfyui:h3:request",
            "model": "route-model",
        }
        assert transport.count("POST", "/v1/runs") == 1
        assert_all_requests_are_authenticated_json(transport, key)


def test_polling_transient_errors_reuse_same_run_and_never_resubmit():
    run_id = "run_poll_retry"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    path = f"/v1/runs/{run_id}"
    transport.fail("GET", path, TimeoutError())
    transport.add("GET", path, 503, {"error": "temporarily unavailable"})
    transport.add("GET", path, 200, {"run_id": run_id, "status": "running"})
    transport.add(
        "GET", path, 200,
        {"run_id": run_id, "status": "completed", "output": "finished"},
    )

    with api_key_environment():
        client, _clock = make_client(transport)
        result = run_once(client)

    assert result.run_id == run_id
    assert transport.count("POST", "/v1/runs") == 1
    polled_paths = [
        request.path for request in transport.requests
        if request.method == "GET" and request.path.startswith("/v1/runs/")
    ]
    assert polled_paths == [path, path, path, path]


def test_429_submission_retry_is_bounded_and_exponential():
    run_id = "run_after_capacity_frees"
    transport = FakeTransport()
    add_supported_preflight(transport)
    transport.add("POST", "/v1/runs", 429, {"error": "busy"})
    transport.add("POST", "/v1/runs", 429, {"error": "busy"})
    add_submission(transport, run_id)
    transport.add(
        "GET", f"/v1/runs/{run_id}", 200,
        {"run_id": run_id, "status": "completed", "output": "done"},
    )
    clock = FakeClock()

    with api_key_environment():
        client, _clock = make_client(
            transport,
            clock=clock,
            max_submission_attempts=3,
            submission_retry_base_seconds=0.25,
        )
        result = run_once(client)

    assert result.run_id == run_id
    assert transport.count("POST", "/v1/runs") == 3
    assert clock.sleeps[:2] == [0.25, 0.5]


def test_429_submission_stops_at_configured_attempt_limit():
    transport = FakeTransport()
    add_supported_preflight(transport)
    for _ in range(3):
        transport.add("POST", "/v1/runs", 429, {"error": "busy"})

    with api_key_environment():
        client, _clock = make_client(transport, max_submission_attempts=3)
        try:
            run_once(client)
            raise AssertionError("expected rate-limit failure")
        except HermesClientError as exc:
            assert "3 attempts" in str(exc)

    assert transport.count("POST", "/v1/runs") == 3


def test_ambiguous_submission_5xx_is_not_retried():
    transport = FakeTransport()
    add_supported_preflight(transport)
    transport.add("POST", "/v1/runs", 500, {"error": "ambiguous"})

    with api_key_environment():
        client, _clock = make_client(transport)
        try:
            run_once(client)
            raise AssertionError("expected ambiguous-submission failure")
        except HermesClientError as exc:
            assert "will not resubmit" in str(exc).lower()

    assert transport.count("POST", "/v1/runs") == 1


# ---------------------------------------------------------------------------
# Timeout, interruption, terminal states, and size enforcement
# ---------------------------------------------------------------------------


def test_hard_timeout_stops_then_grace_polls_and_raises_run_id_only():
    run_id = "run_wall_clock_timeout"
    path = f"/v1/runs/{run_id}"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add("GET", path, 200, {"run_id": run_id, "status": "running"})
    transport.add("GET", path, 200, {"run_id": run_id, "status": "running"})
    transport.add("POST", f"{path}/stop", 200, {"run_id": run_id, "status": "stopping"})
    transport.add("GET", path, 200, {"run_id": run_id, "status": "stopping"})
    transport.add("GET", path, 200, {"run_id": run_id, "status": "cancelled"})
    clock = FakeClock()

    with api_key_environment():
        client, _clock = make_client(transport, clock=clock)
        try:
            run_once(client, timeout_seconds=2.0)
            raise AssertionError("expected timeout")
        except HermesRunTimeoutError as exc:
            assert str(exc) == f"Hermes run timed out (run_id={run_id})."

    assert transport.count("POST", f"{path}/stop") == 1
    assert transport.count("POST", "/v1/runs") == 1


def test_status_returned_after_deadline_times_out_and_clamps_poll_request():
    run_id = "run_completed_too_late"
    path = f"/v1/runs/{run_id}"
    clock = FakeClock()

    class DeadlineCrossingTransport(FakeTransport):
        crossed_deadline = False

        def __call__(
            self, method, url, headers, body, timeout, max_response_bytes
        ):
            result = super().__call__(
                method, url, headers, body, timeout, max_response_bytes
            )
            if (
                method == "GET"
                and urlsplit(url).path == path
                and not self.crossed_deadline
            ):
                self.crossed_deadline = True
                clock.now = 1.0
            return result

    transport = DeadlineCrossingTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET", path, 200,
        {"run_id": run_id, "status": "completed", "output": "too late"},
    )
    transport.add(
        "POST", f"{path}/stop", 200,
        {"run_id": run_id, "status": "stopping"},
    )
    transport.add("GET", path, 200, {"run_id": run_id, "status": "cancelled"})

    with api_key_environment():
        client, _clock = make_client(transport, clock=clock)
        try:
            run_once(client, timeout_seconds=1.0)
            raise AssertionError("expected post-poll timeout")
        except HermesRunTimeoutError as exc:
            assert str(exc) == f"Hermes run timed out (run_id={run_id})."

    status_requests = [
        request for request in transport.requests
        if request.method == "GET" and request.path == path
    ]
    assert status_requests[0].timeout == 1.0
    assert status_requests[1].timeout == 2.0
    assert transport.count("POST", f"{path}/stop") == 1
    assert transport.count("POST", "/v1/runs") == 1


def test_interruption_stops_and_reraises_the_original_exception():
    run_id = "run_interrupted"
    path = f"/v1/runs/{run_id}"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add("POST", f"{path}/stop", 200, {"run_id": run_id, "status": "stopping"})
    transport.add("GET", path, 200, {"run_id": run_id, "status": "cancelled"})

    class UserInterrupted(Exception):
        pass

    original = UserInterrupted()
    calls = []

    def interruption_check():
        calls.append(True)
        raise original

    with api_key_environment():
        client, _clock = make_client(transport)
        try:
            run_once(client, interruption_check=interruption_check)
            raise AssertionError("expected original interruption")
        except UserInterrupted as exc:
            assert exc is original

    assert calls == [True]
    assert transport.count("POST", f"{path}/stop") == 1
    assert transport.count("POST", "/v1/runs") == 1


def test_interruption_survives_a_baseexception_from_cleanup_sleep():
    run_id = "run_interrupted_cleanup_sleep"
    path = f"/v1/runs/{run_id}"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "POST", f"{path}/stop", 200,
        {"run_id": run_id, "status": "stopping"},
    )
    transport.add("GET", path, 200, {"run_id": run_id, "status": "running"})

    class UserInterrupted(BaseException):
        pass

    class CleanupSleepFailed(BaseException):
        pass

    original = UserInterrupted()
    cleanup_failure = CleanupSleepFailed()
    clock = FakeClock()

    def interruption_check():
        raise original

    def failing_sleep(seconds):
        clock.sleeps.append(seconds)
        raise cleanup_failure

    with api_key_environment():
        client, _clock = make_client(transport, clock=clock, sleep=failing_sleep)
        try:
            run_once(client, interruption_check=interruption_check)
            raise AssertionError("expected original interruption")
        except BaseException as exc:
            assert exc is original

    assert clock.sleeps == [1.0]
    assert transport.count("POST", f"{path}/stop") == 1
    assert transport.count("POST", "/v1/runs") == 1


def test_timeout_survives_a_baseexception_from_cleanup_sleep():
    run_id = "run_timeout_cleanup_sleep"
    path = f"/v1/runs/{run_id}"
    transport = FakeTransport()
    transport.add(
        "POST", f"{path}/stop", 200,
        {"run_id": run_id, "status": "stopping"},
    )
    transport.add("GET", path, 200, {"run_id": run_id, "status": "running"})

    class CleanupSleepFailed(BaseException):
        pass

    cleanup_failure = CleanupSleepFailed()
    clock = FakeClock()
    clock.now = 1.0

    def failing_sleep(seconds):
        clock.sleeps.append(seconds)
        raise cleanup_failure

    with api_key_environment():
        client, _clock = make_client(transport, clock=clock, sleep=failing_sleep)
        try:
            client.wait(run_id, timeout_seconds=1.0, started_at=0.0)
            raise AssertionError("expected timeout")
        except BaseException as exc:
            assert isinstance(exc, HermesRunTimeoutError)
            assert str(exc) == f"Hermes run timed out (run_id={run_id})."

    assert clock.sleeps == [1.0]
    assert transport.count("POST", f"{path}/stop") == 1


def test_failed_run_raises_typed_failure():
    run_id = "run_failed"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET", f"/v1/runs/{run_id}", 200,
        {"run_id": run_id, "status": "failed", "error": "provider failure"},
    )

    with api_key_environment():
        client, _clock = make_client(transport)
        try:
            run_once(client)
            raise AssertionError("expected failed-run exception")
        except HermesRunFailedError as exc:
            assert run_id in str(exc)


def test_cancelled_run_raises_typed_cancellation():
    run_id = "run_cancelled"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET", f"/v1/runs/{run_id}", 200,
        {"run_id": run_id, "status": "cancelled"},
    )

    with api_key_environment():
        client, _clock = make_client(transport)
        try:
            run_once(client)
            raise AssertionError("expected cancellation")
        except HermesRunCancelledError as exc:
            assert run_id in str(exc)


def test_unknown_state_is_an_explicit_failure():
    run_id = "run_unknown_state"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET", f"/v1/runs/{run_id}", 200,
        {"run_id": run_id, "status": "teleporting"},
    )

    with api_key_environment():
        client, _clock = make_client(transport)
        try:
            run_once(client)
            raise AssertionError("expected unknown-state failure")
        except HermesClientError as exc:
            assert "unknown run status" in str(exc).lower()
            assert "teleporting" in str(exc)


def test_stopping_is_not_normal_success_or_a_normal_poll_state():
    run_id = "run_stopping_unexpectedly"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET", f"/v1/runs/{run_id}", 200,
        {"run_id": run_id, "status": "stopping"},
    )

    with api_key_environment():
        client, _clock = make_client(transport)
        try:
            run_once(client)
            raise AssertionError("expected stopping-state failure")
        except HermesClientError as exc:
            assert "unknown run status" in str(exc).lower()
            assert "stopping" in str(exc)


def test_completed_output_size_cap_is_enforced_in_utf8_bytes():
    run_id = "run_output_too_large"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET", f"/v1/runs/{run_id}", 200,
        {"run_id": run_id, "status": "completed", "output": "éé"},
    )

    with api_key_environment():
        client, _clock = make_client(transport, max_output_bytes=3)
        try:
            run_once(client)
            raise AssertionError("expected output-size failure")
        except HermesClientError as exc:
            assert "output size limit" in str(exc).lower()
            assert run_id in str(exc)


def test_stop_is_best_effort_and_idempotent_for_known_terminal_run():
    run_id = "run_already_done"
    transport = FakeTransport()
    add_supported_preflight(transport)
    add_submission(transport, run_id)
    transport.add(
        "GET", f"/v1/runs/{run_id}", 200,
        {"run_id": run_id, "status": "completed", "output": "done"},
    )

    with api_key_environment():
        client, _clock = make_client(transport)
        result = run_once(client)
        stopped = client.stop(result.run_id)

    assert stopped is False
    assert transport.count("POST", f"/v1/runs/{run_id}/stop") == 0


if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"All {len(tests)} Hermes client tests passed.")
