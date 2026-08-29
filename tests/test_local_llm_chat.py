"""
Node-flow tests for Ask Local LLM (GGUF).

Same approach as test_h3_skill_promptor.py: a real loopback HTTP server
fakes the OpenAI-compatible endpoint (the openai SDK needs an actual
socket) and records every request body, so message assembly, history
chaining, and think-stripping are asserted without any LLM.

    venv/bin/python custom_nodes/TrentNodes/tests/test_local_llm_chat.py
"""

import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils", "utils.h3_prompt", "utils.h3_skill"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.nodes.local_llm_chat import (  # noqa: E402
    LocalLLMChat,
    _parse_history,
    _strip_think,
)


class _Fake(BaseHTTPRequestHandler):
    requests = []
    replies = []

    def log_message(self, *args):
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path.endswith("/models"):
            self._send({"object": "list", "data": [{"id": "fake-model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _Fake.requests.append(body)
        reply = _Fake.replies.pop(0) if _Fake.replies else "EMPTY"
        if isinstance(reply, tuple):
            text, finish = reply
        else:
            text, finish = reply, "stop"
        self._send({
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                      "total_tokens": 30},
        })


def _start_fake():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _ask(base_url, **overrides):
    node = LocalLLMChat()
    kwargs = dict(
        prompt="say hi",
        system_prompt="sys prompt",
        gguf_model="unused.gguf",
        mmproj="none",
        temperature=0.7,
        reasoning_effort="low",
        seed=0,
        max_tokens=512,
        base_url=base_url,
    )
    kwargs.update(overrides)
    return node.ask(**kwargs)


# --------------------------------------------------------------- helpers

def test_parse_history():
    assert _parse_history("") == []
    assert _parse_history("  \n") == []
    turns = [{"role": "user", "content": "a"},
             {"role": "assistant", "content": "b"}]
    assert _parse_history(json.dumps(turns)) == turns
    for bad in ("not json", "{}", '[{"role": "system", "content": "x"}]',
                '[{"role": "user"}]'):
        try:
            _parse_history(bad)
            assert False, f"must raise on {bad!r}"
        except RuntimeError:
            pass


def test_strip_think():
    assert _strip_think("plain") == ("plain", False)
    assert _strip_think("<think>hmm</think>\nanswer") == ("answer", True)
    # fences are intentional in chat answers - keep them
    assert _strip_think("```py\nx\n```") == ("```py\nx\n```", False)


# ------------------------------------------------------------- node flow

def test_basic_ask_and_history_out():
    server, base_url = _start_fake()
    try:
        _Fake.requests.clear()
        _Fake.replies[:] = ["hello there"]
        response, history_json, info = _ask(base_url)
        assert response == "hello there"
        sent = _Fake.requests[-1]
        assert sent["messages"][0] == {"role": "system",
                                       "content": "sys prompt"}
        assert sent["messages"][1] == {"role": "user", "content": "say hi"}
        assert sent["model"] == "fake-model"  # attach picks up the alias
        assert sent["max_tokens"] == 512 + 2048  # reply + low thinking
        history = json.loads(history_json)
        assert history == [
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        assert "latency" in info
    finally:
        server.shutdown()


def test_history_chains_between_calls():
    server, base_url = _start_fake()
    try:
        _Fake.requests.clear()
        _Fake.replies[:] = ["first", "second"]
        _, history1, _ = _ask(base_url, prompt="q1")
        _, history2, _ = _ask(base_url, prompt="q2", history_json=history1)
        sent = _Fake.requests[-1]
        roles = [m["role"] for m in sent["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert sent["messages"][1]["content"] == "q1"
        assert sent["messages"][2]["content"] == "first"
        assert sent["messages"][3]["content"] == "q2"
        assert len(json.loads(history2)) == 4
    finally:
        server.shutdown()


def test_input_text_appended_and_think_stripped():
    server, base_url = _start_fake()
    try:
        _Fake.requests.clear()
        _Fake.replies[:] = ["<think>pondering</think>\nclean answer"]
        response, _, info = _ask(
            base_url, prompt="improve this", input_text="a cat on a mat"
        )
        assert response == "clean answer"
        assert "stripped a leaked <think> block" in info
        sent_user = _Fake.requests[-1]["messages"][1]["content"]
        assert sent_user.startswith("improve this")
        assert "--- INPUT TEXT ---" in sent_user
        assert sent_user.endswith("a cat on a mat")
    finally:
        server.shutdown()


def test_empty_prompt_raises():
    server, base_url = _start_fake()
    try:
        try:
            _ask(base_url, prompt="   ")
            assert False, "must raise"
        except RuntimeError as exc:
            assert "prompt is empty" in str(exc)
    finally:
        server.shutdown()


def test_thinking_starvation_raises():
    server, base_url = _start_fake()
    try:
        _Fake.replies[:] = [("", "length")]
        try:
            _ask(base_url)
            assert False, "must raise"
        except RuntimeError as exc:
            assert "max_tokens" in str(exc)
    finally:
        server.shutdown()


def test_truncated_reply_warns():
    server, base_url = _start_fake()
    try:
        _Fake.replies[:] = [("partial answer", "length")]
        response, _, info = _ask(base_url)
        assert response == "partial answer"
        assert "cut short" in info
    finally:
        server.shutdown()


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
