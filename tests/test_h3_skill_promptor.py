"""
Transport + node-flow tests for the H3 Skill Promptor.

A real loopback HTTP server fakes the OpenAI-compatible endpoint (the
openai SDK needs an actual socket - same approach rationale as
test_h3_hermes_client.py) and records every request body, so the wire
shape and the corrective-retry flow are asserted without any LLM.

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_skill_promptor.py
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
    for sub in ("nodes", "utils", "utils.h3_prompt", "utils.h3_cowboy",
                "utils.h3_skill"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.utils.h3_cowboy.spec import EXAMPLE_REF_GENERATION  # noqa: E402
from TrentNodes.utils.h3_prompt.backends import SEED_MAX  # noqa: E402
from TrentNodes.utils.h3_skill import client as skill_client  # noqa: E402
from TrentNodes.utils import llamacpp_server  # noqa: E402
from TrentNodes.nodes.h3_skill_promptor import H3SkillPromptor  # noqa: E402


class _Fake(BaseHTTPRequestHandler):
    requests = []          # captured chat bodies
    replies = []           # queue: str, or (text, finish_reason) tuples

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
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


# ------------------------------------------------------------------ client

def test_chat_wire_shape():
    server, base_url = _start_fake()
    try:
        _Fake.requests.clear()
        _Fake.replies[:] = ["hello"]
        message = skill_client.build_user_message(
            "context text", [("<Picture 1> (ref)", "QUJD")]
        )
        text, usage = skill_client.chat(
            base_url,
            [{"role": "system", "content": "sys"}, message],
            seed=2**33 + 7,
            temperature=0.7,
            max_tokens=1234,
            reasoning_effort="low",
            reasoning_budget=777,
            reasoning_budget_message="budget spent",
            model="fake-model",
        )
        assert text == "hello"
        assert usage["completion_tokens"] == 20
        assert usage["finish_reason"] == "stop"
        sent = _Fake.requests[-1]
        assert sent["messages"][0] == {"role": "system", "content": "sys"}
        parts = sent["messages"][1]["content"]
        assert parts[0] == {"type": "text", "text": "context text"}
        assert parts[1]["text"] == "<Picture 1> (ref)"
        assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert sent["temperature"] == 0.7
        assert sent["max_tokens"] == 1234
        assert sent["seed"] == (2**33 + 7) % (SEED_MAX + 1)
        assert sent["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert sent["reasoning_budget_tokens"] == 777
        assert sent["reasoning_budget_message"] == "budget spent"
    finally:
        server.shutdown()


def test_chat_connection_refused_is_actionable():
    try:
        skill_client.chat(
            "http://127.0.0.1:9/v1",
            [{"role": "user", "content": "x"}],
            timeout_s=2.0,
        )
        assert False, "must raise"
    except RuntimeError as exc:
        assert "127.0.0.1:9" in str(exc)


# ------------------------------------------------------------------ node flow

def _run_node(replies, mode="ref2va", duration=30.0,
              reasoning_effort="low", **extra):
    server, base_url = _start_fake()
    try:
        _Fake.requests.clear()
        _Fake.replies[:] = list(replies)
        node = H3SkillPromptor()
        prompt, checkpoint, report = node.generate(
            mode=mode,
            creative_brief="a test brief",
            gguf_model="ignored",
            mmproj="none",
            duration_seconds=duration,
            max_frames_to_analyze=8,
            temperature=0.7,
            reasoning_effort=reasoning_effort,
            seed=1,
            base_url=base_url,
            **extra,
        )
        return prompt, checkpoint, report, list(_Fake.requests)
    finally:
        server.shutdown()


def test_node_valid_first_try():
    good = EXAMPLE_REF_GENERATION.strip()
    prompt, checkpoint, report, requests = _run_node([good])
    assert prompt == good
    assert checkpoint == "MiniMax-H3-Base-Ref2VA"
    assert "validation: PASS" in report
    assert "corrective retry used: no" in report
    assert len(requests) == 1
    system = requests[0]["messages"][0]["content"]
    assert "MiniMax H3 prompting" in system      # the skill document rode along


def test_node_retries_then_passes():
    good = EXAMPLE_REF_GENERATION.strip()
    prompt, checkpoint, report, requests = _run_node(["not a prompt", good])
    assert prompt == good
    assert "corrective retry used: yes" in report
    assert "validation: PASS" in report
    assert len(requests) == 2
    retry_messages = requests[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"   # prior bad output
    assert "violates the skill checklist" in retry_messages[-1]["content"]


def test_node_reports_remaining_failures():
    prompt, checkpoint, report, requests = _run_node(["bad", "still bad"])
    assert "VALIDATION FAILED" in report
    assert prompt == "still bad"                       # never silently edited
    assert len(requests) == 2


def test_node_strips_fence_and_think():
    good = EXAMPLE_REF_GENERATION.strip()
    wrapped = "<think>musing</think>\n```\n" + good + "\n```"
    prompt, _, report, _ = _run_node([wrapped])
    assert prompt == good
    assert "stripped a leaked <think> block" in report
    assert "stripped a markdown code fence" in report


def test_soundscaper_outputs_anchor_audio_sections():
    good = EXAMPLE_REF_GENERATION.strip()
    prompt, _, report, requests = _run_node(
        [good],
        source_soundscape="Rain taps a tin roof over a low room tone.",
        source_music="A sparse felt-piano motif at slow tempo.",
        sound_log="00:00-00:03 Rain. 00:03 Door creak.",
    )
    assert prompt == good
    assert "audio sections anchored to measured source audio" in report
    context = requests[0]["messages"][1]["content"]
    assert "MEASURED AUDIO" in context
    assert "Rain taps a tin roof" in context
    assert "felt-piano motif" in context
    assert "Door creak" in context


def test_video_batch_on_reference_input_auto_routes():
    import torch
    good = EXAMPLE_REF_GENERATION.strip()
    clip = torch.rand(27, 32, 48, 3)  # a whole video, mis-wired
    prompt, _, report, requests = _run_node(
        [good], reference_images=clip, fps=24.0,
    )
    assert prompt == good
    assert "treated as video_frames" in report
    parts = requests[0]["messages"][1]["content"]
    frames = [p for p in parts if p.get("type") == "image_url"]
    assert 2 <= len(frames) <= 8  # keyframed, not 27 pictures


def test_node_scales_token_budget_with_reasoning_effort():
    # The widget means "prompt-text budget"; each effort adds its own
    # capped thinking allowance on top of the default 3072.
    good = EXAMPLE_REF_GENERATION.strip()
    for effort, allowance in (("low", 2048), ("medium", 3072),
                              ("xhigh", 7168)):
        _, _, report, requests = _run_node([good], reasoning_effort=effort)
        sent = requests[0]
        assert sent["max_tokens"] == 3072 + allowance, effort
        assert sent["reasoning_budget_tokens"] == allowance, effort
        assert "thinking budget is exhausted" in sent["reasoning_budget_message"]
        assert f"3072 prompt + {allowance} thinking" in report


def test_node_starved_thinking_is_actionable():
    # Empty content + finish_reason length must raise the specific
    # "thinking ate the budget" error, not fail the checklist - and
    # must NOT burn a corrective retry (it would starve the same way).
    try:
        _run_node([("", "length")], reasoning_effort="xhigh")
        assert False, "must raise"
    except RuntimeError as exc:
        message = str(exc)
        assert "thinking" in message
        assert "max_tokens" in message
        assert "reasoning_effort" in message
        assert "checklist" not in message
    assert len(_Fake.requests) == 1


def test_node_reports_truncated_nonempty_reply():
    # Non-empty text cut off by the limit: warn in the report, do not
    # raise - the checklist still judges what arrived.
    good = EXAMPLE_REF_GENERATION.strip()
    prompt, _, report, _ = _run_node([(good, "length")])
    assert prompt == good
    assert "finish: length" in report
    assert "WARNING" in report and "token limit" in report


def test_verbose_mirrors_report_and_dumps_payloads():
    import contextlib
    import io
    good = EXAMPLE_REF_GENERATION.strip()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _run_node([good], verbose=True)
    out = buffer.getvalue()
    assert "[H3SkillPromptor] mode: ref2va" in out
    assert "---- system prompt" in out
    assert "---- user context" in out
    assert "---- raw reply (pass 1)" in out
    assert "[H3SkillPromptor] validation: PASS" in out
    # thinking is None on the fake server - its dump must be skipped
    assert "---- thinking" not in out
    # default stays silent
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _run_node([good])
    assert buffer.getvalue() == ""


def test_stop_node_reports_when_idle():
    from TrentNodes.nodes.h3_skill_promptor import H3LocalLLMStop
    llamacpp_server._slots.clear()
    # A port nothing uses - with the DEFAULT port the orphan reaper
    # would SIGTERM a genuinely running server (it did, once).
    (status,) = H3LocalLLMStop().stop(port=19999)
    assert "no llama-server was running" in status


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all promptor tests passed")
