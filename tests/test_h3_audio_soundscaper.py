"""
H3 Audio Soundscaper tests: wire shape, parsing, and node flow against
the same loopback fake server pattern as test_h3_skill_promptor.py.

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_audio_soundscaper.py
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
    for sub in ("nodes", "utils", "utils.h3_prompt", "utils.h3_cowboy"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402

from TrentNodes.utils.h3_skill.audio_io import audio_to_wav_b64  # noqa: E402
from TrentNodes.utils.h3_skill.audio_prompts import parse_response  # noqa: E402
from TrentNodes.nodes.h3_audio_soundscaper import H3AudioSoundscaper  # noqa: E402

GOOD_REPLY = """sound_log:
00:00-00:02 Low drone swell. Texture: sub-bass, metallic. Category: Ambience.
00:02-00:04 Door slam. Texture: dry, woody. Category: Foley.

overall_soundscape:
A low mechanical drone underpins the room while a heavy wooden door slams shut, its thud decaying quickly in a small space.

non_diegetic_music:
N/A

dialogue:
(S1) [English] We are closed.
"""


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
        else:
            self._send({"object": "list", "data": [{"id": "fake-omni"}],
                        "models": [{"capabilities": ["completion", "multimodal"]}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _Fake.requests.append(json.loads(self.rfile.read(length)))
        reply = _Fake.replies.pop(0) if _Fake.replies else "EMPTY"
        if isinstance(reply, tuple):
            text, finish = reply
        else:
            text, finish = reply, "stop"
        self._send({
            "id": "x", "object": "chat.completion", "model": "fake-omni",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9,
                      "total_tokens": 14},
        })


def _audio(seconds=2.0, rate=16000):
    t = torch.linspace(0, seconds, int(seconds * rate))
    wave = torch.sin(2 * torch.pi * 440 * t) * 0.3
    return {"waveform": wave.view(1, 1, -1), "sample_rate": rate}


def _run(replies, audio="default", **overrides):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        _Fake.requests.clear()
        _Fake.replies[:] = list(replies)
        node = H3AudioSoundscaper()
        result = node.analyze(
            audio=_audio() if audio == "default" else audio,
            gguf_model="ignored", mmproj="auto",
            temperature=0.6, seed=1, base_url=base_url, **overrides,
        )
        return result, list(_Fake.requests)
    finally:
        server.shutdown()


# ------------------------------------------------------------- audio encoding

def test_audio_to_wav_b64_resamples_and_measures():
    b64, duration, truncated = audio_to_wav_b64(_audio(2.0))
    assert abs(duration - 2.0) < 0.01 and not truncated and len(b64) > 1000
    b64, duration, truncated = audio_to_wav_b64(_audio(3.0, rate=48000))
    assert abs(duration - 3.0) < 0.02  # resampled, same seconds
    _, duration, truncated = audio_to_wav_b64(_audio(5.0), max_seconds=2.0)
    assert truncated and abs(duration - 2.0) < 0.01


# ------------------------------------------------------------------ parsing

def test_parse_good_reply():
    sections, errors = parse_response(GOOD_REPLY)
    assert errors == []
    assert sections["non_diegetic_music"] == "N/A"
    assert sections["dialogue"].startswith("(S1) [English]")
    assert "Door slam" in sections["sound_log"]


def test_parse_rejects_missing_header():
    bad = GOOD_REPLY.replace("non_diegetic_music:\nN/A\n\n", "")
    _, errors = parse_response(bad)
    assert errors and "four headers" in errors[0]


def test_parse_rejects_bad_dialogue_shape():
    bad = GOOD_REPLY.replace("(S1) [English] We are closed.",
                             "S1 says: We are closed.")
    _, errors = parse_response(bad)
    assert any("dialogue lines" in e for e in errors)


# ------------------------------------------------------------------ node flow

def test_node_happy_path_and_wire_shape():
    (scape, music, dialogue, log, report), requests = _run([GOOD_REPLY])
    assert "door slams shut" in scape
    assert music == "N/A"
    assert dialogue == "(S1) [English] We are closed."
    assert "output contract: PASS" in report
    sent = requests[0]
    parts = sent["messages"][1]["content"]
    audio_part = [p for p in parts if p.get("type") == "input_audio"]
    assert len(audio_part) == 1
    assert audio_part[0]["input_audio"]["format"] == "wav"
    assert len(audio_part[0]["input_audio"]["data"]) > 1000
    assert "chat_template_kwargs" not in sent  # omni: no reasoning kwarg
    assert "exactly 2.00 seconds" in parts[0]["text"]


def test_node_retries_on_contract_violation():
    (_, _, _, _, report), requests = _run(["not the contract", GOOD_REPLY])
    assert len(requests) == 2
    assert "corrective retry used: yes" in report
    assert "output contract: PASS" in report
    assert "violates the output contract" in requests[1]["messages"][-1]["content"]


def test_node_warns_on_truncated_reply():
    # finish_reason length = the sound log got cut at max_tokens (the
    # omni Instruct model does not think, so the cap is pure reply
    # budget). Warn, never raise - the parsed sections still stand.
    (_, _, _, _, report), _ = _run([(GOOD_REPLY, "length")])
    assert "WARNING" in report and "max_tokens" in report
    (_, _, _, _, report), _ = _run([GOOD_REPLY])
    assert "WARNING" not in report


def test_verbose_mirrors_report_and_dumps_payloads():
    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _run([GOOD_REPLY], verbose=True)
    out = buffer.getvalue()
    assert "[H3AudioSoundscaper] mode: listening" in out
    assert "---- system prompt" in out
    assert "---- user context" in out
    assert "---- raw reply (pass 1)" in out
    # default stays silent
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _run([GOOD_REPLY])
    assert buffer.getvalue() == ""


def test_node_design_mode_text_only():
    prompt = ("[Shot 1] A blacksmith hammers a glowing blade on an "
              "anvil, sparks flying.")
    (scape, music, dialogue, log, report), requests = _run(
        [GOOD_REPLY], audio=None, video_prompt=prompt,
    )
    assert "mode: design" in report
    assert "output contract: PASS" in report
    sent = requests[0]
    # Text-only wire shape: plain string content, no input_audio part.
    assert isinstance(sent["messages"][1]["content"], str)
    assert prompt in sent["messages"][1]["content"]
    assert "sound designer" in sent["messages"][0]["content"]


def test_node_design_mode_needs_a_prompt():
    try:
        _run([GOOD_REPLY], audio=None)
    except RuntimeError as exc:
        assert "video_prompt" in str(exc)
    else:
        raise AssertionError("no-audio + no-prompt should raise")


def test_node_listening_merges_video_prompt_into_context():
    (_, _, _, _, report), requests = _run(
        [GOOD_REPLY], scene_context="A door closes.",
        video_prompt="[Shot 1] A shop interior.",
    )
    assert "mode: listening" in report
    text = requests[0]["messages"][1]["content"][0]["text"]
    assert "A door closes." in text
    assert "[Shot 1] A shop interior." in text


def test_node_surfaces_unparseable_reply():
    (scape, music, dialogue, log, report), _ = _run(["junk", "more junk"])
    assert scape == "" and dialogue == ""
    assert log == "more junk"
    assert "unparseable" in report


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all audio soundscaper tests passed")
