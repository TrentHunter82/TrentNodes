"""Offline contract, cancellation, cache, reference and subscription transport checks.

Run from the ComfyUI environment: python tests/test_h3_codex.py
"""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for name, path in (("TrentNodes", ROOT), ("TrentNodes.utils", ROOT / "utils"),
                   ("TrentNodes.nodes", ROOT / "nodes")):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

from TrentNodes.utils.h3_codex import client, prompt, skill
from TrentNodes.utils.h3_cowboy import spec
import torch

FAKE_SERVER = r'''
import json, os, sys, time
scenario = os.environ.get("H3_TEST_SCENARIO", "ok")
def send(message):
    print(json.dumps(message), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    with open(os.environ["H3_TEST_LOG"], "a") as handle:
        handle.write(json.dumps(request) + "\n")
    method, rid = request.get("method"), request.get("id")
    if rid is None:
        continue
    result = {}
    if method == "initialize":
        result = {"userAgent": "fake"}
    if method == "account/read":
        result = {"account": {"type": "apiKey" if scenario == "apikey" else "chatgpt"}}
    if method == "config/read":
        result = {"config": {"mcp_servers": {"test": {}}, "plugins": {"plugin@test": {}}}}
    if method == "thread/start":
        result = {"thread": {"id": "thread"}, "model": "configured-model"}
    if method == "turn/start":
        item = {"method": "item/completed", "params": {"threadId": "thread", "item": {
            "type": "agentMessage", "text": json.dumps({"status": "ready"})}}}
        if scenario == "early":
            send(item)
        send({"id": rid, "result": {"turn": {"id": "turn"}}})
        if scenario == "wait":
            time.sleep(30)
            continue
        if scenario == "approval":
            send({"id": "approval", "method": "item/commandExecution/requestApproval", "params": {}})
            continue
        if scenario != "early":
            send(item)
        send({"method": "turn/completed", "params": {"threadId": "thread", "turn": {
            "id": "turn", "status": "failed" if scenario == "expired" else "completed",
            "error": {"codexErrorInfo": "unauthorized"} if scenario == "expired" else None}}})
        continue
    send({"id": rid, "result": result})
'''


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "server.py"
        self.script.write_text(FAKE_SERVER)
        self.log = self.root / "wire.jsonl"
        real_popen = subprocess.Popen
        self.commands = []
        def spawn(command, **kwargs):
            self.commands.append((command, kwargs["env"]))
            return real_popen([sys.executable, str(self.script)], **kwargs)
        self.addCleanup(patch.stopall)
        patch.object(client, "executable", return_value="codex").start()
        patch.object(client.subprocess, "Popen", side_effect=spawn).start()
        patch.dict(os.environ, {"H3_TEST_LOG": str(self.log), "OPENAI_API_KEY": "must-not-forward", "CODEX_API_KEY": "must-not-forward"}).start()

    def session(self, **kwargs):
        return client.CodexSession(self.root, "only write prompts", **kwargs)

    def test_subscription_and_protocol(self):
        with self.session() as session:
            self.assertEqual(session.turn([{"type": "text", "text": "hi"}], {}), {"status": "ready"})
        self.assertIsNotNone(session.process.poll())
        command, env = self.commands[0]
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_API_KEY", env)
        self.assertIn('forced_login_method="chatgpt"', command)
        wire = [json.loads(line) for line in self.log.read_text().splitlines()]
        start = next(x["params"] for x in wire if x.get("method") == "thread/start")
        self.assertNotIn("model", start)
        self.assertEqual(start["environments"], [])
        self.assertTrue(start["ephemeral"])
        self.assertFalse(start["config"]["mcp_servers.test.enabled"])

    def test_event_before_ack_is_kept(self):
        with patch.dict(os.environ, {"H3_TEST_SCENARIO": "early"}), self.session() as session:
            self.assertEqual(session.turn([], {}), {"status": "ready"})

    def test_api_key_login_refused(self):
        with patch.dict(os.environ, {"H3_TEST_SCENARIO": "apikey"}):
            with self.assertRaisesRegex(client.CodexError, "subscription sign-in"):
                with self.session():
                    pass

    def test_expired_login_actionable(self):
        with patch.dict(os.environ, {"H3_TEST_SCENARIO": "expired"}), self.session() as session:
            with self.assertRaisesRegex(client.CodexError, "login has expired"):
                session.turn([], {})

    def test_timeout_stops_process(self):
        with patch.dict(os.environ, {"H3_TEST_SCENARIO": "wait"}):
            with self.assertRaisesRegex(client.CodexError, "timed out"):
                with self.session(timeout=0.5) as session:
                    session.turn([], {})
            self.assertIsNotNone(session.process.poll())

    def test_cancellation_has_no_retry(self):
        def stop():
            if self.log.exists() and "turn/start" in self.log.read_text():
                raise InterruptedError("cancelled")
        with patch.dict(os.environ, {"H3_TEST_SCENARIO": "wait"}):
            with self.assertRaises(InterruptedError):
                with self.session(interrupt=stop) as session:
                    session.turn([], {})
            self.assertIsNotNone(session.process.poll())
            self.assertEqual(self.log.read_text().count('"method": "turn/start"'), 1)

    def test_never_approves_tools(self):
        with patch.dict(os.environ, {"H3_TEST_SCENARIO": "approval"}), self.session() as session:
            with self.assertRaisesRegex(client.CodexError, "no action was approved"):
                session.turn([], {})


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = dict(brief="A baker opens his bakery.", mode="t2va", duration=8.0,
                         action="generate", prompt_text="", refinement="", context="",
                         dialogue="", reference_roles="", source_soundscape="", source_music="",
                         sound_log="", model="", effort="", revision=0, timeout=90)
        self.calls = []
        self.responses = [{"prompt_body": spec.EXAMPLE_BASE_T2VA, "assumptions": []}]
        owner = self
        class FakeSession:
            model_name = "test-model"
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def turn(self, inputs, schema):
                owner.calls.append(inputs)
                return owner.responses.pop(0)
        self.session = FakeSession
        self.addCleanup(patch.stopall)
        patch.object(prompt, "load_official_skill", return_value={name: "Official guide" for name in skill.FILES}).start()

    def run_prompt(self, **changes):
        return prompt.generate(self.temp.name, session_factory=self.session, **(self.args | changes))

    def test_cache_and_revision(self):
        one = self.run_prompt()
        count = len(self.calls)
        two = self.run_prompt(prompt_text="Ignored in generate mode")
        self.assertEqual(one["prompt"], two["prompt"])
        self.assertEqual(count, len(self.calls))
        self.assertIn("Reused saved result", two["report"])
        self.responses.append({"prompt_body": spec.EXAMPLE_BASE_T2VA, "assumptions": []})
        self.run_prompt(revision=1)
        self.assertGreater(len(self.calls), count)

    def test_locked_is_exact_and_offline(self):
        with patch.object(prompt, "load_official_skill", side_effect=AssertionError("network")):
            result = self.run_prompt(action="locked", prompt_text="  edited\nprompt  ")
        self.assertEqual(result["prompt"], "  edited\nprompt  ")
        self.assertEqual(self.calls, [])

    def test_refine_preserves_context(self):
        self.run_prompt(action="refine", prompt_text="old prompt", refinement="slower camera")
        request = self.calls[0][0]["text"]
        self.assertIn("old prompt", request)
        self.assertIn("slower camera", request)

    def test_repair_once_and_report_remaining(self):
        self.responses[:] = [{"prompt_body": "bad prompt", "assumptions": []}] * 2
        result = self.run_prompt()
        self.assertEqual(len(self.calls), 2)
        self.assertIn("review the diagnostics", result["report"])

    def test_mode_inference(self):
        picture = torch.zeros(1, 16, 16, 3)
        self.assertEqual(prompt.resolve_mode("auto"), "t2va")
        self.assertEqual(prompt.resolve_mode("auto", first_frame=picture), "i2va")
        self.assertEqual(prompt.resolve_mode("auto", last_frame=picture), "l2va")
        self.assertEqual(prompt.resolve_mode("auto", first_frame=picture, last_frame=picture), "fl2va")
        self.assertEqual(prompt.resolve_mode("auto", reference_images=picture), "ref2va")

    def test_reference_labels_and_video_evidence(self):
        refs = torch.zeros(2, 16, 16, 3)
        frames = torch.rand(12, 16, 16, 3)
        images, labels, notes, counts = prompt.prepare_references("ref2va", None, None, refs, frames, 6, 4, None, lambda: None)
        self.assertEqual(counts, {"Picture": 2, "Video": 1, "Audio": 0})
        self.assertIn("<Picture 1>", images[0][0])
        self.assertIn("<Picture 2>", images[1][0])
        self.assertIn("Audio was not inspected", notes[0])
        self.assertLessEqual(len(images), 6)
        self.assertTrue(any("Unknown reference" in x for x in prompt.check_body("<Picture 3>", "ref2va", 6, counts)))

    def test_keyframes_require_matching_picture_count(self):
        with self.assertRaisesRegex(ValueError, "needs 2"):
            prompt.prepare_references("fl2va", torch.zeros(1, 8, 8, 3), None, None, None, 24, 8, None, lambda: None)

    def test_official_example_calibration(self):
        for mode, example in spec.EXAMPLE_FOR_BASE_MODE.items():
            body = example[example.index("integrated_multimodal_description:"):]
            short_mode = {"base_T2VA": "t2va", "base_I2VA": "i2va", "base_FL2VA": "fl2va", "base_L2VA": "l2va"}[mode]
            self.assertEqual(prompt.check_body(body, short_mode, 15, {"Picture": 2, "Audio": 0, "Video": 0}), [])
        self.assertEqual(prompt.check_body(spec.EXAMPLE_REF_GENERATION, "ref2va", 15, {"Picture": 4, "Video": 2, "Audio": 1}), [])


class SkillTests(unittest.TestCase):
    def test_pinned_download_and_verified_cache(self):
        import hashlib
        import io
        payload = b"test official skill"
        files = {"SKILL.md": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, patch.dict(skill.FILES, files, clear=True):
            with patch.object(skill.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as fetch:
                self.assertEqual(skill.load_official_skill(directory), {"SKILL.md": payload.decode()})
                self.assertIn(skill.REVISION, fetch.call_args.args[0])
            with patch.object(skill.urllib.request, "urlopen", side_effect=AssertionError("unexpected network")):
                skill.load_official_skill(directory)

    def test_corrupt_download_not_installed(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(skill.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong content")):
                with self.assertRaisesRegex(RuntimeError, "integrity check"):
                    skill.load_official_skill(directory)
            self.assertFalse((Path(directory) / "skills" / skill.REVISION / "SKILL.md").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
