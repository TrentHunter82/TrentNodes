"""Offline tests for the sanitized H3 Hermes Runs-API smoke CLI.

No test in this module opens a socket.  Run directly with:

    /home/trent/ComfyUI/venv/bin/python tests/test_h3_hermes_smoke.py
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from tools import h3_hermes_api_smoke as smoke  # noqa: E402


BASE_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a "
    "medium-wide shot frames a courier ducking under a roller shutter on a "
    "wet loading bay. The camera pushes in with small amplitude at slow "
    "speed as she straightens beyond the doorway.\n\n"
    "overall_soundscape: Rain drums on corrugated steel while the shutter "
    "chain rattles overhead. Boots scuff across wet concrete.\n\n"
    "non_diegetic_music: A low synth pulse moves at a slow tempo, thinning "
    "as she clears the doorway."
)


def prompt_with_literals(
    dialogue: str = "", lyrics: str = "", visible_text: tuple[str, ...] = ()
) -> str:
    additions = []
    if dialogue:
        additions.append(
            f"The courier with a clear voice (S1) says: <d>[English] {dialogue}</d>"
        )
    if lyrics:
        additions.append(
            f"The courier (S1) sings: <d>[English] {lyrics}</d>"
        )
    additions.extend(f'A painted sign reads "{item}" exactly.' for item in visible_text)
    if not additions:
        return BASE_PROMPT
    return BASE_PROMPT.replace(
        "she straightens beyond the doorway.",
        "she straightens beyond the doorway. " + " ".join(additions),
    )


def response_for(request, *, prompt=None, tools=None, extra=None):
    literals = request["exact_literals"]
    if prompt is None:
        prompt = prompt_with_literals(
            literals["dialogue"],
            literals["lyrics"],
            tuple(literals["visible_text"]),
        )
    candidate = {
        "candidate_id": "balanced_1",
        "policy": "literal_minimal",
        "prompt": prompt,
        "score_vector": {"required_intent_coverage": 1.0},
        "critic_findings": [],
    }
    value = {
        "schema_version": "h3_hermes_result/1.0",
        "request_id": request["request_id"],
        "status": "ok",
        "evidence": {
            "observations": [],
            "assumptions": [],
            "uninspected_assets": [],
        },
        "intent_ir": {
            "required_atoms": [request["creative_brief"]],
            "preferred_atoms": [],
            "optional_atoms": [],
            "reference_jobs": [],
        },
        "candidates": [candidate],
        "selected_candidate_id": candidate["candidate_id"],
        "h3_prompt": prompt,
        "repairs": [],
        "quality_report": {
            "hard_errors": [],
            "warnings": [],
            "unresolved_ambiguities": [],
            "reported_tools": ["web_search"] if tools is None else tools,
            "reported_sources": [],
        },
    }
    if extra:
        value.update(extra)
    return value


class FakeClient:
    instances = []
    response_builder = staticmethod(response_for)
    run_id = "run_smoke_123"
    status = "completed"
    elapsed_seconds = 1.23456

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.run_kwargs = None
        type(self).instances.append(self)

    def run(self, **kwargs):
        if self.run_kwargs is not None:
            raise AssertionError("the smoke CLI submitted more than one logical run")
        self.run_kwargs = kwargs
        request = json.loads(kwargs["input"])
        raw = json.dumps(type(self).response_builder(request), ensure_ascii=False)
        return SimpleNamespace(
            run_id=type(self).run_id,
            status=type(self).status,
            output=raw,
            elapsed_seconds=type(self).elapsed_seconds,
        )


@contextlib.contextmanager
def configured_fake(
    *, response_builder=response_for, run_id="run_smoke_123", status="completed"
):
    class ConfiguredFake(FakeClient):
        pass

    ConfiguredFake.instances = []
    ConfiguredFake.response_builder = staticmethod(response_builder)
    ConfiguredFake.run_id = run_id
    ConfiguredFake.status = status
    yield ConfiguredFake


def invoke(argv, client_cls):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = smoke.main(argv, client_cls=client_cls, stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


class SuccessTests(unittest.TestCase):
    def test_success_emits_only_the_stable_compact_sanitized_json_keys(self):
        with configured_fake() as fake:
            code, stdout, stderr = invoke(["--brief", "courier at a loading bay"], fake)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout.count("\n"), 1)
        self.assertNotIn(": ", stdout, "stdout JSON must stay compact")
        result = json.loads(stdout)
        self.assertEqual(set(result), {
            "request_id",
            "run_id_sha256",
            "status",
            "elapsed_seconds",
            "selected_candidate_id_sha256",
            "prompt_char_count",
            "prompt_sha256",
            "local_fixes",
            "local_warnings",
            "reported_tools",
            "verified_tool_events",
        })
        self.assertEqual(
            result["run_id_sha256"],
            hashlib.sha256(b"run_smoke_123").hexdigest(),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["selected_candidate_id_sha256"],
            hashlib.sha256(b"balanced_1").hexdigest(),
        )
        self.assertEqual(result["elapsed_seconds"], 1.235)
        self.assertEqual(result["prompt_char_count"], len(BASE_PROMPT))
        self.assertEqual(
            result["prompt_sha256"],
            hashlib.sha256(BASE_PROMPT.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(result["reported_tools"], ["web_search"])
        self.assertEqual(result["verified_tool_events"], [])
        self.assertNotIn(BASE_PROMPT, stdout)
        self.assertNotIn("creative_brief", stdout)
        self.assertNotIn("raw_response", stdout)

    def test_request_uses_authoritative_contract_stable_instructions_guide_and_grid(self):
        argv = [
            "--base-url", "http://localhost:8642/",
            "--mode", "base_T2VA",
            "--brief", "a precise brief",
            "--dialogue", "Mind the shutter.",
            "--lyrics", "Hold the line.",
            "--visible-text", "BAY 4",
            "--visible-text", "  DÉTOUR  ",
            "--duration", "6.0",
            "--quality-mode", "hero",
            "--research-policy", "always",
            "--timeout-seconds", "321",
            "--poll-interval-seconds", "0.25",
        ]
        with configured_fake() as fake:
            code, _stdout, stderr = invoke(argv, fake)

        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(len(fake.instances), 1)
        client = fake.instances[0]
        self.assertEqual(client.init_kwargs, {
            "base_url": "http://localhost:8642",
            "poll_interval_seconds": 0.25,
        })
        call = client.run_kwargs
        request = json.loads(call["input"])
        self.assertEqual(call["input"], smoke.serialize_request(request))
        self.assertEqual(call["instructions"], smoke.STABLE_INSTRUCTIONS)
        self.assertEqual(call["session_id"], f"h3-smoke:{request['request_id']}")
        self.assertEqual(call["timeout_seconds"], 321.0)
        self.assertNotIn("provider", call)
        self.assertNotIn("model", call)
        self.assertEqual(request["h3_mode"], "base_T2VA")
        self.assertEqual(request["creative_brief"], "a precise brief")
        self.assertEqual(request["quality_mode"], "hero")
        self.assertEqual(request["research_policy"], "always")
        self.assertEqual(request["exact_literals"], {
            "dialogue": "Mind the shutter.",
            "lyrics": "Hold the line.",
            "visible_text": ["BAY 4", "  DÉTOUR  "],
        })
        length, snapped = smoke.snap_length(6.0)
        self.assertEqual(request["generation"], {
            "requested_duration_seconds": 6.0,
            "snapped_duration_seconds": snapped,
            "fps": 24.0,
            "width": 768,
            "height": 768,
            "length": length,
        })
        self.assertEqual(request["task"], {
            "task_types": [],
            "video_role": "none",
            "audio_role": "none",
            "constraints": [],
            "cut_timestamps": [],
        })
        self.assertEqual(request["subjects"], [])
        self.assertEqual(request["assets"], [])
        self.assertEqual(
            request["local_h3_format_guide"],
            smoke.prompts_base.build_system_prompt("base_T2VA"),
        )
        self.assertEqual(request["budgets"]["wall_clock_timeout_seconds"], 321)

    def test_nonblank_routing_is_trimmed_and_blank_routing_is_omitted(self):
        with configured_fake() as fake:
            code, _stdout, _stderr = invoke([
                "--brief", "brief",
                "--provider", "  ",
                "--model", " route-model ",
            ], fake)
        self.assertEqual(code, 0)
        call = fake.instances[0].run_kwargs
        self.assertNotIn("provider", call)
        self.assertEqual(call["model"], "route-model")

        with configured_fake() as fake:
            code, _stdout, _stderr = invoke([
                "--brief", "brief",
                "--provider", " route-provider ",
                "--model", "   ",
            ], fake)
        self.assertEqual(code, 0)
        call = fake.instances[0].run_kwargs
        self.assertEqual(call["provider"], "route-provider")
        self.assertNotIn("model", call)

    def test_remote_identifiers_are_only_hashed_and_tools_use_a_fixed_allowlist(self):
        opaque_run_id = "sk-proj-7f4c9a2b6d1e8f30-run"
        opaque_candidate_id = "sk-proj-7f4c9a2b6d1e8f30-candidate"
        opaque_tool = "sk-proj-7f4c9a2b6d1e8f30-tool"

        def dirty(request):
            value = response_for(
                request,
                tools=[
                    "web_search",
                    "web_extract",
                    "terminal",
                    opaque_tool,
                ] + [f"tool_{index}" for index in range(40)],
            )
            value["candidates"][0]["candidate_id"] = opaque_candidate_id
            value["selected_candidate_id"] = opaque_candidate_id
            return value

        with configured_fake(
            response_builder=dirty,
            run_id=opaque_run_id,
        ) as fake:
            code, stdout, stderr = invoke(["--brief", "brief"], fake)

        self.assertEqual((code, stderr), (0, ""))
        self.assertNotIn(opaque_run_id, stdout)
        self.assertNotIn(opaque_candidate_id, stdout)
        self.assertNotIn(opaque_tool, stdout)
        self.assertNotIn("terminal", stdout)
        result = json.loads(stdout)
        self.assertEqual(
            result["run_id_sha256"],
            hashlib.sha256(opaque_run_id.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            result["selected_candidate_id_sha256"],
            hashlib.sha256(opaque_candidate_id.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(result["reported_tools"], ["web_search", "web_extract"])
        self.assertEqual(result["verified_tool_events"], [])

    def test_remote_status_must_match_the_strict_success_allowlist(self):
        opaque_status = "sk-proj-7f4c9a2b6d1e8f30-status"
        with configured_fake(status=opaque_status) as fake:
            code, stdout, stderr = invoke(["--brief", "brief"], fake)

        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("invalid status metadata", stderr)
        self.assertNotIn(opaque_status, stderr)


class FailureTests(unittest.TestCase):
    def assert_safe_failure(self, response_builder, expected_phrase):
        secret = "never-print-this-smoke-secret"
        with configured_fake(response_builder=response_builder) as fake:
            code, stdout, stderr = invoke(["--brief", secret], fake)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn(expected_phrase, stderr)
        self.assertNotIn(secret, stderr)
        self.assertNotIn("Authorization", stderr)
        self.assertNotIn("/home/", stderr)

    def test_missing_key_guidance_comes_from_client_failure_without_a_key_option(self):
        class MissingKeyClient:
            def __init__(self, **_kwargs):
                raise smoke.HermesClientError(smoke.MISSING_API_KEY_GUIDANCE)

        code, stdout, stderr = invoke(["--brief", "brief"], MissingKeyClient)
        self.assertEqual((code, stdout), (1, ""))
        self.assertEqual(stderr.strip(), f"error: {smoke.MISSING_API_KEY_GUIDANCE}")
        self.assertNotIn("--api-key", smoke.build_parser().format_help())

    def test_authentication_exception_is_replaced_with_fixed_safe_guidance(self):
        secret = "auth-secret-must-not-leak"

        class AuthFailureClient:
            def __init__(self, **_kwargs):
                raise smoke.HermesAuthenticationError(
                    f"Authorization: Bearer {secret}; /home/private/key"
                )

        code, stdout, stderr = invoke(["--brief", "brief"], AuthFailureClient)
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("authentication failed", stderr.lower())
        self.assertNotIn(secret, stderr)
        self.assertNotIn("Authorization", stderr)
        self.assertNotIn("/home/", stderr)

    def test_contract_failure_is_generic_and_never_echoes_raw_response(self):
        secret = "raw-contract-secret"

        def malformed(_request):
            return {"not_the_contract": f"Authorization: Bearer {secret}"}

        self.assert_safe_failure(malformed, "contract validation failed")

    def test_local_h3_failure_is_nonzero_and_sanitized(self):
        self.assert_safe_failure(
            lambda request: response_for(request, prompt="ordinary prose, not H3"),
            "local H3 validation failed",
        )

    def test_hard_prompt_character_limit_is_a_failure_not_only_a_warning(self):
        huge = (
            "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
            + ("visible action continues. " * 400)
            + "\n\noverall_soundscape: Wind moves through the set."
            + "\n\nnon_diegetic_music: A low synth pulse repeats."
        )
        self.assertGreater(len(huge), smoke.MAX_PROMPT_CHARS)
        self.assert_safe_failure(
            lambda request: response_for(request, prompt=huge),
            "hard character limit",
        )

    def test_each_exact_literal_must_survive_byte_for_byte(self):
        cases = (
            (["--dialogue", "Mind the shutter."], "mind the shutter."),
            (["--lyrics", "Hold the line."], "hold the line."),
            (["--visible-text", "BAY 4"], "Bay 4"),
        )
        for extra_args, changed in cases:
            def changed_literal(request, changed=changed):
                exact = request["exact_literals"]
                prompt = prompt_with_literals(
                    exact["dialogue"], exact["lyrics"],
                    tuple(exact["visible_text"]),
                )
                original = extra_args[1]
                return response_for(request, prompt=prompt.replace(original, changed))

            with self.subTest(option=extra_args[0]):
                with configured_fake(response_builder=changed_literal) as fake:
                    code, stdout, stderr = invoke(
                        ["--brief", "brief"] + extra_args, fake
                    )
                self.assertEqual((code, stdout), (1, ""))
                self.assertIn("exact literal validation failed", stderr)
                self.assertNotIn(extra_args[1], stderr)

    def test_spoken_literal_decoys_outside_d_blocks_fail_closed(self):
        cases = (
            ("--dialogue", "Mind  the shutter.", "mind the shutter."),
            ("--lyrics", "Hold  the line.", "hold the line."),
        )
        for option, literal, changed in cases:
            def decoyed(request, *, literal=literal, changed=changed):
                prompt = prompt_with_literals(
                    request["exact_literals"]["dialogue"],
                    request["exact_literals"]["lyrics"],
                    tuple(request["exact_literals"]["visible_text"]),
                )
                prompt = prompt.replace(literal, changed, 1)
                prompt = prompt.replace(
                    "overall_soundscape: ",
                    f"overall_soundscape: {literal} ",
                )
                return response_for(request, prompt=prompt)

            with self.subTest(option=option):
                with configured_fake(response_builder=decoyed) as fake:
                    code, stdout, stderr = invoke(
                        ["--brief", "brief", option, literal], fake
                    )
                self.assertEqual((code, stdout), (1, ""))
                self.assertIn("exact literal validation failed", stderr)
                self.assertNotIn(literal, stderr)

    def test_malformed_spoken_markup_fails_closed(self):
        def unclosed(request):
            literal = request["exact_literals"]["dialogue"]
            prompt = prompt_with_literals(dialogue=literal).replace("</d>", "", 1)
            return response_for(request, prompt=prompt)

        with configured_fake(response_builder=unclosed) as fake:
            code, stdout, stderr = invoke(
                ["--brief", "brief", "--dialogue", "Mind the shutter."], fake
            )
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("local H3 validation failed", stderr)
        self.assertNotIn("Mind the shutter.", stderr)

    def test_model_reported_hard_errors_fail_closed(self):
        def reported_error(request):
            value = response_for(request)
            value["quality_report"]["hard_errors"] = [
                "Authorization: Bearer should-not-print"
            ]
            return value

        self.assert_safe_failure(reported_error, "reported hard errors")


class ParserAndUrlTests(unittest.TestCase):
    def test_only_root_plain_http_loopback_urls_are_accepted(self):
        accepted = {
            "http://127.0.0.1:8642/": "http://127.0.0.1:8642",
            "http://localhost:8642": "http://localhost:8642",
            "http://[::1]:8642/": "http://[::1]:8642",
            "HTTP://LOCALHOST:8642/": "http://localhost:8642",
            "http://[0:0:0:0:0:0:0:1]:8642/": "http://[::1]:8642",
        }
        for value, expected in accepted.items():
            with self.subTest(value=value):
                self.assertEqual(smoke.validate_base_url(value), expected)

        rejected = (
            "",
            "https://127.0.0.1:8642",
            "http://example.com:8642",
            "http://user:password@127.0.0.1:8642",
            "http://localhost:8642/?query=1",
            "http://127.0.0.1:8642/#fragment",
            "http://[::1]:8642/api",
            "http://127.0.0.1:99999",
            "http://127.99.2.3:9999",
            "http://127.0.0.1:0",
            "http://127.0.0.1:",
            "http://127.0.0.1",
            " http://127.0.0.1:8642",
            "http://127.0.0.1:8642 ",
            "http://127.0.0.1:\t8642",
            "http://127.0.0.1:\n8642",
            "http://127.0.0.1:\r8642",
            "http://127.0.0.1:\v8642",
            "http://127.0.0.1:\f8642",
            "http://127.0.0.1:\x008642",
            "http://127.0.0.1:\x1f8642",
            "http://[::1%25lo]:8642",
            "http://[::1%25zone\x7f]:8642",
            "http://[::1%25zone\x80]:8642",
            "http://127.0.0.1:\u00a08642",
            "not a url",
        )
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(smoke.SmokeInputError):
                smoke.validate_base_url(value)

    def test_parser_exits_two_for_missing_brief_unsupported_mode_and_bad_bounds(self):
        bad_argv = (
            [],
            ["--brief", "brief", "--mode", "ref"],
            ["--brief", "brief", "--duration", "0"],
            ["--brief", "brief", "--timeout-seconds", "29"],
            ["--brief", "brief", "--timeout-seconds", "3601"],
            ["--brief", "brief", "--poll-interval-seconds", "0.01"],
            ["--brief", "brief", "--poll-interval-seconds", "11"],
        )
        for argv in bad_argv:
            with self.subTest(argv=argv), mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    smoke.main(argv, client_cls=FakeClient)
                self.assertEqual(raised.exception.code, 2)

    def test_parser_rejects_secret_shaped_arguments_without_echoing_values(self):
        secret = "parser-secret-must-never-print"
        captured = io.StringIO()
        with mock.patch("sys.stderr", captured):
            with self.assertRaises(SystemExit) as raised:
                smoke.main(
                    ["--brief", "brief", "--api-key", secret],
                    client_cls=FakeClient,
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertNotIn(secret, captured.getvalue())
        self.assertNotIn("api-key", captured.getvalue())

    def test_route_arguments_reject_huge_or_control_values_without_echoing_them(self):
        rejected_values = (
            "r" * 257,
            "😀" * 257,
            "sk-proj-7f4c9a2b6d1e8f30\nnext",
            "sk-proj-7f4c9a2b6d1e8f30\x00next",
            "sk-proj-7f4c9a2b6d1e8f30\x7fnext",
        )
        for option in ("--provider", "--model"):
            for value in rejected_values:
                captured = io.StringIO()
                with self.subTest(option=option, value_length=len(value)):
                    with mock.patch("sys.stderr", captured):
                        with self.assertRaises(SystemExit) as raised:
                            smoke.main(
                                ["--brief", "brief", option, value],
                                client_cls=FakeClient,
                            )
                    self.assertEqual(raised.exception.code, 2)
                    self.assertNotIn(value, captured.getvalue())
                    self.assertNotIn("sk-proj-7f4c9a2b6d1e8f30", captured.getvalue())

    def test_route_argument_bounds_accept_the_exact_utf8_limit(self):
        provider = "p" * 256
        model = "😀" * 256
        with configured_fake() as fake:
            code, _stdout, stderr = invoke(
                [
                    "--brief", "brief",
                    "--provider", provider,
                    "--model", model,
                ],
                fake,
            )
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(fake.instances[0].run_kwargs["provider"], provider)
        self.assertEqual(fake.instances[0].run_kwargs["model"], model)

    def test_programmatic_route_values_are_validated_before_client_creation(self):
        constructed = []

        class MustNotConstruct:
            def __init__(self, **_kwargs):
                constructed.append(True)
                raise AssertionError("client must not be created")

        cases = (
            ("provider", object()),
            ("model", ["not", "text"]),
            ("provider", "p" * 257),
            ("model", "route\nmodel"),
        )
        for field, value in cases:
            args = smoke.build_parser().parse_args(["--brief", "brief"])
            setattr(args, field, value)
            with self.subTest(field=field, value_type=type(value).__name__):
                with self.assertRaises(smoke.SmokeInputError):
                    smoke.run_smoke(args, client_cls=MustNotConstruct)
        self.assertEqual(constructed, [])

    def test_parser_defaults_establish_supported_mode_and_h3_grid_inputs(self):
        args = smoke.build_parser().parse_args(["--brief", "brief"])
        self.assertEqual(args.mode, "base_T2VA")
        self.assertEqual(args.duration, smoke.DEFAULT_DURATION_SECONDS)
        self.assertEqual(args.quality_mode, "balanced")
        self.assertEqual(args.research_policy, "when_uncertain")
        self.assertEqual(args.timeout_seconds, 900)
        self.assertEqual(args.poll_interval_seconds, 1.0)
        self.assertEqual(args.visible_text, [])


if __name__ == "__main__":
    unittest.main()
