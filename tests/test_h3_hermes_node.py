"""Task-5/6 tests for the Base H3 Hermes Prompt Director node.

Only Hermes transport is mocked; image staging uses real tiny Torch tensors.
Run directly with the ComfyUI interpreter:

    /home/trent/ComfyUI/venv/bin/python tests/test_h3_hermes_node.py
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import types
import unittest

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

# The production node imports ComfyUI's interruption hook. Direct-script tests
# run from the custom-node directory, so put the ComfyUI root on sys.path just
# like the other node test harnesses do.
sys.path.insert(0, ROOT)

if "TrentNodes" not in sys.modules:
    package = types.ModuleType("TrentNodes")
    package.__path__ = [PKG]
    sys.modules["TrentNodes"] = package
for sub in (
    "nodes", "utils", "utils.h3_prompt", "utils.h3_cowboy",
    "utils.h3_hermes", "utils.cut_detect",
):
    name = f"TrentNodes.{sub}"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[name] = module

from TrentNodes.nodes import h3_hermes_prompt_director as node_mod  # noqa: E402
from TrentNodes.nodes.ultimate_h3_cowboy_promptor import (  # noqa: E402
    UltimateH3CowboyPromptor,
)
import torch  # noqa: E402

from TrentNodes.utils.h3_cowboy import prompts_base, prompts_ref, spec  # noqa: E402
from TrentNodes.utils.h3_cowboy.wiring import canvas_for, snap_length  # noqa: E402
from TrentNodes.utils.h3_hermes.client import RunResult  # noqa: E402
from TrentNodes.utils.h3_hermes.contract import serialize_request  # noqa: E402
from TrentNodes.utils.h3_hermes.schema import (  # noqa: E402
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
)


SIMPLE = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a "
    "medium-wide shot frames a courier ducking under a roller shutter on a "
    "wet loading bay. The camera pushes in with small amplitude at slow "
    "speed as she straightens up beyond the doorway.\n\n"
    "overall_soundscape: Rain drums on corrugated steel while the shutter "
    "chain rattles overhead. Boots scuff across wet concrete.\n\n"
    "non_diegetic_music: A low synth pulse at a slow tempo, thinning out as "
    "she clears the doorway."
)


def _image(value, *, height=12, width=20, batch=1):
    return torch.full((batch, height, width, 3), float(value), dtype=torch.float32)


def _frames(count=6, *, height=12, width=20):
    frames = torch.zeros((count, height, width, 3), dtype=torch.float32)
    for index in range(count):
        frames[index, :, :, 0] = index / max(1, count - 1)
        frames[index, 2:height - 2, 2 + index:6 + index, 1] = 1.0
    return frames


def _audio(seconds=0.1, sample_rate=8000):
    samples = max(1, int(seconds * sample_rate))
    timeline = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = 0.2 * torch.sin(2 * torch.pi * 440.0 * timeline)
    return {
        "waveform": waveform.reshape(1, 1, -1),
        "sample_rate": sample_rate,
    }


class _Video:
    def __init__(self, images, frame_rate, *, component_audio=None, source=None):
        self.images = images
        self.frame_rate = frame_rate
        self.component_audio = component_audio
        self.source = source
        self.component_calls = 0

    def get_components(self):
        self.component_calls += 1
        return types.SimpleNamespace(
            images=self.images,
            frame_rate=self.frame_rate,
            audio=self.component_audio,
        )

    def get_stream_source(self):
        if self.source is None:
            raise RuntimeError("no reusable source")
        return str(self.source)


def _ref_prompt_for_request(request):
    subjects = request["subjects"]
    task_type = " + ".join(request["task"]["task_types"])
    definitions = []
    retentions = []
    body_subjects = []
    for subject in subjects:
        index = subject["index"]
        tag = f"<Subject {index}>"
        name = subject.get("name") or subject["kind"]
        sources = [f"<{source}>" for source in subject.get("sources", [])]
        source_phrase = (" in " + ", ".join(sources)) if sources else ""
        features = subject.get("features") or "the supplied defining features"
        definitions.append(
            f"{tag} is {name}{source_phrase}, with {features}."
        )
        retentions.append(
            f"{tag} (appears in [Shot 1]): fully_preserved - {features} are retained."
        )
        body_subjects.append(tag)

    literals = request["exact_literals"]
    spoken = []
    if literals["dialogue"]:
        spoken.append(
            f"{body_subjects[0]} (S1) says, <d>[English] {literals['dialogue']}</d>"
        )
    if literals["lyrics"]:
        spoken.append(
            f"{body_subjects[0]} (S2) sings, <d>[English] {literals['lyrics']}</d>"
        )
    action = (
        " and ".join(body_subjects)
        + " move through the requested action in one continuous tracking shot."
    )
    if spoken:
        action += " " + " ".join(spoken)
    music = (
        "A steady instrumental pulse supports the performance."
        if literals["lyrics"]
        else "N/A"
    )
    return (
        "subject_definitions:\n" + "\n".join(definitions)
        + "\n\nsummary:\n"
        + f"[{task_type}] The target video follows the supplied reference intent."
        + "\n\nretention_analysis:\n" + "\n".join(retentions)
        + "\n\ndetailed_description:\n"
        + "The target uses a grounded live-action photographic style.\n"
        + "[Shot 1] " + action
        + "\n\noverall_soundscape:\nRoom tone and movement remain synchronized."
        + "\n\nnon_diegetic_music:\n" + music
    )


def _prompt_for_request(request):
    if request["h3_mode"] == "ref":
        return _ref_prompt_for_request(request)
    if request["h3_mode"] != "base_T2VA":
        return spec.EXAMPLE_FOR_BASE_MODE[request["h3_mode"]]
    literals = request["exact_literals"]
    additions = []
    if literals["dialogue"]:
        additions.append(
            "The courier (S1) says: "
            f"<d>[English] {literals['dialogue']}</d>"
        )
    if literals["lyrics"]:
        additions.append(
            "The courier (S1) sings: "
            f"<d>[English] {literals['lyrics']}</d>"
        )
    for visible in literals["visible_text"]:
        additions.append(f'A painted sign reads "{visible}" exactly.')
    if not additions:
        return SIMPLE
    return SIMPLE.replace(
        "she straightens up beyond the doorway.",
        "she straightens up beyond the doorway. " + " ".join(additions),
    )


def _response(request, *, prompt=None, candidates=None, quality_report=None,
              evidence=None, extra=None):
    prompt = _prompt_for_request(request) if prompt is None else prompt
    if candidates is None:
        candidates = [{
            "candidate_id": "balanced_1",
            "policy": "literal_minimal",
            "prompt": prompt,
            "score_vector": {
                "required_intent_coverage": 1.0,
                "contradictions": 0,
            },
            "critic_findings": [],
        }]
    value = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": request["request_id"],
        "status": "ok",
        "evidence": evidence or {
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
        "candidates": candidates,
        "selected_candidate_id": candidates[0]["candidate_id"],
        "h3_prompt": candidates[0]["prompt"],
        "repairs": [],
        "quality_report": quality_report or {
            "hard_errors": [],
            "warnings": [],
            "unresolved_ambiguities": [],
            "reported_tools": ["web_search"],
            "reported_sources": [],
        },
    }
    if extra:
        value.update(extra)
    return value


@contextlib.contextmanager
def _fake_client(*, response_builder=None, usage=None, run_id="run_task5",
                 status="completed", run_exception=None,
                 interruption_exception=None, video_tool_available=False):
    instances = []
    response_builder = response_builder or (lambda request: _response(request))

    class FakeHermesRunsClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.run_kwargs = None
            self.interruption_calls = 0
            self.tool_queries = []
            instances.append(self)

        def has_enabled_tool(self, toolset_name, tool_name, **kwargs):
            self.tool_queries.append((toolset_name, tool_name, kwargs))
            return video_tool_available

        def run(self, **kwargs):
            self.run_kwargs = kwargs
            kwargs["interruption_check"]()
            self.interruption_calls += 1
            if run_exception is not None:
                raise run_exception
            request = json.loads(kwargs["input"])
            raw = json.dumps(response_builder(request), ensure_ascii=False)
            return RunResult(
                run_id=run_id,
                status=status,
                output=raw,
                usage=usage or {"input_tokens": 100, "output_tokens": 50,
                                "total_tokens": 150},
                session_id=kwargs["session_id"],
                model=kwargs.get("model"),
                elapsed_seconds=1.25,
            )

    original_client = node_mod.HermesRunsClient
    original_interrupt = (
        node_mod.comfy.model_management
        .throw_exception_if_processing_interrupted
    )
    interruption_checks = []

    def interruption_check():
        interruption_checks.append(True)
        if interruption_exception is not None:
            raise interruption_exception

    node_mod.HermesRunsClient = FakeHermesRunsClient
    node_mod.comfy.model_management.throw_exception_if_processing_interrupted = (
        interruption_check
    )
    try:
        yield instances, interruption_checks
    finally:
        node_mod.HermesRunsClient = original_client
        node_mod.comfy.model_management.throw_exception_if_processing_interrupted = (
            original_interrupt
        )


@contextlib.contextmanager
def _temp_staging():
    with tempfile.TemporaryDirectory() as temp:
        original = node_mod.folder_paths.get_temp_directory
        node_mod.folder_paths.get_temp_directory = lambda: temp
        try:
            yield Path(temp)
        finally:
            node_mod.folder_paths.get_temp_directory = original


def _generate(**overrides):
    kwargs = {
        "h3_mode": "base_T2VA",
        "subjects": "",
        "target_description": "a courier ducks under a roller shutter",
        "quality_mode": "balanced",
        "research_policy": "when_uncertain",
        "duration_override": 6.0,
    }
    kwargs.update(overrides)
    return node_mod.H3HermesPromptDirector().generate(**kwargs)


def _named(output):
    return dict(zip(node_mod.H3HermesPromptDirector.RETURN_NAMES, output))


class NodeFaceTests(unittest.TestCase):
    def test_registration_display_category_and_v1_output_contract(self):
        key = "TrentH3HermesPromptDirector"
        self.assertIs(node_mod.NODE_CLASS_MAPPINGS[key],
                      node_mod.H3HermesPromptDirector)
        self.assertEqual(node_mod.NODE_DISPLAY_NAME_MAPPINGS[key],
                         "H3 Hermes Prompt Director")
        self.assertEqual(node_mod.H3HermesPromptDirector.CATEGORY, "Trent/VLM")
        self.assertEqual(node_mod.H3HermesPromptDirector.RETURN_TYPES,
                         UltimateH3CowboyPromptor.RETURN_TYPES)
        self.assertEqual(node_mod.H3HermesPromptDirector.RETURN_NAMES,
                         UltimateH3CowboyPromptor.RETURN_NAMES)
        self.assertEqual(node_mod.H3HermesPromptDirector.RETURN_NAMES[:5], (
            "h3_prompt", "duration_seconds", "fps", "analysis_json",
            "h3_checkpoint_hint",
        ))

    def test_input_contract_has_no_secret_or_v1_provider_widget(self):
        inputs = node_mod.H3HermesPromptDirector.INPUT_TYPES()
        required = inputs["required"]
        optional = inputs["optional"]
        names = set(required) | set(optional)

        self.assertNotIn("vlm_provider", names)
        self.assertNotIn("model", names)
        self.assertNotIn("api_key", names)
        self.assertNotIn("fallback_policy", names)
        self.assertEqual(required["quality_mode"][0],
                         ["fast", "balanced", "hero"])
        self.assertEqual(required["quality_mode"][1]["default"], "balanced")
        self.assertEqual(required["research_policy"][0],
                         ["never", "when_uncertain", "always"])
        self.assertEqual(required["research_policy"][1]["default"],
                         "when_uncertain")
        self.assertEqual(required["h3_mode"][1]["default"], "base_T2VA")

        self.assertEqual(optional["hermes_base_url"][1]["default"],
                         "http://127.0.0.1:8642")
        self.assertTrue(optional["hermes_base_url"][1]["advanced"])
        self.assertLess(optional["timeout_seconds"][1]["min"],
                        optional["timeout_seconds"][1]["max"])
        self.assertLess(optional["poll_interval_seconds"][1]["min"],
                        optional["poll_interval_seconds"][1]["max"])
        self.assertEqual(optional["cleanup_policy"][0],
                         ["delete_on_success", "retain_24h", "retain"])
        self.assertEqual(optional["hermes_provider"][1]["default"], "")
        self.assertEqual(optional["hermes_model"][1]["default"], "")
        self.assertTrue(optional["visible_text"][1]["multiline"])

    def test_six_ordered_image_sockets_and_subject_rows_are_preserved(self):
        optional = node_mod.H3HermesPromptDirector.INPUT_TYPES()["optional"]
        ordered = list(optional)
        image_names = [f"subject_{slot}_image" for slot in range(1, 7)]
        self.assertEqual(
            [name for name in ordered if name in image_names], image_names
        )
        for name in image_names:
            self.assertEqual(optional[name][0], "IMAGE")
        for slot in range(1, 7):
            row = [
                f"subject_{slot}_kind", f"subject_{slot}_name",
                f"subject_{slot}_description",
            ]
            start = ordered.index(row[0])
            self.assertEqual(ordered[start:start + 3], row)


class NodeExecutionTests(unittest.TestCase):
    def test_exact_request_guide_session_routing_and_interruption_handoff(self):
        with _fake_client() as (instances, interruption_checks):
            output = _generate(
                target_description="a precise creative brief",
                dialogue="Mind the shutter.",
                lyrics="Hold the line.",
                visible_text="BAY 4\n  DÉTOUR  ",
                constraint_notes="the camera remains locked",
                cut_times="0.0, 3.0",
                hermes_base_url="http://localhost:8642/",
                timeout_seconds=321,
                poll_interval_seconds=0.25,
                hermes_provider="   ",
                hermes_model=" route-model ",
            )
            expected_interruption_check = (
                node_mod.comfy.model_management
                .throw_exception_if_processing_interrupted
            )

        self.assertEqual(interruption_checks, [True])
        self.assertEqual(len(instances), 1)
        client = instances[0]
        self.assertEqual(client.init_kwargs["base_url"],
                         "http://localhost:8642")
        self.assertEqual(client.init_kwargs["poll_interval_seconds"], 0.25)

        call = client.run_kwargs
        request = json.loads(call["input"])
        self.assertEqual(call["input"], serialize_request(request))
        self.assertEqual(request["schema_version"], REQUEST_SCHEMA_VERSION)
        self.assertEqual(request["h3_mode"], "base_T2VA")
        self.assertEqual(request["quality_mode"], "balanced")
        self.assertEqual(request["research_policy"], "when_uncertain")
        self.assertEqual(request["creative_brief"], "a precise creative brief")
        self.assertEqual(request["exact_literals"], {
            "dialogue": "Mind the shutter.",
            "lyrics": "",
            "visible_text": ["BAY 4", "  DÉTOUR  "],
        })
        length, snapped = snap_length(6.0)
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
            "constraints": ["the camera remains locked"],
            "cut_timestamps": [0.0, 3.0],
        })
        self.assertEqual(request["subjects"], [])
        self.assertEqual(request["assets"], [])
        self.assertEqual(
            request["local_h3_format_guide"],
            prompts_base.build_system_prompt("base_T2VA"),
        )
        self.assertEqual(call["instructions"], node_mod.STABLE_INSTRUCTIONS)
        self.assertEqual(call["session_id"],
                         f"comfyui:h3:{request['request_id']}")
        self.assertEqual(call["timeout_seconds"], 321.0)
        self.assertNotIn("provider", call)
        self.assertEqual(call["model"], "route-model")
        self.assertIs(
            call["interruption_check"],
            expected_interruption_check,
        )
        self.assertIn("Mind the shutter.", output[0])
        self.assertNotIn("Hold the line.", output[0])
        self.assertIn("  DÉTOUR  ", output[0])

    def test_blank_provider_and_model_are_not_forwarded(self):
        with _fake_client() as (instances, _checks):
            _generate(hermes_provider=" ", hermes_model="   ")
        call = instances[0].run_kwargs
        self.assertNotIn("provider", call)
        self.assertNotIn("model", call)

    def test_base_ignores_stale_music_literals_but_preserves_v1_warning(self):
        sentinel = "STALE_PRIVATE_LYRIC_DO_NOT_SEND"
        with _fake_client() as (instances, _checks):
            named = _named(_generate(
                h3_mode="base_T2VA",
                music_video=False,
                lyrics=sentinel,
                music_description=sentinel,
            ))
        request = json.loads(instances[0].run_kwargs["input"])
        self.assertEqual(request["exact_literals"]["lyrics"], "")
        self.assertNotIn(sentinel, instances[0].run_kwargs["input"])
        self.assertNotIn(sentinel, named["h3_prompt"])
        analysis = json.loads(named["analysis_json"])
        warnings = analysis["validation"]["warnings"]
        self.assertTrue(warnings)
        self.assertNotIn(sentinel, json.dumps(warnings))

    def test_base_url_matrix_fails_closed_before_staging_or_client(self):
        rejected = [
            None,
            "",
            " http://127.0.0.1:8642",
            "http://127.0.0.1:8642 ",
            "http://127.0.0.1:\t8642",
            "http://127.0.0.1:8642\r",
            "http://127.0.0.1:\x008642",
            "https://127.0.0.1:8642",
            "http://example.com:8642",
            "http://user:password@127.0.0.1:8642",
            "http://localhost:8642/?query=1",
            "http://localhost:8642?",
            "http://127.0.0.1:8642/#fragment",
            "http://127.0.0.1:8642#",
            "http://[::1]:8642/api",
            "http://127.0.0.1:8642/%2f",
            "http://127.0.0.1",
            "http://127.0.0.1:",
            "http://127.0.0.1:0",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:65536",
            "http://[::1%25lo]:8642",
        ]
        rejected.extend(
            f"http://[::1%25zone{chr(codepoint)}]:8642"
            for codepoint in range(0x7F, 0xA0)
        )
        for url in rejected:
            with self.subTest(url=url), _temp_staging() as temp, _fake_client() as (
                instances, _checks
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "loopback|credentials|query|fragment|root|valid|whitespace|http",
                ):
                    _generate(
                        h3_mode="base_I2VA",
                        subject_1_image=_image(0.1),
                        hermes_base_url=url,
                    )
                self.assertEqual(
                    instances, [], "invalid URLs must fail before a client exists"
                )
                self.assertFalse((temp / "h3_hermes").exists())

    def test_malformed_base_url_exception_context_is_sanitized(self):
        cases = (
            (
                "OPAQUE_USERINFO_SECRET_42",
                "http://user:OPAQUE_USERINFO_SECRET_42@127.0.0.1：8642",
            ),
            (
                "opaque_host_secret_7",
                "http://opaque_host_secret_7:8642",
            ),
        )
        for secret, malformed in cases:
            with self.subTest(secret=secret):
                try:
                    node_mod._loopback_base_url(malformed)
                    self.fail("expected malformed-authority rejection")
                except RuntimeError as exc:
                    rendered = "".join(traceback.format_exception(exc))
                    self.assertNotIn(secret, str(exc))
                    self.assertNotIn(secret, rendered)

    def test_accepted_base_urls_are_forwarded_in_canonical_form(self):
        accepted = (
            ("HTTP://LOCALHOST:08642/", "http://localhost:8642"),
            ("http://127.0.0.1:08642", "http://127.0.0.1:8642"),
            (
                "HTTP://[0:0:0:0:0:0:0:1]:08642/",
                "http://[::1]:8642",
            ),
        )
        for supplied, canonical in accepted:
            with self.subTest(supplied=supplied), _fake_client() as (
                instances, _checks
            ):
                named = _named(_generate(hermes_base_url=supplied))
            self.assertEqual(instances[0].init_kwargs["base_url"], canonical)
            self.assertEqual(
                json.loads(named["analysis_json"])["hermes"]["base_url"],
                canonical,
            )

    def test_route_values_fail_before_anchors_staging_or_client(self):
        class StringSubclass(str):
            pass

        rejected = (
            None,
            7,
            StringSubclass("provider"),
            "x" * 257,
            "route\x00name",
            "route\nname",
            "route\x7fname",
            "route\x85name",
            "\ud800",
        )
        for field in ("hermes_provider", "hermes_model"):
            for value in rejected:
                with self.subTest(field=field, value=repr(value)), _temp_staging() as (
                    temp
                ), _fake_client() as (instances, _checks):
                    with self.assertRaisesRegex(
                        RuntimeError, "route|text|large|control|UTF-8"
                    ):
                        _generate(
                            h3_mode="base_I2VA",
                            subject_1_image=_image(0.1),
                            **{field: value},
                        )
                    self.assertEqual(instances, [])
                    self.assertFalse((temp / "h3_hermes").exists())

    def test_route_values_validate_before_trim_and_forward_canonically(self):
        provider = " " + ("🦆" * 254) + " "
        self.assertEqual(len(provider), node_mod.ROUTE_MAX_CHARS)
        self.assertLessEqual(len(provider.encode("utf-8")),
                             node_mod.ROUTE_MAX_UTF8_BYTES)
        with _fake_client() as (instances, _checks):
            _generate(
                hermes_provider=provider,
                hermes_model=" route-model ",
            )
        call = instances[0].run_kwargs
        self.assertEqual(call["provider"], "🦆" * 254)
        self.assertEqual(call["model"], "route-model")

    def test_boolean_controls_require_exact_bool_before_staging_or_client(self):
        rejected = (None, 0, 1, "false", "x" * 100_000, object())
        for field in (
            "fl2va_normalize_picture_tags",
            "snap_duration_to_h3_grid",
            "strict_duration",
        ):
            for value in rejected:
                with self.subTest(
                    field=field,
                    value_type=type(value).__name__,
                    value_chars=(len(value) if isinstance(value, str) else None),
                ), _temp_staging() as temp, _fake_client() as (instances, _checks):
                    with self.assertRaisesRegex(RuntimeError, "boolean"):
                        _generate(
                            h3_mode="base_I2VA",
                            subject_1_image=_image(0.1),
                            **{field: value},
                        )
                    self.assertEqual(instances, [])
                    self.assertFalse((temp / "h3_hermes").exists())

    def test_base_quality_and_research_enums_fail_before_anchor_staging_or_client(self):
        cases = (
            (
                "quality_mode",
                "unsafe-quality-value",
                "quality_mode must be fast, balanced, or hero.",
            ),
            (
                "research_policy",
                "unsafe-research-value",
                "research_policy must be never, when_uncertain, or always.",
            ),
        )
        original_resolve = node_mod.H3HermesPromptDirector._resolve_anchors
        original_stage = node_mod.stage_assets
        anchor_calls = []
        stage_calls = []

        def recording_resolve(self, *args, **kwargs):
            anchor_calls.append((args, kwargs))
            return original_resolve(self, *args, **kwargs)

        def recording_stage(*args, **kwargs):
            stage_calls.append((args, kwargs))
            return original_stage(*args, **kwargs)

        node_mod.H3HermesPromptDirector._resolve_anchors = recording_resolve
        node_mod.stage_assets = recording_stage
        try:
            for field, rejected, expected_error in cases:
                anchor_calls.clear()
                stage_calls.clear()
                with self.subTest(field=field), _temp_staging() as temp, _fake_client() as (
                    instances, _checks
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        _generate(
                            h3_mode="base_I2VA",
                            subject_1_image=_image(0.1),
                            **{field: rejected},
                        )
                    self.assertEqual(str(caught.exception), expected_error)
                    self.assertNotIn(rejected, str(caught.exception))
                    self.assertEqual(anchor_calls, [])
                    self.assertEqual(stage_calls, [])
                    self.assertEqual(instances, [])
                    self.assertFalse((temp / "h3_hermes").exists())
        finally:
            node_mod.H3HermesPromptDirector._resolve_anchors = original_resolve
            node_mod.stage_assets = original_stage

    def test_all_enum_controls_require_exact_builtin_values_before_io(self):
        class StringSubclass(str):
            pass

        cases = (
            ("h3_mode", "forged_mode", "base_I2VA"),
            ("quality_mode", StringSubclass("fast"), "base_I2VA"),
            ("research_policy", StringSubclass("never"), "base_I2VA"),
            ("video_role", "forged_role", "ref"),
            ("audio_role", "forged_role", "ref"),
            ("base_picture_role", "forged_role", "base_I2VA"),
            ("music_source", "forged_source", "ref"),
            ("cleanup_policy", StringSubclass("retain"), "base_I2VA"),
        )
        for field, value, mode in cases:
            with self.subTest(field=field), _temp_staging() as temp, _fake_client() as (
                instances, _checks
            ):
                kwargs = {
                    "h3_mode": mode,
                    "subject_1_image": _image(0.1),
                    field: value,
                }
                if mode == "ref":
                    kwargs.update(
                        subjects="person Courier -- charcoal jacket",
                        frames=_frames(),
                        duration_override=1.0,
                    )
                with self.assertRaisesRegex(RuntimeError, field):
                    _generate(**kwargs)
                self.assertEqual(instances, [])
                self.assertFalse((temp / "h3_hermes").exists())

    def test_ref_missing_subject_or_media_fails_before_staging_or_client(self):
        cases = (
            {
                "subjects": "",
                "frames": _frames(),
            },
            {
                "subjects": "person Courier -- charcoal utility jacket",
                "frames": None,
                "video": None,
            },
        )
        for overrides in cases:
            with self.subTest(overrides=sorted(overrides)), _temp_staging() as temp, _fake_client() as (
                instances, _checks
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "No subjects|Connect either|VIDEO|IMAGE batch"
                ):
                    _generate(h3_mode="ref", **overrides)
                self.assertEqual(instances, [])
                self.assertFalse((temp / "h3_hermes").exists())

    def test_video_frames_and_audio_fail_before_staging_or_client(self):
        cases = {
            "video": object(),
            "frames": _image(0.2),
            "audio": {"waveform": torch.zeros((1, 1, 8)), "sample_rate": 8},
        }
        for field, value in cases.items():
            with self.subTest(field=field), _temp_staging() as temp, _fake_client() as (
                instances, _checks
            ):
                with self.assertRaisesRegex(RuntimeError, "video|frames|audio|unsupported"):
                    _generate(**{field: value})
                self.assertEqual(instances, [])
                self.assertFalse((temp / "h3_hermes").exists())

    def test_picture_slot_gaps_fail_before_staging_or_client(self):
        for mode in ("base_T2VA", "base_I2VA", "base_FL2VA", "base_L2VA"):
            with self.subTest(mode=mode), _temp_staging() as temp, _fake_client() as (
                instances, _checks
            ):
                with self.assertRaisesRegex(RuntimeError, "gap|physical|slot 2"):
                    _generate(
                        h3_mode=mode,
                        subject_1_image=_image(0.1),
                        subject_3_image=_image(0.3),
                    )
                self.assertEqual(instances, [])
                self.assertFalse((temp / "h3_hermes").exists())

    def test_invalid_h3_fails_closed_after_strict_json_parsing(self):
        def broken(request):
            return _response(request, prompt="ordinary prose, not H3")

        with _fake_client(response_builder=broken):
            with self.assertRaisesRegex(
                RuntimeError, "local H3 validation failed.*R[15]"
            ):
                _generate()

    def test_dialogue_and_lyrics_must_preserve_case_and_whitespace_exactly(self):
        cases = (
            ("dialogue", "Mind  the shutter.", "mind  the shutter."),
            ("dialogue", "Mind  the shutter.", "Mind the shutter."),
            ("lyrics", "Hold  the line.", "hold  the line."),
            ("lyrics", "Hold  the line.", "Hold the line."),
        )
        for field, literal, changed in cases:
            def altered(request, *, exact=literal, replacement=changed):
                prompt = _prompt_for_request(request).replace(exact, replacement)
                return _response(request, prompt=prompt)

            with self.subTest(field=field, changed=changed), _fake_client(
                response_builder=altered
            ):
                kwargs: dict[str, object] = {field: literal}
                if field == "lyrics":
                    kwargs.update(
                        h3_mode="ref",
                        subjects="person Courier -- charcoal jacket",
                        frames=_frames(),
                        music_video=True,
                        duration_override=1.0,
                    )
                with self.assertRaisesRegex(
                    RuntimeError, "local H3 validation failed.*R3 VERBATIM"
                ) as caught:
                    _generate(**kwargs)
                self.assertNotIn(literal, str(caught.exception))

    def test_dialogue_decoy_outside_d_does_not_satisfy_exact_check(self):
        literal = "Mind  the shutter."

        def decoyed(request):
            prompt = _prompt_for_request(request).replace(
                f"<d>[English] {literal}</d>",
                "<d>[English] mind the shutter.</d>",
            )
            prompt = prompt.replace(
                "overall_soundscape: ",
                f"overall_soundscape: {literal} ",
            )
            return _response(request, prompt=prompt)

        with _fake_client(response_builder=decoyed):
            with self.assertRaisesRegex(
                RuntimeError, "local H3 validation failed.*R3 VERBATIM"
            ):
                _generate(dialogue=literal)

    def test_lyrics_decoy_outside_d_does_not_satisfy_exact_check(self):
        literal = "Hold  the line."

        def decoyed(request):
            prompt = _prompt_for_request(request).replace(
                f"<d>[English] {literal}</d>",
                "<d>[English] Hold the line.</d>",
            )
            prompt = prompt.replace(
                "overall_soundscape: ",
                f"overall_soundscape: {literal} ",
            )
            return _response(request, prompt=prompt)

        with _fake_client(response_builder=decoyed):
            with self.assertRaisesRegex(
                RuntimeError, "local H3 validation failed.*R3 VERBATIM"
            ):
                _generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    music_video=True,
                    lyrics=literal,
                    duration_override=1.0,
                )

    def test_spoken_block_payload_must_equal_literal_without_extra_words(self):
        cases = (
            ("dialogue", "Mind the shutter.", " Mind the shutter. Then run."),
            ("lyrics", "Hold the line.", " Hold the line. forever"),
        )
        for field, literal, extra in cases:
            def appended(request, literal=literal, extra=extra):
                prompt = _prompt_for_request(request).replace(
                    f" {literal}</d>",
                    f"{extra}</d>",
                )
                return _response(request, prompt=prompt)

            with self.subTest(field=field), _fake_client(
                response_builder=appended
            ):
                kwargs: dict[str, object] = {field: literal}
                if field == "lyrics":
                    kwargs.update(
                        h3_mode="ref",
                        subjects="person Courier -- charcoal jacket",
                        frames=_frames(),
                        music_video=True,
                        duration_override=1.0,
                    )
                with self.assertRaisesRegex(
                    RuntimeError, "local H3 validation failed.*R3 VERBATIM"
                ) as caught:
                    _generate(**kwargs)
            self.assertNotIn(literal, str(caught.exception))

    def test_dialogue_and_lyrics_require_distinct_ordered_role_blocks(self):
        dialogue = "Mind the shutter."
        lyrics = "Hold the line."

        def swapped(request):
            prompt = _prompt_for_request(request)
            first = f"<d>[English] {dialogue}</d>"
            second = f"<d>[English] {lyrics}</d>"
            prompt = prompt.replace(first, "__FIRST_SPOKEN_BLOCK__")
            prompt = prompt.replace(second, first)
            prompt = prompt.replace("__FIRST_SPOKEN_BLOCK__", second)
            return _response(request, prompt=prompt)

        with _fake_client(response_builder=swapped):
            with self.assertRaisesRegex(
                RuntimeError, "local H3 validation failed.*R3 VERBATIM"
            ) as caught:
                _generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    music_video=True,
                    dialogue=dialogue,
                    lyrics=lyrics,
                    duration_override=1.0,
                )
        self.assertNotIn(dialogue, str(caught.exception))
        self.assertNotIn(lyrics, str(caught.exception))

        same = "Keep moving."

        def shared_block(request):
            prompt = _prompt_for_request(request)
            second_sentence = (
                f" <Subject 1> (S2) sings, <d>[English] {same}</d>"
            )
            return _response(request, prompt=prompt.replace(second_sentence, ""))

        with _fake_client(response_builder=shared_block):
            with self.assertRaisesRegex(
                RuntimeError, "local H3 validation failed.*R3 VERBATIM"
            ):
                _generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    music_video=True,
                    dialogue=same,
                    lyrics=same,
                    duration_override=1.0,
                )

    def test_visible_text_decoy_in_soundscape_or_music_does_not_satisfy_literal(self):
        literal = "BAY 4"

        for section in ("overall_soundscape: ", "non_diegetic_music: "):
            def decoyed(request, section=section):
                prompt = _prompt_for_request(request).replace(
                    f'"{literal}"',
                    '"BAY FOUR"',
                )
                prompt = prompt.replace(section, f"{section}{literal} ")
                return _response(request, prompt=prompt)

            with self.subTest(section=section), _fake_client(
                response_builder=decoyed
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "local H3 validation failed.*R3 VERBATIM"
                ) as caught:
                    _generate(visible_text=literal)
            self.assertNotIn(literal, str(caught.exception))

    def test_unclosed_d_markup_fails_closed(self):
        def unclosed(request):
            prompt = _prompt_for_request(request).replace(
                "\n\noverall_soundscape:",
                " A stray <d>unfinished\n\noverall_soundscape:",
            )
            return _response(request, prompt=prompt)

        with _fake_client(response_builder=unclosed):
            with self.assertRaisesRegex(
                RuntimeError, "local H3 validation failed.*R3 VERBATIM"
            ):
                _generate(dialogue="Mind the shutter.")

    def test_structurally_valid_prompt_over_hard_character_cap_fails_closed(self):
        oversized_detail = " ".join(
            "The courier crosses the wet loading bay under a blue work light."
            for _ in range(700)
        )
        oversized = SIMPLE.replace(
            "a medium-wide shot frames a courier ducking under a roller shutter on a "
            "wet loading bay.",
            oversized_detail,
        )
        self.assertGreater(len(oversized), node_mod.MAX_PROMPT_CHARS)

        with _fake_client(
            response_builder=lambda request: _response(request, prompt=oversized)
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                f"local H3 validation failed.*{node_mod.MAX_PROMPT_CHARS} "
                "characters",
            ):
                _generate()

    def test_default_duration_matches_v1_warning_and_exposes_diagnostics(self):
        with _fake_client() as (instances, _checks):
            named = _named(_generate(duration_override=0.0))

        length, physical_duration = snap_length(node_mod.DEFAULT_BASE_DURATION)
        request = json.loads(instances[0].run_kwargs["input"])
        self.assertEqual(request["generation"], {
            "requested_duration_seconds": node_mod.DEFAULT_BASE_DURATION,
            "snapped_duration_seconds": physical_duration,
            "fps": 24.0,
            "width": 768,
            "height": 768,
            "length": length,
        })
        self.assertEqual(named["duration_seconds"], round(physical_duration, 3))
        self.assertEqual(named["length"], length)

        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["duration_source"], "default")
        self.assertEqual(analysis["requested_duration_seconds"], 5.0)
        self.assertEqual(
            analysis["snapped_duration_seconds"], round(physical_duration, 3)
        )
        self.assertTrue(analysis["snap_duration_to_h3_grid"])
        self.assertEqual(analysis["h3_length_frames"], length)
        self.assertIn(
            "no video, no frames and no duration_override, so the target "
            "duration defaulted to 5.00s. That number is written into the "
            "instruction line and the shot times; set duration_override to "
            "the length you actually want.",
            analysis["validation"]["warnings"],
        )

    def test_duration_range_warns_or_fails_strictly_before_io(self):
        cases = (
            ("base_T2VA", 1.0, {}),
            ("base_T2VA", 16.0, {}),
            (
                "ref",
                1.0,
                {
                    "subjects": "person Courier -- charcoal jacket",
                    "frames": _frames(),
                    "fps": 6.0,
                },
            ),
            (
                "ref",
                16.0,
                {
                    "subjects": "person Courier -- charcoal jacket",
                    "frames": _frames(),
                    "fps": 6.0,
                },
            ),
        )
        for mode, duration, extra in cases:
            with self.subTest(mode=mode, duration=duration), _fake_client():
                named = _named(_generate(
                    h3_mode=mode,
                    duration_override=duration,
                    strict_duration=False,
                    **extra,
                ))
            analysis = json.loads(named["analysis_json"])
            self.assertFalse(analysis["duration_supported"])
            self.assertFalse(analysis["strict_duration"])
            self.assertTrue(any(
                "official 4 to 15 second range" in warning
                for warning in analysis["validation"]["warnings"]
            ))

            with self.subTest(mode=mode, duration=duration, strict=True), \
                    _temp_staging() as temp, _fake_client() as (instances, _checks):
                with self.assertRaisesRegex(
                    RuntimeError, "official 4 to 15 second range"
                ):
                    _generate(
                        h3_mode=mode,
                        duration_override=duration,
                        strict_duration=True,
                        **extra,
                    )
                self.assertEqual(instances, [])
                self.assertFalse((temp / "h3_hermes").exists())

    def test_snap_on_records_requested_and_physical_duration(self):
        with _fake_client() as (instances, _checks):
            named = _named(_generate(duration_override=2.0))

        length, physical_duration = snap_length(2.0)
        request = json.loads(instances[0].run_kwargs["input"])
        self.assertEqual(request["generation"]["requested_duration_seconds"], 2.0)
        self.assertEqual(
            request["generation"]["snapped_duration_seconds"], physical_duration
        )
        self.assertEqual(request["generation"]["length"], length)
        self.assertEqual(named["duration_seconds"], round(physical_duration, 3))
        self.assertEqual(named["length"], length)

        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["duration_source"], "override")
        self.assertEqual(analysis["requested_duration_seconds"], 2.0)
        self.assertEqual(
            analysis["snapped_duration_seconds"], round(physical_duration, 3)
        )
        self.assertTrue(analysis["snap_duration_to_h3_grid"])
        self.assertEqual(analysis["h3_length_frames"], length)

    def test_snap_off_keeps_prompt_output_requested_but_reports_physical_grid(self):
        with _fake_client() as (instances, _checks):
            named = _named(_generate(
                duration_override=2.0,
                snap_duration_to_h3_grid=False,
            ))

        length, physical_duration = snap_length(2.0)
        request = json.loads(instances[0].run_kwargs["input"])
        self.assertEqual(request["generation"]["requested_duration_seconds"], 2.0)
        self.assertEqual(
            request["generation"]["snapped_duration_seconds"], physical_duration
        )
        self.assertEqual(request["generation"]["length"], length)
        self.assertEqual(named["duration_seconds"], 2.0)
        self.assertEqual(named["length"], length)

        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["duration_source"], "override")
        self.assertEqual(analysis["requested_duration_seconds"], 2.0)
        self.assertEqual(
            analysis["snapped_duration_seconds"], round(physical_duration, 3)
        )
        self.assertFalse(analysis["snap_duration_to_h3_grid"])
        self.assertEqual(analysis["h3_length_frames"], length)
        self.assertTrue(any(
            "not the 2.000s this prompt states" in warning
            for warning in analysis["validation"]["warnings"]
        ))

    def test_existing_assembler_fixes_are_used_and_reported(self):
        ref_shaped_header = SIMPLE.replace(
            "integrated_multimodal_description: ",
            "integrated_multimodal_description:\n",
        )
        with _fake_client(
            response_builder=lambda request: _response(
                request, prompt=ref_shaped_header
            )
        ):
            output = _generate()
        named = _named(output)
        self.assertTrue(named["h3_prompt"].startswith(
            "integrated_multimodal_description: [Shot 1]"
        ))
        validation = json.loads(named["analysis_json"])["validation"]
        self.assertEqual(validation["hard_errors"], [])
        self.assertTrue(validation["applied_fixes"])
        self.assertGreater(validation["char_count"], 0)

    def test_analysis_handoff_is_complete_bounded_and_secret_free(self):
        opaque = "sk-proj-Q7vM2pL9cR4tN8xK6jH3wZ5y"
        private_path = "/home/alice/.config/hermes/private-key.json"
        credential_url = (
            "https://user:pass@example.invalid/api?api_key=other-secret"
        )
        current_key = "current-env-credential-must-also-stay-private"
        run_id = "run_analysis_probe_7"
        markers = (opaque, private_path, credential_url, current_key)
        alternate_prompt = credential_url + ("x" * 9000)

        def rich_response(request):
            selected_prompt = _prompt_for_request(request)
            candidates = [
                {
                    "candidate_id": opaque,
                    "policy": credential_url,
                    "prompt": selected_prompt,
                    "score_vector": {
                        "required_intent_coverage": 1.0,
                        "contradictions": 0,
                        opaque: 99,
                    },
                    "critic_findings": [opaque, private_path, credential_url],
                },
                {
                    "candidate_id": private_path,
                    "policy": opaque,
                    "prompt": alternate_prompt,
                    "score_vector": {
                        "economy": 0.1,
                        credential_url: private_path,
                    },
                    "critic_findings": [current_key],
                },
            ]
            value = _response(
                request,
                candidates=candidates,
                evidence={
                    "observations": [opaque, private_path, credential_url],
                    "assumptions": [current_key],
                    "uninspected_assets": [
                        opaque, private_path, credential_url, current_key,
                    ],
                },
                quality_report={
                    "hard_errors": [],
                    "warnings": [opaque, private_path, credential_url, current_key],
                    "unresolved_ambiguities": [credential_url],
                    "reported_tools": [
                        "web_search", "web_extract", "vision_analyze", "terminal",
                        opaque, private_path, credential_url,
                    ],
                    "reported_sources": [opaque, private_path, credential_url],
                },
            )
            value["intent_ir"] = {
                "required_atoms": [opaque, request["creative_brief"]],
                "preferred_atoms": [private_path],
                "optional_atoms": [credential_url],
                "reference_jobs": [current_key],
            }
            value["repairs"] = [{
                "warning": opaque,
                "path": private_path,
                "source": credential_url,
            }]
            return value

        old_key = os.environ.get("HERMES_AGENT_API_KEY")
        os.environ["HERMES_AGENT_API_KEY"] = current_key
        try:
            with _fake_client(
                response_builder=rich_response,
                run_id=run_id,
                usage={
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                    "api_key": current_key,
                    "raw_response": "RAW_USAGE_MARKER",
                },
            ):
                named = _named(_generate(cleanup_policy="retain_24h"))
        finally:
            if old_key is None:
                os.environ.pop("HERMES_AGENT_API_KEY", None)
            else:
                os.environ["HERMES_AGENT_API_KEY"] = old_key

        wire = named["analysis_json"]
        for marker in markers:
            self.assertNotIn(marker, wire)
        self.assertNotIn("RAW_USAGE_MARKER", wire)
        self.assertNotIn("Authorization", wire)
        self.assertLess(len(wire.encode("utf-8")), 20_000)
        analysis = json.loads(wire)
        self.assertEqual((analysis["engine_requested"],
                          analysis["engine_used"],
                          analysis["fallback_used"]),
                         ("hermes_agent", "hermes_agent", False))
        self.assertEqual(analysis["hermes"]["base_url"],
                         "http://127.0.0.1:8642")
        self.assertEqual(analysis["hermes"]["run_id"], run_id)
        self.assertEqual(
            analysis["hermes"]["run_id_sha256"],
            hashlib.sha256(run_id.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(analysis["hermes"]["run_id_char_count"], len(run_id))
        self.assertEqual(
            analysis["hermes"]["run_id_utf8_byte_count"],
            len(run_id.encode("utf-8")),
        )
        self.assertEqual(analysis["hermes"]["status"], "completed")
        self.assertEqual(analysis["hermes"]["quality_mode"], "balanced")
        self.assertEqual(analysis["hermes"]["research_policy"],
                         "when_uncertain")
        self.assertEqual(analysis["hermes"]["usage"], {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        })
        self.assertEqual(
            analysis["hermes"]["model_reported_tools"],
            ["web_search", "web_extract", "vision_analyze"],
        )
        self.assertEqual(analysis["hermes"]["verified_tool_events"], [])
        self.assertEqual(analysis["request"]["schema_version"],
                         REQUEST_SCHEMA_VERSION)
        self.assertEqual(analysis["request"]["asset_count"], 0)
        self.assertEqual(analysis["request"]["staged_bytes"], 0)
        self.assertEqual(
            analysis["selection"]["selected_candidate_id_sha256"],
            hashlib.sha256(opaque.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            analysis["selection"]["selected_candidate_id_char_count"],
            len(opaque),
        )
        self.assertEqual(analysis["selection"]["candidate_count"], 2)
        self.assertEqual(analysis["selection"]["score_vector"],
                         {"required_intent_coverage": 1.0,
                          "contradictions": 0})
        alternate = analysis["selection"]["candidates"][1]
        self.assertEqual(
            alternate["prompt_sha256"],
            hashlib.sha256(alternate_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(alternate["prompt_char_count"], len(alternate_prompt))
        self.assertNotIn("prompt", alternate)
        self.assertNotIn("critic_findings", alternate)
        for key in ("evidence", "intent_ir", "repairs", "quality_report"):
            self.assertIn(key, analysis)
        self.assertEqual(analysis["evidence"], {"uninspected_assets": []})
        self.assertEqual(analysis["intent_ir"], {
            "required_atom_count": 2,
            "preferred_atom_count": 1,
            "optional_atom_count": 1,
            "reference_job_count": 1,
        })
        self.assertEqual(analysis["repairs"], {"count": 1})
        self.assertEqual(analysis["quality_report"], {
            "hard_error_count": 0,
            "warning_count": 4,
            "unresolved_ambiguity_count": 1,
            "reported_source_count": 3,
        })
        self.assertEqual(analysis["uninspected_assets"], [])
        self.assertEqual(analysis["cleanup"], {
            "policy": "retain_24h",
            "retained_path": None,
        })
        self.assertEqual(analysis["validation"]["hard_errors"], [])
        self.assertIn("applied_fixes", analysis["validation"])
        self.assertIn("warnings", analysis["validation"])
        self.assertGreater(analysis["validation"]["char_count"], 0)

    def test_remote_run_id_is_bounded_safe_and_never_echoes_invalid_value(self):
        invalid_ids = (
            "«redacted:sk-…»",
            "/home/alice/private-run",
            "run id with spaces",
            "x" * 129,
        )
        for invalid in invalid_ids:
            with self.subTest(kind=type(invalid).__name__, chars=len(invalid)), \
                    _fake_client(run_id=invalid):
                with self.assertRaisesRegex(
                    RuntimeError, "invalid run ID metadata"
                ) as caught:
                    _generate()
            self.assertNotIn(invalid, str(caught.exception))

    def test_remote_run_status_uses_strict_success_allowlist_without_echo(self):
        opaque_status = "sk-proj-status-must-not-appear"
        with _fake_client(status=opaque_status):
            with self.assertRaisesRegex(
                RuntimeError, "invalid status metadata"
            ) as caught:
                _generate()
        self.assertNotIn(opaque_status, str(caught.exception))

    def test_output_order_count_types_and_text_only_pass_throughs(self):
        with _fake_client():
            output = _generate(duration_override=8.0)
        cls = node_mod.H3HermesPromptDirector
        self.assertEqual(len(output), len(cls.RETURN_TYPES))
        self.assertEqual(len(output), len(cls.RETURN_NAMES))
        named = _named(output)
        for slot in range(1, 7):
            self.assertIsNone(named[f"ref_image_{slot}"])
        self.assertIsNone(named["ref_video"])
        self.assertIsNone(named["ref_video_audio"])
        self.assertIsNone(named["ref_audio"])
        self.assertEqual((named["width"], named["height"]), (768, 768))
        self.assertEqual(named["length"], 192)
        self.assertEqual(named["label_map"], "nothing connected")
        self.assertEqual(named["fps"], 24)
        self.assertEqual(named["h3_checkpoint_hint"],
                         "MiniMax-H3-Base-FL2VA")


class RefVideoAudioTests(unittest.TestCase):
    def test_invalid_frames_and_fps_fail_before_staging_or_client(self):
        cases = (
            ("rank", {"frames": torch.zeros((12, 20, 3)), "fps": 8.0}),
            ("empty", {"frames": _frames(0), "fps": 8.0}),
            ("zero_fps", {"frames": _frames(), "fps": 0.0}),
            ("negative_fps", {"frames": _frames(), "fps": -1.0}),
            ("nan_fps", {"frames": _frames(), "fps": float("nan")}),
            ("infinite_fps", {"frames": _frames(), "fps": float("inf")}),
            ("not_video", {"video": object()}),
        )
        for name, media in cases:
            with self.subTest(name=name), _temp_staging() as temp, _fake_client() as (
                instances, _checks
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "non-empty IMAGE batch|fps|VIDEO object"
                ):
                    _generate(
                        h3_mode="ref",
                        subjects="person Courier -- charcoal utility jacket",
                        duration_override=0.0,
                        **media,
                    )
                self.assertEqual(instances, [])
                self.assertFalse((temp / "h3_hermes").exists())

    def test_video_wins_over_frames_and_uses_component_frames_and_real_rate(self):
        video_frames = _frames(6, height=18, width=30)
        ignored_frames = _frames(3, height=8, width=10)
        video = _Video(video_frames, 12.0)
        with _temp_staging(), _fake_client() as (instances, _checks):
            named = _named(_generate(
                h3_mode="ref",
                subjects="person Courier -- charcoal utility jacket",
                video=video,
                frames=ignored_frames,
                fps=3.0,
                duration_override=0.0,
            ))

        self.assertEqual(video.component_calls, 1)
        self.assertIs(named["ref_video"], video_frames)
        self.assertEqual(named["fps"], 12)
        request = json.loads(instances[0].run_kwargs["input"])
        self.assertEqual(request["generation"]["fps"], 12.0)
        self.assertEqual(request["generation"]["requested_duration_seconds"], 0.5)
        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["duration_source"], "video")
        self.assertTrue(any(
            "both video and frames connected; using video" in warning
            for warning in analysis["validation"]["warnings"]
        ))

    def test_subject_rows_assets_directives_and_optional_wav_keep_physical_order(self):
        picture_1 = _image(0.1)
        picture_2 = _image(0.2)
        frames = _frames()
        audio = _audio()
        captured = {}

        def inspect(request):
            captured["request"] = request
            captured["assets"] = request["assets"]
            for item in request["assets"]:
                path = Path(item["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(path.parent.name, request["request_id"])
                if item["h3_label"] == "<Audio 1>":
                    captured["audio_payload"] = path.read_bytes()
            return _response(request)

        with _temp_staging() as temp, _fake_client(response_builder=inspect):
            named = _named(_generate(
                h3_mode="ref",
                subjects="style -- restrained 16mm grain",
                frames=frames,
                fps=6.0,
                audio=audio,
                subject_1_kind="character",
                subject_1_name="Courier",
                subject_1_description="charcoal utility jacket",
                subject_2_kind="environment",
                subject_2_name="Loading bay",
                subject_2_description="wet concrete and corrugated steel",
                subject_1_image=picture_1,
                subject_2_image=picture_2,
                duration_override=1.0,
            ))

        request = captured["request"]
        assets = captured["assets"]
        self.assertEqual(
            [(subject["index"], subject["kind"], subject["sources"], subject["slot"])
             for subject in request["subjects"]],
            [
                (1, "person", ["Picture 1"], 1),
                (2, "scene", ["Picture 2"], 2),
                (3, "style", [], None),
            ],
        )
        evidence_assets = [
            item for item in assets
            if item["h3_label"].startswith("<Video 1 Keyframe ")
        ]
        self.assertGreaterEqual(len(evidence_assets), 1)
        self.assertEqual(
            [(item["h3_label"], item["intended_jobs"], item["prohibited_transfers"])
             for item in assets],
            [
                ("<Picture 1>", ["identity", "appearance"],
                 ["pose", "motion", "audio"]),
                ("<Picture 2>", ["identity", "appearance"],
                 ["pose", "motion", "audio"]),
                ("<Video 1>", ["pose", "motion", "camera", "timing"],
                 ["identity", "appearance", "audio"]),
                *[
                    (
                        f"<Video 1 Keyframe {index}>",
                        ["visual_evidence", "timestamp_context"],
                        ["sampler", "identity", "audio"],
                    )
                    for index in range(1, len(evidence_assets) + 1)
                ],
                ("<Audio 1>", ["audio", "timing"],
                 ["identity", "appearance", "pose", "motion"]),
            ],
        )
        audio_asset = next(item for item in assets if item["h3_label"] == "<Audio 1>")
        self.assertEqual(audio_asset["mime_type"], "audio/wav")
        payload = captured["audio_payload"]
        self.assertEqual((payload[:4], payload[8:12]), (b"RIFF", b"WAVE"))
        self.assertIs(named["ref_image_1"], picture_1)
        self.assertIs(named["ref_image_2"], picture_2)
        self.assertIs(named["ref_video"], frames)
        self.assertIs(named["ref_video_audio"], audio)
        self.assertIs(named["ref_audio"], audio)
        self.assertFalse((temp / "h3_hermes" / request["request_id"]).exists())

    def test_ref_outputs_are_constructed_once_before_success_cleanup(self):
        original_outputs = node_mod.H3HermesPromptDirector._outputs
        output_calls = []

        def fail_on_second_output(self, *args, **kwargs):
            output_calls.append((args, kwargs))
            if len(output_calls) > 1:
                raise AssertionError("outputs were constructed more than once")
            return original_outputs(self, *args, **kwargs)

        node_mod.H3HermesPromptDirector._outputs = fail_on_second_output
        try:
            with _temp_staging() as temp, _fake_client():
                named = _named(_generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    fps=6.0,
                    duration_override=1.0,
                    cleanup_policy="delete_on_success",
                ))
                self.assertTrue(named["h3_prompt"])
                self.assertEqual(len(output_calls), 1)
                self.assertEqual(list((temp / "h3_hermes").glob("*")), [])
                analysis = json.loads(named["analysis_json"])
                self.assertIsNone(analysis["cleanup"]["retained_path"])
        finally:
            node_mod.H3HermesPromptDirector._outputs = original_outputs

    def test_ref_picture_gap_fails_pre_io_and_all_six_outputs_stay_uncompacted(self):
        with _temp_staging() as temp, _fake_client() as (instances, _checks):
            with self.assertRaisesRegex(RuntimeError, "gap|physical|slot 2"):
                _generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    subject_1_image=_image(0.1),
                    subject_3_image=_image(0.3),
                )
            self.assertEqual(instances, [])
            self.assertFalse((temp / "h3_hermes").exists())

        images = [_image(slot / 10.0) for slot in range(1, 7)]
        with _temp_staging(), _fake_client():
            named = _named(_generate(
                h3_mode="ref",
                subjects="person Courier -- charcoal jacket",
                frames=_frames(),
                duration_override=1.0,
                **{
                    f"subject_{slot}_image": image
                    for slot, image in enumerate(images, start=1)
                },
            ))
        for slot, image in enumerate(images, start=1):
            self.assertIs(named[f"ref_image_{slot}"], image)

    def test_tool_capability_and_model_report_never_inflate_verified_events(self):
        cases = (
            (False, ["video_analyze"], [], ["<Video 1>"]),
            (True, [], [], ["<Video 1>"]),
            (True, ["video_analyze"], ["<Video 1>"], ["<Video 1>"]),
        )
        for available, reported, expected_reported, expected_uninspected in cases:
            captured = {}

            def response(request, reported=reported):
                value = _response(request)
                value["quality_report"]["reported_tools"] = reported
                captured["evidence_assets"] = [
                    item for item in request["assets"]
                    if item["h3_label"].startswith("<Video 1 Keyframe ")
                ]
                captured["evidence_payloads"] = [
                    Path(item["path"]).read_bytes()
                    for item in captured["evidence_assets"]
                ]
                return value

            with self.subTest(available=available, reported=reported), _temp_staging(), _fake_client(
                response_builder=response,
                video_tool_available=available,
            ) as (instances, _checks):
                named = _named(_generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    duration_override=1.0,
                ))

            self.assertEqual(
                instances[0].tool_queries,
                [("video", "video_analyze", {})],
            )
            analysis = json.loads(named["analysis_json"])
            self.assertEqual(analysis["hermes"]["verified_tool_events"], [])
            if not available:
                inspection = analysis["inspection"]
                self.assertEqual(
                    {
                        "metadata_only_observations": inspection[
                            "metadata_only_observations"
                        ],
                        "staged_labels": inspection["staged_labels"],
                        "tool_capable_labels": inspection["tool_capable_labels"],
                        "model_reported_inspected_labels": inspection[
                            "model_reported_inspected_labels"
                        ],
                        "verified_inspected_labels": inspection[
                            "verified_inspected_labels"
                        ],
                    },
                    {
                        "metadata_only_observations": [
                            "fps", "duration", "dimensions", "cuts", "keyframes"
                        ],
                        "staged_labels": ["<Video 1>"],
                        "tool_capable_labels": [],
                        "model_reported_inspected_labels": [],
                        "verified_inspected_labels": [],
                    },
                )
                evidence_assets = captured["evidence_assets"]
                evidence_labels = [
                    item["h3_label"] for item in evidence_assets
                ]
                self.assertEqual(
                    evidence_labels,
                    inspection["staged_evidence_labels"],
                )
                for evidence_label in evidence_labels:
                    self.assertIn(evidence_label, analysis["uninspected_assets"])
                self.assertEqual(
                    len(evidence_assets),
                    len(analysis["selected_frame_indices"]),
                )
                self.assertEqual(
                    evidence_labels,
                    [
                        f"<Video 1 Keyframe {index}>"
                        for index in range(1, len(evidence_assets) + 1)
                    ],
                )
                for asset, payload in zip(
                    evidence_assets, captured["evidence_payloads"]
                ):
                    self.assertEqual(asset["kind"], "image")
                    self.assertEqual(asset["mime_type"], "image/jpeg")
                    self.assertEqual(
                        asset["intended_jobs"],
                        ["visual_evidence", "timestamp_context"],
                    )
                    self.assertEqual(
                        asset["prohibited_transfers"],
                        ["sampler", "identity", "audio"],
                    )
                    self.assertTrue(payload.startswith(b"\xff\xd8"))
                    self.assertTrue(payload.endswith(b"\xff\xd9"))
                self.assertEqual(
                    [
                        item["source_frame_index"]
                        for item in inspection["keyframe_evidence"]
                    ],
                    analysis["selected_frame_indices"],
                )
                self.assertNotIn("metadata_capable_labels", inspection)
            else:
                self.assertEqual(captured["evidence_assets"], [])
                self.assertEqual(
                    analysis["inspection"]["staged_evidence_labels"], []
                )
            self.assertEqual(
                analysis["inspection"]["model_reported_inspected_labels"],
                expected_reported,
            )
            self.assertEqual(
                analysis["inspection"]["verified_inspected_labels"], []
            )
            for label in expected_uninspected:
                self.assertIn(label, analysis["uninspected_assets"])

    def test_audio_is_explicitly_uninspected_even_when_model_claims_tools(self):
        def claimed(request):
            value = _response(request)
            value["quality_report"]["reported_tools"] = [
                "video_analyze", "vision_analyze"
            ]
            return value

        with _temp_staging(), _fake_client(
            response_builder=claimed,
            video_tool_available=True,
        ):
            named = _named(_generate(
                h3_mode="ref",
                subjects="person Courier -- charcoal jacket",
                frames=_frames(),
                audio=_audio(),
                audio_role="reference",
                duration_override=1.0,
            ))
        analysis = json.loads(named["analysis_json"])
        self.assertIn("<Audio 1>", analysis["uninspected_assets"])
        self.assertNotIn(
            "<Audio 1>",
            analysis["inspection"]["model_reported_inspected_labels"],
        )
        self.assertEqual(analysis["inspection"]["verified_inspected_labels"], [])

    def test_task_roles_music_and_absent_media_resets_reuse_v1_semantics(self):
        with _temp_staging(), _fake_client() as (instances, _checks):
            reset = _named(_generate(
                h3_mode="ref",
                subjects="person Courier -- charcoal jacket",
                frames=_frames(),
                fps=6.0,
                video_role="edit_source",
                audio_role="reference",
                duration_override=1.0,
            ))
        reset_request = json.loads(instances[0].run_kwargs["input"])
        self.assertEqual(reset_request["task"]["video_role"], "subject_source")
        self.assertEqual(reset_request["task"]["audio_role"], "none")
        self.assertEqual(reset_request["task"]["task_types"], ["reference generation"])
        reset_warnings = json.loads(reset["analysis_json"])["validation"]["warnings"]
        self.assertTrue(any("no video is connected" in warning for warning in reset_warnings))
        self.assertTrue(any("no audio is connected" in warning for warning in reset_warnings))

        video = _Video(_frames(), 6.0)
        with _temp_staging(), _fake_client() as (instances, _checks):
            music = _named(_generate(
                h3_mode="ref",
                subjects="person Singer @Video 1 -- silver jacket",
                video=video,
                audio=_audio(),
                video_role="edit_source",
                audio_role="reference",
                music_video=True,
                music_source="auto",
                lyrics="Hold  the line.",
                music_description="downtempo synthwave at 92 BPM",
                duration_override=1.0,
            ))
        request = json.loads(instances[0].run_kwargs["input"])
        self.assertEqual(request["task"]["task_types"], [
            "video editing", "audio reuse"
        ])
        self.assertEqual(request["task"]["audio_role"], "reuse")
        self.assertIn("MUSIC VIDEO MODE IS ON", request["creative_brief"])
        self.assertIn("downtempo synthwave at 92 BPM", request["creative_brief"])
        analysis = json.loads(music["analysis_json"])
        self.assertTrue(analysis["music_video"])
        self.assertTrue(analysis["music_is_reference"])
        self.assertEqual(analysis["task_type"], "video editing + audio reuse")

    def test_music_controls_are_effectively_blank_when_music_video_is_off(self):
        sentinel = "IGNORED-LYRIC-SENTINEL-Q7vM2pL9"
        captured = {}

        def omit_ignored_lyrics(request):
            captured["request"] = request
            prompt_request = dict(request)
            prompt_request["exact_literals"] = dict(
                request["exact_literals"], lyrics=""
            )
            return _response(request, prompt=_ref_prompt_for_request(prompt_request))

        try:
            with _temp_staging(), _fake_client(
                response_builder=omit_ignored_lyrics
            ):
                named = _named(_generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    fps=6.0,
                    music_video=False,
                    lyrics=sentinel,
                    duration_override=1.0,
                ))
        except RuntimeError as exc:
            self.assertNotIn(sentinel, str(exc))
            raise

        request = captured["request"]
        self.assertEqual(request["exact_literals"]["lyrics"], "")
        self.assertNotIn(sentinel, request["creative_brief"])
        self.assertNotIn(sentinel, named["h3_prompt"])
        self.assertNotIn(sentinel, named["analysis_json"])
        warnings = json.loads(named["analysis_json"])["validation"]["warnings"]
        self.assertTrue(any(
            "lyrics or music_description are set but music_video is off, so "
            "they were ignored" in warning
            for warning in warnings
        ))

    def test_unwired_explicit_audio_reuse_is_external_required_not_staged(self):
        captured = {}

        def inspect(request):
            captured["request"] = request
            return _response(request)

        with _temp_staging(), _fake_client(response_builder=inspect):
            named = _named(_generate(
                h3_mode="ref",
                subjects="person Singer -- silver jacket",
                frames=_frames(),
                fps=6.0,
                music_video=True,
                music_source="reuse_audio_1",
                lyrics="Hold the line.",
                duration_override=1.0,
            ))

        request = captured["request"]
        self.assertEqual(request["task"]["audio_role"], "reuse")
        self.assertIn("audio reuse", request["task"]["task_types"])
        self.assertNotIn("external_required_assets", request)
        request_labels = [item["h3_label"] for item in request["assets"]]
        self.assertEqual(request_labels[0], "<Video 1>")
        self.assertEqual(request_labels[-1], "<Audio 1>")
        evidence_labels = request_labels[1:-1]
        self.assertEqual(
            evidence_labels,
            [
                f"<Video 1 Keyframe {index}>"
                for index in range(1, len(evidence_labels) + 1)
            ],
        )
        external_authority = request["assets"][-1]
        self.assertEqual(
            external_authority,
            {
                "h3_label": "<Audio 1>",
                "authority": "downstream_required_external",
                "inspection_status": "uninspected",
            },
        )
        self.assertTrue(request["assets"][0]["path"].lower().endswith(".mp4"))
        self.assertTrue({
            "path", "sha256", "bytes", "mime_type", "staging_status",
        }.isdisjoint(external_authority))
        self.assertIsNone(named["ref_video_audio"])
        self.assertIsNone(named["ref_audio"])
        self.assertNotIn("<Audio 1>", named["label_map"])

        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["audio_role"], "reuse")
        self.assertIn("audio reuse", analysis["task_types"])
        self.assertEqual(analysis["request"]["labels"], ["<Video 1>"])
        self.assertEqual(analysis["request"]["asset_count"], len(request["assets"]))
        self.assertEqual(
            analysis["request"]["staged_asset_count"],
            1 + len(evidence_labels),
        )
        self.assertEqual(analysis["request"]["physical_staged_asset_count"], 1)
        self.assertEqual(
            analysis["request"]["evidence_staged_asset_count"],
            len(evidence_labels),
        )
        self.assertEqual(analysis["request"]["evidence_labels"], evidence_labels)
        self.assertEqual(
            analysis["request"]["external_required_labels"], ["<Audio 1>"]
        )
        self.assertEqual(
            analysis["inspection"]["staged_labels"], ["<Video 1>"]
        )
        self.assertEqual(
            analysis["inspection"]["external_required_uninspected_labels"],
            ["<Audio 1>"],
        )
        self.assertIn("<Audio 1>", analysis["uninspected_assets"])
        self.assertNotIn(
            "<Audio 1>", analysis["inspection"]["verified_inspected_labels"]
        )
        self.assertTrue(any(
            "The prompt declares <Audio 1> anyway, so the H3 sampler must be "
            "given that track" in warning
            for warning in analysis["validation"]["warnings"]
        ))

    def test_duration_cuts_keyframes_canvas_context_and_exact_cowboy_context(self):
        frames = _frames(10, height=18, width=30)
        captured_contexts = []
        original_process = node_mod.process

        def capture_process(prompt, context):
            captured_contexts.append(context)
            return original_process(prompt, context)

        node_mod.process = capture_process
        try:
            with _temp_staging(), _fake_client() as (instances, _checks):
                named = _named(_generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=frames,
                    fps=5.0,
                    cut_times="0.5, 1.0, 9.0",
                    dialogue="Mind  the shutter.",
                    duration_override=1.5,
                    max_frames_to_analyze=4,
                ))
        finally:
            node_mod.process = original_process

        request = json.loads(instances[0].run_kwargs["input"])
        length, snapped = snap_length(1.5)
        self.assertEqual(request["generation"], {
            "requested_duration_seconds": 1.5,
            "snapped_duration_seconds": snapped,
            "fps": 5.0,
            "width": canvas_for(30, 18)[0],
            "height": canvas_for(30, 18)[1],
            "length": length,
        })
        self.assertEqual(request["task"]["cut_timestamps"], [0.0, 0.5, 1.0])
        self.assertIn("MEASURED SHOT LIST", request["creative_brief"])
        self.assertIn("sampled frame timestamps", request["creative_brief"])
        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["cut_source"], "measured")
        self.assertEqual(analysis["cut_timestamps"], [0.0, 0.5, 1.0])
        self.assertLessEqual(len(analysis["selected_frame_indices"]), 4)
        self.assertEqual((named["width"], named["height"]), canvas_for(30, 18))
        self.assertEqual(named["length"], length)
        self.assertIs(named["ref_video"], frames)
        self.assertEqual(len(named), 18)

        context = captured_contexts[-1]
        self.assertEqual([subject.index for subject in context.subjects], [1])
        self.assertEqual(context.task_type, "reference generation")
        self.assertEqual(context.duration_seconds, snapped)
        self.assertEqual(context.known_shot_times, [0.0, 0.5, 1.0])
        self.assertFalse(context.is_editing)
        self.assertEqual(context.dialogue_text, "Mind  the shutter.")
        self.assertEqual(context.wired_pictures, 0)
        self.assertFalse(context.has_video)
        self.assertFalse(context.has_audio)

    def test_ref_post_run_tamper_and_cancellation_retain_failure_without_masking(self):
        def tamper(request):
            video_asset = next(
                item for item in request["assets"]
                if item["h3_label"] == "<Video 1>"
            )
            Path(video_asset["path"]).write_bytes(b"tampered after submission")
            return _response(request)

        with _temp_staging() as temp, _fake_client(response_builder=tamper):
            with self.assertRaisesRegex(RuntimeError, "integrity verification failed"):
                _generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    duration_override=1.0,
                )
            jobs = list((temp / "h3_hermes").iterdir())
            self.assertEqual(len(jobs), 1)
            self.assertTrue((jobs[0] / ".retention_until").is_file())

        class Cancelled(BaseException):
            pass

        cancellation = Cancelled("cancel sentinel")
        with _temp_staging() as temp, _fake_client(
            interruption_exception=cancellation
        ):
            with self.assertRaises(Cancelled) as caught:
                _generate(
                    h3_mode="ref",
                    subjects="person Courier -- charcoal jacket",
                    frames=_frames(),
                    duration_override=1.0,
                )
            self.assertIs(caught.exception, cancellation)
            jobs = list((temp / "h3_hermes").iterdir())
            self.assertEqual(len(jobs), 1)
            self.assertTrue((jobs[0] / ".retention_until").is_file())

    def test_ref_cleanup_error_never_replaces_original_transport_exception(self):
        sentinel = RuntimeError("transport sentinel")
        original_cleanup = node_mod.cleanup_assets

        def broken_cleanup(*args, **kwargs):
            raise RuntimeError("cleanup sentinel")

        node_mod.cleanup_assets = broken_cleanup
        try:
            with _temp_staging(), _fake_client(run_exception=sentinel):
                with self.assertRaises(RuntimeError) as caught:
                    _generate(
                        h3_mode="ref",
                        subjects="person Courier -- charcoal jacket",
                        frames=_frames(),
                        duration_override=1.0,
                    )
                self.assertIs(caught.exception, sentinel)
        finally:
            node_mod.cleanup_assets = original_cleanup


class ImageReferenceTests(unittest.TestCase):
    def test_i2va_fl2va_l2va_directives_physical_order_and_passthroughs(self):
        cases = {
            "base_I2VA": [
                ("<Picture 1>", ["first_frame", "appearance", "identity"], []),
            ],
            "base_FL2VA": [
                ("<Picture 1>", ["first_frame", "appearance", "identity"], []),
                ("<Picture 2>", ["last_frame", "continuity"], ["audio"]),
            ],
            "base_L2VA": [
                ("<Picture 1>", ["last_frame", "continuity"], ["audio"]),
            ],
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode), _temp_staging() as temp:
                images = [_image(slot / 10.0) for slot in range(1, 7)]
                observed = {}

                def inspect_request(request):
                    assets = request["assets"]
                    observed["request"] = request
                    observed["assets"] = assets
                    self.assertEqual(
                        request["local_h3_format_guide"],
                        prompts_base.build_system_prompt(mode),
                    )
                    self.assertEqual(
                        [(item["h3_label"], item["intended_jobs"],
                          item["prohibited_transfers"]) for item in assets],
                        expected,
                    )
                    self.assertEqual(
                        [item["asset_id"] for item in assets],
                        [f"picture_{slot:02d}" for slot in range(1, len(expected) + 1)],
                    )
                    for item in assets:
                        path = Path(item["path"])
                        self.assertTrue(path.is_file())
                        self.assertEqual(path.parent.name, request["request_id"])
                        self.assertTrue(path.is_relative_to(temp / "h3_hermes"))
                    manifest_path = Path(assets[0]["path"]).parent / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    self.assertEqual(manifest["request_id"], request["request_id"])
                    self.assertEqual(manifest["assets"], assets)
                    return _response(request)

                with _fake_client(response_builder=inspect_request):
                    named = _named(_generate(
                        h3_mode=mode,
                        subjects="person Courier -- a charcoal utility jacket",
                        subject_1_name="ignored row",
                        duration_override=6.0,
                        **{
                            f"subject_{slot}_image": image
                            for slot, image in enumerate(images, start=1)
                        },
                    ))

                request = observed["request"]
                assets = observed["assets"]
                self.assertFalse((temp / "h3_hermes" / request["request_id"]).exists())
                for slot, image in enumerate(images, start=1):
                    self.assertIs(named[f"ref_image_{slot}"], image)
                self.assertIsNone(named["ref_video"])
                self.assertIsNone(named["ref_video_audio"])
                self.assertIsNone(named["ref_audio"])
                self.assertEqual(
                    (named["width"], named["height"]), canvas_for(20, 12)
                )
                length, physical_duration = snap_length(6.0)
                self.assertEqual(named["length"], length)
                self.assertEqual(named["duration_seconds"], round(physical_duration, 3))
                self.assertEqual(
                    named["h3_checkpoint_hint"], "MiniMax-H3-Base-FL2VA"
                )
                expected_map = "\n".join(
                    f"<Picture {slot}> = subject_{slot}_image"
                    for slot in range(1, len(expected) + 1)
                )
                self.assertEqual(named["label_map"], expected_map)

                wire = named["analysis_json"]
                analysis = json.loads(wire)
                self.assertEqual(analysis["request"]["asset_count"], len(expected))
                self.assertEqual(
                    analysis["request"]["staged_bytes"],
                    sum(item["bytes"] for item in assets),
                )
                self.assertEqual(
                    analysis["request"]["labels"],
                    [item["h3_label"] for item in assets],
                )
                self.assertEqual(analysis["inspection"], {
                    "staging_complete": True,
                    "staged_labels": [item["h3_label"] for item in assets],
                    "model_reported_uninspected_assets": [],
                    "verified_inspected_labels": [],
                })
                warnings = " ".join(analysis["validation"]["warnings"])
                self.assertIn("was ignored", warnings)
                self.assertIn("base mode has no reference labels", warnings)
                for item in assets:
                    self.assertNotIn(item["path"], wire)
                    self.assertNotIn(item["sha256"], wire)
                self.assertNotIn("base64", wire.lower())

    def test_fl2va_normalization_and_role_warning_are_locally_authoritative(self):
        with _temp_staging(), _fake_client():
            normalized = _named(_generate(
                h3_mode="base_FL2VA",
                subject_1_image=_image(0.1),
                subject_2_image=_image(0.2),
                fl2va_normalize_picture_tags=True,
            ))
        prompt = normalized["h3_prompt"]
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Picture 2>", prompt)
        analysis = json.loads(normalized["analysis_json"])
        self.assertTrue(analysis["fl2va_normalize_picture_tags"])

        with _temp_staging(), _fake_client():
            conflicted = _named(_generate(
                h3_mode="base_I2VA",
                subject_1_image=_image(0.1),
                base_picture_role="last_frame",
            ))
        analysis = json.loads(conflicted["analysis_json"])
        self.assertEqual(analysis["base_picture_role"], "last_frame")
        self.assertTrue(any(
            "base_picture_role says 'last_frame'" in warning
            for warning in analysis["validation"]["warnings"]
        ))

    def test_local_diagnostics_never_copy_sensitive_user_values(self):
        current_key = "current-env-key-Q7vM2pL9cR4tN8xK6jH3"
        hazards = (
            current_key,
            "sk-proj-Q7vM2pL9cR4tN8xK6jH3wZ5y",
            "/home/alice/.config/private-key.json",
            "https://user:pass@example.invalid/run?token=other",
            "Bearer sk-proj-Q7vM2pL9cR4tN8xK6jH3wZ5y",
            "TOKEN=sk-proj-Q7vM2pL9cR4tN8xK6jH3wZ5y",
        )
        old_key = os.environ.get("HERMES_AGENT_API_KEY")
        os.environ["HERMES_AGENT_API_KEY"] = current_key
        try:
            for hazard in hazards:
                with self.subTest(hazard=hazard), _temp_staging() as temp, _fake_client() as (
                    instances, _checks
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "base_picture_role"
                    ) as caught:
                        _generate(
                            h3_mode="base_I2VA",
                            subject_1_image=_image(0.1),
                            base_picture_role=hazard,
                        )
                    self.assertNotIn(hazard, str(caught.exception))
                    self.assertLess(len(str(caught.exception).encode("utf-8")), 256)
                    self.assertEqual(instances, [])
                    self.assertFalse((temp / "h3_hermes").exists())
        finally:
            if old_key is None:
                os.environ.pop("HERMES_AGENT_API_KEY", None)
            else:
                os.environ["HERMES_AGENT_API_KEY"] = old_key

    def test_cleanup_policies_have_existing_semantics_and_safe_handoff(self):
        for policy in node_mod.CLEANUP_POLICIES:
            with self.subTest(policy=policy), _temp_staging() as temp:
                captured = {}

                def remember(request):
                    captured["request"] = request
                    return _response(request)

                with _fake_client(response_builder=remember):
                    named = _named(_generate(
                        h3_mode="base_I2VA",
                        subject_1_image=_image(0.1),
                        cleanup_policy=policy,
                    ))
                request_id = captured["request"]["request_id"]
                job_dir = temp / "h3_hermes" / request_id
                cleanup = json.loads(named["analysis_json"])["cleanup"]
                self.assertEqual(cleanup["policy"], policy)
                if policy == "delete_on_success":
                    self.assertFalse(job_dir.exists())
                    self.assertIsNone(cleanup["retained_path"])
                else:
                    self.assertTrue(job_dir.is_dir())
                    self.assertEqual(cleanup["retained_path"], str(job_dir))
                    marker = job_dir / ".retention_until"
                    self.assertEqual(marker.is_file(), policy == "retain_24h")

    def test_base_output_preflight_failure_uses_failure_retention(self):
        class OutputFailure(BaseException):
            pass

        sentinel = OutputFailure("output preflight sentinel")
        original_outputs = node_mod.H3HermesPromptDirector._outputs
        output_calls = []

        def broken_outputs(self, *args, **kwargs):
            output_calls.append((args, kwargs))
            raise sentinel

        node_mod.H3HermesPromptDirector._outputs = broken_outputs
        try:
            with _temp_staging() as temp, _fake_client():
                with self.assertRaises(OutputFailure) as caught:
                    _generate(
                        h3_mode="base_I2VA",
                        subject_1_image=_image(0.1),
                    )
                self.assertIs(caught.exception, sentinel)
                self.assertEqual(len(output_calls), 1)
                jobs = list((temp / "h3_hermes").iterdir())
                self.assertEqual(len(jobs), 1)
                self.assertTrue((jobs[0] / ".retention_until").is_file())
        finally:
            node_mod.H3HermesPromptDirector._outputs = original_outputs

    def test_base_output_exception_survives_failure_cleanup_baseexception(self):
        class OutputFailure(BaseException):
            pass

        class CleanupFailure(BaseException):
            pass

        primary = OutputFailure("primary output sentinel")
        cleanup_error = CleanupFailure("cleanup sentinel")
        original_outputs = node_mod.H3HermesPromptDirector._outputs
        original_cleanup = node_mod.cleanup_assets
        cleanup_calls = []

        def broken_outputs(self, *args, **kwargs):
            raise primary

        def broken_cleanup(*args, **kwargs):
            cleanup_calls.append(kwargs["success"])
            raise cleanup_error

        node_mod.H3HermesPromptDirector._outputs = broken_outputs
        node_mod.cleanup_assets = broken_cleanup
        try:
            with _temp_staging(), _fake_client():
                with self.assertRaises(OutputFailure) as caught:
                    _generate(
                        h3_mode="base_I2VA",
                        subject_1_image=_image(0.1),
                    )
                self.assertIs(caught.exception, primary)
                self.assertEqual(cleanup_calls, [False])
        finally:
            node_mod.H3HermesPromptDirector._outputs = original_outputs
            node_mod.cleanup_assets = original_cleanup

    def test_base_outputs_are_constructed_once_before_success_cleanup(self):
        original_outputs = node_mod.H3HermesPromptDirector._outputs
        original_cleanup = node_mod.cleanup_assets
        output_calls = []
        cleanup_calls = []

        def fail_on_second_output(self, *args, **kwargs):
            output_calls.append((args, kwargs))
            if len(output_calls) > 1:
                raise AssertionError("outputs were constructed more than once")
            return original_outputs(self, *args, **kwargs)

        def recording_cleanup(*args, **kwargs):
            cleanup_calls.append(kwargs["success"])
            return original_cleanup(*args, **kwargs)

        node_mod.H3HermesPromptDirector._outputs = fail_on_second_output
        node_mod.cleanup_assets = recording_cleanup
        try:
            with _temp_staging() as temp, _fake_client():
                named = _named(_generate(
                    h3_mode="base_I2VA",
                    subject_1_image=_image(0.1),
                ))
                self.assertTrue(named["h3_prompt"])
                self.assertEqual(len(output_calls), 1)
                self.assertEqual(cleanup_calls, [True])
                self.assertEqual(list((temp / "h3_hermes").glob("*")), [])
                analysis = json.loads(named["analysis_json"])
                self.assertIsNone(analysis["cleanup"]["retained_path"])
        finally:
            node_mod.H3HermesPromptDirector._outputs = original_outputs
            node_mod.cleanup_assets = original_cleanup

    def test_unknown_response_labels_fail_contract_and_trigger_failure_cleanup(self):
        def unknown(request):
            response = _response(request)
            response["evidence"]["uninspected_assets"] = ["<Picture 9>"]
            return response

        with _temp_staging() as temp, _fake_client(response_builder=unknown):
            with self.assertRaisesRegex(
                RuntimeError, "contract validation failed.*absent from"
            ):
                _generate(
                    h3_mode="base_I2VA",
                    subject_1_image=_image(0.1),
                )
            jobs = list((temp / "h3_hermes").iterdir())
            self.assertEqual(len(jobs), 1)
            self.assertTrue((jobs[0] / ".retention_until").is_file())

    def test_mutable_public_manifest_cannot_poison_submitted_request(self):
        original_stage = node_mod.stage_assets
        captured = {}

        def poisoned_stage(*args, **kwargs):
            bundle = original_stage(*args, **kwargs)
            bundle.manifest["assets"][0]["path"] = "/caller/poisoned.jpg"
            bundle.manifest["assets"][0]["sha256"] = "0" * 64
            return bundle

        def remember(request):
            captured["request"] = request
            return _response(request)

        node_mod.stage_assets = poisoned_stage
        try:
            with _temp_staging(), _fake_client(response_builder=remember):
                _generate(
                    h3_mode="base_I2VA",
                    subject_1_image=_image(0.1),
                )
        finally:
            node_mod.stage_assets = original_stage

        asset = captured["request"]["assets"][0]
        self.assertNotEqual(asset["path"], "/caller/poisoned.jpg")
        self.assertNotEqual(asset["sha256"], "0" * 64)

    def test_post_run_staged_file_tamper_fails_integrity_and_retains_failure(self):
        def tamper(request):
            path = Path(request["assets"][0]["path"])
            path.write_bytes(b"tampered after submission")
            return _response(request)

        with _temp_staging() as temp, _fake_client(response_builder=tamper):
            with self.assertRaisesRegex(
                RuntimeError, "integrity verification failed"
            ):
                _generate(
                    h3_mode="base_I2VA",
                    subject_1_image=_image(0.1),
                )
            jobs = list((temp / "h3_hermes").iterdir())
            self.assertEqual(len(jobs), 1)
            self.assertTrue((jobs[0] / ".retention_until").is_file())

    def test_transport_local_and_cancellation_failures_cleanup_without_masking(self):
        class Cancelled(BaseException):
            pass

        transport_error = RuntimeError("transport sentinel")
        cancellation = Cancelled("cancel sentinel")
        cases = (
            ("transport", {"run_exception": transport_error}, RuntimeError,
             "transport sentinel", transport_error),
            ("local", {
                "response_builder": lambda request: _response(
                    request, prompt="ordinary prose, not H3"
                )
            }, RuntimeError, "local H3 validation failed", None),
            ("cancellation", {"interruption_exception": cancellation},
             Cancelled, "cancel sentinel", cancellation),
        )
        for name, client_kwargs, error_type, message, exact in cases:
            with self.subTest(name=name), _temp_staging() as temp, _fake_client(
                **client_kwargs
            ):
                with self.assertRaisesRegex(error_type, message) as caught:
                    _generate(
                        h3_mode="base_I2VA",
                        subject_1_image=_image(0.1),
                    )
                if exact is not None:
                    self.assertIs(caught.exception, exact)
                jobs = list((temp / "h3_hermes").iterdir())
                self.assertEqual(len(jobs), 1)
                self.assertTrue((jobs[0] / ".retention_until").is_file())

    def test_uninspected_evidence_is_honest_bounded_and_not_inspection(self):
        captured = {}

        def uninspected(request):
            captured["assets"] = request["assets"]
            response = _response(request)
            response["evidence"]["uninspected_assets"] = ["<Picture 1>"] * 80
            response["quality_report"]["reported_tools"] = ["vision"]
            return response

        with _temp_staging(), _fake_client(response_builder=uninspected):
            named = _named(_generate(
                h3_mode="base_I2VA",
                subject_1_image=_image(0.1),
            ))
        wire = named["analysis_json"]
        analysis = json.loads(wire)
        self.assertEqual(analysis["uninspected_assets"], ["<Picture 1>"])
        self.assertEqual(
            analysis["inspection"]["model_reported_uninspected_assets"],
            analysis["uninspected_assets"],
        )
        self.assertEqual(analysis["inspection"]["verified_inspected_labels"], [])
        self.assertEqual(analysis["hermes"]["model_reported_tools"], ["vision"])
        self.assertEqual(analysis["hermes"]["verified_tool_events"], [])
        self.assertEqual(analysis["evidence"]["uninspected_assets"],
                         analysis["uninspected_assets"])
        asset = captured["assets"][0]
        self.assertNotIn(asset["path"], wire)
        self.assertNotIn(asset["sha256"], wire)

    def test_base_picture_stays_authoritatively_uninspected_despite_model_claims(self):
        def claims_research_only(request):
            response = _response(request)
            response["evidence"]["uninspected_assets"] = []
            response["quality_report"]["reported_tools"] = ["web_search"]
            return response

        with _temp_staging(), _fake_client(response_builder=claims_research_only):
            named = _named(_generate(
                h3_mode="base_I2VA",
                subject_1_image=_image(0.1),
            ))

        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["uninspected_assets"], ["<Picture 1>"])
        self.assertEqual(
            analysis["inspection"]["model_reported_uninspected_assets"], []
        )
        self.assertEqual(
            analysis["inspection"]["verified_inspected_labels"], []
        )
        self.assertEqual(analysis["hermes"]["model_reported_tools"], ["web_search"])
        self.assertEqual(analysis["hermes"]["verified_tool_events"], [])
        self.assertEqual(analysis["evidence"]["uninspected_assets"], [])

    def test_t2va_uses_explicit_uuid_without_creating_a_job_directory(self):
        captured = {}

        def remember(request):
            captured["request"] = request
            return _response(request)

        with _temp_staging() as temp, _fake_client(response_builder=remember):
            named = _named(_generate())
            self.assertFalse((temp / "h3_hermes").exists())
        request = captured["request"]
        self.assertRegex(
            request["request_id"],
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        )
        analysis = json.loads(named["analysis_json"])
        self.assertEqual(analysis["request"]["labels"], [])
        self.assertEqual(analysis["inspection"], {
            "staging_complete": False,
            "staged_labels": [],
            "model_reported_uninspected_assets": [],
            "verified_inspected_labels": [],
        })
        self.assertEqual(analysis["uninspected_assets"], [])
        self.assertEqual(analysis["evidence"]["uninspected_assets"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
