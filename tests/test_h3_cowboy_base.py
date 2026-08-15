"""
Base-mode conformance tests for the Ultimate H3 Cowboy Promptor.

The anchor is `test_the_four_official_cases_pass_clean`: all four of
MiniMax's own base examples (guide_base section 5) must survive the
validator with zero retry errors and zero warnings, and come back
byte-identical. They are the only base-mode text we know for certain is
correct, and between them they carry every detail that looks like a typo
- the em dash, FL2VA's bare tags, the same-line headers, and a picture
that lands at the END of the video.

Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_cowboy_base.py
"""

import json
import os
import re
import sys
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils", "utils.h3_prompt", "utils.h3_cowboy",
                "utils.cut_detect"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402

from TrentNodes.nodes import (  # noqa: E402
    ultimate_h3_cowboy_promptor as node_mod,
)
from TrentNodes.utils.h3_cowboy import prompts_base, spec  # noqa: E402
from TrentNodes.utils.h3_cowboy.assembler import (  # noqa: E402
    BASE_MODES,
    CowboyContext,
    process,
)
from TrentNodes.utils.h3_cowboy.wiring import snap_length  # noqa: E402
from TrentNodes.utils.h3_prompt.backends import VLMResult  # noqa: E402

EM_DASH = "\u2014"

# A minimal, correct base reply: one shot, three same-line fields.
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


def _ctx(mode="base_T2VA", **overrides):
    kwargs = dict(
        mode=mode, duration_seconds=5.0,
        wired_pictures=0 if mode == "base_T2VA" else (
            2 if mode == "base_FL2VA" else 1
        ),
        task_type="",
    )
    kwargs.update(overrides)
    return CowboyContext(**kwargs)


def _official_ctx(mode, **overrides):
    kwargs = dict(
        mode=mode, duration_seconds=spec.EXAMPLE_BASE_DURATION[mode],
        task_type="",
    )
    kwargs.update(overrides)
    return CowboyContext(**kwargs)


# ---------------------------------------------------------------------------
# The anchors - MiniMax's own four cases
# ---------------------------------------------------------------------------

def test_the_four_official_cases_pass_clean():
    for mode in BASE_MODES:
        result = process(spec.EXAMPLE_FOR_BASE_MODE[mode], _official_ctx(mode))
        assert result.retry_errors == [], (mode, result.retry_errors)
        assert result.warnings == [], (mode, result.warnings)


def test_the_four_official_cases_round_trip_byte_for_byte():
    """
    Including the instruction line, which the node re-renders rather than
    keeps: N comes from counting [Shot N] in the parsed body and S.SS
    from the duration, so an identical line back is evidence the whole
    render-after-the-fact path agrees with the guide.
    """
    for mode in BASE_MODES:
        example = spec.EXAMPLE_FOR_BASE_MODE[mode]
        result = process(example, _official_ctx(mode))
        assert result.prompt.strip() == example.strip(), mode


def test_the_embedded_examples_are_the_guides_text():
    # If someone "tidies" these, the anchors above stop meaning anything.
    # Every assertion here is a detail the guide is easy to disagree with.
    t2va = spec.EXAMPLE_BASE_T2VA
    assert t2va.startswith(
        "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
        "a medium-wide shot frames a baker"
    )
    assert "For the target video" not in t2va      # T2VA has no instruction

    i2va = spec.EXAMPLE_BASE_I2VA
    assert i2va.startswith(
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )

    fl2va = spec.EXAMPLE_BASE_FL2VA
    assert fl2va.startswith(
        "How the reference pictures align with the target video "
        + EM_DASH
        + " Picture 1 (from Shot 1) aligns with the 0.00-second mark"
    )
    assert "Picture 2 (from Shot 1) aligns with the 8.00-second mark" in fl2va
    # The bare spelling holds in the BODY too, which is what makes it
    # systematic rather than a slip in one line.
    assert "established by Picture 1" in fl2va
    assert "established by Picture 2 at the end of the shot" in fl2va
    assert "<Picture" not in fl2va and "[Picture" not in fl2va
    assert fl2va.rstrip().endswith("non_diegetic_music: N/A")

    l2va = spec.EXAMPLE_BASE_L2VA
    assert l2va.startswith(
        "How the reference pictures align with the target video "
        + EM_DASH
        + " <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark"
    )
    assert "final composition established by <Picture 1>." in l2va


# ---------------------------------------------------------------------------
# The three things that look like typos
# ---------------------------------------------------------------------------

def test_the_headers_carry_their_content_on_the_same_line():
    prompt = process(SIMPLE, _ctx()).prompt
    assert re.search(r"^integrated_multimodal_description: \[Shot 1\]",
                     prompt, re.MULTILINE), prompt[:120]
    for key in spec.BASE_SECTION_ORDER:
        assert not re.search(rf"^{key}:$", prompt, re.MULTILINE), key
        assert re.search(rf"^{key}: \S", prompt, re.MULTILINE), key


def test_a_ref_style_header_is_pulled_back_onto_one_line():
    ref_shaped = SIMPLE.replace(
        "integrated_multimodal_description: ",
        "integrated_multimodal_description:\n",
    )
    result = process(ref_shaped, _ctx())
    assert result.retry_errors == [], result.retry_errors
    assert re.search(r"^integrated_multimodal_description: \[Shot 1\]",
                     result.prompt, re.MULTILINE)
    assert any("one line" in f for f in result.applied_fixes), \
        result.applied_fixes


def test_the_fl2va_line_has_no_brackets_and_the_others_do():
    fl2va = spec.render_instruction_line("base_FL2VA", 1, 8.0)
    assert "<" not in fl2va and "[" not in fl2va, fl2va
    assert "Picture 1 (from Shot 1)" in fl2va

    i2va = spec.render_instruction_line("base_I2VA")
    l2va = spec.render_instruction_line("base_L2VA", 1, 6.0)
    assert "<Picture 1>" in i2va and "[Shot 1]" in i2va
    assert "<Picture 1>" in l2va and "[Shot 1]" in l2va


def test_the_separator_is_an_em_dash_and_never_a_hyphen():
    # The older package ships " - " here; that is a real deviation, and
    # this is the test that stops it being copied over.
    for mode, seconds in (("base_FL2VA", 8.0), ("base_L2VA", 6.0)):
        line = spec.render_instruction_line(mode, 1, seconds)
        assert EM_DASH in line, mode
        assert " - " not in line, mode
    # I2VA's line has no dash of any kind in the guide.
    assert EM_DASH not in spec.render_instruction_line("base_I2VA")
    assert spec.render_instruction_line("base_T2VA") == ""


def test_the_instruction_line_is_driven_by_the_parsed_body():
    line = spec.render_instruction_line(
        "base_L2VA", final_shot=3, duration_seconds=7.0
    )
    assert "(from [Shot 3])" in line, line
    assert "7.00-second" in line, line

    three_shots = SIMPLE.replace(
        "she straightens up beyond the doorway.",
        "she straightens up beyond the doorway. [Shot 2] At 00:02.000, the "
        "camera cuts to the corridor beyond. [Shot 3] At 00:04.000, the shot "
        "cuts to her hand meeting the frame exactly as <Picture 1> shows it.",
    )
    result = process(three_shots, _ctx("base_L2VA", duration_seconds=7.0))
    assert result.prompt.startswith(
        "How the reference pictures align with the target video"
    )
    assert "(from [Shot 3])" in result.prompt
    assert "7.00-second" in result.prompt


def test_a_model_written_instruction_line_is_stripped_and_replaced():
    # The worked example contains one, so imitation is likely - and any
    # line written before the body exists is guessing at N and S.SS.
    wrong = (
        "How the reference pictures align with the target video "
        + EM_DASH
        + " <Picture 1> (from [Shot 9]) aligns with the 99.00-second mark "
        "of the target video.\n\n" + SIMPLE
    )
    result = process(wrong, _ctx("base_L2VA", duration_seconds=6.0))
    assert "[Shot 9]" not in result.prompt
    assert "99.00-second" not in result.prompt
    assert "(from [Shot 1])" in result.prompt
    assert "6.00-second" in result.prompt


def test_one_blank_line_separates_the_instruction_from_the_body():
    result = process(SIMPLE, _ctx("base_I2VA"))
    head, _, rest = result.prompt.partition("\n")
    assert head.endswith("is fully referenced.")
    assert rest.startswith("\nintegrated_multimodal_description:"), rest[:60]


# ---------------------------------------------------------------------------
# R1 FORMAT
# ---------------------------------------------------------------------------

def test_base_mode_never_emits_the_ref_sections():
    prompt = process(SIMPLE, _ctx()).prompt
    for key in ("subject_definitions", "summary", "retention_analysis",
                "detailed_description"):
        assert key not in prompt, key


def test_a_ref_shaped_reply_is_exactly_one_r1_error():
    ref_shaped = (
        "subject_definitions:\n<Subject 1> is the courier.\n\n"
        "summary:\n[reference generation] The target video follows "
        "<Subject 1>.\n\n"
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): "
        "fully_preserved - the jacket is retained.\n\n" + SIMPLE
    )
    result = process(ref_shaped, _ctx())
    assert len(result.retry_errors) == 1, result.retry_errors
    message = result.retry_errors[0]
    assert message.startswith("R1 FORMAT")
    assert "subject_definitions" in message
    assert "<Subject 1>" in message


def test_a_missing_field_is_an_r1_error_and_is_never_synthesized():
    # The ref parser invents overall_soundscape and non_diegetic_music
    # when they are absent. Here all three fields ARE the format, so a
    # missing one is a retry, and inventing it would paper over the
    # model writing some other format entirely.
    without = SIMPLE.split("\n\noverall_soundscape:")[0]
    result = process(without, _ctx())
    assert len(result.retry_errors) == 1, result.retry_errors
    assert "overall_soundscape" in result.retry_errors[0]
    assert "non_diegetic_music" in result.retry_errors[0]
    assert "overall_soundscape" not in result.prompt


def test_a_picture_the_mode_does_not_have_is_caught():
    cited = SIMPLE.replace(
        "a courier ducking", "the courier from <Picture 1> ducking"
    )
    result = process(cited, _ctx("base_T2VA"))
    assert any("no reference picture at all" in e
               for e in result.retry_errors), result.retry_errors

    # I2VA has one picture, so <Picture 2> points at nothing.
    two = SIMPLE.replace(
        "a courier ducking",
        "the courier from <Picture 1> ducking, landing on <Picture 2>",
    )
    result = process(two, _ctx("base_I2VA"))
    assert any("only 1 picture" in e for e in result.retry_errors), \
        result.retry_errors


def test_empty_input_is_r5():
    result = process("   ", _ctx())
    assert len(result.retry_errors) == 1
    assert result.retry_errors[0].startswith("R5 EMPTY")


def test_supplied_dialogue_must_reach_a_d_block():
    ctx = _ctx(dialogue_text="Mind the shutter.")
    assert any(e.startswith("R3 VERBATIM")
               for e in process(SIMPLE, ctx).retry_errors)

    spoken = SIMPLE.replace(
        "beyond the doorway.",
        "beyond the doorway. The courier (S1) says: <d>[English] Mind the "
        "shutter.</d>",
    )
    assert process(spoken, ctx).retry_errors == []


# ---------------------------------------------------------------------------
# Deterministic repairs
# ---------------------------------------------------------------------------

def test_a_style_sentence_before_shot_one_is_moved_inside_it():
    # guide_ref 5.2 tables the difference: ref mode puts the style before
    # [Shot 1], base mode puts it after the label on the same line.
    ref_habit = SIMPLE.replace(
        "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
        "a medium-wide",
        "integrated_multimodal_description: The video is live-action and "
        "cinematic. [Shot 1] A medium-wide",
    )
    result = process(ref_habit, _ctx())
    assert re.search(
        r"^integrated_multimodal_description: \[Shot 1\] The video is "
        r"live-action and cinematic\. A medium-wide",
        result.prompt, re.MULTILINE,
    ), result.prompt[:200]
    assert any("inside [Shot 1]" in f for f in result.applied_fixes)


def test_shot_one_never_keeps_a_timestamp():
    stamped = SIMPLE.replace("[Shot 1]", "[Shot 1] At 00:00.000,")
    prompt = process(stamped, _ctx()).prompt
    assert "[Shot 1] At" not in prompt


def test_later_shot_times_are_normalized_and_ordered():
    messy = SIMPLE.replace(
        "she straightens up beyond the doorway.",
        "she straightens up. [Shot 2] At 9.5s, the shot cuts to the "
        "corridor. [Shot 3] At 1.0s, the camera cuts back to the bay.",
    )
    prompt = process(messy, _ctx(duration_seconds=6.0)).prompt
    times = re.findall(r"\[Shot \d+\] At (\d\d:\d\d\.\d\d\d),", prompt)
    assert len(times) == 2, prompt
    assert times == sorted(times), times


# ---------------------------------------------------------------------------
# Warnings - and the things that must NOT be validated
# ---------------------------------------------------------------------------

def test_na_in_non_diegetic_music_is_never_flagged():
    # Legal per guide_base 4.7, and Case 3 uses it.
    silent = SIMPLE.replace(
        "non_diegetic_music: A low synth pulse at a slow tempo, thinning "
        "out as she clears the doorway.",
        "non_diegetic_music: N/A",
    )
    result = process(silent, _ctx())
    assert result.retry_errors == [], result.retry_errors
    assert result.warnings == [], result.warnings
    assert result.prompt.rstrip().endswith("non_diegetic_music: N/A")


def test_an_unusual_cut_phrase_only_warns():
    dissolved = SIMPLE.replace(
        "she straightens up beyond the doorway.",
        "she straightens up. [Shot 2] At 00:03.000, the image "
        "cross-dissolves to the corridor beyond.",
    )
    result = process(dissolved, _ctx(duration_seconds=6.0))
    assert result.retry_errors == [], result.retry_errors
    assert any("cut phrases" in w for w in result.warnings), result.warnings


def test_a_later_shot_must_open_with_its_cut():
    """
    guide_base 4.2 gives the form as a template - "[Shot 2] At 00:03.500,
    the camera cuts to..." - and all six official examples across both
    guides follow it. A cut mentioned after the description reads fine
    and is not the shape, so it gets its own message.
    """
    buried = SIMPLE.replace(
        "she straightens up beyond the doorway.",
        "she straightens up. [Shot 2] At 00:03.000, a wet corridor stretches "
        "away under sodium light as the shot cuts to it.",
    )
    result = process(buried, _ctx(duration_seconds=6.0))
    assert result.retry_errors == [], result.retry_errors
    assert any("does not open with it" in w for w in result.warnings), \
        result.warnings

    for phrase in ("the camera cuts to", "the shot cuts to",
                   "the shot transitions to", "the shot changes to",
                   "the shot switches to"):
        clean = SIMPLE.replace(
            "she straightens up beyond the doorway.",
            f"she straightens up. [Shot 2] At 00:03.000, {phrase} a wet "
            "corridor under sodium light.",
        )
        assert process(clean, _ctx(duration_seconds=6.0)).warnings == [], phrase


def test_l2va_warns_when_the_picture_lands_at_the_start():
    # guide_base 3.3: <Picture 1> "does not inherently belong to Shot 1".
    backwards = SIMPLE.replace(
        "a courier ducking", "the courier shown in <Picture 1> ducking"
    ).replace(
        "she straightens up beyond the doorway.",
        "she straightens up. [Shot 2] At 00:03.000, the shot cuts to the "
        "corridor beyond.",
    )
    result = process(backwards, _ctx("base_L2VA", duration_seconds=6.0))
    assert result.retry_errors == [], result.retry_errors
    assert any("does not inherently belong to Shot 1" in w
               for w in result.warnings), result.warnings


def test_i2va_warns_when_the_first_frame_is_not_cited_in_shot_one():
    result = process(SIMPLE, _ctx("base_I2VA"))
    assert result.retry_errors == [], result.retry_errors
    assert any("first frame" in w for w in result.warnings), result.warnings


def test_fl2va_multiple_shots_warn_unless_they_were_asked_for():
    two_shot = SIMPLE.replace(
        "a courier ducking", "a courier framed as Picture 1 shows, ducking"
    ).replace(
        "she straightens up beyond the doorway.",
        "she straightens up. [Shot 2] At 00:03.000, the shot cuts to the "
        "corridor, settling into the framing of Picture 2.",
    )
    ctx = _ctx("base_FL2VA", duration_seconds=6.0)
    result = process(two_shot, ctx)
    assert result.retry_errors == [], result.retry_errors
    assert any("single shot" in w for w in result.warnings), result.warnings

    asked = _ctx("base_FL2VA", duration_seconds=6.0, multi_shot_requested=True)
    assert not any("single shot" in w for w in process(two_shot, asked).warnings)


def test_the_soundscape_sentence_budget_only_warns():
    long_one = SIMPLE.replace(
        "overall_soundscape: Rain drums on corrugated steel while the "
        "shutter chain rattles overhead. Boots scuff across wet concrete.",
        "overall_soundscape: Rain falls. Steel rattles. Boots scuff. Water "
        "drips. A van idles.",
    )
    result = process(long_one, _ctx())
    assert result.retry_errors == [], result.retry_errors
    assert any("overall_soundscape is 5 sentence(s)" in w
               for w in result.warnings), result.warnings


def test_fl2va_tag_normalization_is_off_by_default_and_reversible():
    example = spec.EXAMPLE_BASE_FL2VA
    plain = process(example, _official_ctx("base_FL2VA"))
    assert "<Picture 1>" not in plain.prompt
    assert "Picture 1 (from Shot 1)" in plain.prompt

    normalized = process(
        example,
        _official_ctx("base_FL2VA", fl2va_normalize_picture_tags=True),
    )
    assert "<Picture 1> (from [Shot 1])" in normalized.prompt
    assert "established by <Picture 2>" in normalized.prompt
    assert any("fl2va_normalize_picture_tags" in f
               for f in normalized.applied_fixes)


# ---------------------------------------------------------------------------
# The composed system prompt
# ---------------------------------------------------------------------------

def test_each_mode_ships_its_own_single_worked_example():
    for mode in BASE_MODES:
        system = prompts_base.build_system_prompt(mode)
        assert system.count("## WORKED EXAMPLE") == 1, mode
        assert system.count(spec.EXAMPLE_FOR_BASE_MODE[mode]) == 1, mode
        for other in BASE_MODES:
            if other != mode:
                assert spec.EXAMPLE_FOR_BASE_MODE[other] not in system, (
                    mode, other
                )


def test_the_system_prompt_forbids_the_ref_furniture():
    system = prompts_base.build_system_prompt("base_T2VA")
    assert "Never write <Subject N>, <Video N> or <Audio N>" in system
    assert "Do NOT write the alignment instruction line" in system
    assert "THE CONTENT GOES ON THE SAME LINE AS ITS HEADER" in system


def test_each_mode_gets_its_own_body_shape_and_only_its_own():
    fl2va = prompts_base.build_system_prompt("base_FL2VA")
    assert "THIS RUN IS FL2VA" in fl2va
    assert "THIS RUN IS L2VA" not in fl2va
    assert "NO angle brackets" in fl2va

    l2va = prompts_base.build_system_prompt("base_L2VA")
    assert "does not inherently belong to Shot 1" in l2va


def test_the_context_says_what_each_picture_is():
    i2va = prompts_base.build_user_context("base_I2VA", "a courier", 5.0, 24.0)
    assert "IS THE FIRST FRAME" in i2va

    fl2va = prompts_base.build_user_context(
        "base_FL2VA", "a courier", 8.0, 24.0
    )
    assert "Picture 2 IS THE LAST FRAME at 8.00 seconds" in fl2va

    t2va = prompts_base.build_user_context("base_T2VA", "a courier", 5.0, 24.0)
    assert "no pictures are attached" in t2va


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class FakeBackend:
    name = "fake"
    supports_audio = False
    supports_video = False

    def __init__(self, reply=SIMPLE):
        self.calls = 0
        self.reply = reply
        self.first_user_text = ""
        self.last_user_text = ""
        self.system = ""
        self.image_labels = []

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None, video=None):
        self.calls += 1
        self.last_user_text = user_text
        if self.calls == 1:
            self.first_user_text = user_text
            self.system = system
            self.image_labels = [im.label for im in images]
        return VLMResult(text=self.reply, usage={"model": "fake-model"})


def _run(fake, **overrides):
    original = node_mod.get_backend
    node_mod.get_backend = lambda *a, **k: fake
    try:
        kwargs = dict(
            h3_mode="base_T2VA",
            subjects="",
            target_description="a courier ducks under a roller shutter",
            vlm_provider="anthropic",
            model="auto",
        )
        kwargs.update(overrides)
        return node_mod.UltimateH3CowboyPromptor().generate(**kwargs)
    finally:
        node_mod.get_backend = original


def test_t2va_runs_with_no_images_no_video_and_no_frames():
    fake = FakeBackend()
    prompt, duration, fps, analysis_json, checkpoint = _run(
        fake, duration_override=6.0
    )[:5]
    assert fake.calls == 1
    assert prompt.startswith("integrated_multimodal_description: [Shot 1]")
    assert duration == round(snap_length(6.0)[1], 3)
    assert checkpoint == "MiniMax-H3-Base-FL2VA"
    assert fake.image_labels == []

    analysis = json.loads(analysis_json)
    assert analysis["mode"] == "base_T2VA"
    assert analysis["unresolved_errors"] == []
    assert analysis["duration_source"] == "override"
    assert analysis["anchor_pictures"] == []


def test_t2va_without_a_duration_says_which_number_it_guessed():
    fake = FakeBackend()
    _p, duration, _f, analysis_json, _c = _run(fake)[:5]
    assert duration == round(snap_length(node_mod.DEFAULT_BASE_DURATION)[1], 3)
    assert any("duration_override" in w
               for w in json.loads(analysis_json)["warnings"])


def test_i2va_labels_the_wired_slot_as_the_first_frame():
    fake = FakeBackend(
        reply=SIMPLE.replace(
            "a courier ducking", "the courier shown in <Picture 1> ducking"
        )
    )
    _p, _d, _f, analysis_json, _c = _run(
        fake, h3_mode="base_I2VA",
        subject_1_image=torch.rand((1, 96, 96, 3)),
        duration_override=5.0,
    )[:5]
    assert len(fake.image_labels) == 1
    assert fake.image_labels[0].startswith("<Picture 1> - the FIRST frame")
    assert json.loads(analysis_json)["anchor_pictures"] == ["Picture 1"]


def test_fl2va_without_two_pictures_is_the_one_hard_error():
    try:
        _run(
            FakeBackend(), h3_mode="base_FL2VA",
            subject_1_image=torch.rand((1, 96, 96, 3)),
            duration_override=8.0,
        )
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "subject_2_image" in str(exc), exc


def test_ref_only_widgets_warn_and_never_raise():
    fake = FakeBackend()
    _p, _d, _f, analysis_json, _c = _run(
        fake,
        subjects="person the courier -- olive rain jacket",
        video_role="edit_source",
        audio_role="reuse",
        frames=torch.rand((24, 64, 64, 3)),
        fps=12.0,
    )[:5]
    text = " ".join(json.loads(analysis_json)["warnings"])
    assert "subjects field" in text
    assert "video_role" in text and "audio_role" in text
    assert "no <Video N>" in text


def test_a_wired_clip_is_context_only_and_not_citable():
    fake = FakeBackend()
    _run(fake, frames=torch.rand((24, 64, 64, 3)), fps=12.0)
    assert fake.image_labels, "context frames should still be sent"
    assert all("not citable" in label for label in fake.image_labels)
    assert "must not be cited" in fake.first_user_text


def test_a_cut_list_is_read_as_the_requested_shot_structure():
    fake = FakeBackend()
    _p, _d, _f, analysis_json, _c = _run(
        fake, h3_mode="base_FL2VA",
        subject_1_image=torch.rand((1, 96, 96, 3)),
        subject_2_image=torch.rand((1, 96, 96, 3)),
        duration_override=8.0, cut_times="0.0, 4.0",
    )[:5]
    analysis = json.loads(analysis_json)
    assert analysis["requested_shot_times"] == [0.0, 4.0]
    # Asking for two shots is guide_base 3.2's "explicitly specified", so
    # the single-shot preference stops warning.
    assert not any("single shot" in w for w in analysis["warnings"])


def test_a_contradictory_picture_role_warns_and_h3_mode_wins():
    fake = FakeBackend()
    _p, _d, _f, analysis_json, _c = _run(
        fake, h3_mode="base_I2VA", base_picture_role="last_frame",
        subject_1_image=torch.rand((1, 96, 96, 3)),
        duration_override=5.0,
    )[:5]
    analysis = json.loads(analysis_json)
    assert any("h3_mode wins" in w for w in analysis["warnings"])
    assert analysis["mode"] == "base_I2VA"


def test_widgets_are_only_ever_appended():
    # A saved workflow stores widget values POSITIONALLY, so inserting an
    # optional input in the middle hands every later widget the value of
    # its neighbour. Everything new goes on the end - this pins the order
    # every graph saved so far was built against.
    optional = list(node_mod.UltimateH3CowboyPromptor.INPUT_TYPES()["optional"])
    assert optional[:22] == [
        "video", "frames", "fps", "audio", "api_key", "video_role",
        "audio_role", "cut_times", "dialogue", "constraint_notes",
        "duration_override", "max_frames_to_analyze", "seed",
        "subject_1_image", "subject_2_image", "subject_3_image",
        "subject_4_image", "subject_5_image", "subject_6_image",
        "base_picture_role", "fl2va_normalize_picture_tags",
        "snap_duration_to_h3_grid",
    ], optional[:22]
    # Then the rows, grouped by row so the face reads down one subject at
    # a time, then the music-video block.
    assert optional[22] == "subject_rows"
    assert optional[23:26] == [
        "subject_1_kind", "subject_1_name", "subject_1_description",
    ]
    assert optional[-4:] == [
        "music_video", "music_source", "lyrics", "music_description",
    ]


# ---------------------------------------------------------------------------
# The timing fix: the prompt states the length H3 really produces
# ---------------------------------------------------------------------------

def _named(out) -> dict:
    return dict(zip(node_mod.UltimateH3CowboyPromptor.RETURN_NAMES, out))


def test_snapping_makes_the_instruction_line_agree_with_length():
    # L2VA writes the duration into the instruction line as S.SS. Ask
    # for 2.00 seconds and H3 renders 56 frames, which is 2.33 - so 2.33
    # is what the line has to say.
    fake = FakeBackend(reply=SIMPLE)
    out = _named(_run(
        fake, h3_mode="base_L2VA",
        subject_1_image=torch.rand((1, 96, 96, 3)),
        duration_override=2.0,
    ))
    assert out["length"] == 56
    assert "2.33-second mark" in out["h3_prompt"], out["h3_prompt"][:400]
    assert out["duration_seconds"] == 2.333

    analysis = json.loads(out["analysis_json"])
    assert analysis["requested_duration_seconds"] == 2.0
    assert analysis["snapped_duration_seconds"] == 2.333
    assert analysis["h3_length_frames"] == 56


def test_snapping_off_keeps_the_requested_number_and_says_so():
    fake = FakeBackend(reply=SIMPLE)
    out = _named(_run(
        fake, h3_mode="base_L2VA",
        subject_1_image=torch.rand((1, 96, 96, 3)),
        duration_override=2.0, snap_duration_to_h3_grid=False,
    ))
    assert out["duration_seconds"] == 2.0
    assert "2.00-second mark" in out["h3_prompt"]
    # The length output still has to be a legal frame count.
    assert out["length"] == 56
    assert any(
        "not the 2.000s this prompt states" in w
        for w in json.loads(out["analysis_json"])["warnings"]
    )


def test_t2va_with_nothing_wired_still_returns_the_whole_tuple():
    # Base mode has no <Video 1> and no <Audio 1>, and T2VA has no
    # pictures either. Empty is the honest answer, not a fault.
    out = _named(_run(FakeBackend(), duration_override=8.0))
    for slot in range(1, node_mod.NUM_SUBJECT_SLOTS + 1):
        assert out[f"ref_image_{slot}"] is None
    assert out["ref_video"] is None
    assert out["ref_video_audio"] is None and out["ref_audio"] is None
    assert (out["width"], out["height"]) == (0, 0)
    assert out["length"] == 192          # 8.00s lands on the grid exactly
    assert out["label_map"] == "nothing connected"


def test_a_base_anchor_picture_passes_through_to_its_own_slot():
    picture = torch.rand((1, 96, 96, 3))
    out = _named(_run(
        FakeBackend(reply=SIMPLE), h3_mode="base_I2VA",
        subject_1_image=picture, duration_override=8.0,
    ))
    assert out["ref_image_1"] is picture
    assert (out["width"], out["height"]) == (768, 768)
    assert out["label_map"] == "<Picture 1> = subject_1_image"


def test_every_mode_reaches_a_checkpoint_hint():
    for mode in spec.MODES:
        assert spec.CHECKPOINT_FOR_MODE[mode].startswith("MiniMax-H3-Base-")
    assert spec.CHECKPOINT_FOR_MODE["ref"] != \
        spec.CHECKPOINT_FOR_MODE["base_T2VA"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All cowboy base-mode tests passed.")
