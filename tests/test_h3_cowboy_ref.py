"""
Ref-mode conformance tests for the Ultimate H3 Cowboy Promptor.

The anchor test is `test_minimax_own_example_passes_clean`: MiniMax's own
complete reference example must survive the validator untouched. It is
the strongest available check, because it is the one piece of text we
know for certain is correct - and the older node produces three retry
errors on it.

Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_cowboy_ref.py
"""

import inspect
import os
import sys
import types

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

from TrentNodes.utils.h3_cowboy import prompts_ref, spec  # noqa: E402
from TrentNodes.utils.h3_cowboy.assembler import (  # noqa: E402
    CowboyContext,
    process,
)
from TrentNodes.utils.h3_cowboy.subjects import parse_subjects  # noqa: E402

# The subjects the official example itself declares: an environment, an
# animal, and two people, across four pictures and two videos.
OFFICIAL_SUBJECTS = """\
scene    the coffee shop  @Picture 1                       -- exposed brick, orange tufted sofa, neon sign
animal   the Samoyed      @Picture 2 @Picture 3 @Picture 4 -- thick white fur, pointed ears, curved tail
person   the blonde woman @Video 1                         -- long blonde hair, light-pink shirt
person   the young man    @Video 2                         -- short wavy brown hair, dark-grey hoodie
"""


def _official_ctx(**overrides):
    kwargs = dict(
        subjects=parse_subjects(OFFICIAL_SUBJECTS, []),
        duration_seconds=7.0,
        task_type="reference generation + audio reference",
        wired_pictures=4, has_video=True, has_audio=True,
    )
    kwargs.update(overrides)
    return CowboyContext(**kwargs)


def _simple_ctx(**overrides):
    kwargs = dict(
        subjects=parse_subjects(
            "person the courier @Picture 1 -- olive rain jacket", []
        ),
        duration_seconds=5.0, task_type="reference generation",
        wired_pictures=1,
    )
    kwargs.update(overrides)
    return CowboyContext(**kwargs)


SIMPLE = """subject_definitions:
<Subject 1> is the courier in <Picture 1>, with short dark hair and an olive rain jacket.

summary:
[reference generation] The target video follows <Subject 1> through a loading bay.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the olive rain jacket and short dark hair are retained.

detailed_description:
The target video has a grounded handheld documentary look.
[Shot 1] A tracking shot holds <Subject 1> center frame as the courier ducks under a roller shutter.

overall_soundscape:
Rain on corrugated steel, with footsteps on wet concrete.

non_diegetic_music:
N/A"""


# ---------------------------------------------------------------------------
# The anchor
# ---------------------------------------------------------------------------

def test_minimax_own_example_passes_clean():
    """
    guide_ref section 7, verbatim, through the validator.

    Zero retry errors and zero warnings. The older node emits three
    retry errors on this same text, because its <Subject 2> does not
    appear in [Shot 3] - which the guide's own retention line states
    outright: "<Subject 2> (appears in [Shot 1], [Shot 2])".
    """
    result = process(spec.EXAMPLE_REF_GENERATION, _official_ctx())
    assert result.retry_errors == [], result.retry_errors
    assert result.warnings == [], result.warnings


def test_the_official_example_round_trips_byte_for_byte():
    result = process(spec.EXAMPLE_REF_GENERATION, _official_ctx())
    assert result.prompt.strip() == spec.EXAMPLE_REF_GENERATION.strip()


def test_the_embedded_example_is_the_guides_text():
    # If someone "tidies" the example, the anchor test above stops
    # meaning anything. Pin its shape.
    text = spec.EXAMPLE_REF_GENERATION
    assert text.startswith("subject_definitions:\n<Subject 1> is the "
                           "coffee-shop environment")
    assert text.rstrip().endswith("non_diegetic_music:\nN/A")
    assert "<Subject 2> (appears in [Shot 1], [Shot 2]):" in text
    assert "[reference generation + audio reference]" in text


# ---------------------------------------------------------------------------
# What must NOT be validated
# ---------------------------------------------------------------------------

def test_a_subject_absent_from_a_shot_is_not_an_error():
    two_shot = SIMPLE.replace(
        "non_diegetic_music:",
        "[Shot 2] At 00:02.000, the shot cuts to an empty corridor.\n\n"
        "non_diegetic_music:",
    ).replace("detailed_description:\n", "detailed_description:\n")
    result = process(two_shot, _simple_ctx())
    assert result.retry_errors == [], result.retry_errors


def test_a_non_spatial_subject_is_never_asked_for_a_frame_position():
    ctx = _simple_ctx(
        subjects=parse_subjects(
            "style -- 16mm grain, halation, teal-shifted shadows", []
        ),
        wired_pictures=0,
    )
    styled = SIMPLE.replace(
        "<Subject 1> is the courier in <Picture 1>, with short dark hair "
        "and an olive rain jacket.",
        "<Subject 1> is the visual style: 16mm grain, halation, "
        "teal-shifted shadows.",
    ).replace("in <Picture 1>", "")
    result = process(styled, ctx)
    assert not any("frame" in w for w in result.warnings), result.warnings


def test_a_subject_with_no_proper_name_is_fine():
    ctx = _simple_ctx(
        subjects=parse_subjects("person @Picture 1 -- olive rain jacket", [])
    )
    result = process(SIMPLE, ctx)
    assert result.retry_errors == [], result.retry_errors


def test_a_measured_shot_count_mismatch_is_not_an_error_outside_editing():
    # The target video's structure is not bound to a reference video's;
    # guide_ref 3 puts a camera/cut/rhythm reference under "reference
    # generation", not "video editing".
    ctx = _simple_ctx(known_shot_times=[0.0, 1.5, 3.0], is_editing=False)
    result = process(SIMPLE, ctx)
    assert result.retry_errors == [], result.retry_errors


def test_a_source_citation_needs_no_retention_line():
    # guide_ref 2: a picture that only identifies where a subject came
    # from is cited inside that subject's line, with no line of its own -
    # so it must not create a retention obligation either.
    result = process(SIMPLE, _simple_ctx())
    assert "<Picture 1>" in result.prompt
    assert result.retry_errors == [], result.retry_errors


# ---------------------------------------------------------------------------
# R2 LABELS - one aggregated message
# ---------------------------------------------------------------------------

def test_a_declared_subject_that_never_appears_is_one_retry_error():
    ctx = _simple_ctx(
        subjects=parse_subjects(
            "person the courier @Picture 1 -- olive jacket\n"
            "animal the dog @Picture 2 -- brown fur", []
        ),
        wired_pictures=2,
    )
    result = process(SIMPLE, ctx)
    assert len(result.retry_errors) == 1, result.retry_errors
    message = result.retry_errors[0]
    assert message.startswith("R2 LABELS")
    assert "<Subject 2>" in message


def test_every_label_problem_lands_in_a_single_message():
    broken = SIMPLE.replace(
        "<Subject 1> (appears in [Shot 1]): fully_preserved - the olive "
        "rain jacket and short dark hair are retained.",
        "(nothing here)",
    ).replace(
        "holds <Subject 1> center frame",
        "holds <Subject 1> and <Subject 7> center frame",
    )
    result = process(broken, _simple_ctx())
    assert len(result.retry_errors) == 1, result.retry_errors
    message = result.retry_errors[0]
    assert "no line in retention_analysis" in message
    assert "<Subject 7>" in message


def test_citing_an_image_that_was_never_attached_is_caught():
    ctx = _simple_ctx(wired_pictures=0)
    result = process(SIMPLE, ctx)
    assert any("only 0 image(s) were attached" in e
               for e in result.retry_errors), result.retry_errors


# ---------------------------------------------------------------------------
# R1 / R3 / structure
# ---------------------------------------------------------------------------

def test_a_missing_section_is_one_format_error():
    without = SIMPLE.split("retention_analysis:")[0] + \
        "detailed_description:\n[Shot 1] A tracking shot holds " \
        "<Subject 1> center frame.\n\noverall_soundscape:\nRain.\n\n" \
        "non_diegetic_music:\nN/A"
    result = process(without, _simple_ctx())
    assert any(e.startswith("R1 FORMAT") for e in result.retry_errors)
    assert any("retention_analysis" in e for e in result.retry_errors)


def test_empty_input_is_r5():
    result = process("   ", _simple_ctx())
    assert len(result.retry_errors) == 1
    assert result.retry_errors[0].startswith("R5 EMPTY")


def test_supplied_dialogue_must_reach_a_d_block():
    ctx = _simple_ctx(dialogue_text="Mind the shutter.")
    result = process(SIMPLE, ctx)
    assert any(e.startswith("R3 VERBATIM") for e in result.retry_errors)

    spoken = SIMPLE.replace(
        "ducks under a roller shutter.",
        "ducks under a roller shutter. <Subject 1> (S1) says, "
        "<d>[English] Mind the shutter.</d>",
    )
    assert process(spoken, ctx).retry_errors == []


def test_a_trailing_no_block_is_dropped():
    with_block = SIMPLE + "\n\nNo extra characters. No added props."
    result = process(with_block, _simple_ctx())
    assert "No extra characters" not in result.prompt
    assert any("dropped" in f for f in result.applied_fixes)


def test_a_later_shot_must_open_with_its_cut():
    """
    Both guides give one form: shot label, timestamp, cut phrase, then
    what the new shot shows. guide_ref 5.1's own opening example writes
    "[Shot 2] At 00:09.000, the shot cuts to an extreme close-up...", and
    section 7's example does it twice - which is why
    test_minimax_own_example_passes_clean still reports zero warnings
    with this check switched on.
    """
    # The second shot goes inside detailed_description, not after it -
    # appending before non_diegetic_music would land it in
    # overall_soundscape and check nothing.
    def _two_shot(second: str) -> str:
        return SIMPLE.replace(
            "ducks under a roller shutter.",
            "ducks under a roller shutter.\n[Shot 2] At 00:02.000, " + second,
        )

    buried = _two_shot(
        "an empty corridor stretches away as the shot cuts to it."
    )
    result = process(buried, _simple_ctx())
    assert result.retry_errors == [], result.retry_errors
    assert any("does not open with it" in w for w in result.warnings), \
        result.warnings

    proper = _two_shot(
        "the shot cuts to an empty corridor stretching away under sodium "
        "light."
    )
    assert process(proper, _simple_ctx()).warnings == []


def test_all_six_sections_survive_in_order():
    prompt = process(SIMPLE, _simple_ctx()).prompt
    positions = [prompt.index(f"{k}:") for k in spec.REF_SECTION_ORDER]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# The composed system prompt
# ---------------------------------------------------------------------------

def test_only_declared_kinds_get_a_feature_card():
    subs = parse_subjects("person A -- hair\nscene B -- brick", [])
    cards = prompts_ref.build_system_prompt(subs).split("BY SUBJECT KIND")[1]
    assert "person:" in cards and "scene:" in cards
    assert "style:" not in cards and "animal:" not in cards


def test_exactly_one_worked_example_ships():
    subs = parse_subjects("person A -- hair", [])
    prompt = prompts_ref.build_system_prompt(subs)
    assert prompt.count("## WORKED EXAMPLE") == 1
    # Two examples of different shapes get blended, so the example body
    # must appear exactly once. (The section list in the output contract
    # also names subject_definitions, which is fine and not an example.)
    assert prompt.count(spec.EXAMPLE_REF_GENERATION) == 1
    assert prompt.count("<Subject 1> is the coffee-shop environment") == 1


def test_the_context_flags_non_spatial_subjects():
    subs = parse_subjects("person A -- hair\nstyle -- grain", [])
    context = prompts_ref.build_user_context(
        subs, "a courier walks", 5.0, 24.0, "reference generation"
    )
    assert "style has no position in the frame" in context
    assert "person has no position in the frame" not in context


def test_the_context_demands_the_task_type_prefix_verbatim():
    subs = parse_subjects("person A -- hair", [])
    context = prompts_ref.build_user_context(
        subs, "", 5.0, 24.0, "reference generation + audio reuse"
    )
    assert "[reference generation + audio reuse]" in context


# ---------------------------------------------------------------------------
# Task types (Phase 2)
# ---------------------------------------------------------------------------

def test_all_six_task_types_are_reachable():
    reached = set()
    for video_role in spec.VIDEO_ROLES:
        for audio_role in spec.AUDIO_ROLES:
            reached |= set(spec.derive_task_types(video_role, audio_role))
    reached |= set(spec.derive_task_types(has_frame_anchor=True))
    assert reached == set(spec.TASK_TYPES), sorted(set(spec.TASK_TYPES) - reached)


def test_a_role_maps_to_the_type_the_guide_names():
    assert spec.derive_task_types("edit_source") == ["video editing"]
    assert spec.derive_task_types("continuation_source") == [
        "video continuation"
    ]
    # guide_ref 3: a video supplying only camera, cuts or rhythm "normally
    # belongs to reference generation".
    assert spec.derive_task_types("structure_reference") == [
        "reference generation"
    ]
    assert spec.derive_task_types("edit_source", "reuse") == [
        "video editing", "audio reuse"
    ]


def test_task_types_never_repeat():
    for video_role in spec.VIDEO_ROLES:
        types = spec.derive_task_types(video_role, "reference", True)
        assert len(types) == len(set(types)), types


def test_the_prefix_is_deterministic_but_validated_as_a_set():
    # The guide states no ordering rule, and its own example writes
    # "[video editing + audio reference + audio reuse]" - out of table
    # order. So we emit one fixed order, and accept any permutation.
    once = spec.format_task_type(spec.derive_task_types("edit_source", "reuse"))
    twice = spec.format_task_type(spec.derive_task_types("edit_source", "reuse"))
    assert once == twice == "video editing + audio reuse"

    permuted = SIMPLE.replace(
        "[reference generation]", "[audio reuse + video editing]"
    )
    ctx = _simple_ctx(task_type="video editing + audio reuse", is_editing=True)
    result = process(permuted, ctx)
    summary = result.prompt.split("summary:\n", 1)[1].split("\n\n", 1)[0]
    assert summary.startswith("[audio reuse + video editing]"), summary[:80]


# ---------------------------------------------------------------------------
# Video editing (Phase 2)
# ---------------------------------------------------------------------------

def test_the_official_editing_example_passes_clean():
    """The README's case-Ref2VA H3-Context-IR output, verbatim."""
    subs = parse_subjects(
        "person the young man @Video 1 -- wavy blonde hair, pink suit", []
    )
    ctx = CowboyContext(
        subjects=subs, duration_seconds=5.0,
        task_type="video editing + audio reference + audio reuse",
        is_editing=True, wired_pictures=0, has_video=True, has_audio=True,
    )
    result = process(spec.EXAMPLE_REF_EDITING, ctx)
    assert result.retry_errors == [], result.retry_errors
    # Already correct, so nothing is injected.
    assert not any("added the" in f for f in result.applied_fixes), \
        result.applied_fixes


def test_the_editing_opener_is_injected_after_the_prefix():
    ctx = _simple_ctx(task_type="video editing", is_editing=True,
                      has_video=True)
    edited = SIMPLE.replace("[reference generation]", "[video editing]")
    result = process(edited, ctx)
    summary = result.prompt.split("summary:\n", 1)[1].split("\n\n", 1)[0]
    assert summary.startswith(
        "[video editing] The target video is an edited version of <Video 1>."
    ), summary[:110]
    assert any("summary opener" in f for f in result.applied_fixes)


def test_editing_adds_the_video_definition_and_retention_lines():
    ctx = _simple_ctx(task_type="video editing", is_editing=True,
                      has_video=True)
    result = process(SIMPLE, ctx)
    definitions = result.prompt.split("subject_definitions:\n", 1)[1]
    assert spec.EDIT_VIDEO_DEFINITION in definitions
    assert "<Video 1> (source video editing): fully_preserved" in result.prompt


def test_a_non_editing_run_gets_no_video_injections():
    result = process(SIMPLE, _simple_ctx())
    assert spec.EDIT_VIDEO_DEFINITION not in result.prompt
    assert "source video editing" not in result.prompt


def test_the_editing_example_ships_only_for_a_whole_video_job():
    assert spec.example_for("ref", ["reference generation"]) == \
        spec.EXAMPLE_REF_GENERATION
    assert spec.example_for("ref", ["video editing"]) == \
        spec.EXAMPLE_REF_EDITING
    assert spec.example_for("ref", ["video continuation"]) == \
        spec.EXAMPLE_REF_EDITING


def test_the_video_role_line_tells_the_truth_about_video_1():
    subs = parse_subjects("person A @Video 1 -- hair", [])
    plain = prompts_ref.build_user_context(
        subs, "", 5.0, 24.0, "reference generation", has_video=True,
        video_role="subject_source",
    )
    assert "no line of its own" in plain

    editing = prompts_ref.build_user_context(
        subs, "", 5.0, 24.0, "video editing", has_video=True,
        video_role="edit_source",
    )
    assert "IS THE SOURCE BEING EDITED" in editing
    assert "its own line in subject_definitions" in editing


# ---------------------------------------------------------------------------
# Locks on the older package, which must stay frozen
# ---------------------------------------------------------------------------

def test_the_old_package_is_still_ref_only_and_unchanged():
    """
    This package shares the old assembler's subject-agnostic helpers, so
    an accidental generalization over there breaks things here. Fail in
    the file the person doing it is already reading.
    """
    from TrentNodes.utils.h3_prompt import prompts as old

    assert old.SECTION_ORDER == [
        "subject_definitions", "summary", "retention_analysis",
        "detailed_description", "overall_soundscape", "non_diegetic_music",
    ]
    params = list(inspect.signature(old.build_task_type).parameters)
    assert params[:2] == ["music_is_reference", "first_frame_alignment"], (
        "tests/test_h3_format.py calls build_task_type positionally; its "
        "first two parameters are frozen"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All cowboy ref-mode tests passed.")
