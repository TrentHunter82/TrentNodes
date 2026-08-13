"""
Conformance tests against the OFFICIAL MiniMax H3 prompt format.

Every assertion here traces to a rule in one of two documents:

    huggingface.co/MiniMaxAI/MiniMax-H3
      docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md    (REF2VA, the mode
                                                    this node writes)
      docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md   (shot labels, camera
                                                    grammar, the
                                                    alignment
                                                    instruction forms)

The rule each test enforces is quoted in its docstring or comment, so a
future change to the emitted format has to argue with the spec rather
than with a preference. Pure string tests, no torch. Run from the
ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_format.py
"""

import os
import re
import sys
import types
from dataclasses import replace

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils", "utils.h3_prompt"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.utils.h3_prompt import prompts  # noqa: E402
from TrentNodes.utils.h3_prompt.assembler import (  # noqa: E402
    AssemblyContext,
    process,
)
from TrentNodes.utils.h3_prompt.prompts import (  # noqa: E402
    RETENTION_MARKERS_AUDIO,
    RETENTION_MARKERS_VISUAL,
    SECTION_ORDER,
    TASK_TYPES,
)

CTX = AssemblyContext(
    subject_name="Aria Voss",
    subject_wardrobe="charcoal utility jacket, black cargo pants",
    duration_seconds=6.0,
)

# A model output already in the official shape. Deviations are
# introduced per-test so each one fails for exactly one reason.
SPEC = """subject_definitions:
<Subject 1> is Aria Voss, whose appearance comes from <Picture 1>: facial identity, dark hair, skin tone, body proportions, and the exact wardrobe of a charcoal utility jacket and black cargo pants.
<Video 1> is the source video supplying only body movement, camera angles, framing, and cut rhythm.

summary:
[reference generation] Generate a new video in which Aria Voss is tracked through every frame of the warehouse walk from <Video 1>, fully preserving her identity and wardrobe from <Picture 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - facial identity, hair, body proportions, and the charcoal utility jacket and black cargo pants, with zero drift.
<Video 1> (cut and pacing structure): attribute_transfer - body movement, positioning, and camera work only.

detailed_description:
The target video has the grounded appearance of a gritty handheld thriller. [Shot 1] A tracking shot moves backward with small amplitude at slow speed, holding Aria Voss <Subject 1> center frame in a medium shot as she strides down a warehouse aisle. [Shot 2] At 00:03.250, the camera cuts to a static wide shot; Aria Voss <Subject 1> occupies the left third of frame, stops beside a pallet stack, and turns away toward the loading door.

overall_soundscape:
Warehouse ambience with a low ventilation hum. Boot footfalls land on Aria Voss's visible steps.

non_diegetic_music:
N/A
"""


def _sections(prompt):
    """Split an emitted prompt into {name: body}, in written order."""
    out = {}
    current = None
    for line in prompt.split("\n"):
        bare = line.strip()
        if bare[:-1] in SECTION_ORDER and bare.endswith(":"):
            current = bare[:-1]
            out[current] = []
        elif current:
            out[current].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


# ---------------------------------------------------------------------------
# Structure: guide_ref_en section 1
# ---------------------------------------------------------------------------

def test_all_six_sections_in_the_official_order():
    """"A complete rewrite output consists of six sections in the
    following order" - subject_definitions, summary, retention_analysis,
    detailed_description, overall_soundscape, non_diegetic_music."""
    prompt = process(SPEC, CTX).prompt
    positions = [prompt.index(f"{key}:") for key in SECTION_ORDER]
    assert positions == sorted(positions), positions
    assert len(set(positions)) == 6


def test_headers_are_lowercase_with_a_colon_and_a_blank_line():
    prompt = process(SPEC, CTX).prompt
    for key in SECTION_ORDER:
        assert f"\n{key}:\n" in f"\n{prompt}", key
    # Sections are separated by exactly one blank line.
    assert "\n\n\n" not in prompt


def test_nothing_follows_non_diegetic_music():
    """No official example writes anything after the last section, and
    the word "exclusion" appears in none of the guides."""
    prompt = process(SPEC, CTX).prompt
    tail = prompt.split("non_diegetic_music:\n", 1)[1]
    assert "No " not in tail, tail
    assert prompt.rstrip().endswith("N/A")


def test_a_stray_no_block_is_dropped_by_default():
    written = SPEC.rstrip() + "\n\nNo extra characters. No added props."
    result = process(written, CTX)
    assert "No extra characters" not in result.prompt
    assert any("dropped" in f for f in result.applied_fixes)


def test_the_no_block_survives_when_explicitly_requested():
    written = SPEC.rstrip() + "\n\nNo extra characters. No added props."
    result = process(written, replace(CTX, append_exclusions=True))
    assert "No extra characters" in result.prompt


# ---------------------------------------------------------------------------
# summary task-type prefix: guide_ref_en section 3
# ---------------------------------------------------------------------------

def test_summary_carries_its_mandatory_task_type_prefix():
    """"It begins with a square-bracketed task-type prefix"."""
    summary = _sections(process(SPEC, CTX).prompt)["summary"]
    assert summary.startswith("[reference generation] "), summary[:60]


def test_a_missing_prefix_is_repaired():
    written = SPEC.replace("[reference generation] ", "")
    result = process(written, CTX)
    summary = _sections(result.prompt)["summary"]
    assert summary.startswith("[reference generation] "), summary[:60]
    assert any("task-type prefix" in f for f in result.applied_fixes)


def test_the_nodes_prefix_wins_over_a_legal_but_wrong_one():
    """The node derives the prefix from what the run uses - alignment
    on means <Picture 1> is a concrete frame anchor, which is
    "keyframe completion". The model cannot see the widget."""
    result = process(
        SPEC,
        replace(CTX, task_type="keyframe completion + reference generation"),
    )
    summary = _sections(result.prompt)["summary"]
    assert summary.startswith(
        "[keyframe completion + reference generation] "
    ), summary[:80]
    assert any("corrected the summary task-type" in f
               for f in result.applied_fixes), result.applied_fixes


def test_a_matching_prefix_is_left_alone():
    result = process(SPEC, CTX)
    assert not any("task-type" in f for f in result.applied_fixes)
    assert not any("task-type" in w for w in result.warnings)


def test_an_invented_task_type_is_replaced_and_flagged():
    written = SPEC.replace("[reference generation]", "[character swap]")
    result = process(written, CTX)
    summary = _sections(result.prompt)["summary"]
    assert summary.startswith("[reference generation] "), summary[:60]
    assert any("outside the official list" in w for w in result.warnings)


def test_every_built_task_type_is_official():
    for reuse in (False, True):
        for align in (False, True):
            built = prompts.build_task_type(reuse, align)
            for part in built.split(" + "):
                assert part in TASK_TYPES, built
            # "do not repeat a type"
            parts = built.split(" + ")
            assert len(parts) == len(set(parts)), built


def test_task_type_grows_with_what_the_run_uses():
    assert prompts.build_task_type() == "reference generation"
    assert prompts.build_task_type(music_is_reference=True) == (
        "reference generation + audio reuse"
    )
    assert prompts.build_task_type(first_frame_alignment=True) == (
        "keyframe completion + reference generation"
    )


# ---------------------------------------------------------------------------
# One line per label: guide_ref_en sections 2 and 4
# ---------------------------------------------------------------------------

def test_the_worked_example_gives_each_label_its_own_line():
    """"Give each item its own line" (subject_definitions) and "Use one
    line for each reference label" (retention_analysis). The system
    prompt's worked example teaches shape more strongly than its rules
    do, so the example itself has to obey."""
    example = prompts.SYSTEM_PROMPT.split("## WORKED EXAMPLE", 1)[1]
    for key in ("subject_definitions", "retention_analysis"):
        body = example.split(f"{key}:\n", 1)[1].split("\n\n", 1)[0]
        lines = [ln for ln in body.split("\n") if ln.strip()]
        assert len(lines) >= 2, (key, body)
        for line in lines:
            assert re.match(r"^<(Subject|Picture|Video|Audio) \d+>", line), (
                key, line
            )


def test_retention_lines_carry_a_scope_parenthetical():
    example = prompts.SYSTEM_PROMPT.split("## WORKED EXAMPLE", 1)[1]
    body = example.split("retention_analysis:\n", 1)[1].split("\n\n", 1)[0]
    for line in [ln for ln in body.split("\n") if ln.strip()]:
        assert re.match(
            r"^<\w+ \d+> \([^)]+\): \w+ - ", line
        ), line


# ---------------------------------------------------------------------------
# Retention markers: guide_ref_en sections 4.1 and 4.2
# ---------------------------------------------------------------------------

def test_the_marker_sets_match_the_guide():
    """"These markers are fixed English values in the output format"."""
    assert RETENTION_MARKERS_VISUAL == (
        "fully_preserved", "partially_preserved", "attribute_transfer",
        "weak_reference",
    )
    assert RETENTION_MARKERS_AUDIO == (
        "fully_copy", "partially_copy", "reference", "weak_reference",
    )


def test_an_audio_marker_on_a_visual_tag_is_flagged():
    written = SPEC.replace(
        "<Video 1> (cut and pacing structure): attribute_transfer",
        "<Video 1> (cut and pacing structure): fully_copy",
    )
    result = process(written, CTX)
    assert any(
        "outside the official set" in w and "fully_copy" in w
        for w in result.warnings
    ), result.warnings


def test_an_invented_marker_is_flagged():
    written = SPEC.replace(
        "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved",
        "<Subject 1> (appears in [Shot 1], [Shot 2]): mostly_kept",
    )
    result = process(written, CTX)
    assert any("mostly_kept" in w for w in result.warnings), result.warnings


def test_a_correct_prompt_raises_no_marker_warning():
    result = process(SPEC, CTX)
    assert not any(
        "outside the official set" in w for w in result.warnings
    ), result.warnings


def test_speaker_ids_are_stripped_from_retention_analysis():
    """"Do not write (Sx) in retention_analysis" - guide section 5.4."""
    written = SPEC.replace(
        "<Subject 1> (appears in [Shot 1], [Shot 2]):",
        "<Subject 1> (S1) (appears in [Shot 1], [Shot 2]):",
    )
    result = process(written, CTX)
    retention = _sections(result.prompt)["retention_analysis"]
    assert "(S1)" not in retention, retention
    assert any("(Sx)" in f for f in result.applied_fixes)


# ---------------------------------------------------------------------------
# Shot labels: guide_ref_en section 5.3, guide_base_en section 4.2
# ---------------------------------------------------------------------------

def test_shot_labels_use_the_official_form():
    """"[Shot 1] marks the opening shot and has no timestamp. Later
    shots use [Shot N] At MM:SS.mmm, ..."."""
    detailed = _sections(process(SPEC, CTX).prompt)["detailed_description"]
    assert "[Shot 1]" in detailed
    assert not re.search(r"\[Shot 1\]\s+At", detailed)
    for found in re.finditer(r"\[Shot (\d+)\]([^\[]*)", detailed):
        if found.group(1) == "1":
            continue
        assert re.match(
            r"\s+At \d{2}:\d{2}\.\d{3},", found.group(2)
        ), found.group(0)[:60]


# ---------------------------------------------------------------------------
# The alignment instruction: guide_base_en section 2.1
# ---------------------------------------------------------------------------

def test_zero_seconds_is_the_i2va_sentence_verbatim():
    """I2VA "always uses" this exact sentence."""
    assert prompts.build_alignment_hook(0.0, 1) == (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )


def test_any_other_moment_is_the_l2va_sentence():
    """L2VA is the form the guide parameterizes: "<Picture 1> (from
    [Shot N]) aligns with the S.SS-second mark of the target video"."""
    hook = prompts.build_alignment_hook(1.3, 3)
    assert hook == (
        "How the reference pictures align with the target video - "
        "<Picture 1> (from [Shot 3]) aligns with the 1.30-second mark "
        "of the target video."
    )
    # Not the I2VA wording with a substituted time - that is in no form.
    assert "is fully referenced" not in hook


def test_the_instruction_is_the_first_line_then_a_blank_one():
    """"The instruction must be the first line of the final prompt,
    followed by one blank line before the core fields"."""
    hook = prompts.build_alignment_hook(0.0, 1)
    prompt = process(SPEC, replace(CTX, alignment_hook=hook)).prompt
    assert prompt.startswith(hook + "\n\nsubject_definitions:")


# ---------------------------------------------------------------------------
# Audio sections: guide_base_en sections 5.2 and 5.3
# ---------------------------------------------------------------------------

def test_na_is_reserved_for_genuine_silence():
    """"Use N/A when there is no non-diegetic music." A trim must not
    write it to save characters - that asserts a silence that is false."""
    bloated = SPEC.replace(
        "non_diegetic_music:\nN/A",
        "non_diegetic_music:\n" + ("A slow piano figure. " * 400),
    )
    result = process(bloated, CTX)
    music = _sections(result.prompt)["non_diegetic_music"]
    assert not music.upper().startswith("N/A"), music[:80]
    assert music.startswith("A slow piano figure."), music[:80]


# ---------------------------------------------------------------------------
# Reference tags: guide_ref_en section 2
# ---------------------------------------------------------------------------

def test_only_the_four_official_tag_names_are_emitted():
    prompt = process(SPEC, CTX).prompt
    for tag in set(re.findall(r"<([A-Za-z]+) \d+>", prompt)):
        assert tag in ("Subject", "Picture", "Video", "Audio"), tag


def test_miscased_and_bracketed_tags_are_canonicalized():
    written = SPEC.replace("<Picture 1>", "[Image 1]").replace(
        "<Video 1>", "<video 1>"
    )
    prompt = process(written, CTX).prompt
    assert "[Image 1]" not in prompt
    assert "<video 1>" not in prompt
    assert "<Picture 1>" in prompt and "<Video 1>" in prompt


# ---------------------------------------------------------------------------
# Length rules: guide_ref_en section 5.2
# ---------------------------------------------------------------------------

def test_the_official_word_target_is_350_to_500():
    """"For generation tasks, detailed_description is normally 350-500
    English words." No guide states a character limit, so the node's
    own cap must not be described to the user as one."""
    assert prompts.DETAILED_DESCRIPTION_WORDS == (350, 500)
    bloated = SPEC.replace(
        "non_diegetic_music:\nN/A",
        "non_diegetic_music:\n" + ("A slow piano figure. " * 400),
    )
    result = process(bloated, CTX)
    assert not any("H3 limit" in w for w in result.warnings), result.warnings


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All H3 format conformance tests passed.")
