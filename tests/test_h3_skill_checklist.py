"""
Checklist validator tests.

The PASS fixtures are MiniMax's own worked examples (via h3_cowboy/spec)
- if the validator rejects an official example, the validator is wrong.
Each FAIL fixture mutates one rule so every test fails for one reason.

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_skill_checklist.py
"""

import os
import sys
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("utils", "utils.h3_prompt", "utils.h3_cowboy", "utils.h3_skill"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.utils.h3_cowboy import spec as cowboy_spec  # noqa: E402
from TrentNodes.utils.h3_cowboy.spec import (  # noqa: E402
    EXAMPLE_BASE_DURATION,
    EXAMPLE_FOR_BASE_MODE,
    EXAMPLE_REF_EDITING,
    EXAMPLE_REF_GENERATION,
    INSTRUCTION_I2VA,
)
from TrentNodes.utils.h3_skill.checklist import (  # noqa: E402
    assemble_final,
    final_shot_index,
    validate,
)
from TrentNodes.utils.h3_skill.skill_loader import (  # noqa: E402
    COWBOY_MODE,
    build_system_prompt,
)

_BASE_MODES = ("t2va", "i2va", "fl2va", "l2va")


def _base_body(mode):
    """Official base example with its alignment line stripped (the
    pipeline renders that line; the model never writes it)."""
    example = EXAMPLE_FOR_BASE_MODE[COWBOY_MODE[mode]].strip()
    if example.startswith("integrated_multimodal_description:"):
        return example
    _, _, rest = example.partition("\n\n")
    return rest.strip()


# ------------------------------------------------------------------ PASS

def test_official_ref_example_passes():
    errors = validate(EXAMPLE_REF_GENERATION.strip(), "ref2va")
    assert errors == [], f"official ref example must pass, got: {errors}"


def test_official_ref_editing_example_passes():
    errors = validate(EXAMPLE_REF_EDITING.strip(), "ref2va")
    assert errors == [], f"official editing example must pass, got: {errors}"


def test_official_base_examples_pass():
    for mode in _BASE_MODES:
        errors = validate(
            _base_body(mode), mode, EXAMPLE_BASE_DURATION[COWBOY_MODE[mode]] + 60
        )
        assert errors == [], f"official {mode} example must pass, got: {errors}"


# ------------------------------------------------------------------ FAIL (ref)

def test_ref_rejects_reordered_headers():
    text = EXAMPLE_REF_GENERATION.replace(
        "summary:", "TEMP:"
    ).replace("retention_analysis:", "summary:").replace(
        "TEMP:", "retention_analysis:"
    )
    assert any("checklist 1" in e for e in validate(text, "ref2va"))


def test_ref_rejects_cut_to():
    text = EXAMPLE_REF_GENERATION.replace("the shot cuts to", "CUT TO:")
    assert text != EXAMPLE_REF_GENERATION, "mutation must actually apply"
    errors = validate(text, "ref2va")
    assert any("CUT TO" in e or "cut phrases" in e for e in errors)


def test_ref_rejects_timed_shot_one():
    # Mutate inside detailed_description - retention scopes also say
    # "[Shot 1]" and sit earlier in the text.
    head, sep, tail = EXAMPLE_REF_GENERATION.partition("detailed_description:")
    text = head + sep + tail.replace("[Shot 1]", "[Shot 1] At 00:00.000,", 1)
    assert text != EXAMPLE_REF_GENERATION, "mutation must actually apply"
    assert any("no timestamp" in e for e in validate(text, "ref2va"))


def test_ref_rejects_mixed_retention_marker():
    text = EXAMPLE_REF_GENERATION.replace(
        ": fully_preserved -", ": fully_copy -", 1
    )
    assert any("marker" in e for e in validate(text, "ref2va"))


def test_ref_rejects_illegal_task_type():
    text = EXAMPLE_REF_GENERATION.replace(
        "reference generation", "style transfer", 1
    )
    assert text != EXAMPLE_REF_GENERATION, "mutation must actually apply"
    assert any("task type" in e for e in validate(text, "ref2va"))


def test_ref_rejects_trailing_exclusion_list():
    text = EXAMPLE_REF_GENERATION.strip() + (
        "\nNo watermarks, no subtitles, no extra characters."
    )
    assert any("exclusion" in e or "positive" in e for e in validate(text, "ref2va"))


def test_ref_rejects_markdown_fence():
    text = "```\n" + EXAMPLE_REF_GENERATION.strip() + "\n```"
    assert any("fence" in e for e in validate(text, "ref2va"))


def test_ref_rejects_timestamp_outside_duration():
    errors = validate(EXAMPLE_REF_GENERATION.strip(), "ref2va", duration_s=2.0)
    assert any("outside" in e for e in errors)


def test_rejects_non_fixed_camera_phrases():
    # Inject a non-fixed qualifier into the style sentence position.
    text = EXAMPLE_REF_GENERATION.replace(
        "detailed_description:\n",
        "detailed_description:\nThe camera work pushes in with medium "
        "amplitude at slow speed.\n",
        1,
    )
    assert text != EXAMPLE_REF_GENERATION, "mutation must actually apply"
    assert any("checklist 7" in e for e in validate(text, "ref2va"))


def test_ref_rejects_missing_editing_opener():
    text = EXAMPLE_REF_EDITING.replace(
        "The target video is an edited version of <Video 1>.", "", 1
    )
    assert text != EXAMPLE_REF_EDITING, "mutation must actually apply"
    assert any("editing summary" in e for e in validate(text, "ref2va"))


def test_ref_rejects_scopeless_visual_retention():
    import re as _re
    text = _re.sub(
        r"<Subject 1> \([^)]*\): fully_preserved",
        "<Subject 1>: fully_preserved",
        EXAMPLE_REF_GENERATION, count=1,
    )
    assert text != EXAMPLE_REF_GENERATION, "mutation must actually apply"
    assert any("scope" in e for e in validate(text, "ref2va"))


def test_base_rejects_missing_blank_line():
    body = _base_body("t2va").replace("\n\noverall_soundscape:",
                                      "\noverall_soundscape:", 1)
    errors = validate(body, "t2va", 60)
    assert any("blank line" in e for e in errors)


# ------------------------------------------------------------------ FAIL (base)

def test_base_rejects_next_line_content():
    body = _base_body("t2va").replace(
        "integrated_multimodal_description: ",
        "integrated_multimodal_description:\n",
        1,
    )
    errors = validate(body, "t2va", 60)
    assert any("SAME line" in e or "same line" in e for e in errors)


def test_base_rejects_ref_constructs():
    body = "subject_definitions:\n<Subject 1> is x.\n\n" + _base_body("t2va")
    errors = validate(body, "t2va", 60)
    assert any("ref sections" in e or "start with" in e for e in errors)


def test_fl2va_rejects_angle_bracket_pictures():
    body = _base_body("fl2va").replace("Picture 1", "<Picture 1>", 1)
    errors = validate(body, "fl2va", 60)
    assert any("BARE" in e for e in errors)


def test_base_rejects_model_written_alignment_line():
    example = EXAMPLE_FOR_BASE_MODE["base_FL2VA"].strip()  # includes the line
    errors = validate(example, "fl2va", 60)
    assert any("alignment" in e for e in errors)


# ------------------------------------------------------------------ assembly

def test_assemble_final_i2va_prepends_verbatim_line():
    body = _base_body("i2va")
    final = assemble_final(body, "i2va", 5.0)
    assert final.startswith(INSTRUCTION_I2VA + "\n\n")
    assert final.endswith(body)


def test_assemble_final_t2va_and_ref_unchanged():
    body = _base_body("t2va")
    assert assemble_final(body, "t2va", 8.0) == body
    ref = EXAMPLE_REF_GENERATION.strip()
    assert assemble_final(ref, "ref2va", 8.0) == ref


def test_assemble_final_fl2va_renders_shot_and_duration():
    body = _base_body("fl2va")
    final = assemble_final(body, "fl2va", 8.0)
    expected = cowboy_spec.render_instruction_line(
        "base_FL2VA", final_shot=final_shot_index(body), duration_seconds=8.0
    )
    assert final.startswith(expected + "\n\n")
    assert "—" in final.splitlines()[0]  # em dash, not hyphen


def test_final_shot_index():
    assert final_shot_index("[Shot 1] a [Shot 2] At 00:02.000, b") == 2
    assert final_shot_index("no shots here") == 1


# ------------------------------------------------------------------ system prompt

def test_system_prompt_embeds_skill_and_contract():
    prompt, source = build_system_prompt("ref2va")
    assert "subject_definitions:" in prompt
    assert "MiniMax H3 prompting" in prompt          # skill doc title
    assert source in ("live skill file", "vendored snapshot")
    base_prompt, _ = build_system_prompt("fl2va")
    assert "integrated_multimodal_description:" in base_prompt
    assert "Do NOT write the picture-alignment" in base_prompt


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checklist tests passed")
