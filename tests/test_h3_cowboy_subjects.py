"""
Unit tests for the Ultimate H3 Cowboy Promptor subject DSL.

Pure string and dataclass work - no torch, no ComfyUI. Run from the
ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_cowboy_subjects.py
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
    for sub in ("nodes", "utils", "utils.h3_prompt", "utils.h3_cowboy"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.utils.h3_cowboy import spec  # noqa: E402
from TrentNodes.utils.h3_cowboy.subjects import (  # noqa: E402
    SubjectRow,
    bind_images,
    declared_kinds,
    merge_text_subjects,
    parse_subjects,
    subjects_from_rows,
)


def _parse(text):
    warnings = []
    return parse_subjects(text, warnings), warnings


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_a_full_line_parses_into_every_field():
    subs, warns = _parse(
        "person Aria Voss @Picture 1 -- dark hair, charcoal jacket"
    )
    assert not warns, warns
    (s,) = subs
    assert (s.index, s.kind, s.name) == (1, "person", "Aria Voss")
    assert s.sources == ["Picture 1"]
    assert s.features == "dark hair, charcoal jacket"


def test_subjects_number_by_declaration_order():
    subs, _w = _parse(
        "person Aria -- hair\nscene the bar -- brick\nstyle -- grain"
    )
    assert [s.index for s in subs] == [1, 2, 3]
    assert [s.tag for s in subs] == [
        "<Subject 1>", "<Subject 2>", "<Subject 3>"
    ]


def test_a_name_is_optional():
    # The guide's own example names no subject: they are "the young
    # blonde woman", not "Aria".
    subs, warns = _parse("scene -- exposed brick wall, neon sign")
    assert not warns, warns
    assert subs[0].name == ""
    assert subs[0].kind == "scene"


def test_a_subject_may_cite_several_sources():
    subs, _w = _parse(
        "animal the Samoyed @Picture 2 @Picture 3 @Picture 4 -- white fur"
    )
    assert subs[0].sources == ["Picture 2", "Picture 3", "Picture 4"]
    assert subs[0].source_phrase() == "<Picture 2>, <Picture 3>, <Picture 4>"


def test_source_spelling_is_tolerant():
    subs, _w = _parse("person A @picture1 @Image 2 @VIDEO 3 -- hair")
    assert subs[0].sources == ["Picture 1", "Picture 2", "Video 3"]


def test_a_description_only_subject_is_legal():
    # The only way to express a style or an action, which often has no
    # image at all.
    subs, warns = _parse("style -- 16mm grain, halation, teal shadows")
    assert not warns, warns
    assert subs[0].sources == []
    assert subs[0].features


def test_blank_lines_and_comments_are_skipped():
    subs, warns = _parse("\n# a note\n\nperson A -- hair\n\n")
    assert len(subs) == 1 and not warns


def test_an_unknown_kind_is_skipped_with_a_useful_warning():
    # Guessing a kind would inject the wrong feature card, and the model
    # would follow it. Skipping plus a precise warning is more honest.
    subs, warns = _parse("Aria Voss | person | wrong syntax")
    assert subs == []
    assert any("not a subject kind" in w and "Aria" in w for w in warns), warns


def test_a_hyphenated_feature_does_not_split_the_line():
    subs, _w = _parse("style -- teal-shifted shadows, hand-held feel")
    assert subs[0].features == "teal-shifted shadows, hand-held feel"


def test_a_subject_with_nothing_to_go_on_warns():
    _subs, warns = _parse("object")
    assert any("no features and no source" in w for w in warns), warns


def test_declared_kinds_are_deduped_in_order():
    subs, _w = _parse(
        "person A -- x\nscene B -- y\nperson C -- z\nstyle -- w"
    )
    assert declared_kinds(subs) == ["person", "scene", "style"]


def test_spatial_kinds_match_the_spec_list():
    subs, _w = _parse(
        "person A -- x\nscene B -- y\nstyle -- z\naction C -- w\nobject D -- v"
    )
    assert [s.is_spatial for s in subs] == [True, False, False, False, True]
    assert spec.SPATIAL_KINDS <= set(spec.SUBJECT_KINDS)


# ---------------------------------------------------------------------------
# Image binding: subject_N_image IS <Picture N>
# ---------------------------------------------------------------------------

def test_slot_number_is_the_picture_number():
    subs, _w = _parse("person A -- hair\nscene B -- brick")
    warns = []
    tags = bind_images(subs, [1, 2], warns)
    assert tags == ["Picture 1", "Picture 2"]
    assert subs[0].sources == ["Picture 1"]
    assert subs[1].sources == ["Picture 2"]
    assert not warns, warns


def test_an_explicit_pin_that_matches_its_slot_is_not_duplicated():
    subs, _w = _parse("person A @Picture 1 -- hair\nscene B -- brick")
    warns = []
    bind_images(subs, [1, 2], warns)
    assert subs[0].sources == ["Picture 1"]
    assert not warns, warns


def test_a_gap_in_the_slots_warns_but_keeps_the_numbering():
    # The alternative - renumbering wired slots 1..M - makes an explicit
    # @Picture 1 mean a different image than subject_1_image, which is
    # exactly the collision this rule exists to prevent.
    subs, _w = _parse("person A -- hair\nscene B -- brick")
    warns = []
    tags = bind_images(subs, [2], warns)
    assert tags == ["Picture 2"]
    assert subs[1].sources == ["Picture 2"]
    assert subs[0].sources == []
    assert any("wired with a gap" in w for w in warns), warns


def test_citing_an_unwired_picture_warns():
    subs, _w = _parse("person A @Picture 4 -- hair")
    warns = []
    bind_images(subs, [1], warns)
    assert any("subject_4_image is not wired" in w for w in warns), warns


def test_a_wired_slot_with_no_subject_warns():
    subs, _w = _parse("person A -- hair")
    warns = []
    bind_images(subs, [1, 2], warns)
    assert any("row 2 is empty" in w for w in warns), warns


def test_binding_is_idempotent():
    subs, _w = _parse("person A -- hair")
    warns = []
    bind_images(subs, [1], warns)
    bind_images(subs, [1], warns)
    assert subs[0].sources == ["Picture 1"]


# ---------------------------------------------------------------------------
# The rows on the node face
# ---------------------------------------------------------------------------

def _rows(*specs):
    """(slot, kind, name, description) tuples -> rows, filling the gaps."""
    filled = {s[0]: s for s in specs}
    return [
        SubjectRow(
            slot=slot,
            kind=filled.get(slot, (slot, "character", "", ""))[1],
            name=filled.get(slot, (slot, "character", "", ""))[2],
            description=filled.get(slot, (slot, "character", "", ""))[3],
        )
        for slot in range(1, 7)
    ]


def test_character_and_environment_are_the_guides_own_kinds():
    # The face says character; the prompt has to say person, because
    # that is the word every kind card and rule is written against.
    assert spec.canonical_kind("character") == "person"
    assert spec.canonical_kind("Environment") == "scene"
    assert spec.canonical_kind("person") == "person"
    assert spec.canonical_kind("nonsense") == ""
    # And the typed field accepts the friendly spellings too.
    subs, _w = _parse("character Aria -- dark hair")
    assert subs[0].kind == "person"


def test_a_filled_row_becomes_its_own_numbered_subject():
    warnings = []
    subs = subjects_from_rows(
        _rows((1, "character", "the courier", "olive rain jacket"),
              (2, "environment", "the loading bay", "wet concrete")),
        warnings,
    )
    assert [s.index for s in subs] == [1, 2]
    assert [s.kind for s in subs] == ["person", "scene"]
    assert subs[0].name == "the courier"
    assert subs[0].features == "olive rain jacket"
    assert warnings == []


def test_an_empty_row_is_not_a_subject():
    # Which is what lets the row count sit wherever it likes, and what
    # keeps a workflow saved before rows existed behaving as it did.
    warnings = []
    assert subjects_from_rows(_rows(), warnings) == []
    assert warnings == []


def test_a_row_keeps_its_number_so_the_picture_numbers_hold():
    # Row 3 is <Subject 3> even with rows 1 and 2 empty, because
    # subject_3_image is <Picture 3>. Closing the gap here would point
    # the prompt at a different image.
    warnings = []
    subs = subjects_from_rows(
        _rows((3, "object", "the satchel", "scuffed canvas")), warnings
    )
    assert [s.index for s in subs] == [3]
    assert any("gap" in w for w in warnings), warnings
    bind_images(subs, [3], warnings)
    assert subs[0].sources == ["Picture 3"]


def test_an_unreadable_kind_falls_back_and_says_so():
    warnings = []
    subs = subjects_from_rows(
        _rows((1, "spaceship", "", "chrome hull")), warnings
    )
    assert subs[0].kind == "person"
    assert any("not a kind" in w for w in warnings), warnings


def test_typed_lines_are_numbered_after_the_rows():
    warnings = []
    rows = subjects_from_rows(
        _rows((1, "character", "the courier", "olive rain jacket")), warnings
    )
    merged = merge_text_subjects(
        rows, "style -- 16mm grain, halation", warnings
    )
    assert [s.index for s in merged] == [1, 2]
    assert merged[1].kind == "style"
    assert any("typed subject line" in w for w in warnings), warnings


def test_typed_lines_alone_still_work():
    # The escape hatch has to stand on its own - every workflow saved
    # before the rows existed uses only this.
    warnings = []
    merged = merge_text_subjects(
        [], "person A -- hair\nscene B -- concrete", warnings
    )
    assert [s.index for s in merged] == [1, 2]
    assert warnings == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All cowboy subject tests passed.")
