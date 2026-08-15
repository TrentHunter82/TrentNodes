"""
End-to-end tests for the Ultimate H3 Cowboy Promptor node, against a
fake backend so no API call is spent.

Deliberately does NOT import the older node's CANNED fixture. Three files
already share that one, and coupling a fourth makes it progressively
harder to change safely.

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_cowboy_node.py
"""

import json
import os
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

from TrentNodes.nodes import ultimate_h3_cowboy_promptor as node_mod  # noqa: E402
from TrentNodes.utils.h3_prompt.backends import VLMResult  # noqa: E402

REPLY = """subject_definitions:
<Subject 1> is the courier in <Picture 1>, with short dark hair, an olive rain jacket, and a canvas satchel.
<Subject 2> is the loading bay in <Picture 2>, featuring wet concrete, sodium toplight from above, and a corrugated roller shutter.

summary:
[reference generation] The target video follows <Subject 1> as the courier crosses <Subject 2> and ducks under the shutter.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the short dark hair, olive rain jacket and canvas satchel are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the wet concrete, sodium toplight and roller shutter are retained.

detailed_description:
The target video has a grounded handheld documentary look with hard sodium toplight.
[Shot 1] A tracking shot moves backward with small amplitude at slow speed, holding <Subject 1> center frame as the courier crosses <Subject 2> toward the shutter.
[Shot 2] At 00:02.500, the camera cuts to a low static wide shot; <Subject 1> occupies the left third of frame and ducks under the roller shutter.

overall_soundscape:
Rain drumming on corrugated steel, with boot scuffs on wet concrete and the rattle of the shutter chain.

non_diegetic_music:
N/A"""


class FakeBackend:
    name = "fake"
    supports_audio = False
    supports_video = False

    def __init__(self, reply=REPLY):
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


SUBJECTS = (
    "person the courier   -- short dark hair, olive rain jacket, canvas satchel\n"
    "scene  the loading bay -- wet concrete, sodium toplight, roller shutter"
)


def _run(fake, **overrides):
    original = node_mod.get_backend
    node_mod.get_backend = lambda *a, **k: fake
    try:
        kwargs = dict(
            h3_mode="ref",
            subjects=SUBJECTS,
            target_description="the courier ducks under a roller shutter",
            vlm_provider="anthropic",
            model="auto",
            frames=torch.rand((24, 64, 64, 3)),
            fps=12.0,
            subject_1_image=torch.rand((1, 96, 96, 3)),
            subject_2_image=torch.rand((1, 96, 96, 3)),
        )
        kwargs.update(overrides)
        return node_mod.UltimateH3CowboyPromptor().generate(**kwargs)
    finally:
        node_mod.get_backend = original


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_node_end_to_end():
    fake = FakeBackend()
    prompt, duration, fps, analysis_json, checkpoint = _run(fake)[:5]
    assert fake.calls == 1, "a clean reply must not trigger a retry"
    assert prompt.startswith("subject_definitions:")
    assert prompt.rstrip().endswith("N/A")
    # 24 frames at 12 fps is 2.00s, which H3 renders as 56 frames.
    assert duration == 2.333 and fps == 12
    assert checkpoint == "MiniMax-H3-Base-Ref2VA"

    analysis = json.loads(analysis_json)
    assert analysis["mode"] == "ref"
    assert analysis["subject_kinds"] == ["person", "scene"]
    assert analysis["unresolved_errors"] == []
    assert analysis["image_tags"] == ["Picture 1", "Picture 2"]


def test_outputs_match_return_names():
    from TrentNodes.nodes.ultimate_h3_cowboy_promptor import (
        UltimateH3CowboyPromptor as N,
    )
    out = _run(FakeBackend())
    assert len(out) == len(N.RETURN_TYPES) == len(N.RETURN_NAMES)
    assert len(N.OUTPUT_TOOLTIPS) == len(N.RETURN_NAMES)


def test_the_first_five_outputs_never_move():
    # A saved workflow stores links by output INDEX. Appending is safe;
    # inserting silently re-points every wire after it in every graph
    # anyone has saved. This test is the one that protects them.
    from TrentNodes.nodes.ultimate_h3_cowboy_promptor import (
        UltimateH3CowboyPromptor as N,
    )
    assert N.RETURN_NAMES[:5] == (
        "h3_prompt", "duration_seconds", "fps", "analysis_json",
        "h3_checkpoint_hint",
    )
    assert N.RETURN_TYPES[:5] == ("STRING", "FLOAT", "INT", "STRING", "STRING")


def test_both_modes_return_the_same_shape():
    # ComfyUI reads outputs by position, so a divergence here breaks the
    # node the moment h3_mode changes. It must fail in the tests, not on
    # the canvas.
    ref = _run(FakeBackend())
    base = _run(FakeBackend(), h3_mode="base_T2VA")
    assert len(ref) == len(base)


def test_the_node_is_registered_under_its_trent_key():
    assert (
        "TrentUltimateH3CowboyPromptor" in node_mod.NODE_CLASS_MAPPINGS
    )
    assert node_mod.NODE_DISPLAY_NAME_MAPPINGS[
        "TrentUltimateH3CowboyPromptor"
    ] == "Ultimate H3 Cowboy Promptor"
    assert node_mod.UltimateH3CowboyPromptor.CATEGORY == "Trent/VLM"


def test_the_old_node_is_untouched_and_still_registered():
    # Both are installed on purpose; this one must not displace it.
    from TrentNodes.nodes import h3_auto_prompt as old
    assert "TrentH3AutoPromptGenerator" in old.NODE_CLASS_MAPPINGS
    assert old.H3AutoPromptGenerator.CATEGORY == "Trent/VLM"


def test_six_subject_image_slots_exist_and_are_optional():
    from TrentNodes.nodes.ultimate_h3_cowboy_promptor import (
        NUM_SUBJECT_SLOTS, UltimateH3CowboyPromptor as N,
    )
    optional = N.INPUT_TYPES()["optional"]
    for slot in range(1, NUM_SUBJECT_SLOTS + 1):
        assert optional[f"subject_{slot}_image"][0] == "IMAGE"
    assert NUM_SUBJECT_SLOTS == 6


# ---------------------------------------------------------------------------
# Subjects reach the model
# ---------------------------------------------------------------------------

def test_each_subject_image_is_labelled_with_its_own_picture_tag():
    fake = FakeBackend()
    _run(fake)
    assert fake.image_labels[0].startswith("Reference image <Picture 1>")
    assert "person" in fake.image_labels[0]
    assert fake.image_labels[1].startswith("Reference image <Picture 2>")
    assert "scene" in fake.image_labels[1]


def test_only_declared_kinds_reach_the_system_prompt():
    fake = FakeBackend()
    _run(fake)
    cards = fake.system.split("BY SUBJECT KIND")[1]
    assert "person:" in cards and "scene:" in cards
    assert "animal:" not in cards and "style:" not in cards


def test_a_style_subject_needs_no_image():
    fake = FakeBackend()
    _run(
        fake,
        subjects=SUBJECTS + "\nstyle -- 16mm grain, halation, teal shadows",
        subject_3_image=None,
    )
    assert "style has no position in the frame" in fake.first_user_text


def test_unreadable_subjects_raise_rather_than_guess():
    try:
        _run(FakeBackend(), subjects="Aria Voss | person | wrong syntax")
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "No subjects could be read" in str(exc), exc


def test_base_mode_runs_and_refuses_a_ref_shaped_reply():
    # The two modes are different formats, not two settings of one. This
    # canned reply is correct ref output and wrong for base mode, so the
    # base validator has to reject it rather than pass it through.
    # tests/test_h3_cowboy_base.py covers base mode properly.
    fake = FakeBackend()
    _p, _d, _f, analysis_json, checkpoint = _run(
        fake, h3_mode="base_T2VA"
    )[:5]
    assert checkpoint == "MiniMax-H3-Base-FL2VA"
    errors = json.loads(analysis_json)["unresolved_errors"]
    assert any("R1 FORMAT" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Validation reaches the retry loop
# ---------------------------------------------------------------------------

def test_a_dangling_subject_forces_exactly_one_retry():
    broken = REPLY.replace(
        "<Subject 2> is the loading bay in <Picture 2>, featuring wet "
        "concrete, sodium toplight from above, and a corrugated roller "
        "shutter.\n", ""
    )
    fake = FakeBackend(reply=broken)
    _p, _d, _f, analysis_json, _c = _run(fake)[:5]
    assert fake.calls == 2, "one retry, not more"
    analysis = json.loads(analysis_json)
    assert any("R2 LABELS" in e for e in analysis["unresolved_errors"])
    # The retry must carry the errors back to the model.
    assert "R2 LABELS" in fake.last_user_text


def test_the_better_attempt_wins():
    # A worse retry must not overwrite a good first pass.
    class Flaky(FakeBackend):
        def generate(self, *a, **k):
            self.calls += 1
            text = REPLY if self.calls == 1 else "total nonsense"
            return VLMResult(text=text, usage={})

    fake = Flaky()
    prompt, _d, _f, analysis_json, _c = _run(fake)[:5]
    assert prompt.startswith("subject_definitions:")
    assert json.loads(analysis_json)["unresolved_errors"] == []


def test_cut_times_pin_the_shot_timeline():
    fake = FakeBackend()
    _p, _d, _f, analysis_json, _c = _run(fake, cut_times="0.0, 1.5")[:5]
    analysis = json.loads(analysis_json)
    assert analysis["cut_source"] == "measured"
    assert analysis["cut_timestamps"] == [0.0, 1.5]
    assert "MEASURED SHOT LIST" in fake.first_user_text


def test_a_shot_count_mismatch_is_not_an_error_outside_editing():
    # The reply writes 2 shots; the measured list says 3. Outside an
    # edit the target's structure is not bound to the reference's.
    fake = FakeBackend()
    _p, _d, _f, analysis_json, _c = _run(fake, cut_times="0.0, 0.8, 1.6")[:5]
    assert fake.calls == 1
    assert json.loads(analysis_json)["unresolved_errors"] == []


def test_supplied_dialogue_must_reach_a_d_block():
    fake = FakeBackend()
    _run(fake, dialogue="Mind the shutter.")
    assert fake.calls == 2
    assert "R3 VERBATIM" in fake.last_user_text


def test_constraint_notes_are_folded_in_positively():
    fake = FakeBackend()
    _run(fake, constraint_notes="the satchel must not change")
    assert "the satchel must not change" in fake.first_user_text
    assert "positive assertion" in fake.first_user_text
    # And never as a trailing block, which the official format forbids.
    assert "Never write a trailing list" in fake.first_user_text


# ---------------------------------------------------------------------------
# Pass-through: the assets come back out, so nothing is wired twice
# ---------------------------------------------------------------------------

def _named(out) -> dict:
    """The return tuple by output name, which is how a graph reads it."""
    from TrentNodes.nodes.ultimate_h3_cowboy_promptor import (
        UltimateH3CowboyPromptor as N,
    )
    return dict(zip(N.RETURN_NAMES, out))


class FakeVideo:
    """The little of ComfyUI's VIDEO the node touches."""

    def __init__(self, images, frame_rate=24.0):
        self.images = images
        self.frame_rate = frame_rate

    def get_components(self):
        return self


def test_what_goes_in_comes_out_by_identity():
    picture = torch.rand((1, 96, 96, 3))
    out = _named(_run(
        FakeBackend(),
        subjects=SUBJECTS + "\nobject the satchel -- scuffed canvas",
        subject_3_image=picture,
    ))
    assert out["ref_image_3"] is picture, "the tensor itself, untouched"


def test_a_gap_passes_through_untouched_and_says_what_it_costs():
    # Compacting here would BE the bug: the prompt says <Picture 3>, so
    # slot 3 has to leave the node as ref_image_3.
    picture = torch.rand((1, 96, 96, 3))
    out = _named(_run(
        FakeBackend(),
        subjects=SUBJECTS + "\nobject the satchel -- scuffed canvas",
        subject_2_image=None, subject_3_image=picture,
    ))
    assert out["ref_image_2"] is None
    assert out["ref_image_3"] is picture

    warnings = " ".join(json.loads(out["analysis_json"])["warnings"])
    assert "every tag after the gap shifts down" in warnings
    # And the map shows the shift the sampler will actually apply.
    assert "<Picture 2> = subject_3_image" in out["label_map"]


def test_the_audio_reaches_both_audio_outputs():
    # Which one to connect depends on audio_role, and the node does not
    # guess: it offers both and the graph picks one.
    audio = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}
    out = _named(_run(FakeBackend(), audio=audio))
    assert out["ref_video_audio"] is audio
    assert out["ref_audio"] is audio


def test_a_wired_video_comes_out_as_frames():
    # The promptor takes a VIDEO; the sampler's ref_video_ socket takes
    # IMAGE. The conversion happens here or it happens by hand.
    images = torch.rand((48, 64, 64, 3))
    out = _named(_run(FakeBackend(), video=FakeVideo(images), frames=None))
    assert out["ref_video"] is images
    assert not hasattr(out["ref_video"], "get_components")


def test_the_canvas_and_length_are_the_samplers_own_numbers():
    out = _named(_run(
        FakeBackend(), frames=torch.rand((24, 64, 128, 3)), fps=12.0,
    ))
    # 128x64 is 2:1: 768 on the short edge would be 1536x768, which is
    # over the sampler's pixel cap, so it comes back scaled to /32.
    assert (out["width"], out["height"]) == (1440, 704)
    assert out["length"] == 56 and out["length"] % 17 == 5


def test_an_off_speed_clip_says_the_reference_will_play_wrong():
    fake = FakeBackend()
    out = _named(_run(fake, frames=torch.rand((60, 64, 64, 3)), fps=30.0))
    warnings = " ".join(json.loads(out["analysis_json"])["warnings"])
    assert "reads reference frames as 24 fps" in warnings
    # 23.976 material drifts by a frame in a thousand and must stay quiet.
    out = _named(_run(
        FakeBackend(), frames=torch.rand((48, 64, 64, 3)), fps=23.976,
    ))
    quiet = " ".join(json.loads(out["analysis_json"])["warnings"])
    assert "24 fps" not in quiet


def test_the_label_map_reads_in_the_samplers_presentation_order():
    audio = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}
    out = _named(_run(FakeBackend(), audio=audio))
    assert out["label_map"].splitlines() == [
        "<Picture 1> = subject_1_image",
        "<Picture 2> = subject_2_image",
        "<Audio 1> = audio (as the video's soundtrack)",
        "<Video 1> = video",
    ]


# ---------------------------------------------------------------------------
# The subject rows: the normal way in
# ---------------------------------------------------------------------------

ROWS = dict(
    subjects="",
    subject_1_kind="character",
    subject_1_name="the courier",
    subject_1_description="short dark hair, olive rain jacket, canvas satchel",
    subject_2_kind="environment",
    subject_2_name="the loading bay",
    subject_2_description="wet concrete, sodium toplight, roller shutter",
)


def test_rows_alone_write_the_prompt():
    # No typed subjects field at all - the rows are the whole input.
    fake = FakeBackend()
    out = _named(_run(fake, **ROWS))
    analysis = json.loads(out["analysis_json"])
    assert analysis["subject_kinds"] == ["person", "scene"]
    assert analysis["image_tags"] == ["Picture 1", "Picture 2"]
    assert "the courier" in fake.first_user_text
    assert "wet concrete" in fake.first_user_text
    # The face says character; the prompt says person, which is the word
    # every kind card is written against.
    assert "person:" in fake.system.split("BY SUBJECT KIND")[1]
    # And no noise about rows nobody hid.
    assert not any("subject_rows is" in w for w in analysis["warnings"])


def test_a_filled_row_past_the_visible_count_still_counts():
    # Only reachable from the API or a failed extension, and a subject
    # that silently stops existing is the worse of the two outcomes.
    fake = FakeBackend()
    out = _named(_run(fake, **ROWS, subject_rows=1))
    warnings = " ".join(json.loads(out["analysis_json"])["warnings"])
    assert "row 2 holds text and was used anyway" in warnings
    assert "the loading bay" in fake.first_user_text


def test_a_row_with_no_image_is_still_a_subject():
    fake = FakeBackend()
    _run(
        fake, **ROWS,
        subject_3_kind="style",
        subject_3_description="16mm grain, halation, teal shadows",
    )
    assert "16mm grain" in fake.first_user_text
    assert "style has no position in the frame" in fake.first_user_text


def test_a_row_keeps_its_number_when_an_earlier_row_is_empty():
    fake = FakeBackend()
    out = _named(_run(
        fake,
        subjects="",
        subject_1_kind="character",
        subject_1_name="the courier",
        subject_1_description="olive rain jacket",
        subject_3_kind="object",
        subject_3_description="scuffed canvas satchel",
        subject_2_image=None,
        subject_3_image=torch.rand((1, 96, 96, 3)),
    ))
    analysis = json.loads(out["analysis_json"])
    assert analysis["image_tags"] == ["Picture 1", "Picture 3"]
    assert any("<Subject 3>" in s for s in analysis["subjects"])
    warnings = " ".join(analysis["warnings"])
    assert "gap" in warnings


def test_an_empty_node_says_where_to_type():
    try:
        _run(FakeBackend(), subjects="")
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "subject_1_description" in str(exc), exc


# ---------------------------------------------------------------------------
# Music video
# ---------------------------------------------------------------------------

MUSIC_REPLY = REPLY.replace(
    "non_diegetic_music:\nN/A",
    "non_diegetic_music:\nDowntempo synthwave at roughly 92 BPM, analog pad "
    "and gated drums. The filter opens into the chorus on the second cut.",
)


def test_music_video_inverts_the_audio_balance():
    fake = FakeBackend(reply=MUSIC_REPLY)
    out = _named(_run(
        fake, **ROWS, music_video=True,
        music_description="downtempo synthwave, ~92 BPM",
    ))
    assert "MUSIC VIDEO MODE IS ON" in fake.first_user_text
    assert "downtempo synthwave, ~92 BPM" in fake.first_user_text
    assert json.loads(out["analysis_json"])["unresolved_errors"] == []


def test_a_music_video_scored_n_a_is_a_retry():
    # The canned reply ends in "non_diegetic_music: N/A", which is a
    # music video with no music.
    fake = FakeBackend()
    out = _named(_run(fake, **ROWS, music_video=True))
    errors = json.loads(out["analysis_json"])["unresolved_errors"]
    assert any("R6 MUSIC" in e for e in errors), errors
    assert fake.calls == 2, "one retry, carrying the error back"


def test_a_reused_song_reaches_the_task_type():
    audio = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}
    fake = FakeBackend(reply=MUSIC_REPLY)
    out = _named(_run(fake, **ROWS, music_video=True, audio=audio))
    analysis = json.loads(out["analysis_json"])
    assert analysis["music_is_reference"] is True
    assert "audio reuse" in analysis["task_type"]
    assert "SUPPLIED TO THE GENERATION AS <Audio 1>" in fake.first_user_text


def test_a_generated_score_is_not_audio_reuse():
    audio = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}
    fake = FakeBackend(reply=MUSIC_REPLY)
    out = _named(_run(
        fake, **ROWS, music_video=True, audio=audio,
        music_source="generate_score",
    ))
    analysis = json.loads(out["analysis_json"])
    assert analysis["music_is_reference"] is False
    assert "audio reuse" not in analysis["task_type"]


def test_supplied_lyrics_must_reach_a_d_block():
    fake = FakeBackend(reply=MUSIC_REPLY)
    out = _named(_run(
        fake, **ROWS, music_video=True, lyrics="Hold the line, hold it."
    ))
    assert fake.calls == 2
    errors = json.loads(out["analysis_json"])["unresolved_errors"]
    assert any("R3 VERBATIM" in e for e in errors), errors


def test_lyrics_without_music_video_warn_instead_of_leaking():
    fake = FakeBackend()
    out = _named(_run(fake, **ROWS, lyrics="Hold the line."))
    assert "MUSIC VIDEO MODE" not in fake.first_user_text
    assert any(
        "music_video is off" in w
        for w in json.loads(out["analysis_json"])["warnings"]
    )


def test_base_mode_says_the_rows_were_ignored():
    fake = FakeBackend()
    out = _named(_run(fake, **ROWS, h3_mode="base_T2VA", music_video=True))
    warnings = " ".join(json.loads(out["analysis_json"])["warnings"])
    assert "subject row 1, 2" in warnings
    assert "music_video is a ref-mode setting" in warnings


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All cowboy node tests passed.")
