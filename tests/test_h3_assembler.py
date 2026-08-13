"""
Unit tests for the H3 prompt assembler. Pure-string tests, no torch,
no ComfyUI. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_assembler.py
"""

import os
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

from TrentNodes.utils.h3_prompt import assembler  # noqa: E402
from TrentNodes.utils.h3_prompt.assembler import (  # noqa: E402
    AssemblyContext,
    format_shot_time,
    parse_shot_time,
    process,
)
from TrentNodes.utils.h3_prompt.prompts import (  # noqa: E402
    MIN_EXCLUSIONS,
    SECTION_ORDER,
)

CTX = AssemblyContext(
    subject_name="Aria Voss",
    subject_wardrobe=(
        "charcoal utility jacket, slate-gray tee, black cargo pants, "
        "scuffed black combat boots"
    ),
    duration_seconds=6.0,
)


def _excl_ctx(**overrides):
    """
    CTX with the off-spec trailing exclusion block turned on.

    The block is opt-in because no official example writes anything
    after non_diegetic_music, so the tests that exercise it have to ask
    for it. See the prompts module docstring.
    """
    return replace(CTX, append_exclusions=True, **overrides)

GOOD = """subject_definitions:
<Subject 1> is Aria Voss as shown in <Picture 1>. Preserve Aria Voss's exact facial identity, dark hair, skin tone, facial structure, apparent age, body proportions, and overall likeness. <Picture 1> also provides the exact wardrobe: a charcoal utility jacket over a slate-gray tee, black cargo pants, and scuffed black combat boots. Do not copy the background, pose, or lighting from <Picture 1>. <Video 1> supplies only the exact body movement, camera angles, framing, cut rhythm, and overall action. Do not copy any identity or facial features from the performer in <Video 1>.

summary:
Generate a new video in which Aria Voss is continuously tracked through every frame of the warehouse walk from <Video 1>, while fully preserving Aria Voss's identity and wardrobe from <Picture 1>.

retention_analysis:
<Subject 1> Aria Voss: fully_preserved - preserve Aria Voss's exact facial identity, hair, skin tone, body proportions, and likeness from <Picture 1> with zero drift in every frame, and fully preserve the charcoal utility jacket, slate-gray tee, black cargo pants, and combat boots. <Video 1>: attribute_transfer - body movement and camera only.

detailed_description:
The target video has the grounded cinematic appearance of a gritty handheld thriller. [Shot 1] A handheld camera tracks backward, holding Aria Voss <Subject 1> center frame in a medium shot as she strides down a warehouse aisle, boots striking the concrete. [Shot 2] At 00:03.250, cut to a static wide shot; Aria Voss <Subject 1> occupies the left third of frame in the midground, stops beside a pallet stack and turns away toward the loading door; identity and wardrobe remain locked to <Picture 1> while she is back-turned.

overall_soundscape:
Warehouse ambience with a low ventilation hum. Boot footfalls land on Aria Voss's visible steps, jacket cloth movement on each arm swing.

non_diegetic_music:
N/A

No face morphing or identity drift of Aria Voss in any frame. No changes to the core body movement defined by <Video 1>. No alteration of the charcoal utility jacket, slate-gray tee, black cargo pants, or combat boots. No extra characters or props.
"""


def test_good_output_passes():
    result = process(GOOD, CTX)
    assert not result.retry_errors, result.retry_errors
    assert result.prompt.startswith("subject_definitions:")
    for key in SECTION_ORDER:
        assert f"{key}:" in result.prompt, f"missing {key}"
    # The official format ends at non_diegetic_music, so the model's
    # trailing "No ..." block is dropped rather than padded.
    assert result.prompt.rstrip().endswith("N/A"), result.prompt[-120:]
    assert any("dropped" in f and "No ..." in f for f in result.applied_fixes)


def test_exclusions_are_padded_when_asked_for():
    result = process(GOOD, _excl_ctx())
    assert result.prompt.count("No ") >= MIN_EXCLUSIONS
    assert "padded exclusions from stock pool" in result.applied_fixes


def test_markdown_and_preamble_stripped():
    wrapped = (
        "Here is your prompt:\n\n```\n" + GOOD + "\n```\n"
        "Let me know if you need changes!"
    )
    result = process(wrapped, CTX)
    assert not result.retry_errors
    assert result.prompt.startswith("subject_definitions:")
    assert "```" not in result.prompt
    assert "Let me know" not in result.prompt


def test_legacy_uppercase_template_converted():
    legacy = """Create a 6-second, 24 fps, 16:9, LIVE-ACTION video with native synchronized audio.

subject_definitions:
<Subject 1> is Aria Voss as shown in <Picture 1>, wearing the charcoal utility jacket. <Video 1> supplies only motion.

summary:
Generate a new video of Aria Voss from <Video 1> preserving <Picture 1>.

retention_analysis:
<Subject 1> Aria Voss: fully_preserved - zero drift. Preserve the charcoal utility jacket and black cargo pants.

detailed_description:
The target video is a gritty thriller.
[0.000s-3.250s] Aria Voss <Subject 1> strides center frame down a warehouse aisle.
[3.250s-6.000s] Aria Voss <Subject 1> stops in the left third of frame, midground, and turns away.

CAMERA
Matches <Video 1> exactly with continuous tracking on Aria Voss.

DIALOGUE
None.

VISIBLE TEXT
The only visible text is none.

OVERALL SOUNDSCAPE
Warehouse ambience, boot footfalls, cloth movement.

NON-DIEGETIC MUSIC
None.

EXCLUSIONS
No face morphing of Aria Voss. No wardrobe changes. No extra people. No text overlays.
"""
    result = process(legacy, CTX)
    assert not result.retry_errors, result.retry_errors
    # Official casing only
    assert "OVERALL SOUNDSCAPE" not in result.prompt
    assert "NON-DIEGETIC MUSIC" not in result.prompt
    assert "EXCLUSIONS" not in result.prompt
    assert "overall_soundscape:" in result.prompt
    assert "non_diegetic_music:" in result.prompt
    # Legacy ranges became official shot labels
    assert "[Shot 1]" in result.prompt
    assert "[Shot 2] At 00:03.250," in result.prompt
    assert "[0.000s" not in result.prompt
    # Preamble header line dropped
    assert "Create a 6-second" not in result.prompt
    # CAMERA content folded into detailed_description
    assert "Camera: Matches <Video 1>" in result.prompt


def test_bad_shot_times_rescaled():
    bad = GOOD.replace("[Shot 2] At 00:03.250,", "[Shot 2] At 00:59.000,")
    result = process(bad, CTX)
    assert not result.retry_errors
    assert "00:59.000" not in result.prompt
    assert any("rescaled" in w for w in result.warnings), result.warnings


def test_missing_subject_name_is_retry_error():
    broken = GOOD.replace(
        "[Shot 2] At 00:03.250, cut to a static wide shot; Aria Voss "
        "<Subject 1> occupies the left third of frame in the midground,",
        "[Shot 2] At 00:03.250, cut to a static wide shot; the subject "
        "occupies the left third of frame in the midground,",
    )
    result = process(broken, CTX)
    assert any("Shot 2" in e and "Aria Voss" in e for e in result.retry_errors), (
        result.retry_errors
    )


def test_bare_name_gets_tagged():
    untagged = GOOD.replace(
        "holding Aria Voss <Subject 1> center frame",
        "holding Aria Voss center frame",
    )
    result = process(untagged, CTX)
    assert not result.retry_errors, result.retry_errors
    assert any("tagged bare subject name" in f for f in result.applied_fixes)


def test_tag_variants_normalized():
    variant = GOOD.replace("<Picture 1>", "[Image 1]").replace(
        "<Video 1>", "<video 1>"
    )
    result = process(variant, CTX)
    assert "<Picture 1>" in result.prompt
    assert "<Video 1>" in result.prompt
    assert "[Image 1]" not in result.prompt


def test_char_cap_trim_ladder():
    bloated = GOOD.replace(
        "non_diegetic_music:\nN/A",
        "non_diegetic_music:\n" + ("A sweeping orchestral score. " * 300),
    )
    result = process(bloated, CTX)
    assert result.char_count <= 7000, result.char_count
    assert any(
        "shortened non_diegetic_music" in f for f in result.applied_fixes
    ), result.applied_fixes
    # It must shorten the score, not blank it: "N/A" is the guide's
    # value for "there is no non-diegetic music", so writing it to save
    # characters asserts a silence that is not true.
    assert "non_diegetic_music:\nA sweeping orchestral score." in result.prompt


def test_the_trim_ladder_never_blanks_a_music_video_score():
    bloated = GOOD.replace(
        "non_diegetic_music:\nN/A",
        "non_diegetic_music:\n" + ("A driving synth pulse. " * 400),
    )
    result = process(bloated, replace(CTX, music_video=True))
    music = result.prompt.split("non_diegetic_music:\n", 1)[1].strip()
    assert not music.upper().startswith("N/A"), music[:80]
    # And the music-video validator must not fire on text the ladder
    # produced; it now runs after the ladder, on the final wording.
    assert not any("non_diegetic_music is 'N/A'" in e
                   for e in result.retry_errors), result.retry_errors


def test_audio_disabled():
    ctx = AssemblyContext(
        subject_name="Aria Voss",
        subject_wardrobe=CTX.subject_wardrobe,
        duration_seconds=6.0,
        enable_audio_prompt=False,
    )
    result = process(GOOD, ctx)
    assert "Quiet natural ambience only" in result.prompt
    assert "non_diegetic_music:\nN/A" in result.prompt


def test_time_helpers():
    assert format_shot_time(3.25) == "00:03.250"
    assert format_shot_time(83.5) == "01:23.500"
    assert abs(parse_shot_time("01:23.500") - 83.5) < 1e-9
    assert abs(parse_shot_time("3.25s") - 3.25) < 1e-9


def test_empty_output_is_retry_error():
    result = process("", CTX)
    assert result.retry_errors


HOOK = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)


def _aligned_ctx(hook=HOOK, profile="official", append_exclusions=True):
    return AssemblyContext(
        subject_name="Aria Voss",
        subject_wardrobe=CTX.subject_wardrobe,
        duration_seconds=6.0,
        profile=profile,
        alignment_hook=hook,
        append_exclusions=append_exclusions,
    )


def test_alignment_hook_opens_the_prompt():
    result = process(GOOD, _aligned_ctx())
    assert result.prompt.startswith(HOOK)
    # The sections still follow, in order, right after it.
    assert result.prompt.split("\n\n")[1].startswith("subject_definitions:")
    assert result.char_count == len(result.prompt)


def test_no_hook_still_opens_on_subject_definitions():
    result = process(GOOD, CTX)
    assert result.prompt.startswith("subject_definitions:")
    assert "fully referenced" not in result.prompt


CONTRADICTING = (
    " No copying of the background, pose, or lighting from <Picture 1>."
)


def test_alignment_drops_the_contradicting_exclusion():
    written = GOOD.rstrip() + CONTRADICTING

    plain = process(written, _excl_ctx())
    assert "No copying of the background" in plain.prompt

    aligned = process(written, _aligned_ctx())
    assert "No copying of the background" not in aligned.prompt
    assert any(
        "contradicted the first-frame alignment hook" in f
        for f in aligned.applied_fixes
    )
    # Dropping one must not leave the block short; padding refills it.
    exclusions = aligned.prompt.rsplit("\n\n", 1)[1]
    assert exclusions.count("No ") >= MIN_EXCLUSIONS


def test_no_padded_exclusion_ever_contradicts_the_hook():
    # Padding runs deepest when the model writes almost nothing.
    thin = GOOD.rsplit("\n\n", 1)[0] + "\n\nNo extra characters."
    exclusions = process(thin, _aligned_ctx()).prompt.rsplit("\n\n", 1)[1]
    assert exclusions.count("No ") >= MIN_EXCLUSIONS
    for sentence in exclusions.split("No ")[1:]:
        assert not ("<Picture 1>" in sentence and any(
            word in sentence
            for word in ("background", "pose", "lighting", "framing")
        )), sentence


def test_alignment_removes_the_contradicting_prose():
    # GOOD's subject_definitions says "Do not copy the background, pose,
    # or lighting from <Picture 1>." - straight from the system prompt's
    # worked example, and false once the hook declares that frame.
    written = process(GOOD, CTX)
    assert "Do not copy the background, pose, or lighting" in written.prompt

    aligned = process(GOOD, _aligned_ctx())
    assert "Do not copy the background, pose, or lighting" not in (
        aligned.prompt
    )
    assert "keeping its framing, background, and lighting" in aligned.prompt
    assert any("denying <Picture 1>" in f for f in aligned.applied_fixes)
    # The identity lock is untouched.
    assert "Aria Voss" in aligned.prompt
    assert "charcoal utility jacket" in aligned.prompt


def _aligned_subject_definitions(body):
    """Run one subject_definitions body through the alignment repair."""
    written = GOOD.split("summary:", 1)[1]
    return process(
        f"subject_definitions:\n{body}\n\nsummary:{written}", _aligned_ctx()
    )


def test_alignment_removes_a_pronoun_follow_on():
    # The denial lands in a second sentence that names no tag at all.
    result = _aligned_subject_definitions(
        "<Subject 1> is Aria Voss as shown in <Picture 1>. "
        "Its background and lighting must not be reproduced. "
        "<Video 1> supplies the motion."
    )
    assert "must not be reproduced" not in result.prompt, result.prompt[:400]
    assert "<Video 1> supplies the motion." in result.prompt


def test_alignment_keeps_an_unrelated_ignore_sentence():
    # "Ignore" governs the camera shake, not the framing. The old
    # unanchored \bignore\b deleted this correct sentence.
    result = _aligned_subject_definitions(
        "<Subject 1> is Aria Voss as shown in <Picture 1>. "
        "Ignore any camera shake in <Video 1> while keeping the framing "
        "from <Picture 1>."
    )
    assert "Ignore any camera shake" in result.prompt


def test_alignment_removes_a_passive_denial():
    result = _aligned_subject_definitions(
        "<Subject 1> is Aria Voss as shown in <Picture 1>. "
        "The background and lighting of <Picture 1> are not copied."
    )
    assert "are not copied" not in result.prompt


def test_alignment_keeps_retention_analysis_line_breaks():
    retention = (
        "<Subject 1> (appears in [Shot 1]): fully_preserved - identity.\n"
        "<Picture 1> supplies only identity and wardrobe.\n"
        "<Video 1> (cut and pacing structure): attribute_transfer - motion."
    )
    written = GOOD.replace(
        GOOD.split("retention_analysis:\n", 1)[1].split("\n\n", 1)[0],
        retention,
    )
    result = process(written, _aligned_ctx())
    body = result.prompt.split("retention_analysis:\n", 1)[1].split(
        "\n\n", 1
    )[0]
    # The denial goes; the other two labels keep their own lines.
    assert "supplies only identity" not in body, body
    assert body.count("\n") == 1, repr(body)
    lines = body.split("\n")
    assert lines[0].startswith("<Subject 1> (appears in [Shot 1])")
    assert lines[1].startswith("<Video 1> (cut and pacing structure)")


def test_injected_wardrobe_lands_on_the_subject_line():
    bare = (
        "<Subject 1> (appears in [Shot 1]): fully_preserved - identity.\n"
        "<Video 1> (cut and pacing structure): attribute_transfer - motion."
    )
    written = GOOD.replace(
        GOOD.split("retention_analysis:\n", 1)[1].split("\n\n", 1)[0], bare
    )
    result = process(written, CTX)
    body = result.prompt.split("retention_analysis:\n", 1)[1].split(
        "\n\n", 1
    )[0]
    subject_line, video_line = body.split("\n")
    assert "charcoal utility jacket" in subject_line, body
    assert "charcoal utility jacket" not in video_line, body


def test_alignment_leaves_unrelated_prose_byte_identical():
    clean = GOOD.replace(
        "Do not copy the background, pose, or lighting from <Picture 1>. ", ""
    ).replace(
        "No copying of the background, pose, or lighting from <Picture 1>. ",
        "",
    )
    aligned = process(clean, _aligned_ctx())
    assert not any("denying <Picture 1>" in f for f in aligned.applied_fixes)


def test_hook_counts_against_the_char_cap():
    long_hook = "For the target video, " + "x" * 6900
    result = process(GOOD, _aligned_ctx(hook=long_hook))
    assert any("7000 characters" in e for e in result.retry_errors)


def _music_ctx(lyrics="", enable_audio=True):
    return AssemblyContext(
        subject_name="Aria Voss",
        subject_wardrobe=CTX.subject_wardrobe,
        duration_seconds=6.0,
        enable_audio_prompt=enable_audio,
        music_video=True,
        lyrics=lyrics,
    )


SCORED = GOOD.replace(
    "non_diegetic_music:\nN/A",
    "non_diegetic_music:\nDowntempo synthwave at about 92 BPM, analog "
    "pad and gated drums, filter opening into the chorus.",
)


def test_music_video_rejects_an_na_score():
    result = process(GOOD, _music_ctx())  # GOOD's score is "N/A"
    assert any(
        "non_diegetic_music is 'N/A'" in e for e in result.retry_errors
    ), result.retry_errors


def test_music_video_accepts_a_described_score():
    result = process(SCORED, _music_ctx())
    assert not result.retry_errors, result.retry_errors
    assert "92 BPM" in result.prompt


def test_na_score_is_fine_without_music_video():
    result = process(GOOD, CTX)
    assert not any("non_diegetic_music" in e for e in result.retry_errors)


def test_missing_lyrics_block_is_a_retry_error():
    result = process(SCORED, _music_ctx(lyrics="I'm lonely lonely lonely"))
    assert any("no <d>" in e for e in result.retry_errors), (
        result.retry_errors
    )


def test_supplied_lyrics_present_in_a_d_block_pass():
    sung = SCORED.replace(
        "[Shot 1] A handheld camera",
        "[Shot 1] Aria Voss <Subject 1> (S1) sings, <d>[English] I'm "
        "lonely lonely lonely.</d> A handheld camera",
    )
    result = process(sung, _music_ctx(lyrics="I'm lonely lonely lonely"))
    assert not result.retry_errors, result.retry_errors
    assert "<d>[English] I'm lonely lonely lonely.</d>" in result.prompt


def test_lyrics_repeated_in_the_audio_sections_are_stripped():
    leaky = SCORED.replace(
        "Downtempo synthwave",
        "<d>[English] I'm lonely lonely lonely.</d> Downtempo synthwave",
    )
    result = process(leaky, _music_ctx())
    music = result.prompt.split("non_diegetic_music:\n")[1]
    assert "<d>" not in music
    assert "Downtempo synthwave" in music
    assert any("removed lyrics repeated" in f for f in result.applied_fixes)


def test_music_video_overrides_a_disabled_audio_prompt():
    result = process(SCORED, _music_ctx(enable_audio=False))
    assert "92 BPM" in result.prompt
    assert "Quiet natural ambience only" not in result.prompt
    assert any(
        "overrides enable_audio_prompt" in w for w in result.warnings
    )


def test_audio_disabled_still_blanks_a_non_music_prompt():
    ctx = AssemblyContext(
        subject_name="Aria Voss",
        subject_wardrobe=CTX.subject_wardrobe,
        duration_seconds=6.0,
        enable_audio_prompt=False,
    )
    result = process(SCORED, ctx)
    assert "non_diegetic_music:\nN/A" in result.prompt
    assert "92 BPM" not in result.prompt


def _measured_ctx(times):
    return AssemblyContext(
        subject_name="Aria Voss",
        subject_wardrobe=CTX.subject_wardrobe,
        duration_seconds=6.0,
        known_shot_times=times,
    )


def test_measured_times_overwrite_model_times():
    # GOOD writes [Shot 2] At 00:03.250; the detector measured 2.500.
    result = process(GOOD, _measured_ctx([0.0, 2.5]))
    assert "[Shot 2] At 00:02.500" in result.prompt
    assert "00:03.250" not in result.prompt
    assert not result.retry_errors
    assert any("measured cut list" in f for f in result.applied_fixes)


def test_measured_times_matching_the_model_apply_no_fix():
    result = process(GOOD, _measured_ctx([0.0, 3.25]))
    assert "[Shot 2] At 00:03.250" in result.prompt
    assert not result.retry_errors
    assert not any("snapped" in f for f in result.applied_fixes)


def test_measured_shot_count_mismatch_is_retry_error():
    # Three measured shots, two written: the model merged one.
    result = process(GOOD, _measured_ctx([0.0, 2.5, 4.75]))
    assert any(
        "measured shot list has 3 shots" in e for e in result.retry_errors
    ), result.retry_errors
    # Snapped anyway so an unfixed retry still lands on a real cut.
    assert "[Shot 2] At 00:02.500" in result.prompt


def test_no_measured_times_keeps_proportional_repair():
    # Regression guard: the old repair path must survive untouched.
    broken = GOOD.replace("At 00:03.250", "At 00:99.000")
    result = process(broken, CTX)
    assert any("non-monotonic" in w for w in result.warnings), result.warnings


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All assembler tests passed.")
