"""
Unit tests for the H3 prompt assembler. Pure-string tests, no torch,
no ComfyUI. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_assembler.py
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
    # Exclusions padded from 4 up to the minimum
    exclusion_count = result.prompt.count("No ")
    assert exclusion_count >= MIN_EXCLUSIONS
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
    assert any("trimmed non_diegetic_music" in f for f in result.applied_fixes)


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All assembler tests passed.")
