"""
Tests for the H3 reference-wiring arithmetic.

The anchor is `test_snap_length_matches_the_samplers_own_arithmetic`: our
copy of align_frame_count has to agree with comfy_extras' original for
every frame count in the usable range. The whole point of this module is
that the prompt and the sampler cannot disagree about a number, so a
drifting copy defeats it silently.

These functions are what the promptor's pass-through outputs will use;
see docs/H3_REFERENCE_WIRING_HANDOFF.md. The node-level tests that used
to live here went with the standalone node they covered - the behaviours
they checked are listed in that document's section 6 and land again on
the promptor.

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_wiring.py
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
    for sub in ("nodes", "utils", "utils.h3_prompt", "utils.h3_cowboy",
                "utils.cut_detect"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.utils.h3_cowboy.wiring import (  # noqa: E402
    H3_FPS,
    build_label_map,
    canvas_for,
    snap_length,
    trim_reference_frames,
)

# ---------------------------------------------------------------------------
# The arithmetic, against comfy_extras itself
# ---------------------------------------------------------------------------

def test_snap_length_matches_the_samplers_own_arithmetic():
    def align_frame_count(n):        # comfy_extras/nodes_minimax_h3.py
        while n % 17 != 5:
            n += 1
        return n

    for frames in range(5, 900):
        seconds = frames / H3_FPS
        ours, duration = snap_length(seconds)
        assert ours == align_frame_count(frames), (frames, ours)
        assert ours % 17 == 5
        assert abs(duration - ours / H3_FPS) < 1e-9


def test_the_documented_drift_numbers_are_real():
    # These are the numbers the handoff and the node tooltip quote. If
    # they change, both are lying.
    assert snap_length(2.0) == (56, 56 / 24)
    assert snap_length(5.0) == (124, 124 / 24)
    assert snap_length(8.0)[0] == 192
    assert abs(snap_length(8.0)[1] - 8.0) < 1e-9      # 8s lands exactly
    assert snap_length(2.0)[1] > 2.3                  # +17%


def test_snap_length_never_goes_below_the_floor():
    for seconds in (0.0, 0.01, 0.2):
        frames, _ = snap_length(seconds)
        assert frames >= 5 and frames % 17 == 5, seconds


def test_trim_reference_frames_rounds_down_not_up():
    # The generation length rounds UP; a reference video is trimmed DOWN.
    assert trim_reference_frames(4) == 0        # sampler raises below 5
    assert trim_reference_frames(5) == 5
    assert trim_reference_frames(21) == 5
    assert trim_reference_frames(22) == 22
    assert trim_reference_frames(360) == 345
    for count in range(5, 400):
        kept = trim_reference_frames(count)
        assert kept <= count and kept % 17 == 5


def test_canvas_for_matches_adapt_canvas():
    import math

    def adapt_canvas(width, height):     # comfy_extras/nodes_minimax_h3.py
        ratio = width / height
        if ratio >= 1.0:
            nom_w, nom_h = 768 * ratio, 768
        else:
            nom_w, nom_h = 768, 768 / ratio
        if nom_w * nom_h > 768 * 1344:
            s = math.sqrt((768 * 1344) / (nom_w * nom_h))
            nom_w, nom_h = nom_w * s, nom_h * s
        return (max(32, round(nom_w / 32) * 32), max(32, round(nom_h / 32) * 32))

    for w, h in ((1920, 1080), (1080, 1920), (1024, 1024), (2560, 1080),
                 (640, 480), (3840, 2160), (900, 1600)):
        assert canvas_for(w, h) == adapt_canvas(w, h), (w, h)


# ---------------------------------------------------------------------------
# The label map - the sampler's presentation order
# ---------------------------------------------------------------------------

def test_the_label_map_follows_the_samplers_order():
    # "images, then videos (each soundtrack's <Audio j> label right before
    # its <Video k>), then standalone audio" - the sampler's docstring.
    mapped = build_label_map(
        ["image_1", "image_2"], has_video=True, has_audio=True
    )
    assert mapped.splitlines() == [
        "<Picture 1> = image_1",
        "<Picture 2> = image_2",
        "<Audio 1> = audio (as the video's soundtrack)",
        "<Video 1> = video",
    ]


def test_the_spong_layout_round_trips():
    # projects/spong_h3/build_prompts.py: N character refs, the clip as
    # <Video 1>, its own audio as <Audio 1>. The one layout known to have
    # produced real videos.
    mapped = build_label_map(
        [f"image_{i}" for i in range(1, 5)], has_video=True, has_audio=True
    )
    assert "<Picture 4> = image_4" in mapped
    assert mapped.index("<Audio 1>") < mapped.index("<Video 1>")


def test_audio_alone_is_still_audio_one():
    assert "<Audio 1> = audio (standalone)" in build_label_map(
        ["image_1"], has_video=False, has_audio=True
    )


def test_nothing_connected_says_so():
    assert build_label_map([], False, False) == "nothing connected"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All H3 wiring arithmetic tests passed.")
