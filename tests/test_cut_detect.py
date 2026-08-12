"""
Unit tests for shot-boundary detection, cut serialization, and the film
strip. CPU-only and model-free: the neural detectors are exercised by
tests/test_cut_detect_models.py, which needs CUDA and downloaded
weights. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_cut_detect.py
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
    for sub in ("nodes", "utils"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402

from TrentNodes.utils.cut_detect import (  # noqa: E402
    Shot,
    ShotList,
    detect_shots,
    format_cut_times,
    format_report,
    format_shot_table,
    parse_cut_times,
    render_film_strip,
    shots_to_json,
)
from TrentNodes.utils.cut_detect.detectors import (  # noqa: E402
    _merge_runts,
    _spans_to_shots,
    _to_uint8_frames,
)


def _solid(n, value, h=48, w=64):
    return torch.full((n, h, w, 3), value, dtype=torch.float32)


def _four_shot_clip():
    """0.0 / 1.25 / 2.5 / 3.75 s at 24 fps."""
    return torch.cat([
        _solid(30, 0.0), _solid(30, 1.0), _solid(30, 0.45), _solid(30, 0.85),
    ])


def _sample_list():
    fps = 24.0
    shots = [
        Shot(1, 0, 30, fps, "start"),
        Shot(2, 30, 60, fps, "hard cut"),
        Shot(3, 60, 96, fps, "dissolve", transition_frames=(56, 60)),
    ]
    return ShotList(shots=shots, fps=fps, total_frames=96,
                    detector="omnishotcut")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_classic_finds_hard_cuts():
    shots = detect_shots(
        _four_shot_clip(), fps=24.0, detector="classic", min_shot_frames=4
    )
    assert shots.cut_times == [0.0, 1.25, 2.5, 3.75], shots.cut_times
    assert shots.num_cuts == 3
    assert shots.shots[0].entry == "start"
    assert all(s.entry == "hard cut" for s in shots.shots[1:])


def test_single_shot_clip_has_no_cuts():
    shots = detect_shots(_solid(60, 0.4), fps=24.0, detector="classic")
    assert len(shots.shots) == 1
    assert shots.cut_times == [0.0]
    assert shots.num_cuts == 0


def test_tiny_and_empty_batches_do_not_raise():
    for count in (0, 1, 2):
        shots = detect_shots(
            _solid(count, 0.3) if count else torch.zeros((0, 8, 8, 3)),
            fps=24.0, detector="classic",
        )
        assert len(shots.shots) == (1 if count else 0)


def test_shot_geometry():
    shots = detect_shots(_four_shot_clip(), fps=24.0, detector="classic")
    assert shots.duration == 5.0
    first = shots.shots[0]
    assert (first.start_frame, first.end_frame) == (0, 30)
    assert first.duration == 1.25
    assert shots.shots[-1].end_frame == 120  # every frame stays covered


def test_frames_are_resized_and_quantized():
    frames = _to_uint8_frames(_solid(3, 1.0, h=100, w=200), 48, 27)
    assert frames.shape == (3, 27, 48, 3)
    assert frames.dtype.name == "uint8"
    assert frames.max() == 255


# ---------------------------------------------------------------------------
# Span folding and runt merging
# ---------------------------------------------------------------------------

def test_transition_span_becomes_the_next_shot_entry():
    spans = [
        (0, 30, "General", "New_Start"),
        (30, 36, "Dissolve", "Transition_Source"),
        (36, 90, "General", "Transition"),
    ]
    shots = _spans_to_shots(spans, fps=24.0, total_frames=90)
    # The dissolve is a boundary, not a shot of its own.
    assert len(shots) == 2
    assert shots[1].entry == "dissolve"
    assert shots[1].transition_frames == (30, 36)
    assert shots[1].start_frame == 36


def test_leading_transition_stays_inside_shot_one():
    spans = [
        (0, 8, "Fade", "New_Start"),
        (8, 60, "General", "Transition"),
    ]
    shots = _spans_to_shots(spans, fps=24.0, total_frames=60)
    assert len(shots) == 1
    assert shots[0].entry == "start"
    assert shots[0].start_frame == 0


def test_mid_video_new_start_reads_as_a_hard_cut():
    spans = [
        (0, 30, "General", "New_Start"),
        (30, 60, "General", "New_Start"),
    ]
    shots = _spans_to_shots(spans, fps=24.0, total_frames=60)
    assert shots[1].entry == "hard cut"
    assert shots[1].raw_labels["inter"] == "New_Start"


def test_sudden_jump_keeps_its_own_name():
    spans = [
        (0, 30, "General", "New_Start"),
        (30, 60, "General", "Sudden_Jump"),
    ]
    shots = _spans_to_shots(spans, fps=24.0, total_frames=60)
    assert shots[1].entry == "sudden jump"
    assert shots[1].is_hard_cut


def test_runts_fold_into_the_previous_shot():
    fps = 24.0
    shots = [
        Shot(1, 0, 30, fps), Shot(2, 30, 32, fps, "hard cut"),
        Shot(3, 32, 70, fps, "hard cut"),
    ]
    merged = _merge_runts(shots, min_shot_frames=4)
    assert len(merged) == 2
    assert merged[0].end_frame == 32
    assert [s.index for s in merged] == [1, 2]


def test_a_runt_first_shot_folds_forward():
    fps = 24.0
    shots = [
        Shot(1, 0, 2, fps), Shot(2, 2, 40, fps, "hard cut"),
    ]
    merged = _merge_runts(shots, min_shot_frames=4)
    assert len(merged) == 1
    assert merged[0].start_frame == 0
    assert merged[0].entry == "start"


# ---------------------------------------------------------------------------
# Serializing
# ---------------------------------------------------------------------------

def test_cut_times_string():
    shots = _sample_list()
    assert format_cut_times(shots) == "0.000, 1.250, 2.500"
    assert format_cut_times(shots, include_first=False) == "1.250, 2.500"


def test_shot_table_names_the_transition_length():
    table = format_shot_table(_sample_list())
    lines = table.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("Shot 1 | 00:00.000 |")
    assert "hard cut" in lines[1]
    assert "dissolve over 0.167s" in lines[2]


def test_report_counts_the_cut_kinds():
    report = format_report(_sample_list())
    assert "Shots:     3  (2 cuts)" in report
    assert "1x dissolve" in report and "1x hard cut" in report


def test_json_carries_frames_and_raw_labels():
    import json
    payload = json.loads(shots_to_json(_sample_list()))
    assert payload["num_shots"] == 3
    assert payload["cut_times"] == [0.0, 1.25, 2.5]
    assert payload["hard_cut_times"] == [1.25]
    assert payload["shots"][2]["transition_frames"] == [56, 60]


# ---------------------------------------------------------------------------
# Reading a cut list back
# ---------------------------------------------------------------------------

def test_every_emitted_format_round_trips():
    shots = _sample_list()
    for text in (
        format_cut_times(shots), format_shot_table(shots),
        shots_to_json(shots),
    ):
        parsed = parse_cut_times(text)
        assert [p.time for p in parsed] == shots.cut_times, text


def test_shot_table_round_trip_keeps_the_kinds():
    parsed = parse_cut_times(format_shot_table(_sample_list()))
    assert [p.kind for p in parsed] == ["start", "hard cut", "dissolve"]


def test_parser_accepts_hand_written_forms():
    cases = {
        "0.0, 2.5, 5.083": [0.0, 2.5, 5.083],
        "00:00.000  00:02.500": [0.0, 2.5],
        "[Shot 2] At 00:02.500, [Shot 3] At 00:05.000": [2.5, 5.0],
        "[1.0, 2.0, 3.0]": [1.0, 2.0, 3.0],
        '{"cut_times": [0, 1.5]}': [0.0, 1.5],
        "1.5\n0.5\n2.5": [0.5, 1.5, 2.5],   # sorted
    }
    for text, expected in cases.items():
        assert [p.time for p in parse_cut_times(text)] == expected, text


def test_shot_numbers_are_not_read_as_times():
    # "Shot 3" must not contribute a 3.0s cut.
    parsed = parse_cut_times("Shot 3 | 00:05.083 | 1.917s | dissolve")
    assert [p.time for p in parsed] == [5.083]


def test_parser_rejects_junk_and_dedupes():
    assert parse_cut_times("") == []
    assert parse_cut_times("   ") == []
    assert parse_cut_times("no numbers here") == []
    assert [p.time for p in parse_cut_times("1.0, 1.0005, 2.0")] == [1.0, 2.0]


def test_hyphen_reads_as_a_range_not_a_minus_sign():
    # The legacy H3 shot form. Both endpoints are boundaries, and the
    # shared 3.250 collapses; the trailing clip end is the caller's to
    # drop against the real duration.
    parsed = parse_cut_times("[0.000s-3.250s] [3.250s-6.000s]")
    assert [p.time for p in parsed] == [0.0, 3.25, 6.0]


# ---------------------------------------------------------------------------
# Film strip
# ---------------------------------------------------------------------------

def test_film_strip_is_a_comfyui_image():
    frames = _four_shot_clip()
    shots = detect_shots(frames, fps=24.0, detector="classic")
    strip = render_film_strip(frames, shots, thumb_width=120)
    assert strip.dim() == 4 and strip.shape[0] == 1 and strip.shape[3] == 3
    assert strip.dtype == torch.float32
    assert 0.0 <= float(strip.min()) and float(strip.max()) <= 1.0


def test_film_strip_wraps_onto_rows():
    frames = _four_shot_clip()
    shots = detect_shots(frames, fps=24.0, detector="classic")
    one_row = render_film_strip(frames, shots, thumb_width=120, columns=4)
    two_rows = render_film_strip(frames, shots, thumb_width=120, columns=2)
    assert two_rows.shape[1] > one_row.shape[1]   # taller
    assert two_rows.shape[2] < one_row.shape[2]   # narrower


def test_film_strip_survives_an_empty_shot_list():
    strip = render_film_strip(_solid(4, 0.5), ShotList(), thumb_width=120)
    assert strip.shape == (1, 64, 64, 3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All cut detection tests passed.")
