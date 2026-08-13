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


def test_a_transition_keeps_both_label_sets():
    # The transition span's labels must not evict the shot's own.
    shots = _spans_to_shots(
        [
            (0, 30, "General", "New_Start"),
            (30, 34, "Dissolve", "Transition_Source"),
            (34, 60, "General", "Hard_Cut"),
        ],
        fps=24.0, total_frames=60,
    )
    assert len(shots) == 2
    assert shots[1].raw_labels == {
        "intra": "General", "inter": "Hard_Cut",
        "transition_intra": "Dissolve", "transition_inter": "Transition_Source",
    }


def test_classic_folds_a_runt_final_shot():
    # detect_boundaries never length-checks the shot that runs to the
    # end of the clip, so min_shot_frames used to mean something weaker
    # here than on the neural paths.
    clip = torch.cat([_solid(30, 0.0), _solid(30, 1.0), _solid(3, 0.4)])
    shots = detect_shots(clip, fps=24.0, detector="classic",
                         min_shot_frames=8)
    assert all(s.frame_count >= 8 for s in shots.shots), shots.cut_times
    assert shots.shots[-1].end_frame == 63


def _break_neural_detectors():
    """Make both neural backends raise, the way a missing package does."""
    from TrentNodes.utils.cut_detect import detectors

    def boom(*_args, **_kwargs):
        raise RuntimeError("not installed")

    saved = (detectors._detect_omnishotcut, detectors._detect_transnetv2)
    detectors._detect_omnishotcut = boom
    detectors._detect_transnetv2 = boom
    return detectors, saved


def _restore_neural_detectors(detectors, saved):
    detectors._detect_omnishotcut, detectors._detect_transnetv2 = saved


def test_auto_records_the_detector_it_fell_back_to():
    detectors, saved = _break_neural_detectors()
    try:
        shots = detect_shots(_four_shot_clip(), fps=24.0, detector="auto")
    finally:
        _restore_neural_detectors(detectors, saved)
    assert shots.detector == "classic"
    assert shots.requested == "auto"
    assert shots.fallback is True
    assert any("omnishotcut unavailable" in n for n in shots.notes)
    assert any("transnetv2 unavailable" in n for n in shots.notes)


def test_a_clean_run_is_not_flagged_as_a_fallback():
    shots = detect_shots(_four_shot_clip(), fps=24.0, detector="classic")
    assert shots.fallback is False
    assert shots.requested == "classic"
    assert "FALLBACK" not in format_report(shots)


def test_neural_only_refuses_to_use_classic():
    detectors, saved = _break_neural_detectors()
    try:
        detect_shots(_four_shot_clip(), fps=24.0, detector="auto",
                     fallback_policy="neural_only")
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "every detector failed" in str(exc), exc
        assert "omnishotcut" in str(exc) and "transnetv2" in str(exc)
    finally:
        _restore_neural_detectors(detectors, saved)


def test_report_headlines_a_fallback():
    shots = _sample_list()
    shots.requested = "omnishotcut"
    shots.detector = "classic"
    shots.fallback = True
    report = format_report(shots)
    assert "FALLBACK:  asked for omnishotcut, ran classic" in report
    # It sits above the numbers it invalidates.
    assert report.index("FALLBACK") < report.index("Shots:")


def test_a_missing_transnetv2_reports_the_install_hint():
    from TrentNodes.utils.cut_detect import detectors
    saved_module = sys.modules.get("transnetv2_pytorch")
    saved_model = detectors._tnv2_model
    sys.modules["transnetv2_pytorch"] = None   # makes `import` raise
    detectors._tnv2_model = None
    try:
        detect_shots(_solid(8, 0.5), fps=24.0, detector="transnetv2")
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "pip install transnetv2-pytorch" in str(exc), exc
    finally:
        detectors._tnv2_model = saved_model
        if saved_module is None:
            del sys.modules["transnetv2_pytorch"]
        else:
            sys.modules["transnetv2_pytorch"] = saved_module


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


def test_an_empty_shot_list_reads_as_no_cuts():
    # The JSON carries "fps": 24.0. Text that parses as JSON must be
    # answered from the JSON alone, or the text reader scrapes that
    # field and reports a phantom cut at 24 seconds.
    empty = ShotList(shots=[], fps=24.0, total_frames=0,
                     detector="omnishotcut", notes=["no frames to analyze"])
    assert parse_cut_times(shots_to_json(empty)) == []
    assert parse_cut_times(format_cut_times(empty)) == []


def test_zero_frame_detection_round_trips_to_no_cuts():
    shots = detect_shots(torch.zeros((0, 8, 8, 3)), fps=24.0,
                         detector="classic")
    assert parse_cut_times(shots_to_json(shots)) == []


def test_unrecognized_json_yields_nothing_rather_than_junk():
    assert parse_cut_times('{"detector": "classic", "fps": 24.0}') == []


def test_multi_line_cut_lists_keep_every_number():
    # A wrapped list used to lose every token but the first per line.
    parsed = parse_cut_times("0.0, 1.5\n3.0, 4.5")
    assert [p.time for p in parsed] == [0.0, 1.5, 3.0, 4.5]


def test_prose_gives_up_only_its_first_time():
    parsed = parse_cut_times("[Shot 2] At 00:02.500, she lifts 3 crates")
    assert [p.time for p in parsed] == [2.5]


def test_a_mixed_paste_reads_rows_and_lists_side_by_side():
    text = "Shot 2 | 00:02.500 | 2.583s | dissolve\n5.5, 7.25"
    parsed = parse_cut_times(text)
    assert [p.time for p in parsed] == [2.5, 5.5, 7.25]


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


def _dissolve_list():
    """
    What the detector really emits around a dissolve: the outgoing shot
    ends at its own span end (56) and the incoming one starts where the
    effect finishes (60), so frames 56-60 belong to no Shot at all.
    """
    fps = 24.0
    shots = [
        Shot(1, 0, 30, fps, "start"),
        Shot(2, 30, 56, fps, "hard cut"),
        Shot(3, 60, 96, fps, "dissolve", transition_frames=(56, 60)),
    ]
    return ShotList(shots=shots, fps=fps, total_frames=96,
                    detector="omnishotcut")


def test_shot_durations_do_not_tile_the_clip_around_a_dissolve():
    # The premise of the next test, stated outright: this is why the
    # ribbon cannot simply draw shot ranges.
    shots = _dissolve_list()
    assert sum(s.duration for s in shots.shots) < shots.duration


def test_ribbon_segments_tile_the_whole_clip():
    from TrentNodes.utils.cut_detect.filmstrip import _ribbon_segments
    shots = _dissolve_list()
    segments = _ribbon_segments(shots)
    assert segments[0][0] == 0.0
    for (_s0, e0, _c0, _t0), (s1, *_rest) in zip(segments, segments[1:]):
        assert abs(s1 - e0) < 1e-6, (e0, s1)
    assert abs(segments[-1][1] - shots.duration) < 1e-6
    # The dissolve gets its own slice rather than reading as a hole.
    ramps = [s for s in segments if s[3]]
    assert len(ramps) == 1, segments
    assert abs(ramps[0][0] - 56 / 24.0) < 1e-6
    assert abs(ramps[0][1] - 60 / 24.0) < 1e-6


def test_a_short_shot_does_not_repeat_a_frame():
    from TrentNodes.utils.cut_detect.filmstrip import _plan_cards
    shots = ShotList(shots=[Shot(1, 0, 2, 24.0, "start")], fps=24.0,
                     total_frames=2)
    cards, _used = _plan_cards(shots, thumbs_per_shot=4)
    assert len(cards) == 2
    assert len({c.frame for c in cards}) == 2


def test_the_card_cap_never_drops_a_shot():
    from TrentNodes.utils.cut_detect.filmstrip import _plan_cards
    shots = ShotList(
        shots=[Shot(i + 1, i * 10, i * 10 + 10, 24.0, "hard cut")
               for i in range(80)],
        fps=24.0, total_frames=800,
    )
    cards, used = _plan_cards(shots, thumbs_per_shot=8)
    assert used < 8                       # degraded, as it must
    leads = {c.shot.index for c in cards if c.lead}
    assert leads == {i + 1 for i in range(80)}


def test_thumbs_per_shot_adds_rows_not_shots():
    frames = _four_shot_clip()
    shots = detect_shots(frames, fps=24.0, detector="classic")
    one = render_film_strip(frames, shots, thumb_width=120, columns=4)
    three = render_film_strip(frames, shots, thumb_width=120, columns=4,
                              thumbs_per_shot=3)
    assert three.shape[2] == one.shape[2]   # same width
    assert three.shape[1] > one.shape[1]    # taller


def test_thumbs_per_shot_one_is_unchanged():
    frames = _four_shot_clip()
    shots = detect_shots(frames, fps=24.0, detector="classic")
    a = render_film_strip(frames, shots, thumb_width=120)
    b = render_film_strip(frames, shots, thumb_width=120, thumbs_per_shot=1)
    assert torch.equal(a, b)


def test_explicit_columns_respect_the_sheet_cap():
    from TrentNodes.utils.cut_detect.filmstrip import MAX_SHEET_WIDTH
    frames = _solid(120, 0.5)
    shots = ShotList(
        shots=[Shot(i + 1, i * 3, i * 3 + 3, 24.0, "hard cut")
               for i in range(40)],
        fps=24.0, total_frames=120,
    )
    strip = render_film_strip(frames, shots, thumb_width=640, columns=32)
    assert strip.shape[2] <= MAX_SHEET_WIDTH, strip.shape


def test_overlap_is_clamped_to_the_model_window():
    from TrentNodes.utils.cut_detect.detectors import (
        OSC_MAX_OVERLAP, _clamp_overlap,
    )
    assert _clamp_overlap(-5) == 0
    assert _clamp_overlap(20) == 20
    assert _clamp_overlap(500) == OSC_MAX_OVERLAP


# ---------------------------------------------------------------------------
# The node itself
# ---------------------------------------------------------------------------

class _FakeComponents:
    def __init__(self, images, frame_rate):
        self.images = images
        self.frame_rate = frame_rate


class _FakeVideo:
    """Stands in for a ComfyUI VIDEO without importing comfy_api."""

    def __init__(self, images, frame_rate):
        self._components = _FakeComponents(images, frame_rate)

    def get_components(self):
        return self._components


def _node():
    from TrentNodes.nodes.cut_detective import CutDetective
    return CutDetective()


def _detect(node, **kwargs):
    """Call detect() with the required widgets at their defaults."""
    call = dict(
        detector="classic", sensitivity=0.5, min_shot_frames=4,
        thumb_width=120, columns=0, show_timeline=True,
        include_first_shot=True,
    )
    call.update(kwargs)
    return node.detect(**call)


def test_node_outputs_match_its_return_names():
    from TrentNodes.nodes.cut_detective import CutDetective
    out = _detect(_node(), frames=_four_shot_clip(), fps=24.0)
    assert len(out) == len(CutDetective.RETURN_TYPES)
    assert len(out) == len(CutDetective.RETURN_NAMES)
    by_name = dict(zip(CutDetective.RETURN_NAMES, out))
    assert isinstance(by_name["cut_times"], str)
    assert isinstance(by_name["num_shots"], int)
    assert by_name["film_strip"].dim() == 4
    assert by_name["num_shots"] == 4


def test_node_prefers_the_video_input_and_its_fps():
    from fractions import Fraction
    video = _FakeVideo(_four_shot_clip(), Fraction(24000, 1001))
    out = _detect(_node(), video=video, frames=_solid(2, 0.5), fps=99.0)
    report = dict(zip(_node().RETURN_NAMES, out))["report"]
    # 24000/1001 fps, and the frames input plus fps=99.0 are both ignored.
    assert "23.976 fps" in report, report
    assert "120 frames" in report, report


def test_node_include_first_shot_off_drops_zero():
    out = _detect(_node(), frames=_four_shot_clip(), fps=24.0,
                  include_first_shot=False)
    cut_times = out[0]
    assert not cut_times.startswith("0.000"), cut_times


def test_node_without_an_input_raises():
    try:
        _detect(_node())
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "Connect either" in str(exc), exc


def test_node_rejects_a_non_video_object():
    try:
        _detect(_node(), video=object())
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "not a ComfyUI VIDEO object" in str(exc), exc


def test_node_reports_the_detector_it_used():
    out = _detect(_node(), frames=_four_shot_clip(), fps=24.0)
    by_name = dict(zip(_node().RETURN_NAMES, out))
    assert by_name["detector_used"] == "classic"


def test_node_passes_the_fallback_policy_through():
    detectors, saved = _break_neural_detectors()
    try:
        _detect(_node(), frames=_four_shot_clip(), fps=24.0,
                detector="auto", fallback_policy="neural_only")
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "every detector failed" in str(exc), exc
    finally:
        _restore_neural_detectors(detectors, saved)


def test_film_strip_marks_a_fallback():
    frames = _four_shot_clip()
    shots = detect_shots(frames, fps=24.0, detector="classic")
    clean = render_film_strip(frames, shots, thumb_width=120)
    shots.requested = "omnishotcut"
    shots.fallback = True
    flagged = render_film_strip(frames, shots, thumb_width=120)
    # Same geometry, different pixels: the marker is drawn in the header.
    assert flagged.shape == clean.shape
    assert not torch.equal(flagged, clean)


def test_node_is_registered_under_its_trent_key():
    from TrentNodes.nodes import cut_detective
    assert "TrentCutDetective" in cut_detective.NODE_CLASS_MAPPINGS
    assert (cut_detective.NODE_DISPLAY_NAME_MAPPINGS["TrentCutDetective"]
            == "Cut Detective")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All cut detection tests passed.")
