"""
Model-backed shot-boundary tests. These load real weights, so they are
opt-in. Run from the ComfyUI root:

    TRENT_TEST_CUT_MODELS=1 venv/bin/python \\
        custom_nodes/TrentNodes/tests/test_cut_detect_models.py

Without the gate every test SKIPs, so the file is safe in a batch run.
omnishotcut additionally needs CUDA and downloads a 164 MB checkpoint
from HuggingFace on first use; transnetv2 ships its weights in the wheel
and runs on CPU.
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

from TrentNodes.utils.cut_detect import detect_shots  # noqa: E402

GATE = os.environ.get("TRENT_TEST_CUT_MODELS") == "1"
SKIP = "SKIP (set TRENT_TEST_CUT_MODELS=1)"


def _solid(n, value, h=64, w=112):
    return torch.full((n, h, w, 3), value, dtype=torch.float32)


def _four_shot_clip():
    """Cuts at 1.25 / 2.50 / 3.75 s at 24 fps."""
    return torch.cat([
        _solid(30, 0.02), _solid(30, 0.98), _solid(30, 0.45),
        _solid(30, 0.80),
    ])


def _has_omnishotcut():
    try:
        import omnishotcut  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def test_omnishotcut_finds_the_synthetic_cuts():
    if not GATE:
        print(SKIP, end=" ")
        return
    if not _has_omnishotcut():
        print("SKIP (omnishotcut or CUDA unavailable)", end=" ")
        return

    shots = detect_shots(
        _four_shot_clip(), fps=24.0, detector="omnishotcut", min_shot_frames=4
    )
    assert shots.detector == "omnishotcut"
    assert shots.cut_times == [0.0, 1.25, 2.5, 3.75], shots.cut_times
    assert shots.shots[-1].end_frame == 120


def test_transnetv2_finds_the_synthetic_cuts():
    if not GATE:
        print(SKIP, end=" ")
        return
    try:
        import transnetv2_pytorch  # noqa: F401
    except ImportError:
        print("SKIP (transnetv2-pytorch unavailable)", end=" ")
        return

    shots = detect_shots(
        _four_shot_clip(), fps=24.0, detector="transnetv2", min_shot_frames=4
    )
    assert shots.detector == "transnetv2"
    assert shots.cut_times == [0.0, 1.25, 2.5, 3.75], shots.cut_times
    assert all(s.entry == "hard cut" for s in shots.shots[1:])
    assert any("hard cuts only" in n for n in shots.notes)


def test_auto_reaches_a_neural_detector():
    if not GATE:
        print(SKIP, end=" ")
        return
    shots = detect_shots(_four_shot_clip(), fps=24.0, detector="auto")
    assert shots.detector in ("omnishotcut", "transnetv2", "classic")
    assert shots.cut_times == [0.0, 1.25, 2.5, 3.75], shots.cut_times


def test_an_unavailable_named_detector_raises_with_a_hint():
    if not GATE:
        print(SKIP, end=" ")
        return
    try:
        detect_shots(_solid(30, 0.5), fps=24.0, detector="nonexistent")
    except RuntimeError as exc:
        assert "unknown detector" in str(exc)
    else:
        raise AssertionError("an unknown detector must raise")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All model-backed cut detection tests passed.")
