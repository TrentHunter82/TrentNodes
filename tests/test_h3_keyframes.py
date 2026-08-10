"""
Unit tests for H3 keyframe selection on synthetic tensors. CPU-only,
no ComfyUI imports. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_keyframes.py
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

import torch  # noqa: E402

from TrentNodes.utils.h3_prompt.keyframes import (  # noqa: E402
    frame_label,
    select_keyframes,
)


def _solid(n, value):
    return torch.full((n, 64, 64, 3), value, dtype=torch.float32)


def test_hard_cuts_selected():
    # Three solid scenes: black, white, mid-gray. Cuts land at 20 and 40.
    frames = torch.cat([_solid(20, 0.0), _solid(20, 1.0), _solid(20, 0.5)])
    result = select_keyframes(frames, fps=24.0, max_frames=8)

    assert 20 in result.scene_boundaries, result.scene_boundaries
    assert 40 in result.scene_boundaries, result.scene_boundaries
    assert result.indices[0] == 0
    assert result.indices[-1] == 59
    assert 20 in result.indices and 40 in result.indices, result.indices
    assert result.indices == sorted(set(result.indices))
    assert len(result.indices) <= 8
    assert result.diff_stats["num_scenes"] == 3


def test_single_scene_motion():
    # One continuous scene with a moving bright patch: no cuts.
    frames = torch.zeros((40, 64, 64, 3))
    for i in range(40):
        x = 4 + i
        frames[i, 20:36, x:x + 8, :] = 1.0
    result = select_keyframes(frames, fps=24.0, max_frames=6)

    assert result.scene_boundaries == [0], result.scene_boundaries
    assert result.indices[0] == 0
    assert result.indices[-1] == 39
    assert 2 <= len(result.indices) <= 6
    assert result.timestamps == [round(i / 24.0, 3) for i in result.indices]


def test_two_frame_edge_case():
    frames = torch.rand((2, 32, 32, 3))
    result = select_keyframes(frames, fps=24.0, max_frames=8)
    assert result.indices == [0, 1]


def test_single_frame_edge_case():
    frames = torch.rand((1, 32, 32, 3))
    result = select_keyframes(frames, fps=24.0, max_frames=8)
    assert result.indices == [0]
    assert result.method == "single"


def test_cap_respected():
    frames = torch.rand((200, 32, 32, 3))
    result = select_keyframes(frames, fps=24.0, max_frames=4)
    assert len(result.indices) <= 4
    assert result.indices[0] == 0
    assert result.indices[-1] == 199


def test_frame_label():
    label = frame_label(2, 6, 1.25, 30)
    assert label == "Frame 2 of 6 - t=1.250s (video frame index 30)"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All keyframe tests passed.")
