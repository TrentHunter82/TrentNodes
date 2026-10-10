"""
Tests for Control Overlay Rodeo.

Run:
    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_control_overlay_rodeo.py

Synthesizes the TrentNodes package so discovery never runs, and points
folder_paths at a temp directory so the real temp folder is untouched.
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")
sys.path.insert(0, ROOT)

pkg = types.ModuleType("TrentNodes")
pkg.__path__ = [PKG]
sys.modules["TrentNodes"] = pkg
_nodes = types.ModuleType("TrentNodes.nodes")
_nodes.__path__ = [os.path.join(PKG, "nodes")]
sys.modules["TrentNodes.nodes"] = _nodes

import folder_paths  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

TEMP_ROOT = tempfile.mkdtemp(prefix="cor_test_")
folder_paths.temp_directory = TEMP_ROOT

mod = importlib.import_module("TrentNodes.nodes.control_overlay_rodeo")
node = mod.ControlOverlayRodeo()
DEFAULTS = json.dumps(mod.DEFAULT_LAYERS)


def settings(**overrides):
    layers = json.loads(DEFAULTS)
    for name, cfg in overrides.items():
        layers[name].update(cfg)
    return json.dumps(layers)


def test_layers_and_downscale():
    images = torch.rand(5, 1080, 1920, 3)
    pose = torch.rand(1, 540, 960, 3)
    hed = torch.rand(5, 1080, 1920, 4)
    res = node.preview(images, 24.0, 512, DEFAULTS, pose=pose, hed=hed)
    ui = res["ui"]["rodeo"][0]
    assert res["result"][0].shape == (5, 1080, 1920, 3)

    assert ui["width"] == 1920 and ui["height"] == 1080
    assert ui["frames"] == 5
    assert [layer["name"] for layer in ui["layers"]] == ["original", "hed", "pose"]
    assert [layer["count"] for layer in ui["layers"]] == [5, 5, 1]

    out = os.path.join(TEMP_ROOT, ui["subfolder"])
    assert len(os.listdir(out)) == 11
    with Image.open(os.path.join(out, "pose_00000.jpg")) as img:
        assert img.size == (512, 288)
    with Image.open(os.path.join(out, "hed_00004.jpg")) as img:
        assert img.size == (512, 288) and img.mode == "RGB"


def test_full_resolution():
    images = torch.rand(1, 64, 48, 3)
    res = node.preview(images, 12.0, 0, DEFAULTS)
    ui = res["ui"]["rodeo"][0]
    assert torch.equal(res["result"][0], images)
    out = os.path.join(TEMP_ROOT, ui["subfolder"])
    with Image.open(os.path.join(out, "original_00000.jpg")) as img:
        assert img.size == (48, 64)
    assert ui["frames"] == 1 and len(ui["layers"]) == 1


def test_blend_math():
    base = torch.full((2, 4, 4, 3), 0.5)
    top = torch.full((2, 4, 4, 3), 0.25)

    def mix(blend, opacity=1.0, on=True):
        cfg = settings(canny={"on": on, "blend": blend, "opacity": opacity})
        return node.preview(base, 24.0, 0, cfg, canny=top)["result"][0]

    assert torch.allclose(mix("screen"), torch.tensor(1 - 0.5 * 0.75))
    assert torch.allclose(mix("multiply", 0.5), torch.tensor(0.5 * 0.5 + 0.5 * 0.125))
    assert torch.allclose(mix("add", 0.5), torch.tensor(0.625))
    assert torch.allclose(mix("difference"), torch.tensor(0.25))
    assert torch.allclose(mix("lighten"), torch.tensor(0.5))
    assert torch.allclose(mix("normal", 0.0), torch.tensor(0.5))
    assert torch.allclose(mix("normal", on=False), torch.tensor(0.5))
    assert torch.allclose(mix("add", 1.0), torch.tensor(0.75))
    off = settings(original={"on": False}, canny={"blend": "normal"})
    assert torch.allclose(node.preview(base, 24.0, 0, off, canny=top)["result"][0], top)


def test_hold_last_frame_and_resize():
    images = torch.zeros(1, 32, 32, 3)
    pose = torch.zeros(3, 16, 16, 3)
    pose[2] = 1.0
    out = node.preview(images, 24.0, 0, DEFAULTS, pose=pose)["result"][0]
    assert out.shape == (3, 32, 32, 3)
    assert out[0].max() == 0 and torch.allclose(out[2], torch.ones(32, 32, 3))


if __name__ == "__main__":
    try:
        test_layers_and_downscale()
        test_full_resolution()
        test_blend_math()
        test_hold_last_frame_and_resize()
        print("ok")
    finally:
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
