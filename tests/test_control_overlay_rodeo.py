"""
Tests for Control Overlay Rodeo.

Run:
    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_control_overlay_rodeo.py

Synthesizes the TrentNodes package so discovery never runs, and points
folder_paths at a temp directory so the real temp folder is untouched.
"""

import importlib
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


def test_layers_and_downscale():
    images = torch.rand(5, 1080, 1920, 3)
    pose = torch.rand(1, 540, 960, 3)
    hed = torch.rand(5, 1080, 1920, 4)
    ui = node.preview(images, 24.0, 512, pose=pose, hed=hed)["ui"]["rodeo"][0]

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
    ui = node.preview(images, 12.0, 0)["ui"]["rodeo"][0]
    out = os.path.join(TEMP_ROOT, ui["subfolder"])
    with Image.open(os.path.join(out, "original_00000.jpg")) as img:
        assert img.size == (48, 64)
    assert ui["frames"] == 1 and len(ui["layers"]) == 1


if __name__ == "__main__":
    try:
        test_layers_and_downscale()
        test_full_resolution()
        print("ok")
    finally:
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
