"""
Tests for Refiner One Look (H3). CPU only.

Run:
    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_h3_refiner_one_look.py
"""

import importlib
import os
import sys
import types
import unittest

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")
sys.path.insert(0, ROOT)

pkg = types.ModuleType("TrentNodes")
pkg.__path__ = [PKG]
sys.modules["TrentNodes"] = pkg
_mod = types.ModuleType("TrentNodes.nodes")
_mod.__path__ = [os.path.join(PKG, "nodes")]
sys.modules["TrentNodes.nodes"] = _mod

import torch  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402

ol = importlib.import_module("TrentNodes.nodes.h3_refiner_one_look")


class OneLookTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(1, 4, 30, 16, 16)
        self.lo = torch.nn.functional.avg_pool2d(self.x.flatten(0, 2), 3, 1, 1).view(self.x.shape)
        self.audio = torch.randn(1, 8, 30)

    def refine(self, gain):
        a = torch.full((1, 4, 30, 1, 1), 0.9)
        b = torch.full_like(a, 0.1)
        a[:, :, :8], b[:, :, :8] = 1.3, 0.4      # head
        a[:, :, 22:], b[:, :, 22:] = 0.8, -0.3   # tail
        return a * self.lo + b + gain * a * (self.x - self.lo)

    def test_one_correction(self):
        y = self.refine(1.0)
        orig = {"samples": NestedTensor([self.x, self.audio])}
        video, audio = ol.H3RefinerOneLook.execute(orig, {"samples": NestedTensor([y, self.audio])}).args[0]["samples"].unbind()
        self.assertIs(audio, self.audio)
        self.assertTrue(torch.allclose(video, 0.9 * self.x + 0.1, atol=1e-4))

    def test_keeps_fine_band(self):
        # band gain is not part of the affine: the body keeps the refiner output
        y = self.refine(0.7)
        out = ol.one_look(self.x, y)
        self.assertLess((out[:, :, 10:20] - y[:, :, 10:20]).abs().mean(), 0.03)
        steps = (out - 0.9 * self.x).mean(dim=(0, 1, 3, 4)).diff().abs()
        self.assertLess(steps.max(), 0.02)


if __name__ == "__main__":
    unittest.main()
