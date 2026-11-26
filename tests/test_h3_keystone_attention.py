"""
Tests for Keystone Attention (H3). CPU only.

Run:
    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_h3_keystone_attention.py
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
import comfy.model_patcher  # noqa: E402
import comfy.patcher_extension  # noqa: E402
from comfy.ldm.minimax.model import PackedLayout  # noqa: E402

ks = importlib.import_module("TrentNodes.nodes.h3_keystone_attention")


def attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores + mask
    return scores.softmax(-1) @ v


class KeystoneAttentionTest(unittest.TestCase):
    def setUp(self):
        self.model = comfy.model_patcher.ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))
        pin = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 4, 4)}
        self.layout = PackedLayout(5, 3, 4, 4, 2, keyframes=[pin])
        self.cond = [(a, b) for a, b, kind in self.layout.segments if kind == "cond"]
        self.assertEqual(len(self.cond), 1)
        s = self.layout.seq_len
        torch.manual_seed(0)
        self.qkv = [torch.randn(1, 2, s, 8) for _ in range(3)]

    def run_block(self, model, block, layout=None):
        seen = {}

        def func(q, k, v, heads, mask=None, **kwargs):
            seen["mask"] = mask
            return attention(q, k, v, heads, mask=mask)

        topts = {"minimax_h3_layout": layout or self.layout, "block_index": block}
        override = model.model_options["transformer_options"]["optimized_attention_override"]
        out = override(func, *self.qkv, 2, skip_reshape=True, transformer_options=topts)
        return out, seen["mask"]

    def test_parse_blocks(self):
        self.assertEqual(ks.parse_blocks("30-32, 34,44"), {30, 31, 32, 34, 44})
        self.assertEqual(ks.parse_blocks(""), set())

    def test_bias_zero_is_unchanged(self):
        self.assertIs(ks.H3KeystoneAttention.execute(self.model, 0.0, "30-39").args[0], self.model)
        self.assertIs(ks.H3KeystoneAttention.execute(self.model, 1.0, "").args[0], self.model)

    def test_bias_on_cond_rows_of_chosen_blocks(self):
        m = ks.H3KeystoneAttention.execute(self.model, 1.5, "30-31, 35").args[0]
        self.assertNotIn("optimized_attention_override", self.model.model_options.get("transformer_options", {}))
        a, b = self.cond[0]
        for block in (30, 31, 35):
            out, mask = self.run_block(m, block)
            expect = torch.zeros(1, self.layout.seq_len)
            expect[:, a:b] = 1.5
            self.assertTrue(torch.equal(mask, expect))
            self.assertTrue(torch.allclose(out, attention(*self.qkv, 2, mask=expect)))
        for block in (0, 29, 32, 49):
            out, mask = self.run_block(m, block)
            self.assertIsNone(mask)
            self.assertTrue(torch.equal(out, attention(*self.qkv, 2)))

    def test_no_pin_no_bias(self):
        m = ks.H3KeystoneAttention.execute(self.model, 1.0, "30-39").args[0]
        plain = PackedLayout(5, 3, 4, 4, 2)
        self.qkv = [torch.randn(1, 2, plain.seq_len, 8) for _ in range(3)]
        _, mask = self.run_block(m, 30, plain)
        self.assertIsNone(mask)

    def test_chains_previous_override(self):
        calls = []

        def previous(func, q, k, v, heads, mask=None, **kwargs):
            calls.append(mask)
            return func(q, k, v, heads, mask=mask, **kwargs)

        self.model.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = previous
        m = ks.H3KeystoneAttention.execute(self.model, 1.0, "30").args[0]
        self.run_block(m, 30)
        self.run_block(m, 10)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0, self.cond[0][0]].item(), 1.0)
        self.assertIsNone(calls[1])

    def test_context_slice(self):
        held = 3
        m = ks.H3KeystoneAttention.execute(self.model, 2.0, "30", "context slice").args[0]
        override = m.model_options["transformer_options"]["optimized_attention_override"]
        func = lambda *a, **k: self.fail("context slice must not call the plain attention")
        topts = {"minimax_h3_layout": self.layout, "block_index": 30, ks.KEY: held}
        out = override(func, *self.qkv, 2, skip_reshape=True, transformer_options=topts)
        key, carried = ks.keystone_rows(self.layout, held)
        va = next(a for a, b, kind in self.layout.segments if kind == "video")
        per = (self.layout.seq_len - va) // 3
        self.assertEqual(key, [self.cond[0], (va, va + per)])
        self.assertEqual(carried, [(va + per, va + held * per)])
        q, k, v = self.qkv
        p = ((q @ k.transpose(-1, -2)) * 8 ** -0.5).softmax(-1)
        kc = torch.zeros(self.layout.seq_len, dtype=torch.bool)
        cc = kc.clone()
        for a, b in key: kc[a:b] = True
        for a, b in carried: cc[a:b] = True
        pk, pc = p[..., kc].sum(-1, keepdim=True), p[..., cc].sum(-1, keepdim=True)
        w = torch.e ** 2.0
        scale = (pk + pc) / (w * pk + pc)
        p2 = p.clone()
        p2[..., kc] *= w * scale
        p2[..., cc] *= scale
        self.assertTrue(torch.allclose(p2.sum(-1), torch.ones(1)))
        expect = (p2 @ v).transpose(1, 2).reshape(1, -1, 16)
        self.assertTrue(torch.allclose(out, expect, atol=1e-5))
        # rest of the sequence keeps its share; no carried frames -> plain attention
        _, mask = self.run_block(m, 30)
        self.assertIsNone(mask)

    def test_count_held(self):
        m = ks.H3KeystoneAttention.execute(self.model, 1.0, "30").args[0]
        wrap = m.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][ks.KEY][0]
        mask = torch.ones(1, 1, 5, 2, 2)
        mask[:, :, :2] = 0
        seen = {}
        wrap(lambda x, t, c, o, **kw: seen.update(o), None, None, None, {}, denoise_mask=mask)
        self.assertEqual(seen[ks.KEY], 2)


if __name__ == "__main__":
    unittest.main()
