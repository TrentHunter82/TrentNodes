import importlib.util
import json
import os
import sys
import tempfile
import unittest

import torch

import comfy.model_patcher
import folder_paths
import nodes as comfy_nodes

import h3_dial_operator as op

spec = importlib.util.spec_from_file_location(
    "dials_nodes", os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-Dials", "nodes.py"))
dials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dials)


class FakeMakeDial:
    calls = []

    @classmethod
    def execute(cls, clip, model, pairs, subjects, name, inject, save):
        cls.calls.append(pairs)
        g = torch.Generator().manual_seed(len(cls.calls))
        dial = {"vec": torch.randn(8, generator=g), "target": "token_refiner", "mode": "post",
                "meta": {"name": name, "pairs": pairs}}
        return dials.io.NodeOutput(dial, "")


class Refiner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.token_refiner = torch.nn.Linear(8, 8)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder_paths.folder_names_and_paths["loras"] = ([self.tmp.name], {".safetensors"})
        folder_paths.filename_list_cache.clear()
        comfy_nodes.NODE_CLASS_MAPPINGS.update({"DialsMakeDial": FakeMakeDial, "DialsLoadDial": dials.LoadDial,
                                                "DialsApplyDial": dials.ApplyDial})
        FakeMakeDial.calls.clear()
        op.made.clear()
        self.model = comfy.model_patcher.ModelPatcher(Refiner(), torch.device("cpu"), torch.device("cpu"))

    def tearDown(self):
        self.tmp.cleanup()

    def run_op(self, state, clip="clip", dial=None):
        folder_paths.filename_list_cache.clear()
        out = op.H3DialOperator().apply(self.model, json.dumps(state), op.SUBJECTS, clip, dial)
        return out["result"][0], out["ui"]["h3_dial_operator"][0]

    def shift(self, m):
        x = torch.zeros(1, 8)
        patched = m.object_patches.get("diffusion_model.token_refiner.forward")
        base = self.model.model.diffusion_model.token_refiner(x)
        return (patched(x) - base)[0] if patched else torch.zeros(8)

    def test_terms_expand_over_templates(self):
        pairs = op.term_pairs("slow motion", "fast motion", op.SUBJECTS).splitlines()
        self.assertEqual(len(pairs), len(op.TEMPLATES))
        self.assertEqual(pairs[0], "{subject}, slow motion. ||| {subject}, fast motion.")
        self.assertEqual(op.term_pairs("{subject} sits.", "{subject} runs.", op.SUBJECTS), "{subject} sits. ||| {subject} runs.")
        with self.assertRaises(ValueError):
            op.term_pairs("{subject} a\n{subject} b", "{subject} c", op.SUBJECTS)
        self.assertEqual(op.term_pairs("slow motion", "fast motion", " ").splitlines()[1],
                         "A video, slow motion. ||| A video, fast motion.")

    def test_sum_matches_strengths_and_off_slots_skip(self):
        slots = [{"id": "1", "a": "still", "b": "morphing", "strength": 2},
                 {"id": "2", "a": "photo", "b": "animation", "strength": -1.5},
                 {"id": "3", "a": "3d", "b": "2d", "strength": 4, "on": False}]
        m, _ = self.run_op({"slots": slots})
        k1, k2 = (op.term_key(s["a"], s["b"], op.SUBJECTS) for s in slots[:2])
        want = 2 * op.made[k1]["vec"] - 1.5 * op.made[k2]["vec"]
        self.assertTrue(torch.allclose(self.shift(m), want, atol=1e-5))
        self.assertEqual(len(FakeMakeDial.calls), 2)

    def test_save_then_reuse_without_clip(self):
        slot = {"id": "1", "a": "slow motion", "b": "fast motion", "strength": 3, "save": True}
        m1, ui = self.run_op({"slots": [slot]})
        self.assertEqual(ui["slots"][0]["saved"], "dials/slow-motion_vs_fast-motion.safetensors")
        op.made.clear()
        m2, ui = self.run_op({"slots": [dict(slot, save=False)]}, clip=None)
        self.assertEqual(len(FakeMakeDial.calls), 1)
        self.assertTrue(torch.allclose(self.shift(m1), self.shift(m2), atol=0.05))  # bf16 file
        listed = op.list_dials()
        self.assertEqual((listed[0]["a"], listed[0]["b"]), ("slow motion", "fast motion"))
        # same terms again: the existing file, no copy
        self.assertEqual(op.save_now(slot, op.SUBJECTS), "dials/slow-motion_vs_fast-motion.safetensors")
        # other terms with the same name get a new one
        self.run_op({"slots": [{"id": "2", "a": "slow-motion", "b": "fast motion!", "strength": 1, "save": True}]})
        folder_paths.filename_list_cache.clear()
        self.assertEqual(sorted(d["name"] for d in op.list_dials()),
                         ["slow-motion_vs_fast-motion", "slow-motion_vs_fast-motion_2"])

    def test_save_now_waits_for_unmade(self):
        self.assertIsNone(op.save_now({"a": "x", "b": "y"}, op.SUBJECTS))

    def test_file_slot_input_dial_and_mix(self):
        self.run_op({"slots": [{"id": "1", "a": "calm", "b": "wild", "strength": 1, "save": True}]})
        ext = {"vec": torch.ones(8), "target": "token_refiner", "mode": "post", "meta": {"name": "ext"}}
        m, ui = self.run_op({"slots": [{"id": "1", "file": "dials/calm_vs_wild.safetensors", "strength": 2}],
                             "input": {"strength": -1}, "save_mix": True}, clip=None, dial=ext)
        self.assertEqual(ui["mix"], "dials/mix_calm-vs-wild+2_ext-1.safetensors")
        folder_paths.filename_list_cache.clear()
        mixed = op.load_file(ui["mix"])
        self.assertTrue(torch.allclose(mixed["vec"], self.shift(m), atol=0.05))

    def test_text_slot_needs_clip(self):
        with self.assertRaisesRegex(ValueError, "connect clip"):
            self.run_op({"slots": [{"id": "1", "a": "a", "b": "b", "strength": 1}]}, clip=None)

    def test_rejects_paths_outside_dials(self):
        with self.assertRaises(ValueError):
            op.load_file("../evil.safetensors")


if __name__ == "__main__":
    unittest.main()
