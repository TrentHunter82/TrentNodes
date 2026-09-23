import unittest
from types import SimpleNamespace

import torch

from h3_chain_batch import H3ChainAudio, H3ChainStep


class ChainTests(unittest.TestCase):
    def manager(self):
        m = SimpleNamespace(unique_id="7", inputs={}, frames_per_batch=141,
                            has_closed_inputs=False)
        m.reset = lambda: setattr(m, 'unique_id', None)
        return m

    def advance(self, manager, step, count=3, seed=10):
        return H3ChainStep().advance(manager, count, seed,
                                     {"7": {"inputs": {"requeue": step}},
                                      "step": {"inputs": {"meta_batch": ["7", 0]}}}, "step")

    def audio(self, value, rate=24000, channels=2):
        return {"waveform": torch.full((1, channels, 24), float(value)), "sample_rate": rate}

    def test_count_seed_and_stop(self):
        m = self.manager()
        for n in range(3):
            self.assertEqual(self.advance(m, n), (n, 10 + n))
            self.assertEqual(m.has_closed_inputs, n == 2)
        self.assertEqual(m.total_frames, 423)
        with self.assertRaises(ValueError):
            self.advance(m, 3)

    def test_one_chunk_and_seed_wrap(self):
        m = self.manager()
        self.advance(m, 0, count=1)
        self.assertTrue(m.has_closed_inputs)
        self.assertEqual(self.advance(m, 1, seed=(1 << 64) - 1)[1], 0)

    def test_audio_order_and_release(self):
        m, node = self.manager(), H3ChainAudio()
        for n in range(3):
            self.advance(m, n)
            out = node.accumulate(self.audio(n), m, n, f"tail{n}")[0]
        expected = torch.cat([self.audio(n)["waveform"] for n in range(3)], -1)
        torch.testing.assert_close(out["waveform"], expected)
        self.assertEqual(node.chunks, [])

    def test_new_run_clears_partial_audio(self):
        m, node = self.manager(), H3ChainAudio()
        node.accumulate(self.audio(99), m, 0, "old")
        self.advance(m, 0, count=1)
        out = node.accumulate(self.audio(1), m, 0, "new")[0]
        torch.testing.assert_close(out["waveform"], self.audio(1)["waveform"])

    def test_missing_chunk_and_format_changes_fail(self):
        for kind in ("skip", "rate", "channels", "missing_save"):
            with self.subTest(kind=kind):
                m, node = self.manager(), H3ChainAudio()
                node.accumulate(self.audio(0), m, 0, "tail0")
                audio = self.audio(1, rate=48000 if kind == "rate" else 24000,
                                   channels=1 if kind == "channels" else 2)
                with self.assertRaises(ValueError):
                    node.accumulate(audio, m, 2 if kind == "skip" else 1,
                                    "" if kind == "missing_save" else "tail1")

    def test_cpu_copy_does_not_alias_input(self):
        m, node = self.manager(), H3ChainAudio()
        audio = self.audio(1)
        node.accumulate(audio, m, 0, "tail")
        audio["waveform"].zero_()
        self.assertTrue(torch.all(node.chunks[0] == 1))

    def test_driver_conflict(self):
        m = self.manager()
        m.inputs["driver"] = object()
        with self.assertRaises(ValueError):
            self.advance(m, 0)

    def test_cached_finalized_manager_restores_id(self):
        m = self.manager()
        m.unique_id = None
        self.advance(m, 0)
        self.assertEqual(m.unique_id, '7')


if __name__ == "__main__":
    unittest.main(verbosity=2)
