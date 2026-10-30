"""
Tests for Video Chain, Continue From Last Clip, Remember Clip Ending and Join Clips.

Run:
    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_h3_video_chain.py

Runs a whole 3-clip chain on the CPU with H3-shaped tensors. Synthesizes the
TrentNodes package so the pack's discovery never runs, and points folder_paths
at a temp directory so the real output folder is never touched.
"""

import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")
sys.path.insert(0, ROOT)

pkg = types.ModuleType("TrentNodes")
pkg.__path__ = [PKG]
sys.modules["TrentNodes"] = pkg
for _sub in ("nodes", "utils"):
    _mod = types.ModuleType("TrentNodes.{}".format(_sub))
    _mod.__path__ = [os.path.join(PKG, _sub)]
    sys.modules["TrentNodes.{}".format(_sub)] = _mod

import folder_paths  # noqa: E402
import torch  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402
from comfy_execution.graph import ExecutionBlocker  # noqa: E402

chain_mod = importlib.import_module("TrentNodes.nodes.h3_video_chain")

FRAMES = 141          # 42 video latents, 235 audio latents
OVERLAP = 39          # 12 video latents, 65 audio latents
SR = 48000


class FakeVAE:
    def decode(self, video):
        # [B, C, T, H, W] -> [B, frames, H*16, W*16, 3], frame value = latent channel 0
        b, _, t, h, w = video.shape
        frames = 1 + 4 * (t - 1)
        out = video[:, :1, :1].mean().expand(b, frames, h * 16, w * 16, 3)
        return out.clone()


class VideoChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="h3vc_test_")
        folder_paths.output_directory = self.tmp

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def manager(self):
        m = SimpleNamespace(unique_id="7", inputs={}, frames_per_batch=FRAMES,
                            has_closed_inputs=False)
        m.reset = lambda: setattr(m, "unique_id", None)
        return m

    def chain(self, m, clip, clips=3, **kw):
        prompt = {"7": {"inputs": {"requeue": clip}},
                  "vc": {"inputs": {"meta_batch": ["7", 0]}}}
        args = dict(clips=clips, seed=10, overlap_frames=OVERLAP, song_offset=7.0,
                    first_prompt="first", later_prompt="later")
        args.update(kw)
        return chain_mod.H3VideoChain().run(m, prompt=prompt, unique_id="vc", **args)

    def empty(self):
        return {"samples": NestedTensor((torch.zeros(1, 24, 42, 4, 6), torch.zeros(1, 32, 2, 235)))}

    def sampled(self, clip):
        # Every latent step holds its clip and step number, so copies can be traced.
        video = torch.arange(42).float().view(1, 1, 42, 1, 1).expand(1, 24, 42, 4, 6) + 1000 * clip
        audio = torch.arange(235).float().view(1, 1, 1, 235).expand(1, 32, 2, 235) + 1000 * clip
        return {"samples": NestedTensor((video.clone(), audio.clone()))}

    def song_cut(self, clip):
        return {"waveform": torch.full((1, 2, round(5.88 * SR)), float(clip)), "sample_rate": SR}

    def test_full_chain(self):
        m = self.manager()
        cont = chain_mod.H3ContinueFromLastClip()
        remember = chain_mod.H3RememberClipEnding()
        image = torch.rand(1, 64, 96, 3)
        all_images = []

        for clip in range(3):
            chain, seed, start, text, first = self.chain(m, clip)
            self.assertEqual(seed, 10 + clip)
            self.assertAlmostEqual(start, 7.0 + clip * 4.25)
            self.assertEqual(text, "first" if clip == 0 else "later")
            self.assertEqual(first, clip == 0)

            latent, start_image = cont.continue_clip(chain, self.empty(), "low res", True,
                                                     first_image=image, vae=FakeVAE())
            video, audio = latent["samples"].tensors
            if clip == 0:
                self.assertIs(start_image, image)
                self.assertNotIn("noise_mask", latent)
            else:
                prev = 1000 * (clip - 1)
                # latents 1..11 = the previous clip's last 12 latents (30..41); latent 0 = clip 1's latent 30
                self.assertTrue(torch.equal(video[0, 0, 1:12, 0, 0], torch.arange(31, 42).float() + prev))
                self.assertEqual(video[0, 0, 0, 0, 0].item(), 30.0)
                self.assertTrue(torch.equal(video[0, 0, 12:], torch.zeros(30, 4, 6)))
                self.assertTrue(torch.equal(audio[0, 0, 0, :65], torch.arange(170, 235).float() + prev))
                vm, am = latent["noise_mask"].tensors
                self.assertTrue((vm[0, 0, :12] == 0).all() and (vm[0, 0, 12:] == 1).all())
                self.assertTrue((am[..., :65] == 0).all() and (am[..., 65:] == 1).all())
                self.assertEqual(start_image.shape, (1, 64, 96, 3))
                self.assertAlmostEqual(start_image.mean().item(), 30.0, places=4)

            saved = remember.remember(chain, self.sampled(clip), "low res")[0]
            frames = torch.full((FRAMES, 64, 96, 3), float(clip))
            # A new instance per clip: ComfyUI may not keep node objects between queues.
            images, sound = chain_mod.H3JoinClips().join(chain, frames, self.song_cut(clip), saved)
            self.assertEqual(images.shape[0], FRAMES if clip == 0 else FRAMES - OVERLAP)
            all_images.append(images)

        self.assertTrue(m.has_closed_inputs)
        total = sum(i.shape[0] for i in all_images)
        self.assertEqual(total, 141 + 2 * 102)
        self.assertEqual(sound["waveform"].shape[-1], round(141 / 24 * SR) + 2 * round(102 / 24 * SR))
        self.assertTrue(torch.equal(sound["waveform"][..., -1], torch.full((1, 2), 2.0)))

    def test_continue_a_finished_run(self):
        remember = chain_mod.H3RememberClipEnding()
        m = self.manager()
        for clip in range(3):
            chain = self.chain(m, clip)[0]
            saved = remember.remember(chain, self.sampled(clip), "low res")[0]
            chain_mod.H3JoinClips().join(chain, torch.zeros(FRAMES, 8, 8, 3), self.song_cut(clip), saved)

        m = self.manager()
        cont = chain_mod.H3ContinueFromLastClip()
        for step in range(2):
            chain, seed, start, text, first = self.chain(m, step, clips=2, clips_already_made=3)
            clip = 3 + step
            self.assertEqual((chain["clip"], seed, first, text), (clip, 10 + clip, False, "later"))
            self.assertAlmostEqual(start, 7.0 + clip * 4.25)
            latent = cont.continue_clip(chain, self.empty(), "low res", True)[0]
            self.assertEqual(latent["samples"].tensors[0][0, 0, 1, 0, 0].item(), 31.0 + 1000 * (clip - 1))
            saved = remember.remember(chain, self.sampled(clip), "low res")[0]
            images, sound = chain_mod.H3JoinClips().join(chain, torch.zeros(FRAMES, 8, 8, 3), self.song_cut(clip), saved)
            self.assertEqual(images.shape[0], FRAMES - OVERLAP)
        # Only the new clips' sound, so it lines up with the new video file.
        self.assertTrue(m.has_closed_inputs)
        self.assertEqual(sound["waveform"].shape[-1], 2 * round(102 / 24 * SR))
        self.assertEqual(sound["waveform"][..., 0].tolist(), [[3.0, 3.0]])

    def test_keep_look_off_and_no_vae(self):
        m = self.manager()
        cont, remember = chain_mod.H3ContinueFromLastClip(), chain_mod.H3RememberClipEnding()
        chain = self.chain(m, 0)[0]
        remember.remember(chain, self.sampled(0), "hi")
        chain = self.chain(m, 1)[0]
        latent, start = cont.continue_clip(chain, self.empty(), "hi", False)
        self.assertEqual(latent["samples"].tensors[0][0, 0, 0, 0, 0].item(), 30.0)
        self.assertIsInstance(start, ExecutionBlocker)

    def test_clip_1_clears_old_endings(self):
        m = self.manager()
        remember = chain_mod.H3RememberClipEnding()
        for clip in range(3):
            remember.remember(self.chain(m, clip)[0], self.sampled(clip), "a")
        remember.remember(self.chain(self.manager(), 0)[0], self.sampled(0), "a")
        folder = os.path.join(self.tmp, "h3_chain", "a")
        self.assertEqual(os.listdir(folder), ["clip_0000.latent"])

    def test_after_collects_both_saves(self):
        m = self.manager()
        chain = self.chain(m, 0)[0]
        remember = chain_mod.H3RememberClipEnding()
        lo = remember.remember(chain, self.sampled(0), "low res")[0]
        both = remember.remember(chain, self.sampled(0), "high res", after=lo)[0]
        self.assertEqual(len(both), 2)

    def test_errors(self):
        m = self.manager()
        cont = chain_mod.H3ContinueFromLastClip()
        self.chain(m, 0)
        chain = self.chain(m, 1)[0]
        with self.assertRaisesRegex(ValueError, "no saved ending"):
            cont.continue_clip(chain, self.empty(), "missing", True)
        with self.assertRaisesRegex(ValueError, "overlap_frames"):
            self.chain(self.manager(), 0, overlap_frames=40)
        chain_mod.H3RememberClipEnding().remember(self.chain(self.manager(), 0)[0], self.sampled(0), "x")
        m = self.manager()
        self.chain(m, 0)
        chain = self.chain(m, 1)[0]
        bigger = {"samples": NestedTensor((torch.zeros(1, 24, 42, 8, 12), torch.zeros(1, 32, 2, 235)))}
        with self.assertRaisesRegex(ValueError, "differ in size"):
            cont.continue_clip(chain, bigger, "x", True)

    def test_one_prompt_for_all_clips(self):
        m = self.manager()
        self.chain(m, 0, later_prompt=None)
        self.assertEqual(self.chain(m, 1, later_prompt=None)[3], "first")


if __name__ == "__main__":
    unittest.main()
