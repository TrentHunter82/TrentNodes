"""Plain-English MiniMax H3 chain: Video Chain, Continue From Last Clip,
Remember Clip Ending and Join Clips.

One VHS Meta Batch Manager runs the graph once per clip. Clip 1 is a normal
generation. Every later clip copies the last `overlap` frames of the clip
before it into its own start, locked with noise mask 0, so it carries on
exactly where that clip ended. Join Clips drops the copied frames again and
keeps every clip's sound for the final mux.

The `chain` wire carries the clip number and the overlap from Video Chain to
the other three nodes, so the overlap is set in one place. Endings are saved
per name under output/h3_chain/<name>/, so a two-pass graph uses one name per
pass (e.g. "low res" and "high res").
"""

import logging
import os
import shutil

import torch
import safetensors.torch

import comfy.utils
import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_execution.graph import ExecutionBlocker

from .h3_chain_batch import H3ChainStep
from .h3_keystone_graft import MiniMaxH3KeystoneGraft
from ..utils.for_next_time import sanitize_slot

FPS = 24
AUDIO_HZ = 40


def _video_steps(frames):
    # H3 video latents cover 1, 4, 4, 4, 4 frames, so a 17k+5 frame run is 5k+2 latents.
    return 2 + 5 * ((frames - 5) // 17)


def _audio_steps(frames):
    return round(frames / FPS * AUDIO_HZ)


def _ending_path(name, clip):
    return os.path.join(folder_paths.get_output_directory(), "h3_chain",
                        sanitize_slot(name), "clip_%04d.latent" % clip)


def _streams(latent, who):
    samples = latent["samples"]
    if not isinstance(samples, NestedTensor):
        raise ValueError("%s: this is not a MiniMax H3 AV latent." % who)
    return samples.tensors


class H3VideoChain:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Video Chain"
    RETURN_TYPES = ("H3_CHAIN", "INT", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("chain", "seed", "song_start", "prompt", "is_first_clip")
    OUTPUT_TOOLTIPS = (
        "Connect to Continue From Last Clip, Remember Clip Ending and Join Clips.",
        "A new seed for each clip. Connect to RandomNoise.",
        "Where this clip starts in the song, in seconds. Connect to Trim Audio Duration start_index.",
        "first_prompt on clip 1, later_prompt on every other clip.",
        "True on clip 1 only.",
    )
    FUNCTION = "run"
    DESCRIPTION = ("Makes a video from several H3 clips with one queue. Connect a VHS Meta "
                   "Batch Manager, set how many clips, and queue once. Every clip after the "
                   "first continues where the clip before it ended.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meta_batch": ("VHS_BatchManager", {"forceInput": True,
                    "tooltip": "VHS Meta Batch Manager. Its frames_per_batch is the length of every clip."}),
                "clips": ("INT", {"default": 4, "min": 1, "max": 1000,
                    "tooltip": "How many clips this run makes."}),
                "seed": ("INT", {"default": 987654321, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed of clip 1. Each later clip adds 1."}),
                "overlap_frames": ("INT", {"default": 39, "min": 39, "max": 9999, "step": 51,
                    "tooltip": "Frames each clip copies from the end of the clip before it. "
                               "39, 90, 141, ... are the lengths where H3 video and audio line up."}),
                "song_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                    "tooltip": "Where clip 1 starts in the song, in seconds."}),
            },
            "optional": {
                "first_prompt": ("STRING", {"forceInput": True, "tooltip": "Prompt for clip 1."}),
                "later_prompt": ("STRING", {"forceInput": True,
                    "tooltip": "Prompt for every later clip. Leave empty to use first_prompt for all clips."}),
                "clips_already_made": ("INT", {"default": 0, "min": 0, "max": 1000,
                    "tooltip": "0 starts a new video. To add clips to the last run, set how many clips "
                               "it made; this run then continues from its last clip. Keep every "
                               "other setting the same."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, meta_batch, clips, seed, overlap_frames, song_offset, prompt, unique_id,
            first_prompt=None, later_prompt=None, clips_already_made=0):
        frames = meta_batch.frames_per_batch
        if (overlap_frames - 39) % 51 or overlap_frames >= frames:
            raise ValueError("Video Chain: overlap_frames must be 39, 90, 141, ... and shorter "
                             "than a clip (%d frames)." % frames)
        step = H3ChainStep().advance(meta_batch, clips - 1, seed, prompt, unique_id, primer_pass=True)[0]
        clip = clips_already_made + step
        first = clip == 0
        seed = (seed + clip) % (1 << 64)
        chain = {"clip": clip, "first_clip": clips_already_made, "overlap": overlap_frames, "manager": meta_batch}
        start = song_offset + clip * (frames - overlap_frames) / FPS
        text = first_prompt if first or later_prompt is None else later_prompt
        return (chain, seed, start, ExecutionBlocker(None) if text is None else text, first)


class H3ContinueFromLastClip:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Continue From Last Clip"
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "start_image")
    OUTPUT_TOOLTIPS = (
        "Latent for the sampler. Clip 1: unchanged. Later clips: starts with the end of the clip before.",
        "first_image on clip 1, the first copied frame on later clips. Connect to first_frame of MiniMax H3 Image to Video.",
    )
    FUNCTION = "continue_clip"
    DESCRIPTION = ("On clip 1 this does nothing. On every later clip it copies the saved end "
                   "of the clip before into the start of this latent and locks it, so the "
                   "video carries on without a jump.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": ("H3_CHAIN", {"tooltip": "From Video Chain."}),
                "latent": ("LATENT", {"tooltip": "This clip's latent before sampling: an empty H3 "
                                                 "AV latent, or the upscaled latent in a second pass."}),
                "name": ("STRING", {"default": "low res",
                    "tooltip": "Same name as on Remember Clip Ending. Use one name per pass size."}),
                "keep_look_of_clip_1": ("BOOLEAN", {"default": True,
                    "tooltip": "Copies one clean frame of clip 1 into the hidden first frame of every "
                               "clip, so colours and faces drift less over a long chain. "
                               "The frame is cut away again, so it never shows."}),
            },
            "optional": {
                "first_image": ("IMAGE", {"tooltip": "Start image of clip 1."}),
                "vae": ("VAE", {"tooltip": "Video VAE. Needed only for start_image on later clips."}),
            },
        }

    def continue_clip(self, chain, latent, name, keep_look_of_clip_1, first_image=None, vae=None):
        if chain["clip"] == 0:
            return (latent, ExecutionBlocker(None) if first_image is None else first_image)

        who = "Continue From Last Clip"
        path = _ending_path(name, chain["clip"] - 1)
        if not os.path.isfile(path):
            raise ValueError("%s: no saved ending at %s. Use the same name on Remember Clip "
                             "Ending, and queue a fresh run." % (who, path))
        data = safetensors.torch.load_file(path, device="cpu")
        src_video, src_audio = data["video"], data["audio"]
        video, audio = _streams(latent, who)
        n = chain["overlap"]
        v_n, a_n = _video_steps(n), _audio_steps(n)
        if src_video.shape[2] < v_n or v_n >= video.shape[2] or a_n >= audio.shape[-1]:
            raise ValueError("%s: the overlap of %d frames does not fit this clip or the saved "
                             "ending." % (who, n))
        if src_video.shape[1:2] + src_video.shape[3:] != video.shape[1:2] + video.shape[3:]:
            raise ValueError("%s: saved ending %s and this latent %s differ in size. Use one "
                             "name per pass size." % (who, tuple(src_video.shape), tuple(video.shape)))

        video = video.clone()
        audio = audio.clone()
        video[:, :, :v_n] = src_video[:, :, -v_n:].to(video)
        audio[..., :a_n] = src_audio[..., -a_n:].to(audio)
        video_mask = torch.ones((1, 1) + video.shape[2:], device=video.device)
        audio_mask = torch.ones((1, 1) + audio.shape[2:], device=audio.device)
        video_mask[:, :, :v_n] = 0.0
        audio_mask[..., :a_n] = 0.0
        out = latent.copy()
        out["samples"] = NestedTensor((video, audio))
        out["noise_mask"] = NestedTensor((video_mask, audio_mask))

        if keep_look_of_clip_1:
            first = safetensors.torch.load_file(_ending_path(name, 0), device="cpu")
            out = MiniMaxH3KeystoneGraft().graft(out, {"samples": [first["video"], first["audio"]]}, 0, 0.0)[0]

        if vae is None:
            start = ExecutionBlocker("%s: connect the video VAE to get start_image on later clips." % who)
        else:
            # The first two latents decode to five frames; the first one is the start image.
            start = vae.decode(out["samples"].tensors[0][:, :, :2])
            start = start.reshape(-1, *start.shape[-3:])[:1]
        logging.info("[Video Chain] clip %d continues from %s (%d frames)", chain["clip"] + 1, path, n)
        return (out, start)


class H3RememberClipEnding:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Remember Clip Ending"
    RETURN_TYPES = ("H3_SAVED",)
    RETURN_NAMES = ("saved",)
    OUTPUT_TOOLTIPS = ("Connect to Join Clips, or to after on another Remember Clip Ending.",)
    FUNCTION = "remember"
    OUTPUT_NODE = True
    DESCRIPTION = ("Saves the end of this clip so the next clip can continue from it. "
                   "Clip 1 also clears endings left over from an earlier run.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": ("H3_CHAIN", {"tooltip": "From Video Chain."}),
                "latent": ("LATENT", {"tooltip": "The sampled latent of this clip."}),
                "name": ("STRING", {"default": "low res",
                    "tooltip": "Same name as on Continue From Last Clip."}),
            },
            "optional": {
                "after": ("H3_SAVED", {"tooltip": "saved from another Remember Clip Ending. "
                                                  "Join Clips then waits for both saves."}),
            },
        }

    def remember(self, chain, latent, name, after=None):
        video, audio = _streams(latent, "Remember Clip Ending")
        n = chain["overlap"]
        path = _ending_path(name, chain["clip"])
        folder = os.path.dirname(path)
        if chain["clip"] == 0:
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        comfy.utils.save_torch_file({
            "video": video[:, :, -_video_steps(n):].detach().cpu().contiguous(),
            "audio": audio[..., -_audio_steps(n):].detach().cpu().contiguous(),
        }, path)
        return ((after or []) + [path],)


class H3JoinClips:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Join Clips"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    OUTPUT_TOOLTIPS = (
        "This clip without the copied frames. Connect to VHS Video Combine.",
        "The whole soundtrack on the last clip. Connect to the same VHS Video Combine.",
    )
    FUNCTION = "join"
    OUTPUT_NODE = True
    DESCRIPTION = ("Cuts the copied frames off every clip after the first and keeps each "
                   "clip's sound, so VHS Video Combine gets one continuous video and song.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": ("H3_CHAIN", {"tooltip": "From Video Chain."}),
                "images": ("IMAGE", {"tooltip": "Decoded frames of this clip."}),
                "audio": ("AUDIO", {"tooltip": "Sound of this clip: the song cut or the decoded audio."}),
                "saved": ("H3_SAVED", {"tooltip": "From Remember Clip Ending. The next clip "
                                                  "starts only after the ending is saved."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def join(self, chain, images, audio, saved):
        clip, manager = chain["clip"], chain["manager"]
        n = 0 if clip == 0 else chain["overlap"]
        if n >= images.shape[0]:
            raise ValueError("Join Clips: the clip has %d frames, the overlap is %d." % (images.shape[0], n))
        images = images[n:]

        # Cut and pad to the exact picture length, so small sound offsets cannot add up down the chain.
        sr = int(audio["sample_rate"])
        start = round(n / FPS * sr)
        length = round(images.shape[0] / FPS * sr)
        wave = audio["waveform"][..., start:start + length]
        wave = torch.nn.functional.pad(wave, (0, length - wave.shape[-1]))
        wave = wave.detach().to(device="cpu", dtype=torch.float32).contiguous()

        # Each clip's sound goes to disk beside the last saved ending, which clip 1 clears.
        folder = os.path.dirname(saved[-1])
        comfy.utils.save_torch_file({"waveform": wave, "sample_rate": torch.tensor(sr)},
                                    os.path.join(folder, "sound_%04d.safetensors" % clip))
        out = {"waveform": wave, "sample_rate": sr}
        if manager.has_closed_inputs:
            # VHS ignores audio until the last clip, then muxes the whole track.
            chunks = []
            for k in range(chain["first_clip"], clip + 1):
                path = os.path.join(folder, "sound_%04d.safetensors" % k)
                if not os.path.isfile(path):
                    raise ValueError("Join Clips: the sound of clip %d is missing (%s). Queue a fresh run." % (k + 1, path))
                part = comfy.utils.load_torch_file(path)
                if int(part["sample_rate"]) != sr or part["waveform"].shape[:-1] != wave.shape[:-1]:
                    raise ValueError("Join Clips: the sample rate or the audio channels changed between clips.")
                chunks.append(part["waveform"])
            out = {"waveform": torch.cat(chunks, dim=-1), "sample_rate": sr}
        logging.info("[Video Chain] clip %d joined after %s", clip + 1, ", ".join(saved))
        return (images, out)


NODE_CLASS_MAPPINGS = {
    "H3VideoChain": H3VideoChain,
    "H3ContinueFromLastClip": H3ContinueFromLastClip,
    "H3RememberClipEnding": H3RememberClipEnding,
    "H3JoinClips": H3JoinClips,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.DISPLAY_NAME for k, v in NODE_CLASS_MAPPINGS.items()}
