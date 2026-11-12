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
        chain = {"clip": clip, "first_clip": clips_already_made, "clips": clips_already_made + clips,
                 "overlap": overlap_frames, "manager": meta_batch}
        start = song_offset + clip * (frames - overlap_frames) / FPS
        text = first_prompt if first or later_prompt is None else later_prompt
        return (chain, seed, start, ExecutionBlocker(None) if text is None else text, first)


STAGES = ("intro", "rising action", "conflict", "climax", "resolution")


def _arc_stage(clip, clips):
    # First clip is the intro, the last the resolution, the one before it the climax;
    # the clips between are split into rising action, then conflict.
    if clip == 0:
        return "intro"
    if clip == clips - 1:
        return "resolution"
    if clip == clips - 2:
        return "climax"
    middle = clips - 3
    return "rising action" if clip - 1 < (middle + 1) // 2 else "conflict"


class H3StoryArc:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Story Arc"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("beat", "stage")
    OUTPUT_TOOLTIPS = (
        "Where this clip sits in the story, with that stage's text. Add it to the Codex brief.",
        "The stage name only: intro, rising action, conflict, climax or resolution.",
    )
    FUNCTION = "beat"
    DESCRIPTION = ("Gives each clip of a Video Chain its place in a five-part story arc: clip 1 is the "
                   "intro, the last clip the resolution, the one before it the climax, and the clips "
                   "between are rising action, then conflict.")

    @classmethod
    def INPUT_TYPES(cls):
        text = lambda default, tip: ("STRING", {"default": default, "multiline": True, "tooltip": tip})
        return {
            "required": {
                "chain": ("H3_CHAIN", {"tooltip": "From Video Chain."}),
                "intro": text("Introduce the character, the place and the mood. Plant one strange detail.",
                              "Text for clip 1."),
                "rising_action": text("Something unexpected happens and the tension grows. Raise one question.",
                                      "Text for the clips after the intro."),
                "conflict": text("The threat becomes real and the character must act. Push the danger closer.",
                                 "Text for the clips between rising action and the climax."),
                "climax": text("The peak: the biggest, most intense moment of the story.",
                               "Text for the clip before the last."),
                "resolution": text("The aftermath. Show what changed, and end on a strong final image.",
                                   "Text for the last clip."),
            },
        }

    def beat(self, chain, intro, rising_action, conflict, climax, resolution):
        clip, clips = chain["clip"], chain["clips"]
        stage = _arc_stage(clip, clips)
        text = dict(zip(STAGES, (intro, rising_action, conflict, climax, resolution)))[stage]
        return ("STORY ARC: clip %d of %d, %s. %s" % (clip + 1, clips, stage.upper(), text), stage)


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
                    "tooltip": "Copies the first frame of clip 1 into the hidden first frame of every "
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
            out = MiniMaxH3KeystoneGraft().graft(out, {"samples": [first["first_video"], first["audio"]]}, 0, 0.0)[0]

        if vae is None:
            start = ExecutionBlocker("%s: connect the video VAE to get start_image on later clips." % who)
        else:
            # The first two latents decode to five frames; the first one is the start image.
            start = vae.decode(out["samples"].tensors[0][:, :, :2])
            start = start.reshape(-1, *start.shape[-3:])[:1]
        logging.info("[Video Chain] clip %d continues from %s (%d frames)", chain["clip"] + 1, path, n)
        return (out, start)


class _HeldNoise:
    """Sampler noise with the held latents swapped for noise unsampled from the clip before."""

    def __init__(self, noise, held):
        self.seed = noise.seed
        self.noise = noise
        self.held = held

    def generate_noise(self, input_latent):
        video, audio = self.noise.generate_noise(input_latent).unbind()
        video = video.clone()
        n = self.held.shape[2]
        held = self.held.to(video)
        # Unsampling a few-step model gives weak noise (std ~0.55). Fill the missing
        # variance with the fresh noise, so H3 still reads these rows as fully noised.
        std = held.std(dim=(0, 1, 3, 4), keepdim=True).clamp(max=1.0)
        video[:, :, :n] = held + (1.0 - std * std).sqrt() * video[:, :, :n]
        return NestedTensor((video, audio))


class H3ContextNoise:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Context Noise"
    RETURN_TYPES = ("LATENT", "NOISE")
    RETURN_NAMES = ("latent", "noise")
    OUTPUT_TOOLTIPS = (
        "Latent for the sampler, with the copied frames partly open to the noise.",
        "Noise for the sampler. With unsampled connected, the copied frames start from their own unsampled noise.",
    )
    FUNCTION = "open_context"
    DESCRIPTION = ("Put between Continue From Last Clip and the sampler. The copied frames get some "
                   "of the noise at every step, most at the start and none at the join, so H3 "
                   "continues them without copying their flaws. Clip 1 passes through.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "From Continue From Last Clip."}),
                "noise": ("NOISE", {"tooltip": "The sampler's noise, e.g. Random Noise."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Noise on the second copied latent (1 = as much as the new frames). "
                               "It fades to 0 at the join. The first latent always stays locked."}),
                "fade": (["linear", "ease"], {"default": "linear",
                    "tooltip": "ease falls faster, so more of the copied frames stay close to locked."}),
            },
            "optional": {
                "unsampled": ("LATENT", {"lazy": True,
                    "tooltip": "The clip before, sampled back with unsample_sigmas (e.g. from Take from "
                               "Last Time). The copied frames then start from noise that leads back to them."}),
                "unsample_sigmas": ("SIGMAS", {
                    "tooltip": "The rising sigmas that made unsampled. Connect both or neither. "
                               "The last sigma must be below 1."}),
            },
        }

    @staticmethod
    def _held(latent):
        mask = latent.get("noise_mask")
        if mask is None:
            return 0
        video_mask = mask.tensors[0]
        held = 0
        while held < video_mask.shape[2] and video_mask[:, :, held].max() < 0.5:
            held += 1
        return held

    def check_lazy_status(self, latent, noise, strength, fade, unsampled=None, unsample_sigmas=None):
        # unsampled is read only with unsample_sigmas connected, so it can stay unconnected
        need = unsample_sigmas is not None and strength > 0 and self._held(latent) > 2
        return ["unsampled"] if need else []

    def open_context(self, latent, noise, strength, fade, unsampled=None, unsample_sigmas=None):
        held = self._held(latent)
        if held <= 2 or strength <= 0:
            return (latent, noise)

        video_mask, audio_mask = latent["noise_mask"].tensors
        video_mask = video_mask.clone()
        ramp = held - 2  # latent 0 stays locked, the last held latent is the join and stays locked
        values = []
        for i in range(ramp):
            w = 1.0 - i / ramp
            m = strength * (w if fade == "linear" else w * w)
            video_mask[:, :, i + 1] = m
            values.append(round(m, 3))
        out = latent.copy()
        out["noise_mask"] = NestedTensor((video_mask, audio_mask))

        if unsampled is not None and unsample_sigmas is not None:
            s = float(unsample_sigmas[-1])
            if not 0.0 < s < 1.0:
                raise ValueError("Context Noise: unsample_sigmas must end between 0 and 1, not %g. "
                                 "Split off the last sigma of the flipped schedule." % s)
            # The sampler returns x_s / (1 - s) with x_s = s * n + (1 - s) * x0, so n = (1 - s) / s * (out - x0).
            x0 = latent["samples"].tensors[0][:, :, :held]
            back = unsampled["samples"].tensors[0][:, :, -held:].to(x0)
            noise = _HeldNoise(noise, (1.0 - s) / s * (back - x0))
        logging.info("[Video Chain] context noise on copied latents 1-%d: %s%s", ramp, values,
                     " (unsampled)" if isinstance(noise, _HeldNoise) else "")
        return (out, noise)


class H3SwapUnsampledContext:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Swap In Unsampled Context"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = ("Latent for the second half of the sampling (no added noise).",)
    FUNCTION = "swap"
    DESCRIPTION = ("Split unsample/resample for chained clips. Sample the first half of the steps with "
                   "the copied frames locked, then this puts the clip before, unsampled to the same "
                   "sigma, into the copied frames and opens them. The second half then denoises them "
                   "from their own unsampled state. The first latent and the join stay locked. "
                   "Clip 1 passes through.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "From Continue From Last Clip (before sampling)."}),
                "sampled": ("LATENT", {"tooltip": "The first half: the sampler run from the top sigma to the split sigma."}),
                "unsampled": ("LATENT", {"tooltip": "The clip before, unsampled from 0 up to the same split sigma "
                                                    "(e.g. Take from Last Time). Ignored on clip 1."}),
            },
        }

    def swap(self, latent, sampled, unsampled):
        held = H3ContextNoise._held(latent)
        if held <= 2:
            return (sampled,)
        clean_video, clean_audio = latent["samples"].tensors
        video_mask, audio_mask = latent["noise_mask"].tensors
        video, audio = sampled["samples"].tensors
        # A sampler that stops above sigma 0 returns x / (1 - sigma), which also scales the locked
        # latents. The next sampler reads locked latents as clean, so put the clean ones back.
        video = torch.where(video_mask.to(video.device) == 0, clean_video.to(video), video)
        audio = torch.where(audio_mask.to(audio.device) == 0, clean_audio.to(audio), audio)
        video[:, :, 1:held - 1] = unsampled["samples"].tensors[0][:, :, -held:][:, :, 1:held - 1].to(video)
        video_mask = video_mask.clone()
        video_mask[:, :, 1:held - 1] = 1.0
        out = sampled.copy()
        out["samples"] = NestedTensor((video, audio))
        out["noise_mask"] = NestedTensor((video_mask, audio_mask))
        logging.info("[Video Chain] unsampled context swapped into copied latents 1-%d", held - 2)
        return (out,)


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
        ending = {
            "video": video[:, :, -_video_steps(n):].detach().cpu().contiguous(),
            "audio": audio[..., -_audio_steps(n):].detach().cpu().contiguous(),
        }
        if chain["clip"] == 0:
            # Frame 0 of clip 1 is the keystone for keep_look_of_clip_1.
            ending["first_video"] = video[:, :, :1].detach().cpu().contiguous()
        comfy.utils.save_torch_file(ending, path)
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
            "optional": {
                "match_colour_frames": ("INT", {"default": 0, "min": 0, "max": 9999,
                    "tooltip": "0 = off. Each clip starts in the colour the clip before ended in, "
                               "measured on the copied frames, and fades to its own colour over "
                               "this many frames. Hides a colour step at the join."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def join(self, chain, images, audio, saved, match_colour_frames=0):
        clip, manager = chain["clip"], chain["manager"]
        n = 0 if clip == 0 else chain["overlap"]
        if n >= images.shape[0]:
            raise ValueError("Join Clips: the clip has %d frames, the overlap is %d." % (images.shape[0], n))
        folder = os.path.dirname(saved[-1])
        look_path = os.path.join(folder, "join_look.safetensors")
        copied, images = images[:n], images[n:]
        if match_colour_frames > 0:
            if n > 0 and os.path.isfile(look_path):
                # The copied frames show the same picture as the end of the clip before, so the
                # difference in their channel mean and spread is the colour step of this join.
                look = comfy.utils.load_torch_file(look_path)
                mean, std = copied.mean(dim=(0, 1, 2)), copied.std(dim=(0, 1, 2)).clamp(min=1e-6)
                gain = look["std"].to(images) / std
                fade = min(match_colour_frames, images.shape[0])
                weight = torch.linspace(1.0, 0.0, fade + 1, device=images.device, dtype=images.dtype)[:fade].view(-1, 1, 1, 1)
                matched = ((images[:fade] - mean) * gain + look["mean"].to(images)).clamp(0.0, 1.0)
                images = torch.cat((torch.lerp(images[:fade], matched, weight), images[fade:]))
            end = images[-chain["overlap"]:]
            comfy.utils.save_torch_file({"mean": end.mean(dim=(0, 1, 2)), "std": end.std(dim=(0, 1, 2))}, look_path)

        # Cut and pad to the exact picture length, so small sound offsets cannot add up down the chain.
        sr = int(audio["sample_rate"])
        start = round(n / FPS * sr)
        length = round(images.shape[0] / FPS * sr)
        wave = audio["waveform"][..., start:start + length]
        wave = torch.nn.functional.pad(wave, (0, length - wave.shape[-1]))
        wave = wave.detach().to(device="cpu", dtype=torch.float32).contiguous()

        # Each clip's sound goes to disk beside the last saved ending, which clip 1 clears.
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


class H3PinFirstFrameFromLatent:
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "Pin First Frame From Latent"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "pin"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive": ("CONDITIONING",), "latent": ("LATENT",)}}

    def pin(self, positive, latent):
        mask = latent.get("noise_mask")
        if mask is None or mask.tensors[0][:, :, :1].max() > 0:
            return (positive,)
        first = _streams(latent, "Pin First Frame From Latent")[0][:, :, :1]
        out = []
        for t, d in positive:
            d = d.copy()
            keyframes = [dict(kf, latent=first.to(kf["latent"])) if kf["resolved_frame_index"] == 0 and "latent" in kf else kf
                         for kf in d.get("minimax_keyframes", [])]
            if not any(kf["resolved_frame_index"] == 0 and "latent" in kf for kf in keyframes):
                # Reference to Video has no first frame guide, so the pin adds one.
                keyframes.append({"resolved_frame_index": 0, "latent": first.clone()})
            d["minimax_keyframes"] = keyframes
            out.append([t, d])
        return (out,)


NODE_CLASS_MAPPINGS = {
    "H3VideoChain": H3VideoChain,
    "H3ContinueFromLastClip": H3ContinueFromLastClip,
    "H3ContextNoise": H3ContextNoise,
    "H3StoryArc": H3StoryArc,
    "H3SwapUnsampledContext": H3SwapUnsampledContext,
    "H3RememberClipEnding": H3RememberClipEnding,
    "H3JoinClips": H3JoinClips,
    "H3PinFirstFrameFromLatent": H3PinFirstFrameFromLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.DISPLAY_NAME for k, v in NODE_CLASS_MAPPINGS.items()}
