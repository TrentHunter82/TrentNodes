"""Graft one clean single-frame latent from an early clip into slot 0 of a chained H3 clip.

H3 video latents come in 17-frame blocks of 1 + 4 + 4 + 4 + 4 frames, so every
fifth latent (0, 5, 10, ...) holds one frame on its own. Copying such a latent
from clip 1 into the first slot of every later clip, locked with noise mask 0,
pulls colour and sharpness back toward clip 1 and slows chain drift. Slot 0 sits
inside the protected context prefix, which the join trims, so the grafted frame
never reaches the output. Idea: roycho87's Clown Keystone (clown_context_suite).
"""

import torch

from comfy.nested_tensor import NestedTensor


class MiniMaxH3KeystoneGraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Chained H3 AV latent to sample, e.g. the output "
                               "of H3 Generated AV Masked Context."}),
                "keystone": ("LATENT", {
                    "tooltip": "H3 AV latent from clip 1 at the same resolution, "
                               "e.g. Load Latent fixed on the primer's slot. A "
                               "plain [video, audio] list is accepted."}),
                "source_index": ("INT", {
                    "default": 0, "min": 0, "max": 9995, "step": 5,
                    "tooltip": "Keystone latent to copy. Must be a single-frame "
                               "latent: 0, 5, 10, ..."}),
                "mask": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Noise mask on slot 0. 0 = kept exactly, "
                               "0.2-0.3 = kept roughly, 1 = free."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "graft"
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "H3 Keystone Graft"
    DESCRIPTION = ("Copy one single-frame latent from clip 1 into slot 0 of a "
                   "chained H3 clip and lock it, so later clips drift less.")

    def graft(self, latent, keystone, source_index, mask):
        samples = latent["samples"]
        if not isinstance(samples, NestedTensor):
            raise ValueError("H3 Keystone Graft: latent is not an H3 AV latent.")
        video, audio = samples.tensors
        src = keystone["samples"]
        src = src.tensors[0] if isinstance(src, NestedTensor) else src[0]
        if source_index % 5 != 0 or source_index >= src.shape[2]:
            raise ValueError("H3 Keystone Graft: source_index %d is not a single-frame latent of the "
                             "%d-latent keystone (use 0, 5, 10, ...)." % (source_index, src.shape[2]))
        if src.shape[1:2] + src.shape[3:] != video.shape[1:2] + video.shape[3:]:
            raise ValueError("H3 Keystone Graft: keystone %s and latent %s differ in channels or size. "
                             "Use a keystone saved at this pass's resolution."
                             % (tuple(src.shape), tuple(video.shape)))

        video = video.clone()
        video[:, :, :1] = src[:, :, source_index:source_index + 1].to(video)

        noise_mask = latent.get("noise_mask")
        if noise_mask is None:
            video_mask = torch.ones((1, 1) + video.shape[2:], device=video.device)
            audio_mask = torch.ones((1, 1) + audio.shape[2:], device=audio.device)
        else:
            video_mask, audio_mask = noise_mask.tensors
            video_mask = video_mask.clone()
        video_mask[:, :, :1] = mask

        out = latent.copy()
        out["samples"] = NestedTensor((video, audio))
        out["noise_mask"] = NestedTensor((video_mask, audio_mask))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3KeystoneGraft": MiniMaxH3KeystoneGraft,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3KeystoneGraft": "H3 Keystone Graft",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
