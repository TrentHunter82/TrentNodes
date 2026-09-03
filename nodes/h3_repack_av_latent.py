"""Bridge node for carrying MiniMax H3 AV latents across workflow runs.

The H3 Motion Context Load Latent node (ComfyUI-H3-Motion-Context-MultiRef)
deliberately returns its two streams as a plain Python list, so the result
cannot be wired into VAEDecode by accident. The MMH3Tools nodes
(MMH3SplitAV, MMH3TrimAV, MMH3ConcatAV, ...) require the core
comfy.nested_tensor.NestedTensor pair instead, and their unpack_av() refuses
the plain list with "is a plain latent, not a MiniMax H3 AV latent".

This node converts the list form back into a NestedTensor pair, so a latent
saved in one run (Meta Batch requeue, manual chain, or a fresh session) can
be split, trimmed, or decoded in the next run. A latent that is already a
NestedTensor passes through untouched.
"""

from comfy.nested_tensor import NestedTensor


class MiniMaxH3RepackAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "An H3 AV latent whose samples are a plain "
                               "[video, audio] list — e.g. the output of "
                               "H3 Motion Context Load Latent. NestedTensor "
                               "latents pass through unchanged."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "repack"
    CATEGORY = "Trent/MiniMax"
    DISPLAY_NAME = "H3 Repack AV Latent"
    DESCRIPTION = ("Rewrap a cross-run H3 AV latent (plain [video, audio] "
                   "list) as the NestedTensor pair that MMH3Tools nodes and "
                   "VAE decodes require.")

    def repack(self, latent):
        samples = latent["samples"]
        if isinstance(samples, NestedTensor):
            return (latent,)
        if isinstance(samples, (list, tuple)):
            if len(samples) != 2:
                raise ValueError(
                    "H3 Repack AV Latent: expected [video, audio], got %d "
                    "stream(s). Wire the output of H3 Motion Context Load "
                    "Latent here." % len(samples))
            video, audio = samples
            if getattr(video, "ndim", 0) != 5:
                raise ValueError(
                    "H3 Repack AV Latent: first stream is not a 5D video "
                    "latent [B,24,T,h,w] (got shape %s)."
                    % (tuple(getattr(video, "shape", ())),))
            out = dict(latent)
            out["samples"] = NestedTensor([video, audio])
            return (out,)
        raise ValueError(
            "H3 Repack AV Latent: expected an H3 AV latent (NestedTensor or "
            "[video, audio] list), got %s. Plain single-tensor latents have "
            "no audio stream and cannot be repacked." % type(samples).__name__)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3RepackAVLatent": MiniMaxH3RepackAVLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3RepackAVLatent": "H3 Repack AV Latent",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
