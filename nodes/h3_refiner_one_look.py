"""
Refiner One Look (H3) - give a whole clip the body's colour correction.

A colour refiner such as noEmbryo's H3 Clip Refiner gives the head, body and
tail of a clip their own per-channel affine. In a chain the steps between them
flash: the VAE carries the head/body step into the first new frames after the
join, and the body/tail step shows 39 frames before the next join. The head is
dropped by Join Clips anyway. This node fits refined = a * original + b for each
latent and channel (on 4x4-pooled latents), takes the body's a and b (median of
the middle latents) and applies them to every latent. What the affine does not
explain (detail transfer, band gain) is kept as is.
"""

import torch
import torch.nn.functional as F

from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io


def split_av(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        return list(samples.unbind())
    return [samples]


def one_look(original, refined):
    x = original.to(refined.device, torch.float32)
    y = refined.float()
    # fit on 4x4-pooled latents: a refiner's fine-band gain would otherwise leak into a per latent
    xp, yp = (F.avg_pool3d(t, (1, 4, 4), ceil_mode=True) for t in (x, y))
    dims = (0, 3, 4)
    mx, my = xp.mean(dims, keepdim=True), yp.mean(dims, keepdim=True)
    var = (xp - mx).square().mean(dims, keepdim=True).clamp(min=1e-8)
    a = ((xp - mx) * (yp - my)).mean(dims, keepdim=True) / var
    b = my - a * mx
    t = x.shape[2]
    mid = slice(max(t // 2 - 2, 0), t // 2 + 3)
    a_body = a[:, :, mid].median(2, keepdim=True).values
    b_body = b[:, :, mid].median(2, keepdim=True).values
    return torch.addcmul(y + (b_body - b), a_body - a, x).to(refined.dtype)


class H3RefinerOneLook(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefinerOneLook",
            display_name="Refiner One Look (H3)",
            category="Trent/MiniMax",
            description="Gives the whole clip the clip refiner's body correction, so its head/body/tail edges do not flash at the chain joins.",
            search_aliases=["refiner flash", "clip refiner", "h3 seam", "seam flash"],
            inputs=[
                io.Latent.Input("original", tooltip="The latent that went into the refiner."),
                io.Latent.Input("refined", tooltip="The refiner output."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, original, refined) -> io.NodeOutput:
        parts = split_av(refined)
        video = one_look(split_av(original)[0], parts[0])
        out = dict(refined)
        out["samples"] = NestedTensor([video] + parts[1:]) if len(parts) > 1 else video
        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"H3RefinerOneLook": H3RefinerOneLook}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RefinerOneLook": "Refiner One Look (H3)"}
