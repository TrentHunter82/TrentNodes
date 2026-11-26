"""
Keystone Attention (H3) - make chosen DiT blocks read the keystone more.

The keystone is clip 1's frame 0: the pinned frame-0 keyframe (the "cond" rows of
the H3 layout, from Pin First Frame From Latent) and held video latent 0 (with
keep_look_of_clip_1). The carried frames are the other held video latents.

context slice: the keystone and the carried frames re-split only their own share
of each query's attention, the keystone weighted e^bias. Everything else keeps
its exact share. With p_k, p_c the two shares and o_k, o_c what each part reads:
    p_k' = (p_k + p_c) w p_k / (w p_k + p_c),   out' = out + (p_k' - p_k)(o_k - o_c)
whole sequence: bias added to the keystone scores, everything gives up a little.

Idea from roycho's Keystone Attention (Geiru Context Suite); own code.
"""

import math

import torch

import comfy.patcher_extension
from comfy_api.latest import io

KEY = "trent_keystone_held"


def parse_blocks(text):
    blocks = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        blocks.update(range(int(first), int(last or first) + 1))
    return blocks


def attention_lse(q, k, v):
    """Attention output [B, H, S, D] and each query's log-sum-exp [B, H, S, 1] (natural log of scaled scores)."""
    if q.is_cuda:
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True)[:2]
        return out, lse[..., :q.shape[2], None].float()
    scores = (q.float() @ k.float().transpose(-1, -2)) * q.shape[-1] ** -0.5
    lse = scores.logsumexp(-1, keepdim=True)
    return ((scores - lse).exp() @ v.float()).to(q.dtype), lse


def keystone_rows(layout, held):
    """Key row indices of the keystone and of the carried frames (lists of (start, stop))."""
    key = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    carried = []
    if held:
        a, b = next((a, b) for a, b, kind in layout.segments if kind == "video")
        per_latent = (b - a) // layout.signature[1]
        key.append((a, a + per_latent))
        if held > 1:
            carried.append((a + per_latent, a + held * per_latent))
    return key, carried


def _index(spans, device):
    return torch.cat([torch.arange(a, b, device=device) for a, b in spans])


class H3KeystoneAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3KeystoneAttention",
            display_name="Keystone Attention (H3)",
            category="Trent/MiniMax",
            description="Makes chosen blocks read the keystone (clip 1's frame 0: the pinned keyframe and held latent 0) harder than the carried frames.",
            search_aliases=["keystone", "pin attention", "h3 drift", "frame 0 bias"],
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("bias", default=1.0, min=-10.0, max=10.0, step=0.05,
                    tooltip="Keystone weight e^bias: 1 = 2.7x, 3 = 20x. 0 is off."),
                io.String.Input("blocks", default="30-39",
                    tooltip="DiT blocks to apply it in, e.g. '30-39' or '31-33, 36-44, 47'."),
                io.Combo.Input("mode", options=["context slice", "whole sequence", "held frames"], default="whole sequence",
                    tooltip="context slice: only the carried frames give up attention to the keystone (3 attention passes on these blocks). "
                            "whole sequence: a bias on the keystone scores, everything gives up a little. "
                            "held frames: the bias goes on the carried frames instead, so every token reads the held frames more "
                            "(idea from roycho's Context Attention)."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, bias, blocks, mode="whole sequence") -> io.NodeOutput:
        want = parse_blocks(blocks)
        if bias == 0 or not want:
            return io.NodeOutput(model)
        weight = math.exp(bias)
        m = model.clone()

        def count_held(executor, x, timestep, context, transformer_options, denoise_mask=None, **kwargs):
            held = 0
            if denoise_mask is not None:
                per_latent = denoise_mask[0].amax(dim=(0, 2, 3))
                while held < per_latent.shape[0] and per_latent[held] < 0.5:
                    held += 1
            options = transformer_options.copy()
            options[KEY] = held
            return executor(x, timestep, context, options, denoise_mask=denoise_mask, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, count_held)
        opts = m.model_options.setdefault("transformer_options", {})
        previous = opts.get("optimized_attention_override")

        def override(func, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            topts = kwargs.get("transformer_options") or {}
            layout = topts.get("minimax_h3_layout")
            rows = k.shape[-2] if skip_reshape else k.shape[1]
            # the token refiner also runs attention, over the text alone
            if layout is not None and rows == layout.seq_len and topts.get("block_index") in want:
                key, carried = keystone_rows(layout, topts.get(KEY, 0))
                spans = key if mode == "whole sequence" else carried if mode == "held frames" else None
                if spans:
                    row = torch.zeros([1, rows], dtype=q.dtype, device=k.device)
                    for a, b in spans:
                        row[:, a:b] = bias
                    mask = row if mask is None else mask + row
                elif mode == "context slice" and key and carried and mask is None and skip_reshape:
                    out, lse = attention_lse(q, k, v)
                    ki, ci = _index(key, k.device), _index(carried, k.device)
                    o_k, lse_k = attention_lse(q, k.index_select(2, ki), v.index_select(2, ki))
                    o_c, lse_c = attention_lse(q, k.index_select(2, ci), v.index_select(2, ci))
                    p_k, p_c = (lse_k - lse).exp(), (lse_c - lse).exp()
                    shift = (p_k + p_c) * weight * p_k / (weight * p_k + p_c) - p_k
                    out = torch.addcmul(out.float(), shift, o_k.float() - o_c.float()).to(q.dtype)
                    b, h, s, d = out.shape
                    return out.transpose(1, 2).reshape(b, s, h * d)
            if previous is not None:
                return previous(func, q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)

        opts["optimized_attention_override"] = override
        return io.NodeOutput(m)


NODE_CLASS_MAPPINGS = {"H3KeystoneAttention": H3KeystoneAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"H3KeystoneAttention": "Keystone Attention (H3)"}
