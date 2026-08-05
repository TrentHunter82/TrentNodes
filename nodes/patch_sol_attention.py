"""Patch node that swaps model attention for NVIDIA's Sol-Attn sparse kernel.

Sol-Attn ("Sol-Attn: Accelerating Video Generation Inference via On-the-Fly
Attention Sparsification", arXiv:2607.24027, NVlabs) is a training-free
sparse attention for long-sequence video DiTs (Wan, HunyuanVideo, LTX, ...).
This node follows the KJNodes patch pattern: it clones the model and installs
an `optimized_attention_override` in transformer_options, so it composes with
other patch nodes and is removed by bypassing the node.

Only long self-attention calls are routed to the sparse kernel; everything
else (cross attention, masked attention, short sequences, head_dim != 128,
the dense warmup steps) falls through to the model's normal attention.
"""

import logging

import torch

from ..utils.sol_attn_triton import HEAD_DIM, sol_attn


def _step_progress(transformer_options, cache):
    """Return sampling progress in [0, 1], best effort.

    Uses the current sigma's position inside the full schedule. The result is
    cached per sigmas-tensor object so the CPU sync happens once per model
    forward, not once per attention call.
    """
    sigmas = transformer_options.get("sigmas")
    sample_sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None or sample_sigmas is None:
        return 1.0
    key = id(sigmas)
    progress = cache.get(key)
    if progress is None:
        try:
            current = float(sigmas.flatten()[0])
            schedule = sample_sigmas.detach().float().flatten().cpu()
            index = int((schedule - current).abs().argmin())
            progress = index / max(len(schedule) - 1, 1)
        except Exception:
            progress = 1.0
        if len(cache) > 256:
            cache.clear()
        cache[key] = progress
    return progress


def make_sol_attention_override(tau, dense_start_percent, min_tokens):
    """Build an optimized_attention_override closure for ComfyUI wrap_attn."""
    progress_cache = {}
    state = {"kernel_broken": False, "logged_active": False}

    @torch.compiler.disable()
    def run_sol(q_bthd, k_bthd, v_bthd, scale):
        return sol_attn(q_bthd, k_bthd, v_bthd, scale=scale, tau=tau)

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        if (
            state["kernel_broken"]
            or mask is not None
            or kwargs.get("enable_gqa", False)
            or not q.is_cuda
            or q.shape != k.shape
            or q.shape != v.shape
        ):
            return dense()

        if skip_reshape:
            b, h, tokens, dim_head = q.shape
        else:
            b, tokens, inner = q.shape
            dim_head = inner // heads

        if dim_head != HEAD_DIM or tokens < min_tokens:
            return dense()

        transformer_options = kwargs.get("transformer_options", {})
        if _step_progress(transformer_options, progress_cache) < dense_start_percent:
            return dense()

        in_dtype = q.dtype
        if skip_reshape:
            # HND (b, heads, T, dim) -> BTHD
            qn, kn, vn = (t.transpose(1, 2) for t in (q, k, v))
        else:
            # flat (b, T, heads*dim) -> BTHD
            qn, kn, vn = (t.view(b, tokens, heads, dim_head) for t in (q, k, v))
        qn, kn, vn = (t.to(torch.bfloat16).contiguous() for t in (qn, kn, vn))

        try:
            out = run_sol(qn, kn, vn, kwargs.get("scale", None))
        except Exception as exc:
            state["kernel_broken"] = True
            logging.warning(
                f"Sol-Attn kernel failed ({type(exc).__name__}: {exc}); "
                "falling back to dense attention for the rest of this run."
            )
            return dense()

        if not state["logged_active"]:
            state["logged_active"] = True
            logging.info(
                f"Sol-Attn active: tokens={tokens}, heads={heads if not skip_reshape else h}, "
                f"tau={tau}, dense_start_percent={dense_start_percent}"
            )

        out = out.to(in_dtype)
        if skip_output_reshape:
            out = out.transpose(1, 2)  # BTHD -> HND
        else:
            out = out.reshape(b, tokens, heads * dim_head)
        return out

    return override


class PatchSolAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Toggle the patch without unplugging the node."}),
                "tau": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Routing threshold. Higher = sparser and faster, "
                               "lower = more exact blocks and slower. 1.0 is the "
                               "paper default. Very low values (e.g. -10) are "
                               "near-exact attention."}),
                "dense_start_percent": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of sampling steps at the start that run "
                               "dense attention. NVIDIA used the first 10 of 50 "
                               "steps (0.2) for Wan. Structure forms early, so "
                               "keep some dense warmup."}),
                "min_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 1048576,
                    "tooltip": "Self-attention sequences shorter than this run "
                               "dense. Sol-Attn only pays off on long video "
                               "sequences."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Trent/Optimization"
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Patches model self-attention with NVIDIA Sol-Attn (arXiv:2607.24027), "
        "training-free sparse attention for long video sequences (Wan, Hunyuan, "
        "LTX...). Uses the portable Triton reference kernel; needs an sm90+ GPU "
        "(H100 / Blackwell) and head_dim 128. Cross attention, masked attention "
        "and short sequences automatically stay dense. Bypass the node to remove "
        "the patch."
    )

    def patch(self, model, enabled, tau, dense_start_percent, min_tokens):
        if not enabled:
            return (model,)

        if not torch.cuda.is_available():
            raise RuntimeError("Sol-Attn needs a CUDA GPU.")
        arch = torch.cuda.get_device_capability()
        if arch < (9, 0):
            raise RuntimeError(
                f"Sol-Attn needs an sm90+ GPU (H100/Blackwell), found sm_{arch[0]}{arch[1]}."
            )

        model_clone = model.clone()
        model_clone.model_options["transformer_options"]["optimized_attention_override"] = \
            make_sol_attention_override(tau, dense_start_percent, min_tokens)
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "PatchSolAttention": PatchSolAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchSolAttention": "Patch Sol-Attention - Trent",
}
