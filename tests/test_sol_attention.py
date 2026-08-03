"""GPU tests for the Sol-Attn kernel and the PatchSolAttention override.

No comfy imports; run with:

    /home/trent/ComfyUI/venv/bin/python tests/test_sol_attention.py
"""

import importlib.util
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Register a synthetic package so the node module's relative imports resolve
# without importing the full TrentNodes package (which pulls in comfy).
_pkg = types.ModuleType("_trent_test_pkg")
_pkg.__path__ = [ROOT]
sys.modules["_trent_test_pkg"] = _pkg
for sub in ("utils", "nodes"):
    mod = types.ModuleType(f"_trent_test_pkg.{sub}")
    mod.__path__ = [os.path.join(ROOT, sub)]
    sys.modules[f"_trent_test_pkg.{sub}"] = mod


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


kmod = _load(
    "_trent_test_pkg.utils.sol_attn_triton",
    os.path.join(ROOT, "utils", "sol_attn_triton.py"),
)
nmod = _load(
    "_trent_test_pkg.nodes.patch_sol_attention",
    os.path.join(ROOT, "nodes", "patch_sol_attention.py"),
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS {name} {detail}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {detail}")


def sdpa_bthd(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def cos_sim(a, b):
    return F.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name()} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}")

    torch.manual_seed(0)

    # --- 1. kernel exactness at very low tau (all blocks exact) ---
    print("\n[1] kernel exactness (tau=-100, T=8200 with 64-tail)")
    b, T, h, d = 2, 8200, 8, 128
    q = torch.randn(b, T, h, d, device=device, dtype=torch.bfloat16) * 0.5
    k = torch.randn_like(q) * 0.5
    v = torch.randn_like(q)
    out = kmod.sol_attn(q, k, v, tau=-100.0)
    ref = sdpa_bthd(q, k, v)
    err = rel_err(out, ref)
    check("exact_matches_sdpa", err < 2e-2, f"rel_err={err:.4f}")

    # --- 2. default tau quality on structured data ---
    print("\n[2] default tau=1.0 quality (structured q/k)")
    base = torch.randn(b, T // 8, h, d, device=device, dtype=torch.bfloat16)
    qs = base.repeat_interleave(8, dim=1) + 0.3 * torch.randn(
        b, T // 8 * 8, h, d, device=device, dtype=torch.bfloat16)
    ks = base.repeat_interleave(8, dim=1) + 0.3 * torch.randn_like(qs)
    vs = torch.randn_like(qs)
    qs, ks, vs = (t.contiguous() for t in (qs, ks, vs))
    out = kmod.sol_attn(qs, ks, vs, tau=1.0)
    ref = sdpa_bthd(qs, ks, vs)
    sim = cos_sim(out, ref)
    check("tau1_cosine_similarity", sim > 0.90, f"cos={sim:.4f}")

    # --- 3. speed vs SDPA at Wan-like sequence length ---
    print("\n[3] speed benchmark (b=1, T=32760, h=12, d=128)")
    b2, T2, h2 = 1, 32760, 12
    qb = torch.randn(b2, T2, h2, d, device=device, dtype=torch.bfloat16) * 0.5
    kb = torch.randn_like(qb) * 0.5
    vb = torch.randn_like(qb)

    for _ in range(3):  # warmup incl. autotune
        kmod.sol_attn(qb, kb, vb, tau=1.0)
        sdpa_bthd(qb, kb, vb)
    torch.cuda.synchronize()

    def bench(fn, iters=10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    t_sol = bench(lambda: kmod.sol_attn(qb, kb, vb, tau=1.0))
    t_sdpa = bench(lambda: sdpa_bthd(qb, kb, vb))
    print(f"  sol_attn: {t_sol:.2f} ms   sdpa: {t_sdpa:.2f} ms   "
          f"speedup: {t_sdpa / t_sol:.2f}x")
    check("kernel_runs_at_scale", True, "")

    # --- 4. override reshape paths ---
    print("\n[4] override layout paths (tau=-100 vs dense func)")
    calls = {"dense": 0}

    def dense_func(q, k, v, heads, mask=None, attn_precision=None,
                   skip_reshape=False, skip_output_reshape=False, **kwargs):
        calls["dense"] += 1
        if skip_reshape:
            qh, kh, vh = q, k, v
            bb = q.shape[0]
            dh = q.shape[-1]
        else:
            bb, _, inner = q.shape
            dh = inner // heads
            qh, kh, vh = (t.view(bb, t.shape[1], heads, dh).transpose(1, 2)
                          for t in (q, k, v))
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        if skip_output_reshape:
            return o
        return o.transpose(1, 2).reshape(bb, -1, heads * dh)

    ov = nmod.make_sol_attention_override(
        tau=-100.0, dense_start_percent=0.0, min_tokens=256)

    Tb = 4096
    q3 = torch.randn(1, Tb, 8 * 128, device=device, dtype=torch.bfloat16) * 0.5
    k3 = torch.randn_like(q3) * 0.5
    v3 = torch.randn_like(q3)
    out_flat = ov(dense_func, q3, k3, v3, 8)
    ref_flat = dense_func(q3, k3, v3, 8)
    err = rel_err(out_flat, ref_flat)
    check("flat_layout", out_flat.shape == ref_flat.shape and err < 2e-2,
          f"rel_err={err:.4f}")

    q4 = torch.randn(1, 8, Tb, 128, device=device, dtype=torch.bfloat16) * 0.5
    k4 = torch.randn_like(q4) * 0.5
    v4 = torch.randn_like(q4)
    out_hnd = ov(dense_func, q4, k4, v4, 8, skip_reshape=True,
                 skip_output_reshape=True)
    ref_hnd = dense_func(q4, k4, v4, 8, skip_reshape=True,
                         skip_output_reshape=True)
    err = rel_err(out_hnd, ref_hnd)
    check("hnd_layout", out_hnd.shape == ref_hnd.shape and err < 2e-2,
          f"rel_err={err:.4f}")

    out_mix = ov(dense_func, q4, k4, v4, 8, skip_reshape=True)
    ref_mix = dense_func(q4, k4, v4, 8, skip_reshape=True)
    err = rel_err(out_mix, ref_mix)
    check("hnd_to_flat_layout", out_mix.shape == ref_mix.shape and err < 2e-2,
          f"rel_err={err:.4f}")

    # --- 5. fallback guards ---
    print("\n[5] fallback guards route to dense")
    before = calls["dense"]
    mask = torch.zeros(Tb, Tb, device=device, dtype=torch.bfloat16)
    ov(dense_func, q3, k3, v3, 8, mask=mask)
    check("mask_falls_back", calls["dense"] == before + 1)

    before = calls["dense"]
    k_short = torch.randn(1, 512, 8 * 128, device=device,
                          dtype=torch.bfloat16)
    ov(dense_func, q3, k_short, k_short.clone(), 8)
    check("cross_attn_falls_back", calls["dense"] == before + 1)

    before = calls["dense"]
    q_small = torch.randn(1, 128, 8 * 128, device=device,
                          dtype=torch.bfloat16)
    ov(dense_func, q_small, q_small.clone(), q_small.clone(), 8)
    check("short_seq_falls_back", calls["dense"] == before + 1)

    before = calls["dense"]
    q64 = torch.randn(1, Tb, 8 * 64, device=device, dtype=torch.bfloat16)
    ov(dense_func, q64, q64.clone(), q64.clone(), 8)
    check("head_dim_64_falls_back", calls["dense"] == before + 1)

    # dense warmup window via sigmas
    ov2 = nmod.make_sol_attention_override(
        tau=-100.0, dense_start_percent=0.5, min_tokens=256)
    schedule = torch.linspace(1.0, 0.0, 21, device=device)
    topts_early = {"sigmas": schedule[2:3], "sample_sigmas": schedule}
    topts_late = {"sigmas": schedule[15:16], "sample_sigmas": schedule}
    before = calls["dense"]
    ov2(dense_func, q3, k3, v3, 8, transformer_options=topts_early)
    check("early_step_dense", calls["dense"] == before + 1)
    before = calls["dense"]
    out_late = ov2(dense_func, q3, k3, v3, 8, transformer_options=topts_late)
    check("late_step_sparse", calls["dense"] == before
          and out_late.shape == ref_flat.shape)

    # --- 6. fp16 input round trip ---
    print("\n[6] fp16 input")
    q5 = q3.to(torch.float16)
    out5 = ov(dense_func, q5, k3.to(torch.float16), v3.to(torch.float16), 8)
    check("fp16_dtype_roundtrip", out5.dtype == torch.float16
          and out5.shape == ref_flat.shape)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
