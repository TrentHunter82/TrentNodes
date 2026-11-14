"""
End-to-end test for nodes/deep_denoise.py. Loads the real models (they
download once to models/upscale_models). Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_deep_denoise.py
"""

import importlib
import math
import os
import sys

import torch

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "custom_nodes"))

deep_denoise = importlib.import_module("TrentNodes.nodes.deep_denoise")
denoise = importlib.import_module("TrentNodes.utils.denoise")

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        FAIL.append(name)


def psnr(a, b):
    return -10.0 * math.log10((a - b).pow(2).mean().item())


def clean_frames(count=4, h=300, w=421):
    """Smooth gradients plus hard-edged boxes; box moves each frame."""
    ys = torch.linspace(0, 1, h).view(h, 1)
    xs = torch.linspace(0, 1, w).view(1, w)
    base = torch.stack(
        [ys * xs, 0.5 + 0.4 * torch.sin(6 * xs) * ys, 1 - xs.expand(h, w)],
        dim=-1,
    )
    frames = base.repeat(count, 1, 1, 1).clone()
    for t in range(count):
        frames[t, 60:140, 40 + 30 * t:120 + 30 * t] = 0.9
    return frames


def main():
    torch.manual_seed(0)
    clean = clean_frames()
    sigma = 25.0 / 255.0
    noisy = (clean + sigma * torch.randn_like(clean)).clamp(0, 1)
    base = psnr(noisy, clean)
    print(f"noisy PSNR {base:.2f} dB")

    est = denoise.estimate_sigma(noisy[:1].movedim(-1, 1)) * 255
    check("sigma estimate", abs(est - 25) < 5, f"{est:.1f}/255")

    node = deep_denoise.DeepDenoise()
    args = dict(
        noise_level=0.0, luma_strength=1.0, chroma_strength=1.0,
        self_ensemble="off", temporal_radius=0, motion_compensation=False,
        temporal_tolerance=0.015,
        tile_size=1024, tile_overlap=64,
    )

    scores = {}
    for name in deep_denoise.MODELS:
        out, noise = node.denoise(noisy, name, **args)
        scores[name] = psnr(out, clean)
        check(name, out.shape == noisy.shape and scores[name] > base + 6,
              f"{scores[name]:.2f} dB")
    check("noise output", noise.shape == noisy.shape
          and abs(noise.mean().item() - 0.5) < 0.02)

    name = "DRUNet (tunable)"
    out, _ = node.denoise(noisy, name, **{**args, "self_ensemble": "x8"})
    check("x8 self-ensemble", psnr(out, clean) >= scores[name] - 0.05,
          f"{psnr(out, clean):.2f} dB")

    out, _ = node.denoise(noisy, name, **{**args, "tile_size": 256})
    check("tiled", abs(psnr(out, clean) - scores[name]) < 0.5,
          f"{psnr(out, clean):.2f} dB")

    out, _ = node.denoise(noisy, name, **{**args, "temporal_radius": 2})
    check("temporal", psnr(out, clean) > scores[name],
          f"{psnr(out, clean):.2f} dB")

    out, _ = node.denoise(noisy, name, **{**args, "luma_strength": 0.0,
                                         "chroma_strength": 0.0})
    check("zero strength is identity", torch.allclose(out, noisy, atol=1e-5))

    rgba = torch.cat([noisy, torch.full_like(noisy[..., :1], 0.25)], dim=-1)
    out, _ = node.denoise(rgba, name, **args)
    check("alpha kept", out.shape[-1] == 4
          and torch.allclose(out[..., 3], rgba[..., 3]))

    print("FAILED: " + ", ".join(FAIL) if FAIL else "all passed")
    sys.exit(1 if FAIL else 0)


main()
