"""
Node-level smoke tests for AlignStylizedFrame on CPU. Loads the node
module inside the real comfy environment (no server) via a synthetic
package so TrentNodes/__init__ discovery does not run. Run from the
ComfyUI root:

    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_node_paths.py
"""

import importlib
import math
import os
import sys
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

sys.path.insert(0, ROOT)

pkg = types.ModuleType("TrentNodes")
pkg.__path__ = [PKG]
sys.modules["TrentNodes"] = pkg
for sub in ("nodes", "utils"):
    m = types.ModuleType(f"TrentNodes.{sub}")
    m.__path__ = [os.path.join(PKG, sub)]
    sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

node_mod = importlib.import_module("TrentNodes.nodes.align_stylized_frame")
alignment = importlib.import_module("TrentNodes.utils.alignment")

DEVICE = torch.device("cpu")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")


def make_image(H=256, W=256, seed=3):
    gen = torch.Generator().manual_seed(seed)
    img = torch.zeros(1, H, W, 3)
    for octave, amp in ((8, 0.5), (16, 0.3), (32, 0.2)):
        noise = torch.rand(1, 3, octave, octave, generator=gen)
        img += F.interpolate(
            noise, size=(H, W), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1) * amp
    img[:, 60:90, 40:120, :] = 0.9
    img[:, 150:200, 170:220, 0] = 0.1
    return img.clamp(0, 1)


def circle_mask(H, W, cy, cx, r):
    yy = torch.arange(H).view(-1, 1).float()
    xx = torch.arange(W).view(1, -1).float()
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float().unsqueeze(0)


def test_align_frames_global():
    print("\n[node-1] align_frames global path (subject disabled, blur fill)")
    node = node_mod.AlignStylizedFrame()
    img = make_image()
    p_gt = (-12.0, 9.0, math.log(0.985), math.log(0.985), 0.0)
    stylized, _ = alignment.warp_image(img, p_gt, DEVICE)

    # Force CPU regardless of mm.get_torch_device()
    import comfy.model_management as mm
    orig_get = mm.get_torch_device
    mm.get_torch_device = lambda: DEVICE
    try:
        out = node.align_frames(
            img, stylized,
            subject_mode="disabled",
            inpaint_method="blur",
            visualization_mode="score_map",
        )
    finally:
        mm.get_torch_device = orig_get

    aligned, diff_map, info, subj_mask, inpaint_mask = out
    err = (aligned - img).abs().mean().item()
    base = (stylized - img).abs().mean().item()
    check(
        "global-align-improves", err < base * 0.35,
        f"mean|diff| {base:.4f} -> {err:.4f}",
    )
    check(
        "output-shapes-types",
        aligned.shape == img.shape and aligned.dtype == torch.float32
        and subj_mask.shape == (1, 256, 256)
        and inpaint_mask.shape == (1, 256, 256)
        and diff_map.shape[-1] == 3 and diff_map.shape[2] == 512,
        f"aligned {tuple(aligned.shape)} {aligned.dtype}, "
        f"diff_map {tuple(diff_map.shape)}",
    )
    check("info-mentions-method", "Global align:" in info, info.splitlines()[0])
    print("  info:", " | ".join(info.strip().splitlines()))


def test_subject_path():
    print("\n[node-2] preserve_subject_inpaint_background (centroid, blur)")
    node = node_mod.AlignStylizedFrame()
    H = W = 256
    bg = make_image(H, W, seed=5)

    # Stylized: subject circle at (80, 80); original target at (170, 170)
    styl = bg.clone()
    styl[:, 55:105, 55:105, :] = 0.0
    styl_mask = circle_mask(H, W, 80, 80, 25)
    styl = styl * (1 - styl_mask.unsqueeze(-1)) + (
        styl_mask.unsqueeze(-1) * torch.tensor([0.2, 0.9, 0.3])
    )
    orig_mask = circle_mask(H, W, 170, 170, 25)
    original = bg.clone()

    # Identity global transform: ghost sits where styl subject is
    aligned_styl_mask = styl_mask.clone()
    edge_strip = torch.zeros(1, H, W)
    edge_strip[:, :, :6] = 1.0  # fake transform-edge gap

    result, info, inpaint_mask = node.preserve_subject_inpaint_background(
        styl, styl.clone(), original,
        orig_mask, styl_mask, aligned_styl_mask,
        conform_to_original=1.0,
        inpaint_method="blur", mask_expand=10,
        inpaint_steps=5, inpaint_denoise=0.9,
        device=DEVICE,
        extra_edge_mask=edge_strip,
    )

    subj_color = result[0, 170, 170]
    check(
        "subject-pasted-at-target",
        abs(subj_color[1].item() - 0.9) < 0.1
        and abs(subj_color[0].item() - 0.2) < 0.1,
        f"color at (170,170) = {[round(v, 2) for v in subj_color.tolist()]}",
    )
    check(
        "inpaint-mask-covers-ghost",
        inpaint_mask[0, 80, 80].item() > 0.5,
        f"mask at ghost center = {inpaint_mask[0, 80, 80].item():.2f}",
    )
    check(
        "inpaint-mask-excludes-subject",
        inpaint_mask[0, 170, 170].item() < 0.01,
        f"mask at pasted subject = {inpaint_mask[0, 170, 170].item():.2f}",
    )
    check(
        "inpaint-mask-includes-edge-strip",
        inpaint_mask[0, 128, 2].item() > 0.5,
        f"mask at edge strip = {inpaint_mask[0, 128, 2].item():.2f}",
    )
    ghost_after = result[0, 80, 80]
    check(
        "ghost-inpainted-away",
        abs(ghost_after[1].item() - 0.9) > 0.15,
        f"color at ghost center = {[round(v, 2) for v in ghost_after.tolist()]}",
    )
    print("  info:", info)

    # conform_to_original = 0.5: paste lands halfway (125, 125)
    result2, _, _ = node.preserve_subject_inpaint_background(
        styl, styl.clone(), original,
        orig_mask, styl_mask, aligned_styl_mask,
        conform_to_original=0.5,
        inpaint_method="blur", mask_expand=10,
        inpaint_steps=5, inpaint_denoise=0.9,
        device=DEVICE,
    )
    mid_color = result2[0, 125, 125]
    check(
        "conform-halfway-paste",
        abs(mid_color[1].item() - 0.9) < 0.1,
        f"color at (125,125) = {[round(v, 2) for v in mid_color.tolist()]}",
    )


def test_none_mode_mask_safety():
    print("\n[node-3] inpaint_method='none': mask safe, ghost kept")
    node = node_mod.AlignStylizedFrame()
    H = W = 256
    bg = make_image(H, W, seed=5)
    styl = bg.clone()
    styl_mask = circle_mask(H, W, 80, 80, 25)
    styl = styl * (1 - styl_mask.unsqueeze(-1)) + (
        styl_mask.unsqueeze(-1) * torch.tensor([0.2, 0.9, 0.3])
    )
    orig_mask = circle_mask(H, W, 170, 170, 25)

    result, info, inpaint_mask = node.preserve_subject_inpaint_background(
        styl, styl.clone(), bg.clone(),
        orig_mask, styl_mask, styl_mask.clone(),
        conform_to_original=1.0,
        inpaint_method="none", mask_expand=10,
        inpaint_steps=5, inpaint_denoise=0.9,
        device=DEVICE,
    )
    ghost_color = result[0, 80, 80]
    check(
        "none-ghost-untouched",
        abs(ghost_color[1].item() - 0.9) < 0.05,
        f"ghost color = {[round(v, 2) for v in ghost_color.tolist()]}",
    )
    check(
        "none-mask-excludes-subject",
        inpaint_mask[0, 170, 170].item() < 0.01,
        f"mask at pasted subject = {inpaint_mask[0, 170, 170].item():.2f}",
    )
    check(
        "none-mask-covers-ghost",
        inpaint_mask[0, 80, 80].item() > 0.5,
        f"mask at ghost = {inpaint_mask[0, 80, 80].item():.2f}",
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    test_align_frames_global()
    test_subject_path()
    test_none_mode_mask_safety()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
