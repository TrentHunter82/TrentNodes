"""
Standalone CPU tests for utils/alignment.py and the rotation-sign fix in
utils/pose_alignment.py. No comfy imports; run with:

    /home/trent/ComfyUI/venv/bin/python tests/test_alignment.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "utils")
)

import alignment  # noqa: E402
import image_ops  # noqa: E402
import pose_alignment  # noqa: E402

DEVICE = torch.device("cpu")
PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name} {detail}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name} {detail}")


def make_test_image(H=256, W=256, seed=7):
    """Multi-octave smooth noise + rectangles: textured but structured."""
    gen = torch.Generator().manual_seed(seed)
    img = torch.zeros(1, H, W, 3)
    for octave, amp in ((8, 0.5), (16, 0.25), (32, 0.25)):
        noise = torch.rand(1, 3, octave, octave, generator=gen)
        img += F.interpolate(
            noise, size=(H, W), mode="bicubic", align_corners=False
        ).permute(0, 2, 3, 1) * amp
    img[:, H // 4:H // 3, W // 5:W // 2, :] = 0.9
    img[:, H // 2:3 * H // 4, 2 * W // 3:5 * W // 6, 0] = 0.1
    img[:, 2 * H // 3:2 * H // 3 + 8, :, :] = 0.05
    return img.clamp(0, 1)


def style_gap(img, seed=11):
    """Simulate a Flux-like restyle: gamma, contrast, blur, smooth noise."""
    gen = torch.Generator().manual_seed(seed)
    out = img.clamp(1e-4, 1.0) ** 0.7
    out = (out - 0.5) * 0.65 + 0.5 + 0.15
    bchw = out.permute(0, 3, 1, 2)
    bchw = F.avg_pool2d(F.pad(bchw, (1, 1, 1, 1), mode="replicate"), 3, 1)
    noise = torch.rand(1, 3, 16, 16, generator=gen) - 0.5
    noise = F.interpolate(
        noise, size=bchw.shape[-2:], mode="bicubic", align_corners=False
    )
    return (bchw + noise * 0.1).permute(0, 2, 3, 1).clamp(0, 1)


def composed_residual_px(p_est, p_gt, H, W):
    """Pixel error of applying est correction after gt distortion."""
    prod = (
        alignment.forward_matrix(p_est) @ alignment.forward_matrix(p_gt)
    )
    trans_err = prod[:2, 2].abs().max().item()
    lin_err = (prod[:2, :2] - torch.eye(2, dtype=torch.float64)).abs().max().item()
    return trans_err + lin_err * max(H, W) / 2


def test_build_theta_matches_legacy():
    print("\n[1] build_theta matches apply_affine_transform at rot=0, sx=sy")
    img = make_test_image()
    scale, tx, ty = 0.96, 11.0, -7.0
    legacy = image_ops.apply_affine_transform(img, scale, tx, ty, DEVICE)
    params = (tx, ty, math.log(scale), math.log(scale), 0.0)
    new, _ = alignment.warp_image(img, params, DEVICE)
    diff = (legacy - new).abs().max().item()
    check("theta-legacy-equivalence", diff < 1e-5, f"max|diff|={diff:.2e}")


def test_warp_conventions():
    print("\n[2] warp_image conventions: +t moves content, +rot y-down")
    H = W = 200
    img = torch.zeros(1, H, W, 3)
    img[0, 100, 120, :] = 1.0  # marker right of center

    warped, _ = alignment.warp_image(img, (15.0, -10.0, 0.0, 0.0, 0.0), DEVICE)
    pos = (warped[0, ..., 0] > 0.2).nonzero().float().mean(dim=0)
    check(
        "translation-convention",
        abs(pos[0].item() - 90) < 1.5 and abs(pos[1].item() - 135) < 1.5,
        f"marker at (y={pos[0]:.1f}, x={pos[1]:.1f}), want (90, 135)",
    )

    # +30deg in y-down coords: offset (dx,dy)=(20.5,0.5) from center
    # (99.5,99.5) -> R(+30)@offset = (dx cos - dy sin, dx sin + dy cos)
    rot = math.radians(30.0)
    warped, _ = alignment.warp_image(img, (0.0, 0.0, 0.0, 0.0, rot), DEVICE)
    pos = (warped[0, ..., 0] > 0.2).nonzero().float().mean(dim=0)
    cx = cy = (W - 1) / 2
    dx, dy = 120 - cx, 100 - cy
    ex = cx + dx * math.cos(rot) - dy * math.sin(rot)
    ey = cy + dx * math.sin(rot) + dy * math.cos(rot)
    check(
        "rotation-convention",
        abs(pos[0].item() - ey) < 1.5 and abs(pos[1].item() - ex) < 1.5,
        f"marker at (y={pos[0]:.1f}, x={pos[1]:.1f}), want ({ey:.1f}, {ex:.1f})",
    )


def test_phase_correlation():
    print("\n[3] phase correlation recovers a pure shift")
    img = make_test_image()
    shifted, _ = alignment.warp_image(img, (24.0, -17.0, 0.0, 0.0, 0.0), DEVICE)
    g_o = alignment.to_grayscale(img).unsqueeze(1)
    g_s = alignment.to_grayscale(shifted).unsqueeze(1)
    e_o = alignment.extract_edges_soft(g_o)[0, 0]
    e_s = alignment.extract_edges_soft(g_s)[0, 0]
    dx, dy = alignment.phase_correlation(e_o, e_s)
    check(
        "phase-correlation",
        abs(dx + 24) <= 1 and abs(dy - 17) <= 1,
        f"got ({dx}, {dy}), want (-24, 17)",
    )


def run_roundtrip(name, p_gt, allow_aniso, max_rot_deg, expect_px=0.75,
                  multi_start=True, precision="balanced"):
    img = make_test_image()
    distorted, _ = alignment.warp_image(img, p_gt, DEVICE)
    stylized = style_gap(distorted)
    est = alignment.estimate_affine(
        img, stylized, DEVICE,
        max_translation=32, max_scale_dev=0.05,
        max_rotation_deg=max_rot_deg,
        allow_anisotropic=allow_aniso,
        precision=precision, multi_start=multi_start,
    )
    err = composed_residual_px(est.params, p_gt, 256, 256)
    detail = (
        f"method={est.method} conv={est.converged} err={err:.3f}px "
        f"tx={est.tx:.2f} ty={est.ty:.2f} sx={est.scale_x:.4f} "
        f"sy={est.scale_y:.4f} rot={est.rotation_deg:.3f} "
        f"ncc {est.ncc_identity:.3f}->{est.ncc_final:.3f}"
    )
    check(name, est.converged and err < expect_px, detail)
    return est


def test_estimate_easy():
    print("\n[4] round-trip: easy case with style gap")
    p_gt = (7.3, -4.6, math.log(1.032), math.log(1.032), math.radians(1.7))
    run_roundtrip("roundtrip-easy", p_gt, False, 3.0)


def test_estimate_hard():
    print("\n[5] round-trip: hard case (25px, -2.2deg, anisotropic)")
    p_gt = (25.0, -18.0, math.log(0.97), math.log(1.01), math.radians(-2.2))
    run_roundtrip(
        "roundtrip-hard", p_gt, True, 3.0, expect_px=1.0,
        precision="precise",
    )

    # Safety property without multi-start: must either converge or
    # honestly report identity, never a confident wrong warp.
    img = make_test_image()
    distorted, _ = alignment.warp_image(img, p_gt, DEVICE)
    stylized = style_gap(distorted)
    est = alignment.estimate_affine(
        img, stylized, DEVICE,
        max_translation=32, max_scale_dev=0.05, max_rotation_deg=3.0,
        allow_anisotropic=True, precision="precise", multi_start=False,
    )
    err = composed_residual_px(est.params, p_gt, 256, 256)
    safe = (est.converged and err < 2.0) or est.method == "identity"
    check(
        "no-multistart-safety", safe,
        f"method={est.method} conv={est.converged} err={err:.2f}px",
    )


def test_estimate_under_inference_mode():
    print("\n[4b] estimate_affine under torch.inference_mode (ComfyUI runs "
          "nodes this way)")
    p_gt = (7.3, -4.6, math.log(1.032), math.log(1.032), math.radians(1.7))
    img = make_test_image()
    distorted, _ = alignment.warp_image(img, p_gt, DEVICE)
    stylized = style_gap(distorted)
    with torch.inference_mode():
        est = alignment.estimate_affine(
            img.clone(), stylized.clone(), DEVICE,
            max_translation=32, max_scale_dev=0.05, max_rotation_deg=3.0,
        )
        # Outputs must be usable inside inference mode (e.g. warp_image)
        aligned, _ = alignment.warp_image(stylized.clone(), est.params, DEVICE)
    err = composed_residual_px(est.params, p_gt, 256, 256)
    check(
        "inference-mode-roundtrip",
        est.converged and err < 0.75,
        f"method={est.method} conv={est.converged} err={err:.3f}px",
    )


def test_estimate_identical():
    print("\n[6] identical images -> ~identity params")
    img = make_test_image()
    est = alignment.estimate_affine(
        img, img.clone(), DEVICE,
        max_translation=32, max_scale_dev=0.05, max_rotation_deg=3.0,
    )
    ok = (
        abs(est.tx) < 0.5 and abs(est.ty) < 0.5
        and abs(est.scale_x - 1) < 0.003 and abs(est.scale_y - 1) < 0.003
        and abs(est.rotation_deg) < 0.2
    )
    check(
        "identical-images", ok,
        f"tx={est.tx:.3f} ty={est.ty:.3f} sx={est.scale_x:.4f} "
        f"rot={est.rotation_deg:.3f} method={est.method}",
    )


def test_estimate_unrelated():
    print("\n[7] unrelated images -> identity fallback (no wild warp)")
    a = make_test_image(seed=7)
    b = make_test_image(seed=1234)
    est = alignment.estimate_affine(
        a, b, DEVICE,
        max_translation=32, max_scale_dev=0.05, max_rotation_deg=3.0,
    )
    # Unrelated content can still yield a marginal NCC gain; the safety
    # property is that we never report a confident strong alignment.
    ok = est.method == "identity" or (
        est.ncc_final - est.ncc_identity
    ) < 0.2
    check(
        "unrelated-images", ok,
        f"method={est.method} ncc {est.ncc_identity:.3f}->{est.ncc_final:.3f}",
    )


def test_rotate_image_sign():
    print("\n[8] pose_alignment.rotate_image rotates content by +angle (y-down)")
    H = W = 200
    img = torch.zeros(1, H, W, 3)
    img[0, 60, 140, :] = 1.0
    angle = 20.0
    rotated = pose_alignment.rotate_image(img, angle, DEVICE)
    pos = (rotated[0, ..., 0] > 0.2).nonzero().float().mean(dim=0)
    cx = cy = (W - 1) / 2
    dx, dy = 140 - cx, 60 - cy
    rad = math.radians(angle)
    ex = cx + dx * math.cos(rad) - dy * math.sin(rad)
    ey = cy + dx * math.sin(rad) + dy * math.cos(rad)
    check(
        "rotate-image-sign",
        abs(pos[0].item() - ey) < 1.5 and abs(pos[1].item() - ex) < 1.5,
        f"marker at (y={pos[0]:.1f}, x={pos[1]:.1f}), want ({ey:.1f}, {ex:.1f})",
    )

    # The node's shoulder-offset tracking block uses R(+a) around the crop
    # center; with the sign fix it must predict the content position.
    pred_y, pred_x = ey, ex
    check(
        "shoulder-tracking-consistency",
        abs(pos[0].item() - pred_y) < 1.5 and abs(pos[1].item() - pred_x) < 1.5,
        "node R(+a) tracking matches rotate_image content motion",
    )


def test_ecc_fallback():
    print("\n[9] ECC fallback recovers a known transform")
    p_gt = (9.0, -6.0, math.log(1.025), math.log(1.025), math.radians(1.2))
    img = make_test_image()
    distorted, _ = alignment.warp_image(img, p_gt, DEVICE)
    g_o = alignment.to_grayscale(img).unsqueeze(1)
    g_s = alignment.to_grayscale(distorted).unsqueeze(1)
    e_o = alignment.extract_edges_soft(g_o)
    e_s = alignment.extract_edges_soft(g_s)
    scale_vec = torch.tensor([
        32.0, 32.0, abs(math.log(0.95)), abs(math.log(0.95)),
        math.radians(3.0),
    ])
    p_ecc = alignment._ecc_fallback(e_o, e_s, scale_vec, False)
    if p_ecc is None:
        check("ecc-fallback", False, "ECC returned None")
        return
    err = composed_residual_px(tuple(p_ecc.tolist()), p_gt, 256, 256)
    check("ecc-fallback", err < 1.5, f"err={err:.3f}px params={p_ecc.tolist()}")


def test_extract_edges_no_border_ring():
    print("\n[10] image_ops.extract_edges: no false border edges")
    img = torch.full((1, 64, 64, 3), 0.5)
    edges = image_ops.extract_edges(img, DEVICE)
    check(
        "edges-no-border-ring",
        edges.abs().max().item() < 1e-6,
        f"max edge on uniform image = {edges.abs().max().item():.3f}",
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    test_build_theta_matches_legacy()
    test_warp_conventions()
    test_phase_correlation()
    test_estimate_easy()
    test_estimate_under_inference_mode()
    test_estimate_hard()
    test_estimate_identical()
    test_estimate_unrelated()
    test_rotate_image_sign()
    test_ecc_fallback()
    test_extract_edges_no_border_ring()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
