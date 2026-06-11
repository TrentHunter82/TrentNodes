"""
Differentiable global affine alignment for TrentNodes.

Estimates the global transform (translation, scale, rotation) between an
original frame and a stylized restyle of it, robust to the style gap
(palette/contrast/texture changes). Used by AlignStylizedFrame in place of
brute-force grid search.

Method:
- FFT phase correlation seeds translation.
- Adam optimizes (tx, ty, log_sx, log_sy, rot) coarse-to-fine through
  affine_grid/grid_sample on blurred Sobel edge maps, with a weighted
  Pearson NCC loss (invariant to global edge-contrast differences).
- A batched multi-start grid at the coarsest level avoids local minima.
- Acceptance test vs identity, cv2.findTransformECC fallback, identity
  last resort: the result is never worse than no alignment.

Importable standalone (torch/numpy at module level; cv2 lazily inside the
ECC fallback only).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from .image_ops import to_grayscale
except ImportError:  # standalone import (tests add utils/ to sys.path)
    from image_ops import to_grayscale


# Stage schedule per search_precision: (coarse, mid, fine) Adam iterations.
# Coarse runs at 1/4 resolution with multi-start, mid at 1/2, fine at full.
PRECISION_ITERS = {
    "fast": (30, 50, 0),
    "balanced": (40, 80, 80),
    "precise": (60, 120, 160),
}
STAGE_LRS = (0.1, 0.03, 0.01)


def phase_correlation(
    map_a: torch.Tensor,
    map_b: torch.Tensor,
) -> tuple:
    """
    Compute translation offset from map_a to map_b via phase correlation.

    Args:
        map_a: Tensor (H, W) float32, reference
        map_b: Tensor (H, W) float32, source

    Returns:
        (offset_x, offset_y) in pixels. Applying this offset to map_b's
        image aligns it to map_a.
    """
    H, W = map_a.shape

    fa = torch.fft.rfft2(map_a)
    fb = torch.fft.rfft2(map_b)

    cross = fa * fb.conj()
    eps = 1e-8
    cross = cross / (cross.abs() + eps)

    correlation = torch.fft.irfft2(cross, s=(H, W))

    peak_flat = correlation.argmax()
    peak_y = (peak_flat // W).item()
    peak_x = (peak_flat % W).item()

    # Wrap negative offsets (phase correlation folds at half-size)
    if peak_y > H // 2:
        peak_y -= H
    if peak_x > W // 2:
        peak_x -= W

    return int(peak_x), int(peak_y)


def extract_edges_soft(gray: torch.Tensor) -> torch.Tensor:
    """
    Sobel edge magnitude with replicate padding (no false border edges)
    followed by a 3x3 binomial blur. The blur widens the convergence basin
    and smooths gradients for sub-pixel optimization.

    Args:
        gray: (B, 1, H, W) grayscale tensor

    Returns:
        (B, 1, H, W) blurred edge magnitude
    """
    device, dtype = gray.device, gray.dtype
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    binomial = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        dtype=dtype, device=device
    ).view(1, 1, 3, 3) / 16.0

    padded = F.pad(gray, (1, 1, 1, 1), mode='replicate')
    ex = F.conv2d(padded, sobel_x)
    ey = F.conv2d(padded, sobel_y)
    mag = torch.sqrt(ex * ex + ey * ey + 1e-8)

    mag = F.pad(mag, (1, 1, 1, 1), mode='replicate')
    return F.conv2d(mag, binomial)


def weighted_ncc_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted Pearson NCC loss between two maps, per batch sample.

    Invariant to any affine intensity change of either map, which removes
    the style-contrast bias of plain L1/L2 edge differences.

    Args:
        a, b: (N, 1, H, W) maps to compare
        w: (N, 1, H, W) non-negative weights

    Returns:
        (N,) loss vector, loss = 1 - correlation (0 = perfect match)
    """
    eps = 1e-8
    wsum = w.sum(dim=(1, 2, 3), keepdim=True) + eps
    ma = (a * w).sum(dim=(1, 2, 3), keepdim=True) / wsum
    mb = (b * w).sum(dim=(1, 2, 3), keepdim=True) / wsum
    ac = a - ma
    bc = b - mb
    cov = (w * ac * bc).sum(dim=(1, 2, 3))
    va = (w * ac * ac).sum(dim=(1, 2, 3))
    vb = (w * bc * bc).sum(dim=(1, 2, 3))
    corr = cov / torch.sqrt(va * vb + eps)
    return 1.0 - corr


def build_theta(params: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Build grid_sample theta for the forward content model
        q = R(rot) @ diag(sx, sy) @ p + t
    (centered pixel coords, x right, y down: content moves by +t pixels,
    scales around center, rotates by +rot).

    grid_sample inverse-maps, so theta is the inverse transform expressed
    in normalized coordinates (note the H/W skew terms on the rotation
    entries for non-square images).

    Args:
        params: (N, 5) tensor of (tx_px, ty_px, log_sx, log_sy, rot_rad)
        H, W: image size the theta will be applied to

    Returns:
        (N, 2, 3) theta for F.affine_grid(..., align_corners=False)
    """
    tx, ty = params[:, 0], params[:, 1]
    sx = torch.exp(params[:, 2])
    sy = torch.exp(params[:, 3])
    cos_r = torch.cos(params[:, 4])
    sin_r = torch.sin(params[:, 4])

    # M^-1 = diag(1/sx, 1/sy) @ R(-rot)
    a00 = cos_r / sx
    a01 = sin_r / sx
    a10 = -sin_r / sy
    a11 = cos_r / sy

    theta = torch.stack([
        torch.stack([a00, a01 * (H / W),
                     -(a00 * tx + a01 * ty) / (W / 2)], dim=1),
        torch.stack([a10 * (W / H), a11,
                     -(a10 * tx + a11 * ty) / (H / 2)], dim=1),
    ], dim=1)
    return theta


def forward_matrix(params) -> torch.Tensor:
    """
    3x3 forward content transform in centered pixel coords for the given
    (tx, ty, log_sx, log_sy, rot_rad) params. Useful for composing and
    testing transforms.
    """
    tx, ty, lsx, lsy, rot = [float(v) for v in params]
    sx, sy = math.exp(lsx), math.exp(lsy)
    c, s = math.cos(rot), math.sin(rot)
    return torch.tensor([
        [sx * c, -sy * s, tx],
        [sx * s, sy * c, ty],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float64)


def warp_image(
    image: torch.Tensor,
    params,
    device: torch.device,
    padding_mode: str = 'border',
) -> tuple:
    """
    Warp a (B, H, W, C) image by alignment params, returning the warped
    image and a validity mask (1.0 = real content, <1.0 = padding fill).

    Generalizes image_ops.apply_affine_transform_with_mask with rotation
    and anisotropic scale; identical to it at rot=0, sx=sy.

    Args:
        image: (B, H, W, C) tensor
        params: sequence or tensor of 5 floats
            (tx_px, ty_px, log_sx, log_sy, rot_rad)
        device: torch device
        padding_mode: grid_sample padding for the image channels

    Returns:
        (warped (B, H, W, C), validity_mask (B, H, W))
    """
    B, H, W, C = image.shape
    if not torch.is_tensor(params):
        params = torch.tensor(params, dtype=image.dtype, device=device)
    params = params.to(device=device, dtype=image.dtype).reshape(1, 5)

    theta = build_theta(params, H, W).expand(B, -1, -1)
    image_bchw = image.permute(0, 3, 1, 2)
    grid = F.affine_grid(theta, image_bchw.shape, align_corners=False)
    warped = F.grid_sample(
        image_bchw, grid, mode='bilinear',
        padding_mode=padding_mode, align_corners=False
    )

    ones = torch.ones(B, 1, H, W, device=device, dtype=image.dtype)
    validity = F.grid_sample(
        ones, grid, mode='bilinear',
        padding_mode='zeros', align_corners=False
    )

    return warped.permute(0, 2, 3, 1), validity.squeeze(1)


@dataclass
class AffineEstimate:
    """Result of estimate_affine."""
    params: tuple          # (tx_px, ty_px, log_sx, log_sy, rot_rad)
    tx: float
    ty: float
    scale_x: float
    scale_y: float
    rotation_deg: float
    method: str            # 'adam' | 'ecc' | 'identity'
    converged: bool
    ncc_identity: float    # edge-NCC before alignment (higher = better)
    ncc_final: float       # edge-NCC after alignment
    score_map_before: torch.Tensor  # (H, W) residual at identity
    score_map_after: torch.Tensor   # (H, W) residual at final params


def _level_loss(eff_params, gray_s, edges_o, weight, factor):
    """Warp stylized gray by params (t in full-res px), edge it, NCC it."""
    N = eff_params.shape[0]
    _, _, h, w = gray_s.shape
    level_params = torch.cat([
        eff_params[:, 0:2] / factor, eff_params[:, 2:5]
    ], dim=1)
    theta = build_theta(level_params, h, w)
    grid = F.affine_grid(theta, (N, 1, h, w), align_corners=False)
    src = gray_s.expand(N, -1, -1, -1)
    warped = F.grid_sample(
        src, grid, mode='bilinear',
        padding_mode='border', align_corners=False
    )
    validity = F.grid_sample(
        torch.ones_like(src), grid, mode='bilinear',
        padding_mode='zeros', align_corners=False
    )
    edges_w = extract_edges_soft(warped)
    return weighted_ncc_loss(
        edges_o.expand(N, -1, -1, -1), edges_w, weight.expand(N, -1, -1, -1) * validity
    )


def _residual_map(eff_params, gray_s, edges_o, weight):
    """Per-pixel normalized edge residual at full res, for visualization."""
    loss_unused = None  # single sample path
    N = 1
    _, _, h, w = gray_s.shape
    theta = build_theta(eff_params.reshape(1, 5), h, w)
    grid = F.affine_grid(theta, (N, 1, h, w), align_corners=False)
    warped = F.grid_sample(
        gray_s, grid, mode='bilinear',
        padding_mode='border', align_corners=False
    )
    validity = F.grid_sample(
        torch.ones_like(gray_s), grid, mode='bilinear',
        padding_mode='zeros', align_corners=False
    )
    edges_w = extract_edges_soft(warped)
    eps = 1e-6
    w_eff = weight * validity
    wsum = w_eff.sum() + eps
    o_n = edges_o / ((edges_o * w_eff).sum() / wsum + eps)
    w_n = edges_w / ((edges_w * w_eff).sum() / wsum + eps)
    return ((o_n - w_n).abs() * validity)[0, 0].detach()


def _ecc_fallback(edges_o, edges_s, scale_vec, allow_anisotropic):
    """
    cv2.findTransformECC fallback on the blurred edge maps.

    Returns a (5,) params tensor or None. Uses MOTION_AFFINE (cv2 has no
    similarity model and MOTION_EUCLIDEAN excludes scale), then projects
    out shear and validates against 1.5x the optimizer bounds.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    t_o = edges_o[0, 0].detach().float().cpu()
    t_s = edges_s[0, 0].detach().float().cpu()
    t_o = (t_o / (t_o.max() + 1e-6)).numpy().copy()
    t_s = (t_s / (t_s.max() + 1e-6)).numpy().copy()

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6
    )
    try:
        _, warp = cv2.findTransformECC(
            t_o, t_s, warp, cv2.MOTION_AFFINE, criteria, None, 5
        )
    except cv2.error:
        return None

    H, W = t_o.shape
    # ECC's warp maps template (output) pixel coords -> input pixel coords,
    # top-left origin: the same inverse-sampling role as our theta. Convert
    # to centered coords, invert to the forward model, decompose.
    A = np.array(warp[:, :2], dtype=np.float64)
    c = np.array([(W - 1) / 2.0, (H - 1) / 2.0], dtype=np.float64)
    b_c = A @ c + np.array(warp[:, 2], dtype=np.float64) - c

    try:
        M = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    t = -M @ b_c

    sx = float(np.hypot(M[0, 0], M[1, 0]))
    sy = float(np.hypot(M[0, 1], M[1, 1]))
    rot = float(np.arctan2(M[1, 0], M[0, 0]))
    if sx <= 0 or sy <= 0:
        return None
    if not allow_anisotropic:
        sx = sy = float(np.sqrt(sx * sy))

    max_t, max_ls, max_rot = (
        float(scale_vec[0]), float(scale_vec[2]), float(scale_vec[4])
    )
    rot_bound = max_rot * 1.5 if max_rot > 0 else 0.005
    if (
        abs(t[0]) > max_t * 1.5 or abs(t[1]) > max_t * 1.5
        or abs(math.log(sx)) > max_ls * 1.5
        or abs(math.log(sy)) > max_ls * 1.5
        or abs(rot) > rot_bound
    ):
        return None
    if max_rot == 0:
        rot = 0.0

    return torch.tensor(
        [float(t[0]), float(t[1]), math.log(sx), math.log(sy), rot],
        dtype=torch.float32,
    )


@torch.no_grad()
def estimate_affine(
    original: torch.Tensor,
    stylized: torch.Tensor,
    device: torch.device,
    max_translation: float = 32.0,
    max_scale_dev: float = 0.05,
    max_rotation_deg: float = 3.0,
    allow_anisotropic: bool = False,
    precision: str = "balanced",
    bg_mask: torch.Tensor = None,
    multi_start: bool = True,
) -> AffineEstimate:
    """
    Estimate the global affine transform aligning stylized to original.

    Args:
        original: (B, H, W, C) tensor in [0, 1]; frame 0 is used
        stylized: (B, H, W, C) tensor, same size
        device: torch device
        max_translation: translation bound in pixels
        max_scale_dev: scale bound (0.05 = +/- ~5%)
        max_rotation_deg: rotation bound in degrees (0 disables rotation)
        allow_anisotropic: estimate separate x/y scale
        precision: "fast" | "balanced" | "precise"
        bg_mask: optional (B, H, W) subject mask; alignment is weighted to
            the background (1 - mask)
        multi_start: keep True; coarse-stage multi-start is required to
            escape local minima (verified failure mode without it)

    Returns:
        AffineEstimate. method='identity' with converged=False means no
        transform beat the unaligned baseline; params are then identity.
    """
    _, H, W, _ = original.shape

    gray_o = to_grayscale(original[0:1]).unsqueeze(1).float()
    gray_s = to_grayscale(stylized[0:1]).unsqueeze(1).float()

    weight_full = torch.ones(1, 1, H, W, device=device, dtype=torch.float32)
    if bg_mask is not None:
        m = bg_mask[0:1].to(device=device, dtype=torch.float32)
        weight_full = (1.0 - m.clamp(0.0, 1.0)).unsqueeze(1)

    # Parameter scaling: optimization variable z is unit-normalized so a
    # single Adam lr is meaningful across px / log-scale / radians. A zero
    # scale freezes that parameter.
    max_ls = max(
        abs(math.log(max(1.0 - max_scale_dev, 1e-3))),
        math.log(1.0 + max_scale_dev),
    )
    max_rot = math.radians(max(max_rotation_deg, 0.0))
    scale_vec = torch.tensor(
        [max_translation, max_translation, max_ls, max_ls, max_rot],
        device=device, dtype=torch.float32,
    )

    def effective(z):
        p = z * scale_vec
        lsy = p[:, 3] if allow_anisotropic else p[:, 2]
        return torch.stack([p[:, 0], p[:, 1], p[:, 2], lsy, p[:, 4]], dim=1)

    factors = [f for f in (4, 2, 1) if min(H, W) // f >= 32] or [1]
    iters_all = PRECISION_ITERS.get(precision, PRECISION_ITERS["balanced"])

    # The caller (align_frames) runs under @torch.no_grad(); optimization
    # must re-enable autograd, and cloning inside the block also defuses
    # potential inference-mode tensors.
    with torch.enable_grad():
        levels = {}
        for f in factors:
            if f > 1:
                g_o = F.avg_pool2d(gray_o, f, f).clone()
                g_s = F.avg_pool2d(gray_s, f, f).clone()
                w_l = F.avg_pool2d(weight_full, f, f).clone()
            else:
                g_o, g_s, w_l = (
                    gray_o.clone(), gray_s.clone(), weight_full.clone()
                )
            levels[f] = (extract_edges_soft(g_o).detach(), g_s, w_l)

        # Translation seed from phase correlation on full-res edge maps
        e_o_full = levels[1][0] if 1 in levels else extract_edges_soft(gray_o)
        e_s_full = extract_edges_soft(gray_s)
        px, py = phase_correlation(e_o_full[0, 0], e_s_full[0, 0])
        px = max(-max_translation, min(max_translation, float(px)))
        py = max(-max_translation, min(max_translation, float(py)))

        # Multi-start grid at the coarsest level (z units)
        starts = [[px / max(max_translation, 1e-6),
                   py / max(max_translation, 1e-6), 0.0, 0.0, 0.0]]
        if multi_start:
            for gy in (-0.6, 0.0, 0.6):
                for gx in (-0.6, 0.0, 0.6):
                    starts.append([gx, gy, 0.0, 0.0, 0.0])

        z = torch.tensor(starts, device=device, dtype=torch.float32)
        stage_factors = (
            [factors[0]] + factors[1:] + [factors[-1]] * 3
        )[:len(iters_all)]

        for stage, (iters, lr) in enumerate(zip(iters_all, STAGE_LRS)):
            if iters <= 0:
                continue
            factor = stage_factors[stage]
            edges_o_l, gray_s_l, w_l = levels[factor]

            if stage > 0 and z.shape[0] > 1:
                # Collapse to the best start after the coarse stage
                with torch.no_grad():
                    losses = _level_loss(
                        effective(z), gray_s_l, edges_o_l, w_l, factor
                    )
                    z = z[losses.argmin().item()].reshape(1, 5)

            z = z.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([z], lr=lr)
            prev_best = float('inf')
            for it in range(iters):
                optimizer.zero_grad(set_to_none=True)
                losses = _level_loss(
                    effective(z), gray_s_l, edges_o_l, w_l, factor
                )
                losses.sum().backward()
                optimizer.step()
                with torch.no_grad():
                    z.clamp_(-1.0, 1.0)
                if (it + 1) % 10 == 0:
                    cur_best = float(losses.detach().min())
                    if prev_best - cur_best < 1e-5:
                        break
                    prev_best = cur_best

            z = z.detach()

        # Final pick at the optimized resolution
        edges_o_l, gray_s_l, w_l = levels[stage_factors[-1]]
        final_losses = _level_loss(
            effective(z), gray_s_l, edges_o_l, w_l, stage_factors[-1]
        )
        z_best = z[final_losses.argmin().item()].reshape(1, 5)

    # --- Acceptance vs identity, scored at full resolution ---
    edges_o_full = e_o_full
    identity = torch.zeros(1, 5, device=device, dtype=torch.float32)

    def full_loss(eff):
        return float(_level_loss(eff, gray_s, edges_o_full, weight_full, 1)[0])

    loss_identity = full_loss(identity)
    cand_params = effective(z_best).detach()
    loss_adam = full_loss(cand_params)

    method = "adam"
    final_params = cand_params
    final_loss = loss_adam

    if loss_adam >= loss_identity - 1e-4:
        ecc = _ecc_fallback(
            edges_o_full, e_s_full, scale_vec, allow_anisotropic
        )
        if ecc is not None:
            ecc_eff = ecc.to(device).reshape(1, 5)
            loss_ecc = full_loss(ecc_eff)
            if loss_ecc < loss_identity - 1e-4:
                method, final_params, final_loss = "ecc", ecc_eff, loss_ecc
            else:
                method, final_params, final_loss = (
                    "identity", identity, loss_identity
                )
        else:
            method, final_params, final_loss = (
                "identity", identity, loss_identity
            )

    converged = method != "identity"
    score_before = _residual_map(identity, gray_s, edges_o_full, weight_full)
    score_after = _residual_map(final_params, gray_s, edges_o_full, weight_full)

    p = [float(v) for v in final_params[0]]
    return AffineEstimate(
        params=tuple(p),
        tx=p[0],
        ty=p[1],
        scale_x=math.exp(p[2]),
        scale_y=math.exp(p[3]),
        rotation_deg=math.degrees(p[4]),
        method=method,
        converged=converged,
        ncc_identity=1.0 - loss_identity,
        ncc_final=1.0 - final_loss,
        score_map_before=score_before,
        score_map_after=score_after,
    )
