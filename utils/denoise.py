"""
Tensor math for the Deep Denoise node. No comfy imports, so it can be
tested standalone.

Layout convention: everything here is (B, C, H, W) float in [0, 1].
"""

import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)

# Dihedral transforms as (rot90 count, flip). The first N entries are used
# for an N-fold self-ensemble.
TTA_TRANSFORMS = (
    (0, False), (0, True), (2, False), (2, True),
    (1, False), (1, True), (3, False), (3, True),
)


def estimate_sigma(image: torch.Tensor) -> float:
    """
    Blind noise standard deviation of one image, in [0, 1] units.

    Immerkaer's method: a Laplacian-difference kernel cancels image
    structure up to second order, so what is left is noise. The median
    (instead of the paper's mean) keeps edges and texture from inflating
    the estimate.
    """
    luma = sum(w * image[:, i:i + 1] for i, w in enumerate(LUMA))
    kernel = torch.tensor(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
        device=image.device, dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    response = F.conv2d(luma, kernel).abs()
    # Kernel L2 norm is 6; median(|N(0,1)|) is 0.6745. Mixing three
    # independent channels into luma shrinks the noise by the L2 norm
    # of the weights, so divide that back out.
    luma_norm = sum(w * w for w in LUMA) ** 0.5
    return response.median().item() / (6.0 * 0.6745 * luma_norm)


def self_ensemble(function, image: torch.Tensor, folds: int) -> torch.Tensor:
    """Average function() over flipped / rotated copies of the input."""
    if folds <= 1:
        return function(image)
    out = None
    for rot, flip in TTA_TRANSFORMS[:folds]:
        x = torch.rot90(image, rot, dims=(2, 3))
        if flip:
            x = x.flip(3)
        y = function(x)
        if flip:
            y = y.flip(3)
        y = torch.rot90(y, -rot, dims=(2, 3))
        out = y if out is None else out.add_(y)
    return out.div_(folds)


def pad_to_multiple(image: torch.Tensor, multiple: int):
    """Replicate-pad bottom/right to a size multiple. Returns (padded, H, W)."""
    h, w = image.shape[2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    return image, h, w


def blend_luma_chroma(noisy, denoised, luma_strength, chroma_strength):
    """
    Apply the denoiser's correction with separate luma / chroma amounts.

    The correction (denoised - noisy) splits linearly into a grey part
    (its luma, the same on all three channels) and a colour part (the
    rest), so no colourspace round trip is needed.
    """
    residual = denoised - noisy
    luma = sum(w * residual[:, i:i + 1] for i, w in enumerate(LUMA))
    chroma = residual - luma
    return noisy + luma_strength * luma + chroma_strength * chroma


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp image (1, C, H, W) by a (2, H, W) pixel flow (x, y)."""
    h, w = image.shape[2:]
    ys = torch.arange(h, device=image.device, dtype=image.dtype)
    xs = torch.arange(w, device=image.device, dtype=image.dtype)
    grid_x = (xs.view(1, w) + flow[0]) * (2.0 / max(w - 1, 1)) - 1.0
    grid_y = (ys.view(h, 1) + flow[1]) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    return F.grid_sample(
        image, grid, mode="bicubic", padding_mode="border",
        align_corners=True,
    )


def patch_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared difference over a 5x5 patch, as (1, 1, H, W)."""
    dist = (a - b).pow(2).mean(dim=1, keepdim=True)
    return F.avg_pool2d(dist, 5, stride=1, padding=2)


def temporal_fuse(frames: torch.Tensor, radius: int, tolerance: float,
                  device, flow_function=None) -> torch.Tensor:
    """
    Motion-safe temporal smoothing of an already-denoised frame batch.

    Each pixel is averaged with the matching pixel in neighbouring
    frames, weighted by how similar the local 5x5 patch is. That removes
    frame-to-frame shimmer; anything that does not match (occlusion, a
    bad flow vector, a cut) gets a near-zero weight, so it cannot ghost.

    flow_function(center, neighbour) returns a (2, H, W) flow with
    center(x) ~ neighbour(x + flow). With it, neighbours are aligned
    first, so moving shots fuse too. Without it only static pixels fuse.
    Each pixel uses whichever of the aligned / unaligned neighbour
    matches better, so flow can only add matches, never remove them.
    """
    count = frames.shape[0]
    out = torch.empty_like(frames)
    scale = -1.0 / (2.0 * tolerance * tolerance)
    for t in range(count):
        center = frames[t:t + 1].to(device)
        total = center.clone()
        weight_sum = torch.ones_like(center[:, :1])
        for n in range(max(0, t - radius), min(count, t + radius + 1)):
            if n == t:
                continue
            neighbour = frames[n:n + 1].to(device)
            dist = patch_distance(neighbour, center)
            if flow_function is not None:
                # Flow is unreliable on flat or held areas, so keep the
                # aligned pixel only where it matches better than the
                # untouched one.
                warped = warp(neighbour, flow_function(center, neighbour))
                warped_dist = patch_distance(warped, center)
                better = warped_dist < dist
                neighbour = torch.where(better, warped, neighbour)
                dist = torch.where(better, warped_dist, dist)
            weight = torch.exp(dist * scale)
            total.addcmul_(neighbour, weight)
            weight_sum.add_(weight)
        out[t] = (total / weight_sum).to(out.device)
    return out


def sigma_label(sigma: float) -> str:
    return f"{sigma * 255.0:.1f}/255"


def clamp_sigma(sigma: float) -> float:
    # DRUNet was trained on sigma 0..50 (of 255).
    return min(max(sigma, 0.0), 50.0 / 255.0)

