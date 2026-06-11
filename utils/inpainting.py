"""
Inpainting utilities for TrentNodes.

Provides multiple inpainting methods for unprompted background fill
(clean-plate / object removal):
- big-lama: purpose-built removal model, single forward pass (default)
- VOID (Netflix): video inpainting via ComfyUI core, HQ tier
- Clone-stamp iterative fill (texture-preserving fallback)
- Simple blur fill (fast fallback)
"""

import torch
import torch.nn.functional as F

import comfy.model_management as mm

from .mask_ops import dilate_mask, erode_mask
from .model_cache import load_lama_model, load_void_models


def lama_inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Fill masked regions with big-lama (purpose-built removal model).

    Single forward pass, no prompt, ~0.1-1s per frame. Falls back to
    clone_stamp if the model can't be loaded.

    Args:
        image: (B, H, W, C) image tensor in [0, 1] range
        mask: (B, H, W) inpaint mask (1 = area to inpaint)
        device: torch device

    Returns:
        Inpainted image tensor (B, H, W, C) on `device`
    """
    B, H, W, C = image.shape
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    try:
        model = load_lama_model(device)
    except Exception as e:
        print(
            f"[TrentNodes] big-lama unavailable ({e}); "
            f"falling back to clone_stamp inpainting"
        )
        return clone_stamp_inpaint(
            image, mask, device, iterations=25, sample_radius=12
        )

    # LaMa wants (1, 3, H, W) in [0, 1] + (1, 1, H, W) binary mask,
    # dims divisible by 8 (replicate-pad, crop after)
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    im = image[..., :3].permute(0, 3, 1, 2).float().to(device)
    m = (mask > 0.5).float().unsqueeze(1).to(device)
    if pad_h or pad_w:
        im = F.pad(im, (0, pad_w, 0, pad_h), mode='replicate')
        m = F.pad(m, (0, pad_w, 0, pad_h), mode='constant', value=0)

    fills = []
    with torch.no_grad():
        for b in range(B):
            fills.append(model(im[b:b + 1], m[b:b + 1]))
    fill = torch.cat(fills, dim=0)[..., :H, :W]
    fill = fill.permute(0, 2, 3, 1).clamp(0, 1)

    # Composite: only masked pixels come from the fill
    m3 = (mask > 0.5).float().unsqueeze(-1).to(device)
    result = image.to(device) * (1 - m3) + fill.to(image.dtype) * m3
    return result


# VOID works at its training-scale resolution; larger frames are filled at
# this cap and the fill upscaled back into the masked region only.
VOID_MAX_W = 1344
VOID_MAX_H = 768
VOID_FRAMES = 5  # minimum valid clip length (latent_t must be even)

_void_empty_cond = None


@torch.no_grad()
def _void_fill_frame(frame_hwc, mask_hw, steps, seed, cfg=6.0):
    """
    Run the 2-pass VOID pipeline on one frame (replicated x5).

    The no_grad decorator is load-bearing outside the ComfyUI executor:
    without it the DDIM loop retains autograd graphs (~13GB/step,
    verified) and OOMs.
    """
    from comfy_extras.nodes_custom_sampler import (
        BasicScheduler, CFGGuider, RandomNoise, SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_void import (
        VOIDInpaintConditioning, VOIDSampler,
        VOIDWarpedNoise, VOIDWarpedNoiseSource,
    )

    global _void_empty_cond
    model1, model2, clip, vae, flow = load_void_models()

    H, W = frame_hwc.shape[:2]
    scale = min(VOID_MAX_W / W, VOID_MAX_H / H, 1.0)
    # /16-divisible work resolution (even latent dims for patch_size 2)
    work_w = max(64, int(W * scale) // 16 * 16)
    work_h = max(64, int(H * scale) // 16 * 16)

    if _void_empty_cond is None:
        tokens = clip.tokenize("")
        _void_empty_cond = clip.encode_from_tokens_scheduled(tokens)
    cond = _void_empty_cond

    video = frame_hwc.unsqueeze(0).repeat(VOID_FRAMES, 1, 1, 1).cpu().float()
    quadmask = (mask_hw > 0.5).float().unsqueeze(0).repeat(
        VOID_FRAMES, 1, 1
    ).cpu()

    pos, neg, latent = VOIDInpaintConditioning.execute(
        cond, cond, vae, video, quadmask, work_w, work_h, VOID_FRAMES, 1
    ).args

    sampler = VOIDSampler.execute().args[0]

    # Pass 1: random noise
    guider1 = CFGGuider().get_guider(model1, pos, neg, cfg)[0]
    sigmas1 = BasicScheduler().get_sigmas(model1, "simple", steps, 1.0)[0]
    noise1 = RandomNoise().get_noise(seed)[0]
    out1, _ = SamplerCustomAdvanced().sample(
        noise1, guider1, sampler, sigmas1, latent
    )
    video1 = vae.decode(out1["samples"])
    if video1.ndim == 5:
        video1 = video1.reshape(-1, *video1.shape[-3:])

    # Pass 2: optical-flow warped noise refinement
    warped = VOIDWarpedNoise.execute(
        flow, video1, work_w, work_h, VOID_FRAMES, 1
    ).args[0]
    noise2 = VOIDWarpedNoiseSource.execute(warped).args[0]
    guider2 = CFGGuider().get_guider(model2, pos, neg, cfg)[0]
    sigmas2 = BasicScheduler().get_sigmas(model2, "simple", steps, 1.0)[0]
    out2, _ = SamplerCustomAdvanced().sample(
        noise2, guider2, sampler, sigmas2, latent
    )
    video2 = vae.decode(out2["samples"])
    if video2.ndim == 5:
        video2 = video2.reshape(-1, *video2.shape[-3:])

    mid = video2[VOID_FRAMES // 2]  # (work_h', work_w', 3)
    if mid.shape[0] != H or mid.shape[1] != W:
        mid = F.interpolate(
            mid.permute(2, 0, 1).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
    return mid.clamp(0, 1)


def void_inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    steps: int = 20,
    seed: int = 43,
) -> torch.Tensor:
    """
    Fill masked regions with Netflix VOID (HQ tier) via ComfyUI core.

    VOID is a video model (CogVideoX-Fun-V1.5); single frames run through
    a 5-frame replicated clip and the middle output frame is used. Two
    diffusion passes per fill. Falls back to lama_inpaint on any failure.

    Args:
        image: (B, H, W, C) image tensor in [0, 1] range
        mask: (B, H, W) inpaint mask (1 = area to inpaint)
        device: torch device
        steps: diffusion steps per pass
        seed: noise seed

    Returns:
        Inpainted image tensor (B, H, W, C) on `device`
    """
    B, H, W, C = image.shape
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    try:
        fills = []
        for b in range(B):
            print(
                f"[TrentNodes] VOID inpainting frame {b + 1}/{B} "
                f"({steps} steps x 2 passes)..."
            )
            fills.append(
                _void_fill_frame(image[b, ..., :3], mask[b], steps, seed)
            )
        fill = torch.stack(fills, dim=0).to(device)
    except Exception as e:
        print(
            f"[TrentNodes] VOID inpainting failed ({e}); "
            f"falling back to big-lama"
        )
        return lama_inpaint(image, mask, device)
    finally:
        mm.soft_empty_cache()

    m3 = (mask > 0.5).float().unsqueeze(-1).to(device)
    return image.to(device) * (1 - m3) + fill.to(image.dtype) * m3


def clone_stamp_inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    iterations: int = 25,
    sample_radius: int = 12
) -> torch.Tensor:
    """
    Clone-stamp style inpainting that samples from background pixels.

    Fills masked region by iteratively extending edges inward,
    sampling only from valid (non-masked) pixels.

    Args:
        image: (B, H, W, C) image tensor
        mask: (B, H, W) or (H, W) mask where 1 = inpaint region
        device: torch device
        iterations: Number of inward-fill passes
        sample_radius: How far to look for source pixels

    Returns:
        Inpainted image tensor
    """
    B, H, W, C = image.shape

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # Blank out the masked region
    original_mask = mask.clone()
    mask_3d = mask.unsqueeze(-1).expand(-1, -1, -1, C)

    result = image.clone()
    result = result * (1 - mask_3d)

    # Track valid source pixels
    valid_source = (1 - original_mask).clone()

    # Create sampling kernel
    kernel_size = sample_radius * 2 + 1
    y_coords = torch.arange(kernel_size, device=device) - sample_radius
    x_coords = torch.arange(kernel_size, device=device) - sample_radius
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    dist = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    weights = torch.exp(-dist / (sample_radius * 0.4))
    weights[sample_radius, sample_radius] = 0
    weights = weights / weights.sum()
    weights = weights.view(1, 1, kernel_size, kernel_size)

    remaining_mask = original_mask.clone()

    for _ in range(iterations):
        # Find edge pixels
        valid_dilated = dilate_mask(valid_source, radius=1, device=device)
        if valid_dilated.dim() == 2:
            valid_dilated = valid_dilated.unsqueeze(0)

        edge_mask = remaining_mask * valid_dilated
        edge_mask = torch.clamp(edge_mask, 0, 1)

        if edge_mask.sum() < 1:
            break

        # Sample from valid neighbors
        pad = sample_radius
        result_padded = F.pad(
            result.permute(0, 3, 1, 2),
            (pad, pad, pad, pad), mode='replicate'
        )
        valid_padded = F.pad(
            valid_source.unsqueeze(1),
            (pad, pad, pad, pad), mode='constant', value=0
        )

        filled_values = torch.zeros_like(result)
        total_weight = torch.zeros(B, H, W, device=device)

        for c in range(C):
            channel = result_padded[:, c:c + 1, :, :]
            weighted_vals = F.conv2d(channel * valid_padded, weights, padding=0)
            weight_sum = F.conv2d(valid_padded, weights, padding=0)

            weight_sum_safe = weight_sum.clamp(min=1e-6)
            filled_c = (weighted_vals / weight_sum_safe).squeeze(1)
            filled_values[:, :, :, c] = filled_c

            if c == 0:
                total_weight = weight_sum.squeeze(1)

        # Update edge pixels
        has_valid_neighbors = (total_weight > 0.01).float()
        update_mask = edge_mask * has_valid_neighbors
        update_mask_3d = update_mask.unsqueeze(-1).expand(-1, -1, -1, C)

        result = result * (1 - update_mask_3d) + filled_values * update_mask_3d
        valid_source = torch.clamp(valid_source + update_mask, 0, 1)
        remaining_mask = remaining_mask * (1 - update_mask)

    return result


def blur_inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    iterations: int = 3
) -> torch.Tensor:
    """
    Simple inpainting that fills the masked region inward from valid
    pixels using normalized convolution, then smooths the fill.

    Fast fallback method. (The previous implementation blurred the masked
    region with its own content, so large regions kept their original
    color and were never actually replaced.)

    Args:
        image: (B, H, W, C) image tensor
        mask: (B, H, W) or (H, W) mask where 1 = inpaint region
        device: torch device
        iterations: Number of final smoothing passes

    Returns:
        Inpainted image tensor
    """
    B, H, W, C = image.shape

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    hard_mask = (mask > 0.5).float()
    remaining = hard_mask.clone()
    result = image.clone()

    # Each pass fills a ~7px ring from currently-valid pixels; loop until
    # the region closes (capped well above any realistic mask size).
    for _ in range(64):
        if remaining.sum() < 1:
            break
        valid = (1.0 - remaining).unsqueeze(1)
        rem_3d = remaining.unsqueeze(-1)
        num = F.avg_pool2d(
            (result * (1 - rem_3d)).permute(0, 3, 1, 2),
            kernel_size=15, stride=1, padding=7
        )
        den = F.avg_pool2d(valid, kernel_size=15, stride=1, padding=7)
        filled = (num / den.clamp(min=1e-6)).permute(0, 2, 3, 1)

        has_source = (den.squeeze(1) > 1e-4).float()
        update = remaining * has_source
        update_3d = update.unsqueeze(-1)
        result = result * (1 - update_3d) + filled * update_3d
        remaining = remaining * (1 - update)

    # Smooth the filled region for seamless blending
    mask_3d = hard_mask.unsqueeze(-1)
    for _ in range(iterations):
        blurred = F.avg_pool2d(
            result.permute(0, 3, 1, 2),
            kernel_size=15, stride=1, padding=7
        ).permute(0, 2, 3, 1)
        result = result * (1 - mask_3d) + blurred * mask_3d

    return result


def inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    method: str = "lama",
    **kwargs
) -> torch.Tensor:
    """
    Unified inpainting interface.

    Args:
        image: (B, H, W, C) image tensor
        mask: (B, H, W) inpaint mask (1 = area to inpaint)
        device: torch device
        method: "lama" (default), "void", "clone_stamp", or "blur".
            "sd_inpaint" is accepted as a legacy alias for "lama" so old
            saved workflows keep working (the SD 1.5 backend was removed).
        **kwargs: Method-specific arguments

    Returns:
        Inpainted image tensor
    """
    if method == "sd_inpaint":
        method = "lama"

    # Each backend takes different knobs; select rather than forward so
    # call sites can pass a uniform kwarg set.
    if method == "lama":
        return lama_inpaint(image, mask, device)
    elif method == "void":
        return void_inpaint(
            image, mask, device,
            steps=kwargs.get("steps", 20),
            seed=kwargs.get("seed", 43),
        )
    elif method == "clone_stamp":
        return clone_stamp_inpaint(
            image, mask, device,
            iterations=kwargs.get("iterations", 25),
            sample_radius=kwargs.get("sample_radius", 12),
        )
    elif method == "blur":
        return blur_inpaint(
            image, mask, device,
            iterations=kwargs.get("iterations", 5),
        )
    else:
        raise ValueError(f"Unknown inpainting method: {method}")


def inpaint_transform_edges(
    image: torch.Tensor,
    validity_mask: torch.Tensor,
    device: torch.device,
    method: str = "lama",
    steps: int = 20,
    denoise: float = 0.9,
    edge_threshold: float = 0.99,
    dilate_radius: int = 4
) -> torch.Tensor:
    """
    Inpaint edge regions created by affine transforms.

    When an image is scaled down or translated, the edges contain
    border-replicated pixels (ugly stretched edges). This function
    detects those regions via the validity mask and inpaints them.

    Args:
        image: (B, H, W, C) transformed image tensor
        validity_mask: (B, H, W) mask from apply_affine_transform_with_mask
                       where 1.0 = real pixels, <1.0 = border-replicated
        device: torch device
        method: Inpainting method ("lama", "void", "clone_stamp", "blur";
                "sd_inpaint" maps to "lama")
        steps: diffusion steps (void only)
        denoise: unused (kept for call-site compatibility)
        edge_threshold: Pixels with validity < threshold need inpainting
        dilate_radius: Pixels to expand edge mask for better blending

    Returns:
        Image with edges inpainted
    """
    # Create edge mask (where validity < threshold = border-replicated)
    edge_mask = (validity_mask < edge_threshold).float()

    # Skip if no edges need inpainting (e.g., scale > 1.0 case)
    edge_pixel_count = edge_mask.sum().item()
    if edge_pixel_count < 10:
        return image

    B, H, W, C = image.shape
    edge_width = int((edge_pixel_count / B) ** 0.5 / 4)  # Rough estimate
    print(
        f"[TrentNodes] Edge inpainting: ~{int(edge_pixel_count/B)} pixels "
        f"(~{edge_width}px border)"
    )

    # Expand edge mask slightly for better blending
    if dilate_radius > 0:
        edge_mask = dilate_mask(edge_mask, radius=dilate_radius, device=device)
        if edge_mask.dim() == 2:
            edge_mask = edge_mask.unsqueeze(0)

    # Use the unified inpaint interface
    if method in ("lama", "sd_inpaint", "void"):
        result = inpaint(
            image, edge_mask, device,
            method=method,
            steps=steps,
        )
    elif method == "clone_stamp":
        result = inpaint(
            image, edge_mask, device,
            method="clone_stamp",
            iterations=25,
            sample_radius=12
        )
    else:
        result = inpaint(
            image, edge_mask, device,
            method="blur",
            iterations=5
        )

    return result
