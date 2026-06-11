"""
Inpainting utilities for TrentNodes.

Provides multiple inpainting methods:
- SD 1.5 diffusion-based inpainting (highest quality)
- Clone-stamp iterative fill (texture-preserving)
- Simple blur fill (fast fallback)
"""

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.sample

from .mask_ops import dilate_mask, erode_mask
from .model_cache import load_sd_inpaint_model


def sd_inpaint(
    image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    checkpoint: str = "sd-v1-5-inpainting.safetensors",
    steps: int = 20,
    denoise: float = 0.9,
    cfg: float = 7.5,
    seed: int = 42
) -> torch.Tensor:
    """
    Use Stable Diffusion 1.5 inpainting model to fill masked regions.

    Provides high-quality context-aware inpainting using diffusion.

    Args:
        image: (B, H, W, C) image tensor in [0, 1] range
        mask: (B, H, W) inpaint mask (1 = area to inpaint)
        device: torch device
        checkpoint: inpainting model checkpoint name
        steps: diffusion steps (more = better quality, slower)
        denoise: denoise strength (0.5-1.0, higher = more change)
        cfg: classifier-free guidance scale
        seed: random seed for reproducibility

    Returns:
        Inpainted image tensor (B, H, W, C)
    """
    B, H, W, C = image.shape
    print(f"[TrentNodes] Running SD inpainting ({steps} steps)...")

    # Load inpainting model (cached); fall back gracefully when the
    # checkpoint can't be loaded (e.g. first-run download failure) instead
    # of aborting the whole node mid-execution.
    try:
        model, clip, vae = load_sd_inpaint_model(checkpoint, device)
    except Exception as e:
        print(
            f"[TrentNodes] SD inpaint checkpoint unavailable ({e}); "
            f"falling back to clone_stamp inpainting"
        )
        return clone_stamp_inpaint(
            image, mask, device, iterations=25, sample_radius=12
        )

    # Encode empty prompts
    tokens = clip.tokenize("")
    positive = clip.encode_from_tokens_scheduled(tokens)
    negative = clip.encode_from_tokens_scheduled(tokens)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # Pad to the VAE downscale multiple with replicate padding (the old
    # center-crop left a 0-7 px un-inpainted strip around the border).
    downscale = getattr(vae, 'downscale_ratio', 8)
    pad_h = (downscale - H % downscale) % downscale
    pad_w = (downscale - W % downscale) % downscale

    if pad_h or pad_w:
        pixels = F.pad(
            image.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h),
            mode='replicate'
        ).permute(0, 2, 3, 1)
        inpaint_mask = F.pad(
            mask.unsqueeze(1), (0, pad_w, 0, pad_h),
            mode='constant', value=0
        ).squeeze(1)
    else:
        pixels = image.clone()
        inpaint_mask = mask.clone()

    # Grow mask for seamless blending
    grow_mask_by = 8
    if grow_mask_by > 0:
        inpaint_mask_4d = inpaint_mask.unsqueeze(1).to(device)
        kernel_size = grow_mask_by * 2 + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)
        padding = grow_mask_by
        mask_grown = torch.clamp(
            F.conv2d(inpaint_mask_4d.round(), kernel, padding=padding),
            0, 1
        ).squeeze(1)
    else:
        mask_grown = inpaint_mask.to(device)

    # Apply mask to image (set inpaint area to gray)
    m = (1.0 - inpaint_mask.round()).to(device)
    pixels = pixels.to(device)
    for i in range(3):
        pixels[:, :, :, i] = pixels[:, :, :, i] * m + 0.5 * (1 - m)

    # VAE encode
    latent_samples = vae.encode(pixels)
    latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

    # Prepare noise
    noise = comfy.sample.prepare_noise(latent_samples, seed, None)

    # Run sampling
    samples = comfy.sample.sample(
        model, noise, steps, cfg,
        "dpmpp_2m", "karras",
        positive, negative, latent_samples,
        denoise=denoise,
        noise_mask=mask_grown,
        seed=seed
    )

    # VAE decode, then crop any padding back off
    samples = samples.to(mm.intermediate_device())
    result = vae.decode(samples)
    result = result[:, :H, :W, :].cpu()

    print("[TrentNodes] SD inpainting complete.")
    return result


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
    method: str = "sd_inpaint",
    **kwargs
) -> torch.Tensor:
    """
    Unified inpainting interface.

    Args:
        image: (B, H, W, C) image tensor
        mask: (B, H, W) inpaint mask (1 = area to inpaint)
        device: torch device
        method: "sd_inpaint", "clone_stamp", or "blur"
        **kwargs: Method-specific arguments

    Returns:
        Inpainted image tensor
    """
    if method == "sd_inpaint":
        return sd_inpaint(image, mask, device, **kwargs)
    elif method == "clone_stamp":
        return clone_stamp_inpaint(image, mask, device, **kwargs)
    elif method == "blur":
        return blur_inpaint(image, mask, device, **kwargs)
    else:
        raise ValueError(f"Unknown inpainting method: {method}")


def inpaint_transform_edges(
    image: torch.Tensor,
    validity_mask: torch.Tensor,
    device: torch.device,
    method: str = "sd_inpaint",
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
        method: Inpainting method ("sd_inpaint", "clone_stamp", "blur")
        steps: SD inpaint diffusion steps
        denoise: SD inpaint denoise strength
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
    if method == "sd_inpaint":
        result = inpaint(
            image, edge_mask, device,
            method="sd_inpaint",
            steps=steps,
            denoise=denoise
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
