"""
GPU-accelerated video degradation utilities for TrentNodes.

Provides temporally coherent degradation effects for generating
synthetic training pairs (clean + degraded) for AI video models.

Effects operate on (B, C, H, W) float32 GPU tensors and are
designed to be composed in a fixed processing order for realism.
"""

import io
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .mask_ops import create_gaussian_kernel


# -----------------------------------------------------------
# Section 1: Temporal Interpolation Engine
# -----------------------------------------------------------

def _linspace_schedule(
    value: float,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Constant schedule: same value every frame."""
    return torch.full(
        (num_frames,), value,
        device=device, dtype=torch.float32,
    )


def _breathing_schedule(
    intensity: float,
    num_frames: int,
    device: torch.device,
    frequency: float = 1.0,
) -> torch.Tensor:
    """Sinusoidal breathing oscillation around intensity."""
    t = torch.linspace(
        0, 2.0 * math.pi * frequency,
        num_frames, device=device,
    )
    wave = 0.5 + 0.5 * torch.sin(t)
    return wave * intensity


def _ramp_schedule(
    start: float,
    end: float,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Linear ramp from start to end."""
    if num_frames == 1:
        return torch.tensor([start], device=device)
    return torch.linspace(start, end, num_frames, device=device)


def _random_smooth_schedule(
    intensity: float,
    num_frames: int,
    device: torch.device,
    rng: torch.Generator,
    smoothing: int = 5,
) -> torch.Tensor:
    """Random but temporally smooth schedule via blur."""
    raw = torch.rand(
        num_frames, device=device, generator=rng,
    ) * intensity
    if num_frames < 3 or smoothing < 1:
        return raw
    # 1D Gaussian smooth along time axis
    k = min(smoothing * 2 + 1, num_frames)
    if k % 2 == 0:
        k -= 1
    half = k // 2
    sigma = half / 2.0
    x = torch.arange(k, device=device, dtype=torch.float32)
    x = x - half
    kernel = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, k)
    raw_3d = raw.view(1, 1, -1)
    padded = F.pad(raw_3d, (half, half), mode='reflect')
    smoothed = F.conv1d(padded, kernel)
    return smoothed.view(-1)


def _angle_schedule(
    mode: str,
    num_frames: int,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate motion blur angle schedule (degrees)."""
    if mode == "horizontal":
        return torch.zeros(num_frames, device=device)
    elif mode == "vertical":
        return torch.full(
            (num_frames,), 90.0, device=device,
        )
    elif mode == "diagonal":
        return torch.full(
            (num_frames,), 45.0, device=device,
        )
    elif mode == "random_per_frame":
        return torch.rand(
            num_frames, device=device, generator=rng,
        ) * 360.0
    else:
        # random_consistent: smooth drift
        base = torch.rand(
            1, device=device, generator=rng,
        ) * 360.0
        drift = _random_smooth_schedule(
            30.0, num_frames, device, rng, smoothing=7,
        )
        return base + drift


# -----------------------------------------------------------
# Section 2: Individual Degradation Effects
# -----------------------------------------------------------

def apply_lens_distortion(
    images: torch.Tensor,
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    """Barrel/pincushion distortion via radial grid warp."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape
    k1 = strength * 0.5  # barrel coefficient

    # Normalized coordinate grid [-1, 1]
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    r2 = grid_x * grid_x + grid_y * grid_y

    # Radial distortion
    factor = 1.0 + k1 * r2
    dx = grid_x * factor
    dy = grid_y * factor

    grid = torch.stack([dx, dy], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    return F.grid_sample(
        images, grid, mode='bilinear',
        padding_mode='border', align_corners=True,
    )


def apply_defocus_blur(
    images: torch.Tensor,
    intensity: float,
    mode: str,
    num_frames: int,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Gaussian defocus blur with temporal modes."""
    if intensity <= 0.0:
        return images
    B, C, H, W = images.shape

    # Map intensity to sigma (0.5 - 15.0)
    max_sigma = 15.0
    base_sigma = 0.5 + intensity * (max_sigma - 0.5)

    # Build per-frame sigma schedule
    if mode == "breathing":
        schedule = _breathing_schedule(
            base_sigma, num_frames, device,
        )
    elif mode == "rack_focus":
        schedule = _ramp_schedule(
            0.5, base_sigma, num_frames, device,
        )
    elif mode == "edge_softness":
        # Will be handled specially below
        schedule = _linspace_schedule(
            base_sigma, num_frames, device,
        )
    else:  # uniform
        schedule = _linspace_schedule(
            base_sigma, num_frames, device,
        )

    # Group frames by quantized radius for efficiency
    radii = (schedule + 0.5).int().clamp(min=1)
    unique_radii = radii.unique()
    # In-place when only one radius; clone only if needed
    if len(unique_radii) == 1:
        result = images
    else:
        result = images.clone()

    for r_val in unique_radii:
        r = r_val.item()
        mask = radii == r_val
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        subset = images[idx]
        kernel_h, kernel_v = create_gaussian_kernel(r, device)
        # Apply separable Gaussian per channel
        sub_flat = subset.reshape(-1, 1, H, W)
        padded = F.pad(
            sub_flat, (r, r, r, r), mode='replicate',
        )
        blurred = F.conv2d(padded, kernel_h)
        blurred = F.conv2d(blurred, kernel_v)
        result[idx] = blurred.reshape(len(idx), C, H, W)

    if mode == "edge_softness":
        # Blend: sharp center, blurred edges
        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        r2 = (gx * gx + gy * gy).clamp(max=1.0)
        edge_mask = r2.view(1, 1, H, W)
        result = torch.lerp(images, result, edge_mask)

    return result


def _create_motion_kernel(
    length: int,
    angle_deg: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a directional motion blur kernel."""
    if length < 1:
        length = 1
    ks = length * 2 + 1
    # 1D line kernel
    kernel = torch.zeros(ks, ks, device=device)
    center = length
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    for i in range(-length, length + 1):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        x = max(0, min(ks - 1, x))
        y = max(0, min(ks - 1, y))
        kernel[y, x] = 1.0
    kernel = kernel / kernel.sum().clamp(min=1.0)
    return kernel.view(1, 1, ks, ks)


def apply_motion_blur(
    images: torch.Tensor,
    intensity: float,
    angle_mode: str,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Directional motion blur via convolution."""
    if intensity <= 0.0:
        return images
    B, C, H, W = images.shape

    # Map intensity to kernel half-length (1-22)
    half_len = max(1, int(intensity * 22))

    angles = _angle_schedule(angle_mode, B, device, rng)

    # Group by quantized angle for efficiency
    q_angles = (angles / 5.0).round() * 5.0
    unique_angles = q_angles.unique()
    if len(unique_angles) == 1:
        result = images
    else:
        result = images.clone()

    for a_val in unique_angles:
        idx = (q_angles == a_val).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        kernel = _create_motion_kernel(
            half_len, a_val.item(), device,
        )
        pad = half_len
        subset = images[idx]
        # Depthwise conv: process each channel independently
        sub_flat = subset.reshape(-1, 1, H, W)
        padded = F.pad(
            sub_flat, (pad, pad, pad, pad), mode='replicate',
        )
        blurred = F.conv2d(padded, kernel)
        result[idx] = blurred.reshape(len(idx), C, H, W)

    return result


def apply_rolling_shutter(
    images: torch.Tensor,
    strength: float,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Simulate rolling shutter via per-row horizontal shift."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape

    # Max pixel shift scales with strength and frame width
    max_shift = strength * 0.05 * W

    # Per-frame shift direction (smooth)
    directions = _random_smooth_schedule(
        1.0, B, device, rng, smoothing=5,
    ) * 2.0 - 1.0

    # Build per-row displacement grid
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    base_y, base_x = torch.meshgrid(yy, xx, indexing='ij')

    # Row factor: linear from -1 to 1 top to bottom
    row_factor = torch.linspace(
        -1, 1, H, device=device,
    ).view(H, 1)

    results = []
    for i in range(B):
        shift = directions[i] * max_shift
        # Pixel shift -> normalized coord shift
        dx = row_factor * (shift / (W * 0.5))
        gx = base_x + dx
        gy = base_y
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        warped = F.grid_sample(
            images[i:i + 1], grid, mode='bilinear',
            padding_mode='border', align_corners=True,
        )
        results.append(warped)

    return torch.cat(results, dim=0)


def apply_chromatic_aberration(
    images: torch.Tensor,
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    """Color fringing via per-channel radial scaling."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape
    if C < 3:
        return images

    # Normalized grid
    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')

    shift = strength * 0.02  # Subtle radial scaling

    channels = []
    for ch, scale in enumerate([1.0 + shift, 1.0, 1.0 - shift]):
        gx = grid_x * scale
        gy = grid_y * scale
        grid = torch.stack([gx, gy], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        ch_data = images[:, ch:ch + 1, :, :]
        warped = F.grid_sample(
            ch_data, grid, mode='bilinear',
            padding_mode='border', align_corners=True,
        )
        channels.append(warped)

    # Keep any extra channels (alpha etc) unchanged
    result = torch.cat(channels, dim=1)
    if C > 3:
        result = torch.cat(
            [result, images[:, 3:, :, :]], dim=1,
        )
    return result


def apply_noise(
    images: torch.Tensor,
    intensity: float,
    noise_type: str,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Add noise with various models."""
    if intensity <= 0.0:
        return images
    B, C, H, W = images.shape

    if noise_type == "gaussian":
        noise = torch.randn(
            B, C, H, W, device=device, generator=rng,
        ) * intensity * 0.3
        return images + noise

    elif noise_type == "poisson":
        peak = max(1.0, 1.0 / (intensity * 0.3 + 1e-6))
        noisy = torch.poisson(
            images.clamp(0, 1) * peak
        ) / peak
        return noisy

    elif noise_type == "film_grain":
        # Luma-weighted, spatially correlated grain
        luma = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )
        # Midtone mask: grain strongest in midtones
        midtone = 1.0 - (2.0 * luma - 1.0).abs()

        # Generate spatially correlated noise
        # Downsample noise for grain texture
        scale = max(1, int(2 + intensity * 3))
        small_h = max(1, H // scale)
        small_w = max(1, W // scale)
        small_noise = torch.randn(
            B, 1, small_h, small_w,
            device=device, generator=rng,
        )
        noise = F.interpolate(
            small_noise, size=(H, W), mode='bilinear',
            align_corners=False,
        )
        # Per-channel variation
        ch_scale = torch.tensor(
            [1.0, 0.8, 1.2],
            device=device,
        ).view(1, 3, 1, 1)
        noise = noise.expand(-1, min(C, 3), -1, -1)
        noise = noise * ch_scale[:, :min(C, 3)]
        noise = noise * midtone * intensity * 0.4

        result = images.clone()
        result[:, :min(C, 3)] += noise
        return result

    elif noise_type == "sensor":
        # Heteroscedastic: read noise + shot noise
        read_std = intensity * 0.15
        shot_factor = intensity * 0.3
        signal = images.clamp(min=1e-6)
        std = torch.sqrt(
            read_std ** 2 + shot_factor * signal,
        )
        noise = torch.randn(
            B, C, H, W, device=device, generator=rng,
        ) * std
        # Slight color bias
        bias = torch.tensor(
            [0.02, -0.01, 0.015],
            device=device,
        ).view(1, 3, 1, 1) * intensity
        result = images + noise
        if C >= 3:
            result[:, :3] += bias[:, :3]
        return result

    else:  # mixed
        w_gauss = 0.5
        w_shot = 0.3
        w_grain = 0.2
        gauss = torch.randn(
            B, C, H, W, device=device, generator=rng,
        ) * intensity * 0.2 * w_gauss
        peak = max(1.0, 1.0 / (intensity * 0.2 + 1e-6))
        shot = (
            torch.poisson(images.clamp(0, 1) * peak) / peak
            - images
        ) * w_shot
        # Simple grain approx
        grain = torch.randn(
            B, 1, H // 2, W // 2,
            device=device, generator=rng,
        )
        grain = F.interpolate(
            grain, size=(H, W), mode='bilinear',
            align_corners=False,
        ).expand(-1, C, -1, -1) * intensity * 0.15 * w_grain
        return images + gauss + shot + grain


def apply_color_degradation(
    images: torch.Tensor,
    strength: float,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Desaturation, color shift, and banding."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape

    result = images.clone()

    # Desaturation
    desat = strength * 0.6
    if desat > 0 and C >= 3:
        gray = (
            0.299 * result[:, 0:1]
            + 0.587 * result[:, 1:2]
            + 0.114 * result[:, 2:3]
        )
        gray = gray.expand(-1, 3, -1, -1)
        result_rgb = torch.lerp(
            result[:, :3], gray, desat,
        )
        result = torch.cat(
            [result_rgb, result[:, 3:]], dim=1,
        ) if C > 3 else result_rgb

    # Color shift
    shift_mag = strength * 0.05
    if shift_mag > 0 and C >= 3:
        shifts = (
            torch.rand(3, device=device, generator=rng)
            * 2.0 - 1.0
        ) * shift_mag
        result[:, :3] += shifts.view(1, 3, 1, 1)

    # Banding (quantize)
    if strength > 0.3:
        levels = max(
            4, int(256 - strength * 200),
        )
        result = (
            (result * levels).round() / levels
        )

    return result


def apply_temporal_flicker(
    images: torch.Tensor,
    strength: float,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Per-frame brightness/contrast jitter."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape

    brightness = _random_smooth_schedule(
        strength * 0.1, B, device, rng, smoothing=3,
    ) - strength * 0.05
    contrast = 1.0 + _random_smooth_schedule(
        strength * 0.15, B, device, rng, smoothing=3,
    ) - strength * 0.075

    b = brightness.view(B, 1, 1, 1)
    c = contrast.view(B, 1, 1, 1)

    return images * c + b


def apply_resolution_degradation(
    images: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Downscale then upscale to lose detail."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape

    # Scale factor: 1.0 (no loss) to 0.25 (4x downscale)
    scale = max(0.1, 1.0 - strength * 0.75)
    small_h = max(4, int(H * scale))
    small_w = max(4, int(W * scale))

    down = F.interpolate(
        images, size=(small_h, small_w),
        mode='bilinear', align_corners=False,
    )
    up = F.interpolate(
        down, size=(H, W),
        mode='bilinear', align_corners=False,
    )
    return up


def apply_compression_jpeg(
    images: torch.Tensor,
    quality: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """JPEG compression simulation via PIL encode/decode."""
    if quality >= 100:
        return images
    B, C, H, W = images.shape
    device = images.device
    dtype = images.dtype

    # Move to CPU for PIL operations
    cpu_images = images.permute(0, 2, 3, 1).cpu()
    cpu_images = (cpu_images.clamp(0, 1) * 255).byte()

    results = []
    # Small per-frame quality variation for realism
    q_var = max(1, quality // 10)
    for i in range(B):
        frame = cpu_images[i].numpy()
        if C >= 3:
            pil_img = Image.fromarray(frame[:, :, :3])
        else:
            pil_img = Image.fromarray(
                frame[:, :, 0], mode='L',
            )
        # Vary quality slightly per frame
        q = quality + int(
            torch.randint(
                -q_var, q_var + 1, (1,),
            ).item()
        )
        q = max(1, min(100, q))

        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        decoded = Image.open(buf)
        arr = torch.from_numpy(
            np.array(decoded, dtype=np.float32) / 255.0,
        )
        if arr.dim() == 2:
            arr = arr.unsqueeze(-1)
        if C >= 3 and arr.shape[-1] == 1:
            arr = arr.expand(-1, -1, 3)
        results.append(arr)

    result = torch.stack(results, dim=0)  # B, H, W, C
    result = result.permute(0, 3, 1, 2)  # B, C, H, W
    # Restore alpha if present
    if C > 3:
        result = torch.cat(
            [result[:, :3], images[:, 3:].cpu()],
            dim=1,
        )
    return result.to(device=device, dtype=dtype)


def apply_compression_h264(
    images: torch.Tensor,
    quality: int,
    device: torch.device,
) -> torch.Tensor:
    """Approximate H.264 macroblock artifacts.

    Blends each 8x8 block toward its mean value. Lower
    quality = heavier blend = more visible block boundaries.
    This produces the flat-patch look of heavy H.264/H.265
    without attempting fragile DCT math.
    """
    if quality >= 100:
        return images
    _, _, H, W = images.shape

    # Blend factor: quality 1 -> 0.95, quality 99 -> 0.01
    blend = max(0.01, min(0.95, 1.0 - quality / 100.0))

    # Block size: 8x8 for normal, 16x16 for very low quality
    bs = 16 if quality < 25 else 8

    # Pad to multiple of block size
    pad_h = (bs - H % bs) % bs
    pad_w = (bs - W % bs) % bs
    if pad_h > 0 or pad_w > 0:
        padded = F.pad(
            images, (0, pad_w, 0, pad_h),
            mode='replicate',
        )
    else:
        padded = images

    _, _, pH, pW = padded.shape

    # Compute per-block mean via avg_pool then upscale
    block_means = F.avg_pool2d(
        padded, kernel_size=bs, stride=bs,
    )
    block_means = F.interpolate(
        block_means, size=(pH, pW), mode='nearest',
    )

    # Blend original toward block means
    result = torch.lerp(padded, block_means, blend)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        result = result[:, :, :H, :W]

    return result


def apply_compression(
    images: torch.Tensor,
    quality: int,
    mode: str,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Route to JPEG or H264 compression simulation."""
    if mode == "h264_sim":
        return apply_compression_h264(
            images, quality, device,
        )
    elif mode == "blockiness":
        # Stronger H264 blocking
        return apply_compression_h264(
            images, max(1, quality - 20), device,
        )
    else:  # jpeg
        return apply_compression_jpeg(
            images, quality, rng,
        )


def apply_vignette(
    images: torch.Tensor,
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    """Radial corner darkening."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape

    yy = torch.linspace(-1, 1, H, device=device)
    xx = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    r2 = gx * gx + gy * gy  # 0 at center, ~2 at corners
    # Smooth falloff
    vignette = 1.0 - strength * r2 * 0.5
    vignette = vignette.clamp(0.0, 1.0).view(1, 1, H, W)

    return images * vignette


def apply_interlacing(
    images: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Simulate interlacing combing artifacts."""
    if strength <= 0.0:
        return images
    B, C, H, W = images.shape
    if B < 2:
        return images

    result = images.clone()

    for i in range(B - 1):
        # Blend odd rows from next frame into current
        blended = torch.lerp(
            images[i, :, 1::2, :],
            images[i + 1, :, 1::2, :],
            strength,
        )
        result[i, :, 1::2, :] = blended

    return result


# -----------------------------------------------------------
# Section 3: Preset Configurations
# -----------------------------------------------------------

PRESETS: Dict[str, Dict] = {
    "mild": {
        "motion_blur_intensity": 0.2,
        "motion_blur_enabled": True,
        "defocus_intensity": 0.15,
        "defocus_enabled": True,
        "noise_intensity": 0.15,
        "noise_enabled": True,
        "compression_quality": 65,
        "compression_enabled": True,
        "temporal_flicker": 0.05,
    },
    "moderate": {
        "motion_blur_intensity": 0.5,
        "motion_blur_enabled": True,
        "defocus_intensity": 0.35,
        "defocus_enabled": True,
        "noise_intensity": 0.35,
        "noise_enabled": True,
        "compression_quality": 35,
        "compression_enabled": True,
        "temporal_flicker": 0.1,
        "chromatic_aberration": 0.15,
        "resolution_degradation": 0.2,
    },
    "severe": {
        "motion_blur_intensity": 0.8,
        "motion_blur_enabled": True,
        "defocus_intensity": 0.6,
        "defocus_enabled": True,
        "noise_intensity": 0.6,
        "noise_enabled": True,
        "compression_quality": 15,
        "compression_enabled": True,
        "temporal_flicker": 0.2,
        "chromatic_aberration": 0.3,
        "resolution_degradation": 0.5,
        "color_degradation": 0.3,
        "interlacing": 0.2,
    },
    # -- Subtle / everyday presets --
    "phone_indoor": {
        # Typical phone video in average indoor lighting.
        # Slight sensor noise from high ISO, mild lens
        # softness, standard social media compression.
        "noise_enabled": True,
        "noise_intensity": 0.08,
        "noise_type": "sensor",
        "defocus_enabled": True,
        "defocus_intensity": 0.06,
        "defocus_mode": "edge_softness",
        "compression_enabled": True,
        "compression_quality": 72,
        "compression_mode": "h264_sim",
        "vignette": 0.08,
        "lens_distortion": 0.04,
    },
    "social_media_reupload": {
        # What happens after 1-2 rounds of platform
        # re-encoding (Instagram, TikTok, Twitter).
        # Noticeable compression but no other artifacts.
        "compression_enabled": True,
        "compression_quality": 52,
        "compression_mode": "h264_sim",
        "resolution_degradation": 0.1,
        "color_degradation": 0.06,
    },
    "zoom_call": {
        # Video conferencing: adaptive bitrate compression,
        # slight temporal flicker from auto-exposure, minor
        # noise from webcam sensor in office lighting.
        "compression_enabled": True,
        "compression_quality": 42,
        "compression_mode": "blockiness",
        "noise_enabled": True,
        "noise_intensity": 0.06,
        "noise_type": "gaussian",
        "temporal_flicker": 0.04,
        "resolution_degradation": 0.15,
    },
    "dashcam": {
        # Cheap dashcam: wide-angle barrel distortion,
        # heavy compression to save storage, vibration
        # blur, some vignetting from wide lens.
        "lens_distortion": 0.15,
        "compression_enabled": True,
        "compression_quality": 38,
        "compression_mode": "h264_sim",
        "motion_blur_enabled": True,
        "motion_blur_intensity": 0.12,
        "motion_blur_angle_mode": "random_consistent",
        "vignette": 0.15,
        "noise_enabled": True,
        "noise_intensity": 0.1,
        "noise_type": "sensor",
    },
    "night_handheld": {
        # Phone/camera at night: high ISO sensor noise,
        # slight motion blur from slow shutter, reduced
        # color fidelity, soft focus.
        "noise_enabled": True,
        "noise_intensity": 0.22,
        "noise_type": "sensor",
        "motion_blur_enabled": True,
        "motion_blur_intensity": 0.15,
        "motion_blur_angle_mode": "random_consistent",
        "defocus_enabled": True,
        "defocus_intensity": 0.08,
        "defocus_mode": "uniform",
        "color_degradation": 0.15,
        "compression_enabled": True,
        "compression_quality": 65,
        "compression_mode": "h264_sim",
    },
    "old_youtube": {
        # Early YouTube era (2007-2012): heavily
        # re-encoded, low resolution, visible blocking,
        # washed out color.
        "compression_enabled": True,
        "compression_quality": 30,
        "compression_mode": "blockiness",
        "resolution_degradation": 0.4,
        "color_degradation": 0.2,
        "noise_enabled": True,
        "noise_intensity": 0.05,
        "noise_type": "gaussian",
    },
    # -- Heavy scenario presets --
    "old_vhs": {
        "noise_enabled": True,
        "noise_intensity": 0.4,
        "noise_type": "sensor",
        "interlacing": 0.7,
        "chromatic_aberration": 0.35,
        "color_degradation": 0.5,
        "resolution_degradation": 0.45,
        "temporal_flicker": 0.15,
        "compression_enabled": True,
        "compression_quality": 45,
        "compression_mode": "h264_sim",
    },
    "shaky_handheld": {
        "motion_blur_enabled": True,
        "motion_blur_intensity": 0.45,
        "motion_blur_angle_mode": "random_per_frame",
        "rolling_shutter": 0.3,
        "defocus_enabled": True,
        "defocus_intensity": 0.25,
        "defocus_mode": "breathing",
        "noise_enabled": True,
        "noise_intensity": 0.15,
        "noise_type": "sensor",
    },
    "security_cam": {
        "compression_enabled": True,
        "compression_quality": 18,
        "compression_mode": "h264_sim",
        "resolution_degradation": 0.55,
        "noise_enabled": True,
        "noise_intensity": 0.35,
        "noise_type": "sensor",
        "vignette": 0.3,
        "color_degradation": 0.4,
        "temporal_flicker": 0.08,
    },
    "livestream": {
        "compression_enabled": True,
        "compression_quality": 22,
        "compression_mode": "blockiness",
        "resolution_degradation": 0.35,
        "temporal_flicker": 0.12,
        "noise_enabled": True,
        "noise_intensity": 0.1,
        "noise_type": "gaussian",
    },
    "old_film": {
        "noise_enabled": True,
        "noise_intensity": 0.35,
        "noise_type": "film_grain",
        "vignette": 0.45,
        "color_degradation": 0.45,
        "temporal_flicker": 0.2,
        "lens_distortion": 0.1,
        "compression_enabled": True,
        "compression_quality": 55,
        "compression_mode": "jpeg",
    },
    "underwater": {
        "defocus_enabled": True,
        "defocus_intensity": 0.5,
        "defocus_mode": "breathing",
        "color_degradation": 0.55,
        "resolution_degradation": 0.35,
        "chromatic_aberration": 0.25,
        "noise_enabled": True,
        "noise_intensity": 0.2,
        "noise_type": "gaussian",
    },
}


def get_preset_params(name: str) -> Optional[Dict]:
    """Get parameter overrides for a preset."""
    return PRESETS.get(name)


# -----------------------------------------------------------
# Section 4: Pipeline Orchestrator
# -----------------------------------------------------------

def _estimate_chunk_size(
    H: int,
    W: int,
    C: int,
    total_frames: int,
) -> int:
    """Estimate safe sub-batch size based on frame dims.

    Each effect can use up to ~8x the frame memory for
    intermediates (padded convolutions, clones, grids).
    Target ~4 GB peak per chunk to leave room for models.
    """
    bytes_per_frame = H * W * C * 4  # float32
    # ~8x multiplier for worst-case intermediate tensors
    mem_per_frame = bytes_per_frame * 8
    target_bytes = 4 * 1024 ** 3  # 4 GB
    chunk = max(1, int(target_bytes / mem_per_frame))
    return min(chunk, total_frames)


def _apply_effects_to_chunk(
    x: torch.Tensor,
    params: Dict,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Apply all non-temporal effects to a BCHW chunk."""
    B, C, H, W = x.shape

    # 1. Lens distortion
    ld_str = params.get("lens_distortion", 0.0)
    if ld_str > 0:
        x = apply_lens_distortion(x, ld_str, device)

    # 2. Defocus blur
    if params.get("defocus_enabled", False):
        df_int = params.get("defocus_intensity", 0.0)
        df_mode = params.get("defocus_mode", "uniform")
        if df_int > 0:
            x = apply_defocus_blur(
                x, df_int, df_mode, B, device, rng,
            )

    # 3. Motion blur
    if params.get("motion_blur_enabled", False):
        mb_int = params.get(
            "motion_blur_intensity", 0.0,
        )
        mb_angle = params.get(
            "motion_blur_angle_mode",
            "random_consistent",
        )
        if mb_int > 0:
            x = apply_motion_blur(
                x, mb_int, mb_angle, device, rng,
            )

    # 4. Rolling shutter
    rs_str = params.get("rolling_shutter", 0.0)
    if rs_str > 0:
        x = apply_rolling_shutter(
            x, rs_str, device, rng,
        )

    # 5. Chromatic aberration
    ca_str = params.get("chromatic_aberration", 0.0)
    if ca_str > 0:
        x = apply_chromatic_aberration(
            x, ca_str, device,
        )

    # 6. Noise
    if params.get("noise_enabled", False):
        n_int = params.get("noise_intensity", 0.0)
        n_type = params.get("noise_type", "gaussian")
        if n_int > 0:
            x = apply_noise(
                x, n_int, n_type, device, rng,
            )

    # 7. Color degradation
    cd_str = params.get("color_degradation", 0.0)
    if cd_str > 0:
        x = apply_color_degradation(
            x, cd_str, device, rng,
        )

    # 8. Temporal flicker
    tf_str = params.get("temporal_flicker", 0.0)
    if tf_str > 0:
        x = apply_temporal_flicker(
            x, tf_str, device, rng,
        )

    # 9. Resolution degradation
    rd_str = params.get("resolution_degradation", 0.0)
    if rd_str > 0:
        x = apply_resolution_degradation(x, rd_str)

    # 10. Compression
    if params.get("compression_enabled", False):
        cq = params.get("compression_quality", 100)
        cm = params.get("compression_mode", "jpeg")
        if cq < 100:
            x = apply_compression(
                x, cq, cm, device, rng,
            )

    # 11. Vignette
    v_str = params.get("vignette", 0.0)
    if v_str > 0:
        x = apply_vignette(x, v_str, device)

    return x


def apply_degradation_pipeline(
    images: torch.Tensor,
    params: Dict,
    seed: int,
) -> Tuple[torch.Tensor, str]:
    """
    Apply the full degradation pipeline in correct order.

    Processes frames in sub-batches to limit GPU memory.
    Interlacing is applied as a full-batch post-pass since
    it needs adjacent frames.

    Args:
        images: (B, H, W, C) float32 tensor in [0, 1]
        params: Dict of all degradation parameters
        seed: Random seed for reproducibility

    Returns:
        Tuple of (degraded_images, degradation_map_json)
    """
    import comfy.model_management as mm

    device = mm.get_torch_device()
    B, H, W, C = images.shape

    # Apply preset overrides
    preset = params.get("degradation_preset", "custom")
    if preset != "custom":
        overrides = get_preset_params(preset)
        if overrides:
            params = {**params, **overrides}

    # Track what was applied for degradation map
    applied = {}
    for key, check in [
        ("lens_distortion", "lens_distortion"),
        ("defocus", "defocus_intensity"),
        ("motion_blur", "motion_blur_intensity"),
        ("rolling_shutter", "rolling_shutter"),
        ("chromatic_aberration", "chromatic_aberration"),
        ("noise", "noise_intensity"),
        ("color_degradation", "color_degradation"),
        ("temporal_flicker", "temporal_flicker"),
        ("resolution_degradation", "resolution_degradation"),
        ("compression", "compression_quality"),
        ("vignette", "vignette"),
        ("interlacing", "interlacing"),
    ]:
        val = params.get(check, 0.0)
        # Build applied map for enabled effects
        if key == "defocus":
            if (params.get("defocus_enabled", False)
                    and val > 0):
                applied[key] = {
                    "intensity": val,
                    "mode": params.get(
                        "defocus_mode", "uniform"
                    ),
                }
        elif key == "motion_blur":
            if (params.get(
                "motion_blur_enabled", False,
            ) and val > 0):
                applied[key] = {
                    "intensity": val,
                    "angle_mode": params.get(
                        "motion_blur_angle_mode",
                        "random_consistent",
                    ),
                }
        elif key == "noise":
            if (params.get("noise_enabled", False)
                    and val > 0):
                applied[key] = {
                    "intensity": val,
                    "type": params.get(
                        "noise_type", "gaussian"
                    ),
                }
        elif key == "compression":
            if (params.get(
                "compression_enabled", False,
            ) and val < 100):
                applied[key] = {
                    "quality": val,
                    "mode": params.get(
                        "compression_mode", "jpeg"
                    ),
                }
        elif isinstance(val, (int, float)) and val > 0:
            applied[key] = {"strength": val}

    # Determine sub-batch size
    chunk_size = _estimate_chunk_size(H, W, C, B)

    # Process in chunks
    output_chunks: List[torch.Tensor] = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk = images[start:end]

        # Per-chunk RNG for reproducibility
        rng = torch.Generator(device=device)
        rng.manual_seed(seed + start)

        # Move chunk to GPU in BCHW format
        x = chunk.to(
            device=device, dtype=torch.float32,
        ).permute(0, 3, 1, 2)

        # Apply all per-frame effects
        x = _apply_effects_to_chunk(
            x, params, device, rng,
        )

        # Clamp and move back to CPU as BHWC
        x = x.clamp_(0.0, 1.0).permute(0, 2, 3, 1)
        output_chunks.append(x.cpu())

        # Free GPU memory between chunks
        del x, chunk
        if end < B:
            mm.soft_empty_cache()

    # Concatenate all chunks
    result = torch.cat(output_chunks, dim=0)
    del output_chunks

    # Interlacing needs adjacent frames (full-batch pass)
    # Runs on CPU to avoid reloading everything to GPU
    il_str = params.get("interlacing", 0.0)
    if il_str > 0 and B >= 2:
        result = result.permute(0, 3, 1, 2)
        result = apply_interlacing(result, il_str)
        result = result.permute(0, 2, 3, 1).clamp_(
            0.0, 1.0,
        )

    # Build degradation map JSON
    deg_map = {
        "seed": seed,
        "preset": preset,
        "frame_count": B,
        "resolution": f"{W}x{H}",
        "chunk_size": chunk_size,
        "degradations": applied,
    }
    deg_json = json.dumps(deg_map, indent=2)

    return result, deg_json
