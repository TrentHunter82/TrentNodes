"""
Shared mask operation utilities for TrentNodes.

Provides GPU-accelerated mask operations used across multiple nodes:
- Dilation (max pooling)
- Erosion (inverted max pooling)
- Gaussian blur/feathering
- Temporal smoothing (cross-frame consistency)
- Mask dimension handling
"""

import torch
import torch.nn.functional as F


def ensure_4d(mask: torch.Tensor) -> tuple:
    """
    Ensure mask is 4D (B, 1, H, W) for F.conv2d/F.max_pool2d operations.

    Args:
        mask: Tensor of shape (H, W), (B, H, W), or (B, 1, H, W)

    Returns:
        Tuple of (4D mask tensor, original number of dimensions)
    """
    orig_dim = mask.dim()

    if orig_dim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 3:
        mask = mask.unsqueeze(1)

    return mask, orig_dim


def restore_dims(mask: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """
    Restore mask to original dimensions.

    Args:
        mask: 4D tensor (B, 1, H, W)
        orig_dim: Original number of dimensions (2 or 3)

    Returns:
        Tensor restored to original shape
    """
    mask = mask.squeeze(1)  # (B, H, W)

    if orig_dim == 2:
        mask = mask.squeeze(0)  # (H, W)

    return mask


def dilate_mask(
    mask: torch.Tensor,
    radius: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Dilate a mask using max pooling.

    Args:
        mask: Tensor of shape (H, W), (B, H, W), or (B, 1, H, W)
        radius: Dilation radius in pixels
        device: torch device (unused, kept for API compatibility)

    Returns:
        Dilated mask with same shape as input
    """
    if radius <= 0:
        return mask

    mask_4d, orig_dim = ensure_4d(mask)

    kernel_size = radius * 2 + 1
    dilated = F.max_pool2d(
        mask_4d, kernel_size=kernel_size, stride=1, padding=radius
    )

    return restore_dims(dilated, orig_dim)


def erode_mask(
    mask: torch.Tensor,
    radius: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Erode a mask using min pooling (via negated max pool).

    Args:
        mask: Tensor of shape (H, W), (B, H, W), or (B, 1, H, W)
        radius: Erosion radius in pixels
        device: torch device (unused, kept for API compatibility)

    Returns:
        Eroded mask with same shape as input
    """
    if radius <= 0:
        return mask

    mask_4d, orig_dim = ensure_4d(mask)

    kernel_size = radius * 2 + 1
    # Erode = invert, dilate, invert
    inverted = 1.0 - mask_4d
    eroded = 1.0 - F.max_pool2d(
        inverted, kernel_size=kernel_size, stride=1, padding=radius
    )

    return restore_dims(eroded, orig_dim)


def create_gaussian_kernel(
    radius: int,
    device: torch.device
) -> tuple:
    """
    Create separable Gaussian kernels for blurring.

    Args:
        radius: Blur radius
        device: torch device for the kernel

    Returns:
        Tuple of (horizontal kernel, vertical kernel)
    """
    kernel_size = radius * 2 + 1
    sigma = radius / 2.0

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - radius
    gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    # Separable kernels
    kernel_h = gaussian_1d.view(1, 1, 1, kernel_size)
    kernel_v = gaussian_1d.view(1, 1, kernel_size, 1)

    return kernel_h, kernel_v


def gaussian_blur(
    mask: torch.Tensor,
    radius: int,
    device: torch.device
) -> torch.Tensor:
    """
    Apply Gaussian blur to a mask using separable convolution.

    Args:
        mask: Tensor of shape (H, W), (B, H, W), or (B, 1, H, W)
        radius: Blur radius in pixels
        device: torch device for computation

    Returns:
        Blurred mask with same shape as input
    """
    if radius <= 0:
        return mask

    mask_4d, orig_dim = ensure_4d(mask)
    kernel_h, kernel_v = create_gaussian_kernel(radius, device)

    # Pad and convolve (separable)
    padded = F.pad(mask_4d, (radius, radius, radius, radius), mode='replicate')
    blurred = F.conv2d(padded, kernel_h)
    blurred = F.conv2d(blurred, kernel_v)

    return restore_dims(blurred, orig_dim)


def feather_mask(
    mask: torch.Tensor,
    radius: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create soft feathered edges on a mask using Gaussian blur.
    Alias for gaussian_blur for semantic clarity.

    Args:
        mask: Tensor of shape (H, W), (B, H, W), or (B, 1, H, W)
        radius: Feather radius in pixels
        device: torch device for computation

    Returns:
        Feathered mask with same shape as input
    """
    return gaussian_blur(mask, radius, device)


def box_blur(
    mask: torch.Tensor,
    radius: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Apply box blur (average pooling) to a mask.

    Args:
        mask: Tensor of shape (H, W), (B, H, W), or (B, 1, H, W)
        radius: Blur radius in pixels
        device: torch device (unused, kept for API compatibility)

    Returns:
        Blurred mask with same shape as input
    """
    if radius <= 0:
        return mask

    mask_4d, orig_dim = ensure_4d(mask)

    kernel_size = radius * 2 + 1
    blurred = F.avg_pool2d(
        mask_4d, kernel_size=kernel_size, stride=1, padding=radius
    )

    return restore_dims(blurred, orig_dim)


def get_mask_bbox(mask: torch.Tensor) -> tuple:
    """
    Get bounding box of mask region.

    Args:
        mask: Tensor of shape (B, H, W) or (H, W)

    Returns:
        Tuple of (y_min, y_max, x_min, x_max)
    """
    if mask.dim() == 3:
        mask = mask[0]  # Take first batch

    # Find non-zero coordinates
    nonzero = torch.nonzero(mask > 0.5, as_tuple=True)

    if len(nonzero[0]) == 0:
        # No mask found, return full image
        return 0, mask.shape[0], 0, mask.shape[1]

    y_min = nonzero[0].min().item()
    y_max = nonzero[0].max().item()
    x_min = nonzero[1].min().item()
    x_max = nonzero[1].max().item()

    return y_min, y_max, x_min, x_max


def get_mask_centroid(mask: torch.Tensor) -> tuple:
    """
    Get center of mass of mask.

    Args:
        mask: Tensor of shape (B, H, W) or (H, W)

    Returns:
        Tuple of (cy, cx) center of mass coordinates
    """
    if mask.dim() == 3:
        mask = mask[0]  # Take first batch

    H, W = mask.shape
    mask_binary = (mask > 0.5).float()
    mask_sum = mask_binary.sum() + 1e-6

    # Create coordinate grids
    y_coords = torch.arange(H, device=mask.device, dtype=torch.float32)
    x_coords = torch.arange(W, device=mask.device, dtype=torch.float32)

    # Weighted average (center of mass)
    cy = (mask_binary * y_coords.view(-1, 1)).sum() / mask_sum
    cx = (mask_binary * x_coords.view(1, -1)).sum() / mask_sum

    return cy.item(), cx.item()


def get_mask_area(mask: torch.Tensor) -> float:
    """
    Get total area of mask (sum of pixels > 0.5).

    Args:
        mask: Tensor of shape (B, H, W) or (H, W)

    Returns:
        Total number of mask pixels
    """
    if mask.dim() == 3:
        mask = mask[0]

    return (mask > 0.5).float().sum().item()


def temporal_smooth(
    masks: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Smooth masks across the time/batch dimension with a 1D Gaussian.

    Reduces per-frame flickering from independent segmentation by
    blending each frame's mask with its temporal neighbors. Uses a
    single F.conv1d call on the reshaped tensor -- fully GPU, no
    Python loops.

    Args:
        masks: (B, H, W) mask batch where B is the time axis
        kernel_size: Temporal window size (must be odd, >=3)
        sigma: Gaussian sigma controlling blend strength

    Returns:
        Temporally smoothed (B, H, W) mask tensor
    """
    if masks.dim() != 3 or masks.shape[0] < 3:
        return masks

    # Force odd kernel size
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    B, H, W = masks.shape
    device = masks.device

    # Build 1D Gaussian kernel on device
    half = kernel_size // 2
    x = torch.arange(
        kernel_size, device=device, dtype=torch.float32
    ) - half
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size)  # (1, 1, K)

    # Reshape (B, H, W) -> (H*W, 1, B) for conv1d along time
    flat = masks.permute(1, 2, 0).reshape(H * W, 1, B)

    # Reflect-pad the time axis to avoid edge artifacts
    flat = F.pad(flat, (half, half), mode='reflect')

    # Single conv1d: Gaussian blend along time dimension
    smoothed = F.conv1d(flat, kernel)

    # Reshape back to (B, H, W)
    smoothed = smoothed.reshape(H, W, B).permute(2, 0, 1)

    return smoothed.clamp_(0.0, 1.0)


def guided_filter(
    guide: torch.Tensor,
    src: torch.Tensor,
    radius: int = 4,
    eps: float = 0.01,
) -> torch.Tensor:
    """
    Edge-aware alpha refinement using the RGB image as guide.

    Preserves fine detail (hair, fur, fabric edges) that blind
    Gaussian blur destroys. Uses the fast O(1) box-filter
    formulation of the guided filter.

    Args:
        guide: (B, H, W, C) RGB image in [0, 1] used as
            edge guide
        src: (B, H, W) input alpha matte to refine
        radius: Filter window radius (larger = smoother
            in flat regions, edges still preserved)
        eps: Regularization; smaller values preserve more
            edges, larger values smooth more

    Returns:
        (B, H, W) refined alpha matte
    """
    if radius <= 0:
        return src

    device = src.device

    # Convert guide to grayscale: (B, H, W)
    if guide.dim() == 4:
        g = guide[..., :3].mean(dim=-1).to(
            device=device, dtype=torch.float32
        )
    else:
        g = guide.to(device=device, dtype=torch.float32)

    p = src.to(dtype=torch.float32)

    # Box filter via avg_pool2d with reflect padding
    def _box(x, r):
        x4 = x.unsqueeze(1)  # (B, 1, H, W)
        k = 2 * r + 1
        return F.avg_pool2d(
            F.pad(x4, (r, r, r, r), mode='reflect'),
            kernel_size=k, stride=1,
        ).squeeze(1)

    mean_g = _box(g, radius)
    mean_p = _box(p, radius)
    mean_gp = _box(g * p, radius)
    cov_gp = mean_gp - mean_g * mean_p

    mean_gg = _box(g * g, radius)
    var_g = mean_gg - mean_g * mean_g

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g

    mean_a = _box(a, radius)
    mean_b = _box(b, radius)

    return (mean_a * g + mean_b).clamp_(0.0, 1.0)


def largest_connected_component(
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Keep only the largest connected component in a mask.

    Useful for filtering BiRefNet output to isolate the
    primary foreground subject (usually the main person)
    and discard small spurious regions.

    Args:
        mask: (H, W) or (B, H, W) binary-ish mask in
            [0, 1]. Values > 0.5 are treated as foreground.

    Returns:
        Filtered mask with same shape, only the largest
        connected component retained.
    """
    import cv2
    import numpy as np

    is_batched = mask.dim() == 3
    device = mask.device
    dtype = mask.dtype

    if not is_batched:
        mask = mask.unsqueeze(0)

    results = []
    for i in range(mask.shape[0]):
        m = (mask[i] > 0.5).cpu().numpy().astype(
            np.uint8
        )
        n_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                m, connectivity=8
            )
        )
        if n_labels <= 2:
            # 0 or 1 foreground component, keep as-is
            results.append(m.astype(np.float32))
            continue

        # Skip background (label 0), find largest fg
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = areas.argmax() + 1
        results.append(
            (labels == largest).astype(np.float32)
        )

    result = torch.tensor(
        np.stack(results),
        device=device, dtype=dtype,
    )

    if not is_batched:
        result = result.squeeze(0)

    return result


def batch_get_centroids(masks: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated batch centroid calculation for multiple masks.

    Computes center of mass for all masks in a single batch operation,
    avoiding Python loops for significant speedup.

    Args:
        masks: Tensor of shape (B, H, W) where B is batch size

    Returns:
        Tensor of shape (B, 2) with [cy, cx] for each mask
    """
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)

    B, H, W = masks.shape
    device = masks.device

    # Binary threshold
    mask_binary = (masks > 0.5).float()

    # Sum per mask for normalization
    mask_sums = mask_binary.view(B, -1).sum(dim=1, keepdim=True) + 1e-6

    # Create coordinate grids (shared across batch)
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)

    # Compute weighted sums for all masks at once
    # cy = sum(mask * y) / sum(mask)
    cy = (mask_binary * y_coords.view(1, -1, 1)).view(B, -1).sum(dim=1)
    cx = (mask_binary * x_coords.view(1, 1, -1)).view(B, -1).sum(dim=1)

    # Normalize
    cy = cy / mask_sums.squeeze(1)
    cx = cx / mask_sums.squeeze(1)

    return torch.stack([cy, cx], dim=1)
