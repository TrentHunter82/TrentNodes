"""
Pose-based alignment utilities using DW Pose.

Provides shoulder-based affine alignment for subject matching using
DW Pose keypoint detection, with fallback to centroid-based alignment.
"""

import numpy as np
import torch
import torch.nn.functional as F

# COCO keypoint indices
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

# Cached DW Pose detector
_dwpose_detector = None


def _get_dwpose_detector():
    """
    Get or initialize the DW Pose detector (cached).

    Returns:
        DwposeDetector instance or None if not available
    """
    global _dwpose_detector

    if _dwpose_detector is not None:
        return _dwpose_detector

    try:
        from custom_controlnet_aux.dwpose import DwposeDetector
        import comfy.model_management as mm

        device = mm.get_torch_device()

        _dwpose_detector = DwposeDetector.from_pretrained(
            "yzd-v/DWPose",
            "yzd-v/DWPose",
            torchscript_device=device
        )
        print("[PoseAlignment] DW Pose detector initialized")
        return _dwpose_detector

    except ImportError as e:
        print(f"[PoseAlignment] DW Pose not available: {e}")
        return None
    except Exception as e:
        print(f"[PoseAlignment] Error initializing DW Pose: {e}")
        return None


def detect_shoulder_positions(image_np):
    """
    Detect shoulder positions using DW Pose.

    Args:
        image_np: (H, W, 3) numpy array, uint8 RGB

    Returns:
        tuple: (left_shoulder, right_shoulder) as numpy arrays [x, y],
               or None if no body detected
    """
    detector = _get_dwpose_detector()
    if detector is None:
        return None

    try:
        from PIL import Image

        # Convert to PIL Image (DW Pose expects this)
        pil_image = Image.fromarray(image_np)

        # Run detection (body only, no hands/face needed)
        result = detector(
            pil_image,
            include_body=True,
            include_hand=False,
            include_face=False,
            output_type="np"
        )

        # Result contains 'bodies' with keypoints
        if not hasattr(result, 'bodies') or len(result.bodies.candidate) == 0:
            return None

        # Get first detected body's keypoints
        keypoints = result.bodies.candidate
        subset = result.bodies.subset

        if len(subset) == 0:
            return None

        # Get keypoint indices for this person
        person = subset[0]
        h, w = image_np.shape[:2]

        # Extract shoulder keypoints
        left_idx = int(person[KEYPOINT_LEFT_SHOULDER])
        right_idx = int(person[KEYPOINT_RIGHT_SHOULDER])

        if left_idx < 0 or right_idx < 0:
            # Shoulders not detected
            return None

        left_shoulder = np.array([
            keypoints[left_idx][0] * w,
            keypoints[left_idx][1] * h
        ], dtype=np.float32)

        right_shoulder = np.array([
            keypoints[right_idx][0] * w,
            keypoints[right_idx][1] * h
        ], dtype=np.float32)

        # Validate - shoulders should be reasonable distance apart
        dist = np.linalg.norm(right_shoulder - left_shoulder)
        if dist < 10:  # Too close, probably bad detection
            return None

        return left_shoulder, right_shoulder

    except Exception as e:
        print(f"[PoseAlignment] Error detecting shoulders: {e}")
        return None


def compute_shoulder_affine_transform(src_shoulders, dst_shoulders):
    """
    Compute affine transform to align source shoulders to destination shoulders.

    The transform includes scale, rotation, and translation to map
    source shoulder positions to destination shoulder positions.

    Args:
        src_shoulders: (left, right) from source image as numpy arrays
        dst_shoulders: (left, right) from destination image

    Returns:
        tuple: (affine_matrix, scale, rotation_degrees)
            - affine_matrix: 2x3 numpy array
            - scale: float scale factor
            - rotation_degrees: float rotation in degrees
    """
    src_left, src_right = src_shoulders
    dst_left, dst_right = dst_shoulders

    # Compute scale from shoulder distance ratio
    src_dist = np.linalg.norm(src_right - src_left)
    dst_dist = np.linalg.norm(dst_right - dst_left)

    if src_dist < 1e-6:
        return None, 1.0, 0.0

    scale = dst_dist / src_dist

    # Compute rotation from shoulder vectors
    src_vec = src_right - src_left
    dst_vec = dst_right - dst_left

    src_angle = np.arctan2(src_vec[1], src_vec[0])
    dst_angle = np.arctan2(dst_vec[1], dst_vec[0])
    rotation = dst_angle - src_angle

    # Compute shoulder centers
    src_center = (src_left + src_right) / 2
    dst_center = (dst_left + dst_right) / 2

    # Build affine matrix
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)

    R = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ], dtype=np.float32)

    # Translation: dst_center - scale * R @ src_center
    translation = dst_center - scale * R @ src_center

    affine = np.zeros((2, 3), dtype=np.float32)
    affine[:2, :2] = scale * R
    affine[:, 2] = translation

    return affine, scale, np.degrees(rotation)


def detect_shoulders_in_masked_region(
    image: torch.Tensor,
    mask: torch.Tensor
) -> tuple:
    """
    Detect shoulder positions within a masked subject region.

    Extracts the bounding box of the mask, crops the image,
    detects shoulders, and returns positions in original image coords.

    Args:
        image: (B, H, W, C) tensor in [0, 1] range
        mask: (B, H, W) tensor where 1 = subject region

    Returns:
        tuple: (left_shoulder, right_shoulder) in original image coordinates,
               or None if no body detected
    """
    # Get bounding box of mask
    mask_np = mask[0].cpu().numpy()
    ys, xs = np.where(mask_np > 0.5)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Add padding for better detection
    pad = 50
    H, W = mask_np.shape
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(W, x_max + pad)
    y_max = min(H, y_max + pad)

    # Crop image for detection
    crop = image[0, y_min:y_max, x_min:x_max, :].cpu().numpy()
    crop_uint8 = (crop * 255).astype(np.uint8)

    # Detect shoulders in crop
    shoulders = detect_shoulder_positions(crop_uint8)

    if shoulders is None:
        return None

    left_shoulder, right_shoulder = shoulders

    # Convert back to original image coordinates
    left_shoulder[0] += x_min
    left_shoulder[1] += y_min
    right_shoulder[0] += x_min
    right_shoulder[1] += y_min

    return left_shoulder, right_shoulder


def rotate_image(
    image: torch.Tensor,
    angle_degrees: float,
    device: torch.device
) -> torch.Tensor:
    """
    Rotate image content by +angle around its center, in y-down image
    coordinates — matching compute_shoulder_affine_transform's
    "rotation = dst_angle - src_angle" convention, so rotating by that
    value turns content from the source toward the destination
    orientation.

    Args:
        image: (B, H, W, C) tensor
        angle_degrees: Rotation angle in degrees (y-down convention)
        device: torch device

    Returns:
        Rotated image tensor (B, H, W, C)
    """
    B, H, W, C = image.shape
    # grid_sample inverse-maps (theta transforms output coords to input
    # coords), so content rotates by the NEGATIVE of theta's angle.
    # Negate here so content rotates by +angle_degrees.
    angle_rad = np.radians(-angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    theta = torch.zeros(B, 2, 3, device=device, dtype=image.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    image_bchw = image.permute(0, 3, 1, 2).to(device)
    grid = F.affine_grid(theta, image_bchw.shape, align_corners=False)
    rotated = F.grid_sample(
        image_bchw, grid, mode='bilinear',
        padding_mode='border', align_corners=False
    )
    return rotated.permute(0, 2, 3, 1)


def rotate_mask(
    mask: torch.Tensor,
    angle_degrees: float,
    device: torch.device
) -> torch.Tensor:
    """
    Rotate mask content by +angle around its center (y-down image
    coordinates, same convention as rotate_image).

    Args:
        mask: (B, H, W) tensor
        angle_degrees: Rotation angle in degrees (y-down convention)
        device: torch device

    Returns:
        Rotated mask tensor (B, H, W)
    """
    B, H, W = mask.shape
    # See rotate_image: negate so content rotates by +angle_degrees.
    angle_rad = np.radians(-angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    theta = torch.zeros(B, 2, 3, device=device, dtype=mask.dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    mask_4d = mask.unsqueeze(1).to(device)
    grid = F.affine_grid(theta, mask_4d.shape, align_corners=False)
    rotated = F.grid_sample(
        mask_4d, grid, mode='bilinear',
        padding_mode='zeros', align_corners=False
    )
    return rotated.squeeze(1)
