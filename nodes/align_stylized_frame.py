"""
ComfyUI Align Stylized Frame Node - Subject-Preserving Version

Automatically aligns a stylized image back to its original source frame.

KEY APPROACH: Preserve the subject EXACTLY as-is, then warp/stretch the
background to fill gaps. This prevents any artifacts or distortions to the
character while fixing background alignment issues.

Workflow:
1. Detect subject (auto or from provided mask)
2. Find optimal subject alignment (scale + position) to match original
3. Extract subject pixels UNTOUCHED
4. Warp background to fill gaps around the preserved subject
5. Composite: warped background + untouched subject

Uses edge-based alignment (Sobel) with multi-scale pyramid search.
"""

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm

from ..utils.image_ops import (
    extract_edges, apply_affine_transform, apply_affine_transform_with_mask
)
from ..utils.mask_ops import (
    dilate_mask, erode_mask, feather_mask,
    get_mask_bbox, get_mask_centroid, get_mask_area
)
from ..utils.birefnet_wrapper import is_birefnet_available
from ..utils.segmentation import birefnet_segment, auto_detect_subject
from ..utils.inpainting import (
    sd_inpaint, clone_stamp_inpaint, blur_inpaint, inpaint_transform_edges
)
from ..utils.pose_alignment import (
    detect_shoulders_in_masked_region, compute_shoulder_affine_transform,
    rotate_image, rotate_mask
)


class AlignStylizedFrame:
    """
    Aligns a stylized image to its original by minimizing edge differences.

    Subject-Preserving Mode:
    - Detects subject (character) automatically or uses provided mask
    - Preserves subject pixels EXACTLY (no warping/scaling of the character)
    - Warps/stretches background to fill gaps around the subject
    - Results in perfect subject fidelity with aligned background
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "stylized_image": ("IMAGE",),
            },
            "optional": {
                "scale_range": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.20,
                    "step": 0.01,
                    "tooltip": "Maximum scale deviation (0.05 = +/- 5%)"
                }),
                "translation_range": ("INT", {
                    "default": 32,
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "Maximum translation in pixels"
                }),
                "search_precision": (["fast", "balanced", "precise"], {
                    "default": "balanced",
                    "tooltip": "Search quality vs speed tradeoff"
                }),
                "visualization_mode": (["heatmap", "difference", "overlay", "subject_mask"], {
                    "default": "overlay",
                    "tooltip": "Visualization type for difference map output"
                }),
                "subject_mode": (["disabled", "auto", "birefnet", "mask"], {
                    "default": "birefnet",
                    "tooltip": (
                        "disabled: global only | auto: simple detection | "
                        "birefnet: high-quality AI | mask: use provided"
                    )
                }),
                "subject_mask": ("MASK", {
                    "tooltip": "Optional mask for subject (for mask mode)"
                }),
                "conform_to_original": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": (
                        "Conform stylized to original "
                        "(0=keep stylized, 1=match original)"
                    )
                }),
                "fill_transform_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Inpaint edges when scaling/repositioning "
                        "creates gaps"
                    )
                }),
                "inpaint_method": (
                    ["none", "sd_inpaint", "clone_stamp", "blur"],
                    {
                        "default": "sd_inpaint",
                        "tooltip": (
                            "none: output mask for external "
                            "inpaint | sd_inpaint: AI diffusion"
                            " | clone_stamp: texture | "
                            "blur: fast"
                        )
                    }
                ),
                "mask_expand": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 2,
                    "tooltip": "Pixels to expand mask before inpainting"
                }),
                "inpaint_steps": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "step": 5,
                    "tooltip": "Diffusion steps (more = better quality, slower)"
                }),
                "inpaint_denoise": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength (higher = more change)"
                }),
            }
        }

    RETURN_TYPES = (
        "IMAGE", "IMAGE", "STRING", "MASK", "MASK"
    )
    RETURN_NAMES = (
        "aligned_image", "difference_map",
        "alignment_info", "subject_mask",
        "inpaint_mask"
    )
    FUNCTION = "align_frames"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Align a stylized image to its original "
        "with optional subject-aware correction"
    )


    def _segment_subject(self, image, device, method="birefnet", reference=None):
        """
        Segment subject using specified method.
        Wrapper around utility functions with fallback logic.
        """
        if method == "birefnet":
            mask = birefnet_segment(image, device)
            if mask is not None:
                return mask
            # Fallback
            print("[AlignStylizedFrame] BiRefNet not available, using auto")
            method = "auto"

        if method == "auto":
            ref = reference if reference is not None else image
            return auto_detect_subject(image, ref, device)

        return auto_detect_subject(image, image, device)


    def compute_edge_difference_masked(self, edges1, edges2, mask=None):
        """Compute difference between edge maps, optionally weighted by mask."""
        if edges1.shape != edges2.shape:
            edges2 = F.interpolate(
                edges2.unsqueeze(1),
                size=edges1.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        diff = torch.abs(edges1 - edges2)

        if mask is not None:
            # Weight by inverse mask (background only)
            bg_mask = 1.0 - mask
            if bg_mask.shape != diff.shape:
                bg_mask = F.interpolate(
                    bg_mask.unsqueeze(1),
                    size=diff.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            weighted_diff = diff * bg_mask
            score = weighted_diff.sum() / (bg_mask.sum() + 1.0)
            return score.item()

        return torch.mean(diff).item()

    def pyramid_search(self, original_edges, stylized_image, scale_range, trans_range,
                       precision, device, bg_mask=None):
        """Multi-scale coarse-to-fine search for optimal alignment."""
        # Precision settings
        if precision == "fast":
            pyramid_levels = 2
            scale_steps = 5
            trans_steps = 5
        elif precision == "balanced":
            pyramid_levels = 3
            scale_steps = 7
            trans_steps = 7
        else:  # precise
            pyramid_levels = 4
            scale_steps = 11
            trans_steps = 9

        best_params = {'scale': 1.0, 'tx': 0.0, 'ty': 0.0}
        best_score = float('inf')

        for level in range(pyramid_levels - 1, -1, -1):
            factor = 2 ** level

            # Downsample edges
            if level > 0:
                orig_level = F.avg_pool2d(
                    original_edges.unsqueeze(1), kernel_size=factor, stride=factor
                ).squeeze(1)
                if bg_mask is not None:
                    mask_level = F.avg_pool2d(
                        bg_mask.unsqueeze(1), kernel_size=factor, stride=factor
                    ).squeeze(1)
                else:
                    mask_level = None
            else:
                orig_level = original_edges
                mask_level = bg_mask

            # Search range
            if level == pyramid_levels - 1:
                s_min, s_max = 1.0 - scale_range, 1.0 + scale_range
                t_range = trans_range // factor
            else:
                s_min = best_params['scale'] - scale_range / (2 ** (pyramid_levels - 1 - level))
                s_max = best_params['scale'] + scale_range / (2 ** (pyramid_levels - 1 - level))
                t_range = max(4, trans_range // (2 ** (pyramid_levels - 1 - level))) // factor

            scales = torch.linspace(
                s_min, s_max, scale_steps,
                device=device
            )
            tx_vals = torch.linspace(
                -t_range * factor, t_range * factor,
                trans_steps, device=device
            )
            ty_vals = torch.linspace(
                -t_range * factor, t_range * factor,
                trans_steps, device=device
            )

            # Pre-downsample stylized for coarse levels
            if level > 0:
                styl_level = F.avg_pool2d(
                    stylized_image.permute(0, 3, 1, 2),
                    kernel_size=factor,
                    stride=factor
                ).permute(0, 2, 3, 1)
            else:
                styl_level = stylized_image

            for scale in scales:
                for tx in tx_vals:
                    for ty in ty_vals:
                        # Scale translation for
                        # downsampled coords
                        level_tx = tx.item() / factor
                        level_ty = ty.item() / factor
                        transformed = (
                            apply_affine_transform(
                                styl_level,
                                scale.item(),
                                level_tx,
                                level_ty,
                                device
                            )
                        )
                        trans_edges = extract_edges(
                            transformed, device
                        )

                        score = (
                            self
                            .compute_edge_difference_masked(
                                orig_level,
                                trans_edges,
                                mask_level
                            )
                        )

                        if score < best_score:
                            best_score = score
                            best_params = {
                                'scale': scale.item(),
                                'tx': tx.item(),
                                'ty': ty.item()
                            }

        return best_params, best_score

    def fine_align_subject(self, original, stylized, orig_mask, device, search_range=15):
        """
        Fine-grained edge-based alignment within the subject region.
        Searches for the best sub-pixel offset to align eyes/head/body.

        Args:
            original: Original image (B, H, W, C)
            stylized: Stylized subject region to align (B, H, W, C)
            orig_mask: Subject mask
            device: torch device
            search_range: Max pixels to search in each direction

        Returns:
            best_dy, best_dx: Optimal offset in pixels
        """
        # Extract edges within masked region
        orig_edges = extract_edges(original, device)
        styl_edges = extract_edges(stylized, device)

        # Weight by mask (only care about subject region)
        if orig_mask.dim() == 2:
            mask = orig_mask.unsqueeze(0)
        else:
            mask = orig_mask

        orig_edges_masked = orig_edges * mask
        styl_edges_masked = styl_edges * mask

        best_score = float('inf')
        best_dy, best_dx = 0, 0

        # Coarse search first
        for dy in range(-search_range, search_range + 1, 3):
            for dx in range(-search_range, search_range + 1, 3):
                # Shift stylized edges
                shifted = torch.roll(styl_edges_masked, shifts=(dy, dx), dims=(1, 2))

                # Compute masked difference
                diff = torch.abs(orig_edges_masked - shifted) * mask
                score = diff.sum() / (mask.sum() + 1)

                if score < best_score:
                    best_score = score
                    best_dy, best_dx = dy, dx

        # Fine search around best coarse result
        coarse_dy, coarse_dx = best_dy, best_dx
        for dy in range(coarse_dy - 3, coarse_dy + 4):
            for dx in range(coarse_dx - 3, coarse_dx + 4):
                if abs(dy) > search_range or abs(dx) > search_range:
                    continue

                shifted = torch.roll(styl_edges_masked, shifts=(dy, dx), dims=(1, 2))
                diff = torch.abs(orig_edges_masked - shifted) * mask
                score = diff.sum() / (mask.sum() + 1)

                if score < best_score:
                    best_score = score
                    best_dy, best_dx = dy, dx

        return best_dy, best_dx

    def _get_shoulder_alignment(self, original_image, stylized_image, orig_mask,
                                  styl_mask, device):
        """
        Detect shoulders and compute alignment transform.

        Uses DW Pose to detect shoulder positions in both images within their
        respective subject mask regions, then computes the affine transform
        needed to align stylized shoulders to original shoulders.

        Args:
            original_image: (B, H, W, C) original image tensor
            stylized_image: (B, H, W, C) stylized image tensor
            orig_mask: (B, H, W) original subject mask
            styl_mask: (B, H, W) stylized subject mask
            device: torch device

        Returns:
            dict with 'scale', 'rotation', 'affine' or None if no body detected
        """
        # Detect shoulders in original image (within subject region)
        orig_shoulders = detect_shoulders_in_masked_region(
            original_image, orig_mask
        )

        if orig_shoulders is None:
            return None

        # Detect shoulders in stylized image (within subject region)
        styl_shoulders = detect_shoulders_in_masked_region(
            stylized_image, styl_mask
        )

        if styl_shoulders is None:
            return None

        # Compute affine transform: stylized shoulders -> original shoulders
        affine, scale, rotation = compute_shoulder_affine_transform(
            styl_shoulders, orig_shoulders
        )

        if affine is None:
            return None

        print(
            f"[AlignStylizedFrame] Shoulder alignment: scale={scale:.3f}, "
            f"rotation={rotation:.1f}deg"
        )

        return {
            'scale': scale,
            'rotation': rotation,
            'affine': affine,
            'orig_shoulders': orig_shoulders,
            'styl_shoulders': styl_shoulders
        }

    def _extract_and_scale_subject(
        self, image, mask, bbox,
        scale_ratio, H, W, pad=20
    ):
        """
        Extract subject crop from bbox and scale it.

        Args:
            image: (B, H, W, C) source image
            mask: (B, H, W) subject mask
            bbox: (y_min, y_max, x_min, x_max)
            scale_ratio: Scale factor (1.0 = no change)
            H: Image height for clamping
            W: Image width for clamping
            pad: Padding around bbox in pixels

        Returns:
            Tuple of (subject, mask, new_h, new_w,
                      y_min, x_min, crop_h, crop_w,
                      was_scaled)
        """
        y_min = max(0, bbox[0] - pad)
        y_max = min(H, bbox[1] + pad)
        x_min = max(0, bbox[2] - pad)
        x_max = min(W, bbox[3] + pad)

        subject = image[
            :, y_min:y_max, x_min:x_max, :
        ]
        mask_crop = mask[
            :, y_min:y_max, x_min:x_max
        ]

        crop_h = y_max - y_min
        crop_w = x_max - x_min

        if abs(scale_ratio - 1.0) > 0.01:
            new_h = max(1, int(crop_h * scale_ratio))
            new_w = max(1, int(crop_w * scale_ratio))

            subject = F.interpolate(
                subject.permute(0, 3, 1, 2),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)

            mask_crop = F.interpolate(
                mask_crop.unsqueeze(1),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            return (
                subject, mask_crop,
                new_h, new_w,
                y_min, x_min,
                crop_h, crop_w, True
            )

        return (
            subject, mask_crop,
            crop_h, crop_w,
            y_min, x_min,
            crop_h, crop_w, False
        )

    def preserve_subject_inpaint_background(self, stylized_before_transform, aligned_bg,
                                             original_image, orig_mask, styl_mask,
                                             aligned_styl_mask,
                                             conform_to_original,
                                             inpaint_method, mask_expand,
                                             inpaint_steps, inpaint_denoise, device):
        """
        CORRECT APPROACH with THREE masks for proper ghost elimination.

        This method:
        1. Extract subject from ORIGINAL stylized image (before any transforms)
        2. Scale subject to match original subject size
        3. Place subject at correct position (matching original)
        4. Inpaint the ALIGNED background where the ghost would appear

        Args:
            stylized_before_transform: Original stylized image BEFORE alignment
            aligned_bg: Background-aligned stylized image (for background pixels)
            original_image: The original image (for reference)
            orig_mask: Subject mask from ORIGINAL image (target position)
            styl_mask: Subject mask from STYLIZED image (extraction position)
            aligned_styl_mask: Subject mask from ALIGNED stylized (ghost position)
            conform_to_original: 0-1, how much to match original position/scale
            inpaint_method: "sd_inpaint", "clone_stamp", or "blur"
            mask_expand: pixels to expand mask before inpainting
            inpaint_steps: diffusion steps for SD inpainting
            inpaint_denoise: denoise strength for SD inpainting
            device: torch device

        Returns:
            Tuple of (result, info, inpaint_mask):
            - result: Final composited image
            - info: String describing what was done
            - inpaint_mask: Mask of inpainted region,
              or None if detection failed
        """
        B, H, W, C = aligned_bg.shape

        # Get bounding boxes for extraction
        orig_bbox = get_mask_bbox(orig_mask)  # (y_min, y_max, x_min, x_max)
        styl_bbox = get_mask_bbox(styl_mask)

        # Calculate bbox dimensions (for extraction bounds)
        orig_h = orig_bbox[1] - orig_bbox[0]
        orig_w = orig_bbox[3] - orig_bbox[2]
        styl_h = styl_bbox[1] - styl_bbox[0]
        styl_w = styl_bbox[3] - styl_bbox[2]

        if styl_h <= 0 or styl_w <= 0 or orig_h <= 0 or orig_w <= 0:
            return (
                aligned_bg,
                "Subject detection failed",
                None
            )

        # Use CENTROID for positioning (center of mass) - needed as fallback
        orig_cy, orig_cx = get_mask_centroid(orig_mask)
        styl_cy, styl_cx = get_mask_centroid(styl_mask)

        # Try shoulder-based alignment first (using DW Pose)
        shoulder_alignment = self._get_shoulder_alignment(
            original_image, stylized_before_transform,
            orig_mask, styl_mask, device
        )

        info_parts = []
        use_shoulder_alignment = False

        if shoulder_alignment is not None and conform_to_original > 0:
            # Shoulder alignment found - use scale from shoulder distance
            use_shoulder_alignment = True
            shoulder_scale = shoulder_alignment['scale']
            shoulder_rotation = shoulder_alignment['rotation']

            # Blend with conform_to_original
            # At 0: no transform; at 1: full shoulder alignment
            scale_ratio = 1.0 + (shoulder_scale - 1.0) * conform_to_original
            rotation_deg = shoulder_rotation * conform_to_original

            print(
                f"[AlignStylizedFrame] Shoulder-based alignment: "
                f"scale={scale_ratio:.3f}, rotation={rotation_deg:.1f}deg"
            )
            info_parts.append(
                f"Shoulder-aligned (scale={scale_ratio:.2f}, rot={rotation_deg:.1f}deg)"
            )

        else:
            # Fallback to centroid-based alignment
            # Use AREA-BASED scaling for more accurate size matching
            orig_area = get_mask_area(orig_mask)
            styl_area = get_mask_area(styl_mask)

            if styl_area > 0:
                area_scale = (orig_area / styl_area) ** 0.5
            else:
                area_scale = 1.0

            # Also compute bbox-based scale for comparison
            bbox_scale_h = orig_h / styl_h
            bbox_scale_w = orig_w / styl_w
            bbox_scale = (bbox_scale_h + bbox_scale_w) / 2

            # Use area-based scaling (more robust to shape differences)
            ideal_scale = area_scale
            scale_ratio = 1.0 + (ideal_scale - 1.0) * conform_to_original

            # Debug output
            print(
                f"[AlignStylizedFrame] Centroid alignment (no face detected): "
                f"scale={scale_ratio:.3f}"
            )
            print(
                f"[AlignStylizedFrame] Centroids: "
                f"orig=({orig_cy:.1f}, {orig_cx:.1f}), "
                f"styl=({styl_cy:.1f}, {styl_cx:.1f})"
            )

            # Position offset using centroids (controlled by conform_to_original)
            dy = (orig_cy - styl_cy) * conform_to_original
            dx = (orig_cx - styl_cx) * conform_to_original

            info_parts.append("Centroid-aligned (no pose)")

        # STEP 1-3: Transform subject based on alignment method
        if use_shoulder_alignment:
            # SHOULDER-BASED: Crop subject first, then scale and position
            # Key: align stylized shoulders to original shoulders

            # Get shoulder positions
            styl_shoulders = shoulder_alignment['styl_shoulders']
            orig_shoulders = shoulder_alignment['orig_shoulders']

            # Stylized shoulder center (in full image coords)
            styl_shoulder_center_y = (styl_shoulders[0][1] + styl_shoulders[1][1]) / 2
            styl_shoulder_center_x = (styl_shoulders[0][0] + styl_shoulders[1][0]) / 2

            # Original shoulder center (target position)
            orig_shoulder_center_y = (orig_shoulders[0][1] + orig_shoulders[1][1]) / 2
            orig_shoulder_center_x = (orig_shoulders[0][0] + orig_shoulders[1][0]) / 2

            # 1. Extract and scale subject
            (
                subject_scaled, mask_scaled,
                new_h, new_w,
                extract_y_min, extract_x_min,
                crop_h, crop_w, _
            ) = self._extract_and_scale_subject(
                stylized_before_transform,
                styl_mask, styl_bbox,
                scale_ratio, H, W, pad=30
            )

            # 2. Shoulder position within crop
            shoulder_in_crop_y = (
                styl_shoulder_center_y - extract_y_min
            )
            shoulder_in_crop_x = (
                styl_shoulder_center_x - extract_x_min
            )

            # 3. Shoulder position in scaled crop
            scaled_shoulder_in_crop_y = shoulder_in_crop_y * scale_ratio
            scaled_shoulder_in_crop_x = shoulder_in_crop_x * scale_ratio

            # 5. Apply rotation if significant (> 1 degree)
            if abs(rotation_deg) > 1.0:
                subject_scaled = rotate_image(subject_scaled, rotation_deg, device)
                mask_scaled = rotate_mask(mask_scaled, rotation_deg, device)

                # Update shoulder position after rotation
                # Rotation is around center of scaled crop
                center_y = new_h / 2
                center_x = new_w / 2

                # Vector from center to shoulder
                offset_y = scaled_shoulder_in_crop_y - center_y
                offset_x = scaled_shoulder_in_crop_x - center_x

                # Rotate this vector (same direction as image rotation)
                angle_rad = np.radians(rotation_deg)
                cos_r = np.cos(angle_rad)
                sin_r = np.sin(angle_rad)

                rotated_offset_x = offset_x * cos_r - offset_y * sin_r
                rotated_offset_y = offset_x * sin_r + offset_y * cos_r

                # Update shoulder position after rotation
                scaled_shoulder_in_crop_y = center_y + rotated_offset_y
                scaled_shoulder_in_crop_x = center_x + rotated_offset_x

                print(
                    f"[AlignStylizedFrame] Rotation {rotation_deg:.1f}° applied, "
                    f"shoulder moved to ({scaled_shoulder_in_crop_x:.0f},"
                    f"{scaled_shoulder_in_crop_y:.0f}) in crop"
                )

            # 6. Position so scaled shoulder aligns with original shoulder
            # paste_origin + scaled_shoulder_in_crop = orig_shoulder_center
            paste_y_min = int(orig_shoulder_center_y - scaled_shoulder_in_crop_y)
            paste_x_min = int(orig_shoulder_center_x - scaled_shoulder_in_crop_x)
            paste_y_max = paste_y_min + new_h
            paste_x_max = paste_x_min + new_w

            print(
                f"[AlignStylizedFrame] Shoulder positioning: "
                f"scale={scale_ratio:.3f}, crop={crop_w}x{crop_h} -> {new_w}x{new_h}"
            )
            print(
                f"[AlignStylizedFrame] Shoulders: "
                f"styl=({styl_shoulder_center_x:.0f},{styl_shoulder_center_y:.0f}) -> "
                f"orig=({orig_shoulder_center_x:.0f},{orig_shoulder_center_y:.0f})"
            )
            print(
                f"[AlignStylizedFrame] Paste region: "
                f"y=[{paste_y_min}:{paste_y_max}], x=[{paste_x_min}:{paste_x_max}]"
            )

        else:
            # CENTROID-BASED: Extract and scale subject
            (
                subject_scaled, mask_scaled,
                new_h, new_w,
                extract_y_min, extract_x_min,
                crop_h, crop_w, was_scaled
            ) = self._extract_and_scale_subject(
                stylized_before_transform,
                styl_mask, styl_bbox,
                scale_ratio, H, W, pad=20
            )

            if was_scaled:
                info_parts.append(
                    f"Scale: {scale_ratio:.3f}"
                )

            # Track centroid position within the crop
            centroid_in_crop_y = styl_cy - extract_y_min
            centroid_in_crop_x = styl_cx - extract_x_min

            # After scaling, centroid position scales too
            scaled_centroid_y = centroid_in_crop_y * (
                new_h / crop_h
            )
            scaled_centroid_x = centroid_in_crop_x * (
                new_w / crop_w
            )

            # Calculate where to place the subject
            target_cy = styl_cy + dy
            target_cx = styl_cx + dx

            # Paste so scaled centroid lands at target
            paste_y_min = int(target_cy - scaled_centroid_y)
            paste_x_min = int(target_cx - scaled_centroid_x)
            paste_y_max = paste_y_min + new_h
            paste_x_max = paste_x_min + new_w

            if abs(dy) > 1 or abs(dx) > 1:
                info_parts.append(f"Move: ({int(dx):+d}, {int(dy):+d})px")

        # STEP 4: Create inpainted background
        # Start with the aligned background
        result = aligned_bg.clone()

        # Inpaint using aligned_styl_mask (where ghost IS
        # in aligned_bg), NOT styl_mask (pre-alignment)
        inpaint_radius = max(8, mask_expand)
        inpaint_mask = dilate_mask(
            aligned_styl_mask,
            radius=inpaint_radius,
            device=device
        )
        if inpaint_mask.dim() == 2:
            inpaint_mask = inpaint_mask.unsqueeze(0)

        # Store mask for output before inpainting
        inpaint_mask_out = inpaint_mask.clone()

        # Use selected inpainting method
        if inpaint_method == "none":
            # Skip -- user will handle externally
            info_parts.append(
                "Inpaint: none (mask output)"
            )
        elif inpaint_method == "sd_inpaint":
            result = sd_inpaint(
                result, inpaint_mask, device,
                steps=inpaint_steps,
                denoise=inpaint_denoise
            )
            result = result.to(device)
            info_parts.append(
                f"SD inpaint ({inpaint_steps} steps)"
            )
        elif inpaint_method == "clone_stamp":
            result = clone_stamp_inpaint(
                result, inpaint_mask, device,
                iterations=20, sample_radius=10
            )
        else:
            result = blur_inpaint(
                result, inpaint_mask, device,
                iterations=5
            )

        # STEP 5: Paste the scaled subject at target position
        # Handle bounds clipping
        src_y_start = max(0, -paste_y_min)
        src_x_start = max(0, -paste_x_min)
        src_y_end = new_h - max(0, paste_y_max - H)
        src_x_end = new_w - max(0, paste_x_max - W)

        dst_y_start = max(0, paste_y_min)
        dst_x_start = max(0, paste_x_min)
        dst_y_end = min(H, paste_y_max)
        dst_x_end = min(W, paste_x_max)

        paste_h = dst_y_end - dst_y_start
        paste_w = dst_x_end - dst_x_start

        valid_paste = (
            paste_h > 0 and paste_w > 0 and
            src_y_end > src_y_start and src_x_end > src_x_start
        )
        if valid_paste:
            subject_paste = subject_scaled[
                :, src_y_start:src_y_end, src_x_start:src_x_end, :
            ]
            mask_paste = mask_scaled[
                :, src_y_start:src_y_end, src_x_start:src_x_end
            ]

            actual_h = min(paste_h, subject_paste.shape[1])
            actual_w = min(paste_w, subject_paste.shape[2])

            if actual_h > 0 and actual_w > 0:
                # Feather mask edges for smooth blending
                mask_region = mask_paste[:, :actual_h, :actual_w]
                mask_feathered = feather_mask(
                    mask_region, radius=5, device=device
                )
                if mask_feathered.dim() == 3:
                    mask_feathered = mask_feathered.unsqueeze(-1)

                # Composite
                y1, y2 = dst_y_start, dst_y_start + actual_h
                x1, x2 = dst_x_start, dst_x_start + actual_w
                bg_region = result[:, y1:y2, x1:x2, :]
                fg_region = subject_paste[:, :actual_h, :actual_w, :]
                result[:, y1:y2, x1:x2, :] = (
                    bg_region * (1 - mask_feathered) +
                    fg_region * mask_feathered
                )

        if info_parts:
            info = (
                "Subject preserved: "
                + ", ".join(info_parts)
            )
        else:
            info = "Subject preserved"
        return result, info, inpaint_mask_out

    def create_difference_visualization(
        self, original, aligned, before_stylized, device,
        mode="heatmap", subject_mask=None
    ):
        """Create a before/after difference visualization."""
        # Ensure all images are 3-channel RGB
        if original.shape[-1] > 3:
            original = original[..., :3]
        if aligned.shape[-1] > 3:
            aligned = aligned[..., :3]
        if before_stylized.shape[-1] > 3:
            before_stylized = before_stylized[..., :3]

        B, H, W, C = original.shape
        label_height = max(20, H // 30)

        if mode == "subject_mask" and subject_mask is not None:
            # Show subject mask overlay
            mask_rgb = subject_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
            red_tint = torch.tensor([1.0, 0.3, 0.3], device=device)
            green_tint = torch.tensor([0.3, 1.0, 0.3], device=device)
            vis_before = original * 0.5 + mask_rgb * 0.5 * red_tint
            vis_after = aligned * 0.5 + mask_rgb * 0.5 * green_tint

        elif mode == "heatmap":
            orig_edges = extract_edges(original, device)
            before_edges = extract_edges(before_stylized, device)
            after_edges = extract_edges(aligned, device)

            diff_before = torch.abs(orig_edges - before_edges)
            diff_after = torch.abs(orig_edges - after_edges)

            max_diff = max(diff_before.max().item(), diff_after.max().item(), 0.001)
            diff_before = torch.pow(diff_before / max_diff, 0.5)
            diff_after = torch.pow(diff_after / max_diff, 0.5)

            def to_heatmap(diff):
                r = torch.clamp(diff * 3.0, 0, 1)
                g = torch.clamp((diff - 0.33) * 3.0, 0, 1)
                b = torch.clamp((diff - 0.66) * 3.0, 0, 1)
                return torch.stack([r, g, b], dim=-1)

            vis_before = to_heatmap(diff_before)
            vis_after = to_heatmap(diff_after)

        elif mode == "difference":
            diff_before = torch.abs(original - before_stylized)
            diff_after = torch.abs(original - aligned)

            weights = (0.299, 0.587, 0.114)
            diff_before = (
                weights[0] * diff_before[..., 0] +
                weights[1] * diff_before[..., 1] +
                weights[2] * diff_before[..., 2]
            )
            diff_after = (
                weights[0] * diff_after[..., 0] +
                weights[1] * diff_after[..., 1] +
                weights[2] * diff_after[..., 2]
            )

            max_diff = max(diff_before.max().item(), diff_after.max().item(), 0.001)
            diff_before = torch.clamp(diff_before / max_diff * 2.0, 0, 1)
            diff_after = torch.clamp(diff_after / max_diff * 2.0, 0, 1)

            vis_before = diff_before.unsqueeze(-1).expand(-1, -1, -1, 3)
            vis_after = diff_after.unsqueeze(-1).expand(-1, -1, -1, 3)

        else:  # overlay
            vis_before = original * 0.5 + before_stylized * 0.5
            vis_after = original * 0.5 + aligned * 0.5

        # Build visualization
        visualization = torch.zeros(B, H + label_height, W * 2, 3,
                                    device=device, dtype=original.dtype)

        visualization[:, :label_height, :, :] = 0.15
        visualization[:, :label_height, 10:100, :] = 0.7
        visualization[:, :label_height, W + 10:W + 90, :] = 0.3
        visualization[:, label_height:, :W, :3] = vis_before[..., :3]
        visualization[:, label_height:, W:, :3] = vis_after[..., :3]
        visualization[:, label_height:, W-1:W+1, :] = 0.5

        return visualization

    @torch.no_grad()
    def align_frames(self, original_image, stylized_image, scale_range=0.05,
                     translation_range=32, search_precision="balanced",
                     visualization_mode="overlay", subject_mode="disabled",
                     subject_mask=None, conform_to_original=1.0,
                     fill_transform_edges=True, inpaint_method="sd_inpaint",
                     mask_expand=10, inpaint_steps=20, inpaint_denoise=0.9):
        """
        Main alignment function with subject-preserving mode.

        When subject_mode is enabled:
        - Subject is preserved EXACTLY (no warping/scaling)
        - Subject is repositioned to match original
        - Background is warped to fill gaps around the subject

        Args:
            conform_to_original: 0-1 slider to match original position/scale
            fill_transform_edges: Inpaint edges when scaling creates gaps
        """

        device = mm.get_torch_device()
        original_image = original_image.to(device)
        stylized_image = stylized_image.to(device)

        # Strip alpha channel - ensure 3ch RGB
        if original_image.shape[-1] > 3:
            original_image = original_image[..., :3]
        if stylized_image.shape[-1] > 3:
            stylized_image = stylized_image[..., :3]

        B, H_orig, W_orig, C = original_image.shape
        B_styl, H_styl, W_styl, C_styl = stylized_image.shape

        # Resize stylized to match original if needed
        if H_styl != H_orig or W_styl != W_orig:
            stylized_image = F.interpolate(
                stylized_image.permute(0, 3, 1, 2),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)

        stylized_before = stylized_image.clone()

        # Handle subject detection - we need SEPARATE masks for original and stylized
        # because the subject may be in different positions!
        orig_mask = None
        styl_mask = None
        subject_info = ""

        if subject_mode == "birefnet":
            print("[AlignStylizedFrame] Using BiRefNet for subject segmentation...")
            orig_mask = self._segment_subject(original_image, device, "birefnet")
            styl_mask = self._segment_subject(stylized_image, device, "birefnet")
            subject_info = "Subject: BiRefNet segmentation\n"
        elif subject_mode == "auto":
            orig_mask = auto_detect_subject(original_image, stylized_image, device)
            styl_mask = auto_detect_subject(stylized_image, original_image, device)
            subject_info = "Subject: auto-detected\n"
        elif subject_mode == "mask" and subject_mask is not None:
            orig_mask = subject_mask.to(device)
            # Handle IMAGE connected to MASK input (B,H,W,C)
            if orig_mask.dim() == 4:
                orig_mask = orig_mask[..., 0]
            if orig_mask.dim() == 2:
                orig_mask = orig_mask.unsqueeze(0)
            # Resize mask if needed
            if orig_mask.shape[-2:] != (H_orig, W_orig):
                orig_mask = F.interpolate(
                    orig_mask.unsqueeze(1),
                    size=(H_orig, W_orig),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            # For stylized, detect where subject is (different position!)
            if is_birefnet_available():
                styl_mask = self._segment_subject(stylized_image, device, "birefnet")
            else:
                styl_mask = auto_detect_subject(
                    stylized_image, original_image, device
                )
            subject_info = (
                "Subject: from provided mask (orig) + detected (stylized)\n"
            )

        # Extract edges for background alignment
        original_edges = extract_edges(original_image, device)

        # PHASE 1: Global background alignment (excluding subject if detected)
        # Use orig_mask for background weighting during alignment
        bg_mask = orig_mask if orig_mask is not None else None
        best_params, best_score = self.pyramid_search(
            original_edges, stylized_image, scale_range, translation_range,
            search_precision, device, bg_mask
        )

        # Apply global alignment with validity mask for edge detection
        aligned_image, validity_mask = apply_affine_transform_with_mask(
            stylized_image,
            best_params['scale'],
            best_params['tx'],
            best_params['ty'],
            device
        )

        # Inpaint mask output (regions needing inpainting)
        output_inpaint_mask = torch.zeros(
            B, H_orig, W_orig, device=device
        )

        # PHASE 1.5: Fill transform edges if enabled
        edge_info = ""
        if fill_transform_edges:
            edge_mask = (
                validity_mask < 0.99
            ).float()
            if edge_mask.dim() == 2:
                edge_mask = edge_mask.unsqueeze(0)
            edge_pixel_count = edge_mask.sum().item()

            if inpaint_method == "none":
                # Add edge mask to output for external use
                if edge_pixel_count > 10:
                    output_inpaint_mask = torch.clamp(
                        output_inpaint_mask
                        + edge_mask,
                        0, 1
                    )
                    edge_info = (
                        "Edge fill: none (mask output)\n"
                    )
            elif edge_pixel_count > 10:
                aligned_image = inpaint_transform_edges(
                    aligned_image,
                    validity_mask,
                    device,
                    method=inpaint_method,
                    steps=inpaint_steps,
                    denoise=inpaint_denoise
                )
                aligned_image = aligned_image.to(device)
                edge_info = "Edge fill: enabled\n"

        # PHASE 2: Subject-preserving correction
        needs_correction = (
            orig_mask is not None
            and conform_to_original > 0
        )
        if needs_correction:
            # Detect where subject is in ALIGNED image
            # (the transform moved the subject!)
            if subject_mode == "birefnet":
                aligned_styl_mask = (
                    self._segment_subject(
                        aligned_image, device, "birefnet"
                    )
                )
            else:
                aligned_styl_mask = auto_detect_subject(
                    aligned_image, original_image, device
                )

            # THREE masks for ghost elimination:
            # orig_mask: target position
            # styl_mask: extraction from stylized_before
            # aligned_styl_mask: where ghost appears
            (
                aligned_image,
                correction_info,
                correction_mask
            ) = self.preserve_subject_inpaint_background(
                stylized_before,
                aligned_image,
                original_image,
                orig_mask,
                styl_mask,
                aligned_styl_mask,
                conform_to_original,
                inpaint_method,
                mask_expand,
                inpaint_steps,
                inpaint_denoise,
                device
            )

            subject_info += correction_info + "\n"
            if correction_mask is not None:
                output_inpaint_mask = torch.clamp(
                    output_inpaint_mask
                    + correction_mask,
                    0, 1
                )

        # Create visualization
        difference_map = (
            self.create_difference_visualization(
                original_image, aligned_image,
                stylized_before, device,
                visualization_mode, orig_mask
            )
        )

        # Format info
        scale_pct = (best_params['scale'] - 1.0) * 100
        alignment_info = (
            f"Background scale: "
            f"{best_params['scale']:.4f} "
            f"({scale_pct:+.2f}%)\n"
            f"Background translation: "
            f"({best_params['tx']:.1f}, "
            f"{best_params['ty']:.1f}) px\n"
            f"Alignment score: {best_score:.6f}\n"
            f"{edge_info}{subject_info}"
        )

        # Prepare mask outputs
        if orig_mask is not None:
            output_mask = orig_mask.cpu()
        else:
            output_mask = torch.zeros(B, H_orig, W_orig)

        return (
            aligned_image.cpu(),
            difference_map.cpu(),
            alignment_info,
            output_mask,
            output_inpaint_mask.cpu()
        )


NODE_CLASS_MAPPINGS = {
    "AlignStylizedFrame": AlignStylizedFrame
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlignStylizedFrame": "Align Stylized Frame"
}
