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

Global alignment uses differentiable affine estimation (phase-correlation
seed + Adam on edge-NCC, see utils/alignment.py) with sub-pixel precision
and optional rotation / anisotropic scale.
"""

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm

from ..utils.alignment import estimate_affine, warp_image
from ..utils.image_ops import extract_edges
from ..utils.mask_ops import (
    dilate_mask, erode_mask, feather_mask,
    get_mask_bbox, get_mask_centroid, get_mask_area
)
from ..utils.birefnet_wrapper import is_birefnet_available
from ..utils.segmentation import birefnet_segment, auto_detect_subject
from ..utils.inpainting import inpaint, inpaint_transform_edges
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
                "enable_rotation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Estimate small global rotation during alignment"
                    )
                }),
                "max_rotation_deg": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "Maximum rotation to search (degrees)"
                }),
                "allow_anisotropic_scale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Separate X/Y scale - fixes aspect-ratio drift, "
                        "slight risk of absorbing content differences"
                    )
                }),
                "visualization_mode": (
                    ["heatmap", "difference", "overlay", "subject_mask",
                     "score_map"],
                    {
                        "default": "overlay",
                        "tooltip": (
                            "Visualization type for difference map output. "
                            "score_map shows the alignment residual "
                            "before/after."
                        )
                    }
                ),
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
                "max_subject_shift": ("INT", {
                    "default": 150,
                    "min": 0,
                    "max": 2048,
                    "step": 10,
                    "tooltip": (
                        "Skip subject correction when the detected move "
                        "exceeds this many pixels - guards against "
                        "segmentation mismatch flinging content across "
                        "the frame (0 = no limit)"
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
                    ["none", "lama", "void", "clone_stamp", "blur"],
                    {
                        "default": "lama",
                        "tooltip": (
                            "none: output mask for external inpaint | "
                            "lama: fast removal model (recommended) | "
                            "void: Netflix VOID diffusion (slow, "
                            "experimental on stills) | clone_stamp: "
                            "texture | blur: fast fallback"
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
                    "tooltip": (
                        "Diffusion steps per pass (void method only)"
                    )
                }),
                "inpaint_denoise": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Unused (kept for old workflow compatibility)"
                    )
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
                                             inpaint_steps, inpaint_denoise, device,
                                             extra_edge_mask=None,
                                             max_subject_shift=150):
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
            extra_edge_mask: optional (B, H, W) transform-edge gap mask to
                merge into the single inpaint pass
            max_subject_shift: skip the centroid-based correction when the
                implied move exceeds this many pixels (0 = no limit);
                guards against original/stylized segmentation mismatch

        Returns:
            Tuple of (result, info, inpaint_mask):
            - result: Final composited image
            - info: String describing what was done
            - inpaint_mask: Mask of the inpainted region (final pasted
              subject footprint excluded, so it is safe for external
              inpainting), or None if detection failed
        """
        B, H, W, C = aligned_bg.shape

        def bail(reason):
            """Skip subject correction, still honoring deferred edge fill."""
            print(f"[AlignStylizedFrame] {reason}")
            if extra_edge_mask is None:
                return aligned_bg, reason, None
            fill = torch.clamp(extra_edge_mask.to(device), 0, 1)
            result = aligned_bg.clone()
            if inpaint_method != "none":
                result = inpaint(
                    result, fill, device,
                    method=inpaint_method, steps=inpaint_steps,
                ).to(device)
            return (
                result,
                reason + " (transform-edge gaps filled)",
                fill
            )

        # Get bounding boxes for extraction
        orig_bbox = get_mask_bbox(orig_mask)  # (y_min, y_max, x_min, x_max)
        styl_bbox = get_mask_bbox(styl_mask)

        # Calculate bbox dimensions (for extraction bounds)
        orig_h = orig_bbox[1] - orig_bbox[0]
        orig_w = orig_bbox[3] - orig_bbox[2]
        styl_h = styl_bbox[1] - styl_bbox[0]
        styl_w = styl_bbox[3] - styl_bbox[2]

        if styl_h <= 0 or styl_w <= 0 or orig_h <= 0 or orig_w <= 0:
            return bail("Subject detection failed")

        # Mask agreement after global alignment, for diagnostics: low IoU
        # means original/stylized segmentation picked different things.
        ob = orig_mask[0] > 0.5
        ab = aligned_styl_mask[0] > 0.5
        union = (ob | ab).float().sum().item()
        mask_iou = (ob & ab).float().sum().item() / union if union > 0 else 0.0

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

            # Plausibility guards: without pose correspondence, a huge
            # implied move or scale means the original/stylized
            # segmentations picked DIFFERENT things (common on non-person
            # scenes) - "correcting" would fling content across the frame.
            move_dist = (
                (orig_cy - styl_cy) ** 2 + (orig_cx - styl_cx) ** 2
            ) ** 0.5
            if max_subject_shift > 0 and move_dist > max_subject_shift:
                return bail(
                    f"Subject correction skipped: centroid move "
                    f"{move_dist:.0f}px exceeds max_subject_shift="
                    f"{max_subject_shift}px (mask IoU {mask_iou:.2f}; "
                    f"original/stylized segmentation likely disagree - "
                    f"try subject_mode=disabled or provide a mask)"
                )
            if not (0.67 <= area_scale <= 1.5):
                return bail(
                    f"Subject correction skipped: implied subject scale "
                    f"{area_scale:.2f} is implausible (mask IoU "
                    f"{mask_iou:.2f}; segmentation likely disagrees)"
                )

            scale_ratio = 1.0 + (area_scale - 1.0) * conform_to_original

            # Debug output
            print(
                f"[AlignStylizedFrame] Centroid alignment (no pose): "
                f"scale={scale_ratio:.3f}, move={move_dist:.0f}px, "
                f"mask IoU={mask_iou:.2f}"
            )
            print(
                f"[AlignStylizedFrame] Centroids: "
                f"orig=({orig_cy:.1f}, {orig_cx:.1f}), "
                f"styl=({styl_cy:.1f}, {styl_cx:.1f})"
            )

            # Position offset using centroids (controlled by conform_to_original)
            dy = (orig_cy - styl_cy) * conform_to_original
            dx = (orig_cx - styl_cx) * conform_to_original

            info_parts.append(
                f"Centroid-aligned (no pose, mask IoU {mask_iou:.2f})"
            )

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

            # 6. Position so the scaled shoulder lands at the
            # conform-blended target (stylized -> original shoulder
            # center), matching the centroid branch's conform_to_original
            # semantics: scale, rotation AND translation all blend.
            target_shoulder_y = (
                styl_shoulder_center_y
                + (orig_shoulder_center_y - styl_shoulder_center_y)
                * conform_to_original
            )
            target_shoulder_x = (
                styl_shoulder_center_x
                + (orig_shoulder_center_x - styl_shoulder_center_x)
                * conform_to_original
            )
            paste_y_min = int(target_shoulder_y - scaled_shoulder_in_crop_y)
            paste_x_min = int(target_shoulder_x - scaled_shoulder_in_crop_x)
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

        # STEP 4a: Compute paste bounds first (pure arithmetic) so the
        # final subject footprint can be excluded from the inpaint mask.
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

        subject_paste = None
        mask_paste = None
        actual_h = actual_w = 0
        if valid_paste:
            subject_paste = subject_scaled[
                :, src_y_start:src_y_end, src_x_start:src_x_end, :
            ]
            mask_paste = mask_scaled[
                :, src_y_start:src_y_end, src_x_start:src_x_end
            ]
            actual_h = min(paste_h, subject_paste.shape[1])
            actual_w = min(paste_w, subject_paste.shape[2])

        # STEP 4b: Final subject footprint, eroded by the paste feather
        # radius so inpainted pixels never bleed a halo through the
        # feathered blend edge.
        footprint = torch.zeros(B, H, W, device=device)
        if valid_paste and actual_h > 0 and actual_w > 0:
            footprint[
                :,
                dst_y_start:dst_y_start + actual_h,
                dst_x_start:dst_x_start + actual_w,
            ] = (mask_paste[:, :actual_h, :actual_w] > 0.5).float()
        eroded_footprint = erode_mask(footprint, radius=5, device=device)
        if eroded_footprint.dim() == 2:
            eroded_footprint = eroded_footprint.unsqueeze(0)

        # STEP 4c: Inpaint mask = ghost (subject at its globally-aligned
        # position) + deferred transform-edge gaps, minus the area the
        # pasted subject will cover anyway.
        inpaint_radius = max(8, mask_expand)
        ghost_mask = dilate_mask(
            aligned_styl_mask,
            radius=inpaint_radius,
            device=device
        )
        if ghost_mask.dim() == 2:
            ghost_mask = ghost_mask.unsqueeze(0)
        if extra_edge_mask is not None:
            ghost_mask = torch.clamp(
                ghost_mask + extra_edge_mask.to(device), 0, 1
            )
        inpaint_mask = torch.clamp(ghost_mask - eroded_footprint, 0, 1)

        # Store mask for output before inpainting
        inpaint_mask_out = inpaint_mask.clone()

        # STEP 4d: Create inpainted background
        result = aligned_bg.clone()

        if inpaint_method == "none":
            # Ghost is NOT removed in this mode; the exported mask already
            # excludes the pasted subject, so external inpainting with it
            # is safe.
            info_parts.append(
                "Inpaint: none (ghost left in image; "
                "use inpaint_mask externally)"
            )
        else:
            result = inpaint(
                result, inpaint_mask, device,
                method=inpaint_method, steps=inpaint_steps,
            ).to(device)
            info_parts.append(
                f"Inpaint: {inpaint_method}"
                + (f" ({inpaint_steps} steps x2)"
                   if inpaint_method == "void" else "")
            )

        # STEP 5: Paste the scaled subject at target position
        if valid_paste and actual_h > 0 and actual_w > 0:
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
        mode="heatmap", subject_mask=None,
        score_map_before=None, score_map_after=None,
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

        def to_heatmap(diff):
            r = torch.clamp(diff * 3.0, 0, 1)
            g = torch.clamp((diff - 0.33) * 3.0, 0, 1)
            b = torch.clamp((diff - 0.66) * 3.0, 0, 1)
            return torch.stack([r, g, b], dim=-1)

        if mode == "score_map" and score_map_after is not None:
            # Per-pixel global-alignment residual from estimate_affine,
            # before (identity) vs after (final transform).
            sm_before = (
                score_map_before if score_map_before is not None
                else score_map_after
            ).to(device)
            sm_after = score_map_after.to(device)
            max_r = max(
                sm_before.max().item(), sm_after.max().item(), 0.001
            )
            sm_before = torch.pow((sm_before / max_r).clamp(0, 1), 0.5)
            sm_after = torch.pow((sm_after / max_r).clamp(0, 1), 0.5)
            vis_before = to_heatmap(
                sm_before.unsqueeze(0).expand(B, -1, -1)
            )
            vis_after = to_heatmap(
                sm_after.unsqueeze(0).expand(B, -1, -1)
            )

        elif mode == "subject_mask" and subject_mask is not None:
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
                     enable_rotation=True, max_rotation_deg=3.0,
                     allow_anisotropic_scale=False,
                     visualization_mode="overlay", subject_mode="birefnet",
                     subject_mask=None, conform_to_original=1.0,
                     max_subject_shift=150,
                     fill_transform_edges=True, inpaint_method="lama",
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

        if inpaint_method == "sd_inpaint":
            # Legacy alias from saved workflows: SD 1.5 backend was
            # replaced by big-lama (faster, better unprompted removal).
            print(
                "[AlignStylizedFrame] inpaint_method 'sd_inpaint' now maps "
                "to 'lama' (SD 1.5 backend removed)"
            )
            inpaint_method = "lama"

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

        size_warning = ""
        orig_aspect = W_orig / H_orig
        styl_aspect = W_styl / H_styl
        if abs(styl_aspect - orig_aspect) / orig_aspect > 0.01:
            size_warning = (
                f"WARNING: aspect ratio mismatch "
                f"({W_styl}x{H_styl} vs {W_orig}x{H_orig}) - resize "
                f"stretches anisotropically; consider "
                f"allow_anisotropic_scale\n"
            )
            print(f"[AlignStylizedFrame] {size_warning.strip()}")
        if B > 1:
            size_warning += (
                "WARNING: batch > 1 - alignment and subject geometry are "
                "estimated from frame 0 and applied to all frames\n"
            )
            print(
                "[AlignStylizedFrame] WARNING: B>1 uses frame-0 geometry "
                "for all frames; split batches for per-frame alignment"
            )

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

        # PHASE 1: Global background alignment (excluding subject if
        # detected). Differentiable affine estimation with phase-correlation
        # seeding; sub-pixel, optional rotation / anisotropic scale.
        est = estimate_affine(
            original_image, stylized_image, device,
            max_translation=float(translation_range),
            max_scale_dev=float(scale_range),
            max_rotation_deg=(
                float(max_rotation_deg) if enable_rotation else 0.0
            ),
            allow_anisotropic=allow_anisotropic_scale,
            precision=search_precision,
            bg_mask=orig_mask,
        )

        aligned_image, validity_mask = warp_image(
            stylized_image, est.params, device, padding_mode='border'
        )

        # Inpaint mask output (regions needing inpainting)
        output_inpaint_mask = torch.zeros(
            B, H_orig, W_orig, device=device
        )

        # Subject correction runs in PHASE 2; decide now so the edge-gap
        # fill can be merged into its single inpaint pass.
        needs_correction = (
            orig_mask is not None
            and styl_mask is not None
            and conform_to_original > 0
        )

        # PHASE 1.5: Fill transform edges if enabled
        edge_info = ""
        deferred_edge_mask = None
        if fill_transform_edges:
            edge_mask = (validity_mask < 0.99).float()
            edge_pixel_count = edge_mask.sum().item()

            if edge_pixel_count <= 10:
                pass
            elif needs_correction:
                # Defer to PHASE 2: one combined inpaint for edge gaps and
                # the subject ghost (mirrors inpaint_transform_edges'
                # dilate_radius=4 expansion).
                deferred_edge_mask = dilate_mask(
                    edge_mask, radius=4, device=device
                )
                if deferred_edge_mask.dim() == 2:
                    deferred_edge_mask = deferred_edge_mask.unsqueeze(0)
                edge_info = "Edge fill: merged with subject inpaint\n"
            elif inpaint_method == "none":
                output_inpaint_mask = torch.clamp(
                    output_inpaint_mask + edge_mask, 0, 1
                )
                edge_info = "Edge fill: none (mask output)\n"
            else:
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
        if needs_correction:
            # The global transform moved the subject; warping the stylized
            # subject mask by the same transform gives the ghost position
            # deterministically (re-segmenting the aligned image can
            # disagree with the actual pixels and leave ghost slivers).
            aligned_styl_mask = warp_image(
                styl_mask.unsqueeze(-1), est.params, device,
                padding_mode='zeros'
            )[0].squeeze(-1).clamp(0.0, 1.0)

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
                device,
                extra_edge_mask=deferred_edge_mask,
                max_subject_shift=max_subject_shift,
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
                visualization_mode, orig_mask,
                score_map_before=est.score_map_before,
                score_map_after=est.score_map_after,
            )
        )

        # Format info
        if allow_anisotropic_scale:
            scale_str = (
                f"{est.scale_x:.4f} x / {est.scale_y:.4f} y"
            )
        else:
            scale_str = f"{est.scale_x:.4f} ({(est.scale_x - 1) * 100:+.2f}%)"
        method_str = est.method
        if not est.converged:
            method_str += (
                " - no improving transform found, image left unwarped"
            )
        alignment_info = (
            f"Global align: {method_str}\n"
            f"Scale: {scale_str}"
            f", rotation: {est.rotation_deg:+.2f} deg\n"
            f"Translation: ({est.tx:.2f}, {est.ty:.2f}) px\n"
            f"Edge NCC: {est.ncc_identity:.4f} -> {est.ncc_final:.4f}\n"
            f"{size_warning}{edge_info}{subject_info}"
        )

        # Prepare mask outputs
        if orig_mask is not None:
            output_mask = orig_mask.to(torch.float32).cpu()
        else:
            output_mask = torch.zeros(B, H_orig, W_orig)

        return (
            aligned_image.to(torch.float32).cpu(),
            difference_map.to(torch.float32).cpu(),
            alignment_info,
            output_mask,
            output_inpaint_mask.to(torch.float32).cpu()
        )


NODE_CLASS_MAPPINGS = {
    "AlignStylizedFrame": AlignStylizedFrame
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlignStylizedFrame": "Align Stylized Frame"
}
