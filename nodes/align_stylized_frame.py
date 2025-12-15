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

import torch
import torch.nn.functional as F
import numpy as np
import comfy.model_management as mm
import folder_paths
import os
import sys

# BiRefNet (BEN2) model cache
_birefnet_model = None
_birefnet_device = None
_birefnet_available = None  # None = not checked yet


def load_birefnet(device):
    """
    Load BiRefNet (BEN2) model for high-quality subject segmentation.
    Model is cached for reuse across calls.
    """
    global _birefnet_model, _birefnet_device, _birefnet_available

    # Return cached model if available and on correct device
    if _birefnet_model is not None and _birefnet_device == device:
        return _birefnet_model

    # Try to import from ComfyUI-Easy-Use
    try:
        # Add ComfyUI-Easy-Use to path if needed
        easy_use_path = os.path.join(folder_paths.base_path,
                                      "custom_nodes", "ComfyUI-Easy-Use", "py")
        if easy_use_path not in sys.path:
            sys.path.insert(0, easy_use_path)

        from modules.ben.model import BEN_Base

        # Model path
        model_dir = os.path.join(folder_paths.models_dir, "rembg")
        model_path = os.path.join(model_dir, "BEN2_Base.pth")

        if not os.path.exists(model_path):
            # Download model
            from torch.hub import download_url_to_file
            os.makedirs(model_dir, exist_ok=True)
            url = "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.pth"
            print(f"[AlignStylizedFrame] Downloading BiRefNet model (~500MB)...")
            download_url_to_file(url, model_path)
            print(f"[AlignStylizedFrame] BiRefNet model downloaded.")

        model = BEN_Base().to(device).eval()
        model.loadcheckpoints(model_path)

        _birefnet_model = model
        _birefnet_device = device
        _birefnet_available = True
        print("[AlignStylizedFrame] BiRefNet model loaded successfully.")
        return model

    except Exception as e:
        print(f"[AlignStylizedFrame] Warning: Could not load BiRefNet: {e}")
        print("[AlignStylizedFrame] Falling back to auto-detection mode.")
        _birefnet_available = False
        return None


def is_birefnet_available():
    """Check if BiRefNet can be loaded (without actually loading it)."""
    global _birefnet_available

    if _birefnet_available is not None:
        return _birefnet_available

    # Check if ComfyUI-Easy-Use is installed
    easy_use_path = os.path.join(folder_paths.base_path,
                                  "custom_nodes", "ComfyUI-Easy-Use", "py")
    ben_model_path = os.path.join(easy_use_path, "modules", "ben", "model.py")

    if os.path.exists(ben_model_path):
        _birefnet_available = True
    else:
        _birefnet_available = False
        print("[AlignStylizedFrame] Note: BiRefNet requires ComfyUI-Easy-Use to be installed.")

    return _birefnet_available


# SD Inpainting model cache
_inpaint_model = None
_inpaint_model_name = None


def load_inpaint_model(checkpoint_name, device):
    """
    Load SD 1.5 inpainting model for clean plate generation.
    Model is cached for reuse across calls.
    """
    global _inpaint_model, _inpaint_model_name
    import comfy.sd

    # Return cached model if available
    if _inpaint_model is not None and _inpaint_model_name == checkpoint_name:
        return _inpaint_model

    # Find or download checkpoint
    ckpt_path = os.path.join(folder_paths.models_dir, "checkpoints", checkpoint_name)

    if not os.path.exists(ckpt_path):
        # Try alternate location
        alt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        if alt_path and os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            # Download the model
            print(f"[AlignStylizedFrame] Downloading SD 1.5 inpainting model (~4GB)...")
            print(f"[AlignStylizedFrame] This is a one-time download.")
            from torch.hub import download_url_to_file
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            url = "https://huggingface.co/webui/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.safetensors"
            download_url_to_file(url, ckpt_path)
            print(f"[AlignStylizedFrame] SD inpainting model downloaded.")

    # Load the model
    print(f"[AlignStylizedFrame] Loading SD inpainting model...")
    model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    _inpaint_model = (model, clip, vae)
    _inpaint_model_name = checkpoint_name
    print(f"[AlignStylizedFrame] SD inpainting model loaded.")

    return model, clip, vae


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
                    "tooltip": "disabled: global only | auto: simple detection | birefnet: high-quality AI segmentation | mask: use provided"
                }),
                "subject_mask": ("MASK", {
                    "tooltip": "Optional mask for subject (required if subject_mode='mask')"
                }),
                "subject_scale_correction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blend factor for subject alignment (0=keep stylized position, 1=match original)"
                }),
                "subject_position_correction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blend factor for subject position (0=keep stylized position, 1=match original)"
                }),
                "inpaint_method": (["sd_inpaint", "clone_stamp", "blur"], {
                    "default": "sd_inpaint",
                    "tooltip": "sd_inpaint: AI diffusion (best quality) | clone_stamp: texture sampling | blur: fast blur fill"
                }),
                "mask_expand": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 2,
                    "tooltip": "Pixels to expand mask before inpainting (larger = safer margin)"
                }),
                "inpaint_steps": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "step": 5,
                    "tooltip": "Diffusion steps for SD inpainting (more = better quality, slower)"
                }),
                "inpaint_denoise": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for SD inpainting (higher = more change)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("aligned_image", "difference_map", "alignment_info", "subject_mask")
    FUNCTION = "align_frames"
    CATEGORY = "Trent/Image"
    DESCRIPTION = "Align a stylized image to its original with optional subject-aware correction"

    def extract_edges(self, image, device):
        """Extract edges using Sobel filters."""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to grayscale
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)

        gray_4d = gray.unsqueeze(1)
        edges_x = F.conv2d(gray_4d, sobel_x, padding=1)
        edges_y = F.conv2d(gray_4d, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

        return edges.squeeze(1)

    def apply_transform(self, image, scale, tx, ty, device):
        """Apply scale and translation transform to image."""
        B, H, W, C = image.shape
        inv_scale = 1.0 / scale

        theta = torch.zeros(B, 2, 3, device=device, dtype=image.dtype)
        theta[:, 0, 0] = inv_scale
        theta[:, 1, 1] = inv_scale
        theta[:, 0, 2] = -tx / (W / 2) * inv_scale
        theta[:, 1, 2] = -ty / (H / 2) * inv_scale

        image_bchw = image.permute(0, 3, 1, 2)
        grid = F.affine_grid(theta, image_bchw.shape, align_corners=False)
        transformed = F.grid_sample(image_bchw, grid, mode='bilinear',
                                    padding_mode='border', align_corners=False)

        return transformed.permute(0, 2, 3, 1)

    def dilate_mask(self, mask, radius, device):
        """Dilate a mask using max pooling."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        kernel_size = radius * 2 + 1
        dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)

        return dilated.squeeze(1)

    def erode_mask(self, mask, radius, device):
        """Erode a mask using min pooling (via negated max pool)."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        kernel_size = radius * 2 + 1
        # Erode = invert, dilate, invert
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=radius)

        return eroded.squeeze(1)

    def auto_detect_subject(self, original, stylized, device):
        """
        Automatically detect subject using edge density, center weighting, and change detection.
        Returns a soft mask where higher values = more likely to be subject.
        """
        B, H, W, C = original.shape

        # 1. Edge density - subjects typically have more detail
        edges = self.extract_edges(original, device)
        edges_norm = edges / (edges.max() + 0.001)

        # 2. Center weighting - subjects are usually centered
        cy, cx = H // 2, W // 2
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_dist = torch.abs(y_coords - cy) / cy
        x_dist = torch.abs(x_coords - cx) / cx
        center_weight = 1.0 - torch.clamp((y_dist.unsqueeze(1) + x_dist.unsqueeze(0)) * 0.4, 0, 0.8)
        center_weight = center_weight.unsqueeze(0).expand(B, -1, -1)

        # 3. Change detection - areas that changed most are likely subject
        diff = torch.mean(torch.abs(original - stylized), dim=-1)
        diff_norm = diff / (diff.max() + 0.001)

        # Combine signals
        saliency = edges_norm * 0.35 + diff_norm * 0.45 + center_weight * 0.20

        # Adaptive threshold
        threshold_val = saliency.mean() + 0.3 * saliency.std()
        mask = (saliency > threshold_val).float()

        # Morphological cleanup: erode to remove noise, then dilate to fill gaps
        mask = self.erode_mask(mask, radius=3, device=device)
        mask = self.dilate_mask(mask, radius=8, device=device)

        # Smooth the mask edges
        mask_4d = mask.unsqueeze(1)
        blur_kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0
        mask_smooth = F.conv2d(mask_4d, blur_kernel, padding=2)
        mask = mask_smooth.squeeze(1)

        return torch.clamp(mask, 0, 1)

    def birefnet_segment(self, image, device):
        """
        Use BiRefNet (BEN2) for high-quality subject segmentation.

        Args:
            image: (B, H, W, C) tensor in [0, 1] range
            device: torch device

        Returns:
            mask: (B, H, W) tensor, 1 = subject, 0 = background
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        model = load_birefnet(device)
        if model is None:
            # Fallback to auto-detection if BiRefNet not available
            print("[AlignStylizedFrame] BiRefNet not available, using auto-detection")
            # Create a dummy comparison image for auto_detect
            return self.auto_detect_subject(image, image, device)

        B, H, W, C = image.shape
        masks = []

        for i in range(B):
            # Convert tensor to PIL Image
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Get mask from BiRefNet
            with torch.no_grad():
                mask_pil, _ = model.inference(pil_img, refine_foreground=False)

            # Convert mask to tensor - mask_pil is a PIL Image in grayscale
            mask_tensor = TF.to_tensor(mask_pil)[0]  # Get single channel [0-1]
            mask_tensor = mask_tensor.to(device)

            # Resize to match input image size if needed
            if mask_tensor.shape[0] != H or mask_tensor.shape[1] != W:
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)

            masks.append(mask_tensor)

        return torch.stack(masks)

    def get_mask_bbox(self, mask):
        """Get bounding box of mask region. Returns (y_min, y_max, x_min, x_max)."""
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

    def get_mask_centroid(self, mask):
        """
        Get center of mass of mask (more accurate than bbox center).

        Args:
            mask: (B, H, W) or (H, W) mask tensor

        Returns:
            (cy, cx): Center of mass coordinates
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

    def get_mask_area(self, mask):
        """
        Get total area of mask (sum of pixels > 0.5).

        Args:
            mask: (B, H, W) or (H, W) mask tensor

        Returns:
            area: Total number of mask pixels
        """
        if mask.dim() == 3:
            mask = mask[0]

        return (mask > 0.5).float().sum().item()

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

            scales = torch.linspace(s_min, s_max, scale_steps, device=device)
            tx_vals = torch.linspace(-t_range * factor, t_range * factor, trans_steps, device=device)
            ty_vals = torch.linspace(-t_range * factor, t_range * factor, trans_steps, device=device)

            for scale in scales:
                for tx in tx_vals:
                    for ty in ty_vals:
                        transformed = self.apply_transform(
                            stylized_image, scale.item(), tx.item(), ty.item(), device
                        )
                        trans_edges = self.extract_edges(transformed, device)

                        if level > 0:
                            trans_edges = F.avg_pool2d(
                                trans_edges.unsqueeze(1), kernel_size=factor, stride=factor
                            ).squeeze(1)

                        score = self.compute_edge_difference_masked(orig_level, trans_edges, mask_level)

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
        orig_edges = self.extract_edges(original, device)
        styl_edges = self.extract_edges(stylized, device)

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

    def preserve_subject_inpaint_background(self, stylized_before_transform, aligned_bg,
                                             original_image, orig_mask, styl_mask,
                                             aligned_styl_mask,
                                             scale_correction, position_correction,
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
            stylized_before_transform: Original stylized image BEFORE any alignment
            aligned_bg: Background-aligned stylized image (for background pixels)
            original_image: The original image (for reference)
            orig_mask: Subject mask from ORIGINAL image (target position)
            styl_mask: Subject mask from STYLIZED image (extraction position)
            aligned_styl_mask: Subject mask from ALIGNED stylized (inpaint position - where ghost is!)
            scale_correction: 0-1, how much to scale subject to match original
            position_correction: 0-1, how much to match original subject position
            inpaint_method: "sd_inpaint", "clone_stamp", or "blur"
            mask_expand: pixels to expand mask before inpainting
            inpaint_steps: diffusion steps for SD inpainting
            inpaint_denoise: denoise strength for SD inpainting
            device: torch device

        Returns:
            result: Final composited image
            info: String describing what was done
        """
        B, H, W, C = aligned_bg.shape

        # Get bounding boxes for extraction
        orig_bbox = self.get_mask_bbox(orig_mask)  # (y_min, y_max, x_min, x_max)
        styl_bbox = self.get_mask_bbox(styl_mask)

        # Calculate bbox dimensions (for extraction bounds)
        orig_h = orig_bbox[1] - orig_bbox[0]
        orig_w = orig_bbox[3] - orig_bbox[2]
        styl_h = styl_bbox[1] - styl_bbox[0]
        styl_w = styl_bbox[3] - styl_bbox[2]

        if styl_h <= 0 or styl_w <= 0 or orig_h <= 0 or orig_w <= 0:
            return aligned_bg, "Subject detection failed"

        # Use CENTROID for more accurate positioning (center of mass)
        orig_cy, orig_cx = self.get_mask_centroid(orig_mask)
        styl_cy, styl_cx = self.get_mask_centroid(styl_mask)

        # Use AREA-BASED scaling for more accurate size matching
        # (sqrt because area scales quadratically with linear dimensions)
        orig_area = self.get_mask_area(orig_mask)
        styl_area = self.get_mask_area(styl_mask)

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
        scale_ratio = 1.0 + (ideal_scale - 1.0) * scale_correction

        # Debug output
        print(f"[AlignStylizedFrame] Scaling: area_scale={area_scale:.3f}, bbox_scale={bbox_scale:.3f}, final={scale_ratio:.3f}")
        print(f"[AlignStylizedFrame] Centroids: orig=({orig_cy:.1f}, {orig_cx:.1f}), styl=({styl_cy:.1f}, {styl_cx:.1f})")

        # Position offset using centroids
        dy = (orig_cy - styl_cy) * position_correction
        dx = (orig_cx - styl_cx) * position_correction

        info_parts = []

        # STEP 1: Extract subject from ORIGINAL stylized (before any transforms!)
        # Add padding around subject for clean extraction
        pad = 20
        extract_y_min = max(0, styl_bbox[0] - pad)
        extract_y_max = min(H, styl_bbox[1] + pad)
        extract_x_min = max(0, styl_bbox[2] - pad)
        extract_x_max = min(W, styl_bbox[3] + pad)

        subject_crop = stylized_before_transform[:, extract_y_min:extract_y_max,
                                                   extract_x_min:extract_x_max, :]
        mask_crop = styl_mask[:, extract_y_min:extract_y_max, extract_x_min:extract_x_max]

        crop_h = extract_y_max - extract_y_min
        crop_w = extract_x_max - extract_x_min

        # STEP 2: Scale subject if needed
        if abs(scale_ratio - 1.0) > 0.01:
            new_h = max(1, int(crop_h * scale_ratio))
            new_w = max(1, int(crop_w * scale_ratio))

            subject_scaled = F.interpolate(
                subject_crop.permute(0, 3, 1, 2),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)

            mask_scaled = F.interpolate(
                mask_crop.unsqueeze(1),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            info_parts.append(f"Scale: {scale_ratio:.3f}")
        else:
            subject_scaled = subject_crop
            mask_scaled = mask_crop
            new_h, new_w = crop_h, crop_w

        # STEP 3: Calculate where to place the subject
        # Use centroids for precise positioning
        # Target center: blend between stylized centroid and original centroid
        target_cy = styl_cy + dy  # dy = (orig_cy - styl_cy) * position_correction
        target_cx = styl_cx + dx  # dx = (orig_cx - styl_cx) * position_correction

        # When position_correction=1.0, target = orig centroid
        # When position_correction=0.0, target = styl centroid (no movement)

        # Paste coordinates (center the scaled subject at target position)
        paste_y_min = int(target_cy - new_h / 2)
        paste_x_min = int(target_cx - new_w / 2)
        paste_y_max = paste_y_min + new_h
        paste_x_max = paste_x_min + new_w

        if abs(dy) > 1 or abs(dx) > 1:
            info_parts.append(f"Move: ({int(dx):+d}, {int(dy):+d})px")

        # STEP 4: Create inpainted background
        # Start with the aligned background
        result = aligned_bg.clone()

        # CRITICAL FIX: Inpaint using aligned_styl_mask (where the ghost IS in aligned_bg)
        # NOT styl_mask (which shows where subject was BEFORE alignment transform)
        inpaint_radius = max(8, mask_expand)
        inpaint_mask = self.dilate_mask(aligned_styl_mask, radius=inpaint_radius, device=device)
        if inpaint_mask.dim() == 2:
            inpaint_mask = inpaint_mask.unsqueeze(0)

        # Use selected inpainting method
        if inpaint_method == "sd_inpaint":
            # Use Stable Diffusion 1.5 inpainting for best quality
            result = self.sd_inpaint_background(
                result, inpaint_mask, device,
                steps=inpaint_steps,
                denoise=inpaint_denoise
            )
            # Ensure result is on the correct device for subsequent operations
            result = result.to(device)
            info_parts.append(f"SD inpaint ({inpaint_steps} steps)")
        elif inpaint_method == "clone_stamp":
            result = self.clone_stamp_inpaint(result, inpaint_mask, device,
                                               iterations=20, sample_radius=10)
        else:
            result = self.simple_inpaint(result, inpaint_mask, device, iterations=5)

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

        if paste_h > 0 and paste_w > 0 and src_y_end > src_y_start and src_x_end > src_x_start:
            subject_paste = subject_scaled[:, src_y_start:src_y_end, src_x_start:src_x_end, :]
            mask_paste = mask_scaled[:, src_y_start:src_y_end, src_x_start:src_x_end]

            actual_h = min(paste_h, subject_paste.shape[1])
            actual_w = min(paste_w, subject_paste.shape[2])

            if actual_h > 0 and actual_w > 0:
                # Feather mask edges for smooth blending
                mask_feathered = self.feather_mask(mask_paste[:, :actual_h, :actual_w],
                                                   radius=5, device=device)
                if mask_feathered.dim() == 3:
                    mask_feathered = mask_feathered.unsqueeze(-1)

                # Composite
                result[:, dst_y_start:dst_y_start+actual_h,
                       dst_x_start:dst_x_start+actual_w, :] = (
                    result[:, dst_y_start:dst_y_start+actual_h,
                           dst_x_start:dst_x_start+actual_w, :] * (1 - mask_feathered) +
                    subject_paste[:, :actual_h, :actual_w, :] * mask_feathered
                )

        info = "Subject preserved" + (": " + ", ".join(info_parts) if info_parts else "")
        return result, info

    def clone_stamp_inpaint(self, image, mask, device, iterations=25, sample_radius=12):
        """
        Clone-stamp style inpainting that samples ONLY from background pixels.
        Fills masked region by iteratively extending edges inward.

        CRITICAL: We first blank out the masked region so we never sample from
        the subject - only from true background pixels.

        Args:
            image: (B, H, W, C) image tensor
            mask: (B, H, W) or (H, W) mask where 1 = inpaint region
            device: torch device
            iterations: Number of inward-fill passes
            sample_radius: How far to look for source pixels
        """
        B, H, W, C = image.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # CRITICAL FIX: First, blank out the masked region entirely
        # This ensures we NEVER sample from subject pixels
        original_mask = mask.clone()
        mask_3d = mask.unsqueeze(-1).expand(-1, -1, -1, C)

        # Start with image where masked region is zeroed out
        # (we'll fill it in from the edges)
        result = image.clone()
        result = result * (1 - mask_3d)  # Zero out the subject region

        # Track which pixels are "valid" (have been filled or were never masked)
        # Initially, only unmasked pixels are valid sources
        valid_source = (1 - original_mask).clone()

        # Create sampling kernel for weighted neighbor averaging
        kernel_size = sample_radius * 2 + 1
        y_coords = torch.arange(kernel_size, device=device) - sample_radius
        x_coords = torch.arange(kernel_size, device=device) - sample_radius
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Distance-based weights (closer pixels have more influence)
        dist = torch.sqrt(yy.float()**2 + xx.float()**2)
        weights = torch.exp(-dist / (sample_radius * 0.4))
        weights[sample_radius, sample_radius] = 0  # Don't sample from self
        weights = weights / weights.sum()
        weights = weights.view(1, 1, kernel_size, kernel_size)

        # Current region still needing to be filled
        remaining_mask = original_mask.clone()

        for _ in range(iterations):
            # Find edge pixels: masked pixels adjacent to valid source pixels
            # Dilate valid_source to find what can be filled this iteration
            valid_dilated = self.dilate_mask(valid_source, radius=1, device=device)
            if valid_dilated.dim() == 2:
                valid_dilated = valid_dilated.unsqueeze(0)

            # Edge = pixels that are still masked but adjacent to valid pixels
            edge_mask = remaining_mask * valid_dilated
            edge_mask = torch.clamp(edge_mask, 0, 1)

            # If no more edges to fill, we're done
            if edge_mask.sum() < 1:
                break

            # For edge pixels, sample from nearby VALID source pixels only
            pad = sample_radius
            result_padded = F.pad(result.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='replicate')
            valid_padded = F.pad(valid_source.unsqueeze(1), (pad, pad, pad, pad), mode='constant', value=0)

            # Compute weighted sum of valid neighbors for each channel
            filled_values = torch.zeros_like(result)
            total_weight = torch.zeros(B, H, W, device=device)

            for c in range(C):
                channel = result_padded[:, c:c+1, :, :]
                # Only count valid source pixels in the weighted average
                weighted_vals = F.conv2d(channel * valid_padded, weights, padding=0)
                weight_sum = F.conv2d(valid_padded, weights, padding=0)

                weight_sum_safe = weight_sum.clamp(min=1e-6)
                filled_values[:, :, :, c] = (weighted_vals / weight_sum_safe).squeeze(1)

                if c == 0:
                    total_weight = weight_sum.squeeze(1)

            # Only update edge pixels that have valid neighbors to sample from
            has_valid_neighbors = (total_weight > 0.01).float()
            update_mask = edge_mask * has_valid_neighbors
            update_mask_3d = update_mask.unsqueeze(-1).expand(-1, -1, -1, C)

            # Fill in the edge pixels
            result = result * (1 - update_mask_3d) + filled_values * update_mask_3d

            # Mark newly filled pixels as valid sources for next iteration
            valid_source = torch.clamp(valid_source + update_mask, 0, 1)

            # Remove filled pixels from remaining mask
            remaining_mask = remaining_mask * (1 - update_mask)

        return result

    def simple_inpaint(self, image, mask, device, iterations=3):
        """
        Simple inpainting using iterative blurring from edges.
        Fills masked regions with content from surrounding areas.
        (Legacy method - use clone_stamp_inpaint for better results)
        """
        B, H, W, C = image.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask_3d = mask.unsqueeze(-1).expand(-1, -1, -1, C)

        result = image.clone()

        # Iteratively blur and fill from edges
        for _ in range(iterations):
            # Heavy blur
            blurred = F.avg_pool2d(
                result.permute(0, 3, 1, 2),
                kernel_size=15, stride=1, padding=7
            ).permute(0, 2, 3, 1)

            # Only replace masked areas
            result = result * (1 - mask_3d) + blurred * mask_3d

            # Erode mask slightly each iteration (fill from edges)
            mask = self.erode_mask(mask, radius=3, device=device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask_3d = mask.unsqueeze(-1).expand(-1, -1, -1, C)

        return result

    def sd_inpaint_background(self, image, mask, device,
                               checkpoint="sd-v1-5-inpainting.safetensors",
                               steps=20, denoise=0.9, cfg=7.5, seed=42):
        """
        Use Stable Diffusion 1.5 inpainting model to create clean plate background.

        This provides much better results than clone-stamp or blur because the
        diffusion model understands image context and can generate realistic
        background content.

        Args:
            image: (B, H, W, C) aligned stylized image in [0, 1] range
            mask: (B, H, W) inpaint mask (1 = area to inpaint)
            device: torch device
            checkpoint: inpainting model checkpoint name
            steps: diffusion steps (more = better quality, slower)
            denoise: denoise strength (0.5-1.0, higher = more change)
            cfg: classifier-free guidance scale
            seed: random seed for reproducibility

        Returns:
            Clean background image with subject area filled by diffusion
        """
        import comfy.sample

        B, H, W, C = image.shape
        print(f"[AlignStylizedFrame] Running SD inpainting ({steps} steps, denoise={denoise})...")

        # 1. Load inpainting model (cached)
        model, clip, vae = load_inpaint_model(checkpoint, device)

        # 2. Encode empty prompts (let model fill based purely on context)
        tokens = clip.tokenize("")
        positive = clip.encode_from_tokens_scheduled(tokens)
        negative = clip.encode_from_tokens_scheduled(tokens)

        # 3. Prepare image for VAE encoding
        # Ensure image is properly sized for VAE (divisible by 8)
        downscale = getattr(vae, 'downscale_ratio', 8)
        new_h = (H // downscale) * downscale
        new_w = (W // downscale) * downscale

        # Crop if needed
        if H != new_h or W != new_w:
            h_offset = (H - new_h) // 2
            w_offset = (W - new_w) // 2
            pixels = image[:, h_offset:h_offset + new_h, w_offset:w_offset + new_w, :].clone()
            inpaint_mask = mask[:, h_offset:h_offset + new_h, w_offset:w_offset + new_w].clone()
        else:
            pixels = image.clone()
            inpaint_mask = mask.clone()
            h_offset, w_offset = 0, 0

        # 4. Prepare mask
        if inpaint_mask.dim() == 2:
            inpaint_mask = inpaint_mask.unsqueeze(0)

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

        # 5. Apply mask to image (set inpaint area to neutral gray for conditioning)
        m = (1.0 - inpaint_mask.round()).to(device)
        pixels = pixels.to(device)
        for i in range(3):
            pixels[:, :, :, i] = pixels[:, :, :, i] * m + 0.5 * (1 - m)

        # 6. VAE encode
        latent_samples = vae.encode(pixels)
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        # 7. Prepare noise
        noise = comfy.sample.prepare_noise(latent_samples, seed, None)

        # 8. Run sampling using comfy.sample.sample (standard ComfyUI API)
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            "dpmpp_2m",  # sampler
            "karras",     # scheduler
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            noise_mask=mask_grown,
            seed=seed
        )

        # 9. VAE decode
        samples = samples.to(mm.intermediate_device())
        result = vae.decode(samples)

        # 10. Handle size differences - paste back into original size if needed
        if H != new_h or W != new_w:
            full_result = image.clone().cpu()
            full_result[:, h_offset:h_offset + new_h, w_offset:w_offset + new_w, :] = result.cpu()
            result = full_result
        else:
            result = result.cpu()

        print(f"[AlignStylizedFrame] SD inpainting complete.")
        return result

    def feather_mask(self, mask, radius, device):
        """Create soft feathered edges on a mask using gaussian-like blur."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Multiple blur passes for smooth falloff
        kernel_size = radius * 2 + 1
        sigma = radius / 2.0

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - radius
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        # Separable convolution (horizontal then vertical)
        kernel_h = gaussian_1d.view(1, 1, 1, kernel_size)
        kernel_v = gaussian_1d.view(1, 1, kernel_size, 1)

        # Pad and convolve
        padded = F.pad(mask, (radius, radius, radius, radius), mode='replicate')
        blurred = F.conv2d(padded, kernel_h)
        blurred = F.conv2d(blurred, kernel_v)

        return blurred.squeeze(1)

    def create_difference_visualization(self, original, aligned, before_stylized, device,
                                        mode="heatmap", subject_mask=None):
        """Create a before/after difference visualization."""
        B, H, W, C = original.shape
        label_height = max(20, H // 30)

        if mode == "subject_mask" and subject_mask is not None:
            # Show subject mask overlay
            mask_rgb = subject_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
            vis_before = original * 0.5 + mask_rgb * 0.5 * torch.tensor([1.0, 0.3, 0.3], device=device)
            vis_after = aligned * 0.5 + mask_rgb * 0.5 * torch.tensor([0.3, 1.0, 0.3], device=device)

        elif mode == "heatmap":
            orig_edges = self.extract_edges(original, device)
            before_edges = self.extract_edges(before_stylized, device)
            after_edges = self.extract_edges(aligned, device)

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

            diff_before = (0.299 * diff_before[..., 0] + 0.587 * diff_before[..., 1] + 0.114 * diff_before[..., 2])
            diff_after = (0.299 * diff_after[..., 0] + 0.587 * diff_after[..., 1] + 0.114 * diff_after[..., 2])

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

    def align_frames(self, original_image, stylized_image, scale_range=0.05,
                     translation_range=32, search_precision="balanced",
                     visualization_mode="overlay", subject_mode="disabled",
                     subject_mask=None, subject_scale_correction=1.0,
                     subject_position_correction=1.0, inpaint_method="sd_inpaint",
                     mask_expand=10, inpaint_steps=20, inpaint_denoise=0.9):
        """
        Main alignment function with subject-preserving mode.

        When subject_mode is enabled:
        - Subject is preserved EXACTLY (no warping/scaling)
        - Subject is repositioned to match original
        - Background is warped to fill gaps around the subject
        """

        device = mm.get_torch_device()
        original_image = original_image.to(device)
        stylized_image = stylized_image.to(device)

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
            orig_mask = self.birefnet_segment(original_image, device)
            styl_mask = self.birefnet_segment(stylized_image, device)
            subject_info = "Subject: BiRefNet segmentation\n"
        elif subject_mode == "auto":
            orig_mask = self.auto_detect_subject(original_image, stylized_image, device)
            styl_mask = self.auto_detect_subject(stylized_image, original_image, device)
            subject_info = "Subject: auto-detected\n"
        elif subject_mode == "mask" and subject_mask is not None:
            orig_mask = subject_mask.to(device)
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
            # For stylized, we need to detect where subject actually is (different position!)
            if is_birefnet_available():
                styl_mask = self.birefnet_segment(stylized_image, device)
            else:
                styl_mask = self.auto_detect_subject(stylized_image, original_image, device)
            subject_info = "Subject: from provided mask (orig) + detected (stylized)\n"

        # Extract edges for background alignment
        original_edges = self.extract_edges(original_image, device)

        # PHASE 1: Global background alignment (excluding subject if detected)
        # Use orig_mask for background weighting during alignment
        bg_mask = orig_mask if orig_mask is not None else None
        best_params, best_score = self.pyramid_search(
            original_edges, stylized_image, scale_range, translation_range,
            search_precision, device, bg_mask
        )

        # Apply global alignment
        aligned_image = self.apply_transform(
            stylized_image,
            best_params['scale'],
            best_params['tx'],
            best_params['ty'],
            device
        )

        # PHASE 2: Subject-preserving correction (if enabled)
        if orig_mask is not None and (subject_scale_correction > 0 or subject_position_correction > 0):
            # CRITICAL: After alignment, we need to detect where subject is in ALIGNED image
            # for correct inpainting position (the transform moved the subject!)
            if subject_mode == "birefnet":
                aligned_styl_mask = self.birefnet_segment(aligned_image, device)
            else:
                aligned_styl_mask = self.auto_detect_subject(aligned_image, original_image, device)

            # Use the CORRECT approach with THREE masks:
            # - orig_mask: target position (where subject should end up)
            # - styl_mask: extraction position (where subject is in stylized_before)
            # - aligned_styl_mask: inpaint position (where ghost would appear in aligned_image)
            aligned_image, correction_info = self.preserve_subject_inpaint_background(
                stylized_before,      # Original stylized BEFORE any transforms
                aligned_image,        # Background-aligned image
                original_image,
                orig_mask,            # Target: where subject should go
                styl_mask,            # For extraction from stylized_before
                aligned_styl_mask,    # For inpainting (where ghost is in aligned_image)
                subject_scale_correction,
                subject_position_correction,
                inpaint_method,
                mask_expand,
                inpaint_steps,
                inpaint_denoise,
                device
            )

            subject_info += correction_info + "\n"

        # Create visualization
        difference_map = self.create_difference_visualization(
            original_image, aligned_image, stylized_before, device,
            visualization_mode, orig_mask
        )

        # Format info
        alignment_info = (
            f"Background scale: {best_params['scale']:.4f} ({(best_params['scale'] - 1.0) * 100:+.2f}%)\n"
            f"Background translation: ({best_params['tx']:.1f}, {best_params['ty']:.1f}) px\n"
            f"Alignment score: {best_score:.6f}\n"
            f"{subject_info}"
        )

        # Prepare mask output (return orig_mask for user reference)
        if orig_mask is not None:
            output_mask = orig_mask.cpu()
        else:
            output_mask = torch.zeros(B, H_orig, W_orig)

        return (
            aligned_image.cpu(),
            difference_map.cpu(),
            alignment_info,
            output_mask
        )


NODE_CLASS_MAPPINGS = {
    "AlignStylizedFrame": AlignStylizedFrame
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlignStylizedFrame": "Align Stylized Frame"
}
