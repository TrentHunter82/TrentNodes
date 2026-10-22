"""
Latent-Aligned Mask Node for ComfyUI
Automatically creates properly blocky masks aligned to VAE latent space
"""

import torch
import torch.nn.functional as F


class LatentMaskProcessor:
    """
    Base class providing shared latent-space mask processing logic.

    Provides a single implementation for latent-aligned mask operations
    that can be reused across different node variants.
    """

    @staticmethod
    def process_mask_latent(
        mask: torch.Tensor,
        compression: int,
        expansion_pixels: int = 8,
        blur_latent_units: int = 1,
        threshold: float = 0.5
    ) -> tuple:
        """
        Core mask processing aligned to VAE latent space.

        Args:
            mask: Input mask tensor (B, H, W) or (H, W)
            compression: VAE spatial compression factor
            expansion_pixels: Pixels to expand in LATENT space
            blur_latent_units: Blur radius in latent space units
            threshold: Binary threshold after processing

        Returns:
            Tuple of (aligned_mask, latent_preview)
        """
        # Track original dimensions
        orig_dim = mask.dim()

        # Ensure mask is 4D (B, 1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        original_shape = mask.shape
        device = mask.device
        dtype = mask.dtype

        # Step 1: Downsample to latent resolution
        latent_h = original_shape[2] // compression
        latent_w = original_shape[3] // compression

        mask_latent = F.interpolate(
            mask.float(),
            size=(latent_h, latent_w),
            mode='area'
        )

        # Step 2: Expand mask in latent space (prevents black edges)
        if expansion_pixels > 0:
            kernel_size = expansion_pixels * 2 + 1
            padding = expansion_pixels
            mask_expanded = F.max_pool2d(
                mask_latent,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
            mask_latent = torch.clamp(mask_expanded, 0, 1)

        # Step 3: Blur in latent space (softens boundaries)
        if blur_latent_units > 0:
            kernel_size = blur_latent_units * 2 + 1
            sigma = blur_latent_units / 2

            # Create Gaussian kernel
            x = torch.arange(
                kernel_size, device=device, dtype=torch.float32
            ) - kernel_size // 2
            gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()

            gauss_2d = gauss_1d.view(-1, 1) * gauss_1d.view(1, -1)
            gauss_kernel = gauss_2d.view(1, 1, kernel_size, kernel_size)

            padding = kernel_size // 2
            mask_latent = F.conv2d(mask_latent, gauss_kernel, padding=padding)

        # Step 4: Apply threshold in latent space
        mask_latent_binary = (mask_latent >= threshold).float()

        # Step 5: Upsample back to original resolution using NEAREST neighbor
        mask_aligned = F.interpolate(
            mask_latent_binary,
            size=(original_shape[2], original_shape[3]),
            mode='nearest'
        )

        # Create preview of latent space
        latent_preview = F.interpolate(
            mask_latent,
            size=(original_shape[2], original_shape[3]),
            mode='nearest'
        )

        # Convert back to original format
        mask_aligned = mask_aligned.squeeze(1).to(dtype)
        latent_preview = latent_preview.squeeze(1).to(dtype)

        # If input was 2D, return 2D
        if orig_dim == 2:
            mask_aligned = mask_aligned.squeeze(0)
            latent_preview = latent_preview.squeeze(0)

        return mask_aligned, latent_preview


class LatentAlignedMask(LatentMaskProcessor):
    """
    Creates masks aligned to VAE latent space to prevent black spots and artifacts.
    Automatically calculates the correct blockiness based on VAE compression.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to snap to the VAE latent grid"}),
                "vae": ("VAE", {"tooltip": "VAE used to auto-detect the spatial compression factor"}),
                "expansion_pixels": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Expand mask in LATENT space (prevents black edges)"
                }),
                "blur_latent_units": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Blur amount in latent units (softens boundaries)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Binary threshold after processing"
                }),
            },
            "optional": {
                "override_compression": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Manual compression factor (0=auto-detect, typically 8)"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "MASK", "INT")
    RETURN_NAMES = ("aligned_mask", "latent_preview", "compression_factor")
    FUNCTION = "align_mask"
    CATEGORY = "Trent/Masks"
    DESCRIPTION = "Snaps a mask to the VAE latent grid so masked regions match whole latent cells, which prevents black spots and edge artifacts in inpainting. Detects the compression factor from the VAE unless you override it."
    
    def get_vae_compression_factor(self, vae):
        """
        Auto-detect VAE's spatial compression factor.
        Handles both standard 2D VAEs and Wan's 3D video VAE.
        Most VAEs use 8x, but some use 4x, 16x, etc.
        """
        try:
            # Check if this is a Wan 3D VAE (has specific attributes)
            if hasattr(vae, 'spatial_compression_ratio'):
                compression = vae.spatial_compression_ratio
                print(f"[LatentAlignedMask] Detected Wan VAE spatial compression: {compression}x")
                return compression
            
            # Check for downsampling_factor attribute
            if hasattr(vae, 'downsampling_factor'):
                return vae.downsampling_factor
            
            # Try to detect from the actual VAE object structure
            # Wan VAEs are often wrapped differently
            if hasattr(vae, 'vae') and hasattr(vae.vae, 'spatial_compression_ratio'):
                compression = vae.vae.spatial_compression_ratio
                print(f"[LatentAlignedMask] Detected Wan VAE (wrapped) spatial compression: {compression}x")
                return compression
            
            # Check encoder structure for standard VAEs
            if hasattr(vae, 'first_stage_model') and hasattr(vae.first_stage_model, 'encoder'):
                encoder = vae.first_stage_model.encoder
                if hasattr(encoder, 'num_resolutions'):
                    # Each resolution typically doubles compression
                    return 2 ** (encoder.num_resolutions - 1)
            
            # For Wan models, check if it has the config
            if hasattr(vae, 'config'):
                config = vae.config
                if hasattr(config, 'spatial_compression_ratio'):
                    compression = config.spatial_compression_ratio
                    print(f"[LatentAlignedMask] Detected VAE compression from config: {compression}x")
                    return compression
            
            # Test encode/decode to measure (works for most VAEs)
            # This might fail for 3D VAEs that require temporal dimension
            try:
                test_size = 512
                test_input = torch.randn(1, 3, test_size, test_size)
                
                # Move to same device as VAE
                if hasattr(vae, 'device'):
                    test_input = test_input.to(vae.device)
                elif hasattr(vae, 'vae') and hasattr(vae.vae, 'device'):
                    test_input = test_input.to(vae.vae.device)
                
                with torch.no_grad():
                    # Handle different VAE encode methods
                    if hasattr(vae, 'encode'):
                        encoded = vae.encode(test_input)
                    elif hasattr(vae, 'vae') and hasattr(vae.vae, 'encode'):
                        encoded = vae.vae.encode(test_input)
                    else:
                        raise AttributeError("No encode method found")
                    
                    if hasattr(encoded, 'sample'):
                        encoded = encoded.sample()
                    
                    # Get spatial dimensions
                    latent_size = encoded.shape[-1]
                
                compression = test_size // latent_size
                print(f"[LatentAlignedMask] Detected VAE compression via test: {compression}x")
                return compression
            except Exception as test_error:
                print(f"[LatentAlignedMask] Test encode failed (expected for 3D VAEs): {test_error}")
            
            # Default to 8x for Wan models if detection fails
            print(f"[LatentAlignedMask] Could not detect compression, defaulting to 8x (standard for Wan/SD)")
            return 8
            
        except Exception as e:
            print(f"[LatentAlignedMask] Error detecting compression, defaulting to 8x: {e}")
            return 8
    
    def align_mask(self, mask, vae, expansion_pixels=8, blur_latent_units=1,
                   threshold=0.5, override_compression=0):
        """
        Process mask to align with VAE latent space.

        Args:
            mask: Input mask tensor (B, H, W) or (H, W)
            vae: VAE model to get compression factor
            expansion_pixels: Pixels to expand in LATENT space
            blur_latent_units: Blur radius in latent space units
            threshold: Binary threshold after processing
            override_compression: Manual compression factor (0=auto)
        """
        # Get compression factor
        if override_compression > 0:
            compression = override_compression
        else:
            compression = self.get_vae_compression_factor(vae)

        # Use shared processing method
        mask_aligned, latent_preview = self.process_mask_latent(
            mask, compression, expansion_pixels, blur_latent_units, threshold
        )

        return (mask_aligned, latent_preview, compression)


class LatentAlignedMaskAdvanced(LatentMaskProcessor):
    """
    Advanced version with character detection and smart expansion.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to snap to the VAE latent grid"}),
                "vae": ("VAE", {"tooltip": "VAE used to auto-detect the spatial compression factor"}),
                "mode": (["conservative", "balanced", "aggressive"], {
                    "default": "balanced",
                    "tooltip": "How much to expand: conservative=less, aggressive=more"
                }),
                "prevent_edge_artifacts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically expand to prevent black edges"
                }),
                "smooth_boundaries": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth mask boundaries in latent space"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "process_mask"
    CATEGORY = "Trent/Masks"
    DESCRIPTION = "Latent-aligned mask with expansion, blur, and threshold picked from a preset mode. Use it when you want latent-safe masks without tuning sliders; the info output reports the settings used."
    
    def process_mask(self, mask, vae, mode="balanced", 
                    prevent_edge_artifacts=True, smooth_boundaries=True):
        
        # Mode presets
        presets = {
            "conservative": {"expansion": 4, "blur": 0, "threshold": 0.5},
            "balanced": {"expansion": 8, "blur": 1, "threshold": 0.5},
            "aggressive": {"expansion": 12, "blur": 2, "threshold": 0.4}
        }
        
        params = presets[mode]
        
        if not prevent_edge_artifacts:
            params["expansion"] = 0
        
        if not smooth_boundaries:
            params["blur"] = 0
        
        # Use the main node
        aligner = LatentAlignedMask()
        mask_aligned, latent_preview, compression = aligner.align_mask(
            mask, vae,
            expansion_pixels=params["expansion"],
            blur_latent_units=params["blur"],
            threshold=params["threshold"]
        )
        
        info = (
            f"VAE Compression: {compression}x\n"
            f"Mode: {mode}\n"
            f"Expansion: {params['expansion']} latent pixels\n"
            f"Blur: {params['blur']} latent units\n"
            f"Threshold: {params['threshold']}"
        )
        
        return (mask_aligned, info)


class LatentAlignedMaskSimple(LatentMaskProcessor):
    """
    Simplified version that doesn't require VAE input.
    Just specify compression factor directly (8 for Wan/SD models).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to snap to the latent grid"}),
                "compression_factor": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Spatial compression (8 for Wan/SD, 4 for others)"
                }),
                "expansion_pixels": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Expand mask in LATENT space (prevents black edges)"
                }),
                "blur_latent_units": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Blur amount in latent units (softens boundaries)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Binary threshold after processing"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("aligned_mask", "latent_preview")
    FUNCTION = "align_mask"
    CATEGORY = "Trent/Masks"
    DESCRIPTION = "Snaps a mask to a latent grid using a compression factor you set by hand, with no VAE input needed. Use 8 for Wan and SD-family VAEs."

    def align_mask(self, mask, compression_factor=8, expansion_pixels=8,
                   blur_latent_units=1, threshold=0.5):
        """Process mask to align with latent space (no VAE needed)."""
        mask_aligned, latent_preview = self.process_mask_latent(
            mask, compression_factor, expansion_pixels, blur_latent_units, threshold
        )
        return (mask_aligned, latent_preview)


class LatentAlignedMaskWan(LatentMaskProcessor):
    """
    Optimized version specifically for Wan 2.1 VAE.
    Hardcoded for 8x spatial compression with sane defaults.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to snap to the Wan 2.1 8x latent grid"}),
                "preset": (["standard", "tight", "loose", "custom"], {
                    "default": "standard",
                    "tooltip": "standard=64px, tight=32px, loose=96px, custom=use sliders"
                }),
            },
            "optional": {
                "expansion_pixels": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Expansion in PIXEL space (only if preset=custom). Multiple of 8!"
                }),
                "blur_pixels": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 8,
                    "tooltip": "Blur in PIXEL space (only if preset=custom). Multiple of 8!"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("aligned_mask",)
    FUNCTION = "align_mask"
    CATEGORY = "Trent/Masks"
    DESCRIPTION = "Snaps a mask to the 8x latent grid of the Wan 2.1 VAE with pixel-space expansion and blur presets. Use it for Wan inpainting masks without needing a VAE input."
    
    def align_mask(self, mask, preset="standard", expansion_pixels=64, blur_pixels=8):
        """Process mask for Wan 2.1 VAE (8x compression, optimized settings)."""
        # Wan 2.1 VAE always uses 8x spatial compression
        compression_factor = 8

        # Presets in PIXEL space for intuitive understanding
        presets = {
            "tight": {"expansion_px": 32, "blur_px": 0},
            "standard": {"expansion_px": 64, "blur_px": 8},
            "loose": {"expansion_px": 96, "blur_px": 16},
            "custom": {"expansion_px": expansion_pixels, "blur_px": blur_pixels}
        }

        params = presets[preset]

        # Convert pixel-space to latent-space (divide by compression factor)
        expansion_latent = params["expansion_px"] // compression_factor
        blur_latent = params["blur_px"] // compression_factor

        # Use shared processing method (returns aligned_mask, latent_preview)
        mask_aligned, _ = self.process_mask_latent(
            mask, compression_factor, expansion_latent, blur_latent, threshold=0.5
        )

        return (mask_aligned,)


NODE_CLASS_MAPPINGS = {
    "LatentAlignedMask": LatentAlignedMask,
    "LatentAlignedMaskAdvanced": LatentAlignedMaskAdvanced,
    "LatentAlignedMaskSimple": LatentAlignedMaskSimple,
    "LatentAlignedMaskWan": LatentAlignedMaskWan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentAlignedMask": "Latent-Aligned Mask (with VAE)",
    "LatentAlignedMaskAdvanced": "Latent-Aligned Mask (Auto)",
    "LatentAlignedMaskSimple": "Latent-Aligned Mask (No VAE)",
    "LatentAlignedMaskWan": "Latent-Aligned Mask (Wan 2.1)",
}
