"""
ComfyUI Bevel/Emboss Node with GPU Support and Extended Ranges
A custom node that applies Photoshop-style bevel and emboss effects to images.
Enforces GPU acceleration when available, with device verification.
Place this file in: ComfyUI/custom_nodes/bevel_emboss_node.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import math
import comfy.model_management as mm  # ComfyUI's model management for device handling

class BevelEmbossNode:
    """
    A ComfyUI node that applies bevel/emboss effects to images with enforced GPU support.
    Features extended parameter ranges and improved bevel algorithms.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI image tensor format
                "depth": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "angle": ("INT", {
                    "default": 135,
                    "min": -360,
                    "max": 360,
                    "step": 1
                }),
                "highlight_opacity": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                "shadow_opacity": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                "soften": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.1
                }),
                "mode": (["emboss", "bevel_inner", "bevel_outer", "pillow_emboss"],),
                "bevel_width": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.5
                }),
                "debug_device": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Show Device",
                    "label_off": "Hide Device"
                }),
            },
            "optional": {
                "mask": ("MASK",),  # Optional mask input
                "mask_mode": (["apply_to_masked", "apply_to_unmasked"],),
                "mask_blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_bevel_emboss"
    CATEGORY = "Trent Nodes"
    
    def apply_bevel_emboss(self, image, depth, angle, highlight_opacity, shadow_opacity, soften, mode, 
                           bevel_width=5.0, debug_device=False, mask=None, mask_mode="apply_to_masked", mask_blur=0.0):
        """
        Main processing function with enforced GPU support.
        Uses ComfyUI's model management for proper device handling.
        """
        
        # Get the optimal device for processing from ComfyUI's model management
        device = mm.get_torch_device()
        
        # Force CPU if no GPU available (fallback)
        if not torch.cuda.is_available() and device.type == 'cuda':
            device = torch.device('cpu')
            if debug_device:
                print(f"[BevelEmboss] WARNING: CUDA not available, falling back to CPU")
        
        # Move image to the processing device
        image = image.to(device)
        
        if debug_device:
            print(f"[BevelEmboss] Processing on device: {device}")
            print(f"[BevelEmboss] Image device: {image.device}")
            print(f"[BevelEmboss] Image shape: {image.shape}")
            if torch.cuda.is_available():
                print(f"[BevelEmboss] GPU Name: {torch.cuda.get_device_name(0)}")
                print(f"[BevelEmboss] GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB used")
        
        batch_size, height, width, channels = image.shape
        
        # Process the mask if provided
        if mask is not None:
            # Move mask to processing device
            mask = mask.to(device)
            
            if debug_device:
                print(f"[BevelEmboss] Mask device: {mask.device}")
            
            # Ensure mask has the same batch size as the image
            if mask.shape[0] != batch_size:
                if mask.shape[0] == 1:
                    mask = mask.repeat(batch_size, 1, 1)
                else:
                    raise ValueError(f"Mask batch size ({mask.shape[0]}) doesn't match image batch size ({batch_size})")
            
            # Resize mask if needed
            if mask.shape[1] != height or mask.shape[2] != width:
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            # Apply blur to mask if requested
            if mask_blur > 0:
                mask = self._blur_mask(mask, mask_blur, device)
            
            # Invert mask if applying to unmasked areas
            if mask_mode == "apply_to_unmasked":
                mask = 1.0 - mask
        
        # Process each image in the batch
        result_batch = []
        
        for i in range(batch_size):
            single_image = image[i]
            
            # Ensure single image is on device
            if single_image.device != device:
                single_image = single_image.to(device)
            
            # Apply the effect based on selected mode
            if mode == "emboss":
                processed = self._apply_emboss(single_image, depth, angle, highlight_opacity, shadow_opacity, soften, device)
            elif mode == "bevel_inner":
                processed = self._apply_bevel_inner(single_image, depth, angle, highlight_opacity, shadow_opacity, soften, bevel_width, device)
            elif mode == "bevel_outer":
                processed = self._apply_bevel_outer(single_image, depth, angle, highlight_opacity, shadow_opacity, soften, bevel_width, device)
            else:  # pillow_emboss
                processed = self._apply_pillow_emboss(single_image, depth, angle, highlight_opacity, shadow_opacity, soften, bevel_width, device)
            
            # Ensure processed is on device
            if processed.device != device:
                processed = processed.to(device)
            
            # Apply mask if provided
            if mask is not None:
                current_mask = mask[i].unsqueeze(2).to(device)
                result = single_image * (1.0 - current_mask) + processed * current_mask
            else:
                result = processed
            
            result_batch.append(result)
        
        # Stack results back into batch format on device
        final_result = torch.stack(result_batch)
        
        if debug_device:
            print(f"[BevelEmboss] Output device: {final_result.device}")
            if torch.cuda.is_available():
                print(f"[BevelEmboss] GPU Memory after processing: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB used")
        
        return (final_result,)
    
    def _blur_mask(self, mask, blur_amount, device):
        """
        Apply Gaussian blur to a mask on the specified device.
        """
        kernel_size = int(blur_amount * 2) * 2 + 1
        sigma = blur_amount / 3
        
        batch_size = mask.shape[0]
        blurred_masks = []
        
        for i in range(batch_size):
            single_mask = mask[i].to(device)
            blurred = self._gaussian_blur(single_mask, kernel_size, sigma, device)
            blurred_masks.append(blurred)
        
        return torch.stack(blurred_masks).to(device)
    
    def _extract_rgb_and_alpha(self, image):
        """
        Extract RGB and alpha channels, handling different formats.
        """
        channels = image.shape[2]
        
        if channels == 4:
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]
            return rgb, alpha, True
        elif channels == 3:
            return image, None, False
        else:
            if channels >= 3:
                rgb = image[:, :, :3]
            else:
                rgb = image[:, :, 0:1].repeat(1, 1, 3)
            return rgb, None, False
    
    def _combine_rgb_and_alpha(self, rgb, alpha):
        """
        Recombine RGB and alpha channels.
        """
        if alpha is not None:
            return torch.cat([rgb, alpha.unsqueeze(2)], dim=2)
        else:
            return rgb
    
    def _apply_emboss(self, image, depth, angle, highlight_opacity, shadow_opacity, soften, device):
        """
        Emboss effect with enforced GPU processing.
        """
        # Ensure image is on device
        image = image.to(device)
        
        rgb, alpha, has_alpha = self._extract_rgb_and_alpha(image)
        
        # Create grayscale with weights on device
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=device).view(1, 1, 3)
        gray = torch.sum(rgb * weights, dim=2)
        
        # Calculate gradient direction
        angle_rad = math.radians(angle)
        dx = math.cos(angle_rad) * depth
        dy = math.sin(angle_rad) * depth
        
        # Create kernels directly on device
        kernel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=device) * dx / 8.0
        
        kernel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=device) * dy / 8.0
        
        # Reshape kernels for conv2d
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution on device
        gray_padded = F.pad(gray.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        
        # Ensure padded tensor is on device
        gray_padded = gray_padded.to(device)
        
        grad_x = F.conv2d(gray_padded, kernel_x)
        grad_y = F.conv2d(gray_padded, kernel_y)
        gradient = (grad_x + grad_y).squeeze()
        
        # Apply softening
        if soften > 0:
            blur_kernel_size = int(soften * 2) * 2 + 1
            gradient = self._gaussian_blur(gradient, blur_kernel_size, soften, device)
        
        # Create highlight and shadow maps
        highlights = torch.clamp(gradient, 0, 1)
        shadows = torch.clamp(-gradient, 0, 1)
        
        # Apply effect to RGB
        result_rgb = rgb.clone()
        highlight_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        result_rgb = result_rgb + (highlight_color.view(1, 1, 3) * highlights.unsqueeze(2) * highlight_opacity)
        
        shadow_factor = 1.0 - (shadows.unsqueeze(2) * shadow_opacity)
        result_rgb = result_rgb * shadow_factor
        
        result_rgb = torch.clamp(result_rgb, 0, 1)
        
        return self._combine_rgb_and_alpha(result_rgb, alpha).to(device)
    
    def _create_distance_field(self, mask, bevel_width, device):
        """
        Create a distance field from mask edges for better bevel effects.
        """
        # Ensure mask is on device
        mask = mask.to(device)
        
        # Ensure mask is binary
        binary_mask = (mask > 0.5).float()
        
        # Create distance field using repeated dilations/erosions
        distance_field = torch.zeros_like(mask, device=device)
        
        # Number of iterations based on bevel width
        iterations = int(bevel_width)
        
        for i in range(iterations):
            kernel_size = 3
            padding = kernel_size // 2
            
            # Create kernel on device
            kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32, device=device) / (kernel_size * kernel_size)
            
            # Dilate the mask
            mask_expanded = binary_mask.unsqueeze(0).unsqueeze(0).to(device)
            dilated = F.conv2d(mask_expanded, kernel, padding=padding).squeeze()
            dilated = (dilated > 0.5).float()
            
            # Erode the mask
            eroded = F.conv2d(mask_expanded, kernel, padding=padding).squeeze()
            eroded = (eroded > 0.9).float()
            
            # Create ring at this distance
            ring = dilated - eroded
            distance_field = torch.maximum(distance_field, ring * (1.0 - i / iterations))
            
            # Update mask for next iteration
            binary_mask = eroded
        
        return distance_field
    
    def _apply_bevel_inner(self, image, depth, angle, highlight_opacity, shadow_opacity, soften, bevel_width, device):
        """
        Improved inner bevel with enforced GPU processing.
        """
        # Ensure image is on device
        image = image.to(device)
        
        rgb, original_alpha, has_alpha = self._extract_rgb_and_alpha(image)
        
        # Create or derive alpha mask
        if has_alpha:
            alpha = original_alpha.to(device)
        else:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=device).view(1, 1, 3)
            gray = torch.sum(rgb * weights, dim=2)
            alpha = (gray > 0.1).float()
        
        # Create distance field for smooth bevel
        distance_field = self._create_distance_field(alpha, bevel_width, device)
        
        # Apply directional lighting
        angle_rad = math.radians(angle)
        
        # Create gradient maps on device
        h, w = distance_field.shape
        y_grad = torch.linspace(-1, 1, h, dtype=torch.float32, device=device).view(-1, 1).expand(h, w)
        x_grad = torch.linspace(-1, 1, w, dtype=torch.float32, device=device).view(1, -1).expand(h, w)
        
        # Calculate light direction components
        light_x = math.cos(angle_rad)
        light_y = -math.sin(angle_rad)
        
        # Create lighting based on distance field and direction
        lighting = distance_field * depth * (x_grad * light_x + y_grad * light_y)
        
        # Apply softening
        if soften > 0:
            blur_kernel_size = int(soften * 2) * 2 + 1
            lighting = self._gaussian_blur(lighting, blur_kernel_size, soften, device)
            distance_field = self._gaussian_blur(distance_field, blur_kernel_size, soften, device)
        
        # Create highlight and shadow maps
        highlights = torch.clamp(lighting * distance_field, 0, 1)
        shadows = torch.clamp(-lighting * distance_field, 0, 1)
        
        # Apply effect
        result_rgb = rgb.clone()
        highlight_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        
        result_rgb = result_rgb + (highlight_color.view(1, 1, 3) * highlights.unsqueeze(2) * highlight_opacity)
        shadow_factor = 1.0 - (shadows.unsqueeze(2) * shadow_opacity)
        result_rgb = result_rgb * shadow_factor
        
        result_rgb = torch.clamp(result_rgb, 0, 1)
        
        return self._combine_rgb_and_alpha(result_rgb, original_alpha).to(device)
    
    def _apply_bevel_outer(self, image, depth, angle, highlight_opacity, shadow_opacity, soften, bevel_width, device):
        """
        Outer bevel using inverted inner bevel approach.
        """
        inverted_angle = (angle + 180) % 360
        return self._apply_bevel_inner(image, depth, inverted_angle, shadow_opacity, highlight_opacity, soften, bevel_width, device)
    
    def _apply_pillow_emboss(self, image, depth, angle, highlight_opacity, shadow_opacity, soften, bevel_width, device):
        """
        Pillow emboss combining both inner and outer bevels.
        """
        inner = self._apply_bevel_inner(image, depth * 0.6, angle, 
                                        highlight_opacity * 0.8, shadow_opacity * 0.8, 
                                        soften, bevel_width * 0.7, device)
        
        outer = self._apply_bevel_outer(image, depth * 0.6, angle, 
                                        highlight_opacity * 0.8, shadow_opacity * 0.8, 
                                        soften, bevel_width * 0.7, device)
        
        # Ensure both are on device
        inner = inner.to(device)
        outer = outer.to(device)
        
        # Extract RGB channels
        rgb_inner, alpha_inner, _ = self._extract_rgb_and_alpha(inner)
        rgb_outer, alpha_outer, _ = self._extract_rgb_and_alpha(outer)
        
        # Blend
        blended_rgb = rgb_inner * 0.5 + rgb_outer * 0.5
        
        return self._combine_rgb_and_alpha(blended_rgb, alpha_inner if alpha_inner is not None else alpha_outer).to(device)
    
    def _gaussian_blur(self, tensor, kernel_size, sigma, device):
        """
        GPU-accelerated Gaussian blur with enforced device placement.
        """
        # Ensure tensor is on device
        tensor = tensor.to(device)
        
        # Create 1D Gaussian kernel on device
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel on device
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        tensor_padded = F.pad(tensor.unsqueeze(0).unsqueeze(0), 
                              (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                              mode='replicate').to(device)
        
        blurred = F.conv2d(tensor_padded, kernel_2d).squeeze()
        
        return blurred.to(device)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BevelEmboss": BevelEmbossNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BevelEmboss": "Bevel/Emboss Effect"
}