import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class CrossDissolveOverlap:
    """
    A ComfyUI node that cross-dissolves between two batches of images
    with a specified overlap amount in frames.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),  # First batch of images
                "images_b": ("IMAGE",),  # Second batch of images
                "overlap_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,  # Increased limit for longer sequences
                    "step": 1,
                    "display": "number"
                }),
                "dissolve_curve": (["linear", "ease_in", "ease_out", "ease_in_out", "smooth", "bounce", "elastic"], {
                    "default": "linear"
                }),
            },
            "optional": {
                "auto_adjust_overlap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically adjust overlap if it exceeds batch sizes"
                }),
                "feather_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply edge feathering to reduce visible seams"
                }),
                "preserve_aspect": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ensure both image batches have matching dimensions"
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light"], {
                    "default": "normal",
                    "tooltip": "Blending mode for the crossfade"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")  # Added output info
    RETURN_NAMES = ("images", "frame_count", "blend_info")
    FUNCTION = "cross_dissolve_overlap"
    CATEGORY = "Trent Nodes"
    DESCRIPTION = "Cross-dissolve two image batches with precise overlap control"
    
    def validate_inputs(self, images_a: torch.Tensor, images_b: torch.Tensor, overlap_frames: int) -> Tuple[bool, str]:
        """
        Validate input parameters and provide helpful error messages.
        This helps users understand what went wrong before the node fails.
        """
        # Check if images are valid tensors
        if not isinstance(images_a, torch.Tensor) or not isinstance(images_b, torch.Tensor):
            return False, "Both image inputs must be valid tensors"
        
        # Check dimensions - ComfyUI images should be 4D: [batch, height, width, channels]
        if len(images_a.shape) != 4 or len(images_b.shape) != 4:
            return False, f"Images must be 4D tensors [batch, height, width, channels]. Got shapes: {images_a.shape}, {images_b.shape}"
        
        # Check if we have at least one image in each batch
        if images_a.shape[0] == 0 or images_b.shape[0] == 0:
            return False, "Both image batches must contain at least one image"
        
        # Check if overlap is reasonable
        min_batch_size = min(images_a.shape[0], images_b.shape[0])
        if overlap_frames > min_batch_size:
            return False, f"Overlap frames ({overlap_frames}) cannot exceed the smaller batch size ({min_batch_size})"
        
        return True, "Inputs are valid"
    
    def resize_to_match(self, images_a: torch.Tensor, images_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resize images to match dimensions if they differ.
        This prevents errors when blending images of different sizes.
        """
        # Get dimensions
        h_a, w_a = images_a.shape[1:3]
        h_b, w_b = images_b.shape[1:3]
        
        # If dimensions match, return as-is
        if h_a == h_b and w_a == w_b:
            return images_a, images_b
        
        # Use the larger dimensions to avoid quality loss
        target_h = max(h_a, h_b)
        target_w = max(w_a, w_b)
        
        # Resize if needed - permute to match F.interpolate expected format [batch, channels, height, width]
        if h_a != target_h or w_a != target_w:
            images_a = images_a.permute(0, 3, 1, 2)  # [batch, channels, height, width]
            images_a = F.interpolate(images_a, size=(target_h, target_w), mode='bilinear', align_corners=False)
            images_a = images_a.permute(0, 2, 3, 1)  # Back to [batch, height, width, channels]
        
        if h_b != target_h or w_b != target_w:
            images_b = images_b.permute(0, 3, 1, 2)
            images_b = F.interpolate(images_b, size=(target_h, target_w), mode='bilinear', align_corners=False)
            images_b = images_b.permute(0, 2, 3, 1)
        
        return images_a, images_b
    
    def get_alpha_curve(self, t, curve_type):
        """
        Generate alpha values based on curve type.
        Added more curve types and improved mathematical accuracy.
        """
        # Clamp t to ensure it's always between 0 and 1
        t = torch.clamp(t, 0.0, 1.0)
        
        if curve_type == "linear":
            return t
        elif curve_type == "ease_in":
            return t * t
        elif curve_type == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif curve_type == "ease_in_out":
            return 3 * t * t - 2 * t * t * t
        elif curve_type == "smooth":
            return t * t * (3 - 2 * t)
        elif curve_type == "bounce":
            # Simple bounce effect - more dramatic at the end
            return t * (2 - t)
        elif curve_type == "elastic":
            # Elastic effect with slight overshoot
            if t == 0:
                return torch.tensor(0.0)
            elif t == 1:
                return torch.tensor(1.0)
            else:
                p = 0.3
                s = p / 4
                return 1 - torch.pow(2, -10 * t) * torch.sin((t - s) * (2 * torch.pi) / p)
        else:
            return t
    
    def apply_blend_mode(self, img_a: torch.Tensor, img_b: torch.Tensor, alpha: float, blend_mode: str) -> torch.Tensor:
        """
        Apply different blending modes for more creative control.
        This gives artists more options for how the images combine.
        """
        if blend_mode == "normal":
            return img_a * (1 - alpha) + img_b * alpha
        elif blend_mode == "multiply":
            # Multiply blend mode - darker results
            blended = img_a * img_b
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "screen":
            # Screen blend mode - lighter results
            blended = 1 - (1 - img_a) * (1 - img_b)
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "overlay":
            # Overlay blend mode - preserves highlights and shadows
            mask = img_a < 0.5
            blended = torch.where(mask, 2 * img_a * img_b, 1 - 2 * (1 - img_a) * (1 - img_b))
            return img_a * (1 - alpha) + blended * alpha
        elif blend_mode == "soft_light":
            # Soft light blend mode - subtle lighting effect
            mask = img_b < 0.5
            blended = torch.where(mask, 
                                img_a - (1 - 2 * img_b) * img_a * (1 - img_a),
                                img_a + (2 * img_b - 1) * (torch.sqrt(img_a) - img_a))
            return img_a * (1 - alpha) + blended * alpha
        else:
            # Default to normal blend
            return img_a * (1 - alpha) + img_b * alpha
    
    def apply_edge_feathering(self, image: torch.Tensor, feather_amount: float = 0.05) -> torch.Tensor:
        """
        Apply edge feathering to reduce visible seams.
        This creates a subtle fade at the edges which helps with seamless blending.
        """
        h, w = image.shape[1:3]
        feather_pixels = int(min(h, w) * feather_amount)
        
        if feather_pixels <= 0:
            return image
        
        # Create feather mask
        mask = torch.ones_like(image[:, :, :, 0:1])  # Single channel mask
        
        # Apply feathering to edges
        for i in range(feather_pixels):
            alpha = i / feather_pixels
            # Top and bottom edges
            mask[:, i, :, 0] *= alpha
            mask[:, -(i+1), :, 0] *= alpha
            # Left and right edges
            mask[:, :, i, 0] *= alpha
            mask[:, :, -(i+1), 0] *= alpha
        
        # Apply mask to all channels
        return image * mask
    
    def generate_blend_info(self, images_a: torch.Tensor, images_b: torch.Tensor, 
                          overlap_frames: int, output_length: int, dissolve_curve: str, 
                          blend_mode: str) -> str:
        """
        Generate informative string about the blend operation.
        This helps users understand what the node did and troubleshoot issues.
        """
        info = f"Cross-dissolve completed:\n"
        info += f"• Input A: {images_a.shape[0]} frames ({images_a.shape[1]}x{images_a.shape[2]})\n"
        info += f"• Input B: {images_b.shape[0]} frames ({images_b.shape[1]}x{images_b.shape[2]})\n"
        info += f"• Overlap: {overlap_frames} frames\n"
        info += f"• Output: {output_length} frames\n"
        info += f"• Curve: {dissolve_curve}\n"
        info += f"• Blend mode: {blend_mode}\n"
        info += f"• Savings: {images_a.shape[0] + images_b.shape[0] - output_length} frames removed due to overlap"
        return info
    
    def cross_dissolve_overlap(self, images_a, images_b, overlap_frames, dissolve_curve, 
                             auto_adjust_overlap=True, feather_edges=False, preserve_aspect=True, 
                             blend_mode="normal"):
        """
        Main function that performs the cross-dissolve with overlap.
        Now includes comprehensive error handling and optional enhancements.
        """
        try:
            # Ensure inputs are tensors - this handles various input types gracefully
            if not isinstance(images_a, torch.Tensor):
                images_a = torch.tensor(images_a, dtype=torch.float32)
            if not isinstance(images_b, torch.Tensor):
                images_b = torch.tensor(images_b, dtype=torch.float32)
            
            # Validate inputs first - this prevents mysterious failures later
            is_valid, error_msg = self.validate_inputs(images_a, images_b, overlap_frames)
            if not is_valid:
                raise ValueError(f"Input validation failed: {error_msg}")
            
            # Auto-adjust overlap if requested and necessary
            max_overlap = min(images_a.shape[0], images_b.shape[0])
            if auto_adjust_overlap and overlap_frames > max_overlap:
                print(f"Warning: Reducing overlap from {overlap_frames} to {max_overlap} frames to fit batch sizes")
                overlap_frames = max_overlap
            
            # Resize images to match dimensions if preserve_aspect is enabled
            if preserve_aspect:
                images_a, images_b = self.resize_to_match(images_a, images_b)
            
            # Calculate output sequence length
            batch_a = images_a.shape[0]
            batch_b = images_b.shape[0]
            output_length = batch_a + batch_b - overlap_frames
            
            # Apply edge feathering if requested - this helps reduce visible seams
            if feather_edges:
                images_a = self.apply_edge_feathering(images_a)
                images_b = self.apply_edge_feathering(images_b)
            
            # Process each frame in the output sequence
            output_images = []
            
            for i in range(output_length):
                if i < batch_a - overlap_frames:
                    # Before overlap region - use only images_a
                    # This preserves the original sequence A completely
                    output_images.append(images_a[i])
                    
                elif i < batch_a:
                    # In overlap region - blend images_a and images_b
                    # This is where the magic happens - we smoothly transition between sequences
                    a_idx = i
                    b_idx = i - (batch_a - overlap_frames)
                    
                    # Calculate blend ratio (0 to 1) with more precision
                    blend_progress = (i - (batch_a - overlap_frames)) / overlap_frames
                    
                    # Convert to tensor for curve calculations if needed
                    if isinstance(blend_progress, (int, float)):
                        blend_progress = torch.tensor(blend_progress, dtype=torch.float32)
                    
                    alpha = self.get_alpha_curve(blend_progress, dissolve_curve)
                    
                    # Get the images to blend
                    img_a = images_a[a_idx]
                    img_b = images_b[b_idx]
                    
                    # Apply the selected blend mode - this gives creative control
                    blended = self.apply_blend_mode(img_a, img_b, alpha, blend_mode)
                    output_images.append(blended)
                    
                else:
                    # After overlap region - use only images_b
                    # This preserves the original sequence B completely
                    b_idx = i - (batch_a - overlap_frames)
                    output_images.append(images_b[b_idx])
            
            # Stack all images into a single tensor - this creates our final sequence
            result = torch.stack(output_images, dim=0)
            
            # Generate informative output about what we accomplished
            blend_info = self.generate_blend_info(images_a, images_b, overlap_frames, 
                                                output_length, dissolve_curve, blend_mode)
            
            return (result, output_length, blend_info)
            
        except Exception as e:
            # Provide helpful error information instead of cryptic failures
            error_info = f"CrossDissolveOverlap failed: {str(e)}\n"
            error_info += f"Input shapes: A={images_a.shape if 'images_a' in locals() else 'unknown'}, "
            error_info += f"B={images_b.shape if 'images_b' in locals() else 'unknown'}\n"
            error_info += f"Overlap frames: {overlap_frames}"
            
            # Return a safe fallback - just concatenate the sequences
            print(f"Error in CrossDissolveOverlap: {error_info}")
            fallback_result = torch.cat([images_a, images_b], dim=0)
            return (fallback_result, fallback_result.shape[0], f"Error occurred - returned concatenated sequences: {str(e)}")

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CrossDissolveOverlap": CrossDissolveOverlap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CrossDissolveOverlap": "Cross Dissolve with Overlap"
}

# Example usage and testing
if __name__ == "__main__":
    # Test the node with dummy data
    node = CrossDissolveOverlap()
    
    # Create dummy image batches (batch_size, height, width, channels)
    images_a = torch.rand(20, 512, 512, 3)  # 20 frames
    images_b = torch.rand(15, 512, 512, 3)  # 15 frames
    
    # Test with 5 frame overlap
    result = node.cross_dissolve_overlap(images_a, images_b, 5, "linear")
    
    print(f"Input A: {images_a.shape[0]} frames")
    print(f"Input B: {images_b.shape[0]} frames")
    print(f"Overlap: 5 frames")
    print(f"Output: {result[0].shape[0]} frames")
    print(f"Expected: {20 + 15 - 5} = 30 frames")