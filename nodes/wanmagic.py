import torch
import numpy as np
from pathlib import Path
import cv2
import os

class Wan21FrameAdjusterNode:
    """
    A ComfyUI node that adjusts video frame counts to meet Wan2.1 requirements.
    It always rounds up to the next valid frame count by adding blank gray frames.
    Wan2.1 needs frame counts that are divisible by 4 plus 1 (e.g., 5, 9, 13, ..., 41, 81, etc.)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("adjusted_images", "final_frame_count", "frames_added")
    FUNCTION = "adjust_frame_count"
    CATEGORY = "Trent Nodes"
    
    def adjust_frame_count(self, images):
        """
        Main function to adjust frame count for Wan2.1 compatibility.
        It always rounds up to the next valid count by adding gray frames.
        """
        current_count = images.shape[0]
        
        # Wan2.1 valid counts are (n*4)+1, starting from n=1 (5 frames).
        # Smallest valid count is 5.
        if current_count < 5:
            target_count = 5
        else:
            remainder = (current_count - 1) % 4
            if remainder == 0:
                target_count = current_count  # Already a valid count
            else:
                target_count = current_count + (4 - remainder)

        frames_to_add = target_count - current_count
        
        if frames_to_add > 0:
            # Get image dimensions (height, width, channels)
            _, h, w, c = images.shape
            
            # Create a single gray frame (0.5 for float tensors, which is common in ComfyUI)
            gray_frame = torch.full((1, h, w, c), 0.5, dtype=images.dtype, device=images.device)
            
            # Duplicate the gray frame for the number of frames to add
            added_frames = gray_frame.repeat(frames_to_add, 1, 1, 1)
            
            # Concatenate original frames with the new gray frames
            adjusted_images = torch.cat([images, added_frames], dim=0)
        else:
            adjusted_images = images

        final_count = adjusted_images.shape[0]
        
        return (adjusted_images, final_count, frames_to_add)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Wan21FrameAdjusterNode": Wan21FrameAdjusterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan21FrameAdjusterNode": "Wan2.1 Frame Adjuster"
}