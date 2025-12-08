"""
Cherry Pick Frames
A ComfyUI custom node for flexible frame selection from image batches.
Supports multiple selection modes: first N, last N, specific indices, or every Nth frame.
"""

import torch
from typing import Dict, Any, Tuple, List

# Maximum number of frame outputs
MAX_OUTPUTS = 16


class CherryPickFrames:
    """
    Flexible frame selector with multiple modes for extracting frames from batches.
    
    Modes:
    - first_n: Get the first N frames from the batch
    - last_n: Get the last N frames from the batch  
    - specific: Pick specific frames by comma-separated indices (e.g., "0,5,10,75")
    - every_nth: Get every Nth frame from the batch
    """
    
    MODES = ["first_n", "last_n", "specific", "every_nth"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of images to pick frames from"
                }),
                "mode": (cls.MODES, {
                    "default": "first_n",
                    "tooltip": "Selection mode: first_n, last_n, specific indices, or every_nth"
                }),
            },
            "optional": {
                "num_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": MAX_OUTPUTS,
                    "step": 1,
                    "tooltip": "Number of frames to extract (for first_n, last_n, every_nth modes)"
                }),
                "frame_indices": ("STRING", {
                    "default": "0",
                    "tooltip": "Comma-separated frame indices, e.g., '0,5,10,75' (for specific mode)"
                }),
                "step": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Step size - grab every Nth frame (for every_nth mode)"
                }),
            },
        }
    
    # Fixed outputs - frame_1 through frame_16
    RETURN_TYPES = tuple(["IMAGE"] * MAX_OUTPUTS)
    RETURN_NAMES = tuple([f"frame_{i+1}" for i in range(MAX_OUTPUTS)])
    OUTPUT_IS_LIST = tuple([False] * MAX_OUTPUTS)
    
    FUNCTION = "pick_frames"
    CATEGORY = "Trent/Image"
    DESCRIPTION = "Flexibly select frames from a batch using various modes: first N, last N, specific indices, or every Nth frame."
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
    
    def parse_indices(self, frame_indices: str, batch_size: int) -> List[int]:
        """Parse comma-separated indices string into list of valid indices."""
        indices = []
        try:
            parts = frame_indices.replace(" ", "").split(",")
            for part in parts:
                if part:
                    idx = int(part)
                    # Support negative indices (from end)
                    if idx < 0:
                        idx = batch_size + idx
                    # Clamp to valid range
                    if 0 <= idx < batch_size:
                        indices.append(idx)
        except ValueError:
            pass
        
        # Remove duplicates while preserving order, limit to MAX_OUTPUTS
        seen = set()
        unique = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
                if len(unique) >= MAX_OUTPUTS:
                    break
        
        return unique if unique else [0]  # Default to first frame if parsing fails
    
    def pick_frames(
        self,
        images: torch.Tensor,
        mode: str,
        num_frames: int = 1,
        frame_indices: str = "0",
        step: int = 1,
        **kwargs
    ) -> Tuple:
        """
        Pick frames based on the selected mode.
        
        Returns tuple of MAX_OUTPUTS frames (unused slots get fallback frame).
        """
        
        batch_size = images.shape[0]
        
        # Determine which indices to grab based on mode
        if mode == "first_n":
            count = min(num_frames, batch_size, MAX_OUTPUTS)
            selected_indices = list(range(count))
            
        elif mode == "last_n":
            count = min(num_frames, batch_size, MAX_OUTPUTS)
            start = batch_size - count
            selected_indices = list(range(start, batch_size))
            
        elif mode == "specific":
            selected_indices = self.parse_indices(frame_indices, batch_size)
            
        elif mode == "every_nth":
            selected_indices = []
            idx = 0
            while idx < batch_size and len(selected_indices) < min(num_frames, MAX_OUTPUTS):
                selected_indices.append(idx)
                idx += step
        else:
            selected_indices = [0]
        
        # Fallback frame for unused outputs
        fallback_frame = images[0:1]
        
        # Build results
        results = []
        for i in range(MAX_OUTPUTS):
            if i < len(selected_indices):
                idx = selected_indices[i]
                frame = images[idx:idx+1]
            else:
                frame = fallback_frame
            results.append(frame)
        
        return tuple(results)


NODE_CLASS_MAPPINGS = {
    "CherryPickFrames": CherryPickFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CherryPickFrames": "Cherry Pick Frames",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
