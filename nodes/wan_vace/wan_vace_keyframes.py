"""
Wan Vace Keyframe Sequence Builder
A ComfyUI custom node for creating keyframe sequences for Wan Vace video generation.
"""

import torch
from typing import Dict, Any, Tuple

class WanVaceKeyframeBuilder:
    """
    Builds keyframe sequences for Wan Vace model.
    
    Dynamically adds image inputs as you connect them.
    Each image can be positioned at any frame using its slider.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frame_count": ("INT", {
                    "default": 16, 
                    "min": 2, 
                    "max": 256, 
                    "step": 1,
                    "tooltip": "Total number of frames in the output sequence"
                }),
            },
            "optional": {
                "default_width": ("INT", {
                    "default": 832,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Width for filler frames (used if no images connected)"
                }),
                "default_height": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Height for filler frames (used if no images connected)"
                }),
                "gray_value": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Gray level for filler frames (0=black, 1=white)"
                }),
                # First dynamic image input - JS will add more as needed
                "image_1": ("IMAGE",),
                "image_1_frame": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Frame position for this image (1 = first frame)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "Batch of frames: keyframes at specified positions, gray filler elsewhere",
        "Batch of masks: white (1.0) for keyframes, black (0.0) for filler frames"
    )
    
    FUNCTION = "build_sequence"
    CATEGORY = "Trent/Keyframes"
    DESCRIPTION = "Creates keyframe image and mask sequences for Wan Vace video generation."
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept any dynamically added inputs
        return True
    
    def build_sequence(
        self,
        frame_count: int,
        default_width: int = 832,
        default_height: int = 480,
        gray_value: float = 0.5,
        unique_id: str = None,
        prompt: dict = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the keyframe sequence and matching mask batch.
        """
        
        # Get widget values from the prompt data for this node
        # This is how we access dynamically-added widget values
        widget_values = {}
        if prompt and unique_id:
            node_data = prompt.get(unique_id, {})
            inputs = node_data.get("inputs", {})
            widget_values = inputs
            print(f"[WanVaceKeyframeBuilder] Node {unique_id} inputs from prompt: {list(inputs.keys())}")
        
        # Collect all connected images and their target frame positions
        keyframes: Dict[int, torch.Tensor] = {}
        
        # Find all image inputs in kwargs (image_1, image_2, etc.)
        for key, value in kwargs.items():
            # Match image_N pattern (but not image_N_frame)
            if key.startswith("image_") and "_frame" not in key and value is not None:
                try:
                    parts = key.split("_")
                    if len(parts) == 2 and parts[1].isdigit():
                        idx = int(parts[1])
                        
                        # Get the corresponding frame position from widget_values (prompt data)
                        frame_key = f"image_{idx}_frame"
                        
                        # First try widget_values from prompt, then kwargs, then default to idx
                        frame_pos = widget_values.get(frame_key, kwargs.get(frame_key, idx))
                        
                        # Ensure frame_pos is an int
                        if isinstance(frame_pos, (int, float)):
                            frame_pos = int(frame_pos)
                        else:
                            frame_pos = idx
                        
                        # Clamp to valid range (1-indexed input, convert to 0-indexed)
                        frame_idx = max(0, min(frame_count - 1, frame_pos - 1))
                        
                        print(f"[WanVaceKeyframeBuilder] {key} -> frame {frame_pos} (0-indexed: {frame_idx})")
                        
                        # Store (if multiple images target same frame, later ones win)
                        keyframes[frame_idx] = value
                except (ValueError, IndexError) as e:
                    print(f"[WanVaceKeyframeBuilder] Error parsing {key}: {e}")
                    continue
        
        print(f"[WanVaceKeyframeBuilder] Keyframes at positions: {sorted(keyframes.keys())}")
        
        # Determine output dimensions from first available keyframe
        if keyframes:
            first_frame_idx = min(keyframes.keys())
            first_img = keyframes[first_frame_idx]
            h = first_img.shape[1]
            w = first_img.shape[2]
            c = first_img.shape[3] if len(first_img.shape) > 3 else 3
            device = first_img.device
            dtype = first_img.dtype
        else:
            h, w, c = default_height, default_width, 3
            device = torch.device('cpu')
            dtype = torch.float32
        
        # Create templates
        gray_frame = torch.full((1, h, w, c), gray_value, dtype=dtype, device=device)
        white_mask = torch.ones((1, h, w), dtype=dtype, device=device)
        black_mask = torch.zeros((1, h, w), dtype=dtype, device=device)
        
        # Build the output sequences
        image_list = []
        mask_list = []
        
        for frame_idx in range(frame_count):
            if frame_idx in keyframes:
                img = keyframes[frame_idx]
                
                # Take first frame if input is batched
                if img.shape[0] > 1:
                    img = img[0:1]
                
                # Resize if dimensions don't match
                if img.shape[1] != h or img.shape[2] != w:
                    img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW
                    img = torch.nn.functional.interpolate(
                        img, 
                        size=(h, w), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    img = img.permute(0, 2, 3, 1)  # BCHW -> BHWC
                
                image_list.append(img)
                mask_list.append(white_mask.clone())
            else:
                image_list.append(gray_frame.clone())
                mask_list.append(black_mask.clone())
        
        # Concatenate into batches
        images_out = torch.cat(image_list, dim=0)
        masks_out = torch.cat(mask_list, dim=0)
        
        print(f"[WanVaceKeyframeBuilder] Output shapes: images={images_out.shape}, masks={masks_out.shape}")
        
        return (images_out, masks_out)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "WanVaceKeyframeBuilder": WanVaceKeyframeBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVaceKeyframeBuilder": "Wan Vace Keyframe Builder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
