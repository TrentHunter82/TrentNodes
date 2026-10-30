"""
Grab Last Multiple Frames - Extract the last N frames from a batch of images.
"""

from typing import Dict, Any, Tuple

import torch


class GrabLastFrames:
    """
    Takes a batch of images and returns the last N frames, in order.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of images"
                }),
                "count": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 100000,
                    "tooltip": "How many frames to take from the end. "
                               "If the batch is shorter, the whole batch "
                               "is returned."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "grab"
    CATEGORY = "Trent/Image"
    DESCRIPTION = "Returns the last N frames from a batch of images."

    def grab(self, images: torch.Tensor, count: int) -> Tuple[torch.Tensor]:
        return (images[-count:],)


NODE_CLASS_MAPPINGS = {
    "GrabLastFrames": GrabLastFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrabLastFrames": "Grab Last Multiple Frames",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
