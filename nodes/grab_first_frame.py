"""
Grab First Frame - Extract the first frame from a batch of images.
"""

from typing import Dict, Any, Tuple

import torch


class GrabFirstFrame:
    """
    Takes a batch of images and returns only the first frame.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Batch of images"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "grab"
    CATEGORY = "Trent/Image"
    DESCRIPTION = "Returns the first frame from a batch of images."

    def grab(self, images: torch.Tensor) -> Tuple[torch.Tensor]:
        return (images[0:1],)


NODE_CLASS_MAPPINGS = {
    "GrabFirstFrame": GrabFirstFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrabFirstFrame": "Grab First Frame",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
