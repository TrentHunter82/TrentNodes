"""
MiniMax H3 Resize -- Resize an image to a MiniMax H3 output size.

Sizes come from the H3 size settings reference table (16:9, all
dimensions multiples of 32). Outputs the resized image plus the
chosen width and height as INTs for wiring into other nodes.
"""

import comfy.utils

# megapixels label -> (width, height), from the H3 size settings reference
H3_SIZES = {
    "0.2 MP - 608x352": (608, 352),
    "0.3 MP - 736x416": (736, 416),
    "0.4 MP - 864x480": (864, 480),
    "0.5 MP - 960x544": (960, 544),
    "0.6 MP - 1056x608": (1056, 608),
    "0.7 MP - 1152x640": (1152, 640),
    "0.8 MP - 1216x672": (1216, 672),
    "0.9 MP - 1280x736": (1280, 736),
    "0.98 MP - 1344x768": (1344, 768),
    "1.0 MP - 1376x768": (1376, 768),
    "1.2 MP - 1504x832": (1504, 832),
    "1.5 MP - 1664x928": (1664, 928),
    "1.8 MP - 1824x1024": (1824, 1024),
    "2.0 MP - 1920x1088": (1920, 1088),
}


class MiniMaxH3Resize:
    """
    Resize an image to one of the MiniMax H3 reference output
    sizes and expose the width/height as INT outputs.
    """

    CATEGORY = "Trent/Image"
    DISPLAY_NAME = "MiniMax H3 Resize"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    DESCRIPTION = (
        "Resize an image to a MiniMax H3 output size (16:9 or "
        "9:16, multiple of 32). Outputs the resized image plus "
        "the width and height."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": (list(H3_SIZES.keys()), {
                    "default": "1.0 MP - 1376x768",
                    "tooltip": "Target H3 output size (megapixels - WxH).",
                }),
                "upscale_method": (
                    ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                    {"default": "lanczos"},
                ),
                "crop": (["center", "disabled"], {
                    "default": "center",
                    "tooltip": (
                        "center: crop to the target aspect before the "
                        "resize (no distortion). disabled: stretch to fit."
                    ),
                }),
                "orientation": (["landscape", "portrait"], {
                    "default": "landscape",
                    "tooltip": (
                        "landscape: use the size as listed (16:9). "
                        "portrait: swap width and height (9:16)."
                    ),
                }),
            }
        }

    def resize(self, image, size, upscale_method, crop, orientation):
        width, height = H3_SIZES[size]
        if orientation == "portrait":
            width, height = height, width
        samples = image.movedim(-1, 1)  # NHWC -> NCHW for common_upscale
        samples = comfy.utils.common_upscale(
            samples, width, height, upscale_method, crop
        )
        return (samples.movedim(1, -1), width, height)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Resize": MiniMaxH3Resize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Resize": "MiniMax H3 Resize",
}
