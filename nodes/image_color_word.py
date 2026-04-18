"""Image Color Word - classify an image into a single color name."""

import torch


def _rgb_to_hsv(r: float, g: float, b: float):
    """Convert 0-1 RGB to HSV (h: 0-360, s/v: 0-1)."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    v = mx
    s = 0.0 if mx == 0 else d / mx
    if d == 0:
        h = 0.0
    elif mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * (((b - r) / d) + 2)
    else:
        h = 60 * (((r - g) / d) + 4)
    return h, s, v


def _classify_color(r: float, g: float, b: float) -> str:
    """Return a one-word color name for a 0-1 RGB triplet."""
    h, s, v = _rgb_to_hsv(r, g, b)

    # Neutrals
    if v < 0.12:
        return "black"
    if s < 0.12 and v > 0.9:
        return "white"
    if s < 0.15:
        return "gray"

    # Brown: warm hues with low value and decent saturation
    if (h < 40 or h > 330) and v < 0.45 and s > 0.25:
        return "brown"

    # Hue wheel buckets
    if h < 15 or h >= 345:
        return "red"
    if h < 40:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 170:
        return "green"
    if h < 200:
        return "cyan"
    if h < 255:
        return "blue"
    if h < 310:
        return "purple"
    return "pink"


class ImageColorWord:
    """Output a single-word color name describing the image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "method": (
                    ["mean", "median"],
                    {"default": "mean"},
                ),
                "ignore_extremes": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("color_word",)
    FUNCTION = "get_color_word"
    CATEGORY = "Trent/Image"

    def get_color_word(
        self,
        image,
        method: str = "mean",
        ignore_extremes: bool = False,
    ):
        # image shape: [B, H, W, C], float32 in [0, 1]
        img = image[0] if image.dim() == 4 else image
        flat = img.reshape(-1, img.shape[-1])[:, :3]

        if ignore_extremes:
            # Drop near-black and near-white pixels so tiny color
            # accents in a mostly neutral image still register.
            luma = flat.mean(dim=1)
            keep = (luma > 0.05) & (luma < 0.95)
            if keep.any():
                flat = flat[keep]

        if method == "median":
            rgb = flat.median(dim=0).values
        else:
            rgb = flat.mean(dim=0)

        r, g, b = (float(c) for c in rgb.cpu())
        word = _classify_color(r, g, b)
        return (word,)


NODE_CLASS_MAPPINGS = {
    "ImageColorWord": ImageColorWord,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageColorWord": "Image Color Word",
}
