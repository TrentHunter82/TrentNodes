"""
Image Slice Cowboy — split an image (or batch) in half.

Takes a single image or a batch and cuts every frame with one straight line,
either vertical (left | right) or horizontal (top | bottom), emitting the two
halves as separate IMAGE outputs. A batch of N in -> two batches of N out.

The cut position is adjustable (default dead-center), so it doubles as a
generic "chop at X%" tool, not just a halver. Each half is returned contiguous
so downstream nodes get clean tensors. Slicing is dimension-only — no resizing,
resampling, or colour change happens to the pixels themselves.
"""

import torch


class ImageSliceCowboy:
    """Split an image / image batch into two halves along one axis."""

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("left_or_top", "right_or_bottom", "size_a", "size_b")
    FUNCTION = "slice"
    CATEGORY = "Trent/Image"

    DESCRIPTION = """Image Slice Cowboy — split an image or batch in half.

Cuts every frame with one straight line and outputs the two halves separately:
  * vertical   -> left_or_top = left half,  right_or_bottom = right half
  * horizontal -> left_or_top = top half,   right_or_bottom = bottom half

A batch of N frames in produces two batches of N frames out (each half keeps
the full batch). split_position lets you slice anywhere from 1%..99% across the
chosen axis, so it works as a "chop at X%" tool too, not just a 50/50 halver.

size_a / size_b report the pixel width (vertical) or height (horizontal) of
each half — handy for re-joining or sanity checks. Use force_even to round the
cut to an even pixel so both halves stay model-friendly (divisible by 2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (
                    ["vertical", "horizontal"],
                    {
                        "default": "vertical",
                        "tooltip": "vertical = cut left|right, "
                                   "horizontal = cut top|bottom",
                    },
                ),
                "split_position": (
                    "FLOAT",
                    {
                        "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                        "tooltip": "Where to cut along the axis (0.5 = halfway)",
                    },
                ),
                "force_even": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Round the cut to an even pixel so both "
                                   "halves stay divisible by 2",
                    },
                ),
            },
        }

    def slice(self, images, direction, split_position, force_even):
        # images: [B, H, W, C] float 0-1
        batch, height, width, _ = images.shape
        axis_len = width if direction == "vertical" else height

        cut = int(round(axis_len * split_position))
        if force_even:
            cut -= cut % 2
        # Keep both halves non-empty even at extreme positions / tiny images.
        cut = max(1, min(cut, axis_len - 1))

        if direction == "vertical":
            first = images[:, :, :cut, :]
            second = images[:, :, cut:, :]
        else:  # horizontal
            first = images[:, :cut, :, :]
            second = images[:, cut:, :, :]

        size_a = cut
        size_b = axis_len - cut

        print(
            f"[Image Slice Cowboy] {direction} cut at {cut}px "
            f"({batch} frame(s), {width}x{height}) -> "
            f"{size_a}px + {size_b}px"
        )

        return (first.contiguous(), second.contiguous(), size_a, size_b)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "ImageSliceCowboy": ImageSliceCowboy
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSliceCowboy": "Image Slice Cowboy"
}
