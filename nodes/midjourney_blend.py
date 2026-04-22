"""
Midjourney Blend node (Yunwu API).

Takes two ComfyUI IMAGE tensors, encodes them as base64 PNGs, and
submits them to /mj/submit/blend. Returns the blended image plus the
new task_id.
"""

from ..utils.midjourney_client import (
    MJClient,
    resolve_credentials,
    tensor_to_base64_png,
    url_to_image_tensor,
)


DIMENSIONS = ["SQUARE", "PORTRAIT", "LANDSCAPE"]
BOT_TYPES = ["MID_JOURNEY", "NIJI_JOURNEY"]


class MidjourneyBlend:
    """Mix two images via the Midjourney Blend endpoint."""

    CATEGORY = "Trent Tools/Midjourney"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "task_id")
    FUNCTION = "blend"
    DESCRIPTION = (
        "Blend two images using Midjourney's /mj/submit/blend "
        "endpoint and return the merged result."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "api_key": (
                    "STRING",
                    {"default": "", "password": True},
                ),
            },
            "optional": {
                "api_url": (
                    "STRING",
                    {"default": "https://yunwu.ai"},
                ),
                "dimensions": (
                    DIMENSIONS,
                    {"default": "SQUARE"},
                ),
                "base_model": (
                    BOT_TYPES,
                    {"default": "MID_JOURNEY"},
                ),
            },
        }

    def blend(
        self,
        image_a,
        image_b,
        api_key,
        api_url="https://yunwu.ai",
        dimensions="SQUARE",
        base_model="MID_JOURNEY",
    ):
        key, url = resolve_credentials(api_key, api_url)
        client = MJClient(api_key=key, api_url=url)

        b64_a = tensor_to_base64_png(image_a)
        b64_b = tensor_to_base64_png(image_b)

        result = client.blend(
            b64_images=[b64_a, b64_b],
            dimensions=dimensions,
            bot_type=base_model,
        )

        image_url = result.get("image_url")
        if not image_url:
            raise RuntimeError(
                "Midjourney blend returned no image URL: "
                + str(result.get("raw"))[:500]
            )

        image_tensor = url_to_image_tensor(image_url)
        return (image_tensor, result["task_id"])
