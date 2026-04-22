"""
Midjourney Imagine node (Yunwu API).

Submits a text prompt to /mj/submit/imagine, polls until SUCCESS,
and returns the 4-image grid as a ComfyUI IMAGE tensor along with
the task_id and a label->customId button map for downstream
upscale/variation nodes.
"""

from ..utils.midjourney_client import (
    MJClient,
    resolve_credentials,
    url_to_image_tensor,
)


def _build_prompt(
    prompt,
    base_model,
    version,
    aspect_ratio,
    stylize,
    chaos,
    weird,
    seed,
):
    """
    Append Midjourney CLI flags to a raw prompt string.

    Skips flags that are at their defaults to keep prompts clean.
    For base_model="niji", emits `--niji <version>` instead of `--v`.
    """
    parts = [prompt.strip()]

    if aspect_ratio and aspect_ratio != "1:1":
        parts.append("--ar " + aspect_ratio)
    if stylize is not None and stylize != 100:
        parts.append("--s " + str(stylize))
    if chaos:
        parts.append("--c " + str(chaos))
    if weird:
        parts.append("--w " + str(weird))

    if base_model == "niji":
        parts.append("--niji " + str(version))
    else:
        parts.append("--v " + str(version))

    if seed is not None and seed >= 0:
        parts.append("--seed " + str(seed))

    return " ".join(parts)


class MidjourneyImagine:
    """Text -> Midjourney 2x2 grid via the Yunwu MJ API."""

    CATEGORY = "Trent Tools/Midjourney"
    RETURN_TYPES = ("IMAGE", "STRING", "DICT")
    RETURN_NAMES = ("image", "task_id", "buttons")
    FUNCTION = "generate"
    DESCRIPTION = (
        "Submit a prompt to the Yunwu Midjourney API and return "
        "the resulting 2x2 grid image, the task_id, and a "
        "label->customId button map for upscale/variation nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a cinematic portrait",
                        "tooltip": (
                            "Plain prompt text. Flags like --ar "
                            "and --s are added automatically from "
                            "the widgets below."
                        ),
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "password": True,
                        "tooltip": (
                            "Yunwu API key. Leave blank to fall "
                            "back to the MJ_API_KEY env var. "
                            "WARNING: pasted keys are saved into "
                            "the workflow JSON."
                        ),
                    },
                ),
            },
            "optional": {
                "api_url": (
                    "STRING",
                    {
                        "default": "https://yunwu.ai",
                        "tooltip": (
                            "Yunwu API base URL. Override if your "
                            "provider uses a different host."
                        ),
                    },
                ),
                "base_model": (
                    ["midjourney", "niji"],
                    {"default": "midjourney"},
                ),
                "version": (
                    ["5.2", "6", "6.1", "7"],
                    {"default": "6.1"},
                ),
                "aspect_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16"],
                    {"default": "1:1"},
                ),
                "stylize": (
                    "INT",
                    {"default": 100, "min": 0, "max": 1000},
                ),
                "chaos": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100},
                ),
                "weird": (
                    "INT",
                    {"default": 0, "min": 0, "max": 3000},
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Use -1 for a random seed.",
                    },
                ),
            },
        }

    def generate(
        self,
        prompt,
        api_key,
        api_url="https://yunwu.ai",
        base_model="midjourney",
        version="6.1",
        aspect_ratio="1:1",
        stylize=100,
        chaos=0,
        weird=0,
        seed=-1,
    ):
        key, url = resolve_credentials(api_key, api_url)
        client = MJClient(api_key=key, api_url=url)

        full_prompt = _build_prompt(
            prompt=prompt,
            base_model=base_model,
            version=version,
            aspect_ratio=aspect_ratio,
            stylize=stylize,
            chaos=chaos,
            weird=weird,
            seed=seed,
        )

        result = client.imagine(full_prompt)
        image_url = result.get("image_url")
        if not image_url:
            raise RuntimeError(
                "Midjourney imagine returned no image URL: "
                + str(result.get("raw"))[:500]
            )

        image_tensor = url_to_image_tensor(image_url)
        return (image_tensor, result["task_id"], result["buttons"])
