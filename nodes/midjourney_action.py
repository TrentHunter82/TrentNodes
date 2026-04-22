"""
Midjourney Action nodes (Yunwu API).

MidjourneyAction       -- one upscale or variation (U1..U4 / V1..V4)
MidjourneyBatchActions -- all four upscales OR all four variations
                          run concurrently via a thread pool.

Both consume the (task_id, buttons) outputs from MidjourneyImagine.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.midjourney_client import (
    MJClient,
    resolve_credentials,
    url_to_image_tensor,
)


ACTIONS_SINGLE = ["U1", "U2", "U3", "U4", "V1", "V2", "V3", "V4"]
BATCH_MODES = ["U1-U4", "V1-V4"]


def _resolve_custom_id(buttons, label):
    """Look up a button label in the buttons dict; raise if missing."""
    if not isinstance(buttons, dict):
        raise RuntimeError(
            "buttons input is not a dict (got "
            + type(buttons).__name__ + "). Wire the `buttons` "
            "output of MidjourneyImagine into this node."
        )
    custom_id = buttons.get(label)
    if not custom_id:
        available = ", ".join(sorted(buttons.keys())) or "(none)"
        raise RuntimeError(
            "Midjourney button '" + label + "' not found in the "
            "buttons map. Available: " + available
        )
    return custom_id


def _run_one_action(client, task_id, label, buttons):
    """Submit one action and download its result image."""
    custom_id = _resolve_custom_id(buttons, label)
    result = client.action(task_id, custom_id)
    image_url = result.get("image_url")
    if not image_url:
        raise RuntimeError(
            "Midjourney action " + label + " returned no image URL"
        )
    return url_to_image_tensor(image_url)


class MidjourneyAction:
    """Run one upscale (U1-U4) or variation (V1-V4) on a prior task."""

    CATEGORY = "Trent Tools/Midjourney"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    DESCRIPTION = (
        "Apply a single Midjourney upscale or variation to a "
        "previous Imagine task using its buttons map."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": (
                    "STRING",
                    {"forceInput": True},
                ),
                "buttons": (
                    "DICT",
                    {"forceInput": True},
                ),
                "action": (
                    ACTIONS_SINGLE,
                    {"default": "U1"},
                ),
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
            },
        }

    def run(
        self,
        task_id,
        buttons,
        action,
        api_key,
        api_url="https://yunwu.ai",
    ):
        key, url = resolve_credentials(api_key, api_url)
        client = MJClient(api_key=key, api_url=url)
        image = _run_one_action(client, task_id, action, buttons)
        return (image,)


class MidjourneyBatchActions:
    """Run all four upscales OR all four variations concurrently."""

    CATEGORY = "Trent Tools/Midjourney"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("slot1", "slot2", "slot3", "slot4")
    FUNCTION = "run"
    DESCRIPTION = (
        "Run all four Midjourney upscales (U1-U4) or all four "
        "variations (V1-V4) in parallel and return them as four "
        "separate IMAGE outputs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": (
                    "STRING",
                    {"forceInput": True},
                ),
                "buttons": (
                    "DICT",
                    {"forceInput": True},
                ),
                "mode": (
                    BATCH_MODES,
                    {"default": "U1-U4"},
                ),
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
            },
        }

    def run(
        self,
        task_id,
        buttons,
        mode,
        api_key,
        api_url="https://yunwu.ai",
    ):
        key, url = resolve_credentials(api_key, api_url)
        client = MJClient(api_key=key, api_url=url)

        if mode == "U1-U4":
            labels = ["U1", "U2", "U3", "U4"]
        else:
            labels = ["V1", "V2", "V3", "V4"]

        results = [None, None, None, None]
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_to_idx = {
                pool.submit(
                    _run_one_action,
                    client,
                    task_id,
                    label,
                    buttons,
                ): idx
                for idx, label in enumerate(labels)
            }
            try:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
            except Exception:
                # Cancel anything still pending so we surface the
                # original error promptly instead of waiting for
                # the rest of the batch to finish.
                for f in future_to_idx:
                    f.cancel()
                raise

        return tuple(results)
