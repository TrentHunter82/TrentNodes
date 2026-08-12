"""
Ref Preview Cowboy - None-tolerant multi-image preview.

Companion to RefFolderCowboy: wire all six image slots straight in.
Slots that carry None (empty ref slots) are silently skipped, so an
empty or partial ref folder never aborts the run - the node just
previews whatever images actually exist, in slot order.
"""

from typing import Any, Dict

import nodes

NUM_SLOTS = 6


class RefPreviewCowboy(nodes.PreviewImage):
    """
    Preview up to six IMAGE inputs, skipping any that are None.

    Every input is optional, and images are saved one slot at a time,
    so mixed sizes are fine and missing slots cost nothing.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                f"image_{i + 1}": ("IMAGE", {
                    "tooltip": (
                        f"Image slot {i + 1} - previewed when "
                        "present, skipped when None"
                    ),
                })
                for i in range(NUM_SLOTS)
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # PreviewImage inherits SaveImage's ("IMAGE",)/("images",) passthrough, but
    # preview() returns no "result", so clear both or /object_info reports a
    # named output slot that does not exist.
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Preview up to six images, silently skipping unconnected or "
        "None inputs. Wire all six RefFolderCowboy slots in once: "
        "empty ref slots pass None and never abort the workflow."
    )

    def preview(self, prompt=None, extra_pnginfo=None, **kwargs):
        shown = []
        for i in range(NUM_SLOTS):
            img = kwargs.get(f"image_{i + 1}")
            if img is None:
                continue
            res = self.save_images(
                img,
                filename_prefix="RefPreviewCowboy",
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )
            shown.extend(res["ui"]["images"])
        return {"ui": {"images": shown}}


NODE_CLASS_MAPPINGS = {
    "RefPreviewCowboy": RefPreviewCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RefPreviewCowboy": "Ref Preview Cowboy",
}
