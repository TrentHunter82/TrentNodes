"""
Int Offset Clamp -- Simple integer math helper.

Computes max(value + offset, min_value). Designed for
driving frame indices in iterative video workflows,
e.g. computing the "previous frame" index as
max(current_frame - 1, 0).
"""


class IntOffsetClamp:
    """
    Add an offset (can be negative) to an integer and
    clamp the result to a minimum. Small helper for
    driving video frame index math in auto-queue
    iterative workflows.
    """

    CATEGORY = "Trent/Utils"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = (
        "max(value + offset, min_value)",
    )
    FUNCTION = "compute"
    DESCRIPTION = (
        "Add offset to value and clamp to a minimum. "
        "Default (offset=-1, min=0) computes the "
        "previous frame index safely for frame 0."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": -999999,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Input integer",
                    "forceInput": True,
                }),
                "offset": ("INT", {
                    "default": -1,
                    "min": -999999,
                    "max": 999999,
                    "step": 1,
                    "tooltip": (
                        "Amount to add to value. "
                        "Negative subtracts."
                    ),
                }),
                "min_value": ("INT", {
                    "default": 0,
                    "min": -999999,
                    "max": 999999,
                    "step": 1,
                    "tooltip": (
                        "Result will not go below "
                        "this value"
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def compute(self, value, offset, min_value):
        return (max(int(value) + int(offset), int(min_value)),)


NODE_CLASS_MAPPINGS = {
    "IntOffsetClamp": IntOffsetClamp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntOffsetClamp": "Int Offset Clamp",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
