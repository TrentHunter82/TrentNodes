"""
Round Up -- Ceiling to the nearest whole number.

Takes a float and always rounds up (math.ceil).
Outputs the result as both INT and FLOAT so it can
feed either socket type downstream.
"""

import math


class RoundUp:
    """
    Round a float up to the nearest whole number and
    output it as both an int and a float.
    """

    CATEGORY = "Trent/Utils"
    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int", "float")
    OUTPUT_TOOLTIPS = (
        "ceil(value) as an integer",
        "ceil(value) as a float",
    )
    FUNCTION = "compute"
    DESCRIPTION = (
        "Always round up to the nearest whole number. "
        "Outputs the same value as both INT and FLOAT."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": -999999.0,
                    "max": 999999.0,
                    "step": 0.01,
                    "tooltip": "Input number to round up",
                    "forceInput": True,
                }),
            },
        }

    def compute(self, value):
        result = math.ceil(float(value))
        return (int(result), float(result))


NODE_CLASS_MAPPINGS = {
    "RoundUp": RoundUp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RoundUp": "Round Up",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
