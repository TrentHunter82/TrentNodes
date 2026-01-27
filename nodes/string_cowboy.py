"""
StringListCowboy - Make a list of strings with optional prepend/append.

Lassos strings together into a list and brands them with prefix/suffix text.
Dynamic inputs expand as you connect more values.
"""


class AnyType(str):
    """Wildcard type that matches any input type."""

    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class StringListCowboy:
    """Make a list of strings with optional prepend/append to each."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prefix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to add before each string"
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to add after each string"
                }),
                "limit": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "tooltip": (
                        "Max items to output (0 = no limit)"
                    ),
                }),
                # First dynamic input - JS will add more as needed
                "value_1": (any_typ,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "wrangle"
    CATEGORY = "Trent/Text"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept any dynamically added inputs
        return True

    def wrangle(self, prefix="", suffix="", limit=0, **kwargs):
        # Collect all connected value inputs in order
        value_items = []
        for key, val in kwargs.items():
            if key.startswith("value_") and val is not None:
                try:
                    idx = int(key.split("_")[1])
                    value_items.append((idx, val))
                except (ValueError, IndexError):
                    continue

        # Sort by index to maintain order
        value_items.sort(key=lambda x: x[0])

        # Build string list
        strings = []
        for idx, val in value_items:
            if isinstance(val, list):
                strings.extend(str(v) for v in val)
            else:
                strings.append(str(val))

        # Apply prefix/suffix to each string
        result = []
        for text in strings:
            result.append(f"{prefix}{text}{suffix}")

        if limit > 0:
            result = result[:limit]

        return (result,)


NODE_CLASS_MAPPINGS = {
    "StringListCowboy": StringListCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringListCowboy": "String List Cowboy",
}
