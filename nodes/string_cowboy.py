"""
StringCowboy - Prepend and/or append text to all strings in a list.

Lassos each string in a list and brands them with prefix and suffix text.
"""


class StringCowboy:
    """Prepend and/or append text to all strings in a list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"forceInput": True}),
                "mode": (["prepend", "append", "both"],),
            },
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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_list",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "garland"
    CATEGORY = "Trent/Text"

    def garland(self, string_list, mode, prefix=None, suffix=None):
        # Handle list inputs from ComfyUI
        mode = mode[0] if isinstance(mode, list) else mode
        prefix_text = ""
        suffix_text = ""

        if prefix:
            prefix_text = prefix[0] if isinstance(prefix, list) else prefix
        if suffix:
            suffix_text = suffix[0] if isinstance(suffix, list) else suffix

        result = []
        for s in string_list:
            text = str(s)
            if mode == "prepend":
                text = prefix_text + text
            elif mode == "append":
                text = text + suffix_text
            elif mode == "both":
                text = prefix_text + text + suffix_text
            result.append(text)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "StringCowboy": StringCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringCowboy": "String Cowboy",
}
