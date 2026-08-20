"""
Cowboy Multi-Line String Concatenate -- join strings with newlines.

Two flavours of the same rope trick:

* CowboyMultiLineStringConcatenate - ten fixed string slots. Type into
  them or wire them up; each filled slot becomes one line.
* CowboyMultiLineStringConcatenateDynamic - starts with one input and
  grows a fresh socket every time you connect the last one. The UI for
  that lives in ../js/multiline_concat_cowboy.js.

Empty slots are skipped by default so a half-filled node doesn't leave
blank lines in the middle of the text.
"""
from ..utils.any_type import any_typ

SLOT_COUNT = 10


def join_lines(values, skip_empty=True):
    """
    Turn an ordered list of values into one newline-joined string.

    A list value contributes one line per element, so upstream list
    outputs stack naturally instead of printing like "['a', 'b']".
    """
    lines = []
    for value in values:
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            text = item if isinstance(item, str) else str(item)
            if skip_empty and not text.strip():
                continue
            lines.append(text)
    return "\n".join(lines)


class CowboyMultiLineStringConcatenate:
    """Join up to ten strings, one per line."""

    CATEGORY = "Trent/Text"
    DISPLAY_NAME = "Cowboy Multi-Line String Concatenate"
    FUNCTION = "concat"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("All the filled slots joined with newlines.",)

    DESCRIPTION = (
        "Joins up to ten strings into one string.\n"
        "Each filled slot becomes one line.\n"
        "Type into a slot, or connect a string to it.\n"
        "An empty slot is skipped when skip_empty is on."
    )

    @classmethod
    def INPUT_TYPES(cls):
        slots = {
            f"string_{i}": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": f"line {i}",
                "tooltip": f"Slot {i}. One line of the output.",
            })
            for i in range(1, SLOT_COUNT + 1)
        }
        return {
            "required": {},
            "optional": {
                **slots,
                "skip_empty": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On - an empty slot adds no line.\n"
                               "Off - an empty slot adds a blank line.",
                }),
            },
        }

    def concat(self, skip_empty=True, **kwargs):
        values = [kwargs.get(f"string_{i}") for i in range(1, SLOT_COUNT + 1)]
        return (join_lines(values, skip_empty),)


class CowboyMultiLineStringConcatenateDynamic:
    """Join any number of connected values, one per line."""

    CATEGORY = "Trent/Text"
    DISPLAY_NAME = "Cowboy Multi-Line String Concatenate (Dynamic)"
    FUNCTION = "concat"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("All the connected inputs joined with newlines.",)

    DESCRIPTION = (
        "Joins the connected inputs into one string.\n"
        "Each input becomes one line.\n"
        "Connect the last input and a new one appears.\n"
        "A list input becomes one line per item."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "skip_empty": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On - an empty input adds no line.\n"
                               "Off - an empty input adds a blank line.",
                }),
                # First dynamic input - the JS extension adds more.
                "string_1": (any_typ,),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Inputs added by the JS extension are not in INPUT_TYPES;
        # accept them all rather than fail the stock check.
        return True

    def concat(self, skip_empty=True, **kwargs):
        indexed = []
        for key, value in kwargs.items():
            if not key.startswith("string_") or value is None:
                continue
            try:
                indexed.append((int(key.split("_")[1]), value))
            except (ValueError, IndexError):
                continue
        indexed.sort(key=lambda pair: pair[0])
        return (join_lines([value for _, value in indexed], skip_empty),)


NODE_CLASS_MAPPINGS = {
    "CowboyMultiLineStringConcatenate": CowboyMultiLineStringConcatenate,
    "CowboyMultiLineStringConcatenateDynamic":
        CowboyMultiLineStringConcatenateDynamic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CowboyMultiLineStringConcatenate":
        "Cowboy Multi-Line String Concatenate",
    "CowboyMultiLineStringConcatenateDynamic":
        "Cowboy Multi-Line String Concatenate (Dynamic)",
}
