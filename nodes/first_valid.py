"""
First Valid - Fallback chain node for ComfyUI.

Checks inputs in priority order and outputs the FIRST truthy value.
Uses lazy evaluation to avoid evaluating later inputs once a truthy is found.
"""
from typing import Any, Dict, List, Tuple

from comfy_execution.graph import ExecutionBlocker

from ..utils.any_type import any_typ
from ..utils.lazy import connected_inputs
from ..utils.truthiness import is_truthy


class FirstValid:
    """
    Fallback chain node - outputs the FIRST truthy value.

    Checks inputs in priority order: first -> second -> third
    Returns the first truthy value found, or blocks if all falsy.

    Uses lazy evaluation to avoid evaluating later inputs
    if an earlier one is already truthy.

    Use case: Provide fallback values, like default images,
    prompts, or parameters when primary sources are unavailable.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "first": (any_typ, {
                    "lazy": True,
                    "tooltip": "Highest priority input - checked first"
                }),
                "second": (any_typ, {
                    "lazy": True,
                    "tooltip": "Second priority - used if first is falsy"
                }),
                "third": (any_typ, {
                    "lazy": True,
                    "tooltip": "Third priority / fallback - used if first "
                               "and second are falsy"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("output",)

    FUNCTION = "select_first"
    CATEGORY = "Trent/Flow"
    DESCRIPTION = (
        "Returns the first truthy input value. "
        "Checks first -> second -> third in priority order. "
        "Skips evaluation of later inputs once a truthy value is found."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> bool:
        """Accept any input types."""
        return True

    def check_lazy_status(
        self,
        first: Any = None,
        second: Any = None,
        third: Any = None,
        unique_id: str = None,
        dynprompt: Any = None,
    ) -> List[str]:
        """
        Request inputs lazily in priority order.

        Only requests inputs that are actually connected. Requesting an
        unconnected input causes ComfyUI to raise NodeInputError in
        make_input_strong_link.
        """
        connected = connected_inputs(
            unique_id, dynprompt, ("first", "second", "third")
        )

        for name, value in (
            ("first", first), ("second", second), ("third", third)
        ):
            if name not in connected:
                continue
            if value is None:
                return [name]
            if is_truthy(value):
                return []

        return []

    def select_first(
        self,
        first: Any = None,
        second: Any = None,
        third: Any = None,
        **_hidden: Any,
    ) -> Tuple[Any]:
        """
        Return the first truthy value, or block if all falsy.

        Returns:
            Tuple containing the first truthy input, or ExecutionBlocker
        """
        # Check in priority order
        if is_truthy(first):
            return (first,)
        if is_truthy(second):
            return (second,)
        if is_truthy(third):
            return (third,)

        # All falsy - block execution
        return (ExecutionBlocker(None),)


NODE_CLASS_MAPPINGS = {
    "FirstValid": FirstValid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FirstValid": "First Valid",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
