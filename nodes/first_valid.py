"""
First Valid - Fallback chain node for ComfyUI.

Checks inputs in priority order and outputs the FIRST truthy value.
Uses lazy evaluation to avoid evaluating later inputs once a truthy is found.
"""
from typing import Any, Dict, List, Tuple

from comfy_execution.graph_utils import ExecutionBlocker

# Import from parent package utils
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.truthiness import is_truthy


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
                "first": ("*", {
                    "lazy": True,
                    "tooltip": "Highest priority input - checked first"
                }),
                "second": ("*", {
                    "lazy": True,
                    "tooltip": "Second priority - used if first is falsy"
                }),
                "third": ("*", {
                    "lazy": True,
                    "tooltip": "Third priority / fallback - used if first "
                               "and second are falsy"
                }),
            },
        }

    RETURN_TYPES = ("*",)
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
    ) -> List[str]:
        """
        Request inputs lazily in priority order.

        Only requests the next input if the current one is falsy.
        This is the key optimization - we avoid evaluating later
        inputs if an earlier one is already valid.
        """
        # Check first - if not evaluated yet, request it
        if first is None:
            return ["first"]

        # First was evaluated - if truthy, we're done (no more needed)
        if is_truthy(first):
            return []

        # First is falsy, check second
        if second is None:
            return ["second"]

        # Second was evaluated - if truthy, we're done
        if is_truthy(second):
            return []

        # Second is falsy, check third
        if third is None:
            return ["third"]

        # All evaluated, nothing more needed
        return []

    def select_first(
        self,
        first: Any = None,
        second: Any = None,
        third: Any = None,
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
