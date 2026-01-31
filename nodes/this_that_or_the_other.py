"""
This, That, or The Other - Parallel gating node for ComfyUI.

Provides 3 independent input/output channels. Each input passes to its
corresponding output ONLY if truthy. Falsy inputs block downstream execution.
"""
from typing import Any, Dict, List, Tuple

from comfy_execution.graph_utils import ExecutionBlocker

# Import from parent package utils
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.truthiness import is_truthy


class ThisThatOrTheOther:
    """
    Parallel gating node with 3 independent input/output channels.

    Each input passes to its corresponding output ONLY if truthy.
    Falsy inputs result in ExecutionBlocker (downstream nodes don't execute).

    Use case: Conditionally route different data types through
    independent processing branches based on whether data is present.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "this": ("*", {
                    "lazy": True,
                    "tooltip": "First input - passes to this_out if truthy"
                }),
                "that": ("*", {
                    "lazy": True,
                    "tooltip": "Second input - passes to that_out if truthy"
                }),
                "the_other": ("*", {
                    "lazy": True,
                    "tooltip": "Third input - passes to the_other_out if truthy"
                }),
            },
        }

    RETURN_TYPES = ("*", "*", "*")
    RETURN_NAMES = ("this_out", "that_out", "the_other_out")

    FUNCTION = "gate"
    CATEGORY = "Trent/Flow"
    DESCRIPTION = (
        "Parallel gate with 3 independent channels. "
        "Each input passes through only if truthy (non-None, non-zero, "
        "non-empty). Falsy inputs block their downstream path."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> bool:
        """Accept any input types."""
        return True

    def check_lazy_status(
        self,
        this: Any = None,
        that: Any = None,
        the_other: Any = None,
    ) -> List[str]:
        """
        Request evaluation of connected inputs.

        Returns list of input names that need to be evaluated.
        For this node, we request all inputs since each channel is
        independent and we need to evaluate all of them.
        """
        needed = []

        # Request each input that hasn't been evaluated yet
        if this is None:
            needed.append("this")
        if that is None:
            needed.append("that")
        if the_other is None:
            needed.append("the_other")

        return needed

    def gate(
        self,
        this: Any = None,
        that: Any = None,
        the_other: Any = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Gate each input independently based on truthiness.

        Returns:
            Tuple of (this_out, that_out, the_other_out)
            Each output is either the input value or ExecutionBlocker
        """
        # Gate each channel independently
        this_out = this if is_truthy(this) else ExecutionBlocker(None)
        that_out = that if is_truthy(that) else ExecutionBlocker(None)
        the_other_out = (
            the_other if is_truthy(the_other) else ExecutionBlocker(None)
        )

        return (this_out, that_out, the_other_out)


NODE_CLASS_MAPPINGS = {
    "ThisThatOrTheOther": ThisThatOrTheOther,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThisThatOrTheOther": "This, That, or The Other",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
