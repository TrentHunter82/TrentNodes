"""
This, That, or The Other - Parallel gating node for ComfyUI.

Provides 3 independent input/output channels. Each input passes to its
corresponding output ONLY if truthy. Falsy inputs block downstream execution.
"""
from typing import Any, Dict, List, Tuple

from comfy_execution.graph import ExecutionBlocker

from ..utils.any_type import any_typ
from ..utils.lazy import connected_inputs
from ..utils.truthiness import is_truthy


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
                "this": (any_typ, {
                    "lazy": True,
                    "tooltip": "First input - passes to this_out if truthy"
                }),
                "that": (any_typ, {
                    "lazy": True,
                    "tooltip": "Second input - passes to that_out if truthy"
                }),
                "the_other": (any_typ, {
                    "lazy": True,
                    "tooltip": "Third input - passes to the_other_out if truthy"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = (any_typ, any_typ, any_typ)
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
        unique_id: str = None,
        dynprompt: Any = None,
    ) -> List[str]:
        """
        Request evaluation of connected inputs only.

        Requesting an unconnected input causes ComfyUI to raise
        NodeInputError in make_input_strong_link, so we inspect the
        prompt graph and skip unconnected slots.
        """
        connected = connected_inputs(
            unique_id, dynprompt, ("this", "that", "the_other")
        )
        return [
            name for name, value in (
                ("this", this), ("that", that), ("the_other", the_other)
            )
            if name in connected and value is None
        ]

    def gate(
        self,
        this: Any = None,
        that: Any = None,
        the_other: Any = None,
        **_hidden: Any,
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
