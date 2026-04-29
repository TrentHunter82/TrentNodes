"""
Helpers for ComfyUI lazy-evaluation nodes.

ComfyUI's executor calls check_lazy_status to learn which inputs to
evaluate next. If that list contains an unconnected slot,
make_input_strong_link raises NodeInputError. connected_inputs inspects
the prompt graph (via the hidden UNIQUE_ID and DYNPROMPT inputs) so
nodes can request only inputs that actually have a link.
"""
from typing import Any, Iterable


def connected_inputs(
    unique_id: Any,
    dynprompt: Any,
    all_input_names: Iterable[str],
) -> set:
    """
    Return the subset of input names that have a link in the prompt
    graph. Falls back to all names when the hidden inputs are missing
    (e.g. when called outside a normal execution context).
    """
    if unique_id is None or dynprompt is None:
        return set(all_input_names)
    try:
        inputs = dynprompt.get_node(unique_id).get("inputs", {})
    except Exception:
        return set()
    return {
        name for name, val in inputs.items()
        if isinstance(val, list) and len(val) == 2
    }
