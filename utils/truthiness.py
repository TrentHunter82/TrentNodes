"""
Truthiness utilities for conditional flow nodes.

Defines what constitutes a 'truthy' or 'falsy' value in the context
of ComfyUI node execution for conditional gating.
"""
import torch
from typing import Any


def is_truthy(value: Any) -> bool:
    """
    Determine if a value is truthy for conditional execution.

    FALSY values (return False):
        - None
        - 0 (int) or 0.0 (float)
        - Empty string ""
        - Empty list, tuple, or dict
        - Empty tensor (numel() == 0)
        - All-zero tensor

    TRUTHY values (return True):
        - Everything else (non-empty containers, non-zero numbers,
          tensors with non-zero elements, model objects, etc.)

    Args:
        value: Any value to check for truthiness

    Returns:
        bool: True if value is truthy, False if falsy
    """
    # None is always falsy
    if value is None:
        return False

    # Handle booleans explicitly (before numeric check)
    if isinstance(value, bool):
        return value

    # Handle tensors (IMAGE, MASK, LATENT samples, etc.)
    if isinstance(value, torch.Tensor):
        # Empty tensor is falsy
        if value.numel() == 0:
            return False
        # All-zero tensor is falsy (GPU-efficient check)
        return bool(value.any().item())

    # Handle numeric types
    if isinstance(value, (int, float)):
        return value != 0

    # Handle string
    if isinstance(value, str):
        return len(value) > 0

    # Handle LATENT type (dict with "samples" key)
    if isinstance(value, dict) and "samples" in value:
        return is_truthy(value["samples"])

    # Handle other containers (list, tuple, dict)
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0

    # Everything else (MODEL, CLIP, VAE, CONDITIONING, etc.) is truthy
    return True


def is_falsy(value: Any) -> bool:
    """
    Convenience function - inverse of is_truthy.

    Args:
        value: Any value to check

    Returns:
        bool: True if value is falsy, False if truthy
    """
    return not is_truthy(value)
