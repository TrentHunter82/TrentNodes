"""
Wan Vace nodes for ComfyUI
"""
from .wan_vace_keyframes import (
    NODE_CLASS_MAPPINGS as _keyframe_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as _keyframe_names,
)
from .vace_mask_autocomping import (
    NODE_CLASS_MAPPINGS as _autocomp_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as _autocomp_names,
)

NODE_CLASS_MAPPINGS = {**_keyframe_mappings, **_autocomp_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_keyframe_names,
    **_autocomp_names,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
