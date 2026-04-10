"""
Remotion Build Props - constructs a JSON props file from ComfyUI
inputs for Remotion CLI rendering.

Accepts up to 8 key-value pairs plus an optional base JSON string.
Writes the props to a file the Remotion CLI reads via --props.
"""

import json
import os
from typing import Dict, Any, Tuple

from ..utils.remotion_utils import auto_type


class RemotionBuildProps:
    """
    Build a JSON props file for Remotion rendering from ComfyUI
    widget inputs.

    Accepts a base JSON object, up to 8 key-value overrides,
    and optional typed inputs (frame_dir, frame_count) that
    wire directly from RemotionExportFrames.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs = {
            "required": {
                "project": ("REMOTION_PROJECT",),
            },
            "optional": {
                "base_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": (
                        "Base JSON object. Key-value pairs below "
                        "and connected inputs are merged on top."
                    ),
                }),
                "frame_dir": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Connect from Export Frames. Sets "
                        "props.frameDir automatically."
                    ),
                }),
                "frame_count": ("INT", {
                    "default": 0,
                    "forceInput": True,
                    "tooltip": (
                        "Connect from Export Frames. Sets "
                        "props.frameCount automatically."
                    ),
                }),
            },
        }

        for i in range(1, 9):
            inputs["optional"][f"key_{i}"] = ("STRING", {
                "default": "",
            })
            inputs["optional"][f"value_{i}"] = ("STRING", {
                "default": "",
            })

        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("props_path",)
    FUNCTION = "build_props"
    CATEGORY = "Trent/Remotion Get Down"
    DESCRIPTION = (
        "Builds a JSON props file for Remotion rendering. "
        "Accepts a base JSON, up to 8 key-value overrides, "
        "and optional frame_dir/frame_count inputs from "
        "Export Frames."
    )

    def build_props(
        self,
        project: dict,
        base_json: str = "{}",
        frame_dir: str = "",
        frame_count: int = 0,
        **kwargs,
    ) -> Tuple:
        try:
            props = json.loads(base_json)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid base JSON: {e}"
            )

        if not isinstance(props, dict):
            raise ValueError(
                "base_json must be a JSON object (dict), "
                f"got {type(props).__name__}"
            )

        # Inject connected typed inputs (override base_json)
        if frame_dir:
            props["frameDir"] = frame_dir
        if frame_count > 0:
            props["frameCount"] = frame_count

        for i in range(1, 9):
            key = kwargs.get(f"key_{i}", "")
            value = kwargs.get(f"value_{i}", "")
            if key.strip():
                props[key.strip()] = auto_type(value)

        props_path = os.path.join(
            project["project_path"], ".comfyui_props.json"
        )
        with open(props_path, "w", encoding="utf-8") as f:
            json.dump(props, f, indent=2)

        print(
            f"[Remotion Get Down] Props written to "
            f"{props_path} ({len(props)} keys)"
        )

        return (props_path,)


NODE_CLASS_MAPPINGS = {
    "RemotionBuildProps": RemotionBuildProps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionBuildProps": "Remotion Build Props",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
