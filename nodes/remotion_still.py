"""
Remotion Still - renders a single still frame from a Remotion
composition via npx remotion still.

Returns both the file path and the image as a ComfyUI IMAGE tensor.
"""

import os
import subprocess
from typing import Dict, Any, Tuple

from ..utils.remotion_utils import build_env, load_image_as_tensor


class RemotionStill:
    """
    Render a single still frame from a Remotion composition.

    Useful for thumbnails, preview frames, or single-frame
    composites.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "project": ("REMOTION_PROJECT",),
                "composition_id": ("STRING", {
                    "default": "MyComposition",
                    "tooltip": (
                        "The composition ID as defined in your "
                        "Remotion Root.tsx"
                    ),
                }),
            },
            "optional": {
                "frame": ("INT", {
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": (
                        "Which frame number to render as a still."
                    ),
                }),
                "props_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to props JSON file. Connect from "
                        "Remotion Build Props."
                    ),
                }),
                "format": (
                    ["png", "jpeg", "webp"], {"default": "png"},
                ),
                "jpeg_quality": ("INT", {
                    "default": 95, "min": 1, "max": 100,
                    "tooltip": (
                        "Quality for jpeg/webp output."
                    ),
                }),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 7680,
                    "tooltip": (
                        "Override width. 0 = use composition "
                        "default."
                    ),
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 4320,
                    "tooltip": (
                        "Override height. 0 = use composition "
                        "default."
                    ),
                }),
                "output_filename": ("STRING", {
                    "default": "still_output",
                    "tooltip": (
                        "Output filename without extension."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "output_path")
    FUNCTION = "render_still"
    CATEGORY = "Trent/Remotion Get Down"
    DESCRIPTION = (
        "Renders a single still frame from a Remotion "
        "composition. Returns the image as a ComfyUI tensor "
        "and the file path."
    )

    def render_still(
        self,
        project: dict,
        composition_id: str,
        frame: int = 0,
        props_path: str = "",
        format: str = "png",
        jpeg_quality: int = 95,
        width: int = 0,
        height: int = 0,
        output_filename: str = "still_output",
    ) -> Tuple:
        ext_map = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
        ext = ext_map[format]
        output_path = os.path.join(
            project["out_dir"], f"{output_filename}{ext}"
        )

        entry_point = os.path.join(
            project["project_path"], "src", "index.ts"
        )

        cmd = [
            project["npx_executable"],
            "remotion", "still",
            entry_point,
            composition_id,
            output_path,
            "--frame", str(frame),
            "--image-format", format,
        ]

        if format in ("jpeg", "webp"):
            cmd.extend(["--jpeg-quality", str(jpeg_quality)])
        if props_path and os.path.isfile(props_path):
            cmd.extend(["--props", props_path])
        if width > 0:
            cmd.extend(["--width", str(width)])
        if height > 0:
            cmd.extend(["--height", str(height)])

        env = build_env(project)

        try:
            proc = subprocess.run(
                cmd,
                cwd=project["project_path"],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Remotion still render timed out after 120s."
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Remotion still render failed:\n"
                f"{proc.stderr[-2000:]}"
            )

        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"Still render succeeded but output not found: "
                f"{output_path}"
            )

        image_tensor = load_image_as_tensor(output_path)

        print(
            f"[Remotion Get Down] Still rendered: {output_path}"
        )

        return (image_tensor, output_path)


NODE_CLASS_MAPPINGS = {
    "RemotionStill": RemotionStill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionStill": "Remotion Still",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
