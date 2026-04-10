"""
Remotion Render - the core render node that executes
npx remotion render as a subprocess.

Supports all major codecs, overrides for resolution/fps/duration,
and custom CLI flags.
"""

import os
import subprocess
from typing import Dict, Any, Tuple

from ..utils.remotion_utils import build_env


class RemotionRender:
    """
    Render a Remotion composition to video or image sequence
    via CLI subprocess.

    Blocks until rendering completes, then outputs the path to
    the rendered file and a log of stdout/stderr.
    """

    CODECS = [
        "h264", "h265", "vp8", "vp9",
        "prores", "gif", "png-sequence",
    ]
    PRORES_PROFILES = [
        "4444-xq", "4444", "hq", "standard", "light", "proxy",
    ]
    PIXEL_FORMATS = [
        "yuv420p", "yuva420p", "yuv422p",
        "yuv444p", "yuv420p10le",
    ]

    EXT_MAP = {
        "h264": ".mp4",
        "h265": ".mp4",
        "vp8": ".webm",
        "vp9": ".webm",
        "prores": ".mov",
        "gif": ".gif",
        "png-sequence": "",
    }

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
                "props_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to props JSON file. Connect from "
                        "Remotion Build Props."
                    ),
                }),
                "output_filename": ("STRING", {
                    "default": "render_output",
                    "tooltip": (
                        "Output filename without extension. "
                        "Extension is determined by codec."
                    ),
                }),
                "codec": (cls.CODECS, {"default": "h264"}),
                "prores_profile": (cls.PRORES_PROFILES, {
                    "default": "hq",
                    "tooltip": "Only used when codec is prores.",
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
                "fps": ("INT", {
                    "default": 0, "min": 0, "max": 120,
                    "tooltip": (
                        "Override FPS. 0 = use composition default."
                    ),
                }),
                "duration_frames": ("INT", {
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": (
                        "Override duration in frames. 0 = use "
                        "composition default."
                    ),
                }),
                "crf": ("INT", {
                    "default": 18, "min": 0, "max": 51,
                    "tooltip": (
                        "Constant Rate Factor for h264/h265. "
                        "Lower = better quality."
                    ),
                }),
                "pixel_format": (cls.PIXEL_FORMATS, {
                    "default": "yuv420p",
                }),
                "concurrency": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": (
                        "CPU threads for rendering. "
                        "0 = auto (half of available)."
                    ),
                }),
                "timeout_seconds": ("INT", {
                    "default": 600, "min": 30, "max": 7200,
                    "tooltip": (
                        "Maximum seconds to wait for render "
                        "to complete."
                    ),
                }),
                "extra_flags": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Additional CLI flags passed verbatim. "
                        "E.g., '--scale=2 --muted'"
                    ),
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "render_log")
    FUNCTION = "render"
    CATEGORY = "Trent/Remotion Get Down"
    DESCRIPTION = (
        "Renders a Remotion composition to video or image "
        "sequence via the Remotion CLI. Supports h264, h265, "
        "vp8, vp9, prores, gif, and png-sequence codecs."
    )

    def render(
        self,
        project: dict,
        composition_id: str,
        props_path: str = "",
        output_filename: str = "render_output",
        codec: str = "h264",
        prores_profile: str = "hq",
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        duration_frames: int = 0,
        crf: int = 18,
        pixel_format: str = "yuv420p",
        concurrency: int = 0,
        timeout_seconds: int = 600,
        extra_flags: str = "",
        verbose: bool = False,
    ) -> Tuple:
        is_sequence = codec == "png-sequence"
        ext = self.EXT_MAP.get(codec, ".mp4")

        if is_sequence:
            output_path = os.path.join(
                project["out_dir"], output_filename
            )
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = os.path.join(
                project["out_dir"],
                f"{output_filename}{ext}",
            )

        # CLI: npx remotion render <entry-point> <comp-id> <output>
        entry_point = os.path.join(
            project["project_path"], "src", "index.ts"
        )

        cmd = [
            project["npx_executable"],
            "remotion", "render",
            entry_point,
            composition_id,
            output_path,
        ]

        if props_path and os.path.isfile(props_path):
            cmd.extend(["--props", props_path])

        if width > 0:
            cmd.extend(["--width", str(width)])
        if height > 0:
            cmd.extend(["--height", str(height)])
        if fps > 0:
            cmd.extend(["--fps", str(fps)])
        if duration_frames > 0:
            cmd.extend([
                "--duration-in-frames", str(duration_frames),
            ])

        if is_sequence:
            cmd.extend(["--codec", "png"])
            cmd.append("--sequence")
        else:
            cmd.extend(["--codec", codec])

        if codec == "prores":
            cmd.extend(["--prores-profile", prores_profile])

        if codec in ("h264", "h265"):
            cmd.extend(["--crf", str(crf)])

        cmd.extend(["--pixel-format", pixel_format])

        if concurrency > 0:
            cmd.extend(["--concurrency", str(concurrency)])

        if verbose:
            cmd.extend(["--log", "verbose"])

        if extra_flags.strip():
            cmd.extend(extra_flags.strip().split())

        env = build_env(project)

        if verbose:
            print(
                f"[Remotion Get Down] Running: {' '.join(cmd)}"
            )

        try:
            proc = subprocess.run(
                cmd,
                cwd=project["project_path"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Remotion render timed out after "
                f"{timeout_seconds}s. Increase timeout_seconds "
                f"or simplify the composition."
            )

        render_log = (
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}"
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Remotion render failed "
                f"(exit code {proc.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDERR: {proc.stderr[-2000:]}"
            )

        if not is_sequence and not os.path.isfile(output_path):
            raise RuntimeError(
                f"Render appeared to succeed but output file "
                f"not found: {output_path}"
            )

        print(
            f"[Remotion Get Down] Render complete: {output_path}"
        )

        return (output_path, render_log)


NODE_CLASS_MAPPINGS = {
    "RemotionRender": RemotionRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionRender": "Remotion Render",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
