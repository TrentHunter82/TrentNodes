"""
Remotion List Compositions - queries a Remotion project for
available composition IDs so the user can see what is renderable.
"""

import os
import subprocess
from typing import Dict, Any, Tuple

from ..utils.remotion_utils import build_env


class RemotionListCompositions:
    """
    List available Remotion compositions from a project.

    Runs npx remotion compositions and outputs the result as
    a string for inspection.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "project": ("REMOTION_PROJECT",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("compositions",)
    FUNCTION = "list_compositions"
    CATEGORY = "Trent/Remotion Get Down"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Lists available Remotion compositions from a project. "
        "Useful for discovering composition IDs to pass to the "
        "Remotion Render node."
    )

    def list_compositions(self, project: dict) -> Tuple:
        entry_point = os.path.join(
            project["project_path"], "src", "index.ts"
        )

        cmd = [
            project["npx_executable"],
            "remotion", "compositions",
            entry_point,
        ]

        env = build_env(project)

        try:
            proc = subprocess.run(
                cmd,
                cwd=project["project_path"],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Remotion compositions query timed out after 60s."
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to list compositions:\n"
                f"{proc.stderr[-2000:]}"
            )

        output = proc.stdout.strip()

        print(
            f"[Remotion Get Down] Compositions:\n{output}"
        )

        return (output,)


NODE_CLASS_MAPPINGS = {
    "RemotionListCompositions": RemotionListCompositions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionListCompositions": "Remotion List Compositions",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
