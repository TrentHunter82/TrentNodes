"""
Remotion Project Config - validates a Remotion project directory
and Node.js installation for the Remotion Get Down bridge.

This is the foundation node that all other Remotion nodes depend on.
It outputs a REMOTION_PROJECT config dict consumed downstream.
"""

import json
import os
import subprocess
from typing import Dict, Any, Tuple

from ..utils.remotion_utils import (
    build_env,
    validate_node_js,
    validate_npx,
)

# Bundled Remotion project ships inside TrentNodes
_TRENT_NODES_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
_BUNDLED_PROJECT = os.path.join(_TRENT_NODES_DIR, "remotion_project")


class RemotionProjectConfig:
    """
    Point to a Remotion project directory, validate it, and
    detect Node.js availability.

    Outputs a REMOTION_PROJECT config dict used by all other
    Remotion Get Down nodes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "project_path": ("STRING", {
                    "default": _BUNDLED_PROJECT,
                    "multiline": False,
                    "tooltip": (
                        "Path to a Remotion project root "
                        "(must contain package.json with remotion "
                        "in dependencies). Defaults to the "
                        "bundled project inside TrentNodes."
                    ),
                }),
            },
            "optional": {
                "node_executable": ("STRING", {
                    "default": "node",
                    "multiline": False,
                    "tooltip": (
                        "Path to Node.js executable. Default 'node' "
                        "assumes it is on PATH. For WSL2 you may "
                        "need the full path like /usr/bin/node"
                    ),
                }),
                "npx_executable": ("STRING", {
                    "default": "npx",
                    "multiline": False,
                    "tooltip": "Path to npx executable.",
                }),
                "auto_install_deps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Automatically run npm install if "
                        "node_modules/ is missing"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("REMOTION_PROJECT",)
    RETURN_NAMES = ("project",)
    FUNCTION = "configure"
    CATEGORY = "Trent/Remotion Get Down"
    DESCRIPTION = (
        "Validates a Remotion project directory and Node.js "
        "installation. Outputs a config used by all other "
        "Remotion Get Down nodes."
    )

    def configure(
        self,
        project_path: str,
        node_executable: str = "node",
        npx_executable: str = "npx",
        auto_install_deps: bool = True,
    ) -> Tuple:
        project_path = os.path.abspath(project_path.strip())

        if not os.path.isdir(project_path):
            raise ValueError(
                f"Project path does not exist: {project_path}"
            )

        pkg_json_path = os.path.join(project_path, "package.json")
        if not os.path.isfile(pkg_json_path):
            raise ValueError(
                f"No package.json found in {project_path}. "
                f"Is this a Remotion project?"
            )

        with open(pkg_json_path, "r", encoding="utf-8") as f:
            pkg = json.load(f)

        deps = pkg.get("dependencies", {})
        dev_deps = pkg.get("devDependencies", {})
        all_deps = {**deps, **dev_deps}

        if "remotion" not in all_deps:
            raise ValueError(
                f"'remotion' not found in dependencies or "
                f"devDependencies of {pkg_json_path}. "
                f"Is this a Remotion project?"
            )

        remotion_version = all_deps.get("remotion", "unknown")

        node_version = validate_node_js(node_executable)
        validate_npx(npx_executable)

        node_modules = os.path.join(project_path, "node_modules")
        if auto_install_deps and not os.path.isdir(node_modules):
            print(
                "[Remotion Get Down] node_modules not found, "
                "running npm install..."
            )
            proc = subprocess.run(
                ["npm", "install"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"npm install failed:\n{proc.stderr[:2000]}"
                )
            print("[Remotion Get Down] npm install complete.")

        public_dir = os.path.join(project_path, "public")
        out_dir = os.path.join(project_path, "out")
        os.makedirs(public_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        config = {
            "project_path": project_path,
            "public_dir": public_dir,
            "out_dir": out_dir,
            "node_executable": node_executable,
            "npx_executable": npx_executable,
            "node_version": node_version,
            "remotion_version": remotion_version,
        }

        print(
            f"[Remotion Get Down] Project validated: "
            f"{project_path} "
            f"(Node {node_version}, "
            f"Remotion {remotion_version})"
        )

        return (config,)


NODE_CLASS_MAPPINGS = {
    "RemotionProjectConfig": RemotionProjectConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemotionProjectConfig": "Remotion Project Config",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
