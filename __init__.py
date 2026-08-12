"""
@author: Trent
@title: Trent's ComfyUI Nodes
@nickname: Trent Nodes
@description: Custom nodes for video processing, keyframe management,
              scene detection, and video analysis
"""
from .utils.discovery import discover_nodes
from .utils.banner import (
    setup_terminal,
    print_banner,
    should_show_banner,
    should_use_color
)
from .utils.validation import run_self_test

# Import server routes (registers API endpoints on import).
# Outside a live ComfyUI the route decorators raise AttributeError, not
# ImportError, so catch broadly: a failure here must never take the nodes
# down with it. Report it instead of swallowing it silently.
try:
    from . import server  # noqa: F401
except Exception as _server_err:  # noqa: BLE001
    print(
        f"[TrentNodes] API routes unavailable "
        f"({type(_server_err).__name__}: {_server_err}); nodes still load."
    )

# Node registries
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# WEB_DIRECTORY for serving custom JavaScript files
WEB_DIRECTORY = "./js"

# Discover and register all nodes
discover_nodes(__name__, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)


def _initialize():
    """Run one-time initialization tasks."""
    setup_terminal()

    if should_show_banner():
        print_banner(use_color=should_use_color())

    run_self_test(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)


# Run initialization once per interpreter session
if not globals().get("_TRENT_INIT_RAN"):
    globals()["_TRENT_INIT_RAN"] = True
    _initialize()


__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'WEB_DIRECTORY'
]
