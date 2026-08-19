"""
WorkflowTimer: reports elapsed time since the current workflow started.

A module-level hook wraps PromptServer.send_sync to catch the
"execution_start" event and record a start timestamp per prompt_id.
The node reads the current prompt_id from the executing context and
outputs the elapsed time as a formatted string and a float.

Timing semantics: the node measures from execution_start until the
moment *it* executes. Route your final output through the passthrough
input so the timer runs after the slow work. Anything scheduled after
the timer (e.g. a downstream save) is not included.
"""
import time

from ..utils.any_type import any_typ

_start_times = {}  # prompt_id -> (perf_counter, wall clock)
_MAX_TRACKED = 50
_hook_installed = False


def _install_hook():
    global _hook_installed
    if _hook_installed:
        return
    try:
        from server import PromptServer
    except ImportError:
        return  # headless test context without a server
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return
    original_send_sync = instance.send_sync

    def send_sync(event, data, sid=None):
        if event == "execution_start" and isinstance(data, dict):
            prompt_id = data.get("prompt_id")
            if prompt_id is not None:
                _start_times[prompt_id] = (time.perf_counter(), time.time())
                while len(_start_times) > _MAX_TRACKED:
                    _start_times.pop(next(iter(_start_times)))
        return original_send_sync(event, data, sid)

    instance.send_sync = send_sync
    _hook_installed = True


_install_hook()


def _format_elapsed(seconds: float, fmt: str) -> str:
    if fmt == "seconds":
        return f"{seconds:.2f}"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if fmt == "clock":
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    # "hms"
    parts = []
    if hours >= 1:
        parts.append(f"{int(hours)}h")
    if minutes >= 1 or hours >= 1:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{secs:.1f}s")
    return " ".join(parts)


class WorkflowTimer:
    """Outputs the time elapsed since the current workflow run started."""

    CATEGORY = "Trent/Utils"
    RETURN_TYPES = (any_typ, "STRING", "FLOAT")
    RETURN_NAMES = ("passthrough", "time_string", "seconds")
    FUNCTION = "measure"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (["hms", "clock", "seconds"], {"default": "hms"}),
            },
            "optional": {
                "passthrough": (any_typ,),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always re-run, even on a fully cached prompt

    def measure(self, format, passthrough=None):
        _install_hook()  # retry in case the server appeared after import
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        prompt_id = ctx.prompt_id if ctx is not None else None
        start = _start_times.get(prompt_id)
        if start is None:
            text = "unknown (start not captured)"
            seconds = -1.0
        else:
            seconds = time.perf_counter() - start[0]
            text = _format_elapsed(seconds, format)
        return {
            "ui": {"text": (text,)},
            "result": (passthrough, text, seconds),
        }
