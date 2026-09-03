"""
MetaBatchTimer: measures wall-clock time across an entire VHS Meta Batch run.

The VHS Meta Batch Manager re-queues the workflow once per batch, so each
batch executes as its own prompt with its own prompt_id. A per-prompt timer
(like WorkflowTimer) resets every batch. This node instead stamps the start
time onto the BatchManager instance itself, which VHS keeps alive across
re-queues, so elapsed time accumulates over the whole run.

Wiring:
- Connect meta_batch from the VHS Meta Batch Manager output.
- Route the VHS Video Combine "Filenames" output (or any late output)
  through the passthrough input so the timer executes at the END of each
  batch. Without the passthrough the timer runs near the start of each
  batch, so ETA lags one batch and the final total misses the last batch.
"""
import time

from ..utils.any_type import any_typ
from .workflow_timer import _format_elapsed, _install_hook, _start_times

_STATE_ATTR = "_trent_meta_batch_timer"


class MetaBatchTimer:
    """Total elapsed time / speed / ETA across a full VHS Meta Batch run."""

    CATEGORY = "Trent/Utils"
    RETURN_TYPES = (any_typ, "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("passthrough", "time_string", "seconds", "status")
    FUNCTION = "measure"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meta_batch": ("VHS_BatchManager",),
                "format": (["hms", "clock", "seconds"], {"default": "hms"}),
            },
            "optional": {
                "passthrough": (any_typ, {
                    "tooltip": "Route the Video Combine Filenames output "
                               "through here so the timer runs after each "
                               "batch finishes."}),
            },
            "hidden": {
                "prompt": "PROMPT",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always re-run, even on a fully cached prompt

    def measure(self, meta_batch, format, passthrough=None, prompt=None):
        _install_hook()
        now = time.perf_counter()

        # Which requeue (batch) is this? VHS injects a growing 'requeue'
        # counter into the BatchManager node's prompt inputs: 0, 1, 2, ...
        requeue = 0
        uid = getattr(meta_batch, "unique_id", None)
        if prompt is not None and uid is not None and str(uid) in prompt:
            requeue = prompt[str(uid)]["inputs"].get("requeue", 0)

        # First batch of a fresh run: stamp the run start on the manager.
        # BatchManager.reset() re-inits only its own attributes, so a stale
        # stamp from a previous run survives — requeue == 0 overwrites it.
        state = getattr(meta_batch, _STATE_ATTR, None)
        if requeue == 0 or state is None:
            start = None
            try:
                from comfy_execution.utils import get_executing_context
                ctx = get_executing_context()
                if ctx is not None:
                    captured = _start_times.get(ctx.prompt_id)
                    if captured is not None:
                        start = captured[0]  # true execution_start of run
            except Exception:
                pass
            state = {"start": start if start is not None else now}
            setattr(meta_batch, _STATE_ATTR, state)

        elapsed = now - state["start"]

        fpb = getattr(meta_batch, "frames_per_batch", -1)
        total_frames = getattr(meta_batch, "total_frames", float("inf"))
        num_batches = None
        if fpb and fpb > 0 and total_frames != float("inf"):
            num_batches = int(-(-total_frames // fpb))  # ceil division

        # With the passthrough wired, this node runs after the batch's
        # outputs, so the current batch counts as done. Without it, the
        # node runs early in the batch and only prior batches are done.
        after_batch = passthrough is not None
        batches_done = requeue + 1 if after_batch else requeue

        finished = bool(getattr(meta_batch, "has_closed_inputs", False))
        if num_batches is not None and after_batch \
                and batches_done >= num_batches:
            finished = True

        time_str = _format_elapsed(elapsed, format)
        total_txt = f"/{num_batches}" if num_batches is not None else ""
        parts = []
        if batches_done > 0 and elapsed > 0:
            per_batch = elapsed / batches_done
            frames_done = batches_done * fpb if fpb > 0 else 0
            if total_frames != float("inf"):
                frames_done = min(frames_done, total_frames)
            if finished:
                parts.append(f"COMPLETE — total {time_str}")
                parts.append(f"{batches_done}{total_txt} batches")
            else:
                parts.append(f"batch {requeue + 1}{total_txt}"
                             f" — elapsed {time_str}")
                if num_batches is not None:
                    eta = per_batch * (num_batches - batches_done)
                    parts.append(f"ETA {_format_elapsed(eta, format)}")
            parts.append(f"avg {_format_elapsed(per_batch, format)}/batch")
            if frames_done > 0:
                parts.append(f"{frames_done / elapsed:.1f} frames/s")
        else:
            parts.append(f"batch {requeue + 1}{total_txt}"
                         f" — elapsed {time_str}")
        status = " — ".join(parts)

        return {
            "ui": {"text": (status,)},
            "result": (passthrough, time_str, elapsed, status),
        }
