"""
TextHoldupCowboy - Pause execution to preview/edit a string mid-run.

Holds up the stagecoach: execution blocks while the frontend shows the
incoming text in an editable dialog. The user can edit and send it on,
pass it through unchanged, or let a timeout release it automatically.
"""
import threading
import uuid

import comfy.model_management as mm


# gate_id -> {"event": Event, "text": str | None}
# The server route (server.py /trent/text_holdup) resolves entries here.
PENDING_GATES = {}
PENDING_GATES_LOCK = threading.Lock()


def resolve_gate(gate_id: str, text):
    """Called by the API route to release a waiting node."""
    with PENDING_GATES_LOCK:
        gate = PENDING_GATES.get(gate_id)
        if gate is None:
            return False
        gate["text"] = text
        gate["event"].set()
        return True


class TextHoldupCowboy:
    """Pause the run and let the user preview/edit a string before it moves on."""

    CATEGORY = "Trent/Text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "holdup"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "timeout": ("FLOAT", {
                    "default": 120.0,
                    "min": 5.0,
                    "max": 3600.0,
                    "step": 5.0,
                    "tooltip": "Seconds to wait before passing the text through unchanged",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = pass text straight through without pausing",
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run: the whole point is a fresh human checkpoint each run
        return float("NaN")

    def holdup(self, text, timeout, enabled, unique_id=None):
        if not enabled:
            return (text,)

        from server import PromptServer

        gate_id = uuid.uuid4().hex
        event = threading.Event()
        with PENDING_GATES_LOCK:
            PENDING_GATES[gate_id] = {"event": event, "text": None}

        try:
            print(f"[TextHoldupCowboy] holding up node {unique_id} "
                  f"for up to {timeout:.0f}s (gate {gate_id[:8]})")
            PromptServer.instance.send_sync("trent-text-holdup", {
                "gate_id": gate_id,
                "node_id": unique_id,
                "text": text,
                "timeout": timeout,
            })

            waited = 0.0
            step = 0.2
            while waited < timeout:
                if event.wait(step):
                    break
                waited += step
                if mm.processing_interrupted():
                    PromptServer.instance.send_sync(
                        "trent-text-holdup-close", {"gate_id": gate_id}
                    )
                    mm.throw_exception_if_processing_interrupted()

            with PENDING_GATES_LOCK:
                result = PENDING_GATES[gate_id]["text"]

            if result is None:
                # Timed out: close the dialog and ride on with the original
                PromptServer.instance.send_sync(
                    "trent-text-holdup-close", {"gate_id": gate_id}
                )
                return (text,)
            return (result,)
        finally:
            with PENDING_GATES_LOCK:
                PENDING_GATES.pop(gate_id, None)
