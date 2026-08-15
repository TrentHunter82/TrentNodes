"""
Save for Next Time / Take from Last Time.

A pair of nodes that carry real typed data between separate queue
runs. Save writes what it receives into a small ring buffer under
output/for_next_time/<slot>/; Take reads it back on a later run,
newest first by default.

Use them for "start from what I made last time" loops: stash a
render, a prompt, or a latent at the end of one run and pick it up
at the start of the next without saving and re-loading by hand.

The storage rules live in utils/for_next_time.py. Two behaviours
here are worth knowing:

* Save has no outputs on purpose. It needs IS_CHANGED = NaN to run
  every queue, and ComfyUI folds a node's IS_CHANGED into the cache
  key of everything downstream of it. Outputs plus NaN would force
  the whole chain, sampler included, to re-run every time. Save
  reports through the node preview instead.

* Take blocks the sockets it has no data for, silently. An entry
  that holds only an image leaves the text, audio, video, latent
  and mask sockets blocked, so only the branches that actually
  need the missing member get skipped. found and entry_name are
  never blocked, so you can always branch on them.

Save and Take on the same slot in the SAME prompt is not
supported: ComfyUI does not order two unconnected nodes, and
Take's cache check runs before any node executes.
"""

import logging
import os
from typing import Any, Dict, Tuple

from comfy_execution.graph import ExecutionBlocker

from ..utils import for_next_time as store

log = logging.getLogger(__name__)

_FALLBACK_MODES = ["block", "empty", "error"]


class SaveForNextTime:
    """
    Stash data for a later queue run.

    Writes every connected input into one entry inside
    output/for_next_time/<slot_name>/, then trims the slot to the
    newest max_entries entries. Connect as many or as few inputs
    as you like: one entry holds whatever arrived together.

    Use case: save the last frame of a render, the prompt that
    made it, or a latent, then pick any of them back up with Take
    from Last Time on the next run.
    """

    CATEGORY = "Trent/Utilities"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    NOT_IDEMPOTENT = True
    DESCRIPTION = (
        "Save data into a named slot for a later queue run. "
        "Keeps the newest max_entries saves and deletes the rest. "
        "Read it back with Take from Last Time. Has no outputs "
        "by design - wire your data to this node and to whatever "
        "else needs it."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "slot_name": ("STRING", {
                    "default": "default",
                    "multiline": False,
                    "tooltip": (
                        "Folder name under output/for_next_time/. "
                        "Use the same name on Take from Last Time. "
                        "Letters, digits, spaces, . - _ only"
                    ),
                }),
                "max_entries": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "tooltip": (
                        "How many saves to keep. Older ones are "
                        "deleted. Keep this low for video - each "
                        "entry holds a full copy of the file"
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Saved as PNG (one file per batch frame)",
                }),
                "mask": ("MASK", {
                    "tooltip": "Saved as a grayscale PNG",
                }),
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Saved as UTF-8 text",
                }),
                "audio": ("AUDIO", {
                    "tooltip": (
                        "Saved as FLAC. Lossless for real audio, "
                        "but not bit-exact for float32 waveforms"
                    ),
                }),
                "video": ("VIDEO", {
                    "tooltip": (
                        "Saved as MP4. Copied without re-encoding "
                        "when the source already matches"
                    ),
                }),
                "latent": ("LATENT", {
                    "tooltip": "Saved as a .latent safetensors file",
                }),
                "note": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Optional label stored in the entry's "
                        "meta.json. Does not affect the filename"
                    ),
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        # Always save. Safe to return NaN here only because this
        # node has no outputs, so no downstream cache key carries
        # the NaN with it.
        return float("NaN")

    def save(
        self,
        slot_name: str,
        max_entries: int = 10,
        image=None,
        mask=None,
        text=None,
        audio=None,
        video=None,
        latent=None,
        note: str = "",
        prompt=None,
        extra_pnginfo=None,
    ) -> Dict[str, Any]:
        members = {
            "image": image,
            "mask": mask,
            "text": text,
            "audio": audio,
            "video": video,
            "latent": latent,
        }

        extra_meta = {}
        if prompt is not None:
            extra_meta["prompt"] = prompt
        if extra_pnginfo is not None:
            extra_meta["extra_pnginfo"] = extra_pnginfo

        entry_name, written = store.save_entry(
            slot_name,
            members,
            max_entries=max_entries,
            note=note,
            extra_meta=extra_meta or None,
        )

        log.info(
            "Saved %s to slot %r as %s",
            "+".join(written),
            slot_name,
            entry_name,
        )

        summary = "{}  ({})".format(entry_name, ", ".join(written))
        result: Dict[str, Any] = {"ui": {"text": (summary,)}}

        # Show the saved image in the node so you can see what
        # went into the slot without opening the folder.
        if "image" in written:
            files = store.read_meta(
                os.path.join(store.slot_dir(slot_name, create=False), entry_name)
            ).get("members", {}).get("image", [])
            subfolder = os.path.join(
                store.ROOT_FOLDER, store.sanitize_slot(slot_name), entry_name
            )
            result["ui"]["images"] = [
                {"filename": f, "subfolder": subfolder, "type": "output"}
                for f in files
            ]

        return result


class TakeFromLastTime:
    """
    Read data back that Save for Next Time stashed earlier.

    Defaults to the newest entry in the slot. Set steps_back to
    reach further back (1 is the one before the newest), or type
    an entry_name to pin an exact entry - a name prefix is enough
    when it is unambiguous.

    Sockets with no data in the chosen entry are blocked, so a
    branch that needs a member the entry does not hold is skipped
    instead of being fed a wrong value. found tells you whether
    an entry was read at all.

    Use case: pick up the image, prompt or latent you saved on the
    previous queue run.
    """

    CATEGORY = "Trent/Utilities"
    RETURN_TYPES = (
        "IMAGE", "MASK", "STRING", "AUDIO", "VIDEO", "LATENT",
        "STRING", "BOOLEAN", "INT",
    )
    RETURN_NAMES = (
        "image", "mask", "text", "audio", "video", "latent",
        "entry_name", "found", "entry_count",
    )
    OUTPUT_TOOLTIPS = (
        "Saved image, or blocked if this entry has none",
        "Saved mask, or blocked if this entry has none",
        "Saved text, or blocked if this entry has none",
        "Saved audio, or blocked if this entry has none",
        "Saved video, or blocked if this entry has none",
        "Saved latent, or blocked if this entry has none",
        "Name of the entry that was read (empty if none)",
        "True when an entry was read",
        "How many entries the slot holds right now",
    )
    FUNCTION = "take"
    DESCRIPTION = (
        "Read back data saved by Save for Next Time on an earlier "
        "queue run. Defaults to the most recent save. Sockets the "
        "saved entry has no data for are blocked, so only the "
        "branches that need them are skipped."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "slot_name": ("STRING", {
                    "default": "default",
                    "multiline": False,
                    "tooltip": (
                        "Must match the slot_name on Save for "
                        "Next Time"
                    ),
                }),
                "steps_back": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "tooltip": (
                        "0 is the most recent save, 1 the one "
                        "before it, and so on. Ignored when "
                        "entry_name is filled in"
                    ),
                }),
            },
            "optional": {
                "entry_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Exact entry folder name, or a unique "
                        "prefix of one. Leave empty to use "
                        "steps_back"
                    ),
                }),
                "fallback_mode": (_FALLBACK_MODES, {
                    "default": "block",
                    "tooltip": (
                        "What to do when the slot is empty or "
                        "steps_back goes too far. block = skip "
                        "the branches downstream. empty = return "
                        "blank values. error = stop with an error"
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        slot_name="",
        steps_back=0,
        entry_name="",
        fallback_mode="block",
        **kwargs,
    ):
        """
        Report whether the entry this node would read has changed.

        Deliberately NOT NaN: ComfyUI folds this value into the
        cache key of every downstream node, and NaN never compares
        equal, so NaN here would re-run the whole chain - sampler
        included - on every queue even when the entry is identical.

        Every failure is swallowed. An uncaught exception makes
        ComfyUI substitute NaN itself, which is the exact
        behaviour being avoided.
        """
        try:
            path = store.slot_dir(slot_name, create=False)

            # A linked input arrives as None here, because
            # IS_CHANGED only sees constant widget values. Fall
            # back to fingerprinting the whole slot, which is a
            # superset of anything the linked value could pick.
            if steps_back is None or entry_name is None or not slot_name:
                return "dir:" + ";".join(store.list_entries(path))

            name = store.resolve_entry(path, steps_back, entry_name)
            if name is None:
                return "missing"
            return store.fingerprint(path, name)
        except Exception:
            return "missing"

    def take(
        self,
        slot_name: str,
        steps_back: int = 0,
        entry_name: str = "",
        fallback_mode: str = "block",
    ) -> Tuple[Any, ...]:
        path = store.slot_dir(slot_name, create=True)
        entries = store.list_entries(path)
        name = store.resolve_entry(path, steps_back, entry_name)

        if name is None:
            return self._handle_missing(
                slot_name, steps_back, entry_name, fallback_mode, len(entries)
            )

        loaded = store.load_entry(path, name)
        if not loaded:
            log.warning(
                "Entry %s in slot %r has no readable members", name, slot_name
            )

        outputs = [
            loaded.get(member, ExecutionBlocker(None))
            for member in store.MEMBER_NAMES
        ]
        return tuple(outputs) + (name, True, len(entries))

    # ---- helpers ----

    @staticmethod
    def _handle_missing(
        slot_name: str,
        steps_back: int,
        entry_name: str,
        mode: str,
        count: int,
    ) -> Tuple[Any, ...]:
        if mode == "error":
            if entry_name.strip():
                detail = "no entry matching {!r}".format(entry_name)
            elif count:
                detail = "steps_back={} but the slot holds {} entries".format(
                    steps_back, count
                )
            else:
                detail = "the slot is empty"
            raise ValueError(
                "Take from Last Time: {} in slot {!r}. Run Save for "
                "Next Time first, or set fallback_mode to 'block' or "
                "'empty'.".format(detail, slot_name)
            )

        if mode == "empty":
            # blank_member returns None for VIDEO - there is no
            # safe empty video - so that socket stays blocked.
            outputs = []
            for member in store.MEMBER_NAMES:
                blank = store.blank_member(member)
                outputs.append(
                    ExecutionBlocker(None) if blank is None else blank
                )
        else:
            outputs = [ExecutionBlocker(None) for _ in store.MEMBER_NAMES]

        return tuple(outputs) + ("", False, count)


NODE_CLASS_MAPPINGS = {
    "SaveForNextTime": SaveForNextTime,
    "TakeFromLastTime": TakeFromLastTime,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveForNextTime": "Save for Next Time",
    "TakeFromLastTime": "Take from Last Time",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
