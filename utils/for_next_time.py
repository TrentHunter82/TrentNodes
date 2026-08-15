"""
Storage layer for Save for Next Time / Take from Last Time.

Keeps a small ring buffer of "entries" per named slot under
output/for_next_time/<slot>/. One entry is one save: a folder
holding whatever members that save received (image, mask, text,
audio, video, latent) plus a meta.json manifest.

Design notes that are load-bearing:

* An entry is a FOLDER, published with a single os.rename from a
  hidden staging folder. A rename of one directory is atomic, so
  an entry is either fully present or absent. Writing loose files
  that share a filename stem has no equivalent - a crash between
  two files leaves a half entry that cannot be told apart from a
  legitimately single-member entry.

* Entry names are "<seq:08d>-<YYYYmmdd-HHMMSS>". The sequence
  number comes from the folder itself, not the clock. Wall clock
  time is not monotonic: DST roll-back, NTP steps and WSL clock
  drift after suspend all reorder a purely time-based name. The
  timestamp stays in the name for humans.

* Sorting is by name, never by mtime. A folder's mtime changes
  when anything inside it changes, so pruning one entry could
  reorder the rest. Copies that do not preserve mtime (plain cp,
  moves across filesystems, restores from backup) would scramble
  it too.

* Names that do not parse are never pruned and sort as oldest.
  Rename an entry to something like "hero-shot" and it becomes a
  permanent pin: reachable by entry_name, unable to take over
  steps_back=0.

This module holds no node classes. utils/ is not scanned by
discover_nodes (it walks nodes/ only), so nothing here registers
as a node by accident.
"""

import io
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:  # pragma: no cover - only hit outside ComfyUI
    folder_paths = None


# Members in the order they appear on both nodes' sockets.
MEMBER_NAMES = ("image", "mask", "text", "audio", "video", "latent")

ROOT_FOLDER = "for_next_time"
META_FILENAME = "meta.json"
META_VERSION = 1

STAGING_PREFIX = ".tmp-"
TRASH_PREFIX = ".trash-"
STALE_STAGING_SECONDS = 3600

# "<seq:08d>-<timestamp>". Only the leading sequence is parsed.
_ENTRY_RE = re.compile(r"^(\d{8})-")
_SLOT_ALLOWED_RE = re.compile(r"^[A-Za-z0-9._ -]+$")
_MAX_SLOT_LEN = 64
_PUBLISH_ATTEMPTS = 20

# One lock per slot, so two nodes in one process cannot race to
# the same sequence number. Cross-process races are handled by the
# publish retry loop instead.
_SLOT_LOCKS: Dict[str, threading.Lock] = {}
_SLOT_LOCKS_GUARD = threading.Lock()


# ---- paths ----


def output_root() -> str:
    """Return the for_next_time root inside ComfyUI's output dir."""
    if folder_paths is not None:
        base = folder_paths.get_output_directory()
    else:  # pragma: no cover - only hit outside ComfyUI
        base = os.path.join(os.getcwd(), "output")
    return os.path.join(base, ROOT_FOLDER)


def sanitize_slot(slot_name: str) -> str:
    """
    Validate a user-supplied slot name.

    Raises:
        ValueError: if the name is empty, too long, reserved, or
            holds anything outside [A-Za-z0-9._ -].
    """
    name = (slot_name or "").strip()
    if not name:
        raise ValueError("slot_name is required")
    if len(name) > _MAX_SLOT_LEN:
        raise ValueError(
            "slot_name is longer than {} characters".format(_MAX_SLOT_LEN)
        )
    if name in (".", ".."):
        raise ValueError("slot_name cannot be '.' or '..'")
    if name.startswith("."):
        raise ValueError("slot_name cannot start with '.'")
    if not _SLOT_ALLOWED_RE.match(name):
        raise ValueError(
            "slot_name may only use letters, digits, spaces, dot, "
            "dash and underscore"
        )
    return name


def slot_dir(slot_name: str, create: bool = True) -> str:
    """
    Resolve a slot folder, creating it when asked.

    Creating on read as well as write means the first run of a
    save/take loop works with no manual setup, the same way
    GetLatestImageFromFolder does it.
    """
    name = sanitize_slot(slot_name)
    root = output_root()
    path = os.path.abspath(os.path.join(root, name))

    # Belt and braces. sanitize_slot already rejects separators,
    # but is_within_directory realpaths both sides, so this also
    # catches a symlink planted at the slot path.
    if folder_paths is not None:
        os.makedirs(root, exist_ok=True)
        if not folder_paths.is_within_directory(root, path):
            raise ValueError("Invalid slot_name: {!r}".format(slot_name))
    elif os.path.commonpath([root, path]) != os.path.abspath(root):
        raise ValueError("Invalid slot_name: {!r}".format(slot_name))

    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _slot_lock(path: str) -> threading.Lock:
    with _SLOT_LOCKS_GUARD:
        lock = _SLOT_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _SLOT_LOCKS[path] = lock
        return lock


# ---- entry enumeration ----


def entry_sort_key(name: str) -> Tuple[int, int, str]:
    """
    Sort key for entry folder names, oldest first.

    Unparseable names get a leading 0 so they always sort older
    than every real entry. That keeps a hand-renamed keeper from
    hijacking steps_back=0.
    """
    match = _ENTRY_RE.match(name)
    if match:
        return (1, int(match.group(1)), name)
    return (0, 0, name)


def list_entries(slot_path: str) -> List[str]:
    """
    Return entry folder names in this slot, oldest first.

    Skips hidden folders (staging and trash), loose files, and
    symlinks. A folder with no meta.json is not an entry.
    """
    try:
        raw = os.listdir(slot_path)
    except OSError:
        return []

    names = []
    for name in raw:
        if name.startswith("."):
            continue
        full = os.path.join(slot_path, name)
        if os.path.islink(full) or not os.path.isdir(full):
            continue
        if not os.path.isfile(os.path.join(full, META_FILENAME)):
            continue
        names.append(name)

    names.sort(key=entry_sort_key)
    return names


def resolve_entry(
    slot_path: str,
    steps_back: int = 0,
    entry_name: str = "",
) -> Optional[str]:
    """
    Pick one entry name, or None when there is nothing to pick.

    entry_name wins when non-empty. It is matched against the
    enumerated folder names - exact first, then a unique prefix -
    so no user string is ever joined into a path. steps_back
    counts back from the newest; 0 is the newest. An overrun
    returns None rather than clamping.
    """
    names = list_entries(slot_path)
    if not names:
        return None

    wanted = (entry_name or "").strip()
    if wanted:
        if wanted in names:
            return wanted
        matches = [n for n in names if n.startswith(wanted)]
        if len(matches) == 1:
            return matches[0]
        return None

    steps = int(steps_back or 0)
    if steps < 0 or steps >= len(names):
        return None
    return names[len(names) - 1 - steps]


def read_meta(entry_path: str) -> Dict[str, Any]:
    """Load an entry's manifest, or {} when it is unreadable."""
    try:
        with open(
            os.path.join(entry_path, META_FILENAME), "r", encoding="utf-8"
        ) as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def fingerprint(slot_path: str, entry_name: str) -> str:
    """
    Cheap change token for one entry, used by IS_CHANGED.

    meta.json is written last inside staging and the whole entry
    arrives by one rename, so its size and mtime stand in for the
    entry's content. The entry name is included so that moving
    steps_back onto a different entry always busts the token.
    """
    meta = os.path.join(slot_path, entry_name, META_FILENAME)
    stat = os.stat(meta)
    return "{}:{}:{}".format(entry_name, stat.st_mtime_ns, stat.st_size)


# ---- writing ----


def _next_seq(slot_path: str) -> int:
    highest = 0
    for name in list_entries(slot_path):
        match = _ENTRY_RE.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def _entry_name_for(seq: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return "{:08d}-{}".format(seq, stamp)


def sweep_stale(slot_path: str, max_age: float = STALE_STAGING_SECONDS):
    """Delete abandoned staging and trash folders."""
    now = time.time()
    try:
        names = os.listdir(slot_path)
    except OSError:
        return

    for name in names:
        if not name.startswith((STAGING_PREFIX, TRASH_PREFIX)):
            continue
        full = os.path.join(slot_path, name)
        if os.path.islink(full) or not os.path.isdir(full):
            continue
        try:
            if now - os.path.getmtime(full) < max_age:
                continue
        except OSError:
            continue
        shutil.rmtree(full, ignore_errors=True)


def _publish(slot_path: str, staging: str) -> str:
    """
    Move a finished staging folder into place under a fresh name.

    os.rename onto an existing non-empty directory fails, which is
    exactly the collision signal we want when another process
    claimed the same sequence number. Retry with a recomputed one.
    """
    last_error = None
    for _ in range(_PUBLISH_ATTEMPTS):
        name = _entry_name_for(_next_seq(slot_path))
        target = os.path.join(slot_path, name)
        if os.path.exists(target):
            time.sleep(0.001)
            continue
        try:
            os.rename(staging, target)
            return name
        except OSError as exc:
            last_error = exc
            time.sleep(0.001)

    raise RuntimeError(
        "Could not publish an entry into {} after {} attempts: {}".format(
            slot_path, _PUBLISH_ATTEMPTS, last_error
        )
    )


def prune(slot_path: str, max_entries: int) -> List[str]:
    """
    Trim the slot down to the newest max_entries entries.

    Runs after a successful publish, never before: a crash partway
    through leaves too many entries, which is harmless, instead of
    too few. Unparseable names are left alone so a renamed entry
    acts as a permanent pin.
    """
    keep = max(1, int(max_entries))
    prunable = [n for n in list_entries(slot_path) if _ENTRY_RE.match(n)]
    victims = prunable[:-keep] if len(prunable) > keep else []

    removed = []
    for name in victims:
        source = os.path.join(slot_path, name)
        trash = os.path.join(
            slot_path, "{}{}-{}".format(TRASH_PREFIX, name, os.getpid())
        )
        try:
            os.rename(source, trash)
        except OSError:
            # Already gone, or another process is pruning it too.
            continue
        shutil.rmtree(trash, ignore_errors=True)
        removed.append(name)

    return removed


def save_entry(
    slot_name: str,
    members: Dict[str, Any],
    max_entries: int = 10,
    note: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    """
    Write one entry and prune the slot.

    Args:
        slot_name: user-facing slot name.
        members: member name -> value, for members that are set.
        max_entries: ring size.
        note: free text stored in the manifest.
        extra_meta: extra manifest keys (prompt provenance).

    Returns:
        (entry_name, list of member names written)
    """
    written = [name for name in MEMBER_NAMES if members.get(name) is not None]
    if not written:
        raise ValueError(
            "Save for Next Time got no data. Connect at least one of: "
            + ", ".join(MEMBER_NAMES)
        )

    path = slot_dir(slot_name, create=True)

    with _slot_lock(path):
        sweep_stale(path)

        staging = os.path.join(
            path,
            "{}{}-{}".format(STAGING_PREFIX, os.getpid(), uuid.uuid4().hex[:8]),
        )
        os.makedirs(staging)

        try:
            manifest: Dict[str, List[str]] = {}
            for name in written:
                manifest[name] = _WRITERS[name](staging, members[name])

            meta = {
                "version": META_VERSION,
                "created": datetime.now(timezone.utc).isoformat(),
                "slot": sanitize_slot(slot_name),
                "note": note or "",
                "members": manifest,
            }
            if extra_meta:
                meta.update(extra_meta)

            # Written last: its presence is what makes the folder
            # a real entry.
            with open(
                os.path.join(staging, META_FILENAME), "w", encoding="utf-8"
            ) as handle:
                json.dump(meta, handle, indent=2)

            entry_name = _publish(path, staging)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        prune(path, max_entries)

    return entry_name, written


# ---- reading ----


def load_entry(slot_path: str, entry_name: str) -> Dict[str, Any]:
    """
    Load every member of one entry.

    Returns:
        member name -> value, holding only members present on disk
        and readable.
    """
    entry_path = os.path.join(slot_path, entry_name)
    meta = read_meta(entry_path)
    manifest = meta.get("members") or {}

    loaded: Dict[str, Any] = {}
    for name in MEMBER_NAMES:
        files = manifest.get(name)
        if not files:
            continue
        if isinstance(files, str):
            files = [files]
        paths = [os.path.join(entry_path, f) for f in files]
        if not all(os.path.isfile(p) for p in paths):
            continue
        loaded[name] = _READERS[name](paths)

    return loaded


# ---- codecs ----


def _write_image(staging: str, value: torch.Tensor) -> List[str]:
    tensor = value.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    names = []
    single = tensor.shape[0] == 1
    for index in range(tensor.shape[0]):
        frame = tensor[index].numpy()
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        name = "image.png" if single else "image.{:03d}.png".format(index)
        Image.fromarray(frame).save(os.path.join(staging, name))
        names.append(name)
    return names


def _read_image(paths: List[str]) -> torch.Tensor:
    frames = []
    for path in paths:
        img = Image.open(path)
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        frames.append(torch.from_numpy(arr))
    return torch.stack(frames, dim=0)


def _write_mask(staging: str, value: torch.Tensor) -> List[str]:
    tensor = value.detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4:
        # [B,H,W,1] or [B,1,H,W] both show up in the wild.
        tensor = tensor.squeeze(-1) if tensor.shape[-1] == 1 else tensor[:, 0]

    names = []
    single = tensor.shape[0] == 1
    for index in range(tensor.shape[0]):
        plane = tensor[index].numpy()
        plane = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
        name = "mask.png" if single else "mask.{:03d}.png".format(index)
        Image.fromarray(plane, mode="L").save(os.path.join(staging, name))
        names.append(name)
    return names


def _read_mask(paths: List[str]) -> torch.Tensor:
    planes = []
    for path in paths:
        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float32) / 255.0
        planes.append(torch.from_numpy(arr))
    return torch.stack(planes, dim=0)


def _write_text(staging: str, value: Any) -> List[str]:
    text = value if isinstance(value, str) else str(value)
    with open(os.path.join(staging, "text.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return ["text.txt"]


def _read_text(paths: List[str]) -> str:
    with open(paths[0], "r", encoding="utf-8") as handle:
        return handle.read()


def _write_audio(staging: str, value: Dict[str, Any]) -> List[str]:
    """
    Write AUDIO as FLAC, one file per batch item.

    Mirrors AudioSaveHelper.save_audio in comfy_api/latest/_ui.py
    rather than calling it: that helper routes through
    folder_paths.get_save_image_path and picks its own counter
    based filenames, which fights the staging folder.
    """
    import av

    waveform = value["waveform"]
    sample_rate = int(value["sample_rate"])
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    names = []
    for index, item in enumerate(waveform.detach().cpu()):
        layout = "mono" if item.shape[0] == 1 else "stereo"
        name = "audio.{:03d}.flac".format(index)

        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="flac")
        stream = container.add_stream("flac", rate=sample_rate, layout=layout)

        frame = av.AudioFrame.from_ndarray(
            item.movedim(0, 1).reshape(1, -1).float().numpy(),
            format="flt",
            layout=layout,
        )
        frame.sample_rate = sample_rate
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
        container.close()

        buffer.seek(0)
        with open(os.path.join(staging, name), "wb") as handle:
            handle.write(buffer.getbuffer())
        names.append(name)

    return names


def _read_audio(paths: List[str]) -> Dict[str, Any]:
    from comfy_extras.nodes_audio import load as load_audio

    waveforms = []
    sample_rate = 44100
    for path in paths:
        waveform, sample_rate = load_audio(path)
        waveforms.append(waveform)
    return {
        "waveform": torch.stack(waveforms, dim=0),
        "sample_rate": int(sample_rate),
    }


def _write_video(staging: str, value: Any) -> List[str]:
    # save_to reuses the source streams when the container and
    # codec already match, so this is a remux, not a re-encode.
    value.save_to(os.path.join(staging, "video.mp4"))
    return ["video.mp4"]


def _read_video(paths: List[str]) -> Any:
    from comfy_api.latest import InputImpl

    # VideoFromFile only stores the path and decodes lazily, so
    # handing back a path inside the ring buffer would break as
    # soon as a later save prunes this entry. Copy the bytes into
    # the temp dir first - no transcode.
    source = paths[0]
    if folder_paths is not None:
        temp_dir = folder_paths.get_temp_directory()
    else:  # pragma: no cover - only hit outside ComfyUI
        temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    target = os.path.join(
        temp_dir,
        "for_next_time_{}{}".format(
            uuid.uuid4().hex[:12], os.path.splitext(source)[1]
        ),
    )
    shutil.copyfile(source, target)
    return InputImpl.VideoFromFile(target)


def _write_latent(staging: str, value: Dict[str, Any]) -> List[str]:
    import comfy.utils

    payload = {
        "latent_tensor": value["samples"].detach().cpu().contiguous(),
        "latent_format_version_0": torch.tensor([]),
    }
    comfy.utils.save_torch_file(payload, os.path.join(staging, "latent.latent"))
    return ["latent.latent"]


def _read_latent(paths: List[str]) -> Dict[str, Any]:
    import safetensors.torch

    data = safetensors.torch.load_file(paths[0], device="cpu")
    return {"samples": data["latent_tensor"].float()}


_WRITERS = {
    "image": _write_image,
    "mask": _write_mask,
    "text": _write_text,
    "audio": _write_audio,
    "video": _write_video,
    "latent": _write_latent,
}

_READERS = {
    "image": _read_image,
    "mask": _read_mask,
    "text": _read_text,
    "audio": _read_audio,
    "video": _read_video,
    "latent": _read_latent,
}


def blank_member(name: str) -> Any:
    """
    Type-appropriate empty value for fallback_mode='empty'.

    VIDEO has no safe blank - synthesising a real container just
    to hand back an empty clip is worse than blocking - so it
    returns None and the caller blocks that socket instead.
    """
    if name == "image":
        return torch.zeros(1, 64, 64, 3, dtype=torch.float32)
    if name == "mask":
        return torch.zeros(1, 64, 64, dtype=torch.float32)
    if name == "text":
        return ""
    if name == "latent":
        return {"samples": torch.zeros(1, 4, 8, 8, dtype=torch.float32)}
    if name == "audio":
        return {
            "waveform": torch.zeros(1, 1, 1, dtype=torch.float32),
            "sample_rate": 44100,
        }
    return None
