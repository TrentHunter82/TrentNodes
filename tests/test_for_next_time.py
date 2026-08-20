"""
Tests for Save for Next Time / Take from Last Time.

Run:
    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_for_next_time.py

Synthesizes the TrentNodes package so TrentNodes/__init__.py
discovery (and every other node's imports) never runs, then points
folder_paths at a temp directory so the real ComfyUI output folder
is never touched.
"""

import importlib
import os
import shutil
import sys
import tempfile
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")
sys.path.insert(0, ROOT)

pkg = types.ModuleType("TrentNodes")
pkg.__path__ = [PKG]
sys.modules["TrentNodes"] = pkg
for _sub in ("nodes", "utils"):
    _mod = types.ModuleType("TrentNodes.{}".format(_sub))
    _mod.__path__ = [os.path.join(PKG, _sub)]
    sys.modules["TrentNodes.{}".format(_sub)] = _mod

import folder_paths  # noqa: E402
import torch  # noqa: E402
from comfy_execution.graph import ExecutionBlocker  # noqa: E402

TEMP_ROOT = tempfile.mkdtemp(prefix="fnt_test_")
folder_paths.output_directory = os.path.join(TEMP_ROOT, "output")
folder_paths.temp_directory = os.path.join(TEMP_ROOT, "temp")
os.makedirs(folder_paths.output_directory, exist_ok=True)
os.makedirs(folder_paths.temp_directory, exist_ok=True)

store = importlib.import_module("TrentNodes.utils.for_next_time")
nodes_mod = importlib.import_module("TrentNodes.nodes.for_next_time")

SaveForNextTime = nodes_mod.SaveForNextTime
TakeFromLastTime = nodes_mod.TakeFromLastTime

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("{}{}".format(name, "\n    " + detail if detail else ""))


def fresh_slot(name):
    """Return a slot name that has no entries yet."""
    path = os.path.join(folder_paths.output_directory, store.ROOT_FOLDER, name)
    shutil.rmtree(path, ignore_errors=True)
    return name


def take_all(slot, **kwargs):
    return TakeFromLastTime().take(slot, **kwargs)


def blocked(value):
    return isinstance(value, ExecutionBlocker)


# ---- slot name safety ----


def test_slot_names():
    for bad in ("", "   ", ".", "..", "../escape", "a/b", "a\\b", ".hidden",
                "x" * 65, "semi;colon"):
        try:
            store.sanitize_slot(bad)
            check("reject slot {!r}".format(bad), False, "was accepted")
        except ValueError:
            check("reject slot {!r}".format(bad), True)

    for good in ("default", "my slot", "a-b_c.1"):
        try:
            check("accept slot {!r}".format(good),
                  store.sanitize_slot(good) == good)
        except ValueError as exc:
            check("accept slot {!r}".format(good), False, str(exc))

    root = os.path.abspath(store.output_root())
    path = store.slot_dir("nested_check", create=True)
    check("slot_dir stays under the root",
          os.path.abspath(path).startswith(root),
          "{} not under {}".format(path, root))


# ---- round trips ----


def test_text_round_trip():
    slot = fresh_slot("text_slot")
    name, written = store.save_entry(slot, {"text": "hello next time"})
    check("text save reports the member", written == ["text"], str(written))

    out = take_all(slot)
    check("text comes back", out[2] == "hello next time", repr(out[2]))
    check("text entry_name matches", out[6] == name, out[6])
    check("text found is True", out[7] is True)
    check("text entry_count is 1", out[8] == 1, str(out[8]))
    check("unset members are blocked",
          all(blocked(out[i]) for i in (0, 1, 3, 4, 5)))


def test_image_round_trip():
    slot = fresh_slot("image_slot")
    image = torch.rand(1, 12, 16, 3)
    store.save_entry(slot, {"image": image})

    out = take_all(slot)
    got = out[0]
    check("image shape survives", tuple(got.shape) == (1, 12, 16, 3),
          str(tuple(got.shape)))
    # PNG is 8-bit, so allow one quantisation step.
    check("image values survive", torch.allclose(got, image, atol=1.0 / 255.0),
          "max delta {}".format((got - image).abs().max().item()))


def test_image_batch_round_trip():
    slot = fresh_slot("image_batch_slot")
    image = torch.rand(3, 8, 8, 3)
    _, written = store.save_entry(slot, {"image": image})
    check("batch image saved", written == ["image"])

    out = take_all(slot)
    check("batch image shape survives", tuple(out[0].shape) == (3, 8, 8, 3),
          str(tuple(out[0].shape)))


def test_mask_round_trip():
    slot = fresh_slot("mask_slot")
    mask = torch.rand(1, 10, 10)
    store.save_entry(slot, {"mask": mask})

    out = take_all(slot)
    check("mask shape survives", tuple(out[1].shape) == (1, 10, 10),
          str(tuple(out[1].shape)))
    check("mask values survive",
          torch.allclose(out[1], mask, atol=1.0 / 255.0))


def test_latent_round_trip():
    slot = fresh_slot("latent_slot")
    latent = {"samples": torch.rand(1, 4, 8, 8)}
    store.save_entry(slot, {"latent": latent})

    out = take_all(slot)
    check("latent is a dict with samples",
          isinstance(out[5], dict) and "samples" in out[5])
    check("latent survives exactly",
          torch.equal(out[5]["samples"], latent["samples"]))


def test_audio_round_trip():
    slot = fresh_slot("audio_slot")
    waveform = torch.zeros(1, 2, 800)
    waveform[0, 0, :] = torch.linspace(-0.5, 0.5, 800)
    audio = {"waveform": waveform, "sample_rate": 16000}
    store.save_entry(slot, {"audio": audio})

    out = take_all(slot)
    got = out[3]
    check("audio is a dict", isinstance(got, dict) and "waveform" in got)
    check("audio sample_rate survives", got["sample_rate"] == 16000,
          str(got.get("sample_rate")))
    check("audio channel count survives", got["waveform"].shape[1] == 2,
          str(tuple(got["waveform"].shape)))
    check("audio length survives", got["waveform"].shape[2] == 800,
          str(tuple(got["waveform"].shape)))


def _make_video(path, frames=3):
    """Write a tiny real mp4 so the VIDEO codec can be exercised."""
    import av
    import numpy as np

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    for index in range(frames):
        arr = np.full((32, 32, 3), index * 40, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        container.mux(stream.encode(frame))
    container.mux(stream.encode(None))
    container.close()


def test_video_round_trip_and_temp_copy():
    from comfy_api.latest import InputImpl

    slot = fresh_slot("video_slot")
    source = os.path.join(TEMP_ROOT, "source.mp4")
    _make_video(source)

    store.save_entry(slot, {"video": InputImpl.VideoFromFile(source)},
                     max_entries=1)
    out = take_all(slot)
    video = out[4]
    check("video comes back", not blocked(video) and video is not None)

    path = video.get_stream_source()
    slot_path = store.slot_dir(slot, create=False)
    check("video is copied out of the ring buffer",
          isinstance(path, str)
          and not os.path.abspath(path).startswith(os.path.abspath(slot_path)),
          "returned {}".format(path))

    # The whole point of the copy: it must still decode after the
    # entry that produced it has been pruned away.
    store.save_entry(slot, {"text": "pushes the video out"}, max_entries=1)
    check("pruning removed the video entry",
          len(store.list_entries(slot_path)) == 1)
    try:
        width, height = video.get_dimensions()
        check("copied video still decodes after pruning",
              (width, height) == (32, 32), "{}x{}".format(width, height))
    except Exception as exc:
        check("copied video still decodes after pruning", False, str(exc))


def test_multi_member_entry():
    slot = fresh_slot("multi_slot")
    _, written = store.save_entry(
        slot,
        {"image": torch.rand(1, 8, 8, 3), "text": "both"},
    )
    check("both members recorded", sorted(written) == ["image", "text"],
          str(written))

    out = take_all(slot)
    check("multi entry returns image", not blocked(out[0]))
    check("multi entry returns text", out[2] == "both", repr(out[2]))
    check("multi entry blocks latent", blocked(out[5]))
    check("multi entry counts as one", out[8] == 1, str(out[8]))


def test_empty_save_rejected():
    slot = fresh_slot("empty_save_slot")
    try:
        store.save_entry(slot, {name: None for name in store.MEMBER_NAMES})
        check("saving nothing raises", False, "no error")
    except ValueError:
        check("saving nothing raises", True)


# ---- ring buffer behaviour ----


def test_pruning_and_steps_back():
    slot = fresh_slot("ring_slot")
    for index in range(13):
        store.save_entry(slot, {"text": "run {}".format(index)},
                         max_entries=10)

    path = store.slot_dir(slot, create=False)
    entries = store.list_entries(path)
    check("ring settles at max_entries", len(entries) == 10, str(len(entries)))

    check("steps_back 0 is newest", take_all(slot)[2] == "run 12",
          take_all(slot)[2])
    check("steps_back 3 is the 4th newest",
          take_all(slot, steps_back=3)[2] == "run 9")
    check("steps_back 9 is the oldest kept",
          take_all(slot, steps_back=9)[2] == "run 3")

    out = take_all(slot, steps_back=10)
    check("steps_back overrun does not wrap", out[7] is False)
    check("steps_back overrun blocks text", blocked(out[2]))
    check("entry_count still reports the ring size", out[8] == 10, str(out[8]))


def test_seq_survives_a_clock_rewind():
    slot = fresh_slot("clock_slot")
    store.save_entry(slot, {"text": "first"})
    path = store.slot_dir(slot, create=False)

    # Simulate the machine's clock jumping backwards a year: the
    # timestamp half of the name goes back, the sequence does not.
    first = store.list_entries(path)[0]
    rewound = "{:08d}-19990101-000000".format(
        int(first.split("-")[0]) + 1
    )
    shutil.copytree(
        os.path.join(path, first), os.path.join(path, rewound)
    )

    entries = store.list_entries(path)
    check("a rewound clock still sorts newest last",
          entries[-1] == rewound, str(entries))
    check("next seq keeps climbing past the rewind",
          store._next_seq(path) == int(rewound.split("-")[0]) + 1)


def test_unparseable_names_are_pinned():
    slot = fresh_slot("pin_slot")
    store.save_entry(slot, {"text": "real"})
    path = store.slot_dir(slot, create=False)

    # A rename moves the whole entry folder, members and all.
    shutil.copytree(
        os.path.join(path, store.list_entries(path)[0]),
        os.path.join(path, "hero-shot"),
    )

    check("a renamed entry sorts oldest",
          store.list_entries(path)[0] == "hero-shot",
          str(store.list_entries(path)))

    for index in range(5):
        store.save_entry(slot, {"text": str(index)}, max_entries=1)

    check("a renamed entry is never pruned",
          "hero-shot" in store.list_entries(path),
          str(store.list_entries(path)))
    check("renamed entries do not hijack steps_back 0",
          take_all(slot)[2] == "4", take_all(slot)[2])
    check("a renamed entry is reachable by name",
          take_all(slot, entry_name="hero-shot")[2] == "real")


def test_incomplete_entries_are_ignored():
    slot = fresh_slot("partial_slot")
    store.save_entry(slot, {"text": "complete"})
    path = store.slot_dir(slot, create=False)

    # A save killed before meta.json was written.
    os.makedirs(os.path.join(path, "00000009-20260814-120000"))
    with open(
        os.path.join(path, "00000009-20260814-120000", "text.txt"),
        "w", encoding="utf-8",
    ) as handle:
        handle.write("half written")

    # Abandoned staging, and a file dropped in by hand.
    os.makedirs(os.path.join(path, ".tmp-999-deadbeef"))
    with open(os.path.join(path, "stray.png"), "w", encoding="utf-8") as fh:
        fh.write("not an entry")

    check("an entry with no meta.json is skipped",
          store.list_entries(path) == [
              n for n in store.list_entries(path) if "00000009" not in n
          ] and len(store.list_entries(path)) == 1,
          str(store.list_entries(path)))
    check("the newest complete entry still wins",
          take_all(slot)[2] == "complete", take_all(slot)[2])


def test_stale_staging_is_swept():
    slot = fresh_slot("sweep_slot")
    path = store.slot_dir(slot, create=True)
    stale = os.path.join(path, ".tmp-1-oldoldold")
    os.makedirs(stale)
    os.utime(stale, (0, 0))

    store.sweep_stale(path)
    check("stale staging folders are swept", not os.path.exists(stale))


def test_entry_name_lookup():
    slot = fresh_slot("name_slot")
    first, _ = store.save_entry(slot, {"text": "one"})
    second, _ = store.save_entry(slot, {"text": "two"})

    check("entry_name exact match wins",
          take_all(slot, entry_name=first)[2] == "one")
    check("entry_name prefix match works",
          take_all(slot, entry_name=first.split("-")[0])[2] == "one")
    check("entry_name beats steps_back",
          take_all(slot, steps_back=0, entry_name=first)[2] == "one")

    out = take_all(slot, entry_name="0000")
    check("an ambiguous prefix finds nothing", out[7] is False, str(out[7]))

    out = take_all(slot, entry_name="../../../etc")
    check("entry_name cannot escape the slot", out[7] is False)
    check("second entry is still the newest", take_all(slot)[2] == "two",
          second)


# ---- fallback modes ----


def test_fallback_modes():
    slot = fresh_slot("fallback_slot")

    out = take_all(slot, fallback_mode="block")
    check("block mode blocks every typed socket",
          all(blocked(out[i]) for i in range(6)))
    check("block mode still reports found", out[7] is False)
    check("block mode still reports entry_name", out[6] == "")
    check("block mode still reports entry_count", out[8] == 0)

    out = take_all(slot, fallback_mode="empty")
    check("empty mode returns a blank image",
          not blocked(out[0]) and tuple(out[0].shape) == (1, 64, 64, 3),
          str(getattr(out[0], "shape", out[0])))
    check("empty mode returns a blank mask",
          not blocked(out[1]) and tuple(out[1].shape) == (1, 64, 64))
    check("empty mode returns empty text", out[2] == "")
    check("empty mode returns silent audio",
          isinstance(out[3], dict) and "waveform" in out[3])
    check("empty mode returns a zero latent",
          isinstance(out[5], dict) and "samples" in out[5])
    check("empty mode still blocks video (no safe blank)", blocked(out[4]))
    check("empty mode reports found False", out[7] is False)

    try:
        take_all(slot, fallback_mode="error")
        check("error mode raises", False, "no error")
    except ValueError as exc:
        check("error mode raises", True)
        check("error mode explains itself", "Save for Next Time" in str(exc),
              str(exc))


def test_fallback_image():
    seed = torch.rand(1, 16, 16, 3)

    # Empty slot, block mode: image carries the seed, the rest block.
    slot = fresh_slot("seed_slot")
    out = take_all(slot, fallback_image=seed)
    check("seed image fills an empty slot", out[0] is seed)
    check("seed image leaves found False", out[7] is False)
    check("seed image leaves other sockets blocked",
          all(blocked(out[i]) for i in range(1, 6)))

    # Empty mode: the seed beats the 64x64 blank.
    out = take_all(slot, fallback_mode="empty", fallback_image=seed)
    check("seed image beats the blank in empty mode", out[0] is seed)

    # Error mode still stops - the user asked to be stopped.
    try:
        take_all(slot, fallback_mode="error", fallback_image=seed)
        check("error mode ignores the seed image", False, "no error")
    except ValueError:
        check("error mode ignores the seed image", True)

    # An entry without an image also gets the seed, but found is True.
    store.save_entry(slot, {"text": "words only"})
    out = take_all(slot, fallback_image=seed)
    check("seed image fills a text-only entry", out[0] is seed)
    check("a text-only entry still reports found", out[7] is True)

    # A saved image wins over the seed.
    store.save_entry(slot, {"image": torch.rand(1, 8, 8, 3)})
    out = take_all(slot, fallback_image=seed)
    check("a saved image beats the seed",
          out[0] is not seed and tuple(out[0].shape) == (1, 8, 8, 3),
          str(getattr(out[0], "shape", out[0])))


# ---- clearing ----


def test_clear_slot():
    slot = fresh_slot("clear_slot")
    for text in ("one", "two", "three"):
        store.save_entry(slot, {"text": text})
    path = store.slot_dir(slot, create=False)

    # Pin one entry by renaming it, the documented keeper trick.
    victim = store.list_entries(path)[0]
    os.rename(os.path.join(path, victim), os.path.join(path, "keeper"))

    removed, kept = store.clear_slot(slot)
    check("clear removes the numbered entries", len(removed) == 2,
          str(removed))
    check("clear keeps the pinned entry", kept == ["keeper"], str(kept))
    check("only the pin remains on disk",
          store.list_entries(path) == ["keeper"],
          str(store.list_entries(path)))

    removed, kept = store.clear_slot(slot, include_pinned=True)
    check("include_pinned removes the pin", removed == ["keeper"],
          str(removed))
    check("the slot is empty after a full clear",
          store.list_entries(path) == [], str(store.list_entries(path)))

    # A cleared slot behaves like a fresh one: seq restarts safely
    # and Take reports nothing found.
    store.save_entry(slot, {"text": "after clear"})
    out = take_all(slot)
    check("a cleared slot accepts new saves", out[2] == "after clear")

    removed, kept = store.clear_slot("never_used_slot")
    check("clearing a missing slot is a no-op", removed == [] and kept == [])


# ---- caching ----


def test_is_changed():
    slot = fresh_slot("cache_slot")

    missing = TakeFromLastTime.IS_CHANGED(slot_name=slot, steps_back=0)
    check("IS_CHANGED reports missing on an empty slot",
          missing == "missing", repr(missing))

    store.save_entry(slot, {"text": "one"})
    first = TakeFromLastTime.IS_CHANGED(slot_name=slot, steps_back=0)
    again = TakeFromLastTime.IS_CHANGED(slot_name=slot, steps_back=0)
    check("IS_CHANGED is stable when nothing changed", first == again,
          "{!r} vs {!r}".format(first, again))
    check("IS_CHANGED is never NaN", first == first, repr(first))

    store.save_entry(slot, {"text": "two"})
    second = TakeFromLastTime.IS_CHANGED(slot_name=slot, steps_back=0)
    check("IS_CHANGED changes after a new save", first != second,
          "{!r} vs {!r}".format(first, second))

    check("IS_CHANGED distinguishes steps_back",
          TakeFromLastTime.IS_CHANGED(slot_name=slot, steps_back=1) != second)

    linked = TakeFromLastTime.IS_CHANGED(slot_name=slot, steps_back=None)
    check("a linked steps_back falls back to the slot listing",
          isinstance(linked, str) and linked.startswith("dir:"), repr(linked))

    check("IS_CHANGED swallows a bad slot name",
          TakeFromLastTime.IS_CHANGED(slot_name="../nope") == "missing")

    save_token = SaveForNextTime.IS_CHANGED(slot_name=slot)
    check("Save always re-runs (NaN)", save_token != save_token,
          repr(save_token))


# ---- node contract ----


def test_node_contract():
    check("Save has no outputs", SaveForNextTime.RETURN_TYPES == (),
          str(SaveForNextTime.RETURN_TYPES))
    check("Save is an output node", SaveForNextTime.OUTPUT_NODE is True)
    check("Save is not idempotent", SaveForNextTime.NOT_IDEMPOTENT is True)
    check("Take is not marked NOT_IDEMPOTENT",
          not getattr(TakeFromLastTime, "NOT_IDEMPOTENT", False))

    check("Take names every output",
          len(TakeFromLastTime.RETURN_TYPES)
          == len(TakeFromLastTime.RETURN_NAMES)
          == len(TakeFromLastTime.OUTPUT_TOOLTIPS),
          "{} / {} / {}".format(
              len(TakeFromLastTime.RETURN_TYPES),
              len(TakeFromLastTime.RETURN_NAMES),
              len(TakeFromLastTime.OUTPUT_TOOLTIPS)))
    check("Take returns the members first",
          TakeFromLastTime.RETURN_NAMES[:6] == store.MEMBER_NAMES,
          str(TakeFromLastTime.RETURN_NAMES[:6]))

    for cls in (SaveForNextTime, TakeFromLastTime):
        name = cls.__name__
        spec = cls.INPUT_TYPES()
        check("{} declares required inputs".format(name), "required" in spec)
        check("{} is in Trent/Utilities".format(name),
              cls.CATEGORY == "Trent/Utilities", cls.CATEGORY)
        check("{} points FUNCTION at a real method".format(name),
              hasattr(cls, cls.FUNCTION), cls.FUNCTION)
        check("{} has a DESCRIPTION".format(name), bool(cls.DESCRIPTION))

    check("mappings expose both nodes",
          set(nodes_mod.NODE_CLASS_MAPPINGS) == {
              "SaveForNextTime", "TakeFromLastTime"},
          str(set(nodes_mod.NODE_CLASS_MAPPINGS)))
    check("mappings name both nodes",
          set(nodes_mod.NODE_DISPLAY_NAME_MAPPINGS)
          == set(nodes_mod.NODE_CLASS_MAPPINGS))


def test_save_node_end_to_end():
    slot = fresh_slot("node_save_slot")
    result = SaveForNextTime().save(
        slot, max_entries=2, image=torch.rand(1, 8, 8, 3), text="via the node",
        note="a note",
    )
    check("Save returns a ui payload",
          isinstance(result, dict) and "ui" in result, str(result)[:120])
    check("Save previews the saved image", "images" in result["ui"],
          str(result["ui"].keys()))

    path = store.slot_dir(slot, create=False)
    meta = store.read_meta(os.path.join(path, store.list_entries(path)[0]))
    check("the note reaches meta.json", meta.get("note") == "a note",
          str(meta.get("note")))
    check("meta.json records both members",
          sorted(meta.get("members", {})) == ["image", "text"],
          str(meta.get("members")))

    out = take_all(slot)
    check("the node round trips text", out[2] == "via the node")
    check("the node round trips an image", not blocked(out[0]))


if __name__ == "__main__":
    try:
        test_slot_names()
        test_text_round_trip()
        test_image_round_trip()
        test_image_batch_round_trip()
        test_mask_round_trip()
        test_latent_round_trip()
        test_audio_round_trip()
        test_video_round_trip_and_temp_copy()
        test_multi_member_entry()
        test_empty_save_rejected()
        test_pruning_and_steps_back()
        test_seq_survives_a_clock_rewind()
        test_unparseable_names_are_pinned()
        test_incomplete_entries_are_ignored()
        test_stale_staging_is_swept()
        test_entry_name_lookup()
        test_fallback_modes()
        test_fallback_image()
        test_clear_slot()
        test_is_changed()
        test_node_contract()
        test_save_node_end_to_end()
    finally:
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    print("{} passed".format(len(PASS)))
    if FAIL:
        print("FAIL: {} case(s)".format(len(FAIL)))
        for item in FAIL:
            print("  - {}".format(item))
        sys.exit(1)
    print("OK: all for_next_time cases passed")
