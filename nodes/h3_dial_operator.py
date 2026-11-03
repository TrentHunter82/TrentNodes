"""H3 Dial Operator: several ComfyUI-Dials dials on one node, each with its own knob.

A slot is either a saved dial file or two terms (A and B). Terms expand into prompt
pairs over a list of subjects and go through Dials' Make Dial, so a text slot is the
same kind of dial as one made by hand. All active dials are summed in float32 and
applied once with Apply Dial, which matches applying one combined dial file.

Made dials are kept in memory by a key built from their terms, and a saved dial
stores that key, so a slot whose terms match a saved file loads the file instead of
running the text encoder again.
"""

import hashlib
import json
import os
import re
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import folder_paths
import nodes as comfy_nodes

DIAL_DIR = "dials"
TEMPLATES = ("{subject}, {term}.", "A video of {subject}, {term}.",
             "{subject} in a scene, {term}.", "{term}. {subject}.")
BARE_TEMPLATES = ("{term}.", "A video, {term}.", "A scene, {term}.", "A shot, {term}.")  # empty subject list
SUBJECTS = "A young woman\nAn old man\nA dancer\nA golden dog\nA bouquet of flowers\nA human face"
KEY_SUFFIX = {"pre": "diff_b", "post": "dials_shift"}

made = OrderedDict()  # term key -> dial; a few KB each


def dials_node(name):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError("H3 Dial Operator needs the ComfyUI-Dials node pack (missing node " + name + ")")
    return cls


def term_key(a, b, subjects):
    return hashlib.sha1(json.dumps([a.strip(), b.strip(), subjects.strip(), TEMPLATES]).encode()).hexdigest()[:12]


def term_pairs(a, b, subjects):
    # Lines with {subject} are used as written, paired by line; plain terms go through the templates.
    if "{subject}" in a or "{subject}" in b:
        la, lb = [x.strip() for x in a.splitlines() if x.strip()], [x.strip() for x in b.splitlines() if x.strip()]
        if len(la) != len(lb):
            raise ValueError(f"A has {len(la)} sentence lines and B has {len(lb)}; give each A line a B line")
        return "\n".join(f"{x} ||| {y}" for x, y in zip(la, lb))
    templates = TEMPLATES if subjects.strip() else BARE_TEMPLATES
    return "\n".join(t.format(subject="{subject}", term=a.strip()) + " ||| " + t.format(subject="{subject}", term=b.strip())
                     for t in templates)


def slug(text, size=24):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:size].strip("-") or "x"


def dial_root():
    root = os.path.join(folder_paths.get_folder_paths("loras")[0], DIAL_DIR)
    os.makedirs(root, exist_ok=True)
    return root


def list_dials():
    out = []
    for name in folder_paths.get_filename_list("loras"):
        name = name.replace("\\", "/")
        if not name.startswith(DIAL_DIR + "/") or not name.endswith(".safetensors"):
            continue
        with safe_open(folder_paths.get_full_path_or_raise("loras", name), "pt") as f:
            meta = f.metadata() or {}
        op = json.loads(meta.get("operator", "{}"))
        out.append({"file": name, "name": name[len(DIAL_DIR) + 1:-len(".safetensors")],
                    "a": op.get("a", ""), "b": op.get("b", ""), "key": op.get("key", ""),
                    "mix": meta.get("operator_mix", "")})
    return out


def load_file(name):
    if not name.replace("\\", "/").startswith(DIAL_DIR + "/"):
        raise ValueError(f"not a dial file: {name}")
    return dials_node("DialsLoadDial").execute(name).result[0]


def free_name(base, key=None):
    # The first free <base>, <base>_2, ...; an existing file with the same key is reused.
    names = {d["name"]: d for d in list_dials()}
    for n in range(1, 1000):
        name = base if n == 1 else f"{base}_{n}"
        if name not in names:
            return name, False
        if key and names[name]["key"] == key:
            return name, True
    raise ValueError(f"no free dial name for {base}")


def write(dial, name, meta):
    meta = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in {**dial["meta"], **meta, "name": name}.items()}
    key = f"diffusion_model.{dial['target']}.{KEY_SUFFIX[dial['mode']]}"
    save_file({key: dial["vec"].to(torch.bfloat16).contiguous()}, os.path.join(dial_root(), name + ".safetensors"), metadata=meta)
    return f"{DIAL_DIR}/{name}.safetensors"


def save_slot(slot, subjects, dial):
    key = term_key(slot["a"], slot["b"], subjects)
    name, exists = free_name(slot.get("name") or f"{slug(slot['a'])}_vs_{slug(slot['b'])}", key)
    if exists:
        return f"{DIAL_DIR}/{name}.safetensors"
    return write(dial, name, {"operator": {"a": slot["a"], "b": slot["b"], "subjects": subjects, "key": key}})


def cached(slot, subjects):
    """The slot's dial without running the text encoder, or None."""
    if slot.get("file"):
        return load_file(slot["file"])
    key = term_key(slot["a"], slot["b"], subjects)
    if key in made:
        return made[key]
    for d in list_dials():
        if d["key"] == key:
            made[key] = load_file(d["file"])
            return made[key]
    return None


def resolve(slot, subjects, clip, model):
    dial = cached(slot, subjects)
    if dial is not None:
        return dial
    if not slot.get("a", "").strip() or not slot.get("b", "").strip():
        raise ValueError(f"dial '{slot.get('name') or 'new'}' needs both an A term and a B term, or a saved concept")
    if clip is None:
        raise ValueError(f"dial '{slot['a']} -> {slot['b']}' is made from text, so connect clip")
    key = term_key(slot["a"], slot["b"], subjects)
    made[key] = dials_node("DialsMakeDial").execute(clip, model, term_pairs(slot["a"], slot["b"], subjects), subjects,
                                                     f"{slug(slot['a'])}_vs_{slug(slot['b'])}", "post_refiner (hook)", False).result[0]
    while len(made) > 64:
        made.popitem(last=False)
    return made[key]


def mix_name(parts):
    return "mix_" + "_".join(f"{slug(label, 12)}{strength:+g}" for label, strength, _ in parts)[:70]


class H3DialOperator:
    DISPLAY_NAME = "H3 Dial Operator"
    DESCRIPTION = ("Several dials on one node. Each slot is a saved dial or two terms, A and B; the knob moves "
                   "toward A (negative) or B (positive). Terms need clip the first time; saved dials do not. "
                   "Needs ComfyUI-Dials.")
    CATEGORY = "Trent/MiniMax"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "dials": ("STRING", {"default": '{"slots": []}', "hidden": True}),
                "subjects": ("STRING", {"default": SUBJECTS, "multiline": True, "advanced": True,
                    "tooltip": "Subjects that text dials are averaged over, one per line, so only the A -> B change is left."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Only needed to make a dial from terms that are not saved yet."}),
                "dial": ("DIAL", {"tooltip": "An extra dial from another node. It gets its own knob."}),
            },
        }

    def apply(self, model, dials, subjects, clip=None, dial=None):
        state = json.loads(dials)
        parts, report = [], []
        for slot in state.get("slots", []):
            active = slot.get("on", True) and slot.get("strength", 0) != 0
            if not active and not slot.get("save"):
                continue
            d = resolve(slot, subjects, clip, model)
            label = slot.get("name") or (slot["file"].split("/")[-1][:-len(".safetensors")] if slot.get("file") else slot["a"] + " vs " + slot["b"])
            entry = {"id": slot["id"]}
            if slot.get("save") and not slot.get("file"):
                entry["saved"] = save_slot(slot, subjects, d)
            if active:
                parts.append((label, float(slot["strength"]), d))
            report.append(entry)
        inp = state.get("input", {})
        if dial is not None and inp.get("on", True) and inp.get("strength", 1.0) != 0:
            parts.append((dial["meta"].get("name", "input"), float(inp.get("strength", 1.0)), dial))

        groups = {}
        for _, strength, d in parts:
            g = groups.setdefault((d["target"], d["mode"]), {"vec": 0, "target": d["target"], "mode": d["mode"], "meta": {}})
            g["vec"] = g["vec"] + d["vec"].float() * strength
        apply_dial = dials_node("DialsApplyDial")
        for g in groups.values():
            model = apply_dial.execute(model, g, 1.0).result[0]

        ui = {"slots": report}
        if state.get("save_mix") and parts:
            if len(groups) > 1:
                raise ValueError("these dials use different injection points, so they cannot be saved as one mix")
            g = next(iter(groups.values()))
            name, _ = free_name(slug(state.get("mix_name", ""), 60) if state.get("mix_name", "").strip() else mix_name(parts))
            ui["mix"] = write(g, name, {"operator_mix": [[label, strength] for label, strength, _ in parts]})
        return {"ui": {"h3_dial_operator": [ui]}, "result": (model,)}


def save_now(slot, subjects):
    """Save a slot from memory or from a matching file; None when it has to be made on the next run."""
    dial = cached(slot, subjects)
    return None if dial is None else save_slot(slot, subjects, dial)


NODE_CLASS_MAPPINGS = {"H3DialOperator": H3DialOperator}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DialOperator": "H3 Dial Operator"}
