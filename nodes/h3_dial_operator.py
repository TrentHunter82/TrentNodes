"""H3 Dial Operator and Image Dial Operator: several ComfyUI-Dials dials on one node, each with its own knob.

A text slot is a saved dial file or two terms (A and B). Terms expand into prompt
pairs over a list of subjects and go through Dials' Make Dial, so a text slot is the
same kind of dial as one made by hand. An image slot is a saved dial file or two
lists of pictures paired by order; the dial is B minus A.

Image methods: "pair" averages every token of each picture read through a few plain
texts over six crops, with no debias (how h3_img_pair_multi was made). The framing
methods read each picture once through a sentence about it, average only the text
tokens and remove the generic wording directions like Make Dial's debias.

All active dials are summed in float32 and applied once with Apply Dial, which
matches applying one combined dial file. Made dials are kept in memory by a key built
from their inputs, and a saved dial stores that key, so a slot that matches a saved
file loads the file instead of running the text encoder again.
"""

import hashlib
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import save_file

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers
import nodes as comfy_nodes

DIAL_DIR = "dials"
TEMPLATES = ("{subject}, {term}.", "A video of {subject}, {term}.",
             "{subject} in a scene, {term}.", "{term}. {subject}.")
BARE_TEMPLATES = ("{term}.", "A video, {term}.", "A scene, {term}.", "A shot, {term}.")  # empty subject list
SUBJECTS = "A young woman\nAn old man\nA dancer\nA golden dog\nA bouquet of flowers\nA human face"
KEY_SUFFIX = {"pre": "diff_b", "post": "dials_shift"}
PAIR_TEXTS = ("A video.", "A person in a scene.", "An animated film scene with music.", "A landscape in motion.")
FRAMINGS = {"look": "A video with the colours, light, lines and texture of <Picture 1>.",
            "content": "A video of what <Picture 1> shows.",
            "bare": "A video of <Picture 1>."}
IMAGE_METHODS = ("pair", *FRAMINGS)
VIEW = (1024, 576)

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
        img = json.loads(meta.get("image_operator", "{}"))
        out.append({"file": name, "name": name[len(DIAL_DIR) + 1:-len(".safetensors")],
                    "a": op.get("a", ""), "b": op.get("b", ""), "key": op.get("key") or img.get("key", ""),
                    "pictures_a": img.get("a", []), "pictures_b": img.get("b", []), "method": img.get("method", ""),
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


def picture_path(name):
    path = os.path.realpath(folder_paths.get_annotated_filepath(name))
    root = os.path.realpath(folder_paths.get_input_directory())
    if os.path.commonpath([path, root]) != root or not os.path.isfile(path):
        raise ValueError(f"picture not found in input: {name}")
    return path


def image_key(slot):
    def sig(name):
        st = os.stat(picture_path(name))
        return [name, st.st_size, st.st_mtime_ns]
    return hashlib.sha1(json.dumps([slot["method"], [sig(x) for x in slot["pictures_a"]], [sig(x) for x in slot["pictures_b"]],
                                    PAIR_TEXTS, FRAMINGS]).encode()).hexdigest()[:12]


def slot_key(slot, subjects):
    return image_key(slot) if slot.get("kind") == "image" else term_key(slot["a"], slot["b"], subjects)


def slot_label(slot):
    if slot.get("name"):
        return slot["name"]
    if slot.get("file"):
        return slot["file"].split("/")[-1][:-len(".safetensors")]
    if slot.get("kind") == "image":
        return f"{slot['method']} {len(slot['pictures_a'])}v{len(slot['pictures_b'])}"
    return slot["a"] + " vs " + slot["b"]


def save_slot(slot, subjects, dial):
    key = slot_key(slot, subjects)
    if slot.get("kind") == "image":
        stem = lambda xs: slug(os.path.splitext(xs[0].split("/")[-1])[0], 16) if xs else "none"
        base = slot.get("name") or f"img_{slot['method']}_{stem(slot['pictures_a'])}_vs_{stem(slot['pictures_b'])}"
        meta = {"image_operator": {"a": slot["pictures_a"], "b": slot["pictures_b"], "method": slot["method"], "key": key}}
    else:
        base = slot.get("name") or f"{slug(slot['a'])}_vs_{slug(slot['b'])}"
        meta = {"operator": {"a": slot["a"], "b": slot["b"], "subjects": subjects, "key": key}}
    name, exists = free_name(base, key)
    if exists:
        return f"{DIAL_DIR}/{name}.safetensors"
    return write(dial, name, meta)


def cached(slot, subjects):
    """The slot's dial without running the text encoder, or None."""
    if slot.get("file"):
        return load_file(slot["file"])
    key = slot_key(slot, subjects)
    if key in made:
        return made[key]
    for d in list_dials():
        if d["key"] == key:
            made[key] = load_file(d["file"])
            return made[key]
    return None


def load_picture(name):
    image = ImageOps.exif_transpose(node_helpers.pillow(Image.open, picture_path(name))).convert("RGB")
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]


def fit(image, w, h):
    return comfy.utils.common_upscale(image.movedim(-1, 1), w, h, "lanczos", "center").movedim(1, -1).clamp(0, 1)


def views(image):
    # The whole picture and five 70% crops (corners and centre), on a 16:9 canvas.
    image = fit(image, 1456, 816)
    w, h = int(1456 * .7), int(816 * .7)
    crops = [image] + [image[:, y:y + h, x:x + w] for x, y in ((0, 0), (1456 - w, 0), (0, 816 - h), (1456 - w, 816 - h), ((1456 - w) // 2, (816 - h) // 2))]
    return [fit(c, *VIEW) for c in crops]


def make_image_dial(clip, model, slot):
    dials = sys.modules[dials_node("DialsMakeDial").__module__]
    _, proj, refiner = dials.text_projection(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype_inference()
    method = slot["method"]

    def encode(text, image=None):
        ctx, extra = clip.encode_from_tokens_scheduled(clip.tokenize(text, images=[] if image is None else [image]))[0]
        comfy.model_management.load_models_gpu([model])
        h = refiner(proj(ctx[0].to(device, dtype)))
        if method != "pair" and image is not None:  # framings read out only the text tokens
            h = h[torch.as_tensor(extra["minimax_token_tags"], device=device).reshape(-1) == 1]
        return h.float().mean(0).cpu()

    def side(names):
        if method == "pair":
            return torch.stack([torch.stack([encode(t, v) for v in views(load_picture(n)) for t in PAIR_TEXTS]).mean(0) for n in names])
        return torch.stack([encode(FRAMINGS[method], fit(load_picture(n), *VIEW)) for n in names])

    a, b = side(slot["pictures_a"]), side(slot["pictures_b"])
    paired = len(a) == len(b)
    diffs = b - a if paired else None
    vec = diffs.mean(0) if paired else b.mean(0) - a.mean(0)
    removed = "off"
    if method != "pair":
        nuisance = dials.parse_pairs("\n".join(dials.NUISANCE_PAIRS), dials.NUISANCE_SUBJECTS)
        basis = torch.linalg.svd(torch.stack([encode(y) - encode(x) for x, y in nuisance]), full_matrices=False)[2][:3].T
        raw = vec
        vec = vec - basis @ (basis.T @ vec)
        diffs = diffs - (diffs @ basis) @ basis.T if paired else None
        removed = f"{1 - float(vec.norm() ** 2 / raw.norm() ** 2):.3f}"
    agree = f"{torch.nn.functional.cosine_similarity(diffs, vec[None]).mean().item():.3f}" if paired else "n/a"
    return {"vec": vec, "target": "token_refiner", "mode": "post",
            "meta": {"name": slot_label(slot), "source": f"image dial ({method})", "n_pairs": str(len(a)) if paired else "0",
                     "pair_agreement_cos": agree, "inject": "post_refiner (hook)", "debias_removed": removed,
                     "calibrated": "False", "format": "ComfyUI-Dials v1"}}


def resolve(slot, subjects, clip, model):
    dial = cached(slot, subjects)
    if dial is not None:
        return dial
    image = slot.get("kind") == "image"
    if image and not (slot.get("pictures_a") and slot.get("pictures_b")):
        raise ValueError(f"image dial '{slot_label(slot)}' needs pictures on both sides, or a saved concept")
    if not image and (not slot.get("a", "").strip() or not slot.get("b", "").strip()):
        raise ValueError(f"dial '{slot.get('name') or 'new'}' needs both an A term and a B term, or a saved concept")
    if clip is None:
        raise ValueError(f"dial '{slot_label(slot)}' is not made yet, so connect clip")
    key = slot_key(slot, subjects)
    if image:
        made[key] = make_image_dial(clip, model, slot)
    else:
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
        inp = state.get("input", {})
        extra = []
        if dial is not None and inp.get("on", True) and inp.get("strength", 1.0) != 0:
            extra.append((dial["meta"].get("name", "input"), float(inp.get("strength", 1.0)), dial))
        return run(model, state, subjects, clip, extra)


def run(model, state, subjects, clip, extra):
    parts, report = [], []
    for slot in state.get("slots", []):
        active = slot.get("on", True) and slot.get("strength", 0) != 0
        if not active and not slot.get("save"):
            continue
        d = resolve(slot, subjects, clip, model)
        entry = {"id": slot["id"]}
        if slot.get("save") and not slot.get("file"):
            entry["saved"] = save_slot(slot, subjects, d)
        if active:
            parts.append((slot_label(slot), float(slot["strength"]), d))
        report.append(entry)
    parts += extra

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


class H3ImageDialOperator:
    DISPLAY_NAME = "Image Dial Operator"
    DESCRIPTION = ("Image dials on one node. Each slot is a saved dial or two sets of pictures, A and B, paired by "
                   "order; the knob moves toward A (negative) or B (positive). Method 'pair' is the tested one; "
                   "look / content / bare read each picture through a sentence. New pictures need clip. "
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
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Only needed to make a dial from pictures that are not saved yet."}),
            },
        }

    def apply(self, model, dials, clip=None):
        return run(model, json.loads(dials), "", clip, [])


def save_now(slot, subjects):
    """Save a slot from memory or from a matching file; None when it has to be made on the next run."""
    dial = cached(slot, subjects)
    return None if dial is None else save_slot(slot, subjects, dial)


NODE_CLASS_MAPPINGS = {"H3DialOperator": H3DialOperator, "H3ImageDialOperator": H3ImageDialOperator}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DialOperator": "H3 Dial Operator", "H3ImageDialOperator": "Image Dial Operator"}
