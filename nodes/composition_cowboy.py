"""Composition Cowboy — layout-preset prompt builder.

Wrangles curated graphic-design layouts (magazine covers, comic pages, book covers,
posters) into an Ideogram-4-compatible JSON caption + a plain natural-language layout
prompt. Presets are selectable or auto-iterable on auto-queue, each region carries a
built-in description so it works with zero input, and per-region content can be overridden
manually or piped in from an external VLM node.

No model is loaded or called here — all VLM interaction is via sockets:
  * `vlm_brief` output  -> feed to your VLM node alongside a reference image
  * `overrides_json` in -> wire the VLM's returned JSON back to fill regions
  * `import_json` in    -> a full Ideogram-4 caption from another node / KJ's editor
  * `bboxes` in         -> region geometry from a grounding / detection node

Several pure helpers (`_render_preview`, `_norm_bbox`, `_dumps`, ...) are adapted from
KJNodes' Ideogram 4 Prompt Builder so the JSON round-trips cleanly into that node.
"""

import copy
import glob
import json
import math
import os
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


_PACK_DIR = os.path.dirname(os.path.dirname(__file__))
_FONT_PATH = os.path.join(_PACK_DIR, "fonts", "FreeMono.ttf")
_PRESET_DIR = os.path.join(_PACK_DIR, "presets", "layout")

# Quick per-role override widgets (also the VLM brief's role vocabulary).
ROLE_SLOTS = ["title", "subtitle", "hero", "body", "brand", "extra"]
# Display + iteration order for categories.
CATEGORY_ORDER = ["Magazine", "Comic", "Book", "Poster", "Custom"]

# A built-in fallback so the node always loads even if the preset dir is missing.
_FALLBACK_PRESET = {
    "id": "fallback_thirds",
    "category": "Poster",
    "name": "Rule of Thirds (fallback)",
    "aspect": {"width": 1024, "height": 1024},
    "image_kind": "photo",
    "background": "simple composition on a rule-of-thirds grid",
    "boxes": [
        {"role": "hero", "type": "obj", "x": 0.08, "y": 0.08, "w": 0.84, "h": 0.66,
         "desc": "main subject placed on a rule-of-thirds intersection"},
        {"role": "title", "type": "text", "x": 0.10, "y": 0.80, "w": 0.80, "h": 0.12,
         "text": "TITLE", "desc": "headline along the lower third"},
    ],
}

# A built-in "build it yourself" preset: starts empty. You supply regions via the
# elements_json widget, the bboxes socket, or import_json, plus the global style fields.
_USER_PRESET = {
    "id": "user_defined",
    "category": "Custom",
    "name": "User Defined",
    "aspect": {"width": 1024, "height": 1024},
    "image_kind": "photo",
    "background": "",
    "boxes": [],
}


# --------------------------------------------------------------------------------------
# Rendering helpers (adapted from KJNodes/nodes/ideogram4_nodes.py)
# --------------------------------------------------------------------------------------
def _hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) if len(h) == 6 else (255, 255, 255)


def _readable(rgb):
    # Lighten toward white if too dark, so box-colored text stays legible on the dark canvas.
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 130:
        t = (130 - lum) / (255 - lum)
        r, g, b = round(r + (255 - r) * t), round(g + (255 - g) * t), round(b + (255 - b) * t)
    return (r, g, b)


def _font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = word if not line else line + " " + word
            if line and draw.textlength(test, font=font) > max_w:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def _render_preview(boxes, width, height, bg=None, brightness=50, caption=None):
    # Render the regions + prompts over the reference image (or a black canvas).
    if bg is not None:
        iw, ih = bg.size
        long_edge = max(iw, ih)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        base = bg.convert("RGB").resize((rw, rh), Image.LANCZOS)
        if brightness < 100:
            base = ImageEnhance.Brightness(base).enhance(max(0.0, brightness / 100.0))
        img = base.convert("RGBA")
    else:
        long_edge = max(width, height)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw = max(1, round(width * scale))
        rh = max(1, round(height * scale))
        img = Image.new("RGBA", (rw, rh), (0, 0, 0, 255))
    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fs = max(10, round(rh / 64))
    font = _font(fs)
    tag_font = _font(max(9, fs - 2))
    lh = fs + 2

    for i, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get("nobbox"):
            continue
        palette = [c for c in (box.get("palette") or []) if c]
        r, g, b = _hex_rgb(palette[0]) if palette else (140, 140, 140)
        x1 = max(0, min(rw, round(box.get("x", 0) * rw)))
        y1 = max(0, min(rh, round(box.get("y", 0) * rh)))
        x2 = max(0, min(rw, round((box.get("x", 0) + box.get("w", 0)) * rw)))
        y2 = max(0, min(rh, round((box.get("y", 0) + box.get("h", 0)) * rh)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)

        pal5 = palette[:5]
        if pal5 and (x2 - x1) > 2:
            sh = max(5, fs // 2)
            seg = (x2 - x1) / len(pal5)
            for p, hexc in enumerate(pal5):
                sx = x1 + round(p * seg)
                draw.rectangle([sx, y1, x1 + round((p + 1) * seg), y1 + sh], fill=_hex_rgb(hexc))

        etype = "text" if box.get("type") == "text" else "obj"
        tag = "%s:%s" % (str(i + 1).zfill(2), box.get("role", etype))
        tw = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + fs + 2], fill=(r, g, b, 255))
        tagfill = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x1 + 3, y1 + 1), tag, fill=tagfill, font=tag_font)

        body = box.get("desc", "") or ""
        if etype == "text" and box.get("text"):
            body = '"%s"%s' % (box["text"], " — " + body if body else "")
        if body and (x2 - x1) > 8:
            ty = y1 + fs + 5
            for line in _wrap(draw, body, font, x2 - x1 - 8):
                if ty > y2:
                    break
                draw.text((x1 + 4, ty), line, fill=_readable((r, g, b)) + (255,), font=font)
                ty += lh

    if caption:
        cfs = max(11, round(rh / 50))
        cfont = _font(cfs)
        bar_h = cfs + 8
        draw.rectangle([0, 0, rw, bar_h], fill=(20, 20, 20, 210))
        cap = caption if draw.textlength(caption, font=cfont) < rw - 12 else caption[:60] + "…"
        draw.text((6, 4), cap, fill=(255, 255, 255, 255), font=cfont)

    img = Image.alpha_composite(img, overlay).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _norm_bbox(box):
    # Normalized {x, y, w, h} fractions (0-1) -> [ymin, xmin, ymax, xmax] on a 0-1000 grid.
    def c(v):
        return max(0, min(1000, round(v * 1000)))
    x, y, w, h = box.get("x", 0.0), box.get("y", 0.0), box.get("w", 0.0), box.get("h", 0.0)
    ymin, xmin, ymax, xmax = c(y), c(x), c(y + h), c(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _palette(colors):
    if isinstance(colors, dict):
        colors = colors.values()
    return [c.upper() for c in colors if c]


def _dumps(v, lvl=0):
    # Like json.dumps(ensure_ascii=False, indent=4), but scalar arrays stay on one line.
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(_dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + _dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + _dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def _parse_json_list(s):
    if s:
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _parse_json_obj(s):
    if s and isinstance(s, str) and s.strip():
        try:
            v = json.loads(s)
            if isinstance(v, dict):
                return v
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _caption_to_boxes(cap):
    # Ideogram caption dict -> editor box list ({x,y,w,h, type, text, desc, palette}).
    cd = cap.get("compositional_deconstruction") or {}
    boxes = []
    for el in (cd.get("elements") or []):
        if not isinstance(el, dict):
            continue
        box = {"type": "text" if el.get("type") == "text" else "obj",
               "text": el.get("text", "") or "", "desc": el.get("desc", "") or "",
               "palette": list(el.get("color_palette") or [])}
        bb = el.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4 and _all_finite(*bb):
            ymin, xmin, ymax, xmax = (float(v) for v in bb)
            box.update(x=xmin / 1000.0, y=ymin / 1000.0,
                       w=(xmax - xmin) / 1000.0, h=(ymax - ymin) / 1000.0)
        else:
            box.update(x=0.03, y=0.03, w=0.22, h=0.14, nobbox=True)
        boxes.append(box)
    return boxes


def _parse_palette(s):
    # "#fff, #000" / "#fff #000" / ["#fff", "#000"] -> ["#FFF...", ...] (deduped, upper).
    if not s:
        return []
    if isinstance(s, (list, tuple)):
        toks = list(s)
    else:
        v = None
        try:
            v = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            v = None
        toks = v if isinstance(v, list) else re.split(r"[\s,;]+", str(s).strip())
    return _palette([str(t).strip() for t in toks if str(t).strip()])


def _all_finite(*vals):
    # JSON allows Infinity/NaN (json.loads parses them), and round(inf) raises — so any geometry
    # coming from elements_json / import_json / a bbox socket must be screened before it reaches
    # _norm_bbox / _render_preview / the pixel-bbox output.
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


def _coerce_box(d):
    # Forgiving region dict -> internal box {role, type, x, y, w, h, text, desc, palette[, nobbox]}.
    if not isinstance(d, dict):
        return None
    etype = "text" if d.get("type") == "text" else "obj"
    box = {"role": d.get("role", etype), "type": etype,
           "text": d.get("text", "") or "", "desc": d.get("desc", "") or "",
           "palette": list(d.get("palette") or d.get("color_palette") or [])}
    try:
        if all(k in d for k in ("x", "y", "w", "h")) and _all_finite(d["x"], d["y"], d["w"], d["h"]):
            box.update(x=float(d["x"]), y=float(d["y"]), w=float(d["w"]), h=float(d["h"]))
        elif isinstance(d.get("bbox"), (list, tuple)) and len(d["bbox"]) == 4 and _all_finite(*d["bbox"]):
            ymin, xmin, ymax, xmax = (float(v) for v in d["bbox"])  # [ymin,xmin,ymax,xmax] on 0-1000
            box.update(x=xmin / 1000.0, y=ymin / 1000.0,
                       w=(xmax - xmin) / 1000.0, h=(ymax - ymin) / 1000.0)
        else:
            box.update(x=0.03, y=0.03, w=0.22, h=0.14, nobbox=True)
    except (TypeError, ValueError):
        box.update(x=0.03, y=0.03, w=0.22, h=0.14, nobbox=True)
    return box


def _position_words(box):
    # Coarse human-readable position from the box center.
    cx = box.get("x", 0.0) + box.get("w", 0.0) / 2.0
    cy = box.get("y", 0.0) + box.get("h", 0.0) / 2.0
    v = "top" if cy < 0.34 else ("bottom" if cy > 0.66 else "middle")
    h = "left" if cx < 0.34 else ("right" if cx > 0.66 else "center")
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    if h == "center":
        return v
    return "%s-%s" % (v, h)


# --------------------------------------------------------------------------------------
# Preset loading (cached at import; rebuilt only on restart)
# --------------------------------------------------------------------------------------
_PRESETS_CACHE = None


def _category_rank(cat):
    return CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)


def _load_presets():
    global _PRESETS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE
    presets = []
    for path in sorted(glob.glob(os.path.join(_PRESET_DIR, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("[CompositionCowboy] Failed to load %s: %s" % (path, e))
            continue
        for p in (data.get("presets") or []):
            if isinstance(p, dict) and p.get("boxes"):
                presets.append(p)
    if not presets:
        presets = [_FALLBACK_PRESET]
    presets.append(copy.deepcopy(_USER_PRESET))
    # Stable sort by category order, preserving within-file order.
    presets.sort(key=lambda p: _category_rank(p.get("category", "")))
    _PRESETS_CACHE = presets
    return presets


def _preset_label(p):
    return "%s / %s" % (p.get("category", "?"), p.get("name", p.get("id", "preset")))


def _pool_options(presets):
    cats = []
    for p in presets:
        c = p.get("category", "")
        if c and c not in cats:
            cats.append(c)
    cats.sort(key=_category_rank)
    return ["All"] + cats


# --------------------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------------------
class CompositionCowboy:
    """Layout-preset prompt builder with VLM piping and optional bboxes."""

    @classmethod
    def INPUT_TYPES(cls):
        presets = _load_presets()
        labels = [_preset_label(p) for p in presets]
        pools = _pool_options(presets)
        return {
            "required": {
                "preset": (labels, {"default": labels[0],
                                    "tooltip": "Pick a layout template (used when select_by = name)."}),
                "select_by": (["name", "index"], {"default": "name",
                              "tooltip": "Which selector wins: the 'preset' dropdown (name) or 'preset_index'."}),
                "pool": (pools, {"default": "All",
                         "tooltip": "Scopes which presets 'preset_index' cycles through."}),
                "preset_index": ("INT", {"default": 0, "min": 0, "max": 9999, "control_after_generate": True,
                                 "tooltip": "Index into the pool (used when select_by = index). Set the "
                                            "control dropdown to 'increment' and turn on Auto Queue to cycle "
                                            "styles each run. Wraps around the pool."}),
                "include_bboxes": ("BOOLEAN", {"default": True,
                                   "tooltip": "On: emit Ideogram-style region bboxes. Off: coordinate-free "
                                              "prompt (for models that don't use a region grid) and empty bboxes output."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                          "tooltip": "0 = use the preset's aspect; >0 overrides it (multiples of 16)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                           "tooltip": "0 = use the preset's aspect; >0 overrides it (multiples of 16)."}),
                "json_mode": (["replace", "merge"], {"default": "replace",
                              "tooltip": "How a wired import_json combines with the preset."}),
                "bg_brightness": ("INT", {"default": 25, "min": 0, "max": 100,
                                  "tooltip": "Dim the reference image behind the preview (%)."}),
            },
            "optional": {
                "title": ("STRING", {"default": "", "multiline": True,
                          "tooltip": "Override the 'title' regions (text = words shown; obj = subject)."}),
                "subtitle": ("STRING", {"default": "", "multiline": True}),
                "hero": ("STRING", {"default": "", "multiline": True,
                         "tooltip": "Override the 'hero' / main-subject regions."}),
                "body": ("STRING", {"default": "", "multiline": True}),
                "brand": ("STRING", {"default": "", "multiline": True}),
                "extra": ("STRING", {"default": "", "multiline": True}),
                "high_level_description": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "One-line overview of the whole image (KJ parity). Blank = auto from the preset."}),
                "background": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Scene / background description (KJ parity). Blank = the preset's built-in background."}),
                "style": (["none", "photo", "art_style"], {"default": "none",
                    "tooltip": "Emit an Ideogram style_description block (KJ parity). 'none' omits it entirely."}),
                "style_detail": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "The 'photo' text (camera/film/lens) or the 'art_style' text, per the style selector."}),
                "aesthetics": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "style_description.aesthetics (used only when style != none)."}),
                "lighting": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "style_description.lighting (used only when style != none)."}),
                "medium": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "style_description.medium (used only when style != none)."}),
                "style_palette": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Global style colors as hex, e.g. '#1A1A1A, #E0C080'. Added to style_description.color_palette."}),
                "overrides_json": ("STRING", {"default": "", "multiline": True,
                                   "tooltip": "Per-region overrides as JSON, e.g. {\"hero\":\"...\",\"2\":\"...\"}. "
                                              "Role keys hit all matching regions; 1-based index keys hit one. "
                                              "Wire your VLM node's JSON output here (convert to input)."}),
                "import_json": ("STRING", {"default": "", "multiline": True,
                                "tooltip": "A full Ideogram-4 caption JSON to seed from (e.g. from KJ's editor "
                                           "or a VLM). Combined per json_mode. Convert to input to wire it."}),
                "elements_json": ("STRING", {"default": "", "multiline": True,
                                  "tooltip": "Define your own regions as a JSON list, e.g. "
                                             "[{\"type\":\"text\",\"x\":0.1,\"y\":0.1,\"w\":0.8,\"h\":0.1,\"text\":\"HELLO\"}]. "
                                             "Appended to the active preset's regions — pick the 'Custom / User Defined' "
                                             "preset to start from an empty page. Each region takes x/y/w/h as 0-1 "
                                             "fractions, or a KJ-style bbox [ymin,xmin,ymax,xmax] on a 0-1000 grid; "
                                             "plus optional type/text/desc/palette/role."}),
                "image": ("IMAGE", {"tooltip": "Optional reference image shown behind the preview."}),
                "bboxes": ("BOUNDING_BOX", {"tooltip": "Optional pixel-space boxes from a detection/grounding "
                                            "node; seeds regions when the preset has none, else repositions by index."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "BOUNDING_BOX", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("prompt_json", "prompt_text", "preview", "bboxes", "width", "height", "preset_name", "vlm_brief")
    FUNCTION = "build"
    CATEGORY = "TrentNodes/Prompt"
    DESCRIPTION = (
        "Build print/cover layout prompts from curated, design-best-practice templates "
        "(magazine, comic, book, poster). Pick a preset or set select_by=index + the "
        "preset_index 'increment' control + Auto Queue to iterate styles automatically. "
        "Fill the role slots or pipe descriptions in from any VLM node via overrides_json "
        "(use the vlm_brief output to instruct it). Outputs Ideogram-4 JSON (wire into KJ's "
        "Ideogram 4 Prompt Builder import_json) and a plain-language layout prompt. "
        "Toggle include_bboxes off for models without a region grid. "
        "For full control pick 'Custom / User Defined' and supply your own regions via "
        "elements_json / the bboxes socket / import_json, plus the KJ-parity style fields "
        "(high_level_description, background, style + aesthetics/lighting/medium/style_palette) "
        "which are available on every preset."
    )

    # ---- content application --------------------------------------------------------
    @staticmethod
    def _apply_content(box, value):
        if not value or not value.strip():
            return
        if box.get("type") == "text":
            box["text"] = value
        else:
            box["desc"] = value

    def _apply_overrides(self, boxes, slots, overrides):
        # 1) role-slot widgets
        for role, value in slots.items():
            if value and value.strip():
                for box in boxes:
                    if box.get("role") == role:
                        self._apply_content(box, value)
        # 2) overrides_json role keys, then index keys (index beats role)
        for key, value in overrides.items():
            if not isinstance(value, str) or str(key).isdigit():
                continue
            for box in boxes:
                if box.get("role") == key:
                    self._apply_content(box, value)
        for key, value in overrides.items():
            if isinstance(value, str) and str(key).isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(boxes):
                    self._apply_content(boxes[idx], value)

    @staticmethod
    def _apply_bboxes(boxes, bboxes, width, height):
        if not bboxes:
            return boxes
        if isinstance(bboxes, dict):
            frame = [bboxes]
        elif bboxes and isinstance(bboxes[0], (list, tuple)):
            frame = bboxes[0]
        else:
            frame = bboxes
        w = max(1, width)
        h = max(1, height)
        for i, bb in enumerate(frame):
            if not isinstance(bb, dict):
                continue
            if not _all_finite(bb.get("x", 0), bb.get("y", 0), bb.get("width", 0), bb.get("height", 0)):
                continue
            geom = {"x": float(bb.get("x", 0)) / w, "y": float(bb.get("y", 0)) / h,
                    "w": float(bb.get("width", 0)) / w, "h": float(bb.get("height", 0)) / h}
            if i < len(boxes):
                boxes[i].update(geom)
                boxes[i].pop("nobbox", None)
            else:
                boxes.append({"role": "extra", "type": "obj", "text": "", "desc": "", "palette": [], **geom})
        return boxes

    # ---- output builders ------------------------------------------------------------
    @staticmethod
    def _build_caption(hld, background, boxes, include_bboxes, style_description=None):
        elements = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            etype = "text" if box.get("type") == "text" else "obj"
            elem = {"type": etype}
            if include_bboxes and not box.get("nobbox"):
                elem["bbox"] = _norm_bbox(box)
            if etype == "text":
                elem["text"] = box.get("text", "")
            elem["desc"] = box.get("desc", "")
            pal = _palette(box.get("palette", []))
            if pal:
                elem["color_palette"] = pal[:5]
            elements.append(elem)
        caption = {}
        if hld:
            caption["high_level_description"] = hld
        if style_description:
            caption["style_description"] = style_description
        caption["compositional_deconstruction"] = {"background": background, "elements": elements}
        return caption

    @staticmethod
    def _build_text(hld, background, boxes, width, height, include_bboxes, style_description=None):
        lines = []
        if hld:
            lines.append(hld + (" (%dx%d)" % (width, height) if width and height else ""))
        if background:
            lines.append("Background: %s." % background)
        if style_description:
            parts = ["%s: %s" % (k, v) for k, v in style_description.items() if v and k != "color_palette"]
            if style_description.get("color_palette"):
                parts.append("palette: %s" % ", ".join(style_description["color_palette"]))
            if parts:
                lines.append("Style — " + "; ".join(parts) + ".")
        for box in boxes:
            if not isinstance(box, dict):
                continue
            role = (box.get("role") or box.get("type") or "region").title()
            pos = "" if (not include_bboxes or box.get("nobbox")) else " (%s)" % _position_words(box)
            if box.get("type") == "text":
                txt = box.get("text", "")
                desc = box.get("desc", "")
                content = ('"%s"' % txt if txt else "") + ((" — " + desc) if (txt and desc) else (desc if not txt else ""))
            else:
                content = box.get("desc", "")
            lines.append("- %s%s: %s" % (role, pos, content.strip()))
        return "\n".join(lines)

    @staticmethod
    def _build_brief(preset, boxes, include_bboxes):
        cat = preset.get("category", "layout")
        name = preset.get("name", "")
        head = ("You are filling in a %s layout (%s). For each region below write a concise, vivid "
                "description for an image generator. For TEXT regions return the exact words to display; "
                "for OBJECT regions describe the subject. Return ONLY a JSON object mapping each region "
                "number (as a string) to its description.\n\nRegions:") % (cat, name)
        rlines = []
        skeleton = {}
        for i, box in enumerate(boxes):
            n = str(i + 1)
            etype = "text" if box.get("type") == "text" else "object"
            role = box.get("role", etype)
            pos = "" if (not include_bboxes or box.get("nobbox")) else " at %s" % _position_words(box)
            hint = box.get("desc", "")
            rlines.append('  "%s": %s region (%s)%s — %s' % (n, role, etype, pos, hint))
            skeleton[n] = ""
        return head + "\n" + "\n".join(rlines) + "\n\nReturn JSON like:\n" + json.dumps(skeleton, ensure_ascii=False)

    # ---- main -----------------------------------------------------------------------
    def build(self, preset, select_by, pool, preset_index, include_bboxes, width, height,
              json_mode, bg_brightness, title="", subtitle="", hero="", body="", brand="",
              extra="", high_level_description="", background="", style="none", style_detail="",
              aesthetics="", lighting="", medium="", style_palette="",
              overrides_json="", import_json="", elements_json="", image=None, bboxes=None):
        presets = _load_presets()
        label_to_preset = {_preset_label(p): p for p in presets}

        # Resolve the active preset.
        if select_by == "index":
            pool_list = presets if pool == "All" else [p for p in presets if p.get("category") == pool]
            if not pool_list:
                pool_list = presets
            active = pool_list[int(preset_index) % len(pool_list)]
        else:
            active = label_to_preset.get(preset, presets[0])

        # Resolve dimensions.
        aspect = active.get("aspect") or {}
        out_w = int(width) or int(aspect.get("width", 1024)) or 1024
        out_h = int(height) or int(aspect.get("height", 1024)) or 1024

        # Base boxes + overview from the preset.
        boxes = copy.deepcopy(active.get("boxes", []))
        auto_hld = "%s layout — %s (%s)" % (active.get("category", "Composition"),
                                            active.get("name", ""), active.get("image_kind", "image"))
        hld = auto_hld
        bg_text = active.get("background", "")

        # import_json: a full caption from KJ's editor or a VLM.
        imported = None
        if import_json and import_json.strip():
            try:
                c = json.loads(import_json)
                if isinstance(c, dict) and c.get("compositional_deconstruction"):
                    imported = c
            except json.JSONDecodeError:
                imported = None
        imported_style = None
        if imported is not None:
            imp_boxes = _caption_to_boxes(imported)
            cd = imported.get("compositional_deconstruction") or {}
            sd_in = imported.get("style_description")
            if isinstance(sd_in, dict) and sd_in:
                imported_style = sd_in            # preserved unless a style widget is set below
            if json_mode == "replace":
                boxes = imp_boxes
                if cd.get("background"):           # don't let a blank import clobber the preset bg
                    bg_text = cd["background"]
                if imported.get("high_level_description"):
                    hld = imported["high_level_description"]
            else:
                boxes = boxes + imp_boxes

        # User-defined regions typed into elements_json (a JSON list, or a single dict),
        # appended to whatever's already there. Use the 'User Defined' preset to start empty.
        custom = _parse_json_list(elements_json)
        if not custom and elements_json and elements_json.strip():
            one = _parse_json_obj(elements_json)
            if one:
                custom = [one]
        for d in custom:
            cb = _coerce_box(d)
            if cb:
                boxes.append(cb)

        # Per-region overrides: slot widgets, then overrides_json.
        slots = {"title": title, "subtitle": subtitle, "hero": hero,
                 "body": body, "brand": brand, "extra": extra}
        self._apply_overrides(boxes, slots, _parse_json_obj(overrides_json))

        # Geometry piped in from a detection / grounding node.
        boxes = self._apply_bboxes(boxes, bboxes, out_w, out_h)

        # Global KJ-parity overrides: a non-blank widget wins over the preset / import value.
        if isinstance(background, str) and background.strip():
            bg_text = background
        if isinstance(high_level_description, str) and high_level_description.strip():
            hld = high_level_description

        # Optional Ideogram style_description block (KJ parity), emitted only when style != none.
        # Keys are kept present-but-blank once a style is chosen, matching KJ's verifier order.
        style_description = None
        if style and style != "none":
            sd = {"aesthetics": aesthetics or "", "lighting": lighting or ""}
            if style == "photo":
                sd["photo"] = style_detail or ""
                sd["medium"] = medium or ""
            else:
                sd["medium"] = medium or ""
                sd["art_style"] = style_detail or ""
            pal = _parse_palette(style_palette)
            if pal:
                sd["color_palette"] = pal
            style_description = sd
        elif imported_style:
            style_description = imported_style   # round-trip an imported style when no widget is set

        # Assemble outputs.
        caption = self._build_caption(hld, bg_text, boxes, include_bboxes, style_description)
        prompt_json = _dumps(caption)
        prompt_text = self._build_text(hld, bg_text, boxes, out_w, out_h, include_bboxes, style_description)
        vlm_brief = self._build_brief(active, boxes, include_bboxes)

        # Preview.
        bg = None
        if image is not None:
            try:
                bg = Image.fromarray((image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            except Exception:
                bg = None
        preview = _render_preview(boxes, out_w, out_h, bg, bg_brightness, caption=hld)

        # Pixel-space bboxes (empty when bboxes are turned off).
        bbox_dicts = []
        if include_bboxes:
            for box in boxes:
                if not isinstance(box, dict) or box.get("nobbox"):
                    continue
                x, y = box.get("x", 0.0), box.get("y", 0.0)
                bw, bh = box.get("w", 0.0), box.get("h", 0.0)
                if bw < 0:
                    x += bw
                    bw = -bw
                if bh < 0:
                    y += bh
                    bh = -bh
                bbox_dicts.append({"x": round(x * out_w), "y": round(y * out_h),
                                   "width": max(1, round(bw * out_w)),
                                   "height": max(1, round(bh * out_h))})
        bboxes_out = [bbox_dicts] if bbox_dicts else []

        preset_name = active.get("name", _preset_label(active))
        return (prompt_json, prompt_text, preview, bboxes_out, out_w, out_h, preset_name, vlm_brief)


NODE_CLASS_MAPPINGS = {"CompositionCowboy": CompositionCowboy}
NODE_DISPLAY_NAME_MAPPINGS = {"CompositionCowboy": "🤠 Composition Cowboy"}
