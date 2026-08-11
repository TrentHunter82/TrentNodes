"""Render ComfyUI-style node card PNGs for the README.

Regenerates assets/images/nodes/<Class>.png for every node in the pack,
drawn to match the ComfyUI default dark theme (slot colors taken from the
shipped frontend palette). Run it from the ComfyUI venv after adding or
changing nodes, then reference new cards in README.md as:

    <img src="assets/images/nodes/<Class>.png" width="<W>" alt="<Name> node">

The printed report lists each card's CSS width (half the PNG width).

Usage:
    cd ComfyUI && venv/bin/python custom_nodes/TrentNodes/tools/render_node_cards.py
"""
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = REPO.parent.parent
OUT_DIR = REPO / "assets" / "images" / "nodes"

# machine-specific combo defaults -> generic examples for public docs
VALUE_OVERRIDES = {
    ("VRAMGatedCheckpointLoader", "ckpt_name"): "sd_xl_base_1.0.safetensors",
    ("VRAMGatedVAELoader", "vae_name"): "ae.safetensors",
    ("VRAMGatedUNETLoader", "unet_name"): "flux1-dev.safetensors",
    ("VRAMGatedLoraLoaderModelOnly", "lora_name"): "my_lora.safetensors",
}

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

S = 2  # supersample scale; embed at half the PNG width for crisp text

BODY = (53, 53, 53, 255)
TITLE_BG = (48, 48, 48, 255)
OUTLINE = (24, 24, 24, 255)
TITLE_TEXT = (222, 222, 222, 255)
SLOT_TEXT = (170, 170, 170, 255)
WIDGET_BG = (34, 34, 34, 255)
WIDGET_OUTLINE = (102, 102, 102, 255)
WIDGET_NAME = (153, 153, 153, 255)
WIDGET_VAL = (221, 221, 221, 255)
ARROW = (190, 190, 190, 255)
PLACEHOLDER = (120, 120, 120, 255)
SLOT_STROKE = (22, 22, 22, 255)

# ComfyUI "Dark (Default)" node_slot palette; unlisted types render gray
TYPE_COLORS = {
    "IMAGE": "#64B5F6", "MASK": "#81C784", "LATENT": "#FF9CF9",
    "CONDITIONING": "#FFA931", "MODEL": "#B39DDB", "CLIP": "#FFD500",
    "VAE": "#FF6E6E", "CLIP_VISION": "#A8DADC", "CLIP_VISION_OUTPUT": "#AD7452",
    "CONTROL_NET": "#6EE7B7", "STYLE_MODEL": "#C2FFAE", "TAESD": "#DCC274",
    "NOISE": "#B0B0B0", "GUIDER": "#66FFFF", "SAMPLER": "#ECB4B4",
    "SIGMAS": "#CDFFCD",
}
DEFAULT_SLOT = "#999999"

TITLE_H = 27 * S
SLOT_H = 20 * S
WIDGET_H = 20 * S
WIDGET_GAP = 4 * S
MULTI_H = 58 * S
INSET = 10 * S
RADIUS = 8 * S
MARGIN = 14 * S
MIN_W = 205 * S
MAX_W = 340 * S


def load_font(bold, size):
    candidates = ["/mnt/c/Windows/Fonts/arialbd.ttf" if bold else
                  "/mnt/c/Windows/Fonts/arial.ttf"]
    try:
        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        candidates.append(str(ttf / ("DejaVuSans-Bold.ttf" if bold
                                     else "DejaVuSans.ttf")))
    except ImportError:
        pass
    candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
                      % ("-Bold" if bold else ""))
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    raise FileNotFoundError("no usable sans font found")


f_title = load_font(True, 14 * S)
f_slot = load_font(False, 12 * S)
f_widget = load_font(False, 12 * S)
f_multi = load_font(False, 11 * S)


def load_pack():
    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(COMFY_ROOT / "custom_nodes"))
    import server as comfy_server
    if getattr(comfy_server.PromptServer, "instance", None) is None:
        from types import SimpleNamespace
        from aiohttp import web
        comfy_server.PromptServer.instance = SimpleNamespace(
            routes=web.RouteTableDef(), app=web.Application())
    import TrentNodes
    return TrentNodes.NODE_CLASS_MAPPINGS, TrentNodes.NODE_DISPLAY_NAME_MAPPINGS


def classify(name, spec):
    if not isinstance(spec, (tuple, list)) or len(spec) == 0:
        return {"name": name, "kind": "socket", "type": str(spec)}
    typ = spec[0]
    cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(typ, (list, tuple)):
        choices = [str(c) for c in typ]
        default = cfg.get("default", choices[0] if choices else "")
        return {"name": name, "kind": "combo", "type": "COMBO",
                "value": str(default)}
    typ = str(typ)
    if typ in WIDGET_TYPES and not cfg.get("forceInput"):
        default = cfg.get("default", "")
        if isinstance(default, (dict, list)):
            default = str(default)
        return {"name": name, "kind": "widget", "type": typ, "value": default,
                "multiline": bool(cfg.get("multiline")),
                "placeholder": cfg.get("placeholder", "")}
    return {"name": name, "kind": "socket", "type": typ}


def node_schema(cls_name, cls, display):
    it = cls.INPUT_TYPES()
    inputs = []
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            inputs.append(classify(name, spec))
    ret_types = [str(t) if not isinstance(t, (list, tuple)) else "COMBO"
                 for t in getattr(cls, "RETURN_TYPES", ())]
    ret_names = list(getattr(cls, "RETURN_NAMES", ()) or ())
    outputs = [{"name": str(ret_names[i]) if i < len(ret_names) else t,
                "type": t} for i, t in enumerate(ret_types)]
    return {"class": cls_name, "display": display, "inputs": inputs,
            "outputs": outputs}


def hex_rgba(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


def tw(font, text):
    return font.getlength(text)


def trunc(font, text, maxw):
    if tw(font, text) <= maxw:
        return text
    while text and tw(font, text + "…") > maxw:
        text = text[:-1]
    return text + "…"


def fmt_val(slot):
    t, v = slot["type"], slot.get("value", "")
    if t == "INT":
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return str(v)
    if t == "FLOAT":
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        s = f"{fv:.2f}"
        if abs(float(s) - fv) > 1e-9:
            s = f"{fv:.3f}"
        if abs(float(s) - fv) > 1e-9:
            s = f"{fv:g}"
        return s
    if t == "BOOLEAN":
        return "true" if v in (True, "True", "true", 1) else "false"
    return str(v)


def wrap_px(font, text, maxw, max_lines):
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split(" "):
            cand = (cur + " " + w).strip()
            if tw(font, cand) <= maxw or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = trunc(font, lines[-1] + " …", maxw)
    return lines


def render(node):
    cls = node["class"]
    title = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", node["display"]).strip()

    in_sockets = [i for i in node["inputs"] if i["kind"] == "socket"]
    widgets = [i for i in node["inputs"] if i["kind"] != "socket"]
    outputs = node["outputs"]

    for wdg in widgets:
        key = (cls, wdg["name"])
        if key in VALUE_OVERRIDES:
            wdg["value"] = VALUE_OVERRIDES[key]
        wdg["_val"] = fmt_val(wdg)

    need = MIN_W
    need = max(need, tw(f_title, title) + 28 * S)
    rows = max(len(in_sockets), len(outputs))
    for i in range(rows):
        w = 44 * S
        if i < len(in_sockets):
            w += tw(f_slot, in_sockets[i]["name"])
        if i < len(outputs):
            w += tw(f_slot, outputs[i]["name"])
        need = max(need, w + 24 * S)
    for wdg in widgets:
        if wdg["type"] == "STRING" and wdg.get("multiline"):
            continue
        val = trunc(f_widget, wdg["_val"], 150 * S)
        need = max(need, tw(f_widget, wdg["name"]) + tw(f_widget, val) + 66 * S)
    W = int(min(MAX_W, need))

    body_h = 8 * S + rows * SLOT_H
    for wdg in widgets:
        multi = wdg["type"] == "STRING" and wdg.get("multiline")
        body_h += (MULTI_H if multi else WIDGET_H) + WIDGET_GAP
    body_h += 6 * S
    H = TITLE_H + body_h

    img = Image.new("RGBA", (W + 2 * MARGIN, H + 2 * MARGIN), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle(
        [MARGIN, MARGIN + 3 * S, MARGIN + W, MARGIN + H + 3 * S],
        radius=RADIUS, fill=(0, 0, 0, 115))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(4 * S)))

    d = ImageDraw.Draw(img)
    x0, y0 = MARGIN, MARGIN

    d.rounded_rectangle([x0, y0, x0 + W, y0 + H], radius=RADIUS, fill=BODY,
                        outline=OUTLINE, width=S)
    d.rounded_rectangle([x0, y0, x0 + W, y0 + TITLE_H + RADIUS], radius=RADIUS,
                        fill=TITLE_BG)
    d.rectangle([x0, y0 + TITLE_H, x0 + W, y0 + TITLE_H + RADIUS], fill=BODY)
    d.line([x0, y0 + TITLE_H, x0 + W, y0 + TITLE_H], fill=(43, 43, 43, 255),
           width=S)
    d.rounded_rectangle([x0, y0, x0 + W, y0 + H], radius=RADIUS,
                        outline=OUTLINE, width=S)
    d.text((x0 + 14 * S, y0 + TITLE_H // 2), trunc(f_title, title, W - 24 * S),
           font=f_title, fill=TITLE_TEXT, anchor="lm")

    slot_top = y0 + TITLE_H + 4 * S
    r = 4.4 * S
    for i, s_in in enumerate(in_sockets):
        cy = slot_top + i * SLOT_H + SLOT_H // 2
        cx = x0 + 9 * S
        col = hex_rgba(TYPE_COLORS.get(s_in["type"], DEFAULT_SLOT))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col,
                  outline=SLOT_STROKE, width=S)
        d.text((cx + 9 * S, cy), trunc(f_slot, s_in["name"], W * 0.55),
               font=f_slot, fill=SLOT_TEXT, anchor="lm")
    for i, s_out in enumerate(outputs):
        cy = slot_top + i * SLOT_H + SLOT_H // 2
        cx = x0 + W - 9 * S
        col = hex_rgba(TYPE_COLORS.get(s_out["type"], DEFAULT_SLOT))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col,
                  outline=SLOT_STROKE, width=S)
        d.text((cx - 9 * S, cy), trunc(f_slot, s_out["name"], W * 0.55),
               font=f_slot, fill=SLOT_TEXT, anchor="rm")

    wy = slot_top + rows * SLOT_H + 4 * S
    for wdg in widgets:
        multi = wdg["type"] == "STRING" and wdg.get("multiline")
        wx0, wx1 = x0 + INSET, x0 + W - INSET
        if multi:
            d.rounded_rectangle([wx0, wy, wx1, wy + MULTI_H], radius=5 * S,
                                fill=WIDGET_BG, outline=WIDGET_OUTLINE, width=1)
            text = wdg["_val"] or wdg.get("placeholder", "")
            fill = WIDGET_VAL if wdg["_val"] else PLACEHOLDER
            ty = wy + 6 * S
            for ln in wrap_px(f_multi, text, (wx1 - wx0) - 14 * S, 4):
                d.text((wx0 + 7 * S, ty), ln, font=f_multi, fill=fill)
                ty += int(13.5 * S)
            wy += MULTI_H + WIDGET_GAP
            continue

        d.rounded_rectangle([wx0, wy, wx1, wy + WIDGET_H],
                            radius=WIDGET_H // 2, fill=WIDGET_BG,
                            outline=WIDGET_OUTLINE, width=1)
        cyw = wy + WIDGET_H // 2
        has_arrows = wdg["kind"] == "combo" or wdg["type"] in ("INT", "FLOAT")
        pad = 20 * S if has_arrows else 10 * S
        if has_arrows:
            ah = 3.5 * S
            d.polygon([(wx0 + 8 * S, cyw), (wx0 + 8 * S + ah, cyw - ah),
                       (wx0 + 8 * S + ah, cyw + ah)], fill=ARROW)
            d.polygon([(wx1 - 8 * S, cyw), (wx1 - 8 * S - ah, cyw - ah),
                       (wx1 - 8 * S - ah, cyw + ah)], fill=ARROW)
        avail = (wx1 - wx0) - 2 * pad
        if wdg["type"] == "BOOLEAN":
            on = wdg["_val"] == "true"
            dot_r = 4 * S
            dcx = wx1 - 12 * S
            d.ellipse([dcx - dot_r, cyw - dot_r, dcx + dot_r, cyw + dot_r],
                      fill=(139, 195, 74, 255) if on else (85, 85, 85, 255))
            d.text((dcx - 8 * S, cyw), wdg["_val"], font=f_widget,
                   fill=WIDGET_VAL, anchor="rm")
            d.text((wx0 + 10 * S, cyw),
                   trunc(f_widget, wdg["name"],
                         avail - tw(f_widget, wdg["_val"]) - 20 * S),
                   font=f_widget, fill=WIDGET_NAME, anchor="lm")
        else:
            val_w = min(tw(f_widget, wdg["_val"]), avail * 0.62)
            val = trunc(f_widget, wdg["_val"], val_w)
            name_w = avail - tw(f_widget, val) - 8 * S
            d.text((wx0 + pad, cyw), trunc(f_widget, wdg["name"], name_w),
                   font=f_widget, fill=WIDGET_NAME, anchor="lm")
            d.text((wx1 - pad, cyw), val, font=f_widget, fill=WIDGET_VAL,
                   anchor="rm")
        wy += WIDGET_H + WIDGET_GAP

    return img


def main():
    class_map, display_map = load_pack()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    readme = (REPO / "README.md").read_text()
    report, errors = [], []
    for cls_name, cls in class_map.items():
        display = display_map.get(cls_name, cls_name)
        try:
            schema = node_schema(cls_name, cls, display)
            img = render(schema)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls_name}: {e}")
            continue
        referenced = f"assets/images/nodes/{cls_name}.png" in readme
        # only keep repo images for nodes the README references, plus
        # refresh anything already on disk
        path = OUT_DIR / f"{cls_name}.png"
        if referenced or path.exists():
            img.save(path, optimize=True)
        report.append((cls_name, display, img.width // S, referenced))
    print(f"{'class':34} {'css width':>9}  in README")
    for cls_name, display, w, referenced in sorted(report):
        print(f"{cls_name:34} {w:>9}  {'yes' if referenced else 'NO'}")
    for e in errors:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
