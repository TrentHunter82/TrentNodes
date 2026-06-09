#!/usr/bin/env python3
"""Build a ComfyUI (litegraph v0.4) workflow JSON that wires 13 Mikey
'Wildcard Processor' nodes into the Composition Cowboy node's string inputs.

Schema/serialization matched to the user's frontend (comfyui-frontend 1.45.15) by
inspecting their own saved workflows:
  - link tuple = [id, from_node, from_slot, to_node, to_slot, type]
  - a widget converted to an input appears in `inputs` with {"widget":{"name":...}}
    AND its value REMAINS in widgets_values (positional).
  - outputs use "links":[...]; plain socket inputs use "link": id|null.

CompositionCowboy widget order + defaults are taken verbatim from the live node.
"""
import json
import os

# 13 string widgets to expose as inputs, in CompositionCowboy's own order.
WIRED = ["title", "subtitle", "hero", "body", "brand", "extra",
         "high_level_description", "background", "style_detail",
         "aesthetics", "lighting", "medium", "style_palette"]

# CompositionCowboy widgets_values, exactly as derived from CompositionCowboy.INPUT_TYPES()
# (preset_index carries a control_after_generate companion at index 4; converted widgets
#  keep their default value here even though they are also exposed as inputs).
CC_WIDGETS_VALUES = [
    "Magazine / Classic Masthead",  # preset
    "name",                          # select_by
    "All",                           # pool
    0,                                # preset_index
    "fixed",                         # preset_index.control_after_generate
    True,                             # include_bboxes
    0,                                # width
    0,                                # height
    "replace",                       # json_mode
    25,                               # bg_brightness
    "", "", "", "", "", "",          # title..extra
    "", "",                           # high_level_description, background
    "none",                          # style
    "", "", "", "", "",              # style_detail..style_palette
    "", "", "",                       # overrides_json, import_json, elements_json
]

CC_OUTPUTS = [("prompt_json", "STRING"), ("prompt_text", "STRING"), ("preview", "IMAGE"),
              ("bboxes", "BOUNDING_BOX"), ("width", "INT"), ("height", "INT"),
              ("preset_name", "STRING"), ("vlm_brief", "STRING")]

CC_ID = 1


def main():
    nodes = []
    links = []

    # --- Composition Cowboy node (id 1) ---
    cc_inputs = []
    for slot, name in enumerate(WIRED):
        cc_inputs.append({
            "localized_name": name, "name": name, "type": "STRING",
            "widget": {"name": name}, "link": slot + 1,   # link ids 1..13
        })
    cc_inputs.append({"localized_name": "image", "name": "image", "type": "IMAGE", "link": None})
    cc_inputs.append({"localized_name": "bboxes", "name": "bboxes", "type": "BOUNDING_BOX", "link": None})

    # Unconnected outputs use null (matches the canonical reference convention).
    cc_outputs = [{"localized_name": n, "name": n, "type": t, "links": None, "slot_index": i}
                  for i, (n, t) in enumerate(CC_OUTPUTS)]

    # Single-column layout: all 13 Wildcard Processor nodes stacked on the left, Composition
    # Cowboy alone on the right. Nothing sits between a source node and the target, so every
    # wire is fully visible (no links routed behind other nodes).
    WC_X, WC_Y0, WC_DY = 40, 40, 184
    WC_W, WC_H = 360, 164
    CC_X = WC_X + WC_W + 220                    # clear gap to the right of the column

    nodes.append({
        "id": CC_ID, "type": "CompositionCowboy", "pos": [CC_X, 40], "size": [460, 1040],
        "flags": {}, "order": len(WIRED), "mode": 0,
        "inputs": cc_inputs, "outputs": cc_outputs,
        "properties": {"Node name for S&R": "CompositionCowboy"},
        "widgets_values": CC_WIDGETS_VALUES,
    })

    # --- 13 Wildcard Processor nodes (ids 2..14) ---
    for i, name in enumerate(WIRED):
        nid = 2 + i
        link_id = i + 1
        seed = 100000 + (i * 777)              # distinct starting seeds -> decorrelated lines
        nodes.append({
            "id": nid, "type": "Wildcard Processor",
            "pos": [WC_X, WC_Y0 + i * WC_DY], "size": [WC_W, WC_H],
            "flags": {}, "order": i, "mode": 0,
            "inputs": [],
            "outputs": [{"localized_name": "STRING", "name": "STRING", "type": "STRING",
                         "links": [link_id], "slot_index": 0}],
            "properties": {"Node name for S&R": "Wildcard Processor"},
            "widgets_values": ["__composition_cowboy/%s__" % name, seed, "randomize"],
            "title": "WC → %s" % name,
        })
        # link: from wildcard output (slot 0) -> CC input slot i
        links.append([link_id, nid, 0, CC_ID, i, "STRING"])

    wf = {
        "id": "composition-cowboy-wildcards",
        "revision": 0,
        "last_node_id": 14,
        "last_link_id": 13,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "definitions": {},
        "config": {},
        "extra": {},
        "version": 0.4,
    }

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "user", "default", "workflows", "composition_cowboy_wildcards.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    print("wrote:", out)
    print("nodes:", len(nodes), "links:", len(links))


if __name__ == "__main__":
    main()
