"""
JSON ➜ Multi-Line Summary · ComfyUI custom node
- Accepts dict or raw-JSON string
- Supports sub_path with bracket indexing, e.g. "loras[0]" or "foo.bar[2][1]"
- If the resolved target is a list, it falls back to the first item (if any)
"""

import json
import re


class JSONSummary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Accept both socket labels. Keeps the plug simple.
                "json_in": (["JSON", "STRING"], {"forceInput": True}),
            },
            "optional": {
                # Examples:
                #   "inference_params"
                #   "loras[0]"
                #   "foo.bar[2][1]"
                "sub_path": ("STRING", {"default": "inference_params"}),
                "keys": ("STRING", {
                    "default": (
                        "height,width,num_frames,fps,guidance_scale,seed,steps,"
                        "use_timestep_transform,shift_value,use_guidance_schedule,"
                        "add_quality_guidance,clip_value,use_negative_prompts,"
                        "skip_control,caching_coefficient,caching_warmup,"
                        "caching_cooldown"
                    )
                }),
                "separator": ("STRING", {"default": ": "}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Trent/Utilities"

    # ------------------------------- Helpers ------------------------------- #
    def _parse_json(self, json_in):
        """Coerce the input to a Python dict if possible."""
        if isinstance(json_in, str):
            try:
                return json.loads(json_in)
            except json.JSONDecodeError:
                return "[JSONSummary] ERROR: input isn’t valid JSON"
        if isinstance(json_in, dict):
            return json_in
        return "[JSONSummary] ERROR: unsupported input type"

    def _descend(self, node, token):
        """
        Walk one token that may include list indexes.
        Examples of token:
          - "inference_params"
          - "loras[0]"
          - "bar[2][1]"
        """
        # Split "key[3][1]" into key="key", idx=[3,1]
        m = re.match(r'^([^\[\]]+)', token)
        key = m.group(0) if m else token
        idxs = [int(x) for x in re.findall(r'\[(\d+)\]', token)]

        # First descend by key (dict)
        if isinstance(node, dict):
            node = node.get(key, {})
        else:
            return {}

        # Then apply any list indices
        for i in idxs:
            if isinstance(node, list) and 0 <= i < len(node):
                node = node[i]
            else:
                return {}
        return node

    # --------------------------------- Run --------------------------------- #
    def run(self, json_in, sub_path, keys, separator):
        # 1) Load input
        node = self._parse_json(json_in)
        if isinstance(node, str):  # error message from _parse_json
            return (node,)

        # 2) Walk the sub-path like "foo.bar[2][1]"
        #    Empty sub_path means "use the whole object".
        if sub_path.strip():
            for part in filter(None, sub_path.split(".")):
                node = self._descend(node, part)

        # 3) If we ended up on a list, use the first element (common case: loras)
        if isinstance(node, list):
            node = node[0] if node and isinstance(node[0], dict) else {}

        if not isinstance(node, dict):
            return ("[JSONSummary] ERROR: target is not a dict",)

        # 4) Build the lines for requested keys
        wanted = [k.strip() for k in keys.split(",") if k.strip()]
        lines = [f"{k}{separator}{node.get(k, 'N/A')}" for k in wanted]

        return ("\n".join(lines),)


# ---- ComfyUI discovery ----
NODE_CLASS_MAPPINGS = {"JSONSummary": JSONSummary}
NODE_DISPLAY_NAME_MAPPINGS = {"JSONSummary": "JSON ➜ Multi-Line Summary"}
