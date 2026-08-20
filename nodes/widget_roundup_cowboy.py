"""
Cowboy Widget Roundup -- read titles and widget values off other nodes.

Two nodes:

* CowboyWidgetRoundup - connect up to MAX_NODES nodes, name up to
  MAX_WIDGETS_PER_NODE widgets on each, and get one multiline string
  with every node's title and widget values. The node_count and
  widgets_per_node widgets drive the dynamic UI in
  ../js/widget_roundup_cowboy.js (extra inputs and name fields appear
  and disappear as the counts change).
* CowboyNodeTitle - connect one node, get its title as a string.

Unlike KJNodes' WidgetToString there is no id / title search: both
nodes resolve the actual link, walk the workflow stored in
EXTRA_PNGINFO to find the upstream node, and read widget values from
the executing PROMPT. Reroute nodes are skipped through automatically.
"""
from ..utils.any_type import any_typ

MAX_NODES = 6
MAX_WIDGETS_PER_NODE = 4

# ComfyUI's global display-name registry, used as a nicer fallback when
# a node has no custom title (e.g. "KSampler" -> "KSampler", but
# "CheckpointLoaderSimple" -> "Load Checkpoint").
try:
    from nodes import NODE_DISPLAY_NAME_MAPPINGS as _COMFY_DISPLAY_NAMES
except Exception:  # noqa: BLE001 - not running inside a live ComfyUI
    _COMFY_DISPLAY_NAMES = {}

_REROUTE_TYPES = {"Reroute", "Reroute (rgthree)"}


def _split_unique_id(unique_id):
    """Return (node_id_int, subgraph_prefix_or_None) from a UNIQUE_ID."""
    text = str(unique_id)
    if ":" in text:
        parts = text.split(":")
        return int(parts[-1]), ":".join(parts[:-1])
    return int(text), None


def _collect_graph(workflow):
    """
    Flatten the workflow (main graph + subgraph definitions) into:

    * all_nodes:       every node dict
    * link_to_origin:  link id -> origin node id
    * node_to_parent:  subgraph-node id -> id of the subgraph node that
                       hosts it in the main graph (for PROMPT keys)
    """
    all_nodes = list(workflow.get("nodes", []))
    node_to_parent = {}

    # A subgraph instance in the main graph has its subgraph UUID as its
    # node type; map UUID -> hosting node id.
    subgraph_id_to_parent = {}
    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")
        if isinstance(node_type, str) and len(node_type) == 36 \
                and node_type.count("-") == 4:
            subgraph_id_to_parent[node_type] = node["id"]

    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        parent_id = subgraph_id_to_parent.get(subgraph.get("id", ""))
        for node in subgraph.get("nodes", []):
            if parent_id is not None:
                node_to_parent[node["id"]] = parent_id
            all_nodes.append(node)

    link_to_origin = {}
    for node in all_nodes:
        for output in node.get("outputs") or []:
            for link in output.get("links") or []:
                link_to_origin[link] = node["id"]

    return all_nodes, link_to_origin, node_to_parent


def _find_node(all_nodes, node_id, node_type=None):
    """Find a node dict by id (and optionally by type, to dodge subgraph
    id collisions)."""
    for node in all_nodes:
        if node.get("id") != node_id:
            continue
        if node_type is not None and node.get("type") != node_type:
            continue
        return node
    return None


def _resolve_origin(me, input_name, all_nodes, link_to_origin):
    """
    Follow the link on one of our inputs back to its origin node dict.
    Skips through Reroute nodes. Returns None when unconnected.
    """
    link_id = None
    for node_input in me.get("inputs") or []:
        if node_input.get("name") == input_name:
            link_id = node_input.get("link")
            break
    if link_id is None:
        return None

    origin_id = link_to_origin.get(link_id)
    for _ in range(32):  # reroute-chain guard
        origin = _find_node(all_nodes, origin_id)
        if origin is None or origin.get("type") not in _REROUTE_TYPES:
            return origin
        reroute_inputs = origin.get("inputs") or []
        link_id = reroute_inputs[0].get("link") if reroute_inputs else None
        if link_id is None:
            return None
        origin_id = link_to_origin.get(link_id)
    return None


def _node_title(node):
    """A node's user title, or its display name, or its raw type."""
    title = node.get("title")
    if title:
        return title
    node_type = node.get("type", "?")
    return _COMFY_DISPLAY_NAMES.get(node_type, node_type)


def _prompt_entry(prompt, node_id, node_to_parent, my_prefix):
    """Look up a node's PROMPT entry, trying subgraph-prefixed keys."""
    keys = []
    parent = node_to_parent.get(node_id)
    if parent is not None:
        keys.append(f"{parent}:{node_id}")
    if my_prefix is not None:
        keys.append(f"{my_prefix}:{node_id}")
    keys.append(str(node_id))
    for key in keys:
        if key in prompt:
            return prompt[key]
    return None


def _widget_value(entry, node, widget_name):
    """Read one widget value from a PROMPT entry, with clear errors."""
    inputs = entry.get("inputs", {})
    if widget_name not in inputs:
        available = [
            name for name, value in inputs.items()
            if not isinstance(value, list)
        ]
        raise ValueError(
            f"Node '{_node_title(node)}' has no widget '{widget_name}'. "
            f"Available widgets: {', '.join(available) or '(none)'}"
        )
    value = inputs[widget_name]
    if isinstance(value, list):
        raise ValueError(
            f"'{widget_name}' on node '{_node_title(node)}' is a "
            f"connected input, not a widget value."
        )
    return value


def _format_value(value, float_decimals):
    if isinstance(value, float):
        return f"{value:.{float_decimals}f}"
    return str(value)


class CowboyWidgetRoundup:
    """Round up titles and widget values from the connected nodes."""

    CATEGORY = "Trent/Text"
    DISPLAY_NAME = "Cowboy Widget Roundup"
    FUNCTION = "round_up"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = (
        "One multiline string: each node's title and widget values.",
    )

    DESCRIPTION = (
        "Connect nodes to the node_N inputs and type widget names into\n"
        "the fields. Outputs each node's title and widget values as one\n"
        "multiline string. node_count and widgets_per_node grow and\n"
        "shrink the inputs and fields. Leave a name field empty to skip\n"
        "it. Works through real links only - no id lookup."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Upstream widget edits don't change our direct inputs, so
        # always re-run.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        name_fields = {
            f"node_{i}_widget_{j}": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": f"node {i} widget name",
                "tooltip": f"Name of a widget on the node_{i} input. "
                           f"Leave empty to skip.",
            })
            for i in range(1, MAX_NODES + 1)
            for j in range(1, MAX_WIDGETS_PER_NODE + 1)
        }
        return {
            "required": {
                "node_count": ("INT", {
                    "default": 1, "min": 1, "max": MAX_NODES, "step": 1,
                    "tooltip": "How many node inputs to show.",
                }),
                "widgets_per_node": ("INT", {
                    "default": 1, "min": 1, "max": MAX_WIDGETS_PER_NODE,
                    "step": 1,
                    "tooltip": "How many widget-name fields per node.",
                }),
            },
            "optional": {
                **name_fields,
                "include_titles": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Put each node's title above its values.",
                }),
                "include_widget_names": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prefix each value with 'widget_name: '.",
                }),
                "float_decimals": ("INT", {
                    "default": 2, "min": 0, "max": 10,
                    "tooltip": "Decimal places for float values.",
                }),
                # First dynamic link input - the JS extension adds more.
                "node_1": (any_typ,),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # node_2..node_N are added by the JS extension and are not in
        # INPUT_TYPES; accept them rather than fail the stock check.
        return True

    def round_up(self, node_count, widgets_per_node, prompt=None,
                 extra_pnginfo=None, unique_id=None, include_titles=True,
                 include_widget_names=True, float_decimals=2, **kwargs):
        workflow = (extra_pnginfo or {}).get("workflow", {})
        all_nodes, link_to_origin, node_to_parent = _collect_graph(workflow)

        my_id, my_prefix = _split_unique_id(unique_id)
        me = _find_node(all_nodes, my_id, node_type="CowboyWidgetRoundup")
        if me is None:
            raise ValueError("CowboyWidgetRoundup could not find itself "
                             "in the workflow.")

        sections = []
        for i in range(1, node_count + 1):
            origin = _resolve_origin(me, f"node_{i}", all_nodes,
                                     link_to_origin)
            if origin is None:
                continue

            names = []
            for j in range(1, widgets_per_node + 1):
                name = (kwargs.get(f"node_{i}_widget_{j}") or "").strip()
                if name:
                    names.append(name)

            lines = []
            if include_titles:
                lines.append(_node_title(origin))

            if names:
                entry = _prompt_entry(prompt or {}, origin["id"],
                                      node_to_parent, my_prefix)
                if entry is None:
                    raise ValueError(
                        f"Node '{_node_title(origin)}' is not in the "
                        f"prompt (muted or bypassed?), so its widgets "
                        f"can't be read."
                    )
                for name in names:
                    value = _format_value(
                        _widget_value(entry, origin, name), float_decimals)
                    if include_widget_names:
                        lines.append(f"{name}: {value}")
                    else:
                        lines.append(value)

            if lines:
                sections.append("\n".join(lines))

        return ("\n\n".join(sections),)


class CowboyNodeTitle:
    """Output the title of the connected node as a string."""

    CATEGORY = "Trent/Text"
    DISPLAY_NAME = "Cowboy Node Title"
    FUNCTION = "get_title"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("title",)
    OUTPUT_TOOLTIPS = ("The connected node's title.",)

    DESCRIPTION = (
        "Connect any node output and get that node's title as a\n"
        "string. Uses the custom title if one is set, otherwise the\n"
        "node's display name."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A title edit doesn't change our direct inputs; always re-run.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": (any_typ, {
                    "tooltip": "Connect any output of the node whose "
                               "title you want.",
                }),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    def get_title(self, any_input=None, extra_pnginfo=None, unique_id=None):
        workflow = (extra_pnginfo or {}).get("workflow", {})
        all_nodes, link_to_origin, _ = _collect_graph(workflow)

        my_id, _ = _split_unique_id(unique_id)
        me = _find_node(all_nodes, my_id, node_type="CowboyNodeTitle")
        if me is None:
            raise ValueError("CowboyNodeTitle could not find itself in "
                             "the workflow.")

        origin = _resolve_origin(me, "any_input", all_nodes, link_to_origin)
        if origin is None:
            raise ValueError("CowboyNodeTitle: any_input is not "
                             "connected to a node.")

        return (_node_title(origin),)


NODE_CLASS_MAPPINGS = {
    "CowboyWidgetRoundup": CowboyWidgetRoundup,
    "CowboyNodeTitle": CowboyNodeTitle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CowboyWidgetRoundup": "Cowboy Widget Roundup",
    "CowboyNodeTitle": "Cowboy Node Title",
}
