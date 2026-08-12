// Graph semantics. No DOM in this file.
//
// The whole trick behind Cable Management: a pass-through output pin owns no graph state. When you
// drag off it, we look up what actually feeds the corresponding input and connect from
// THERE. The graph gains one ordinary link, indistinguishable from one you drew by hand,
// and uninstalling the extension leaves the workflow byte-identical.

/**
 * The graph currently on screen.
 *
 * `app.graph` is NOT it. Entering a subgraph swaps what the canvas renders but leaves
 * `app.graph` pointing at the root, so every id in the DOM misses and the extension
 * concludes the nodes do not exist. `canvas.setGraph()` -- which `openSubgraph()` calls --
 * is what actually tracks the view.
 */
export function activeGraph(app) {
  return app?.canvas?.graph ?? app?.graph ?? null;
}

// litegraph/src/constants.ts -- the subgraph boundary nodes carry fixed negative ids and are
// NOT members of graph.nodes, so getNodeById() misses them.
const SUBGRAPH_INPUT_ID = -10;
const SUBGRAPH_OUTPUT_ID = -20;

/**
 * Resolve a node id against a graph, including the subgraph boundary.
 *
 * Inside a subgraph, anything fed from outside has `origin_id === -10` and lives at
 * `graph.inputNode` rather than in `graph.nodes`. Without this, every input wired to the
 * subgraph's own inputs looks unresolvable and silently loses its pass-through pin.
 */
export function nodeById(graph, id) {
  if (!graph) return null;
  const n = Number(id);
  if (n === SUBGRAPH_INPUT_ID) return graph.inputNode ?? null;
  if (n === SUBGRAPH_OUTPUT_ID) return graph.outputNode ?? null;
  return graph.getNodeById(n) ?? graph.getNodeById(id) ?? null;
}

/** The litegraph node behind a `.lg-node` element's data-node-id. */
export function nodeFromId(graph, id) {
  return nodeById(graph, id);
}

/** The link object feeding `node.inputs[index]`, or null when the input is unconnected. */
function linkInto(graph, node, index) {
  const slot = node?.inputs?.[index];
  if (!slot || slot.link == null) return null;
  return graph.getLink ? graph.getLink(slot.link) : graph.links?.[slot.link];
}

/**
 * Resolve an input back to whatever truly drives it.
 * Returns {node, slot, type} of the upstream OUTPUT, or null if nothing feeds it.
 */
export function originOf(graph, node, index) {
  const link = linkInto(graph, node, index);
  if (!link) return null;
  const origin = nodeById(graph, link.origin_id);
  if (!origin) return null;
  return { node: origin, slot: link.origin_slot, type: link.type };
}

/**
 * Every input eligible to sprout a pass-through output, in two flavours:
 *
 *   kind "link"   -- something upstream already drives it, so a drop resolves to that origin
 *                    and the graph gains one ordinary link. Zero extension state.
 *   kind "widget" -- it holds a literal. Nothing produces that value yet, so a drop has to
 *                    materialise a source first (see primitive.js).
 *
 * Widget-backed inputs are ordinary inputs that happen to render inside the widget grid, so
 * a connected widget input is kind "link" like any other.
 */
export function passThroughInputs(graph, node) {
  const out = [];
  const inputs = node?.inputs ?? [];
  for (let i = 0; i < inputs.length; i++) {
    const slot = inputs[i];
    const common = { index: i, name: slot.name, type: slot.type };
    if (slot.link != null) {
      const origin = originOf(graph, node, i);
      if (origin) out.push({ ...common, kind: "link", origin });
    } else if (slot.widget) {
      out.push({ ...common, kind: "widget", origin: null });
    }
  }
  return out;
}

