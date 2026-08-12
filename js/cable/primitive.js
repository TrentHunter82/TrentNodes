// Materialising the pass-through source for a widget that holds a literal value.
//
// A connected input can be resolved -- something upstream already produces the value. A
// widget holding a literal has no such source, so one has to exist before it can be shared.
//
// Core already ships exactly the right thing: `PrimitiveNode` from
// src/extensions/core/widgetInputs.ts. It is `isVirtualNode`, so it never reaches the
// backend; on first connection it adopts the target widget's type AND config (which is why
// COMBO widgets work, where PrimitiveInt/Float/String cannot); and at prompt-build time it
// copies its value into every connected consumer's widget as a literal.
//
// We create one, mark it ours, and park it behind the host node. Ownership lives in
// `properties`, which is litegraph's sanctioned serialised bag -- PrimitiveNode already
// adds a property of its own in its constructor. Disable this extension and an owned
// primitive is simply a normal Primitive node with one unread property.

const OWNED = "cablemanagement.owned";
// Primitives taking part in an in-flight gesture. Without this, reapIfUnused deletes a
// primitive the instant it is created: on pointerdown its only link is the one driving its
// own host, which is precisely the "unused" condition -- so it vanished before the drag
// could ever use it, DOM element and all.
const inFlight = new Set();
const HOST = "cablemanagement.host";
const HOST_SLOT = "cablemanagement.hostSlot";

/** The host node an owned primitive republishes, or null if it no longer exists. */
function hostOf(graph, prim) {
  const id = prim?.properties?.[HOST];
  if (id == null || !graph) return null;
  return graph.getNodeById?.(Number(id)) ?? graph.getNodeById?.(id) ?? null;
}

/** Did we create this node, or did the user (input double-click, node search)? */
export function isOwned(node) {
  // Type-guarded like isStitch: the brand makes a node INVISIBLE and click-through,
  // so honouring it on arbitrary node types would let a crafted workflow conceal
  // executing nodes. Only core's virtual PrimitiveNode is ours to hide (and reap).
  return (
    node?.properties?.[OWNED] === true &&
    node.isVirtualNode === true &&
    node.type === "PrimitiveNode"
  );
}

/** The host's current literal for a widget-backed input. */
export function widgetValueFor(node, slot) {
  const name = slot?.widget?.name;
  if (!name) return undefined;
  return node.widgets?.find((w) => w.name === name)?.value;
}

/**
 * Create a primitive that can feed CONSUMERS of a host widget's value.
 *
 * Deliberately NOT connected to the host. Wiring it into the host's own widget input is what
 * disables that widget, which then forces hiding it and faking an editable replacement --
 * and that replacement can never sit exactly where the original did, so the node's layout
 * breaks. The host already holds the literal; it just needs mirroring onto the primitive.
 *
 * Consequence: the host's widget stays completely stock -- enabled, in place, untouched.
 * A consumer's widget is driven by the primitive and goes disabled, which is precisely how
 * core's own Primitive nodes have always behaved.
 */
export function materialise(app, host, index) {
  return materialiseIn(app?.canvas?.graph ?? app?.graph, host, index);
}

/** Same, into an explicit graph -- coverage reverts heal closed subgraphs too. */
export function materialiseIn(graph, host, index) {
  const LiteGraph = window.LiteGraph;
  const slot = host?.inputs?.[index];
  if (!LiteGraph || !graph || !slot) return null;

  const seed = widgetValueFor(host, slot);

  const prim = LiteGraph.createNode("PrimitiveNode");
  if (!prim) return null;

  prim.pos = [host.pos[0] + 10, host.pos[1] + 10];
  graph.add(prim);
  prim.properties[OWNED] = true;
  prim.properties[HOST] = host.id;
  prim.properties[HOST_SLOT] = index;

  // A PrimitiveNode adopts its type and config from the FIRST widget input it connects to,
  // and it has none yet. Seed it from the host's widget spec directly so the drag has a
  // typed output and the right control.
  const w = host.widgets?.find((x) => x.name === slot.widget?.name);
  if (w) {
    prim.outputs[0].type = slot.type;
    prim.outputs[0].name = String(slot.type);
    prim.outputs[0].widget = { name: w.name };
    carryRefSymbols(slot.widget, prim.outputs[0].widget);
    if (typeof prim.addWidget === "function" && !prim.widgets?.length) {
      const added = prim.addWidget(w.type, w.name, seed, () => {}, { ...(w.options ?? {}) });
      if (added) added.value = seed;
    }
  }
  prim.title = slot.widget?.name ?? slot.name;
  // The first drop can land on a plain input before any sync pass runs.
  ensureConnectGate(prim);
  return prim;
}

/**
 * Core's refreshComboInNode dereferences `outputs[0].widget[GET_CONFIG]()` unguarded on
 * combo primitives. GET_CONFIG is a Symbol -- unreachable by import from an extension
 * and never serialized, so a bare `{ name }` ref (fresh OR reloaded) turns Refresh Node
 * Definitions into a TypeError that aborts the refresh for the WHOLE graph. The host's
 * own input ref carries the live getter; copy every symbol it has onto the primitive's.
 */
function carryRefSymbols(from, to) {
  if (!from || !to) return;
  for (const s of Object.getOwnPropertySymbols(from)) {
    if (!(s in to)) to[s] = from[s];
  }
}

/** Push the host widget's current value onto the primitive that republishes it. */
export function mirrorFromHost(graph, prim) {
  const host = graph?.getNodeById?.(prim?.properties?.[HOST]);
  const index = prim?.properties?.[HOST_SLOT];
  const slot = host?.inputs?.[index];
  // Every sync, not just materialise: a reloaded primitive's ref lost its symbols to
  // serialization, and the gate-first case never passes core's own re-seeding.
  carryRefSymbols(slot?.widget, prim?.outputs?.[0]?.widget);
  const w = host?.widgets?.find((x) => x.name === slot?.widget?.name);
  let pw = prim?.widgets?.[0];
  // A revived primitive whose only link targets the OUTPUT GATE has no widget at
  // all: core builds a Primitive's widget from its connected target's type, and a
  // gate resolves to no node (conversion adoption and subgraph reloads both land
  // here). Rebuild it from the host's spec, exactly as materialise seeded it.
  if (w && !pw && typeof prim?.addWidget === "function") {
    pw = prim.addWidget(w.type, w.name, w.value, () => {}, { ...(w.options ?? {}) });
  }
  if (!w || !pw || pw.value === w.value) return;
  pw.value = w.value;
  pw.callback?.(w.value);
}

/**
 * Serialize idempotency (sweep cell E): configure() rebuilds a Primitive's widgets
 * through _onFirstConnection -- value PLUS control_after_generate (and the combo
 * filter) -- while materialise seeds only the value widget. The two states
 * serialize differently, so after every undo/redo restore core's
 * captureCanvasState saw a "user edit" and WIPED THE REDO QUEUE. Once the first
 * link resolves to a real consumer, run core's own recreateWidget() one time so
 * the live widget set is exactly what a reload would build. Gate-first primitives
 * are left alone: _onFirstConnection cannot resolve the gate (id -20) and the
 * recreate would strip the widget mirrorFromHost rebuilt.
 */
export function ensureStableSerial(graph, prim) {
  if (!prim || prim.__cablemanagementRebuilt) return;
  if ((prim.widgets?.length ?? 0) !== 1) { prim.__cablemanagementRebuilt = true; return; }
  const t = prim.widgets[0].type;
  if (t !== "number" && t !== "combo") { prim.__cablemanagementRebuilt = true; return; }
  if (typeof prim.recreateWidget !== "function") { prim.__cablemanagementRebuilt = true; return; }
  const ll = graph?._links?.get?.(prim.outputs?.[0]?.links?.[0]);
  const target = ll && graph.getNodeById?.(ll.target_id);
  if (!target?.inputs?.[ll.target_slot]) return; // no real consumer yet; retry next sync
  prim.__cablemanagementRebuilt = true;
  prim.recreateWidget();
}

/**
 * Mirror AT PROMPT-BUILD TIME, not only on sync.
 *
 * The sync loop runs off a MutationObserver, and typing in a widget mutates no DOM the
 * observer watches -- so an edit made just before Queue Prompt would ship the primitive's
 * stale value. PrimitiveNode.applyToGraph is what graphToPrompt calls on virtual nodes to
 * push their value into consumers; mirroring right before it closes the gap for good.
 *
 * Installed from the sync walk (not materialise) so primitives loaded from a saved
 * workflow get it too. Instance patch only: nothing serialises, core class untouched.
 */
export function ensureApplyHook(graph, prim) {
  if (!prim || prim.__cablemanagementApply) return;
  const orig = prim.applyToGraph;
  if (typeof orig !== "function") return;
  prim.__cablemanagementApply = true;
  prim.applyToGraph = function (...args) {
    mirrorFromHost(this.graph ?? graph, this);
    return orig.apply(this, args);
  };
}

/**
 * Let an owned primitive feed PLAIN inputs, not only widget inputs.
 *
 * Core's PrimitiveNode.onConnectOutput refuses a non-widget input twice over: outright when
 * the input's TYPE is not a ComfyWidgets key (kills wildcard and custom-typed inputs), and
 * via _isValidConnection -- which requires input.widget[GET_CONFIG] -- the moment the
 * primitive already has any link, which an owned primitive always does. Type compatibility
 * between our seeded output and the input is litegraph's general check and still applies;
 * for widget targets the original runs so combo config merging stays intact.
 *
 * The connection alone is not enough: core delivers values by writing into consumer
 * WIDGETS, so a plain input would resolve to nothing at prompt build. injectLiterals()
 * completes the pair at graphToPrompt time.
 */
export function ensureConnectGate(prim) {
  if (!prim || prim.__cablemanagementGate) return;
  const orig = prim.onConnectOutput;
  if (typeof orig !== "function") return;
  prim.__cablemanagementGate = true;
  prim.onConnectOutput = function (slot, type, input, target_node, target_slot) {
    if (input && !input.widget) return true;
    return orig.call(this, slot, type, input, target_node, target_slot);
  };
}

/**
 * Write virtual primitives' literals into consumer inputs that core cannot serve.
 *
 * Core's PrimitiveNode delivers its value by writing into the consumer's WIDGET at
 * applyToGraph time -- so any input without a live widget instance loses the value
 * silently: a plain input has no widget ref at all, and a forceInput pin carries a
 * `widget` ref but no widget instance behind it (found via a custom node's forceInput override pin
 * that "did nothing": the link to the virtual node resolves to nothing and the field
 * is simply missing from the payload). Both cases get the literal written here.
 * Covers hand-placed core Primitives too, not only cablemanagement-owned ones -- the delivery
 * gap is identical. Subgraph execution ids are composite ("path:nodeId"), so target
 * ids match either exactly or as the last path segment.
 */
export function injectLiterals(output, graph) {
  if (!output || !graph) return;
  // (graph, parent) pairs -- parent names the subgraph NODE a graph hangs from, so a
  // link into the output gate can continue from that node's output in the outer graph.
  const entries = [{ g: graph, parent: null }];
  for (const n of graph.nodes ?? []) {
    if (n.subgraph) entries.push({ g: n.subgraph, parent: { g: graph, node: n } });
  }
  for (const { g, parent } of entries) {
    for (const prim of g.nodes ?? []) {
      if (!isOwned(prim) && !(prim.isVirtualNode && prim.type === "PrimitiveNode")) continue;
      let value = prim.widgets?.[0]?.value;
      // Widgetless revived primitive (see mirrorFromHost): the host holds the truth,
      // and it may never have been on screen to trigger the rebuild.
      if (value === undefined) {
        const host = g.getNodeById?.(prim.properties?.[HOST]);
        const slot = host?.inputs?.[prim.properties?.[HOST_SLOT]];
        if (host && slot) value = widgetValueFor(host, slot);
      }
      if (value === undefined) continue;
      for (const linkId of prim.outputs?.[0]?.links ?? []) {
        const link = g.getLink ? g.getLink(linkId) : g.links?.[linkId];
        for (const c of consumersOf(g, parent, link, 0)) {
          // A live widget already received the value via applyToGraph; a widget REF
          // without a widget instance (forceInput) did not. Neither did ANY consumer
          // reached through a boundary hop -- core's delivery follows the raw link
          // and dies at the gate, so the live widget still holds its stale default.
          const live =
            c.input.widget && c.node.widgets?.find((w) => w.name === c.input.widget.name);
          if (c.hops && live && live.value !== value) {
            // Do core's widget write for the hop the gate blocked: the live widget is
            // what SERIALISES, and a receiver without the extension executes that --
            // not this payload injection. Same write + callback shape as core's
            // applyFirstWidgetValueToGraph.
            live.value = value;
            live.callback?.(value, window.app?.canvas, c.node,
              window.app?.canvas?.graph_mouse ?? [0, 0], {});
          }
          if (!c.hops && live) continue;
          const tid = String(c.node.id);
          for (const key of Object.keys(output)) {
            if (key !== tid && !key.endsWith(`:${tid}`)) continue;
            if (output[key]?.inputs) output[key].inputs[c.input.name] = value;
          }
        }
      }
    }
  }
}

/**
 * The REAL consumers of a link, seen through subgraph plumbing: a link into the
 * output gate continues from the subgraph node's output in the parent graph, and a
 * link into a subgraph node's input continues from the matching gate slot inside.
 * Both hops matter to delivery -- the prompt is flattened, so only real nodes hold
 * payload entries, while the graph-level links stop at the boundary objects.
 */
function consumersOf(g, parent, link, depth) {
  if (!link || depth > 8) return [];
  if ((link.targetIsIoNode ?? String(link.target_id) === "-20") && parent) {
    const res = [];
    for (const lid of parent.node.outputs?.[link.target_slot]?.links ?? []) {
      const l = parent.g.getLink ? parent.g.getLink(lid) : parent.g.links?.[lid];
      res.push(...consumersOf(parent.g, null, l, depth + 1));
    }
    return res;
  }
  const target = g.getNodeById(link.target_id) ?? null;
  const input = target?.inputs?.[link.target_slot];
  if (!input) return [];
  if (target.subgraph) {
    const res = [];
    for (const lid of target.subgraph.inputs?.[link.target_slot]?.linkIds ?? []) {
      const l = target.subgraph.getLink?.(lid);
      res.push(...consumersOf(target.subgraph, { g, node: target }, l, depth + 1));
    }
    return res;
  }
  // A virtual Reroute (cablemanagement's gate stitches included) republishes its input --
  // the real consumers are past it, it holds no prompt entry to inject into, and
  // core's own applyToGraph stops at it just the same.
  if (target.isVirtualNode && target.type === "Reroute") {
    const res = [];
    for (const lid of target.outputs?.[0]?.links ?? []) {
      const l = g.getLink ? g.getLink(lid) : g.links?.[lid];
      res.push(...consumersOf(g, parent, l, depth + 1));
    }
    return res;
  }
  return [{ node: target, input, hops: depth }];
}

/** The owned primitive republishing this host widget, if one exists. */
export function ownerOf(graph, host, index) {
  if (!graph || !host) return null;
  for (const n of graph.nodes ?? []) {
    if (!isOwned(n)) continue;
    if (String(n.properties?.[HOST]) === String(host.id) && n.properties?.[HOST_SLOT] === index) {
      return n;
    }
  }
  return null;
}

/** Owned primitives should be invisible but still laid out -- see hide()/CSS. */
export function ownedIds(graph) {
  const ids = new Set();
  for (const n of graph?.nodes ?? []) if (isOwned(n)) ids.add(String(n.id));
  return ids;
}

/** Drop an owned primitive once its last consumer is gone. */
export function hold(prim) {
  if (prim) inFlight.add(prim.id);
}

export function release(prim) {
  if (prim) inFlight.delete(prim.id);
}

export function reapIfUnused(graph, prim) {
  if (!isOwned(prim)) return false;
  if (inFlight.has(prim.id)) return false;

  // Host gone: nothing left to republish, and every wire it still feeds would draw from a
  // node that is deliberately invisible. The apparent origin died; its links die with it,
  // exactly as they would had the origin been an ordinary visible node.
  if (!hostOf(graph, prim)) {
    graph.remove(prim);
    return true;
  }

  // It is never connected to its host, so "unused" is simply: no consumers left.
  if ((prim.outputs?.[0]?.links?.length ?? 0) > 0) return false;
  // Floating links are consumers too -- they live in graph.floatingLinks, NOT in
  // the output's link list (measured: parking a widget passthrough on a comb gate
  // left outLinks 0, the next press reaped the primitive, core cascaded the
  // floating link away and the whole lane imploded).
  for (const f of graph.floatingLinks?.values?.() ?? []) {
    if (String(f.origin_id) === String(prim.id)) return false;
  }
  graph.remove(prim);
  return true;
}

/**
 * Keep the primitive parked inside its host's footprint after the host moves.
 *
 * The park position is set once at materialise time and nothing ever updated it, so a
 * dragged-away host left its machinery behind -- and any honest draw (ledger unavailable,
 * stale frame) started a wire at a patch of empty canvas. setPos, not a pos write: the
 * layout store must hear about it or links keep drawing from the old spot anyway.
 */
export function parkWithHost(graph, prim) {
  const host = hostOf(graph, prim);
  if (!host) return;
  const x = host.pos[0] + 10;
  const y = host.pos[1] + 10;
  if (prim.pos[0] === x && prim.pos[1] === y) return;
  if (typeof prim.setPos === "function") prim.setPos(x, y);
  else {
    prim.pos[0] = x;
    prim.pos[1] = y;
  }
}
