// Pin rendering. DOM only -- this file never touches the graph beyond reading positions.
//
// Pins live in one absolutely-positioned layer per node, appended to `.lg-node` (which is
// position:absolute, contain:layout and overflow:visible, so the layer anchors to the node
// and pins may legally overhang its right edge).
//
// Placement note for v1: pins sit in a rail OUTSIDE the node's right border. A node's own
// outputs occupy the right edge of the same rows its inputs occupy, so an inboard pin would
// collide with them. Relocating outputs to a bottom block is the fix; until then, outboard.

const LAYER = "cablemanagement-layer";
const PIN = "cablemanagement-pin";

/** The pin layer for a node element, created on first use. */
function layerFor(nodeEl) {
  let layer = nodeEl.querySelector(`:scope > .${LAYER}`);
  if (!layer) {
    layer = document.createElement("div");
    layer.className = LAYER;
    nodeEl.appendChild(layer);
  }
  return layer;
}

/**
 * Current canvas zoom. Rects are screen pixels; the layer lives inside the scaled
 * transform, so screen deltas must be divided by scale to become layout pixels.
 */
function scaleOf(app) {
  const s = app?.canvas?.ds?.scale;
  return typeof s === "number" && s > 0 ? s : 1;
}

/** Vertical centre of a slot dot, in the node's own unscaled coordinate space.
 *  Takes the node rect pre-measured -- one read per node, not one per dot. */
function rowCentre(nodeRect, dotEl, scale) {
  const d = dotEl.getBoundingClientRect();
  return (d.top + d.height / 2 - nodeRect.top) / scale;
}

/**
 * Reconcile the pins on one node against the list of pass-through inputs it should have.
 * Updates positions in place; only adds and removes what actually changed.
 *
 * Collapsed: every pin sits at one point on the right edge, just below core's stacked
 * output dot -- the same visual language core uses for the node's own slots. Slot rows do
 * not exist to measure, so the point is derived from the title bar (the whole element).
 */
export function syncNode(app, nodeEl, nodeId, passThroughs, collapsed = false, slotEls = null) {
  const scale = scaleOf(app);
  const layer = layerFor(nodeEl);
  const wanted = new Map();

  // Slot elements are descendants of the node element; a document-wide attribute
  // scan per slot was the sync pass's dominant cost on large graphs.
  const nodeRect = passThroughs.length ? nodeEl.getBoundingClientRect() : null;
  const collapsedTop = collapsed && nodeRect ? nodeRect.height / scale / 2 + 7 : 0;
  for (const pt of passThroughs) {
    if (collapsed) {
      wanted.set(String(pt.index), { pt, top: collapsedTop });
      continue;
    }
    const dot = slotEls
      ? slotEls.get(`${nodeId}-in-${pt.index}`)
      : nodeEl.querySelector(
          `[data-slot-key="${CSS.escape(`${nodeId}-in-${pt.index}`)}"]`
        );
    if (!dot) continue; // slot not rendered (virtualised row)
    wanted.set(String(pt.index), { pt, top: rowCentre(nodeRect, dot, scale) });
  }

  // Remove pins whose input is no longer connected or no longer rendered.
  for (const el of [...layer.children]) {
    if (!wanted.has(el.dataset.cablemanagementIndex)) el.remove();
  }

  for (const [index, { pt, top }] of wanted) {
    let pin = layer.querySelector(`:scope > [data-cablemanagement-index="${CSS.escape(index)}"]`);
    if (!pin) {
      pin = document.createElement("div");
      pin.className = PIN;
      pin.dataset.cablemanagementIndex = index;
      pin.dataset.cablemanagementNode = String(nodeId);
      layer.appendChild(pin);
    }
    // Update only what changed -- these run on every graph mutation.
    const topPx = `${Math.round(top)}px`;
    if (pin.style.top !== topPx) pin.style.top = topPx;
    const title =
      pt.kind === "link"
        ? `${pt.name}: pass through from ${pt.origin.node.title ?? pt.origin.node.type}`
        : `${pt.name}: share this value`;
    if (pin.title !== title) pin.title = title;
    if (pin.dataset.cablemanagementType !== String(pt.type)) {
      pin.dataset.cablemanagementType = String(pt.type);
      // Same resolution core's getSlotColor() uses, so a pin matches the slots either side
      // of it under any theme.
      const t = String(pt.type ?? "").toUpperCase();
      pin.style.setProperty("--cablemanagement-slot-color", t ? `var(--color-datatype-${t}, #AAA)` : "#AAA");
    }
    if (pin.dataset.cablemanagementKind !== pt.kind) pin.dataset.cablemanagementKind = pt.kind;
    if (collapsed) pin.dataset.cablemanagementCollapsed = "1";
    else delete pin.dataset.cablemanagementCollapsed;
  }
}

/** Drop every pin belonging to a node element (node removed, or extension disabled). */
export function clearNode(nodeEl) {
  nodeEl.querySelector(`:scope > .${LAYER}`)?.remove();
}

/** Remove every pin everywhere. */
export function clearAll() {
  for (const l of document.querySelectorAll(`.${LAYER}`)) l.remove();
}

/** Identify a pin element under the pointer. */
export function pinAt(target) {
  const el = target?.closest?.(`.${PIN}`);
  if (!el) return null;
  return { el, nodeId: el.dataset.cablemanagementNode, index: Number(el.dataset.cablemanagementIndex) };
}
