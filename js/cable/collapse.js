// Collapsible input and output blocks.
//
// Collapsed means "show only what is information" -- a linked slot, or a REQUIRED input
// (core marks optional inputs with the hollow-circle shape, 7). An unconnected optional
// slot is a possibility, not information, and on busy nodes the possibilities outnumber
// the facts. A mandatory socket the workflow cannot run without stays visible.
//
// Hiding is `display:none`, which normally is dangerous: a slot with no measurable rect gets
// dropped from the layout store and its links snap back to litegraph's default anchors. It is
// safe HERE, and only here, because the only rows hidden are unconnected ones -- and an
// unconnected slot anchors no link. Connected slots always stay mounted.
//
// State lives in `node.properties` -- litegraph's serialised bag, same as cablemanagement.owned and
// cablemanagement.from -- so it survives refresh (Comfy's snapshot serialises properties), save/load,
// and copy/paste. Written only when non-default: an untouched node serialises clean.
//
// Height follows core's own floor semantics: `--node-height` on the node element is a
// MIN-height, content grows above it, and only mount/manual-resize/setSize move it (the
// advanced-inputs toggle works because the floor sits at the small state's height). So a
// drawer collapse LOWERS the floor to the folded natural height -- the node shrinks with
// its rows and grows back on expand -- unless the floor already sat ABOVE the open natural
// height, which is a deliberate manual size and is preserved.

import { app } from "../../../scripts/app.js";
import { activeGraph, nodeFromId } from "./graph.js";

const DRAWERS = "cablemanagement.drawers";
const OPTIONAL_SHAPE = 7; // RenderShape.HollowCircle -- core's optional-input marker

function flags(node) {
  const p = node?.properties?.[DRAWERS];
  return { inputs: !!p?.inputs, outputs: !!p?.outputs };
}

function titleHeight() {
  return window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
}

/**
 * The LGraphNode behind a node element, resolved AT CLICK TIME. Undo/redo replaces
 * node instances via graph.configure() while Vue reuses the DOM elements -- a handler
 * closed over the mount-time instance toggles a dead object's properties, a silent
 * no-op until something rebuilds the buttons.
 */
function liveNode(nodeEl, fallback) {
  const id = nodeEl?.dataset?.nodeId;
  return (id != null && nodeFromId(activeGraph(app), id)) || fallback;
}

/**
 * Register the toggle with the change tracker, or refreshes lose it.
 *
 * The unsaved-workflow draft persists on the tracker's `graphChanged` event, and the draft
 * of a SAVED-and-unmodified workflow is deliberately deleted after each persist -- so a
 * bare property write survives refresh only if something else happens to dirty the
 * workflow. captureCanvasState() is the real thing: snapshots into activeState, marks the
 * workflow modified, dispatches graphChanged, and makes the toggle undoable -- consistent
 * with drawer state now living in the file.
 */
function commit() {
  const t = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
  if (!t) return;
  (t.captureCanvasState ?? t.checkState)?.call(t);
}

export function toggle(node, which) {
  if (!node) return false;
  const f = flags(node);
  f[which] = !f[which];
  if (f.inputs || f.outputs) node.properties[DRAWERS] = f;
  else delete node.properties[DRAWERS];
  return f[which];
}

export function isCollapsed(node, which) {
  return flags(node)[which];
}

/**
 * Mark each slot row with whether it carries a link, and each input row with whether the
 * slot is required, so the stylesheet can hide the rest when a block is collapsed. Truth
 * comes from the graph, never from a CSS class core happens to set.
 */
export function syncNode(nodeEl, nodeId, node, scale = 1, slotEls = null) {
  const f = flags(node);
  const wantIn = f.inputs ? "collapsed" : "open";
  const wantOut = f.outputs ? "collapsed" : "open";
  const oldIn = nodeEl.dataset.cablemanagementInputs;
  const oldOut = nodeEl.dataset.cablemanagementOutputs;
  const fresh = oldIn === undefined && oldOut === undefined;

  // Rect reads force layout; the steady state -- no dataset transition, no row flip --
  // must do none at all, or every collapsed node costs a reflow per sync pass.
  const transition = oldIn !== wantIn || oldOut !== wantOut;
  const restoring = fresh && (f.inputs || f.outputs);
  const boxBefore =
    (transition && !fresh) || restoring
      ? nodeEl.getBoundingClientRect().height / scale
      : null;

  const mark = (slots, kind, folded) => {
    // Track the last visible SLOT row (widget rows are never hidden, so they never carry
    // the drawer's reserved-space gap).
    let lastSlotRow = null;
    let moved = false;
    const marked = []; // rows currently carrying the last-row marker
    for (let i = 0; i < (slots?.length ?? 0); i++) {
      // From the caller's one-scan-per-node map when provided -- a document-wide scan
      // per slot made the sync pass quadratic in graph size, and even node-scoped
      // per-slot queries dominated it.
      const dot = slotEls
        ? slotEls.get(`${nodeId}-${kind}-${i}`)
        : nodeEl.querySelector(
            `[data-slot-key="${CSS.escape(`${nodeId}-${kind}-${i}`)}"]`
          );
      // Widget rows FIRST: a widget row nests its own .lg-slot hover wrapper around the
      // dot, and resolving that inner wrapper as "the row" put the linked marker (and the
      // drawer's hiding CSS) on an element inside the widget -- killing the hover dot and
      // corrupting pin anchors whenever the drawer was collapsed.
      const row = dot?.closest(".lg-node-widget") ?? dot?.closest(".lg-slot");
      if (!row) continue;
      const isSlotRow = row.classList.contains("lg-slot");
      const linked =
        kind === "in" ? slots[i].link != null : (slots[i].links?.length ?? 0) > 0;
      // Required inputs hold their row even unlinked -- a mandatory socket is
      // information, not a possibility.
      const req =
        kind === "in" && isSlotRow && slots[i].shape !== OPTIONAL_SHAPE;
      if (req) {
        if (row.dataset.cablemanagementReq !== "1") row.dataset.cablemanagementReq = "1";
      } else if (row.dataset.cablemanagementReq) {
        delete row.dataset.cablemanagementReq;
      }
      const want = linked ? "1" : "0";
      if (row.dataset.cablemanagementLinked !== want) {
        // A hideable row folding or unfolding IN PLACE (implied connection into a hidden
        // slot) shifts everything after it without any watched resize firing.
        if (row.dataset.cablemanagementLinked !== undefined && folded && !req) moved = true;
        row.dataset.cablemanagementLinked = want;
      }
      if ((linked || req) && isSlotRow) lastSlotRow = row;
      if (row.dataset.cablemanagementLast) marked.push(row);
    }
    // The collapsed drawer reserves a row of space below its last visible slot (CSS keys
    // off this marker), so the ellipsis indicator occupies layout rather than overlaying
    // the neighbouring row. Cleared/set only on CHANGE -- the steady state writes
    // nothing (attribute churn invalidates style per pass).
    for (const r of marked) if (r !== lastSlotRow) delete r.dataset.cablemanagementLast;
    if (lastSlotRow && lastSlotRow.dataset.cablemanagementLast !== "1") lastSlotRow.dataset.cablemanagementLast = "1";
    return { any: !!lastSlotRow, moved };
  };
  const mIn = mark(node.inputs, "in", f.inputs);
  const mOut = mark(node.outputs, "out", f.outputs);

  const collapsing =
    (oldIn === "open" && wantIn === "collapsed") ||
    (oldOut === "open" && wantOut === "collapsed");
  const drift = transition
    ? !fresh || restoring // fresh mount only matters when restoring a fold
    : mIn.moved || mOut.moved;
  // An in-place row flip is only discovered DURING mark's writes, so its "before" is
  // measured after the fact -- on an unpinned node that reads as "box unchanged" and
  // fires a spurious nudge, which is safe (floor-restoring) and rare (gesture-driven).
  const box0 =
    drift && boxBefore == null
      ? nodeEl.getBoundingClientRect().height / scale
      : boxBefore;

  // The manual-size check needs the OPEN natural height, measurable only before the fold.
  let floor0 = null;
  let natOpen = null;
  if (collapsing) {
    floor0 = readFloor(nodeEl) ?? boxBefore;
    natOpen = naturalHeight(nodeEl, scale);
  }

  nodeEl.dataset.cablemanagementInputs = wantIn;
  nodeEl.dataset.cablemanagementOutputs = wantOut;

  // With every row of a collapsed block hidden there is no row to carry the gap; the body
  // reserves it as padding instead.
  const setFlag = (name, on) => {
    if (on) nodeEl.dataset[name] = "1";
    else delete nodeEl.dataset[name];
  };
  setFlag("cablemanagementInputsEmpty", f.inputs && !mIn.any);
  setFlag("cablemanagementOutputsEmpty", f.outputs && !mOut.any);

  // Drop the floor with the rows, unless the node was manually made larger than it
  // needs open -- then the size is the user's and the fold happens inside it.
  if (collapsing && floor0 != null && floor0 <= natOpen + 2) {
    const natFolded = naturalHeight(nodeEl, scale);
    const title = titleHeight();
    if (natFolded > title && typeof node.setSize === "function") {
      node.setSize([node.size[0], Math.round(natFolded - title)]);
      node.graph?.setDirtyCanvas?.(true, true);
    }
  }

  ensureHandles(nodeEl, nodeId, node, scale, drift);
  ensureMore(nodeEl, node, scale, drift);
  if (drift) verifySlotSync(nodeEl, node, box0, scale);
}

/** The `--node-height` floor in CSS px (title bar included), or null when unset. */
function readFloor(nodeEl) {
  const v = parseFloat(nodeEl.style.getPropertyValue("--node-height"));
  return Number.isFinite(v) ? v : null;
}

/**
 * How tall the node WANTS to be right now. `--node-height` is a min-height floor, so ask
 * the layout engine directly by zeroing it for one measure. Costs a forced reflow --
 * called only on drawer toggles.
 */
function naturalHeight(nodeEl, scale) {
  const s = nodeEl.style;
  const prev = s.getPropertyValue("--node-height");
  s.setProperty("--node-height", "0px");
  const h = nodeEl.getBoundingClientRect().height / scale;
  if (prev) s.setProperty("--node-height", prev);
  else s.removeProperty("--node-height");
  return h; // CSS px, title bar included
}

/**
 * Fold geometry moved slot rows. Core re-measures slot anchors only when the node's
 * border box CHANGES (its ResizeObserver feeds syncNodeSlotLayoutsFromDOM) -- when the
 * box moved, that pipeline runs on its own. When it did NOT -- the box is pinned by a
 * floor above the content -- nothing fires and link anchors go stale: nudge.
 */
function verifySlotSync(nodeEl, node, boxBefore, scale) {
  requestAnimationFrame(() => {
    if (!nodeEl.isConnected || !node.graph) return;
    const now = nodeEl.getBoundingClientRect().height / scale;
    if (boxBefore == null || Math.abs(now - boxBefore) > 0.5) return;
    nudgeSlotSync(nodeEl, node);
  });
}

/**
 * Make core re-measure this node's slot positions: what a user resize does, a real 1px
 * setSize round trip. Ends by restoring the PRE-nudge floor, never at whatever height
 * the box happened to hold -- setSize raises the floor to the value it is given, and a
 * nudge that re-emits a measured box height ratchets grown nodes tall permanently (the
 * old form of this function did exactly that when a load-time ResizeObserver measure
 * snuck in between mount and the first sync).
 */
function nudgeSlotSync(nodeEl, node) {
  if (!node || node.__cablemanagementNudging || typeof node.setSize !== "function") return;
  node.__cablemanagementNudging = true;
  const floor0 = readFloor(nodeEl);
  node.setSize([node.size[0], node.size[1] + 1]);
  node.graph?.setDirtyCanvas?.(true, true);
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      const h =
        floor0 != null ? Math.round(floor0 - titleHeight()) : node.size[1] - 1;
      node.setSize([node.size[0], h]);
      node.graph?.setDirtyCanvas?.(true, true);
      node.__cablemanagementNudging = false;
    })
  );
}

/** One chevron per block, only where the block has something to collapse. */
function ensureHandles(nodeEl, nodeId, node, scale = 1, drift = false) {
  const f = flags(node);
  let headerH = null; // lazy one-per-node header measure, shared by both carets
  const hidden = (slots, kind) =>
    (slots ?? []).filter((s) =>
      kind === "in"
        ? s.link == null && !s.widget && s.shape === OPTIONAL_SHAPE
        : (s.links?.length ?? 0) === 0
    ).length;

  for (const [which, kind, slots] of [
    ["inputs", "in", node.inputs],
    ["outputs", "out", node.outputs],
  ]) {
    const need = hidden(slots, kind) > 0 || f[which];
    let el = nodeEl.querySelector(`:scope > .cablemanagement-collapse[data-which="${which}"]`);
    if (!need) {
      el?.remove();
      continue;
    }
    if (!el) {
      el = document.createElement("button");
      el.type = "button";
      el.className = "cablemanagement-collapse";
      el.dataset.which = which;
      el.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
      el.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggle(liveNode(nodeEl, node), which);
        commit();
        window.dispatchEvent(new CustomEvent("cablemanagement:resync"));
      });
      nodeEl.appendChild(el);
    }
    const label = f[which] ? `Show all ${which}` : `Show only connected ${which}`;
    if (el.title !== label) el.title = label;
    // The caret shows its action, uniform for both blocks: down closes the drawer, up
    // opens it.
    const glyph = f[which] ? "▴" : "▾";
    if (el.textContent !== glyph) el.textContent = glyph;

    // Both carets are header-height minus a 2px inset top and bottom -- deliberately the
    // SAME height, measured off the header (the bottom strip's own height varies and a
    // caret matched to it goes weird on short badge rows). Each sits beside the block it
    // controls -- inputs top-left, outputs bottom-right, matching the slot contract.
    // The header height is stable in layout px once measured -- re-measure only for a
    // fresh caret or on drift, one rect pair per node (these reads force layout).
    if (el.style.height && !drift) continue;
    if (!headerH) {
      const nr = nodeEl.getBoundingClientRect();
      const body = nodeEl.querySelector('[data-testid^="node-body-"]');
      headerH = { h: body ? (body.getBoundingClientRect().top - nr.top) / scale - 4 : null };
    }
    const h = headerH.h;
    if (h != null && h >= 10) {
      const px = Math.round(h);
      const hPx = `${px}px`;
      if (el.style.height !== hPx) {
        el.style.height = hPx;
        el.style.lineHeight = hPx;
        el.style.width = hPx; // square
        const edge = which === "inputs" ? "left" : "right";
        el.style[edge] = `${-(px + 2)}px`; // keep the 2px gap off the node edge
      }
      const side = which === "inputs" ? "top" : "bottom";
      if (el.style[side] !== "2px") el.style[side] = "2px";
    }
  }
}

/**
 * The collapsed-state indicator: an ellipsis on the seam below the last visible row of a
 * collapsed block -- left for inputs, right for outputs. Always visible while collapsed
 * (the carets are hover-only actions; this is the standing "rows are hidden here" signal),
 * and itself a click target to reopen.
 */
function ensureMore(nodeEl, node, scale, drift = false) {
  const f = flags(node);
  // Fast path: nothing collapsed on this node -- no measuring, just reap strays.
  if (!f.inputs && !f.outputs) {
    for (const el of nodeEl.querySelectorAll(":scope > .cablemanagement-more")) el.remove();
    return;
  }
  // Steady collapsed nodes with their indicators in place re-measure only on drift --
  // the ellipsis rides layout that our own dataset changes are the movers of.
  if (
    !drift &&
    (!f.inputs || nodeEl.querySelector(':scope > .cablemanagement-more[data-which="inputs"]')) &&
    (!f.outputs || nodeEl.querySelector(':scope > .cablemanagement-more[data-which="outputs"]'))
  ) {
    return;
  }
  const nr = nodeEl.getBoundingClientRect();
  for (const [which, rowSel] of [
    [
      "inputs",
      '.lg-slot--input[data-cablemanagement-linked="1"], .lg-slot--input[data-cablemanagement-req]',
    ],
    ["outputs", '.lg-slot--output[data-cablemanagement-linked="1"]'],
  ]) {
    let el = nodeEl.querySelector(`:scope > .cablemanagement-more[data-which="${which}"]`);
    if (!f[which]) {
      el?.remove();
      continue;
    }
    if (!el) {
      el = document.createElement("button");
      el.type = "button";
      el.className = "cablemanagement-more";
      el.dataset.which = which;
      el.textContent = "…";
      el.title = `Show all ${which}`;
      el.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
      el.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        toggle(liveNode(nodeEl, node), which);
        commit();
        window.dispatchEvent(new CustomEvent("cablemanagement:resync"));
      });
      nodeEl.appendChild(el);
    }
    // Centred in the reserved gap: below the last visible row of the block, or inside the
    // propped-open slot column when every row is hidden -- measured, so the indicator
    // lands where the BLOCK is (outputs sit above NodeBadges, not below them).
    const rows = nodeEl.querySelectorAll(rowSel);
    const last = rows[rows.length - 1];
    let top = null;
    if (last) {
      top = (last.getBoundingClientRect().bottom - nr.top) / scale + 3;
    } else {
      const column = nodeEl.querySelector(
        which === "inputs"
          ? '[data-testid^="node-body-"] > div:has(> .ml-auto) > div:not(.ml-auto)'
          : '[data-testid^="node-body-"] > div > .ml-auto'
      );
      if (column) {
        top = (column.getBoundingClientRect().top - nr.top) / scale + 3;
      } else {
        top = which === "inputs" ? 30 : nr.height / scale - 13;
      }
    }
    const topPx = `${Math.round(top)}px`;
    if (el.style.top !== topPx) el.style.top = topPx;
  }
}

export function clearNode(nodeEl) {
  for (const el of nodeEl.querySelectorAll(":scope > .cablemanagement-collapse, :scope > .cablemanagement-more")) el.remove();
  for (const el of nodeEl.querySelectorAll("[data-cablemanagement-last], [data-cablemanagement-req]")) {
    delete el.dataset.cablemanagementLast;
    delete el.dataset.cablemanagementReq;
  }
  delete nodeEl.dataset.cablemanagementInputs;
  delete nodeEl.dataset.cablemanagementOutputs;
  delete nodeEl.dataset.cablemanagementInputsEmpty;
  delete nodeEl.dataset.cablemanagementOutputsEmpty;
}

export function clearAll() {
  for (const el of document.querySelectorAll(".cablemanagement-collapse, .cablemanagement-more")) el.remove();
  for (const el of document.querySelectorAll("[data-cablemanagement-linked], [data-cablemanagement-last], [data-cablemanagement-req]")) {
    delete el.dataset.cablemanagementLinked;
    delete el.dataset.cablemanagementLast;
    delete el.dataset.cablemanagementReq;
  }
  for (const el of document.querySelectorAll("[data-cablemanagement-inputs], [data-cablemanagement-outputs]")) {
    delete el.dataset.cablemanagementInputs;
    delete el.dataset.cablemanagementOutputs;
    delete el.dataset.cablemanagementInputsEmpty;
    delete el.dataset.cablemanagementOutputsEmpty;
  }
}
