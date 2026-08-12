// The drag gesture, handed to core.
//
// A pass-through pin is a slot in every way the user cares about, so dragging one should BE
// core's drag: the user's Link Render Mode, core's snapping and target highlighting, and the
// node search menu on release over empty canvas. Reimplementing it produced a dashed SVG
// spline that honoured none of those.
//
// The handover is a replayed press. Nothing in the app tree or in litegraph checks
// `isTrusted`, `useSlotLinkInteraction` never captures the pointer, and `dispatchEvent`
// bypasses hit-testing -- so dispatching `pointerdown` on the ORIGIN slot's dot starts a real
// core drag and we stand down. The pin then only supplies the visible start point.
//
// Two constraints that are not obvious and silently break this if ignored:
//
//   * Trackers must live on `window` with capture:true. Core's own move/up handlers are
//     window-capture listeners that call stopPropagation() unconditionally; window is the
//     outermost node, so a capture-phase stop there kills the rest of capture AND all of
//     bubble. A document-capture listener goes deaf mid-gesture.
//   * Pins must be pointer-events:none while dragging. Core resolves hover and drop with
//     document.elementFromPoint, and a pin sitting inside .lg-node shadows the real slot
//     underneath it.

import { activeGraph, nodeFromId, originOf } from "./graph.js";
import { pinAt } from "./pins.js";
import { materialise, ownerOf, hold, release } from "./primitive.js";
import { recordProvenance, recordGateProvenance, provenanceOf, forgetProvenance, severSource } from "./reanchor.js";
import { invalidate, anchorOf } from "./ledger.js";
import { floatSrc, routeSrc, stampFloats, unclaimIfSrc } from "./routestore.js";

/** The element core listens on. Key format is `${nodeId}-${'in'|'out'}-${index}`. */
function slotDot(nodeId, index, isInput) {
  return document.querySelector(
    `[data-slot-key="${CSS.escape(`${nodeId}-${isInput ? "in" : "out"}-${index}`)}"]`
  );
}

/**
 * Replay the press on the real slot. Modifiers are stripped by default: a plain
 * pass-through drag intends neither "move existing links" (shift) nor
 * "disconnect all" (ctrl+alt). The pin's OWN modifier gestures re-add shift
 * deliberately -- the bundle move rides core's machinery -- and handle
 * ctrl+alt before ever forwarding.
 *
 * The replayed event BUBBLES, and onDown is a document-capture listener -- so it sees our
 * own synthetic press before core does, and its retire-the-previous-gesture step would
 * finish() the very gesture being forwarded (reaping the just-made primitive mid-dispatch).
 * dispatchEvent is synchronous, so a flag around it is airtight.
 */
let forwarding = false;
function forward(app, dot, ev, shift = false) {
  forwarding = true;
  try {
    dispatchPress(dot, ev, null, shift);
  } finally {
    forwarding = false;
  }
  // The real gate is layoutStore.getSlotLayout -- module-private, unreadable from here, and
  // its failure is a silent no-op. Verify against the one handle we do have.
  return (app.canvas?.linkConnector?.renderLinks?.length ?? 0) > 0;
}

function dispatchPress(dot, ev, at, shift = false) {
  dot.dispatchEvent(
    new PointerEvent("pointerdown", {
      bubbles: true,
      cancelable: true,
      composed: true,
      pointerId: ev.pointerId,
      pointerType: ev.pointerType || "mouse",
      isPrimary: true,
      button: 0,
      buttons: 1,
      clientX: at ? at[0] : ev.clientX,
      clientY: at ? at[1] : ev.clientY,
      ctrlKey: false,
      metaKey: false,
      altKey: false,
      shiftKey: shift,
    })
  );
}

const FROM = "cablemanagement.from";

/** Every wire currently drawn from a pin: consumers whose record names it. */
function pinWires(graph, host, index) {
  const out = [];
  for (const node of graph?._nodes ?? []) {
    const from = node.properties?.[FROM];
    if (!from) continue;
    for (const [slot, rec] of Object.entries(from)) {
      if (!rec || String(rec[0]) !== String(host.id) || Number(rec[1]) !== index) continue;
      const idx = Number(slot);
      const linkId = node.inputs?.[idx]?.link;
      if (linkId == null) continue;
      const link = graph.getLink ? graph.getLink(linkId) : graph._links?.get?.(linkId);
      if (!link) continue;
      out.push({ node, index: idx, linkId, parentId: link.parentId ?? null });
    }
  }
  return out;
}

/**
 * A wire genuinely leaving a pin takes the pin's apparent-source stamps on its
 * reroute chain with it -- a stale stamp resurrects the pin as source on the
 * next reconnect through the chain (the routefrom lesson). Only stamps naming
 * THIS pin are cleared; another pin's riders keep theirs.
 */
function clearChainStamps(graph, parentId, host, index) {
  let rid = parentId;
  let guard = 0;
  while (rid != null && guard++ < 100) {
    unclaimIfSrc(graph, "floats", rid, host.id, index);
    unclaimIfSrc(graph, "reroutes", rid, host.id, index);
    rid = graph.getReroute?.(rid)?.parentId;
  }
}

/**
 * Hand the drag to a SUBGRAPH BOUNDARY slot.
 *
 * The IO nodes are canvas-drawn litegraph objects with no DOM, so there is no slot element
 * to press. But the canvas itself routes a press inside `slot.boundingRect` to
 * SubgraphInputNode.onPointerDown, which arms linkConnector.dragNewFromSubgraphInput on the
 * pointer's drag-start -- so the press goes to the CANVAS, positioned over the slot. The
 * coordinates carry all the meaning; visibility is irrelevant to canvas hit-testing, so this
 * works even when the IO panel is scrolled off screen.
 *
 * Nothing is verifiable at press time: the drag only materialises when the pointer next
 * moves. The caller defers preview anchoring instead.
 */
function forwardToBoundary(app, graph, originSlot, ev) {
  const slot = graph?.inputNode?.slots?.[originSlot];
  const canvasEl = app.canvas?.canvas;
  const ds = app.canvas?.ds;
  const r = slot?.boundingRect; // Rectangle extends Float64Array: [x, y, w, h]
  if (!slot || !canvasEl || !ds || !r) return false;
  const view = canvasEl.getBoundingClientRect();
  const at = [
    view.left + (r[0] + r[2] * 0.5 + ds.offset[0]) * ds.scale,
    view.top + (r[1] + r[3] * 0.5 + ds.offset[1]) * ds.scale,
  ];
  forwarding = true;
  try {
    dispatchPress(canvasEl, ev, at);
  } finally {
    forwarding = false;
  }
  return true;
}

/**
 * Re-anchor the in-flight preview onto the pin.
 *
 * RenderLink objects are constructed fresh per drag and discarded on reset, so redefining
 * fromPos as a getter needs no cleanup. This is a DIFFERENT mechanism from the ledger, which
 * substitutes arguments to _renderAllLinkSegments for committed links and never sees a
 * RenderLink.
 */
function anchorPreview(app, pinEl, liteNode) {
  const links = app.canvas?.linkConnector?.renderLinks;
  if (!links?.length || !pinEl || !liteNode) return;
  const TITLE = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
  for (const rl of links) {
    try {
      Object.defineProperty(rl, "fromPos", {
        configurable: true,
        get: () => [
          liteNode.pos[0] + pinEl.offsetLeft + pinEl.offsetWidth * 0.5,
          liteNode.pos[1] - TITLE + pinEl.offsetTop + pinEl.offsetHeight * 0.5,
        ],
      });
    } catch {
      /* non-configurable: the preview starts at the true origin, which is cosmetic only */
    }
  }
}

/**
 * Re-anchor the preview once it exists. A boundary handover arms the drag on the pointer's
 * NEXT movement, so at press time there is nothing to anchor yet.
 */
function deferPreview(app, pinEl, liteNode) {
  let tries = 0;
  const tick = () => {
    if ((app.canvas?.linkConnector?.renderLinks?.length ?? 0) > 0) {
      return anchorPreview(app, pinEl, liteNode);
    }
    if (++tries < 30) requestAnimationFrame(tick); // ~half a second, then cosmetic give-up
  };
  requestAnimationFrame(tick);
}

/** Set by install(); call once app.canvas exists. Returns true once subscribed. */
export let attachCanvasEvents = () => false;
// Reassigned per install(); read by the once-only connectFloatingReroute patch.
let currentPending = () => null;
/**
 * The pin the in-flight drag was pulled from, or null. Gate drops (drops.js)
 * consult this: a wire pulled from a pass-through pin must record the pin as
 * its apparent source even when the drop is handled outside core's events.
 */
export let pendingPin = () => null;

export function install(app) {
  // Which pin the in-flight core drag came from, plus any primitive created for it.
  let pending = null;

  const clear = () => {
    pending = null;
    document.body.classList.remove("cablemanagement-dragging");
  };

  const onDown = (ev) => {
    if (forwarding) return; // our own replayed press on its way to core -- not a new gesture
    // Retire any gesture still open (an abandoned drag, or a search box dismissed by a
    // click elsewhere) -- but only for presses on the GRAPH SURFACE. A press in overlaid UI
    // arrives mid-gesture: choosing from the node search box is itself a pointerdown, and
    // the link it creates comes after it. Retiring on that press loses the provenance, and
    // the new node wires up drawn from the true origin instead of the pin.
    if (pending && ev.target?.closest?.(".lg-node, .lgraphcanvas")) finish();
    if (ev.button !== 0) return;
    const pin = pinAt(ev.target);
    if (!pin) return;

    const graph = activeGraph(app);
    const host = nodeFromId(graph, pin.nodeId);
    const slot = host?.inputs?.[pin.index];
    if (!slot) return;

    ev.preventDefault();
    ev.stopPropagation();
    document.body.classList.add("cablemanagement-dragging");

    // Modifier gestures, output-pin contract (audit 2g, greenlit): the pin's
    // wires are the recorded consumers only -- the host's own feed and the
    // true origin's hand-wired consumers are never part of the bundle.
    const modMove = ev.shiftKey && !((ev.ctrlKey || ev.metaKey) && ev.altKey);
    const modCut = (ev.ctrlKey || ev.metaKey) && ev.altKey && !ev.shiftKey;
    if (modMove || modCut) {
      const wires = pinWires(graph, host, pin.index);
      if (wires.length) {
        if (modCut) {
          // Core's ctrl+alt on an output: every link cut at the press, no
          // drag. A direct user cut keeps no memory (disconnect policy):
          // records and the chains' pin stamps go. The cut is at the SOURCE
          // side, matching ctrl+alt on a real output dot: a wire riding a
          // reroute/ribbon chain leaves the chain floating toward its
          // consumer (A, ribbon(floating)->B) -- NOT dangling from the true
          // origin, which would keep feeding the ribbon from a source the
          // user just cut away (2g QA: A.passthrough vs A.output asymmetry).
          for (const w of wires) {
            forgetProvenance(w.node, w.index);
            clearChainStamps(graph, w.parentId, host, pin.index);
            severSource(graph, w.node, w.index, w.linkId);
          }
          invalidate();
          graph.setDirtyCanvas(true, true);
          return clear();
        }
        // Shift: core's move-all-links, filtered to the pin. The press is
        // replayed WITH shift so core builds its moving bundle from the true
        // origin's (or owned primitive's) output; a one-shot veto on
        // before-move-output drops every link that is not the pin's -- and
        // unhides it, because core hides before it asks. Floating wires ride
        // along only when their chain's stamp names this pin.
        let dot = null;
        if (slot.link != null) {
          const origin = originOf(graph, host, pin.index);
          if (!origin || origin.node === graph.inputNode) return clear(); // boundary: no output slot to move from
          dot = slotDot(origin.node.id, origin.slot, false);
        } else if (slot.widget) {
          const prim = ownerOf(graph, host, pin.index);
          if (prim) dot = slotDot(prim.id, 0, false);
        }
        if (!dot || !dot.getBoundingClientRect().width) return clear();
        const lc0 = app.canvas?.linkConnector;
        const keep = new Set(wires.map((w) => w.linkId));
        const veto = (e) => {
          const rl = e?.detail;
          const link = rl?.link;
          if (link && keep.has(link.id)) return;
          const ff = floatFromOf(graph, link);
          if (ff && String(ff[0]) === String(host.id) && Number(ff[1]) === pin.index) return;
          e.preventDefault();
          if (link) delete link._dragging;
          const fr = rl?.fromReroute;
          if (fr) {
            delete fr._dragging;
            lc0?.hiddenReroutes?.delete?.(fr);
          }
        };
        lc0?.events?.addEventListener?.("before-move-output", veto);
        let started = false;
        try {
          started = forward(app, dot, ev, true);
        } finally {
          lc0?.events?.removeEventListener?.("before-move-output", veto);
        }
        if (!started) return clear();
        // NO anchorPreview here: a moving bundle's fixed end is each CONSUMER
        // (core draws moved wires consumer -> pointer). Re-anchoring fromPos
        // onto the pin collapsed the whole bundle into one pin -> pointer line
        // -- indistinguishable from a plain new drag while the real wires sat
        // hidden (Barney's QA: "a new dragging line is created instead").
        pending = { host, index: pin.index, graph, moved: wires };
        return;
      }
      // No wires: core parity -- shift or ctrl+alt on an output with nothing
      // to move or cut degrades to the plain drag below.
    }

    if (slot.link != null) {
      const origin = originOf(graph, host, pin.index);
      if (!origin) return clear();

      // Fed from the subgraph boundary: the origin is the canvas-drawn IO node.
      if (origin.node === graph.inputNode) {
        if (!forwardToBoundary(app, graph, origin.slot, ev)) return clear();
        pending = { host, index: pin.index };
        // If this drag ends on the OUTPUT gate, the stitch it creates should draw
        // from this pin -- stamp it where the stitch seam can see it (cleared on
        // connector reset; stitch.js consumes it once).
        const c = app.canvas?.linkConnector;
        if (c) c.__cablemanagementBoundaryPin = [String(host.id), pin.index];
        deferPreview(app, pin.el, host);
        return;
      }

      const dot = slotDot(origin.node.id, origin.slot, false);
      // A reroute has no DOM element at all: data-slot-key is written only by
      // InputSlot/OutputSlot, and node data is built solely from graph._nodes.
      if (!dot || !dot.getBoundingClientRect().width) return clear();
      if (!forward(app, dot, ev)) return clear();
      anchorPreview(app, pin.el, host);
      pending = { host, index: pin.index, graph };
      return;
    }

    if (!slot.widget) return;

    // Widget literal: nothing produces this value yet, and core cannot drag from a node that
    // does not exist. So materialise the primitive NOW and hand over from its output -- that
    // is what makes the wire appear at all, and makes it core's wire rather than a bespoke
    // one. If the drag is abandoned, the primitive is reaped on reset, leaving the graph
    // exactly as it was.
    const existing = ownerOf(graph, host, pin.index);
    const prim = existing ?? materialise(app, host, pin.index);
    if (!prim) return clear();
    hold(prim); // survive reapOrphans for the duration of the gesture

    // The primitive exists in the graph immediately, but its DOM does not: Vue renders on a
    // later frame, so there is no slot dot to forward to on this tick. Retry across a few
    // frames, then give up and undo. Verified per attempt against renderLinks, because the
    // real gate (layoutStore.getSlotLayout) is unreadable and fails silently.
    // `held` names whatever hold() pinned -- for a PRE-EXISTING primitive `created` is
    // null and the old created-only release paths left it in inFlight forever, making
    // it unreapable (and serialised into every save) for the rest of the session.
    pending = { host, index: pin.index, created: existing ? null : prim, held: prim, graph };
    let tries = 0;
    const tryForward = () => {
      if (!pending) return; // gesture already ended
      const primDot = slotDot(prim.id, 0, false);
      if (primDot && primDot.getBoundingClientRect().width && forward(app, primDot, ev)) {
        anchorPreview(app, pin.el, host);
        return;
      }
      if (++tries < 10) return requestAnimationFrame(tryForward);
      release(prim);
      if (!existing) reap(graph, prim);
      clear();
      graph.setDirtyCanvas(true, true);
    };
    tryForward();
  };

  /** Remove a primitive we speculatively created for a drag that came to nothing. */
  function reap(graph, prim) {
    try {
      for (const linkId of [...(prim.outputs?.[0]?.links ?? [])]) {
        const link = graph.getLink ? graph.getLink(linkId) : graph._links?.get?.(linkId);
        const target = link && nodeFromId(graph, link.target_id);
        target?.disconnectInput?.(link.target_slot);
      }
      graph.remove(prim);
    } catch {
      /* leave it rather than throw mid-gesture */
    }
  }

  document.addEventListener("pointerdown", onDown, true);

  /**
   * Float provenance: a pin drag released on empty canvas can become a floating
   * reroute (link-release menu, "Add Reroute"). The float knows only its TRUE
   * origin, so it would draw from the datasource (or the parked primitive's
   * slot) instead of the pin it was pulled from. Record the pin on the CHAIN'S
   * REROUTE -- unlike the floating link, the reroute survives completion, so
   * every link later pulled through the chain can inherit real provenance.
   * Lives in graph.extra: serialises with the workflow, one unread key without
   * the extension.
   */
  const floatFromOf = (graph, link) => {
    let rid = link?.parentId;
    let guard = 0;
    while (rid != null && guard++ < 100) {
      const rec = floatSrc(graph, rid);
      if (rec) return rec;
      rid = graph.getReroute?.(rid)?.parentId;
    }
    return null;
  };
  // Route provenance (reanchor's prune stamps it): the apparent source of the
  // links that ride a reroute/ribbon chain. A link reconnected THROUGH the
  // chain inherits it -- without this, every reroute-pull reconnect minted a
  // provenance-less link and the ribbon's apparent source decayed one
  // reconnect at a time (churn QA). Null stamp = conflicting sources, no
  // inheritance.
  const routeFromOf = (graph, link) => {
    let rid = link?.parentId;
    let guard = 0;
    while (rid != null && guard++ < 100) {
      const rec = routeSrc(graph, rid);
      if (rec) return rec;
      rid = graph.getReroute?.(rid)?.parentId;
    }
    return null;
  };
  // The prototype patch survives disable/enable cycles but each install() creates a
  // fresh `pending`; the patch reads through this reassigned accessor so it always
  // sees the CURRENT generation (same pattern as attachCanvasEvents).
  currentPending = () => pending;
  pendingPin = () => (pending?.host ? { host: pending.host, index: pending.index } : null);
  const LGN = window.LiteGraph?.LGraphNode;
  if (LGN?.prototype?.connectFloatingReroute && !LGN.prototype.__cablemanagementFloatFrom) {
    LGN.prototype.__cablemanagementFloatFrom = true;
    const origCFR = LGN.prototype.connectFloatingReroute;
    LGN.prototype.connectFloatingReroute = function (...args) {
      const reroute = origCFR.apply(this, args);
      try {
        const p = currentPending();
        if (p && reroute && this.graph) {
          stampFloats(this.graph, [reroute.id], [String(p.host.id), p.index]);
          invalidate();
        }
      } catch {
        /* recording is cosmetic provenance; never break the gesture */
      }
      return reroute;
    };
  }

  /**
   * Provenance for a core-owned drag comes from core's own event, not from pointerup.
   *
   * Core's move/up handlers are window-capture listeners registered at app boot that call
   * stopPropagation() unconditionally. This extension installs long after that, and on the
   * same target and phase the earlier registration runs first -- so a pointerup tracker here
   * is silenced for exactly the gestures core owns. `link-created` carries the new LLink, so
   * the target is in the payload anyway.
   */
  let lc = null;
  const onLinkCreated = (e) => {
    const link = e?.detail;
    if (pending?.held) release(pending.held);
    else if (pending?.created) release(pending.created);
    if (pending) {
      pending.created = null; // it found a consumer; keep it
      pending.held = null;
    }
    if (pending && link) {
      const graph = activeGraph(app);
      // A gate target resolves via nodeFromId (deliberately -- boundary-fed pins need it)
      // but carries NO properties bag: recordProvenance onto it throws mid-gesture.
      if (link.targetIsIoNode ?? String(link.target_id) === "-20") {
        recordGateProvenance(graph, link, pending.host, pending.index);
      } else {
        const target = nodeFromId(graph, link.target_id);
        if (target) {
          recordProvenance(target, link.target_slot, pending.host, pending.index, link.id);
        }
      }
    } else if (link) {
      // No gesture of ours in flight -- but a completion THROUGH a float chain
      // (comb out-pull, drop on the floating reroute) still descends from the
      // pin the float was pulled from. The chain's reroute remembers. Failing
      // that, the route stamp: a reconnect pulled through a live reroute or
      // ribbon inherits the chain's apparent source.
      const graph = activeGraph(app);
      const rec = floatFromOf(graph, link) ?? routeFromOf(graph, link);
      const host = rec ? nodeFromId(graph, rec[0]) : null;
      const target = host ? nodeFromId(graph, link.target_id) : null;
      if (host && target) {
        recordProvenance(target, link.target_slot, host, rec[1], link.id);
      }
    }
    invalidate();
    finish();
    // A link can land on a row the DOM observer cannot see change -- a zero-height slot in
    // a collapsed drawer gains its link with no watched attribute flipping.
    window.dispatchEvent(new CustomEvent("cablemanagement:resync"));
  };

  /**
   * The input end of an EXISTING re-anchored link being picked up and dragged elsewhere.
   *
   * Core builds the preview from the link's true origin -- for a widget pass-through that
   * is the hidden primitive, so the wire would visibly snap to a node that is not there.
   * The provenance names the pin it belongs to: re-anchor the preview onto it, and adopt
   * the gesture as pending so the drop re-records provenance for wherever it lands.
   */
  const onMoveInput = (e) => {
    const rl = e?.detail;
    const link = rl?.link;
    if (!link) return;
    const graph = activeGraph(app);
    const target = nodeFromId(graph, link.target_id);
    const from = provenanceOf(target, link.target_slot);
    if (!from) return;
    const host = nodeFromId(graph, from[0]);
    if (!host) return;
    pending = { host, index: from[1] };
    document.body.classList.add("cablemanagement-dragging"); // pins must not shadow drop targets
    try {
      Object.defineProperty(rl, "fromPos", {
        configurable: true,
        get: () => {
          const el = document.querySelector(
            `.cablemanagement-pin[data-cablemanagement-node="${CSS.escape(from[0])}"][data-cablemanagement-index="${CSS.escape(String(from[1]))}"]`
          );
          return (el && anchorOf(el, host)) ?? [host.pos[0], host.pos[1]];
        },
      });
    } catch {
      /* non-configurable: preview starts at the true origin, cosmetic only */
    }
  };

  /**
   * A moved link landing on a normal input dispatches `input-moved`, NOT `link-created` --
   * that one is reserved for fresh drags (and one subgraph corner). The detail is the
   * render link, which only knows the OLD link; the new one is whatever the graph just
   * assigned its highest id to, created synchronously before the dispatch.
   */
  const onInputMoved = () => {
    if (!pending) return;
    const graph = activeGraph(app);
    const lastId = graph?.state?.lastLinkId;
    const link = graph?.getLink ? graph.getLink(lastId) : graph?._links?.get?.(lastId);
    const target = link && nodeFromId(graph, link.target_id);
    if (target) {
      recordProvenance(target, link.target_slot, pending.host, pending.index, link.id);
    }
    invalidate();
    finish();
    window.dispatchEvent(new CustomEvent("cablemanagement:resync"));
  };
  /**
   * End of gesture.
   *
   * `reset` is NOT reliably the end: releasing on empty canvas opens the node search box,
   * which cancels the reset and resolves later, when the chosen node is created and linked.
   * Clearing `pending` here would lose the provenance for exactly that link -- the new node
   * would be wired from the true producer with no memory of the pin it was dragged from.
   * So a reset only finalises once nothing is still pending a link.
   */
  /**
   * Settle a shift-moved bundle: every wire that actually LEFT the pin (new
   * link id, or cut on an empty release) takes its record and its chain's pin
   * stamps with it -- a surviving record would let reconcile re-source the
   * wire back to the pin's provider (the routefrom lesson). Wires that ended
   * the gesture exactly where they started keep theirs; a drop on another
   * pin has already re-recorded for the new pin, which this never touches.
   * Idempotent: runs on every reset (cancelled ones included) and at finish.
   */
  const settleMoved = () => {
    const moved = pending?.moved;
    if (!moved) return;
    const graph = pending.graph ?? activeGraph(app);
    for (const w of moved) {
      const rec = provenanceOf(w.node, w.index);
      if (!rec || String(rec[0]) !== String(pending.host.id) || Number(rec[1]) !== pending.index) continue;
      if (w.node.inputs?.[w.index]?.link === w.linkId) continue; // never moved
      forgetProvenance(w.node, w.index);
      clearChainStamps(graph, w.parentId, pending.host, pending.index);
    }
    invalidate();
  };

  const finish = () => {
    settleMoved();
    if (pending?.created) {
      // The gesture's own graph, not whatever is on screen NOW -- an Escape-dismissed
      // search box followed by a tab/subgraph switch would otherwise reap (or leak)
      // against the wrong graph.
      const graph = pending.graph ?? activeGraph(app);
      const consumers = (pending.created.outputs?.[0]?.links ?? []).filter((id) => {
        const l = graph.getLink ? graph.getLink(id) : graph._links?.get?.(id);
        return l && String(l.target_id) !== String(pending.host.id);
      });
      // A floating link is a consumer too: parking the drag (comb gate, dangling
      // limbo) leaves outputs[0].links empty while the float lives in
      // graph.floatingLinks. Reaping there implodes the parked lane on the NEXT
      // graph-surface press (this finish() runs from onDown's retire step,
      // before any other handler sees the event).
      let floating = false;
      for (const f of graph.floatingLinks?.values?.() ?? []) {
        if (String(f.origin_id) === String(pending.created.id)) {
          floating = true;
          break;
        }
      }
      if (!consumers.length && !floating) {
        reap(graph, pending.created);
        graph.setDirtyCanvas(true, true);
      }
    }
    if (pending?.held) release(pending.held);
    else if (pending?.created) release(pending.created);
    pending = null;
    invalidate();
    document.body.classList.remove("cablemanagement-dragging");
  };

  // No timer. Releasing on empty canvas opens the node search box, which the user may sit on
  // for as long as they like before choosing -- and only then is the link created. Any
  // timeout races that. `pending` therefore survives `reset` and is retired by whichever
  // comes first: the link being created, or the next gesture starting.
  const onReset = () => {
    settleMoved();
    document.body.classList.remove("cablemanagement-dragging");
  };
  // app.canvas does not exist during setup(), so linkConnector cannot be subscribed there --
  // the listeners would silently never attach. Called from the sync loop once it appears.
  attachCanvasEvents = () => {
    const next = app.canvas?.linkConnector?.events;
    if (!next || next === lc) return !!lc;
    lc = next;
    lc.addEventListener?.("link-created", onLinkCreated);
    lc.addEventListener?.("reset", onReset);
    lc.addEventListener?.("before-move-input", onMoveInput);
    lc.addEventListener?.("input-moved", onInputMoved);
    return true;
  };
  attachCanvasEvents();

  return () => {
    document.removeEventListener("pointerdown", onDown, true);
    lc?.removeEventListener?.("link-created", onLinkCreated);
    lc?.removeEventListener?.("reset", onReset);
    lc?.removeEventListener?.("before-move-input", onMoveInput);
    lc?.removeEventListener?.("input-moved", onInputMoved);
    clear();
  };
}
