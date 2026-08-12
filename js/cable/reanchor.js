// Deriving ledger entries from recorded provenance.
//
// Built from our own markers, never from a scan of every link and never from `renderedPaths`
// -- the latter is cleared before drawConnections' early-outs, so a ledger derived from it
// goes blank in Hidden-links mode and for the frames right after a workflow load or an undo,
// which is exactly when the machinery would show.
//
// PROVENANCE, not inference. A link created by dragging a pass-through pin is
// indistinguishable, in the graph, from one wired by hand -- that is the point of resolving
// to a real origin. So we cannot infer which links to re-anchor; guessing would relocate
// wires the user drew deliberately. Instead the drop records where it came from, on the
// TARGET node, keyed by the target input index (an input holds at most one link).
//
// It lives in `properties` for the same reasons `cablemanagement.owned` does: litegraph's sanctioned
// serialised bag, survives save/load so the picture is stable across reloads, and without the
// extension it is one unread property on a node.

import { SUPPRESS, anchorOf, invalidate } from "./ledger.js";
import { isOwned } from "./primitive.js";
import { isStitch } from "./stitch.js";
import { nodeById } from "./graph.js";
import { stampGate, stampFloats, stampRouteChain, floatSrc, gateClaims, unclaim, gc as routesGc } from "./routestore.js";

const FROM = "cablemanagement.from";

/**
 * Record that `target.inputs[targetIndex]` was created by dragging the pin on (host, hostIndex).
 *
 * The LINK ID is part of the record. Provenance describes one specific link; if the user
 * later hand-wires something else into the same input, the slot index alone would still
 * match and the ledger would relocate a wire the user drew deliberately. The id makes the
 * record self-invalidating: link gone or replaced, record ignored and pruned.
 */
export function recordProvenance(target, targetIndex, host, hostIndex, linkId) {
  if (!target || !host) return;
  // Refuse a record naming the target's OWN pin for the same input -- dropping
  // a link whose apparent source is C's pin back onto C would otherwise write
  // "C came from C", legitimising a self-reconnect the graph never made
  // (churn QA: broken.json carried exactly this). The link stays and draws
  // honestly from its true origin.
  if (String(host.id) === String(target.id) && Number(hostIndex) === Number(targetIndex)) return;
  const map = { ...(target.properties?.[FROM] ?? {}) };
  map[String(targetIndex)] = [String(host.id), hostIndex, linkId];
  target.properties[FROM] = map;
}

/**
 * Gate-target flavour of the record above. A link landing on the subgraph OUTPUT GATE has
 * no node to carry properties -- the IO nodes are canvas-drawn, property-less objects (and
 * writing to one crashes) -- so the record lives in graph.extra keyed by LINK ID. Same
 * self-invalidation: link gone, entry pruned. graph.extra serialises with the subgraph.
 */
export function recordGateProvenance(graph, link, host, hostIndex) {
  if (!graph || !link || !host) return;
  stampGate(graph, link.id, [String(host.id), hostIndex]);
}

/** A link whose target is the output gate; tolerates plain objects without the getter. */
function targetsGate(link) {
  return link?.targetIsIoNode ?? String(link?.target_id) === "-20";
}

/**
 * Re-key a float's pin record when its reroute is about to be replaced.
 *
 * Comb operations restructure chains: absorbing a floating reroute into a lane
 * mints teeth and REMOVES the original reroute -- the floatfrom entry keyed by
 * it would orphan, prune would collect it, and the float would snap from its
 * pin back to the true origin (QA: "floating lines pop from their passthrough
 * origin"). Copied onto every survivor: the resolver walks the float's parent
 * chain, so whichever tooth ends up on it wins.
 */
export function migrateFloatFrom(graph, fromId, toIds) {
  const src = floatSrc(graph, fromId);
  if (src == null) return;
  unclaim(graph, "floats", fromId);
  stampFloats(graph, toIds.filter((id) => id != null), src);
}

/**
 * The valid provenance entry for an input, or null (none recorded, or stale).
 * With `graph`, an entry whose host no longer exists is stale too -- the link id alone
 * still matches after the HOST dies (deleting a node mid-chain removes the host but not
 * the link), and a dead-host entry blocks the ledger's own fallbacks.
 */
export function provenanceOf(node, index, graph) {
  const entry = node?.properties?.[FROM]?.[String(index)];
  const slot = node?.inputs?.[index];
  if (!entry || !slot || slot.link == null) return null;
  if (entry[2] != null && entry[2] !== slot.link) return null;
  if (graph && !nodeById(graph, entry[0])) return null;
  return entry;
}

/** Forget provenance for an input (its link was removed or replaced by hand). */
export function forgetProvenance(target, targetIndex) {
  const map = target?.properties?.[FROM];
  if (!map || !(String(targetIndex) in map)) return;
  const next = { ...map };
  delete next[String(targetIndex)];
  if (Object.keys(next).length) target.properties[FROM] = next;
  else delete target.properties[FROM];
}

/**
 * Cut a wire at its SOURCE side. A wire riding a reroute/ribbon chain keeps
 * the consumer's claim as a floating link through the chain -- core's own
 * parked shape via LLink.disconnect(graph, "input"), whose floating marker on
 * the terminus reroute is what makes the parked wire actually RENDER (a
 * hand-built float draws nothing; 2g QA: "all links disappear but the pins
 * remain"). A plain wire has nothing to park and simply disconnects -- the
 * same asymmetry core's own ctrl+alt shows on A.output.
 */
export function severSource(graph, node, slot, linkId) {
  const link = graph?.getLink ? graph.getLink(linkId) : graph?._links?.get?.(linkId);
  if (!link) return;
  if (link.parentId != null && typeof link.disconnect === "function") {
    link.disconnect(graph, "input");
    const input = node.inputs?.[slot];
    if (input && input.link === linkId) input.link = null;
  } else {
    node.disconnectInput(slot, false);
  }
}

/**
 * The live graph link a render link is CARRYING, or null. Core's moved-link
 * drags (MovingInputLink / MovingOutputLink -- pick an existing link up by
 * either end) hold the LLink in .link; fresh drags carry nothing, and a
 * FloatingRenderLink's .link is a floating link that never sits in the graph's
 * link map, so the map lookup is the whole discriminator.
 */
export function carriedLinkOf(graph, rl) {
  const l = rl?.link;
  if (!l || l.id == null || l.id < 0) return null;
  const live = graph?.getLink ? graph.getLink(l.id) : graph?._links?.get?.(l.id);
  return live === l ? l : null;
}

/**
 * Core's move contract, applied by hand: a moved link that LANDS somewhere
 * dies at its donor end (measured baseline P8: move never copies; the
 * empty-canvas drop disconnects via the same method). Our gate surfaces
 * connect through raw graph calls instead of the render link's polymorphic
 * connect methods, so the donor kill they skip happens here: forget the donor
 * input's provenance first (a dead link's record would let reconcile
 * resurrect the feed), then run the render link's own disconnect --
 * MovingInputLink empties the donor input, MovingOutputLink removes the donor
 * output's link. No-op for fresh drags, and for carried links the connect
 * already destroyed by replacement (an input holds one link).
 */
export function settleCarriedLink(graph, rl) {
  if (!carriedLinkOf(graph, rl)) return false;
  if (rl.inputNode) forgetProvenance(rl.inputNode, rl.inputIndex);
  if (typeof rl.disconnect === "function") rl.disconnect();
  return true;
}

/**
 * The true-source pin of a link whose origin is an owned primitive: its host's
 * widget input. Null for links from visible nodes -- an all-disconnected walk
 * there should end in an honest draw from the real origin.
 */
function terminusOf(graph, link) {
  const origin = nodeById(graph, link?.origin_id);
  if (!isOwned(origin)) return null;
  const host = origin.properties?.["cablemanagement.host"];
  const slot = origin.properties?.["cablemanagement.hostSlot"];
  return host != null && slot != null ? [String(host), Number(slot)] : null;
}

function pinEl(nodeId, index) {
  return document.querySelector(
    `.cablemanagement-pin[data-cablemanagement-node="${CSS.escape(String(nodeId))}"][data-cablemanagement-index="${CSS.escape(String(index))}"]`
  );
}

/**
 * A resolver that survives re-renders.
 *
 * Anchors must be computed per frame (a cached point freezes when the origin node is
 * dragged), and the element itself must be re-fetched when it detaches -- Vue replaces a
 * node's DOM on many updates, and a detached element reports offsetLeft 0, which would slam
 * the anchor to the node's top-left corner.
 */
function anchorResolver(graph, nodeId, index, terminus, link) {
  let el = pinEl(nodeId, index);
  return () => {
    if (!el?.isConnected) el = pinEl(nodeId, index);
    const node = nodeById(graph, nodeId);
    if (el && node) {
      // Memoized happy path -- one honesty check, no walk, no re-query.
      const connected = node.inputs?.[Number(index)]?.link != null;
      if (connected || (!!terminus && String(terminus[0]) === String(nodeId) && Number(terminus[1]) === Number(index))) {
        return anchorOf(el, node);
      }
    }
    return fallbackAnchor(graph, nodeId, index, terminus, link);
  };
}

/**
 * Resolve the recorded pin, or the last honest anchor upstream of it.
 *
 * The hop-0 happy path (pin present, input connected) is the old resolver. The
 * rest is the mid-chain-disconnect fallback: a pass-through pin disappears when
 * its input loses its feed, so the wire should fall back to the LAST SURVIVING
 * anchor on its route, not snap to the true origin -- walk upstream through
 * each host's own (possibly dormant) record until an honest pin turns up.
 *
 * "Honest" is the crux, in three guards:
 * - A pin is accepted only while its input is CONNECTED -- still part of the
 *   route -- or when it is the route's true source (`terminus`: the owned
 *   primitive's host widget). A WIDGET input that lost its feed reverts to its
 *   own literal and keeps a pin, but anchoring there would claim the consumer
 *   follows this node's literal while its link still rides the chain's hidden
 *   primitive (edit the widget, nothing moves).
 * - A pin missing while its link is LIVE is a Vue re-render transient; drawing
 *   from further upstream for a frame would flicker the wire across the graph.
 * - Dormant records are memory, not truth: a dead host or missing record ends
 *   the walk (honest draw from the true endpoint beats a guess).
 */
function fallbackAnchor(graph, nodeId, index, terminus, link) {
  const seen = new Set();
  let id = String(nodeId);
  let idx = Number(index);
  let brokenAt = null; // first disconnected node walked past -- proximity hint for the ribbon tip
  for (let guard = 0; guard < 100; guard++) {
    const key = `${id}:${idx}`;
    if (seen.has(key)) return null;
    seen.add(key);
    const node = nodeById(graph, id);
    if (!node) return null;
    const el = pinEl(id, idx);
    const connected = node.inputs?.[idx]?.link != null;
    const isTerminus = !!terminus && String(terminus[0]) === id && Number(terminus[1]) === idx;
    if (el && (connected || isTerminus)) {
      // Reroutes and ribbons along the route beat a pin further upstream: if the
      // broken hop's feed survived as a dangling chain (core disconnect-to-float
      // parks it on the last reroute), its tip is the LAST element of the route
      // -- anchor there, so the wire visibly rejoins the ribbon exit.
      return (brokenAt && floatTipNear(graph, link, brokenAt)) ?? anchorOf(el, node);
    }
    if (connected) return null; // transient unmount, not a disconnect
    brokenAt ??= node;
    const rec = node.properties?.[FROM]?.[String(idx)];
    if (!rec) return floatTipNear(graph, link, brokenAt); // no pin memory -- the chain may still hold the route
    id = String(rec[0]);
    idx = Number(rec[1]);
  }
  return null;
}

/**
 * The surviving remains of a broken hop's feed: a floating link descending from
 * the SAME origin as the consumer's own link (both were pulled from one source)
 * and parked on a reroute chain. Its tip reroute is the ribbon/reroute exit the
 * consumer's wire should rejoin. Nearest-to-the-break wins when the origin fans
 * out into several dangling chains; the consumer's own floating self is skipped
 * (a float resolving through this walk must not anchor to its own tip).
 */
function floatTipNear(graph, link, nearNode) {
  if (!link) return null;
  let best = null;
  let bd = Infinity;
  for (const f of graph.floatingLinks?.values?.() ?? []) {
    if (f === link) continue;
    if (String(f.origin_id) !== String(link.origin_id) || f.origin_slot !== link.origin_slot) continue;
    if (f.parentId == null) continue;
    const r = graph.getReroute?.(f.parentId) ?? graph.reroutes?.get?.(f.parentId);
    if (!r?.pos) continue;
    const d = nearNode?.pos ? Math.hypot(r.pos[0] - nearNode.pos[0], r.pos[1] - nearNode.pos[1]) : 0;
    if (d < bd) {
      bd = d;
      best = [r.pos[0], r.pos[1]];
    }
  }
  return best;
}

/**
 * @returns Map<linkId, SUPPRESS | { startPos }>
 */
export function build(graph) {
  const out = new Map();
  if (!graph) return out;

  const getLink = (id) =>
    graph.getLink ? graph.getLink(id) : graph._links?.get?.(id) ?? graph.links?.[id];

  // 1. Owned primitives -- pure machinery.
  //    The wire driving the host's own widget is suppressed; wires to daisy-chain consumers
  //    are re-anchored onto the widget pin the user dragged from.
  for (const node of graph.nodes ?? []) {
    if (!isOwned(node)) continue;
    const hostId = node.properties?.["cablemanagement.host"];
    const host = nodeById(graph, hostId);

    for (const linkId of node.outputs?.[0]?.links ?? []) {
      const link = getLink(linkId);
      if (!link) continue;
      // The primitive is never wired to its host, so there is no machinery link to
      // suppress -- only consumer links to re-anchor onto the pin they were pulled from.
      const target = nodeById(graph, link.target_id);
      const from = provenanceOf(target, link.target_slot, graph);
      const slot = host ? node.properties?.["cablemanagement.hostSlot"] : null;
      const anchor = from ?? (host && slot != null ? [String(host.id), slot] : null);
      // The host's widget pin is the route's true source: the walk may accept it
      // even though a widget input holds no link (the literal lives there).
      const terminus = host && slot != null ? [String(host.id), slot] : null;
      if (anchor) out.set(link.id, { resolve: anchorResolver(graph, anchor[0], anchor[1], terminus, link) });
    }
  }

  // 2. Input pass-throughs -- re-anchor onto the pin the drop came from.
  for (const node of graph.nodes ?? []) {
    const map = node.properties?.[FROM];
    if (!map) continue;
    for (const slotIdx of Object.keys(map)) {
      const from = provenanceOf(node, Number(slotIdx), graph);
      if (!from) continue; // stale record: draw honestly, prune() will drop or retarget it
      const link = getLink(node.inputs[Number(slotIdx)].link);
      if (!link || out.has(link.id)) continue; // owned-primitive pass above wins

      if (!nodeById(graph, from[0])) continue; // source node gone: draw it honestly
      out.set(link.id, { resolve: anchorResolver(graph, from[0], from[1], terminusOf(graph, link), link) });
    }
  }

  // 3. Floating links -- half-links parked by a pin drag (link-release menu's
  //    "Add Reroute", comb lanes). They render through the same choke point as
  //    real links but carry no target to hold provenance, so the pulled-from
  //    pin is recorded on the CHAIN'S REROUTE (drag.js writes graph.extra).
  //    Fallback: an owned-primitive origin is self-describing (host + slot).
  for (const link of graph.floatingLinks?.values?.() ?? []) {
    let rec = null;
    let rid = link.parentId;
    let guard = 0;
    while (rid != null && guard++ < 100 && !rec) {
      rec = floatSrc(graph, rid) ?? null;
      if (!rec) rid = graph.getReroute?.(rid)?.parentId;
    }
    if (!rec) {
      const origin = nodeById(graph, link.origin_id);
      if (isOwned(origin)) {
        rec = [String(origin.properties["cablemanagement.host"]), Number(origin.properties["cablemanagement.hostSlot"])];
      }
    }
    if (rec && nodeById(graph, rec[0])) {
      // 'f'-prefixed key: floats have their own id counter, so a bare float
      // id can collide with a REAL link's -- this entry used to overwrite the
      // real link's substitution (and the lookup handed either entry to
      // either object; Barney's floating_input workflow, float 68 vs link 68).
      out.set("f" + link.id, { resolve: anchorResolver(graph, rec[0], rec[1], terminusOf(graph, link), link) });
    }
  }

  // 4. Links INTO the output gate -- provenance from graph.extra (the gate cannot
  //    carry it; see recordGateProvenance).
  for (const [lid, rec] of gateClaims(graph)) {
    const link = getLink(Number(lid));
    if (!link || !targetsGate(link) || out.has(link.id) || !rec) continue;
    if (!nodeById(graph, rec[0])) continue;
    out.set(link.id, { resolve: anchorResolver(graph, rec[0], rec[1]) });
  }

  // 5. Stitch reroutes created from a pin -- the primitive treatment: every
  //    outgoing segment re-anchors onto the pin, and the incoming gate segment
  //    is suppressed (the pin already tells that half of the story). A stitch
  //    with no record, or whose pin host died, draws honestly.
  for (const node of graph.nodes ?? []) {
    const rec = node.properties?.["cablemanagement.stitchFrom"];
    if (!rec || !isStitch(node)) continue;
    if (!nodeById(graph, rec[0])) continue;
    for (const linkId of node.outputs?.[0]?.links ?? []) {
      const link = getLink(linkId);
      if (!link || out.has(link.id)) continue;
      out.set(link.id, { resolve: anchorResolver(graph, rec[0], rec[1]) });
    }
    const inId = node.inputs?.[0]?.link;
    const inLink = inId != null ? getLink(inId) : null;
    if (inLink && !out.has(inLink.id)) out.set(inLink.id, SUPPRESS);
  }

  return out;
}

/**
 * Drop provenance whose link no longer exists or no longer matches, so a hand-rewired input
 * stops pretending it came from a pin.
 *
 * One case is retargeted instead of dropped: a record whose HOST died while its link
 * survives -- deleting a node in the middle of an apparent chain does exactly this. If the
 * surviving link's origin is hidden machinery, the machinery's own host pin is the only
 * honest place left to draw from; rewrite the record onto it. Any other origin is a
 * visible node, and honest drawing needs no record at all.
 */
export function prune(graph) {
  let cured = false;
  const getLink = (id) =>
    graph.getLink ? graph.getLink(id) : graph._links?.get?.(id) ?? graph.links?.[id];
  // Core seeds createReroute(pos, link) with floatingLinkIds=[link.id] -- a
  // phantom (never pruned live, stock-reproducible; sweep bug 1) that defeats
  // reroute GC and VETOES disconnect-to-float, which is gated on the set being
  // empty. combPass already scrubs teeth; scrub every reroute here so a plain
  // reroute chain survives a mid-chain disconnect as a dangling float -- the
  // route memory the fallback walk rejoins.
  for (const r of graph?.reroutes?.values?.() ?? []) {
    for (const fid of [...(r.floatingLinkIds ?? [])]) {
      if (!graph.floatingLinks?.has?.(fid)) r.floatingLinkIds.delete(fid);
    }
  }
  // Route-registry claims whose carrier (reroute / gate link) or src host
  // died are dead weight; routes die with their last claim (routestore gc).
  routesGc(graph, {
    liveReroute: (rid) => !!graph.reroutes?.get?.(Number(rid)),
    liveLink: (lid) => {
      const link = getLink(Number(lid));
      return !!link && targetsGate(link);
    },
    liveNode: (nid) => !!nodeById(graph, nid),
  });
  // Stitch pin records whose host died: the stitch stays (it still bridges),
  // only the pretty-draw record goes.
  for (const node of graph?.nodes ?? []) {
    const rec = node.properties?.["cablemanagement.stitchFrom"];
    if (rec && !nodeById(graph, rec[0])) delete node.properties["cablemanagement.stitchFrom"];
  }
  // Route provenance: a reroute (or ribbon tooth) REMEMBERS the apparent
  // source of the links riding it, so a link later reconnected THROUGH the
  // chain can inherit a record (drag.js consumes this on link-created). Stamps
  // persist in graph.extra once written -- the link whose record stamped them
  // may die, and the whole point is surviving that -- and go null on conflict
  // (two provenance-carrying links with different sources on one reroute:
  // ambiguous, inherit nothing). Cleaned when the reroute or host dies.
  {
    const getRoute = (rid) => graph.getReroute?.(rid) ?? graph.reroutes?.get?.(rid);
    for (const node of graph?.nodes ?? []) {
      const map = node.properties?.[FROM];
      if (!map) continue;
      for (const slotIdx of Object.keys(map)) {
        const from = provenanceOf(node, Number(slotIdx), graph);
        if (!from) continue;
        const link = getLink(node.inputs[Number(slotIdx)].link);
        // The record's whole chain shares ONE route (one wire = one identity);
        // stampRouteChain keeps the v0 per-hop semantics -- absent stamps
        // claim, matching stamps hold, a differing stamp turns the hop into a
        // conflict claim that inherits nothing.
        const rids = [];
        let rid = link?.parentId;
        let guard = 0;
        while (rid != null && guard++ < 100) {
          rids.push(rid);
          rid = getRoute(rid)?.parentId;
        }
        if (rids.length) stampRouteChain(graph, rids, [String(from[0]), Number(from[1])]);
      }
    }
  }

  for (const node of graph?.nodes ?? []) {
    const map = node.properties?.[FROM];
    if (!map) continue;
    for (const slotIdx of Object.keys(map)) {
      const idx = Number(slotIdx);
      // Self-referential record ("this input came from its own pin") -- written
      // by pre-guard versions during churn; poison for the walk and the move
      // gesture. Purge on sight.
      const rec = map[slotIdx];
      if (rec && String(rec[0]) === String(node.id) && Number(rec[1]) === idx) {
        forgetProvenance(node, idx);
        cured = true;
        continue;
      }
      if (provenanceOf(node, idx, graph)) continue;
      const linkId = node.inputs?.[idx]?.link;
      // Input disconnected (link GONE, not replaced): keep the record dormant.
      // It is the route memory the fallback walk follows -- and what lets a
      // re-established feed restore the chain -- while the stale link id inside
      // it keeps it from ever applying to a hand-wired replacement.
      if (linkId == null && node.inputs?.[idx]) continue;
      const link = linkId != null ? getLink(linkId) : null;
      const origin = link ? nodeById(graph, link.origin_id) : null;
      const host = isOwned(origin) ? nodeById(graph, origin.properties["cablemanagement.host"]) : null;
      if (host) {
        recordProvenance(node, idx, host, Number(origin.properties["cablemanagement.hostSlot"]), link.id);
      } else {
        forgetProvenance(node, idx);
      }
      cured = true;
    }
  }
  // Cures rewrite PROPERTIES, and the ledger only rebuilds on link mutations --
  // without this, a purged record kept drawing its stale entry until the next
  // unrelated rebuild (observed: a poisoned workflow showed its self-anchor
  // until a page refresh).
  if (cured) invalidate();
}
