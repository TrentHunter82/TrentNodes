// GROUP COMB -- one hotkey that places one of the ordinary combs from this pack at a
// group's corner and enrolls every wire crossing the group boundary.
//
// This is the ONLY file in js/cable/ that is not upstream comfyui-cable-management code.
// It touches none of theirs: it drives the public window.__cablemanagementCombs API, so
// what you get is a completely normal comb -- same record, same gestures, same drag,
// collapse, sort, flip and clipboard behaviour you already know. The hotkey only decides
// WHERE it goes and WHICH wires ride it.
//
// Keeping it at arm's length is the point. Upstream's tree can be re-copied over the top
// without touching this file.

import { app } from "../../../scripts/app.js";

// Mirrors the private geometry in pathing/combs.js. Duplicated rather than exported so
// this file stays a pure consumer of the public API; if a gate ever changes size the
// only symptom is a comb placed a few pixels off, not a break.
const GATE_W = 24;
const PIN_PITCH = 20;
const PAD = 12;
const MARGIN = 18; // clear air between the group frame and the comb

const gateH = (n) => PAD * 2 + Math.max(2, Math.max(0, n - 1)) * PIN_PITCH;

function toast(severity, summary, detail) {
  app.extensionManager?.toast?.add?.({ severity, summary, detail, life: 4000 });
}

// The graph ON SCREEN. app.graph stays pointed at the root even inside a subgraph, and
// resolving against the root there would enroll root links from subgraph coordinates.
const activeGraph = () => app?.canvas?.graph ?? app?.graph ?? null;

function selectedGroups(graph) {
  const out = [];
  for (const it of app.canvas?.selectedItems ?? []) {
    if (it && typeof it.recomputeInsideNodes === "function") out.push(it);
  }
  if (!out.length && app.canvas?.selected_group) out.push(app.canvas.selected_group);
  return out.filter((g) => (graph?._groups ?? []).includes(g));
}

// Wires crossing the group boundary, split by direction. Membership is recomputed here
// because it goes stale the moment a node is dragged in or out of the frame -- and with
// Vue nodes off it is empty until a draw pass has measured the node bounds.
function boundaryLinks(graph, group) {
  group.recomputeInsideNodes?.();
  const inside = new Set((group._nodes ?? []).map((n) => n.id));
  const inbound = [], outbound = [];
  if (!inside.size) return { inbound, outbound };
  for (const link of graph._links?.values?.() ?? []) {
    const src = inside.has(link.origin_id);
    const dst = inside.has(link.target_id);
    if (src === dst) continue; // wholly inside or wholly outside: not our business
    (dst ? inbound : outbound).push(link.id);
  }
  return { inbound, outbound };
}

// Is this wire already riding a comb? The teeth carry the link id, so ask the graph
// rather than keeping a second index that can drift out of sync.
function enrolledLinkIds(graph) {
  const out = new Set();
  for (const comb of graph.extra?.cablemanagement_combs ?? []) {
    for (const lane of comb.lanes ?? []) {
      for (const id of graph.reroutes?.get?.(lane.in)?.linkIds ?? []) out.add(Number(id));
    }
  }
  return out;
}

// Where a comb of n lanes sits for this group and direction.
function placement(group, side, n) {
  const [gx, gy] = group.pos;
  const [gw, gh] = group.size;
  const pairW = GATE_W * 2;
  if (side === "in") {
    // Top-left, hanging off the left edge, below the title strip core draws inside
    // the frame.
    return [gx - pairW - MARGIN, gy + (group.font_size ?? 24) * 1.4];
  }
  // Bottom-right, hanging off the right edge, its BOTTOM level with the group's so it
  // grows upward as lanes are added rather than sinking past the corner.
  return [gx + gw + MARGIN, gy + gh - gateH(n)];
}

function buildSide(graph, api, group, side, links) {
  const already = enrolledLinkIds(graph);
  const fresh = links.filter((id) => !already.has(Number(id)));
  // One wire is not a ribbon -- a single-lane comb is a wire with two extra dots in it,
  // and upstream's own pass decomposes it on the next frame anyway.
  if (fresh.length < 2) return { added: 0, skipped: fresh.length };

  const [x, y] = placement(group, side, fresh.length);
  const combId = api.create(fresh[0], fresh[1], x, y);
  if (combId == null) return { added: 0, skipped: fresh.length };
  for (const id of fresh.slice(2)) api.add(combId, id);

  // Re-place now that the final lane count is known: the gate grew taller as lanes were
  // added, and for the bottom-right comb the anchor is its bottom edge.
  const [fx, fy] = placement(group, side, fresh.length);
  api.move(combId, "in", fx, fy);
  api.move(combId, "out", fx + GATE_W, fy);
  return { added: fresh.length, skipped: 0 };
}

function runCombGroup() {
  const graph = activeGraph();
  const api = window.__cablemanagementCombs;
  if (!graph) return;
  if (!api) {
    toast("warn", "Cable Management not ready", "The comb API has not installed yet. Try again in a moment.");
    return;
  }
  const groups = selectedGroups(graph);
  if (!groups.length) {
    toast("warn", "No group selected", "Select a group first, then press the hotkey.");
    return;
  }
  let added = 0, skipped = 0;
  for (const group of groups) {
    const { inbound, outbound } = boundaryLinks(graph, group);
    for (const [side, links] of [["in", inbound], ["out", outbound]]) {
      const r = buildSide(graph, api, group, side, links);
      added += r.added; skipped += r.skipped;
    }
  }
  graph.setDirtyCanvas(true, true);
  if (added) {
    toast("success", "Combed the group", `${added} wire${added === 1 ? "" : "s"} gathered.`);
  } else if (skipped) {
    toast("info", "Nothing to comb", "Fewer than two un-combed wires cross the boundary.");
  } else {
    toast("info", "Nothing to comb", "No wires cross the boundary of the selected group.");
  }
}

app.registerExtension({
  name: "TrentNodes.GroupComb",

  commands: [
    {
      id: "TrentNodes.GroupComb.CombGroup",
      label: "Cables: comb the selected group's wires",
      function: runCombGroup
    }
  ],

  // Alt+Shift+<letter> is the house pattern for TrentNodes commands, and core rejects a
  // duplicate combo with a red toast on every load -- so R is out (Reroute Roundup) and
  // G is out (too close to KJNodes' Ctrl+Shift+G for comfort). T is free on both counts.
  keybindings: [
    { combo: { key: "t", alt: true, shift: true }, commandId: "TrentNodes.GroupComb.CombGroup" }
  ]
});
