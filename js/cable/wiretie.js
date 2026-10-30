// WIRE TIE -- Shift + right-drag a line across wires; on release every wire it cut is
// gathered into a comb sitting on the cut. Inspired by Dan Fessler's wire-tie demo.
//
// Not upstream comfyui-cable-management code. Like groupcomb.js it only builds an
// ORDINARY comb, so everything after the tie -- drag, collapse, sort, flip, clipboard --
// is the pack's own behaviour. Unlike groupcomb it cannot use the public API: api.add
// always inserts the teeth in a link's LAST segment, and a cut can land on any segment
// of a rerouted wire. So it mints the teeth itself, ahead of the exact segment that was
// cut, through the helpers combs.js exports for the clipboard.

import { app } from "../../../scripts/app.js";
import { activeGraph, nodeById } from "./graph.js";
import { isEnabled } from "./enabled.js";
import { combCrossing, combRecords, layoutComb, mintTooth } from "./pathing/combs.js";
import { getCircuit } from "../noodle_wrangler.js";

// Mirrors the private geometry in pathing/combs.js (same as groupcomb.js).
const GATE_W = 24;
const PIN_PITCH = 20;
const PAD = 12;

const gateH = (n) => PAD * 2 + Math.max(2, Math.max(0, n - 1)) * PIN_PITCH;

// Identity-transform context for hit tests: segment paths are built in graph space.
const hitCtx = document.createElement("canvas").getContext("2d");

let cut = null; // { a, b, hits } in graph space while the drag is live
let hitsFrame = 0;

function toast(severity, summary, detail) {
  app.extensionManager?.toast?.add?.({ severity, summary, detail, life: 4000 });
}

// Same commit collapse.js uses: snapshot, mark modified, make it undoable.
function commit() {
  const t = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
  if (!t) return;
  (t.captureCanvasState ?? t.checkState)?.call(t);
}

// canvas.renderedPaths holds every segment drawn last frame: an LLink is the segment
// ending at its target input, a Reroute the segment ending at that reroute. Either one
// is exactly the `before` createReroute wants, so the teeth land on the cut segment.
function segmentLinks(graph, seg) {
  if (seg instanceof window.LiteGraph.Reroute) return [...(seg.linkIds ?? [])];
  return graph.links.has(seg.id) ? [seg.id] : []; // floating links are not tieable
}

function findHits(canvas, graph, a, b) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (len < 4) return [];
  // Sample no farther apart than the stroke is wide, or a wire crossing square-on
  // can slip between two samples.
  hitCtx.lineWidth = (canvas.connections_width ?? 3) + 4;
  const steps = Math.ceil(len / (hitCtx.lineWidth * 0.8));
  const found = [];
  for (const seg of canvas.renderedPaths ?? []) {
    if (!seg.path) continue;
    // The ribbon run inside an existing comb is not a wire you can tie.
    if (seg instanceof window.LiteGraph.Reroute && combCrossing(seg.parentId, seg.id)) continue;
    const links = segmentLinks(graph, seg);
    if (!links.length) continue;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = a[0] + (b[0] - a[0]) * t;
      const y = a[1] + (b[1] - a[1]) * t;
      if (hitCtx.isPointInStroke(seg.path, x, y)) {
        found.push({ seg, links, t, x, y });
        break;
      }
    }
  }
  // One lane per wire: a wire that crosses the cut twice, or a junction reroute whose
  // branches were cut again further down, keeps only the hit nearest the drag start.
  found.sort((p, q) => p.t - q.t);
  const taken = new Set();
  const hits = found.filter((h) => {
    if (h.links.some((id) => taken.has(id))) return false;
    for (const id of h.links) taken.add(id);
    return true;
  });
  // Lane order is top pin first, so order along the gate's vertical axis.
  return hits.sort((p, q) => p.y - q.y || p.x - q.x);
}

// Most wires run left to right; a bundle that mostly flows the other way gets its
// gates mirrored so the pins still face their own ends.
function flowsLeft(graph, hits) {
  let left = 0;
  for (const h of hits) {
    const link = graph.links.get(h.links[0]);
    const src = link && nodeById(graph, link.origin_id);
    const dst = link && nodeById(graph, link.target_id);
    if (src?.pos && dst?.pos && dst.pos[0] < src.pos[0]) left++;
  }
  return left * 2 > hits.length;
}

function tie(graph, hits) {
  const n = hits.length;
  const cx = hits.reduce((s, h) => s + h.x, 0) / n;
  const cy = hits.reduce((s, h) => s + h.y, 0) / n;
  const y = cy - gateH(n) / 2;
  const recs = combRecords(graph, true);
  const id = recs.reduce((m, c) => Math.max(m, c.id), 0) + 1;
  const comb = flowsLeft(graph, hits)
    ? { id, in: { pos: [cx, y], pins: "right" }, out: { pos: [cx - GATE_W, y], pins: "left" }, lanes: [] }
    : { id, in: { pos: [cx - GATE_W, y], pins: "left" }, out: { pos: [cx, y], pins: "right" }, lanes: [] };
  recs.push(comb);
  for (const h of hits) {
    // Both teeth go in ahead of the same segment end, so the chain reads
    // ... -> in -> out -> end, which is the order every lane must have.
    const tIn = mintTooth(graph, h.seg);
    const tOut = mintTooth(graph, h.seg);
    if (tIn && tOut) comb.lanes.push({ in: tIn.id, out: tOut.id });
  }
  layoutComb(graph, comb);
  graph.setDirtyCanvas(true, true);
  commit();
}

function drawCut(canvas, ctx) {
  if (!cut) return;
  const s = canvas.ds.scale;
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = (canvas.connections_width ?? 3) + 4;
  for (const h of cut.hits) ctx.stroke(h.seg.path);
  ctx.setLineDash([6 / s, 4 / s]);
  ctx.lineWidth = 1.5 / s;
  ctx.strokeStyle = "#fff";
  ctx.beginPath();
  ctx.moveTo(cut.a[0], cut.a[1]);
  ctx.lineTo(cut.b[0], cut.b[1]);
  ctx.stroke();
  const n = cut.hits.length;
  if (n) {
    const text = n === 1 ? "Tie needs 2+ wires" : `Tie ${n} wires`;
    ctx.font = `${12 / s}px Arial`;
    const w = ctx.measureText(text).width + 12 / s;
    const x = cut.b[0] + 12 / s, y = cut.b[1] + 10 / s;
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(20,20,24,0.9)";
    ctx.beginPath();
    ctx.roundRect(x, y, w, 20 / s, 4 / s);
    ctx.fill();
    ctx.fillStyle = "#ddd";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 6 / s, y + 10 / s);
  }
  ctx.restore();
}

function refreshHits() {
  hitsFrame = 0;
  const canvas = app.canvas, graph = activeGraph(app);
  if (!cut || !canvas || !graph) return;
  cut.hits = findHits(canvas, graph, cut.a, cut.b);
  canvas.setDirty(true, false);
}

function finish() {
  const canvas = app.canvas, graph = activeGraph(app);
  const done = cut;
  cut = null;
  cancelAnimationFrame(hitsFrame);
  hitsFrame = 0;
  canvas?.setDirty(true, false);
  if (!done || !canvas || !graph) return;
  if (Math.hypot(done.b[0] - done.a[0], done.b[1] - done.a[1]) < 4) return;
  const hits = findHits(canvas, graph, done.a, done.b);
  if (hits.length >= 2) {
    tie(graph, hits);
  } else if (getCircuit() && !window.LiteGraph?.vueNodesMode) {
    toast("info", "Nothing to tie", "Circuit noodles hide the wires from hit tests. Turn them off in Trent Tools.");
  } else if (hits.length === 1) {
    toast("info", "Nothing to tie", "A tie needs at least two wires.");
  }
}

function install() {
  const canvasEl = () => app.canvas?.canvas;
  const graphPt = (e) => {
    const p = app.canvas.convertEventToCanvasOffset(e);
    return [p[0], p[1]];
  };

  // Window CAPTURE listeners: litegraph binds its pointer handlers at construction, so
  // prototype patches never see real events (see noodle_wrangler.js).
  window.addEventListener("pointerdown", (e) => {
    if (cut || e.button !== 2 || !e.shiftKey || e.target !== canvasEl() || !isEnabled()) return;
    const p = graphPt(e);
    cut = { a: p, b: p, hits: [] };
    e.preventDefault();
    e.stopPropagation();
  }, true);

  window.addEventListener("pointermove", (e) => {
    if (!cut) return;
    cut.b = graphPt(e);
    if (!hitsFrame) hitsFrame = requestAnimationFrame(refreshHits);
    app.canvas.setDirty(true, false);
  }, true);

  window.addEventListener("pointerup", (e) => {
    if (!cut || e.button !== 2) return;
    cut.b = graphPt(e);
    e.preventDefault();
    e.stopPropagation();
    finish();
  }, true);

  // Windows raises contextmenu AFTER the button comes up, so the gesture is over by
  // then; a Shift + right-click on the canvas is ours either way.
  window.addEventListener("contextmenu", (e) => {
    if (!cut && !(e.shiftKey && e.target === canvasEl() && isEnabled())) return;
    e.preventDefault();
    e.stopPropagation();
  }, true);

  window.addEventListener("keydown", (e) => {
    if (cut && e.key === "Escape") {
      cut = null;
      app.canvas?.setDirty(true, false);
      e.stopPropagation();
    }
  }, true);
  window.addEventListener("blur", () => {
    cut = null;
  });

  const canvas = app.canvas;
  const prev = canvas.onDrawForeground;
  canvas.onDrawForeground = function (ctx, area) {
    if (prev) prev.call(this, ctx, area);
    try { drawCut(this, ctx); } catch (e) { /* never break drawing */ }
  };
}

app.registerExtension({
  name: "TrentNodes.WireTie",
  setup: install
});
