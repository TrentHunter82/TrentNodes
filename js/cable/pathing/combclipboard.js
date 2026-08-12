// Clipboard survival for ribbons (combs).
//
// Why ribbons flatten on paste while plain reroutes travel: combPass's marquee proxy
// PURGES selected teeth from canvas.selectedItems (converting them to gate
// selection), and core's _serializeItems only serializes reroutes it finds in the
// selection -- so links copy with parentId chains pointing at reroutes absent from
// the clipboard, and the paste drops the chain. We stole the teeth from the
// clipboard; this module hands them back, plus the comb record.
//
// Copy (_serializeItems wrap): for every comb with a lane wired to a selected node,
//   - lane with BOTH endpoint nodes selected -> serialize both teeth (sanitized:
//     linkIds filtered to links that will actually paste, in-tooth parentId kept
//     only when that upstream reroute pastes too) so core's own paste rebuilds the
//     chain and threads the pasted links through it;
//   - lane with ONE side selected -> no teeth (core's validateLinks would reap a
//     linkless reroute anyway); the bag records the connected pin and the paste
//     side synthesizes a floating lane from it;
//   - lane with NO side selected -> dropped.
// The comb geometry rides an extra `cablemanagement_combs` key on the clipboard
// object -- stock frontends ignore unknown keys, so degradation is unchanged.
//
// Paste (_deserializeItems wrap, after core): rebuild comb records from the bag via
// the results remap tables (old reroute id -> new Reroute, old node id -> new node).
// Floating lanes reuse the gestureFloatingEnroll recipe: connectFloatingReroute from
// the pasted pin mints the terminus (= out tooth), mintTooth inserts the second
// tooth ahead of it. A pasted comb under two lanes auto-decomposes next combPass,
// teeth left as plain reroutes -- same rule as [del]. Runs inside _deserializeItems
// so the rebuilt record lands in the paste's own undo step.
import { app } from '../../../../scripts/app.js'
import { isEnabled } from '../enabled.js'
import { combRecords, layoutComb, mintTooth } from './combs.js'

const KEY = 'cablemanagement_combs'

let installed = false

export function installClipboard() {
  if (installed) return true
  const proto = app?.canvas?.constructor?.prototype
  if (!proto || typeof proto._serializeItems !== 'function' || typeof proto._deserializeItems !== 'function') return false
  if (!proto.__cablemanagementCombClipboard) {
    proto.__cablemanagementCombClipboard = true
    const origSer = proto._serializeItems
    proto._serializeItems = function (items) {
      const out = origSer.call(this, items)
      if (out && isEnabled()) {
        try {
          augment(this.graph, out)
        } catch (err) {
          console.warn('[cablemanagement] comb copy failed:', err)
        }
      }
      return out
    }
    const origDe = proto._deserializeItems
    proto._deserializeItems = function (parsed, options) {
      const results = origDe.call(this, parsed, options)
      if (results && isEnabled()) {
        try {
          rebuild(this, parsed, options, results)
        } catch (err) {
          console.warn('[cablemanagement] comb paste failed:', err)
        }
      }
      return results
    }
  }
  installed = true
  return true
}

/** results maps are keyed by number-normalised old ids; be liberal in the lookup. */
function mapGet(map, id) {
  if (!map || id == null) return undefined
  return map.get(id) ?? map.get(Number(id)) ?? map.get(String(id))
}

function augment(graph, serialisable) {
  if (!graph) return
  const sel = new Set((serialisable.nodes ?? []).map((n) => String(n.id)))
  if (!sel.size) return
  const getLink = (id) => (graph.getLink ? graph.getLink(id) : graph._links?.get?.(id))
  const bag = []
  for (const comb of combRecords(graph)) {
    const lanes = []
    for (const lane of comb.lanes) {
      const tin = graph.reroutes?.get?.(lane.in)
      const tout = graph.reroutes?.get?.(lane.out)
      if (!tin || !tout) continue
      // Real links riding the lane (the out tooth carries the lane's full set).
      const links = [...(tout.linkIds ?? [])].map(getLink).filter(Boolean)
      const keep = links.filter((l) => sel.has(String(l.origin_id)) && sel.has(String(l.target_id)))
      // A parked/hanging float's real end counts as a connected pin too -- an
      // already-floating lane whose attached node is selected pastes floating.
      let srcPin = null
      let tgtPin = null
      for (const l of links) {
        if (!srcPin && sel.has(String(l.origin_id))) srcPin = { node: String(l.origin_id), slot: l.origin_slot }
        if (!tgtPin && sel.has(String(l.target_id))) tgtPin = { node: String(l.target_id), slot: l.target_slot }
      }
      for (const f of graph.floatingLinks?.values?.() ?? []) {
        if (f.parentId !== lane.in && f.parentId !== lane.out) continue
        if (!tgtPin && sel.has(String(f.target_id))) tgtPin = { node: String(f.target_id), slot: f.target_slot }
        if (!srcPin && sel.has(String(f.origin_id))) srcPin = { node: String(f.origin_id), slot: f.origin_slot }
      }
      if (keep.length) {
        // Full lane. Sanitized tooth entries, not asSerialisable(): a stale linkId
        // resolves against the ORIGINAL graph's links on a same-graph paste and
        // validateLinks would keep it -- the pasted tooth then claims a link whose
        // chain never passes through it.
        const ids = keep.map((l) => l.id)
        const parentPastes = tin.parentId != null && serialisable.reroutes.some((r) => r.id === tin.parentId)
        serialisable.reroutes.push(
          { id: tin.id, parentId: parentPastes ? tin.parentId : undefined, pos: [tin.pos[0], tin.pos[1]], linkIds: ids },
          { id: tout.id, parentId: tin.id, pos: [tout.pos[0], tout.pos[1]], linkIds: ids }
        )
        lanes.push({ full: { in: lane.in, out: lane.out } })
      } else if (srcPin) {
        lanes.push({ float: 'src', ...srcPin })
      } else if (tgtPin) {
        lanes.push({ float: 'tgt', ...tgtPin })
      }
    }
    if (!lanes.length) continue
    const gate = (g) => ({ pos: [g.pos[0], g.pos[1]], pins: g.pins, ...(g.labels ? { labels: true } : {}) })
    bag.push({
      in: gate(comb.in),
      out: gate(comb.out),
      ...(comb.collapsed ? { collapsed: true } : {}),
      lanes
    })
  }
  if (bag.length) serialisable[KEY] = bag
}

function rebuild(canvas, parsed, options, results) {
  const graph = canvas.graph
  const bag = parsed?.[KEY]
  if (!graph || !Array.isArray(bag) || !bag.length) return
  // The paste offset, derived exactly as core derives it (parsed pos fields are
  // not mutated by the paste; position defaults like _deserializeItems' own).
  let ox = Infinity
  let oy = Infinity
  for (const it of [...(parsed.nodes ?? []), ...(parsed.reroutes ?? [])]) {
    if (!Array.isArray(it?.pos)) continue
    if (it.pos[0] < ox) ox = it.pos[0]
    if (it.pos[1] < oy) oy = it.pos[1]
  }
  if (!Number.isFinite(ox) || !Number.isFinite(oy)) return
  const position = options?.position ?? canvas.graph_mouse
  const dx = position[0] - ox
  const dy = position[1] - oy

  let recs = null // created lazily -- an all-dead bag must not stamp graph.extra
  for (const entry of bag) {
    if (!Array.isArray(entry?.lanes) || !Array.isArray(entry.in?.pos) || !Array.isArray(entry.out?.pos)) continue
    const lanes = []
    for (const l of entry.lanes) {
      if (l?.full) {
        const nin = mapGet(results.reroutes, l.full.in)
        const nout = mapGet(results.reroutes, l.full.out)
        // validateLinks may have reaped a tooth whose links did not paste.
        if (nin && nout && graph.reroutes?.has?.(nin.id) && graph.reroutes?.has?.(nout.id)) {
          lanes.push({ in: nin.id, out: nout.id })
        }
      } else if (l?.float === 'src' || l?.float === 'tgt') {
        const n = mapGet(results.nodes, l.node)
        const slot = l.float === 'src' ? n?.outputs?.[l.slot] : n?.inputs?.[l.slot]
        if (!n || !slot || typeof n.connectFloatingReroute !== 'function') continue
        // Shift-paste already wired this input through the ORIGINAL ribbon
        // (afterRerouteId falls back to the un-remapped parentId) -- leave it.
        if (l.float === 'tgt' && slot.link != null) continue
        const terminus = n.connectFloatingReroute([0, 0], slot)
        const rIn = terminus && mintTooth(graph, terminus)
        if (rIn) lanes.push({ in: rIn.id, out: terminus.id })
      }
    }
    if (!lanes.length) continue
    recs ??= combRecords(graph, true)
    const id = recs.reduce((m, c) => Math.max(m, c.id), 0) + 1
    const gate = (g) => ({
      pos: [g.pos[0] + dx, g.pos[1] + dy],
      pins: g.pins === 'right' ? 'right' : 'left',
      ...(g.labels ? { labels: true } : {})
    })
    const comb = {
      id,
      in: gate(entry.in),
      out: gate(entry.out),
      ...(entry.collapsed ? { collapsed: true } : {}),
      lanes
    }
    recs.push(comb)
    layoutComb(graph, comb)
  }
}
