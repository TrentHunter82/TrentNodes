// Unpack survival: translating identity-keyed records across core's re-mint.
//
// LGraph.unpackSubgraph re-creates every interior node, link and reroute with a
// fresh id, but carries node PROPERTIES verbatim -- so immediately after an
// unpack the pass-through machinery is fully intact and every record points at
// ids that no longer exist. Left alone, the extension's own sync pass then
// destroys what core preserved: prune() forgets provenance (stale link ids) and
// reapIfUnused() deletes a still-consumed hidden primitive because its host
// pointer dangles -- severing real links (measured: probe-unpack.mjs, the
// "unpack removes any internal structure" report).
//
// Core builds exactly the maps we need (nodeIdMap / linkIdMap / rerouteIdMap)
// but they are method-local, and unlike conversion (which fires
// "subgraph-converted") unpack fires no event. So: wrap the method, snapshot
// the interior before, and rebuild the mappings after from two facts the
// implementation guarantees -- new nodes are created in subgraph.nodes order,
// and migrated reroutes are created in subgraph.reroutes iteration order.
//
// The translation runs SYNCHRONOUSLY in the same task as the unpack, before the
// rAF-deferred sync pass can reap anything, and it is NOT gated on the master
// toggle: it repairs serialised records, not rendering -- a disabled extension
// must not lose a workflow's machinery to an unpack either (records going stale
// while off would get the primitive reaped on the next enable).
//
// Known core limitations this cannot mend (they are dropped BY CORE, not by us):
// floating links do not survive unpack at all (only _links is iterated), so
// dangling comb lanes and parked floats die with the subgraph, and gate-target
// records (gatefrom) describe links onto io-nodes that stop existing.

const FROM = "cablemanagement.from";
const HOST = "cablemanagement.host";
const STITCHFROM = "cablemanagement.stitchFrom";
const COMBS = "cablemanagement_combs";

let installed = false;

export function install(app) {
  if (installed) return;
  // unpackSubgraph lives on LGraph.prototype; subgraph instances inherit it, so
  // one patch covers nested unpacks too. Walk the chain rather than assume the
  // constructor -- the app's graph may be a subclass.
  let proto = app?.graph ? Object.getPrototypeOf(app.graph) : null;
  while (proto && !Object.prototype.hasOwnProperty.call(proto, "unpackSubgraph")) {
    proto = Object.getPrototypeOf(proto);
  }
  if (!proto) return;
  const orig = proto.unpackSubgraph;
  proto.unpackSubgraph = function (subgraphNode, options) {
    let snap = null;
    try {
      snap = snapshot(this, subgraphNode);
    } catch {
      snap = null; // never let bookkeeping block the unpack itself
    }
    const ret = orig.call(this, subgraphNode, options);
    try {
      if (snap) translate(this, snap);
    } catch (err) {
      console.warn(
        "[cablemanagement] unpack record translation failed; pass-through records from this subgraph may be stale",
        err
      );
    }
    return ret;
  };
  installed = true;
}

function snapshot(graph, subgraphNode) {
  const sub = subgraphNode?.subgraph;
  if (!sub) return null;
  return {
    oldNodes: [...(sub.nodes ?? [])].map((n) => ({
      id: String(n.id),
      type: String(n.type),
      pos: [n.pos[0], n.pos[1]],
    })),
    oldRerouteIds: [...(sub.reroutes?.keys?.() ?? [])],
    preNodes: new Set(graph.nodes ?? []),
    preRerouteIds: new Set(graph.reroutes?.keys?.() ?? []),
    combs: (sub.extra?.[COMBS] ?? []).map((c) => ({
      in: { ...c.in, pos: [c.in.pos[0], c.in.pos[1]] },
      out: { ...c.out, pos: [c.out.pos[0], c.out.pos[1]] },
      lanes: (c.lanes ?? []).map((l) => ({ ...l })),
    })),
  };
}

function translate(graph, snap) {
  const added = (graph.nodes ?? []).filter((n) => !snap.preNodes.has(n));
  if (added.length !== snap.oldNodes.length) return;
  const nodeMap = new Map();
  for (let i = 0; i < added.length; i++) {
    // Order is the mapping; a type mismatch means the order assumption broke.
    // Bail wholesale -- a stale record draws honestly, a WRONG one relocates
    // wires onto the wrong node.
    if (String(added[i].type) !== snap.oldNodes[i].type) return;
    nodeMap.set(snap.oldNodes[i].id, String(added[i].id));
  }
  // Hosts OUTSIDE the subgraph kept their ids; only map interior ones.
  const mapNode = (id) => nodeMap.get(String(id)) ?? String(id);

  for (const n of added) {
    const p = n.properties;
    if (!p) continue;
    if (p[HOST] != null) p[HOST] = mapNode(p[HOST]);
    if (Array.isArray(p[STITCHFROM])) p[STITCHFROM] = [mapNode(p[STITCHFROM][0]), p[STITCHFROM][1]];
    const from = p[FROM];
    if (from) {
      const next = {};
      for (const k of Object.keys(from)) {
        const [hostId, hostIdx, oldLid] = from[k];
        const lid = n.inputs?.[Number(k)]?.link;
        // Severed memory (null link id) survives the unpack: the fragment's
        // shape is worth exactly as much outside the subgraph as in it. Its
        // tail reroute cannot follow -- core drops floating links on unpack,
        // so the parked chain is gone -- the ghost falls back to the pin.
        if (oldLid === null && lid == null) {
          next[k] = [mapNode(hostId), hostIdx, null];
          continue;
        }
        // A live record's link id is re-stamped from the recreated slot: core
        // rebuilt the same topology, so the input's CURRENT link is the same
        // wire under a new id. No link survived -> drop rather than lie.
        if (lid == null) continue;
        next[k] = [mapNode(hostId), hostIdx, lid];
      }
      if (Object.keys(next).length) p[FROM] = next;
      else delete p[FROM];
    }
  }

  // Comb records died with the subgraph's extra, but their teeth (reroutes)
  // were migrated -- in iteration order, the second guaranteed fact. Rebuild
  // the records against the new reroute ids; gates follow the same positional
  // offset core applied to the nodes.
  if (snap.combs.length) {
    const newRids = [...(graph.reroutes?.keys?.() ?? [])].filter((id) => !snap.preRerouteIds.has(id));
    if (newRids.length !== snap.oldRerouteIds.length) return;
    const rMap = new Map(snap.oldRerouteIds.map((old, i) => [old, newRids[i]]));
    const dx = added.length ? added[0].pos[0] - snap.oldNodes[0].pos[0] : 0;
    const dy = added.length ? added[0].pos[1] - snap.oldNodes[0].pos[1] : 0;
    const recs = ((graph.extra ??= {})[COMBS] ??= []);
    let nextId = recs.reduce((m, c) => Math.max(m, c?.id ?? 0), 0);
    for (const c of snap.combs) {
      const lanes = c.lanes
        .map((l) => ({ in: rMap.get(l.in), out: rMap.get(l.out) }))
        .filter((l) => l.in != null && l.out != null);
      // Below two lanes combPass would dissolve it on sight; skipping leaves
      // the surviving tooth as the plain reroute it is about to become anyway.
      if (lanes.length < 2) continue;
      recs.push({
        id: ++nextId,
        in: { ...c.in, pos: [c.in.pos[0] + dx, c.in.pos[1] + dy] },
        out: { ...c.out, pos: [c.out.pos[0] + dx, c.out.pos[1] + dy] },
        lanes,
      });
    }
  }
  graph.setDirtyCanvas?.(true, true);
}
