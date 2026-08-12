// The v1 route registry -- the versioned store behind SCHEMA.md.
//
// One shape, one owner: every apparent-source stamp that used to live in three
// unversioned graph.extra tables (cablemanagement_routefrom / _floatfrom /
// _gatefrom) now lives under cablemanagement_routes as CLAIMS pointing at
// shared ROUTE objects. Identity is the point: stamps written by one gesture
// share one route, so the satellites (route lifecycle, death cleanup) get an
// object to hang off instead of re-deriving groups from scattered keys.
//
// Discipline rules (the float-id collision lesson):
//   - claims.reroutes / claims.floats key REROUTE ids; claims.gatelinks keys
//     REAL link ids. No table ever keys a floating-link id.
//   - Reads NEVER create state: a graph without legacy keys and without the
//     v1 key stays byte-identical through open+save (the NO_RECORDS lesson).
//     Legacy keys present = the one exception: first touch adapts, verbatim.
//
// The adapter is VERBATIM by ruling ('not fixing stuff during translation,
// not trying to infer user intention in any way'): every legacy entry becomes
// its own single-claim route -- no grouping is guessed, conflict-null stamps
// stay per-entry. Multi-claim routes only ever emerge from fresh gestures.

const KEY = "cablemanagement_routes";
const LEGACY = {
  reroutes: "cablemanagement_routefrom",
  floats: "cablemanagement_floatfrom",
  gatelinks: "cablemanagement_gatefrom",
};
const KINDS = ["reroutes", "floats", "gatelinks"];

/** The v1 bag, or null. Adapts legacy keys on sight; never creates otherwise. */
function bagOf(graph, create = false) {
  if (!graph) return null;
  let x = graph.extra;
  const hasLegacy = !!x && KINDS.some((k) => x[LEGACY[k]]);
  if (!x?.[KEY] && !hasLegacy && !create) return null;
  x ??= graph.extra = {};
  const bag = (x[KEY] ??= { v: 1, seq: 1, routes: {}, claims: { reroutes: {}, floats: {}, gatelinks: {} } });
  if (hasLegacy) {
    adapt(x, bag);
    scrubAdopted(graph, bag);
  }
  return bag;
}

/**
 * Adoption-time defect detection (RULING: 'while adopting a v0 workflow v1
 * should detect that type of a defect'). v0 could not guarantee the
 * annotation layer matched the graph -- a claim can outlive its carrier
 * reroute/link or its source host, describing a route with nothing left
 * alive. Translation stays verbatim; THIS pass then drops every claim the
 * graph itself proves dead, and says so. Checks run only where the graph can
 * answer them (a bare {extra} object adapts untouched -- there is no truth
 * to check against).
 */
function scrubAdopted(graph, bag) {
  const canReroute = !!(graph.reroutes?.get || graph.getReroute);
  const canLink = !!(graph.getLink || graph._links?.get);
  const canNode = typeof graph.getNodeById === "function";
  if (!canReroute && !canLink && !canNode) return;
  const liveReroute = (id) =>
    !canReroute || !!(graph.getReroute?.(Number(id)) ?? graph.reroutes?.get?.(Number(id)));
  const liveLink = (id) => {
    if (!canLink) return true;
    const l = graph.getLink ? graph.getLink(Number(id)) : graph._links?.get?.(Number(id));
    return !!l && (l.targetIsIoNode ?? String(l.target_id) === "-20");
  };
  const liveNode = (id) => !canNode || !!(graph.getNodeById(id) ?? graph.getNodeById(Number(id)));
  const dropped = { reroutes: 0, floats: 0, gatelinks: 0 };
  const liveCarrier = { reroutes: liveReroute, floats: liveReroute, gatelinks: liveLink };
  for (const kind of KINDS) {
    const claims = bag.claims[kind];
    for (const id of Object.keys(claims)) {
      const rid = claims[id];
      const src = bag.routes[rid]?.src ?? null;
      if (liveCarrier[kind](id) && (src == null || liveNode(src[0]))) continue;
      delete claims[id];
      dropIfUnclaimed(bag, rid);
      dropped[kind]++;
    }
  }
  const total = dropped.reroutes + dropped.floats + dropped.gatelinks;
  if (total) {
    console.info(
      `[cablemanagement] v0 adoption: dropped ${total} claim(s) with no living justification`,
      dropped
    );
  }
}

/** Verbatim v0 -> v1: one single-claim route per legacy entry, keys removed. */
function adapt(extra, bag) {
  for (const kind of KINDS) {
    const legacy = extra[LEGACY[kind]];
    if (!legacy) continue;
    for (const id of Object.keys(legacy)) {
      if (bag.claims[kind][id] !== undefined) continue; // v1 claim wins over a stale leftover
      const rec = legacy[id];
      const src = rec == null ? null : [String(rec[0]), Number(rec[1])];
      const rid = String(bag.seq++);
      bag.routes[rid] = { src };
      bag.claims[kind][id] = rid;
    }
    delete extra[LEGACY[kind]];
  }
}

function srcOf(graph, kind, id) {
  const bag = bagOf(graph);
  const rid = bag?.claims?.[kind]?.[String(id)];
  if (rid === undefined) return undefined;
  return bag.routes[rid]?.src ?? null;
}

/** Apparent source stamped on a hop: [host, idx], null = conflict, undefined = none. */
export function routeSrc(graph, rerouteId) {
  return srcOf(graph, "reroutes", rerouteId);
}

/** Draw source of a float whose chain holds this reroute, or undefined. */
export function floatSrc(graph, rerouteId) {
  return srcOf(graph, "floats", rerouteId);
}

/** Draw source of a gate-terminating link, or undefined. */
export function gateSrc(graph, linkId) {
  return srcOf(graph, "gatelinks", linkId);
}

/** Every gate-link claim, as [linkId, src] pairs (iteration order unspecified). */
export function gateClaims(graph) {
  const bag = bagOf(graph);
  if (!bag) return [];
  return Object.keys(bag.claims.gatelinks).map((lid) => [lid, bag.routes[bag.claims.gatelinks[lid]]?.src ?? null]);
}

const sameSrc = (a, b) =>
  a === b || (a != null && b != null && String(a[0]) === String(b[0]) && Number(a[1]) === Number(b[1]));

function mintRoute(bag, src) {
  const rid = String(bag.seq++);
  bag.routes[rid] = { src: src == null ? null : [String(src[0]), Number(src[1])] };
  return rid;
}

function dropIfUnclaimed(bag, routeId) {
  for (const kind of KINDS) {
    for (const id of Object.keys(bag.claims[kind])) if (bag.claims[kind][id] === routeId) return;
  }
  delete bag.routes[routeId];
}

/**
 * Stamp route provenance along a chain -- one shared route for the whole call
 * (one gesture = one identity). Per hop, the v0 re-stamp semantics hold
 * exactly: absent -> claim; same src -> keep; different src -> the hop
 * becomes a CONFLICT claim (src null) and inherits nothing.
 */
export function stampRouteChain(graph, rerouteIds, src) {
  const bag = bagOf(graph, true);
  let fresh = null;
  let conflict = null;
  for (const rid of rerouteIds) {
    const key = String(rid);
    const cur = bag.claims.reroutes[key];
    if (cur === undefined) {
      fresh ??= mintRoute(bag, src);
      bag.claims.reroutes[key] = fresh;
      continue;
    }
    const curSrc = bag.routes[cur]?.src ?? null;
    if (sameSrc(curSrc, src)) continue;
    conflict ??= mintRoute(bag, null);
    bag.claims.reroutes[key] = conflict;
    dropIfUnclaimed(bag, cur);
  }
}

/**
 * Stamp float draw-sources -- one route per call; ifAbsent preserves the v0
 * `??=` writes (an existing claim is the older gesture's and wins).
 */
export function stampFloats(graph, rerouteIds, src, { ifAbsent = false } = {}) {
  const bag = bagOf(graph, true);
  let route = null;
  for (const rid of rerouteIds) {
    const key = String(rid);
    if (bag.claims.floats[key] !== undefined) {
      if (ifAbsent) continue;
      const old = bag.claims.floats[key];
      route ??= mintRoute(bag, src);
      bag.claims.floats[key] = route;
      dropIfUnclaimed(bag, old);
      continue;
    }
    route ??= mintRoute(bag, src);
    bag.claims.floats[key] = route;
  }
}

/** Stamp a gate-link claim (one link, its own route; replaces an existing claim). */
export function stampGate(graph, linkId, src) {
  const bag = bagOf(graph, true);
  const key = String(linkId);
  const old = bag.claims.gatelinks[key];
  bag.claims.gatelinks[key] = mintRoute(bag, src);
  if (old !== undefined) dropIfUnclaimed(bag, old);
}

/** Remove one claim; the route dies with its last claim. */
export function unclaim(graph, kind, id) {
  const bag = bagOf(graph);
  const key = String(id);
  const rid = bag?.claims?.[kind]?.[key];
  if (rid === undefined) return false;
  delete bag.claims[kind][key];
  dropIfUnclaimed(bag, rid);
  return true;
}

/** Remove a claim only when it names this pin (drag.js chain-stamp clearing). */
export function unclaimIfSrc(graph, kind, id, hostId, hostIdx) {
  const src = srcOf(graph, kind, id);
  if (src == null) return false;
  if (String(src[0]) !== String(hostId) || Number(src[1]) !== Number(hostIdx)) return false;
  return unclaim(graph, kind, id);
}

/** Move a claim to a new carrier id (tooth GC handoff, unpack id translation). */
export function renameClaim(graph, kind, oldId, newId) {
  const bag = bagOf(graph);
  const from = String(oldId);
  const to = String(newId);
  const rid = bag?.claims?.[kind]?.[from];
  if (rid === undefined) return false;
  delete bag.claims[kind][from];
  bag.claims[kind][to] ??= rid;
  dropIfUnclaimed(bag, rid); // no-op unless the target key was taken
  return true;
}

/**
 * Prune pass: drop claims whose carrier died (reroute/link) or whose route's
 * src host died. Routes die with their last claim. Predicates come from the
 * caller so the store stays graph-model agnostic.
 */
export function gc(graph, { liveReroute, liveLink, liveNode }) {
  const bag = bagOf(graph);
  if (!bag) return false;
  let changed = false;
  const liveCarrier = { reroutes: liveReroute, floats: liveReroute, gatelinks: liveLink };
  for (const kind of KINDS) {
    const claims = bag.claims[kind];
    for (const id of Object.keys(claims)) {
      const rid = claims[id];
      const src = bag.routes[rid]?.src ?? null;
      const dead = !liveCarrier[kind]?.(id) || (src != null && !liveNode?.(src[0]));
      if (dead) {
        delete claims[id];
        dropIfUnclaimed(bag, rid);
        changed = true;
      }
    }
  }
  return changed;
}

/** Debug/harness view of the whole bag (adapting legacy on sight). */
export function dump(graph) {
  return bagOf(graph) ?? null;
}
