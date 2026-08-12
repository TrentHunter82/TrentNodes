// Orthogonal router on a sparse Hanan grid whose lanes are the edges of inflated
// obstacle rects. Pure geometry: no DOM, no litegraph, no app.
//
// route() returns graph-coord waypoints [start, ...bends..., end] or null when the
// trivial case applies (caller then leaves the link to core's renderer).

const BEND_COST = 24
// Best-effort (R1): a blocked grid edge is not forbidden, it costs its length times this.
// The search then minimises hidden stretch instead of failing.
const BLOCKED_MULT = 8
// Clearance is advisory, not solid: a lane inside an obstacle's INFLATED rect but
// outside the real node costs only this. Without the distinction a tight-but-clear
// corridor (deliberately aligned pins with a neighbour within 16px) priced as fully
// blocked, and a wrap around the whole node beat the straight line (QA round).
// Nearly nominal on purpose: the toll only tie-breaks toward roomier lanes at equal
// geometry. Anything bigger taxes long tight corridors until a parallel lane a few px
// off the pin line wins, turning a deliberate alignment into a whole-run shimmy
// (measured: at 1.1 a 436px soft stretch still lost to a 6px detour).
const SOFT_MULT = 1.02
const EPS = 0.01

const DIR = {
  right: [1, 0],
  left: [-1, 0],
  up: [0, -1],
  down: [0, 1],
  none: [1, 0]
}

export function route({ start, end, startDir = 'right', endDir = 'left', obstacles, clearance = 16, stubStart = 24, stubEnd = 24, enforceStart = false, enforceEnd = false, vPad = 0, flatTol = 0 }) {
  // Reroute ends (core passes CENTER -> 'none') are resolved by the registry to the
  // upstream's side before this call; the mapping here is only a fallback.
  const sv = DIR[startDir === 'none' ? 'right' : startDir] ?? DIR.right
  const ev = DIR[endDir === 'none' ? 'left' : endDir] ?? DIR.left
  const sStub = [start[0] + sv[0] * stubStart, start[1] + sv[1] * stubStart]
  const eStub = [end[0] + ev[0] * stubEnd, end[1] + ev[1] * stubEnd]

  // Local obstacle set, inflated. The corridor bbox is padded so lanes exist around
  // obstacles sitting just outside the endpoints' span.
  const pad = clearance + 80
  const bb = [
    Math.min(sStub[0], eStub[0], start[0], end[0]) - pad,
    Math.min(sStub[1], eStub[1], start[1], end[1]) - pad,
    Math.max(sStub[0], eStub[0], start[0], end[0]) + pad,
    Math.max(sStub[1], eStub[1], start[1], end[1]) + pad
  ]
  const rects = []
  const hard = [] // the real node boxes: crossing these is the 8x sin
  for (const r of obstacles) {
    // vPad makes every obstacle taller FOR THE GRID ONLY (Barney's shadow rule):
    // horizontal lanes form that much further from node tops/bottoms, so wires
    // stand clear of the drop shadow instead of grazing it. Vertical lanes stay
    // put -- pin-column corridors are deliberate geometry.
    const x0 = r[0] - clearance, y0 = r[1] - clearance - vPad
    const x1 = r[0] + r[2] + clearance, y1 = r[1] + r[3] + clearance + vPad
    if (x1 < bb[0] || x0 > bb[2] || y1 < bb[1] || y0 > bb[3]) continue
    rects.push([x0, y0, x1, y1])
    hard.push([r[0] - 4, r[1] - 4, r[0] + r[2] + 4, r[1] + r[3] + 4])
  }

  // Trivial forward case: straight stub-to-stub corridor with nothing in the way.
  // ONLY for canonical exit-right/enter-left directions -- with any other faces
  // (flipped comb gates), a clear straight corridor still exists geometrically but
  // runs THROUGH the gate body against the stub direction, and the shortcut made
  // flips appear to do nothing unless the layout already forced a wrap (QA
  // find). Non-canonical dirs always take the full search, which honours the stubs.
  // flatTol (Barney's straightening rule): endpoints within the link stroke's
  // thickness of colinear fall back to the spline too -- a jog smaller than the
  // line is drawing noise, and the near-flat spline reads as a straight wire.
  if (
    sv[0] === 1 && ev[0] === -1 &&
    eStub[0] - sStub[0] > EPS && Math.abs(sStub[1] - eStub[1]) < Math.max(EPS, flatTol) &&
    clearH(sStub[1], sStub[0], eStub[0], rects) && clearH(eStub[1], sStub[0], eStub[0], rects)
  ) {
    return null
  }

  // Lanes: obstacle edges plus the two stub coordinates.
  const xs = [sStub[0], eStub[0]]
  const ys = [sStub[1], eStub[1]]
  for (const [x0, y0, x1, y1] of rects) { xs.push(x0, x1); ys.push(y0, y1) }
  const X = dedupe(xs), Y = dedupe(ys)

  const si = [idx(X, sStub[0]), idx(Y, sStub[1])]
  const ei = [idx(X, eStub[0]), idx(Y, eStub[1])]

  const path = astar(X, Y, si, ei, rects, hard, sv, ev, enforceStart, enforceEnd)
  if (!path) return null

  // Waypoints: real endpoints, stubs, grid path. Collinear runs collapsed.
  const pts = [start, sStub, ...path.map(([i, j]) => [X[i], Y[j]]), eStub, end]
  return simplify(pts)
}

function dedupe(vals) {
  const s = [...vals].sort((a, b) => a - b)
  const out = [s[0]]
  for (const v of s) if (v - out[out.length - 1] > EPS) out.push(v)
  return out
}

function idx(lanes, v) {
  for (let i = 0; i < lanes.length; i++) if (Math.abs(lanes[i] - v) <= EPS) return i
  return 0
}

// A horizontal run at y crossing [xa,xb] is blocked by a rect when y lies strictly
// inside it and the spans overlap with positive length. Strict bounds let paths ride
// exactly on inflated edges (the lanes ARE those edges).
function clearH(y, xa, xb, rects) {
  const a = Math.min(xa, xb), b = Math.max(xa, xb)
  for (const [x0, y0, x1, y1] of rects) {
    if (y > y0 + EPS && y < y1 - EPS && b > x0 + EPS && a < x1 - EPS) return false
  }
  return true
}

function clearV(x, ya, yb, rects) {
  const a = Math.min(ya, yb), b = Math.max(ya, yb)
  for (const [x0, y0, x1, y1] of rects) {
    if (x > x0 + EPS && x < x1 - EPS && b > y0 + EPS && a < y1 - EPS) return false
  }
  return true
}

function astar(X, Y, si, ei, rects, hard, sv, ev, enforceStart, enforceEnd) {
  const W = X.length, H = Y.length
  const id = (i, j) => j * W + i
  // State carries entry axis (0 h, 1 v, -1 start) for the bend penalty.
  const g = new Map(), from = new Map()
  const open = [[0, si[0], si[1], -1]]
  const hCost = (i, j) => Math.abs(X[i] - X[ei[0]]) + Math.abs(Y[j] - Y[ei[1]])
  g.set(id(si[0], si[1]) * 4 + 0, 0) // axis -1 -> offset (axis+1) = 0, matching the pop

  // Problem-size cap: a pathological grid (hundreds of nodes = dense Hanan lines)
  // must not freeze the main thread on every obstacle change. Bailing returns null
  // and the caller falls back to the plain spline for that link -- degraded, alive.
  let expansions = 0
  while (open.length) {
    if (++expansions > 20000) return null
    let bi = 0
    for (let k = 1; k < open.length; k++) if (open[k][0] < open[bi][0]) bi = k
    const [, i, j, axis] = open.splice(bi, 1)[0]
    const skey = id(i, j) * 4 + (axis + 1)
    const base = g.get(skey)
    if (i === ei[0] && j === ei[1]) {
      const pts = [[i, j]]
      let cur = skey
      while (from.has(cur)) { cur = from.get(cur); pts.push([Math.floor(cur / 4) % W, Math.floor(cur / 4 / W)]) }
      return pts.reverse()
    }
    const steps = [
      [i - 1, j, 0], [i + 1, j, 0],
      [i, j - 1, 1], [i, j + 1, 1]
    ]
    for (const [ni, nj, nax] of steps) {
      if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue
      // ENFORCED stubs (comb gate faces) are binding, not cosmetic: retracing one
      // is free on this grid (same-lane travel, no bend), so A* folded straight
      // back over any stub the layout found inconvenient and simplify() collapsed
      // the fold -- flipped gates looked like no-ops (QA find). Ban the one
      // step that reverses out of the start stub / enters the end stub from the
      // endpoint's side. OPT-IN per end: for node pins the fold is load-bearing
      // -- tight side-by-side pairs route THROUGH the collapsed fold, and a
      // global ban turned their doglegs into colliding shimmies (measured:
      // adjacent 0 -> 17 crossings).
      if (enforceStart || enforceEnd) {
        const dx = Math.sign(X[ni] - X[i]), dy = Math.sign(Y[nj] - Y[j])
        if (enforceStart && i === si[0] && j === si[1] && dx === -sv[0] && dy === -sv[1]) continue
        if (enforceEnd && ni === ei[0] && nj === ei[1] && dx === ev[0] && dy === ev[1]) continue
      }
      const len = nax === 0 ? Math.abs(X[ni] - X[i]) : Math.abs(Y[nj] - Y[j])
      const free = nax === 0 ? clearH(Y[j], X[i], X[ni], rects) : clearV(X[i], Y[j], Y[nj], rects)
      let mult = 1
      if (!free) {
        const overNode = nax === 0 ? !clearH(Y[j], X[i], X[ni], hard) : !clearV(X[i], Y[j], Y[nj], hard)
        mult = overNode ? BLOCKED_MULT : SOFT_MULT
      }
      // Vertical travel carries an epsilon surcharge so equal-cost bend placements
      // resolve identically for every link (horizontal-first). Without it, sibling
      // links tie-break by expansion order into different shapes and braid. The
      // distance-from-target term additionally orders the two L-shapes of a plain
      // dogleg (identical H and V totals): descending AT the allocated stub lane
      // always prices under descending early at the source -- without it the choice
      // fell to expansion order and single links ignored their entry lane.
      let cost = len * mult * (nax === 1 ? 1.0005 + Math.abs(X[i] - X[ei[0]]) * 1e-6 : 1)
      if (axis !== -1 && axis !== nax) cost += BEND_COST
      const nkey = id(ni, nj) * 4 + (nax + 1)
      const ng = base + cost
      if (ng < (g.get(nkey) ?? Infinity)) {
        g.set(nkey, ng)
        from.set(nkey, skey)
        open.push([ng + hCost(ni, nj), ni, nj, nax])
      }
    }
  }
  return null
}

// Reuse a sibling's routed shape for a link between the same two nodes, translated to
// this link's own slots. Sibling stub tips differ only in y (same nodes, stacked
// slots), so the translation is pure-y plus one orthogonal tail alignment. Identical
// shapes cannot cross each other -- this is what kills same-pair homotopy braids.
// Returns null (caller falls back to A*) when the template has no mid section, the
// x-geometry does not match, or the translated route fails obstacle validation.
// Ribbon-cable bundle construction. A bundle of links between the same two nodes must
// share one homotopy AND nest correctly around wrapped obstacles -- uniform y-shifts
// copy the source-side stacking onto every run, which inverts the required nesting on
// the far side of a wrap (measured: 6 inherent-looking crossings that a perpendicular
// offset eliminates entirely). Each strand is the template offset along the path's
// LEFT NORMAL by its rank offset; orthogonal corner joins come free (a vertex takes x
// from its vertical neighbour, y from its horizontal one). The first and last segments
// re-anchor on the member's own pins.
//
// offsetStrand returns one member's polyline, or null when the template is degenerate
// or the pin-attachment axes do not match.
export function offsetStrand(tpl, start, end, o) {
  const n = tpl.length
  if (n < 4) return null
  // Per-segment free coordinate: vertical segments carry x, horizontal carry y.
  const segs = []
  for (let k = 0; k < n - 1; k++) {
    const vert = Math.abs(tpl[k][0] - tpl[k + 1][0]) < 0.1
    const horiz = Math.abs(tpl[k][1] - tpl[k + 1][1]) < 0.1
    if (!vert && !horiz) return null
    // Left normal of travel: (dy, -dx). Offsetting a vertical shifts x, a horizontal
    // shifts y, signed by travel direction.
    const dx = Math.sign(tpl[k + 1][0] - tpl[k][0])
    const dy = Math.sign(tpl[k + 1][1] - tpl[k][1])
    segs.push({ vert, coord: vert ? tpl[k][0] : tpl[k][1], nx: dy, ny: -dx })
  }
  // Pin re-anchor on the attachment segments; perpendicular offset on the mid ones.
  for (let k = 0; k < segs.length; k++) {
    const s = segs[k]
    if (k === 0) s.coord = s.vert ? start[0] : start[1]
    else if (k === segs.length - 1) s.coord = s.vert ? end[0] : end[1]
    else s.coord += o * (s.vert ? s.nx : s.ny)
  }
  // Consecutive segments must alternate orientation for corner joins to be defined.
  const own = [[start[0], start[1]]]
  for (let k = 1; k < segs.length; k++) {
    const a = segs[k - 1], b = segs[k]
    if (a.vert === b.vert) return null
    own.push(a.vert ? [a.coord, b.coord] : [b.coord, a.coord])
  }
  own.push([end[0], end[1]])
  const pts = simplify(own)
  return pts.length >= 3 ? pts : null
}

// Check a route's mid segments (between the stub tips) against obstacle interiors.
function simplify(pts) {
  const out = [pts[0]]
  for (let k = 1; k < pts.length - 1; k++) {
    const [a, b, c] = [out[out.length - 1], pts[k], pts[k + 1]]
    const abx = Math.abs(a[0] - b[0]) < EPS, aby = Math.abs(a[1] - b[1]) < EPS
    const bcx = Math.abs(b[0] - c[0]) < EPS, bcy = Math.abs(b[1] - c[1]) < EPS
    if ((abx && bcx) || (aby && bcy)) continue
    if (Math.abs(a[0] - b[0]) < EPS && Math.abs(a[1] - b[1]) < EPS) continue
    out.push(b)
  }
  out.push(pts[pts.length - 1])
  return out
}
