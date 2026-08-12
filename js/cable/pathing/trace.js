// Trace geometry helpers: emit a filleted orthogonal polyline into a Path2D, and
// arc-length math for the centre point and point-at-t (arrows, flow dots).

const RADIUS = 10

export function emit(path, pts) {
  path.moveTo(pts[0][0], pts[0][1])
  for (let k = 1; k < pts.length - 1; k++) {
    const p = pts[k - 1], c = pts[k], n = pts[k + 1]
    // Clamp to a THIRD of the adjacent segments, not half: dense bundles have
    // 12-30px segments, and half-length fillets turn the whole trace into curve --
    // it then reads as a spline instead of a routed trace.
    const r = Math.min(
      RADIUS,
      dist(p, c) / 3,
      dist(c, n) / 3
    )
    path.arcTo(c[0], c[1], n[0], n[1], Math.max(0.5, r))
  }
  const last = pts[pts.length - 1]
  path.lineTo(last[0], last[1])
}

export function measure(pts) {
  const seg = []
  let total = 0
  for (let k = 1; k < pts.length; k++) {
    const d = dist(pts[k - 1], pts[k])
    seg.push(d)
    total += d
  }
  return { seg, total }
}

// Point at fraction t of total polyline length (fillets ignored; the error is
// bounded by the fillet radius, fine for markers).
export function pointAt(pts, m, t) {
  let target = Math.max(0, Math.min(1, t)) * m.total
  for (let k = 0; k < m.seg.length; k++) {
    if (target <= m.seg[k] || k === m.seg.length - 1) {
      const f = m.seg[k] > 0 ? target / m.seg[k] : 0
      const a = pts[k], b = pts[k + 1]
      return {
        x: a[0] + (b[0] - a[0]) * f,
        y: a[1] + (b[1] - a[1]) * f,
        angle: Math.atan2(b[1] - a[1], b[0] - a[0])
      }
    }
    target -= m.seg[k]
  }
  const last = pts[pts.length - 1]
  return { x: last[0], y: last[1], angle: 0 }
}

function dist(a, b) {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
}
