// The link display ledger: hide the machinery, show the intent.
//
// Cable Management's graph edits are deliberately ordinary -- that is the whole point of resolving to a
// real origin -- but "ordinary" means links render from their TRUE endpoints, which exposes
// exactly what we went to trouble to make invisible:
//
//   Case A  A feeds B; user drags B' to C. Graph holds A->B and A->C.
//           Wanted: A->B unchanged, and A->C drawn as if it started at B'.
//
//   Case B  User drags A' (a widget pin) to B. Hidden primitive H drives both.
//           Graph holds H->A and H->B. Wanted: one stroke from A' to B; H invisible.
//
// Both reduce to one operation per real link: SUPPRESS, or re-anchor its start point.
//
// The choke point is LGraphCanvas.prototype._renderAllLinkSegments -- all five call sites
// funnel through it, and both endpoints arrive as plain parameters. Patching lower (renderLink
// / renderLinkDirect) would leave the viewport cull computed on the ORIGINAL endpoints, and
// would leave `renderedPaths.add(link)` running in the caller -- yielding an invisible but
// still hit-testable phantom at stale coordinates.
//
// Patch the PROTOTYPE, never the instance: setGraph() replaces this.linkRenderer per graph on
// workflow load and on subgraph enter/exit.

export const SUPPRESS = Symbol("cablemanagement.suppress");

const entries = new Map(); // linkId -> SUPPRESS | { startPos, endPos, startDir }
let dirty = true;
let lastLinkCount = -1;
let lastLinkId = -1;
let installed = false;
let buildFn = null;
let lastError = null;
let applied = 0;
let passActive = true;

/**
 * The prototype patches are permanent once installed; this is the honest off switch.
 * Inactive, the pass forwards untouched endpoints and never SUPPRESSes -- with the
 * master toggle off, a stitch's inbound gate wire must be visible like everything else.
 */
export function setActive(v) {
  passActive = !!v;
  if (!passActive) entries.clear();
  dirty = true;
}

/**
 * Debug view: draw the TRUTH under the illusion. Every link the ledger alters
 * (re-anchored or suppressed) also paints its REAL geometry -- the endpoints
 * core handed us, before substitution -- as a faint dashed spline, so "what
 * actually feeds what" is one glance instead of console archaeology. The sync
 * pass ghosts hidden machinery nodes to match (see index.js).
 */
let debugView = false;
export function setDebugView(v) {
  debugView = !!v;
  dirty = true;
}
export function debugViewActive() {
  return debugView;
}

/** Mark the ledger stale. Cheap; the rebuild is deferred to the next repaint. */
export function invalidate() {
  dirty = true;
}

export function get(linkId) {
  return entries.get(linkId);
}

export function size() {
  return entries.size;
}

/** Debug surface -- read from the console or a test harness. */
export function debug() {
  return {
    installed,
    size: entries.size,
    applied,
    lastError,
    keys: [...entries.keys()],
    values: [...entries.entries()].map(([k, v]) => [k, v === SUPPRESS ? "SUPPRESS" : v]),
  };
}

/**
 * Graph-space anchor for one of our pins.
 *
 * The Vue node root is translated in GRAPH units and is position:absolute, so it is the
 * offsetParent of the pins inside it. offsetLeft/offsetTop are therefore already unscaled
 * graph units -- no getBoundingClientRect, no dividing by canvas scale, nothing that drifts
 * mid-zoom. The title offset is because the node's transform subtracts it.
 */
export function anchorOf(pinEl, liteNode) {
  const TITLE = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
  if (!pinEl || !liteNode) return null;
  return [
    liteNode.pos[0] + pinEl.offsetLeft + pinEl.offsetWidth * 0.5,
    liteNode.pos[1] - TITLE + pinEl.offsetTop + pinEl.offsetHeight * 0.5,
  ];
}

/** Replace the ledger wholesale. Called by the rebuild the extension supplies. */
export function replace(next) {
  entries.clear();
  for (const [k, v] of next) entries.set(k, v);
}

/**
 * @param app        ComfyUI app
 * @param rebuild    (graph) => Map<linkId, SUPPRESS|{startPos,endPos,startDir}>
 */
export function install(app, rebuild) {
  if (installed) return true;
  const proto = app?.canvas?.constructor?.prototype;
  const orig = proto?._renderAllLinkSegments;
  // On 1.27.10 and earlier this is a true ES private (#renderAllLinkSegments) and cannot be
  // reached. Degrade silently rather than breaking the canvas.
  if (typeof orig !== "function") return false;

  buildFn = rebuild;
  const drawConnections = proto.drawConnections;

  proto.drawConnections = function (ctx) {
    const g = this.graph;
    if (g && passActive) {
      // O(1) backstop so a programmatic connect() by third-party code -- which fires no event
      // we subscribe to -- still refreshes, without scanning every link every frame.
      const count = g._links?.size ?? -1;
      const last = g.state?.lastLinkId ?? -1;
      if (dirty || count !== lastLinkCount || last !== lastLinkId) {
        try {
          replace(buildFn(g) ?? new Map());
          lastError = null;
        } catch (err) {
          entries.clear(); // never let a ledger bug take the canvas down
          lastError = String(err);
        }
        lastLinkCount = count;
        lastLinkId = last;
        dirty = false;
      }
    }
    return drawConnections.call(this, ctx);
  };

  // Arity is 9. `disabled` must be forwarded explicitly or floating links change style.
  proto._renderAllLinkSegments = function (
    ctx, link, startPos, endPos, visibleReroutes, now, startDir, endDir, disabled
  ) {
    // Identity decides the key space: core's FLOATING links have their own id
    // counter and render through this same path, so a float whose id collides
    // with a pin-recorded real link inherited that link's substitution and
    // drew from a stranger's pin (Barney's floating_input workflow: float 68
    // vs pin link 68 -- 'a link between a random node and the ribbon').
    // Real links look up bare ids; floats look up 'f'-prefixed entries (the
    // floating-ribbon pin illusion lives there, built by the same convention).
    const isReal = link?.id != null && this.graph?.getLink?.(link.id) === link;
    const e = passActive && link?.id != null ? entries.get(isReal ? link.id : "f" + link.id) : undefined;
    // `_dragging` alone is NOT "being dragged": core stamps it on every
    // output-attached floating link as a permanent style flag, each frame
    // (renderFloatingLinks). Nor is a live connector enough -- during ANY drag
    // isConnecting is true globally, and requiring only that snapped every
    // on-screen float back to its true origin for the drag's duration (QA find).
    // A genuinely dragged link is a MEMBER of the connector's moved-link sets.
    const lc = app?.canvas?.linkConnector;
    const member = !!link?._dragging && !!lc?.isConnecting &&
      (!!lc.inputLinks?.includes(link) || !!lc.outputLinks?.includes(link) ||
        !!lc.floatingLinks?.includes(link));
    // Even then, only the LOOSE end follows the pointer -- the other end keeps
    // its substitution. Dropping the whole entry drew a moved link's chain from
    // the hidden primitive while its input end was in hand (QA find: the ribbon
    // entry segment "connected to true origin while dragging").
    const looseStart = member && lc?.state?.connectingTo === "output";
    const looseEnd = member && lc?.state?.connectingTo === "input";
    // Debug view: the true wire, faint and dashed, from the ORIGINAL endpoints
    // -- drawn under the substituted stroke (or under nothing, for SUPPRESS).
    if (debugView && e !== undefined) {
      try {
        ctx.save();
        ctx.globalAlpha = 0.28;
        ctx.setLineDash([6, 5]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#7ec8ff";
        ctx.beginPath();
        ctx.moveTo(startPos[0], startPos[1]);
        const bend = Math.max(40, Math.abs(endPos[0] - startPos[0]) * 0.4);
        ctx.bezierCurveTo(
          startPos[0] + bend, startPos[1],
          endPos[0] - bend, endPos[1],
          endPos[0], endPos[1]
        );
        ctx.stroke();
        ctx.restore();
        window.__cablemanagementDebugDraws = (window.__cablemanagementDebugDraws ?? 0) + 1;
      } catch {
        /* a debug overlay must never take the canvas down */
      }
    }
    // Anchors are resolved per frame, not cached at rebuild time. A cached point freezes the
    // moment the ORIGIN node moves: the ledger only rebuilds on link mutation, and dragging a
    // node changes no link, so the re-anchored end would stay put while the far end tracks.
    let start = startPos;
    let end = endPos;
    let sDir = startDir;
    // A dragged machinery link un-suppresses wholesale: the user is holding it.
    const active = e !== undefined && !(member && e === SUPPRESS);
    if (active) applied++;
    if (active && e !== SUPPRESS) {
      if (!looseStart) {
        start = (e.resolve ? e.resolve() : e.startPos) ?? startPos;
        sDir = e.startDir ?? startDir;
      }
      if (!looseEnd) end = e.endPos ?? endPos;
    }
    // Opt-in draw capture for the test harness: the EFFECTIVE endpoints -- after any ledger
    // substitution -- are the one honest record of where links are actually drawn (layout
    // store internals are unreadable, and getConnectionPos lies in Vue mode).
    if (window.__cablemanagementCapture && link && !(active && e === SUPPRESS)) {
      // Floats key with an 'f' prefix: their id space is separate from real
      // links, and a bare-id collision let a float overwrite a real link's
      // captured endpoints (same conflation the substitution guard above ends).
      (window.__cablemanagementDraw ??= new Map()).set(isReal ? link.id : "f" + link.id, [start[0], start[1], end[0], end[1]]);
    }
    // Returning early means renderedPaths.add(link) never runs, so a suppressed link also
    // leaves hit-testing, the midpoint dot and the tooltip for this frame -- which is what we
    // want: it should be as if the link were not there.
    if (active && e === SUPPRESS) return;
    // The original recomputes link_bounding from whatever points we pass, so the viewport cull
    // tracks the DRAWN geometry, and the interaction path is rebuilt from it too -- hit-testing
    // follows the re-anchored stroke for free.
    return orig.call(this, ctx, link, start, end, visibleReroutes, now, sDir, endDir, disabled);
  };

  installed = true;
  return true;
}

/** Subscribe to the events that actually mean "links moved". */
export function watch(app) {
  const g = app?.canvas?.graph ?? app?.graph;
  // Only widget-backed inputs emit node:slot-links:changed, so this is a hint, not a guarantee
  // -- the O(1) backstop in drawConnections covers the rest.
  for (const ev of ["node:slot-links:changed", "node:before-removed", "configured"]) {
    g?.events?.addEventListener?.(ev, invalidate);
  }
  const lc = app?.canvas?.linkConnector?.events;
  for (const ev of ["link-created", "input-moved", "output-moved", "after-drop-links", "reset"]) {
    lc?.addEventListener?.(ev, invalidate);
  }
}
