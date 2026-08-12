// Comb gesture layer (PATHING.md section 9). Verbs:
//   reroute dropped on a reroute        -> comb is born (two lanes, gates adjacent)
//   reroute dropped on a gate or tooth  -> enroll as the last lane
//   in-tooth pulled away                -> lane detaches, the dot stays free --
//                                          UNLESS the lane dangles at the source
//                                          (floating tip): then the pull is an
//                                          output-seeking link drag that resumes
//                                          the float, mirroring the out-pull
//   out-tooth pulled                    -> NEW LINK drag through the lane (output
//                                          semantics: branch a complete lane,
//                                          resume a dangling one; never moves)
//   gate body dragged                   -> gate moves (teeth follow via combPass)
//   group title dragged                 -> gates inside the group rect ride the
//                                          drag (core already carries the teeth)
//   hover glyph clicked                 -> gate flips horizontally
// Core does all reroute dragging; we only observe presses and interpret drops --
// except the out-pull, which we start (dragFromReroute) and finish (dropLinks +
// reset) ourselves, because core never saw the swallowed pointerdown and its own
// pointer state machine will not fire the drop. Gate-body, glyph, and out-tooth
// presses are swallowed via DOCUMENT capture -- core's own listeners sit on the
// canvas element, and same-target capture runs in registration order, so an
// element-level listener could never preempt them.
import {
  clearGateSelection, combAt, combRecords, detachLane, dissolveComb,
  gestureCreate, gestureEnroll, gestureJoin, isGateSelected, selectGate,
  selectedGates, setDotDrag, setHover, toothOf
} from './combs.js'
// Every graph resolution here must be the graph ON SCREEN -- the root graph is not it
// inside a subgraph, which made all comb gestures dead there and let presses hit-test
// invisible root gates from subgraph coordinates.
import { activeGraph } from '../graph.js'
import { openSortModal } from './sortmodal.js'
import { handleGateInDrop, dimCoreSlots, undimCoreSlots } from '../drops.js'
import { forgetProvenance, severSource } from '../reanchor.js'
import { invalidate } from '../ledger.js'
import { routeSrc, stampFloats, unclaim } from '../routestore.js'

export function installGestures(app, active) {
  let gateDrag = null // {press: [x,y], gates: [{comb, which, origin}]}
  let press = null // reroute press being core-dragged: {rid, x, y}
  let pullDrag = false // out-tooth pull: we own this link drag end to end
  // Link drags born on the CANVAS element (reroute slot pulls, our tooth pulls)
  // have no Vue drag session -- the composable that snaps the preview to a
  // compatible slot and keeps it moving over node DOM only serves drags started
  // on slot DOM. For canvas-born connector drags we drive snapLinksPos ourselves
  // (QA find: float pulls didn't snap; the comb pull's preview froze the moment
  // the pointer entered a node, because graph_mouse only updates over the canvas).
  let canvasPress = false
  // Out-pin shift gesture, deferred until the pointer proves it is a drag:
  // {lane, riders, start: [clientX, clientY]}. A release before the threshold
  // is a click, and core's shift+click on an output touches nothing.
  let shiftPull = null
  // Where the current pull began (client px). A release within the click
  // threshold resets silently -- core's slot click is a pure no-op, and the
  // old fall-through opened the release menu and ate the next press.
  let pullStart = null

  // The wires leaving a lane's out-pin: real links riding the out-tooth.
  const outToothRiders = (g, lane) => {
    const outR = g.reroutes?.get?.(lane?.out)
    const riders = []
    for (const lid of outR?.linkIds ?? []) {
      const link = g.getLink ? g.getLink(lid) : g._links?.get?.(lid)
      const node = link && (g.getNodeById(link.target_id) ?? g.getNodeById(Number(link.target_id)))
      if (node?.inputs?.[link.target_slot]) riders.push({ node, slot: link.target_slot, linkId: lid })
    }
    return riders
  }

  // A consumer-side cut parks the lane's origin float with its PHYSICAL
  // origin -- the true source, not the pin the wire was drawn from. With a
  // passthrough source those differ, and the parked wire visibly jumped to
  // the upstream node (Barney's regression: "A's source->ribbon(floating)").
  // Same rule as routes' own sever: the float inherits the lane's apparent
  // source into the floatfrom stamp, so it keeps drawing from the pin. A
  // provenance-less lane has nothing to inherit and honestly draws from its
  // real origin.
  const inheritFloatFrom = (g, lane) => {
    const rec = routeSrc(g, lane.out) ?? routeSrc(g, lane.in)
    if (!rec) return
    stampFloats(g, [lane.in, lane.out], rec, { ifAbsent: true })
  }

  // Ctrl+alt on an IN pin severs the lane's SOURCE (core's input contract:
  // disconnect at the press, then a fresh source-seeking drag). The riders
  // stay parked on the lane as consumer-side floats -- the 2f "waiting for a
  // source" shape -- added BEFORE their real links go: the floats are what
  // hold the teeth through the removal. A source cut ends the lane's
  // apparent-source illusion (stamps go), and the riders' records go with
  // their links; a new source dropped on the in-pin writes fresh ones.
  const cutLaneSource = (g, lane, riders) => {
    // severSource = core's own LLink.disconnect(graph, "input"): each rider's
    // claim survives as a floating link through the lane, with the floating
    // marker that makes the parked wire render (hand-built floats drew
    // nothing -- Barney's "all links disappear but the pins remain").
    for (const w of riders) {
      forgetProvenance(w.node, w.slot)
      severSource(g, w.node, w.slot, w.linkId)
    }
    for (const kind of ['floats', 'reroutes']) {
      unclaim(g, kind, lane.in)
      unclaim(g, kind, lane.out)
    }
    invalidate()
  }
  // Selected gates follow node-group drags (marquee semantics): a press on a
  // selected NODE arms this; deltas are read off that node's own pos, which works
  // for both the legacy canvas drag and the Vue node drag.
  let follow = null // {refNode, refPos, gates: [{comb, which, origin}]}

  // LGraphGroups among core's selected items (duck-typed: only groups carry
  // recomputeInsideNodes). A drag that moves a group moves everything the group
  // captured -- teeth included, they are ordinary reroutes -- so the gates the
  // group holds must ride the same drag, or combPass snaps the teeth straight
  // back onto the stranded gates (measured: ribbons refused to move with their
  // group).
  const selectedGroups = () =>
    [...(app.canvas?.selectedItems ?? [])].filter(
      (it) => typeof it?.recomputeInsideNodes === 'function'
    )

  // Arm the follow for a drag core is about to run: refItem's pos carries the
  // per-frame delta (nodes and groups both expose pos). Riders: the selected
  // gates (marquee semantics, as before), plus the gates every dragged GROUP
  // captures -- gate anchor point-in-rect, the same rule core's
  // recomputeInsideNodes applies to reroutes. Callers pass groups: [] on
  // ctrl/meta drags, mirroring core's getDraggedItems (a group frame dragged
  // WITHOUT its children must leave its gates behind too).
  const armFollow = (g, refItem, groups, withSelected) => {
    const gates = []
    const seen = new Set()
    const ride = (comb, which) => {
      const k = `${comb.id}|${which}`
      if (seen.has(k)) return
      seen.add(k)
      gates.push({ comb, which, origin: [...comb[which].pos] })
    }
    if (withSelected) for (const s of selectedGates(g)) ride(s.comb, s.which)
    for (const grp of groups) {
      const b = grp.boundingRect ?? [grp.pos[0], grp.pos[1], grp.size[0], grp.size[1]]
      for (const comb of combRecords(g)) {
        for (const which of ['in', 'out']) {
          const p = comb[which].pos
          if (p[0] >= b[0] && p[0] <= b[0] + b[2] && p[1] >= b[1] && p[1] <= b[1] + b[3]) {
            ride(comb, which)
          }
        }
      }
    }
    follow = gates.length
      ? { refNode: refItem, refPos: [refItem.pos[0], refItem.pos[1]], gates }
      : null
  }
  let cursorSet = false // we own the canvas cursor only while over a glyph

  const graphPt = (e) => {
    const c = app.canvas
    if (c?.convertEventToCanvasOffset) {
      const p = c.convertEventToCanvasOffset(e)
      return [p[0], p[1]]
    }
    const r = c.canvas.getBoundingClientRect()
    return [
      (e.clientX - r.left) / c.ds.scale - c.ds.offset[0],
      (e.clientY - r.top) / c.ds.scale - c.ds.offset[1]
    ]
  }

  const rerouteNear = (g, x, y, not) => {
    let best = null, bd = 12
    for (const r of g.reroutes?.values?.() ?? []) {
      if (r === not) continue
      const d = Math.hypot(r.pos[0] - x, r.pos[1] - y)
      if (d < bd) { bd = d; best = r }
    }
    return best
  }

  const elCentre = (el) => {
    const r = el.getBoundingClientRect()
    const c = app.canvas, cr = c.canvas.getBoundingClientRect()
    return [
      (r.x + r.width / 2 - cr.left) / c.ds.scale - c.ds.offset[0],
      (r.y + r.height / 2 - cr.top) / c.ds.scale - c.ds.offset[1]
    ]
  }

  // Snap the in-flight preview like the Vue session would: pin hover snaps to the
  // pin when compatible, node hover snaps to the first compatible slot (free ones
  // preferred), otherwise the preview simply follows the pointer -- which is also
  // what keeps it moving over node DOM. elementsFromPoint, not e.target: core
  // captures the pointer on the canvas element, so targets lie during its drags.
  const driveSnap = (e) => {
    const lc = app.canvas?.linkConnector
    if (!lc?.isConnecting || !lc.renderLinks?.length) return
    const g = activeGraph(app)
    const to = lc.state?.connectingTo
    const want = to === 'input' ? 'in' : 'out'
    const fits = (node, slot) => lc.renderLinks.some((rl) =>
      to === 'input' ? rl.canConnectToInput?.(node, slot) : rl.canConnectToOutput?.(node, slot)
    )
    let slotEl = null, nodeEl = null
    for (const el of document.elementsFromPoint(e.clientX, e.clientY)) {
      slotEl ??= el.closest?.('[data-slot-key]')
      nodeEl ??= el.closest?.('[data-node-id]')
      if (nodeEl) break
    }
    let snap = null
    const m = slotEl && /^(.+)-(in|out)-(\d+)$/.exec(slotEl.getAttribute('data-slot-key') ?? '')
    if (m && m[2] === want) {
      const node = g?.getNodeById(m[1]) ?? g?.getNodeById(Number(m[1]))
      const slot = node && (to === 'input' ? node.inputs : node.outputs)?.[+m[3]]
      if (slot && fits(node, slot)) snap = elCentre(slotEl)
    }
    if (!snap && nodeEl) {
      const nid = nodeEl.getAttribute('data-node-id')
      const node = g?.getNodeById(nid) ?? g?.getNodeById(Number(nid))
      const slots = node && (to === 'input' ? node.inputs : node.outputs)
      let pick = -1
      for (let j = 0; slots && j < slots.length; j++) {
        if (!fits(node, slots[j])) continue
        if (pick < 0) pick = j
        const free = to === 'input' ? slots[j].link == null : !slots[j].links?.length
        if (free) { pick = j; break }
      }
      if (pick >= 0) {
        const el = document.querySelector(`[data-slot-key="${node.id}-${want}-${pick}"]`)
        if (el) snap = elCentre(el)
        else {
          const p = node.getConnectionPos(to === 'input', pick)
          if (p) snap = [p[0], p[1]]
        }
      }
    }
    const pt = snap ?? graphPt(e)
    const cur = lc.state.snapLinksPos
    if (!cur || cur[0] !== pt[0] || cur[1] !== pt[1]) {
      lc.state.snapLinksPos = [pt[0], pt[1]]
      g?.setDirtyCanvas(true, true)
    }
  }

  document.addEventListener(
    'pointerdown',
    (e) => {
      if (!active() || e.button !== 0) return
      const g = activeGraph(app)
      const canvas = app.canvas
      canvasPress = e.target === canvas?.canvas

      // Presses off the canvas element (node DOM, widgets): keep gate selection
      // only when pressing an already-selected node -- that press starts a group
      // drag the gates must follow. Anything else clears (node semantics).
      if (e.target !== canvas?.canvas) {
        const nodeEl = e.target instanceof Element ? e.target.closest('[data-node-id]') : null
        const nid = nodeEl?.getAttribute('data-node-id')
        const refNode = nid != null
          ? [...(canvas?.selectedItems ?? [])].find(
              (it) => it?.pos && it.size && String(it.id) === String(nid)
            )
          : null
        if (refNode) {
          armFollow(g, refNode, e.ctrlKey || e.metaKey ? [] : selectedGroups(), true)
        } else if (!e.shiftKey) {
          clearGateSelection()
          g?.setDirtyCanvas(true, true)
        }
        return
      }

      const [x, y] = graphPt(e)
      const hit = combAt(g, x, y)
      if (hit?.zone === 'flip') {
        hit.comb[hit.which].pins = hit.comb[hit.which].pins === 'left' ? 'right' : 'left'
        g.setDirtyCanvas(true, true)
        e.stopPropagation(); e.preventDefault()
        return
      }
      if (hit?.zone === 'sort') {
        openSortModal(g, hit.comb)
        e.stopPropagation(); e.preventDefault()
        return
      }
      if (hit?.zone === 'collapse' || hit?.zone === 'expand') {
        // Per-ribbon collapse (node-collapse semantics): one bit on the record,
        // both gates and the lane geometry follow on the next combPass.
        if (hit.comb.collapsed) delete hit.comb.collapsed
        else hit.comb.collapsed = true
        g.setDirtyCanvas(true, true)
        e.stopPropagation(); e.preventDefault()
        return
      }
      if (hit?.zone === 'labels') {
        // Labels toggle (#4): per gate, persisted on the record; the bulge and
        // the label list themselves are combPass transients.
        const gate = hit.comb[hit.which]
        if (gate.labels) delete gate.labels
        else gate.labels = true
        g.setDirtyCanvas(true, true)
        e.stopPropagation(); e.preventDefault()
        return
      }
      if (hit?.zone === 'body') {
        // Node semantics: pressing an unselected gate selects it exclusively
        // (clearing core's node selection too); shift adds; pressing a selected
        // gate keeps the whole selection. The drag then moves EVERY selected gate.
        if (e.shiftKey) selectGate(g, hit.comb, hit.which, true)
        else if (!isGateSelected(hit.comb, hit.which)) {
          selectGate(g, hit.comb, hit.which, false)
          app.canvas.deselectAll?.()
        } else {
          // Already selected: keep the selection, still bump z (node semantics --
          // every click on a node brings it to front).
          selectGate(g, hit.comb, hit.which, true)
        }
        // Node semantics both ways: the gate drags with the selection, so the
        // selection also drags with the gate (core-selected nodes ride along).
        gateDrag = {
          press: [x, y],
          gates: selectedGates(g).map((s) => ({ ...s, origin: [...s.comb[s.which].pos] })),
          items: [...(app.canvas.selectedItems ?? [])]
            .filter((it) => it?.pos && (it.size || it.linkIds))
            .map((it) => ({ it, origin: [it.pos[0], it.pos[1]] }))
        }
        g.setDirtyCanvas(true, true)
        e.stopPropagation(); e.preventDefault()
        // preventDefault also suppressed the browser's focus change -- without
        // this, keydown targets stay outside graph-canvas-container and BOTH
        // delete paths (core's binding and ours) bail on their scope check.
        canvas.canvas.focus?.()
        return
      }
      // Legacy (non-Vue) nodes are drawn ON the canvas element, so a node press
      // arrives here instead of the node-DOM branch above. Same semantics: a press
      // on a selected node arms the follow (gates ride the group drag core is
      // about to run); an unselected node clears gate selection. Core owns the
      // drag either way. Vue sessions never reach this: node presses target node
      // DOM -- except over our visibility-hidden primitives, whose graph rects
      // getNodeOnPos would still hit, so the probe is gated to legacy sessions.
      const nodeHit = document.querySelector('.lg-node') ? null : g?.getNodeOnPos?.(x, y)
      if (nodeHit) {
        const refNode = [...(canvas?.selectedItems ?? [])].find((it) => it === nodeHit)
        if (refNode) {
          armFollow(g, refNode, e.ctrlKey || e.metaKey ? [] : selectedGroups(), true)
        } else if (!e.shiftKey) {
          clearGateSelection()
          g?.setDirtyCanvas(true, true)
        }
        return
      }

      const r = rerouteNear(g, x, y)

      // GROUP-TITLE press (guarded on !r: a press that also lands on a tooth is
      // a tooth gesture, and an armed follow would swallow its pointerup drop
      // resolution below). Core is about to drag the group and everything
      // recomputeInsideNodes captured; the gates live in graph.extra and know
      // nothing of groups, so the ribbon stayed behind while its teeth were
      // dragged and snapped back. Arm the same follow node drags use, with the
      // GROUP as the reference positionable -- only its pos is read, and a
      // resize-corner press never changes pos (zero delta, harmless).
      if (!r) {
        const grp = [...(g?.groups ?? [])].find((gr) => gr.isPointInTitlebar?.(x, y))
        if (grp) {
          // Core keeps the selection when the pressed group is already in it
          // (shift adds it), and the whole selection rides the drag; an
          // unselected plain press replaces the selection at drag start, so
          // the gate selection clears with it (node semantics).
          const rides = app.canvas?.selectedItems?.has?.(grp) || e.shiftKey
          if (!rides) clearGateSelection()
          const dragged = rides ? [grp, ...selectedGroups()] : [grp]
          armFollow(g, grp, e.ctrlKey || e.metaKey ? [] : dragged, rides)
          g?.setDirtyCanvas(true, true)
          return
        }
      }

      // Empty canvas / pins / free reroutes: clear gate selection (a marquee will
      // re-select through the teeth proxies).
      if (!e.shiftKey) clearGateSelection()

      if (r) {
        const t = toothOf(r.id)
        if (t?.side === 'out') {
          const lc = app.canvas.linkConnector
          if (lc && !lc.isConnecting) {
            // Modifier gestures, output-pin contract (2g round 3): the
            // out-pin's wires are the lane's riders. Ctrl+alt cuts exactly
            // those at the press -- consumer side only, so the lane parks as
            // a float and its apparent-source stamps survive. Shift lifts
            // them as core's moving bundle (a veto excludes every other link
            // of the true origin, unhiding what core hid before asking); the
            // release rides the normal pullDrag finish. No riders: both fall
            // through to the plain branch pull, core parity.
            const lane = t.comb?.lanes?.[t.lane]
            const riders = ((e.shiftKey || ((e.ctrlKey || e.metaKey) && e.altKey)) && lane)
              ? outToothRiders(g, lane) : []
            if (riders.length && (e.ctrlKey || e.metaKey) && e.altKey && !e.shiftKey) {
              for (const w of riders) {
                forgetProvenance(w.node, w.slot)
                w.node.disconnectInput(w.slot, true)
              }
              inheritFloatFrom(g, lane)
              invalidate()
              g.setDirtyCanvas(true, true)
              e.stopPropagation(); e.preventDefault()
              return
            }
            if (riders.length && e.shiftKey && !((e.ctrlKey || e.metaKey) && e.altKey)) {
              // Shift = cut at the pin + carry the wires by their consumer
              // ends -- but DEFERRED: nothing mutates until the pointer
              // travels past the click threshold, because core's shift+CLICK
              // on an output is a pure no-op and cutting at the press would
              // turn a stray click into silent wire loss (audit 2g click
              // deviation, same guard). The arm itself happens in
              // pointermove once real dragging is evident.
              shiftPull = { lane, riders, start: [e.clientX, e.clientY] }
              e.stopPropagation(); e.preventDefault()
              return
            }
            lc.dragFromReroute(g, r)
            dimCoreSlots(app) // canvas-born drag: no Vue session, so core's dim never fires
            pullDrag = true
            pullStart = [e.clientX, e.clientY]
            g.setDirtyCanvas(true, true)
            e.stopPropagation(); e.preventDefault()
          }
          return
        }
        // Ctrl+alt on a populated IN pin: sever the lane's source and arm the
        // fresh source-seeking drag, core's input contract (2g greenlight).
        if (t?.side === 'in' && (e.ctrlKey || e.metaKey) && e.altKey && !e.shiftKey) {
          const lc = app.canvas.linkConnector
          const lane = t.comb?.lanes?.[t.lane]
          const riders = lane && lc && !lc.isConnecting ? outToothRiders(g, lane) : []
          if (riders.length) {
            cutLaneSource(g, lane, riders)
            lc.dragFromRerouteToOutput(g, r)
            if (lc.isConnecting) {
              dimCoreSlots(app)
              pullDrag = true
              pullStart = [e.clientX, e.clientY]
            }
            g.setDirtyCanvas(true, true)
            e.stopPropagation(); e.preventDefault()
            return
          }
        }
        // In-tooth of a lane dangling at the SOURCE (input-drag parked on the
        // gate): the tip carries only a floating link with no origin. Detaching
        // would rip the thread out of the ribbon for nothing -- instead resume
        // the float as an output-seeking drag (core's dragFromRerouteToOutput
        // reconnects the real target inputs through the surviving chain).
        if (t?.side === 'in' && !r.linkIds?.size) {
          const fl = r.firstFloatingLink
          if (fl && Number(fl.origin_id) === -1) {
            const lc = app.canvas.linkConnector
            if (lc && !lc.isConnecting) {
              lc.dragFromRerouteToOutput(g, r)
              dimCoreSlots(app)
              pullDrag = true
              pullStart = [e.clientX, e.clientY]
              g.setDirtyCanvas(true, true)
              e.stopPropagation(); e.preventDefault()
            }
            return
          }
        }
        press = { rid: r.id, x, y } // observe only; core owns the drag
        // Free-dot rides show the "+" squares (teeth never enroll themselves).
        if (!toothOf(r.id)) setDotDrag(true)
      }
    },
    true
  )

  window.addEventListener(
    'pointermove',
    (e) => {
      if (!active()) return
      const [x, y] = graphPt(e)
      if (shiftPull) {
        // The deferred out-pin shift gesture becomes real once the pointer
        // proves it is a drag: cut at the pin (records die, the lane parks
        // its source float via keepReroutes) and arm the consumer-anchored
        // bundle -- dragNewFromInput for the first wire, the rest built with
        // the same render-link class off the first instance.
        if (Math.hypot(e.clientX - shiftPull.start[0], e.clientY - shiftPull.start[1]) > 6) {
          const sp = shiftPull
          shiftPull = null
          const lc = app.canvas?.linkConnector
          const g2 = activeGraph(app)
          if (lc && !lc.isConnecting && g2) {
            for (const w of sp.riders) {
              forgetProvenance(w.node, w.slot)
              w.node.disconnectInput(w.slot, true)
            }
            inheritFloatFrom(g2, sp.lane)
            let armed = false
            for (const w of sp.riders) {
              const input = w.node.inputs?.[w.slot]
              if (!input) continue
              if (!armed) {
                lc.dragNewFromInput(g2, w.node, input)
                armed = lc.isConnecting
              } else {
                const Ctor = lc.renderLinks[0]?.constructor
                if (Ctor) lc.renderLinks.push(new Ctor(g2, w.node, input))
              }
            }
            invalidate()
            if (armed) {
              if (lc.renderLinks.length > 1) lc.state.multi = true
              dimCoreSlots(app)
              pullDrag = true
              pullStart = [...sp.start]
              driveSnap(e)
            }
            g2.setDirtyCanvas(true, true)
          }
        }
        return
      }
      if (pullDrag) {
        // Preview rides core's connecting_links; we drive the snap position (and
        // with it, motion over node DOM -- graph_mouse freezes there).
        driveSnap(e)
        activeGraph(app)?.setDirtyCanvas(true, true)
        return
      }
      if (gateDrag) {
        const L = window.LiteGraph
        // Grid snap when the setting demands it OR shift is held (core's
        // shift-drag convention, extended to gates).
        const wantSnap = (L?.alwaysSnapToGrid || e.shiftKey) && L?.CANVAS_GRID_SIZE > 0
        const grid = wantSnap ? L.CANVAS_GRID_SIZE : 0
        const snap = (v) => (grid ? Math.round(v / grid) * grid : v)
        const dx = x - gateDrag.press[0], dy = y - gateDrag.press[1]
        for (const gd of gateDrag.gates) {
          gd.comb[gd.which].pos = [snap(gd.origin[0] + dx), snap(gd.origin[1] + dy)]
        }
        for (const s of gateDrag.items) {
          const nx = snap(s.origin[0] + dx), ny = snap(s.origin[1] + dy)
          const mdx = nx - s.it.pos[0], mdy = ny - s.it.pos[1]
          if (!mdx && !mdy) continue
          // The store-syncing path is INVERTED between the two kinds: reroutes
          // sync via move() (raw pos setter), nodes sync via the pos SETTER --
          // LGraphNode.move() is a deliberate no-op in Vue mode (measured:
          // nodes silently ignored the gate drag).
          if (s.it.linkIds) s.it.move(mdx, mdy)
          else s.it.pos = [nx, ny]
        }
        activeGraph(app).setDirtyCanvas(true, true) // combPass re-lays the teeth
        return
      }
      if (follow) {
        // Node deltas already carry the grid snap; adding them keeps alignment.
        const dx = follow.refNode.pos[0] - follow.refPos[0]
        const dy = follow.refNode.pos[1] - follow.refPos[1]
        if (dx || dy) {
          for (const gd of follow.gates) {
            gd.comb[gd.which].pos = [gd.origin[0] + dx, gd.origin[1] + dy]
          }
          activeGraph(app)?.setDirtyCanvas(true, true)
        }
        return
      }
      if (canvasPress && app.canvas?.linkConnector?.isConnecting) {
        driveSnap(e) // core-owned drag born on the canvas (reroute slot pulls)
        return
      }
      if (!app.canvas?.pointer?.isDown) {
        setHover(combAt(activeGraph(app), x, y), activeGraph(app))
      }
    },
    true
  )

  // Interactive glyphs advertise themselves: pointer cursor over any clickable
  // zone. BUBBLE phase on purpose -- core's own element-level mousemove writes
  // the cursor every frame, so the override must run AFTER it (a capture-phase
  // write was overwritten to 'default' the moment core had ever dragged).
  document.addEventListener('pointermove', (e) => {
    if (!active() || app.canvas?.pointer?.isDown) return
    const g = activeGraph(app)
    if (!g) return
    const [x, y] = graphPt(e)
    const hit = combAt(g, x, y)
    const clicky = hit && ['flip', 'labels', 'sort', 'collapse', 'expand'].includes(hit.zone)
    const cv = app.canvas?.canvas
    if (!cv) return
    if (clicky) {
      cv.style.cursor = 'pointer'
      cursorSet = true
    } else if (cursorSet) {
      cv.style.cursor = 'default'
      cursorSet = false
    }
  })

  // Delete/Backspace on selected gates. Core's binding runs the
  // DeleteSelectedItems command off a WINDOW bubble listener -- gates are not
  // core items, so a gates-only selection reads as empty and toasts "Nothing
  // selected". Document capture preempts it. Mixed selections compose: we
  // dissolve the combs and let the event through so core deletes its own items;
  // swallowed only when core's set is empty (its handler would only toast).
  document.addEventListener(
    'keydown',
    (e) => {
      if (!active()) return
      if (e.key !== 'Delete' && e.key !== 'Backspace') return
      if (e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return
      const t = e.composedPath?.()[0] ?? e.target
      if (
        t instanceof Element &&
        (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT' || t.isContentEditable)
      ) return
      // Same scoping as core's own Delete binding.
      const container = document.getElementById('graph-canvas-container')
      if (container && t instanceof Node && !container.contains(t)) return
      const g = activeGraph(app)
      const sel = selectedGates(g)
      if (!sel.length) return
      const byComb = new Map()
      for (const { comb, which } of sel) {
        if (!byComb.has(comb)) byComb.set(comb, new Set())
        byComb.get(comb).add(which)
      }
      for (const [comb, sides] of byComb) dissolveComb(g, comb, sides)
      clearGateSelection()
      g.setDirtyCanvas(true, true)
      if (!app.canvas?.selectedItems?.size) { e.stopPropagation(); e.preventDefault() }
    },
    true
  )

  // A cancelled pointer (window blur, pen lift, browser gesture takeover) fires NO
  // pointerup; a stranded flag would make the next unrelated release run drop
  // resolution.
  window.addEventListener(
    'pointercancel',
    () => {
      pullDrag = false
      gateDrag = null
      follow = null
      press = null
      canvasPress = false
      shiftPull = null
      pullStart = null
      setDotDrag(false)
      undimCoreSlots()
    },
    true
  )

  window.addEventListener(
    'pointerup',
    (e) => {
      canvasPress = false
      setDotDrag(false) // any release retires the "+" squares next frame
      // Source drag released on a ribbon IN pin. The Vue finalize cannot be
      // trusted with this drop: its reroute-at-pointer stage claims success
      // even when core's connectToRerouteInput refuses an occupied input, so
      // the release dies with no fallback and no menu. This install-time
      // capture listener runs BEFORE the composable's per-drag listener --
      // handle the gate connect here; the guarded seams stay quiet after.
      {
        const lc = app.canvas?.linkConnector
        const g = activeGraph(app)
        if (lc?.isConnecting && lc.state?.connectingTo === 'input' && g && active()) {
          try {
            const [x, y] = graphPt(e)
            handleGateInDrop(app, lc, { canvasX: x, canvasY: y })
          } catch {
            /* never break a release; the normal drop pipeline continues */
          }
        }
      }
      if (shiftPull) {
        // Never left the click threshold: core's shift+click on an output is
        // a pure no-op, and nothing was mutated yet. Just disarm.
        shiftPull = null
        return
      }
      if (pullDrag) {
        pullDrag = false
        const lc = app.canvas?.linkConnector
        const g = activeGraph(app)
        // Zero-motion release: a core slot click is a pure no-op -- reset
        // silently instead of running drop resolution, which fell through to
        // the release menu and ate the user's next press (audit 2g).
        if (pullStart && Math.hypot(e.clientX - pullStart[0], e.clientY - pullStart[1]) <= 6) {
          pullStart = null
          lc?.reset?.(true)
          g?.setDirtyCanvas(true, true)
          undimCoreSlots()
          return
        }
        pullStart = null
        if (lc && g) {
          const [x, y] = graphPt(e)
          // Vue pins live in the DOM with centres ON the node's boundary, where the
          // legacy getNodeOnPos misses (measured: a drop dead on the pin fell through
          // to dropOnNothing and the search box kept the connector alive). The
          // [data-slot-key] element is not in the hit element's ancestry either
          // (separate layer), so resolve by PROXIMITY: nearest keyed slot within
          // 20px of the drop. Legacy dropLinks stays the fallback and still handles
          // gates, reroutes, node bodies, and release-on-empty.
          let connected = false
          let slotEl = null, bd = 400
          for (const el of document.querySelectorAll('[data-slot-key]')) {
            const r = el.getBoundingClientRect()
            if (!r.width) continue
            const dx = e.clientX - (r.x + r.width / 2)
            const dy = e.clientY - (r.y + r.height / 2)
            const d = dx * dx + dy * dy
            if (d < bd) { bd = d; slotEl = el }
          }
          const m = slotEl && /^(.+)-(in|out)-(\d+)$/.exec(slotEl.getAttribute('data-slot-key') ?? '')
          if (m) {
            const node = g.getNodeById(m[1]) ?? g.getNodeById(Number(m[1]))
            if (node && m[2] === 'in' && lc.state?.connectingTo === 'input') {
              const input = node.inputs?.[+m[3]]
              for (const rl of lc.renderLinks) {
                if (input && rl.canConnectToInput?.(node, input)) {
                  rl.connectToInput(node, input, lc.events)
                  connected = true
                }
              }
            } else if (node && m[2] === 'out' && lc.state?.connectingTo === 'output') {
              const output = node.outputs?.[+m[3]]
              for (const rl of lc.renderLinks) {
                if (output && rl.canConnectToOutput?.(node, output)) {
                  rl.connectToOutput(node, output, lc.events)
                  connected = true
                }
              }
            }
          }
          if (!connected) {
            // Core adorns pointer events the same way before handing them to dropLinks.
            e.canvasX = x; e.canvasY = y
            lc.dropLinks(g, e) // wrapped seam: gate drops become dangling lanes too
          }
          lc.reset?.(true)
          g.setDirtyCanvas(true, true)
        }
        undimCoreSlots()
        return
      }
      if (gateDrag) { gateDrag = null; activeGraph(app)?.setDirtyCanvas(true, true); return }
      if (follow) {
        // The node's drop-time grid snap lands AFTER this capture handler runs --
        // apply the final delta once core has settled, then disarm.
        const f = follow
        follow = null
        requestAnimationFrame(() => requestAnimationFrame(() => {
          const dx = f.refNode.pos[0] - f.refPos[0]
          const dy = f.refNode.pos[1] - f.refPos[1]
          for (const gd of f.gates) {
            gd.comb[gd.which].pos = [gd.origin[0] + dx, gd.origin[1] + dy]
          }
          activeGraph(app)?.setDirtyCanvas(true, true)
        }))
        return
      }
      if (!active() || !press) return
      const p = press
      press = null
      setDotDrag(false)
      const g = activeGraph(app)
      const [x, y] = graphPt(e)
      if (Math.hypot(x - p.x, y - p.y) < 6) return // click, not a pull
      const t = toothOf(p.rid)
      if (t) {
        // Detach runs SYNCHRONOUSLY: it only deletes the PARTNER tooth, so core's
        // in-flight drag of this dot is untouched. Deferring it loses the race
        // against combPass, which snaps the still-enrolled tooth back to its pin
        // before the deferred removal lands (measured: freed dot parked ON the pin
        // slot, overlapping the next lane's tooth and poisoning the next grab).
        if (t.side === 'in') {
          // Dropped back on its own in-gate (#4 ruling): the peel is aborted,
          // not completed -- the lane stays enrolled and the next combPass
          // snaps the tooth home. Detaching here left the user with a freed
          // dot sitting exactly where the lane used to be (2g QA).
          const at = combAt(g, x, y)
          if (at && at.comb === t.comb && at.which === 'in') return
          detachLane(g, t.comb, p.rid)
          g.setDirtyCanvas(true, true)
        }
        // out-teeth: no-op, the next combPass snaps them home
        return
      }
      // Create/enroll DELETE the dot core is finalizing -- those wait until core's
      // own pointerup handling is done with it.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const r = g.reroutes?.get?.(p.rid)
        if (!r || toothOf(p.rid)) return
        const gate = combAt(g, r.pos[0], r.pos[1])
        const other = gate ? null : rerouteNear(g, r.pos[0], r.pos[1], r)
        const otherTooth = other ? toothOf(other.id) : null
        // Lane creation lives on the "+" square only; gate body and teeth are
        // join-or-nothing (a same-source dot merges into its lane, anything
        // else stays a free dot where it landed).
        if (gate) {
          if (gate.zone === 'newlane') gestureEnroll(g, gate.comb, r)
          else gestureJoin(g, gate.comb, r)
        } else if (otherTooth) gestureJoin(g, otherTooth.comb, r)
        else if (other) gestureCreate(g, other, r)
        g.setDirtyCanvas(true, true)
      }))
    },
    true
  )
}
