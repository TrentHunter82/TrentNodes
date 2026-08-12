// Lane sort modal (#4, jscammie's re-ordering ask). One row per lane -- type
// dot + resolved source label -- drag rows to reorder, Apply writes the order
// into comb.lanes (both gates follow: lane order IS pin order), Cancel/close
// discards.
//
// The modal itself is COMFY'S: the Vue dialog store (the template picker's
// showDialog) accepts a plain tag as `component`, so we get the real chrome
// around an empty div we fill with vanilla DOM. LANDMINE: the store is reached
// through private seams ([data-v-app].__vue_app__ -> pinia._s) -- a frontend
// bump can move them; the legacy comfyAPI ComfyDialog is the degradation path
// (public export, dated skin, same content).
import { laneLabel } from './combs.js'

const KEY = 'cablemanagement-sort'

function dialogStore() {
  try {
    const vueApp = document.querySelector('[data-v-app]')?.__vue_app__
    return vueApp?.config?.globalProperties?.$pinia?._s?.get?.('dialog') ?? null
  } catch {
    return null
  }
}

function typeColor(type) {
  const colors = window.LGraphCanvas?.link_type_colors ?? {}
  return colors[type] ?? '#9a9a9a'
}

function laneType(graph, lane) {
  const t = graph.reroutes?.get?.(lane.in)
  const lid = t?.linkIds?.values?.().next?.().value
  const l = lid != null ? (graph.getLink ? graph.getLink(lid) : graph._links?.get?.(lid)) : null
  return l?.type ?? t?.firstFloatingLink?.type ?? null
}

export function openSortModal(graph, comb) {
  const content = document.createElement('div')
  content.className = 'cablemanagement-sort'

  const list = document.createElement('div')
  const rows = comb.lanes.map((lane, i) => {
    const row = document.createElement('div')
    row.className = 'cablemanagement-sort-row'
    const dot = document.createElement('span')
    dot.className = 'cablemanagement-sort-dot'
    dot.style.background = typeColor(laneType(graph, lane))
    row.appendChild(dot)
    row.appendChild(document.createTextNode(laneLabel(graph, lane)))
    row.__lane = comb.lanes[i]
    list.appendChild(row)
    return row
  })
  content.appendChild(list)

  // Row drag: pointer capture on the pressed row; crossing a sibling's midpoint
  // swaps it in the DOM immediately -- the DOM order IS the pending order.
  let drag = null
  list.addEventListener('pointerdown', (e) => {
    const row = e.target.closest('.cablemanagement-sort-row')
    if (!row) return
    drag = row
    row.classList.add('cablemanagement-sort-dragging')
    row.setPointerCapture(e.pointerId)
    e.preventDefault()
  })
  list.addEventListener('pointermove', (e) => {
    if (!drag) return
    for (const sib of [...list.children]) {
      if (sib === drag) continue
      const r = sib.getBoundingClientRect()
      const mid = r.top + r.height / 2
      const di = [...list.children].indexOf(drag)
      const si = [...list.children].indexOf(sib)
      if (si < di && e.clientY < mid) list.insertBefore(drag, sib)
      else if (si > di && e.clientY > mid) list.insertBefore(drag, sib.nextSibling)
    }
  })
  const endDrag = () => {
    drag?.classList.remove('cablemanagement-sort-dragging')
    drag = null
  }
  list.addEventListener('pointerup', endDrag)
  list.addEventListener('pointercancel', endDrag)

  const buttons = document.createElement('div')
  buttons.className = 'cablemanagement-sort-buttons'
  const cancel = document.createElement('button')
  cancel.textContent = 'Cancel'
  const apply = document.createElement('button')
  apply.textContent = 'Apply'
  buttons.appendChild(cancel)
  buttons.appendChild(apply)
  content.appendChild(buttons)

  const store = dialogStore()
  let legacy = null
  const close = () => {
    if (store) store.closeDialog({ key: KEY })
    else legacy?.close?.()
  }
  cancel.addEventListener('click', close)
  apply.addEventListener('click', () => {
    comb.lanes = [...list.children].map((row) => row.__lane).filter(Boolean)
    graph.setDirtyCanvas?.(true, true)
    close()
  })

  if (store) {
    store.showDialog({ key: KEY, title: 'Sort lanes', component: 'div', props: { id: 'cablemanagement-sort-mount' } })
    // The mount renders on the dialog's next tick; poll briefly.
    const t0 = performance.now()
    const tryMount = () => {
      const mount = document.getElementById('cablemanagement-sort-mount')
      if (mount) mount.appendChild(content)
      else if (performance.now() - t0 < 2000) requestAnimationFrame(tryMount)
    }
    requestAnimationFrame(tryMount)
    return
  }
  const D = window.comfyAPI?.dialog?.ComfyDialog
  if (!D) return
  legacy = new D()
  legacy.show(content)
}
