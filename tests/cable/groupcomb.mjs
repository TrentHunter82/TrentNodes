// Group comb hotkey, over the VENDORED cable-management copy in TrentNodes/js/cable.
// Asserts that the hotkey produces an ORDINARY comb -- upstream's own record, in
// upstream's own graph.extra key -- placed at the group corner.
//
//   s1 the vendored pack boots: comb API, PCB mode, and the group command all present
//   s2 the hotkey combs a group: upstream records, one lane per crossing wire
//   s3 the comb lands at the group's top-left, outside the frame
//   s4 re-running enrolls nothing twice
//   s5 upstream's own machinery is intact (gestures + clipboard modules loaded)
import { chromium } from 'playwright'

const URL = process.env.COMFY_URL ?? 'http://127.0.0.1:8188'
const b = await chromium.launch({ headless: true })
const page = await b.newPage({ viewport: { width: 1800, height: 1000 } })
const errs = []
page.on('pageerror', (e) => errs.push(String(e).split('\n')[0]))
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 90_000 })
await page.waitForFunction(() => window.app?.graph && window.__cablemanagementCombs, null, { timeout: 120_000 })
await page.waitForTimeout(2500)

let pass = true
const ok = (label, cond, extra = '') => {
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}${extra ? ' -- ' + extra : ''}`)
  if (!cond) pass = false
}

// ---- s1 ----
const boot = await page.evaluate(() => ({
  api: typeof window.__cablemanagementCombs?.create === 'function',
  patched: !!window.__cablemanagementPathing?.state?.patched,
  pcb: window.__cablemanagementPathing?.PCB?.(),
  ext: (window.app.extensions ?? []).filter((e) => /CableManagement|GroupComb/.test(e.name)).map((e) => e.name),
  cmd: !!window.app.extensionManager.command.commands?.some?.((c) => c.id === 'TrentNodes.GroupComb.CombGroup')
}))
ok('s1: vendored pack booted with the group command',
  boot.api && boot.patched && boot.pcb >= 64 && boot.cmd && boot.ext.length === 2, JSON.stringify(boot))

// Build: two feeders outside, two consumers inside a group, one sink outside.
const built = await page.evaluate(async () => {
  const g = window.app.graph, L = window.LiteGraph
  g.clear()
  const ds = window.app.canvas.ds; ds.scale = 0.8; ds.offset = [60, 40]
  const mk = (t, x, y) => { const n = L.createNode(t); n.pos = [x, y]; g.add(n); return n }
  const feedA = mk('EmptyLatentImage', 80, 120)
  const feedB = mk('CheckpointLoaderSimple', 80, 420)
  const k = mk('KSampler', 780, 120)
  const v = mk('VAEDecode', 780, 560)
  const sink = mk('PreviewImage', 1420, 300)
  feedA.connect(0, k, k.inputs.findIndex((s) => s.name === 'latent_image'))
  feedB.connect(0, k, k.inputs.findIndex((s) => s.name === 'model'))
  feedB.connect(2, v, v.inputs.findIndex((s) => s.name === 'vae'))
  k.connect(0, v, v.inputs.findIndex((s) => s.name === 'samples'))
  v.connect(0, sink, 0)
  const grp = new L.LGraphGroup('Sampling', 1)
  grp.pos = [700, 60]; grp.size = [560, 700]
  g.add(grp)
  g.setDirtyCanvas(true, true)
  await new Promise((r) => setTimeout(r, 1200))
  grp.recomputeInsideNodes()
  return { inside: grp._nodes.map((n) => n.type), pos: [...grp.pos], size: [...grp.size] }
})
ok('setup: group holds the two inner nodes', built.inside.length === 2, JSON.stringify(built.inside))

const combs = () => page.evaluate(async () => {
  for (let i = 0; i < 6; i++) {
    window.app.graph.setDirtyCanvas(false, true)
    await new Promise((r) => setTimeout(r, 120))
  }
  return (window.app.graph.extra.cablemanagement_combs ?? []).map((c) => ({
    id: c.id, lanes: c.lanes.length, in: [...c.in.pos], out: [...c.out.pos]
  }))
})

// ---- s2 ----
await page.evaluate(() => {
  window.app.canvas.selectedItems = new Set([window.app.graph._groups[0]])
  return window.app.extensionManager.command.execute('TrentNodes.GroupComb.CombGroup')
})
await page.waitForTimeout(700)
const s2 = await combs()
// 3 wires enter (latent, model, vae); 1 leaves, below the two-lane floor.
ok('s2: one upstream comb, one lane per inbound wire',
  s2.length === 1 && s2[0].lanes === 3, JSON.stringify(s2))

// ---- s3 ----
const placed = s2[0] && s2[0].in[0] < built.pos[0] && s2[0].in[1] > built.pos[1] &&
  s2[0].in[1] < built.pos[1] + built.size[1] / 2
ok('s3: comb sits outside the left edge, near the top', placed,
  JSON.stringify({ comb: s2[0]?.in, group: built.pos }))

// ---- s4 ----
await page.evaluate(() => {
  window.app.canvas.selectedItems = new Set([window.app.graph._groups[0]])
  return window.app.extensionManager.command.execute('TrentNodes.GroupComb.CombGroup')
})
await page.waitForTimeout(700)
const s4 = await combs()
ok('s4: re-running enrolls nothing twice',
  s4.length === 1 && s4[0].lanes === 3, JSON.stringify(s4.map((c) => c.lanes)))

// ---- s5: upstream machinery still present ----
const intact = await page.evaluate(() => ({
  gestures: typeof window.__cablemanagementCombs?.selection === 'function',
  labels: typeof window.__cablemanagementCombs?.labels === 'function',
  flip: typeof window.__cablemanagementCombs?.flip === 'function',
  routes: (window.__cablemanagementPathing?.routes?.() ?? []).filter((r) => r.key.includes('|X')).length
}))
ok('s5: upstream comb features intact (flip, labels, selection, ribbon routes)',
  intact.gestures && intact.labels && intact.flip && intact.routes === 3, JSON.stringify(intact))

// ---- s6: the real keypress, not just command.execute ----
// Bare `t` is only useful if it survives the keybinding store, so drive it from the
// keyboard on a focused canvas rather than calling the command by id.
await page.evaluate(() => {
  const g = window.app.graph
  for (const c of g.extra.cablemanagement_combs ?? []) window.__cablemanagementCombs.decompose(c.id)
  g.extra.cablemanagement_combs = []
  g.setDirtyCanvas(true, true)
})
await page.waitForTimeout(500)
// No click to focus: the toast stack from the earlier steps sits over the canvas and
// intercepts pointer events. The keybinding handler listens at the document level, so
// an unfocused body is enough -- just make sure no text field is holding the keys.
await page.evaluate(() => {
  document.activeElement?.blur?.()
  window.app.canvas.selectedItems = new Set([window.app.graph._groups[0]])
})
await page.keyboard.press('t')
await page.waitForTimeout(900)
const s6 = await combs()
ok('s6: bare T fires the command from the keyboard', s6.length === 1 && s6[0].lanes === 3, JSON.stringify(s6))

await page.locator('canvas').first().screenshot({ path: process.env.SHOT ?? '/tmp/groupcomb.png' })
ok('no page errors', errs.length === 0, errs.slice(0, 3).join(' | '))
await b.close()
if (!pass) process.exit(1)
console.log('PASS')
