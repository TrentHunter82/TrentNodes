// Wire tie gesture (js/cable/wiretie.js), driven with the real mouse.
//
//   s1 Shift + right-drag across three wires makes one comb with three lanes
//   s2 a cut on the upstream segment of a rerouted wire puts the teeth on THAT segment
//   s3 the gesture never opens the canvas context menu
//   s4 Escape cancels a live cut
//   s5 a second tie chains onto wires that already ride a comb
//   s6 cutting only one wire ties nothing
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

// Two feeders on the left, two consumers on the right; the vae wire carries a reroute
// to the right of the cut, so the cut lands on its upstream (reroute-ending) segment.
const built = await page.evaluate(async () => {
  const g = window.app.graph, L = window.LiteGraph
  g.clear()
  const ds = window.app.canvas.ds; ds.scale = 1; ds.offset = [0, 0]
  const mk = (t, x, y) => { const n = L.createNode(t); n.pos = [x, y]; g.add(n); return n }
  const a = mk('EmptyLatentImage', 60, 80)
  const c = mk('CheckpointLoaderSimple', 60, 380)
  const k = mk('KSampler', 1100, 80)
  const v = mk('VAEDecode', 1100, 640)
  a.connect(0, k, k.inputs.findIndex((s) => s.name === 'latent_image'))
  c.connect(0, k, k.inputs.findIndex((s) => s.name === 'model'))
  const vae = c.connect(2, v, v.inputs.findIndex((s) => s.name === 'vae'))
  const r = g.createReroute([900, 700], vae)
  g.setDirtyCanvas(true, true)
  await new Promise((res) => setTimeout(res, 800))
  return { vae: vae.id, reroute: r.id }
})

const combs = () => page.evaluate(async () => {
  for (let i = 0; i < 4; i++) {
    window.app.graph.setDirtyCanvas(true, true)
    await new Promise((r) => setTimeout(r, 120))
  }
  return (window.app.graph.extra.cablemanagement_combs ?? []).map((c) => ({
    lanes: c.lanes.map((l) => ({ ...l })), in: [...c.in.pos], out: [...c.out.pos]
  }))
})

// Graph point -> page point.
const client = (x, y) => page.evaluate(([x, y]) => {
  const c = window.app.canvas, r = c.canvas.getBoundingClientRect()
  return [(x + c.ds.offset[0]) * c.ds.scale + r.left, (y + c.ds.offset[1]) * c.ds.scale + r.top]
}, [x, y])

async function rightDrag(from, to, { escape = false } = {}) {
  const [x0, y0] = await client(...from)
  const [x1, y1] = await client(...to)
  await page.keyboard.down('Shift')
  await page.mouse.move(x0, y0)
  await page.mouse.down({ button: 'right' })
  await page.mouse.move(x1, y1, { steps: 12 })
  await page.waitForTimeout(100)
  if (escape) await page.keyboard.press('Escape')
  await page.mouse.up({ button: 'right' })
  await page.keyboard.up('Shift')
  await page.waitForTimeout(400)
}

// ---- s4 first, while the graph is clean ----
await rightDrag([700, 40], [700, 900], { escape: true })
ok('s4: Escape cancels the cut', (await combs()).length === 0)

// ---- s1 ----
await rightDrag([700, 40], [700, 900])
const s1 = await combs()
ok('s1: one comb, three lanes', s1.length === 1 && s1[0].lanes.length === 3, JSON.stringify(s1))

// ---- s2 ----
const chain = await page.evaluate(({ vae, reroute }) => {
  const g = window.app.graph
  const comb = g.extra.cablemanagement_combs?.[0]
  const lane = comb?.lanes.find((l) => g.reroutes.get(l.in)?.linkIds?.has(vae))
  if (!lane) return null
  return {
    rerouteParent: g.reroutes.get(reroute)?.parentId,
    outParent: g.reroutes.get(lane.out)?.parentId,
    lane,
    linkParent: g.links.get(vae)?.parentId
  }
}, built)
ok('s2: rerouted wire reads source -> in -> out -> reroute -> target',
  chain && chain.rerouteParent === chain.lane.out && chain.outParent === chain.lane.in &&
  chain.linkParent === built.reroute, JSON.stringify(chain))

// ---- s3 ----
const menus = await page.evaluate(() => document.querySelectorAll('.litecontextmenu, .p-contextmenu:not([style*="display: none"])').length)
ok('s3: no context menu opened', menus === 0, `menus=${menus}`)

// ---- s5 ----
// Model + vae again, upstream of the first comb: wires already riding a comb can be
// tied a second time, which is how a bundle gets its shape.
await rightDrag([520, 330], [560, 470])
const s5 = await combs()
ok('s5: a second tie chains onto wires that already ride a comb',
  s5.length === 2 && s5[1].lanes.length === 2 && s5[0].lanes.length === 3, JSON.stringify(s5.map((c) => c.lanes.length)))

// ---- s6 ----
// The latent wire alone, where it runs clear of the others.
await rightDrag([450, 130], [450, 230])
ok('s6: a single wire ties nothing', (await combs()).length === 2)

await page.locator('canvas').first().screenshot({ path: process.env.SHOT ?? '/tmp/wiretie.png' })
ok('no page errors', errs.length === 0, errs.slice(0, 3).join(' | '))
await b.close()
if (!pass) process.exit(1)
console.log('PASS')
