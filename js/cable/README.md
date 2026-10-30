# Cable Management (vendored) + group comb

A full copy of [comfyui-cable-management](https://github.com/vtokic/comfyui-cable-management)
by Vedran Aberle Tokić, MIT licensed — see `LICENSE-cable-management`. Everything
you already know about it works exactly as before: combs, pass-through pins,
drawers, PCB routing, gestures, sorting, clipboard.

Vendored on 2026-08-11 from the local `group-drag-ribbons` working tree, so this
copy already includes:

- the committed group-drag patch (ribbons ride group drags), upstream PR #4
- the "Ribbon runs match the link style" setting and its node-pin spline geometry

## What was changed from upstream

Deliberately as little as possible, so a future upstream copy can be dropped
straight over the top:

1. **Import depth.** Files here sit one directory deeper than upstream
   (`/extensions/TrentNodes/cable/…`), so `../../scripts/` became
   `../../../scripts/`, and `pathing/` gained one more level.
2. **Extension name** in `index.js` — `cablemanagement` → `TrentNodes.CableManagement`,
   only so it is identifiable and does not hard-collide if the original folder is
   re-enabled.
3. **`graph._links` → `graph.links`** in `combs.js`, `registry.js`, `ledger.js` and
   `primitive.js`. Frontend 1.57 dropped `LGraph._links`; without this every comb's
   teeth were scrubbed as wireless on the first frame. Upstream (v1.2.1) still reads
   `_links`.

Setting ids and the `graph.extra.cablemanagement_combs` record key are **left
alone on purpose**. Your stored preferences carry over, and every workflow you
have already rigged still opens correctly.

## The one addition: `groupcomb.js`

The only non-upstream file. Select a group, press **T**, and every wire
crossing the group boundary is gathered into a comb placed just outside the
group — inbound off the top-left, outbound off the bottom-right.

What it makes is an **ordinary comb**. It drives the public
`window.__cablemanagementCombs` API and touches none of the upstream modules, so
the result drags, collapses, flips, sorts and copies exactly like one you placed
by hand. The hotkey only decides where it goes and which wires ride it.

Needs at least two un-combed wires crossing the boundary. Re-running is safe —
already-combed wires are skipped.

A bare `T`, matching the rest of the keymap (`g`, `l`, `s`, `o`, `w`, `0`) rather
than the `Alt+Shift+<letter>` pattern this pack's own defaults use. Bare keys only
fire on the canvas, never while a text field has focus. T because core rejects a
duplicate combo with a red toast on every load: `R` is taken by Reroute Roundup
and `G` by core's Fit Group To Contents.

## Second addition: `wiretie.js`

**Shift + right-drag** a line across wires. The wires it crosses are highlighted with
a "Tie N wires" label, and on release they are gathered into a comb sitting on the
cut, lanes ordered top to bottom along the line. Escape cancels.

- The teeth go on the exact segment you cut, so a wire that already has reroutes
  keeps its shape.
- Wires already riding a comb can be tied again further along; chained ties are
  how a bundle gets its shape. The ribbon run inside a comb is never cut.
- Needs at least two wires. A bundle that mostly flows right to left gets mirrored
  gates.
- Shift + right-click on the canvas no longer opens the context menu.
- Does not work with circuit noodles on: they draw no hit paths, so a toast says
  to turn them off.

## Do not run the original alongside

Both patch `pathRenderer.drawLink` and blind `LiteGraph.Reroute`; whichever loads
last wins. The original now lives at
`custom_nodes/comfyui-cable-management.disabled` — ComfyUI skips any folder ending
`.disabled`. Its git history and the uncommitted work on `group-drag-ribbons`
(including PR #4) are intact there.

## Pulling upstream changes later

Copy `web/*.js`, `web/pathing/*.js` and the CSS over this directory, then redo the
three changes above. `groupcomb.js` and `wiretie.js` are not upstream's and should
survive untouched.

## Tests

`tests/cable/groupcomb.mjs` — Playwright, needs a running ComfyUI:

```
COMFY_URL=http://127.0.0.1:8188 node tests/cable/groupcomb.mjs
COMFY_URL=http://127.0.0.1:8188 node tests/cable/wiretie.mjs
```

`groupcomb.mjs` step s5 (ribbon routes) reports 0 routes with or without the 1.57
fix; not investigated yet.

Keep `node_modules` out of `js/` — ComfyUI globs `js/**/*.js` and imports every
hit as an extension.
