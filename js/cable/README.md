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

Setting ids and the `graph.extra.cablemanagement_combs` record key are **left
alone on purpose**. Your stored preferences carry over, and every workflow you
have already rigged still opens correctly.

## The one addition: `groupcomb.js`

The only non-upstream file. Select a group, press **Alt+Shift+T**, and every wire
crossing the group boundary is gathered into a comb placed just outside the
group — inbound off the top-left, outbound off the bottom-right.

What it makes is an **ordinary comb**. It drives the public
`window.__cablemanagementCombs` API and touches none of the upstream modules, so
the result drags, collapses, flips, sorts and copies exactly like one you placed
by hand. The hotkey only decides where it goes and which wires ride it.

Needs at least two un-combed wires crossing the boundary. Re-running is safe —
already-combed wires are skipped.

`Alt+Shift+T`, because core rejects a duplicate combo with a red toast on every
load: `Alt+Shift+R` is taken by Reroute Roundup, and `Alt+Shift+G` sits too close
to KJNodes' `Ctrl+Shift+G` for comfort.

## Do not run the original alongside

Both patch `pathRenderer.drawLink` and blind `LiteGraph.Reroute`; whichever loads
last wins. The original now lives at
`custom_nodes/comfyui-cable-management.disabled` — ComfyUI skips any folder ending
`.disabled`. Its git history and the uncommitted work on `group-drag-ribbons`
(including PR #4) are intact there.

## Pulling upstream changes later

Copy `web/*.js`, `web/pathing/*.js` and the CSS over this directory, then redo the
two changes above. `groupcomb.js` is not upstream's and should survive untouched.

## Tests

`tests/cable/groupcomb.mjs` — Playwright, needs a running ComfyUI:

```
COMFY_URL=http://127.0.0.1:8188 node tests/cable/groupcomb.mjs
```

Keep `node_modules` out of `js/` — ComfyUI globs `js/**/*.js` and imports every
hit as an extension.
