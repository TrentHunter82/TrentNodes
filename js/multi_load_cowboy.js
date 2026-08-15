/**
 * Multi-Load Cowboy - a six slot image loader grid on the node face.
 *
 * The node keeps six plain combo widgets (image_1 .. image_6) for its
 * state. This extension hides them and draws a grid of cells over the
 * top: click a cell to pick or upload, drop files on a cell, drag one
 * cell onto another to swap. Every change writes straight back to the
 * combo it belongs to, so save, load, undo and the API all keep working
 * with no extra state of our own.
 *
 * The look follows the "soft-ui" design system: one surface colour, a
 * raised shadow for anything you can act on, a sunken well for anything
 * empty, mono numerals and uppercase letterspaced labels.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "MultiLoadCowboy";
const EMPTY = "(empty)";
const SLOTS = 6;
const SLOT_NAMES = Array.from({ length: SLOTS }, (_, i) => `image_${i + 1}`);

/* Layout constants, in unscaled canvas pixels. */
const DOM_MARGIN = 10;     /* the frontend's own inset for a DOM widget */
const PAD = 10;
const GAP = 9;
const HEAD_H = 20;
const CELL_RATIO = 0.72;   /* cell height / cell width */
const SLOT_H = 20;         /* one output row */
const LABEL_RESERVE = 96;  /* keep the output labels clear of the grid */
const MIN_GRID_BOX = 170;  /* narrower than this, sit under the slots */
const MIN_BAND = 90;       /* too little room to bother sharing the band */
const TOP_INSET = 12;      /* never ride up into the title bar */
const CLEARANCE = 8;       /* gap kept between the grid and the settings */
const ROW_SPACING = 4;     /* the frontend adds this to every widget box */

/* Bumped on every upload so replaced files re-fetch instead of showing
   a stale browser cache entry. */
let cacheBust = 0;

/* ------------------------------------------------------------------ */
/* Icons                                                               */
/* ------------------------------------------------------------------ */

const ICON_PLUS =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round">
        <path d="M12 6v12M6 12h12"/></svg>`;

const ICON_X =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round">
        <path d="M7 7l10 10M17 7L7 17"/></svg>`;

const ICON_SWEEP =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M5 7h14M10 7V5h4v2M8 7l1 12h6l1-12"/></svg>`;

const ICON_WARN =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>`;

/* ------------------------------------------------------------------ */
/* Styles                                                              */
/* ------------------------------------------------------------------ */

const CSS = `
.mlc-root, .mlc-pop {
    --mlc-surface: #202226;
    --mlc-d1: #07080a;
    --mlc-l1: #32363d;
    --mlc-well: linear-gradient(145deg, #191b1e, #26282e);
    --mlc-text: #e6e6ea;
    --mlc-muted: #8a8b92;
    --mlc-faint: #62646b;
    --mlc-hair: rgba(255, 255, 255, .07);
    --mlc-ring: rgba(255, 255, 255, .22);
    --mlc-font: "Segoe UI", system-ui, -apple-system,
                "Helvetica Neue", Arial, sans-serif;
    --mlc-mono: "Cascadia Code", ui-monospace, "SF Mono", Consolas,
                "Roboto Mono", monospace;
    --mlc-raise: 6px 6px 14px var(--mlc-d1), -5px -5px 12px var(--mlc-l1);
    --mlc-raise-sm: 3px 3px 7px var(--mlc-d1), -3px -3px 7px var(--mlc-l1);
    --mlc-sink: inset 4px 4px 8px var(--mlc-d1),
                inset -3px -3px 7px var(--mlc-l1);
}
.mlc-root.mlc-light, .mlc-pop.mlc-light {
    --mlc-surface: #e8eaee;
    --mlc-d1: #c3c7d1;
    --mlc-l1: #ffffff;
    --mlc-well: linear-gradient(145deg, #dfe2e8, #f0f2f5);
    --mlc-text: #2b2d33;
    --mlc-muted: #787b84;
    --mlc-faint: #9ea2ac;
    --mlc-hair: rgba(30, 32, 38, .10);
    --mlc-ring: rgba(44, 46, 52, .30);
}

/* The root fills the widget box and is only a positioning frame; the
   panel inside is the part that gets a width and the soft-ui shell.
   The frame takes no pointer events: it is wider than the panel, and
   anything it caught would be a dead patch of canvas over the output
   labels - no panning, no dragging a wire out of a socket. */
.mlc-root {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: stretch;
    font-family: var(--mlc-font);
    color: var(--mlc-text);
    user-select: none;
    -webkit-font-smoothing: antialiased;
    pointer-events: none;
}
.mlc-root * { box-sizing: border-box; }

/* The frontend sizes its own wrapper to the whole widget box and writes
   pointer-events: auto onto it from a Vue computed style on every
   render, so the only way to hand that strip back to the canvas is a
   rule that outranks an inline style. */
.dom-widget:has(> .mlc-root),
.dom-widget:has(> .mlc-spacer) {
    pointer-events: none !important;
}

.mlc-panel {
    pointer-events: auto;
    width: 100%;
    height: 100%;
    min-height: 0;
    box-sizing: border-box;
    padding: ${PAD}px;
    display: flex;
    flex-direction: column;
    gap: 0;
    border-radius: 16px;
    background: var(--mlc-surface);
    box-shadow: var(--mlc-raise);
    overflow: hidden;
}

.mlc-head {
    height: ${HEAD_H}px;
    flex: 0 0 ${HEAD_H}px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 0 2px 0 4px;
}
.mlc-eyebrow {
    font-family: var(--mlc-mono);
    font-size: 9px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--mlc-faint);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.mlc-head-right { display: flex; align-items: center; gap: 7px; }
.mlc-count {
    font-family: var(--mlc-mono);
    font-size: 10px;
    color: var(--mlc-muted);
    font-variant-numeric: tabular-nums;
    letter-spacing: .04em;
}
.mlc-sweep {
    width: 19px; height: 19px;
    padding: 0; border: 0; border-radius: 50%;
    display: grid; place-items: center;
    cursor: pointer;
    background: var(--mlc-surface);
    box-shadow: var(--mlc-raise-sm);
    color: var(--mlc-muted);
    transition: color .18s, box-shadow .18s, opacity .18s;
}
.mlc-sweep svg { width: 11px; height: 11px; }
.mlc-sweep:hover { color: var(--mlc-text); }
.mlc-sweep:active { box-shadow: var(--mlc-sink); }
.mlc-sweep[disabled] { opacity: .25; pointer-events: none; }

/* Rows share whatever height the widget box really has, so the last row
   can never be clipped by a layout that disagrees with our arithmetic. */
.mlc-grid {
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: repeat(var(--mlc-cols, 3), 1fr);
    grid-template-rows: repeat(var(--mlc-rows, 2), 1fr);
    gap: ${GAP}px;
}

.mlc-cell {
    position: relative;
    border-radius: 11px;
    background: var(--mlc-well);
    box-shadow: var(--mlc-sink);
    cursor: pointer;
    overflow: hidden;
    transition: box-shadow .18s, transform .14s;
}
.mlc-cell:hover { transform: translateY(-1px); }
.mlc-cell.is-filled { box-shadow: var(--mlc-raise-sm); }
/* Hairline so a photo reads as set into the tile, not pasted on it. */
.mlc-cell.is-filled::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 11px;
    box-shadow: inset 0 0 0 1px var(--mlc-hair);
    pointer-events: none;
}
.mlc-cell.is-drop {
    box-shadow: var(--mlc-sink), 0 0 0 2px var(--mlc-ring);
    transform: none;
}
.mlc-cell.is-dragging { opacity: .35; }

.mlc-thumb {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    display: none;
    border-radius: 11px;
}
.mlc-cell.is-filled .mlc-thumb { display: block; }

.mlc-plus {
    position: absolute;
    inset: 0;
    display: grid; place-items: center;
    color: var(--mlc-faint);
    transition: color .18s, transform .18s;
}
.mlc-plus svg {
    width: 22px; height: 22px;
    filter: drop-shadow(1px 1px 1px var(--mlc-d1));
}
.mlc-cell:hover .mlc-plus { color: var(--mlc-text); }
.mlc-cell.is-drop .mlc-plus { transform: scale(1.18); color: var(--mlc-text); }
.mlc-cell.is-filled .mlc-plus,
.mlc-cell.is-missing .mlc-plus { display: none; }

.mlc-warn {
    position: absolute;
    inset: 0;
    display: none;
    place-items: center;
    color: var(--mlc-muted);
}
.mlc-warn svg { width: 18px; height: 18px; }
.mlc-cell.is-missing .mlc-warn { display: grid; }

.mlc-num {
    position: absolute;
    left: 6px; bottom: 5px;
    font-family: var(--mlc-mono);
    font-size: 9px;
    letter-spacing: .06em;
    color: var(--mlc-faint);
    font-variant-numeric: tabular-nums;
    pointer-events: none;
    transition: color .18s;
}
.mlc-cell.is-filled .mlc-num {
    left: 5px; bottom: 4px;
    padding: 1px 5px;
    border-radius: 999px;
    color: #e6e6ea;
    background: rgba(10, 11, 13, .55);
    backdrop-filter: blur(3px);
}

.mlc-meta {
    position: absolute;
    left: 0; right: 0; bottom: 0;
    padding: 14px 6px 4px 6px;
    font-family: var(--mlc-mono);
    font-size: 8.5px;
    line-height: 1.25;
    color: rgba(230, 230, 234, .92);
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    background: linear-gradient(180deg,
        rgba(10, 11, 13, 0), rgba(10, 11, 13, .72));
    opacity: 0;
    transition: opacity .18s;
    pointer-events: none;
    display: none;
}
.mlc-cell.is-filled .mlc-meta { display: block; }
.mlc-cell.is-filled:hover .mlc-meta { opacity: 1; }

.mlc-clear {
    position: absolute;
    top: 4px; right: 4px;
    width: 17px; height: 17px;
    padding: 0; border: 0; border-radius: 50%;
    display: none; place-items: center;
    cursor: pointer;
    color: #e6e6ea;
    background: rgba(10, 11, 13, .58);
    backdrop-filter: blur(3px);
    opacity: 0;
    transition: opacity .18s, background .18s;
}
.mlc-clear svg { width: 9px; height: 9px; }
.mlc-cell.is-filled .mlc-clear,
.mlc-cell.is-missing .mlc-clear { display: grid; }
.mlc-cell:hover .mlc-clear { opacity: 1; }
.mlc-clear:hover { background: rgba(10, 11, 13, .85); }

/* ---- picker popover ---- */

.mlc-pop {
    position: fixed;
    z-index: 10010;
    width: 322px;
    padding: 14px;
    border-radius: 18px;
    background: var(--mlc-surface);
    box-shadow: var(--mlc-raise), 0 18px 40px rgba(0, 0, 0, .45);
    font-family: var(--mlc-font);
    color: var(--mlc-text);
    display: flex;
    flex-direction: column;
    gap: 11px;
}
.mlc-pop-head {
    display: flex; align-items: center; justify-content: space-between;
}
.mlc-pop-title {
    font-family: var(--mlc-mono);
    font-size: 9.5px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--mlc-faint);
}
.mlc-pop-row { display: flex; gap: 9px; }
.mlc-btn {
    flex: 1;
    border: 0;
    cursor: pointer;
    font-family: var(--mlc-font);
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: .05em;
    color: var(--mlc-text);
    background: var(--mlc-surface);
    box-shadow: var(--mlc-raise-sm);
    border-radius: 999px;
    padding: 9px 14px;
    transition: box-shadow .18s, transform .14s, color .2s;
}
.mlc-btn:hover { transform: translateY(-1px); }
.mlc-btn:active { box-shadow: var(--mlc-sink); color: var(--mlc-muted); }
.mlc-btn.mlc-btn-quiet { flex: 0 0 auto; color: var(--mlc-muted); }
.mlc-btn.mlc-btn-quiet:hover { color: var(--mlc-text); }
.mlc-btn.is-busy { pointer-events: none; color: var(--mlc-muted); box-shadow: var(--mlc-sink); }

.mlc-search {
    border: 0;
    width: 100%;
    font-family: var(--mlc-font);
    font-size: 12px;
    color: var(--mlc-text);
    background: var(--mlc-surface);
    box-shadow: var(--mlc-sink);
    border-radius: 12px;
    padding: 9px 12px;
    outline: none;
}
.mlc-search::placeholder { color: var(--mlc-faint); }
.mlc-search:focus { box-shadow: var(--mlc-sink), 0 0 0 2px var(--mlc-ring); }

.mlc-pop-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    max-height: 258px;
    overflow-y: auto;
    padding: 2px;
    scrollbar-width: thin;
    scrollbar-color: var(--mlc-l1) transparent;
}
.mlc-pop-grid::-webkit-scrollbar { width: 8px; }
.mlc-pop-grid::-webkit-scrollbar-thumb {
    background: var(--mlc-l1); border-radius: 999px;
}
.mlc-opt {
    position: relative;
    height: 62px;
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    background: var(--mlc-surface);
    box-shadow: var(--mlc-sink);
    transition: box-shadow .16s, transform .14s;
}
.mlc-opt:hover { transform: translateY(-1px); box-shadow: var(--mlc-raise-sm); }
.mlc-opt.is-current { box-shadow: var(--mlc-raise-sm), 0 0 0 2px var(--mlc-ring); }
.mlc-opt img {
    width: 100%; height: 100%; object-fit: cover; display: block;
}
.mlc-opt .mlc-opt-name {
    position: absolute; left: 0; right: 0; bottom: 0;
    padding: 10px 5px 3px;
    font-family: var(--mlc-mono);
    font-size: 8px;
    color: #e6e6ea;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    background: linear-gradient(180deg,
        rgba(10, 11, 13, 0), rgba(10, 11, 13, .78));
}
.mlc-pop-foot {
    font-family: var(--mlc-mono);
    font-size: 9px;
    letter-spacing: .08em;
    color: var(--mlc-faint);
    text-align: center;
}
@media (prefers-reduced-motion: reduce) {
    .mlc-root *, .mlc-pop * { transition: none !important; }
}
`;

function injectCSS() {
    if (document.getElementById("mlc-css")) return;
    const style = document.createElement("style");
    style.id = "mlc-css";
    style.textContent = CSS;
    document.head.appendChild(style);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/**
 * Let the canvas have the pointer back.
 *
 * The frontend wraps every DOM widget in its own div sized to the whole
 * widget box, which is wider than the panel we draw in it. The stylesheet
 * rule above is what actually holds that wrapper open, because the
 * frontend rewrites its inline pointer-events on every render. This is
 * the fallback for an engine without :has(), and it only ever switches
 * the wrapper off: the panel inside sets pointer-events back to auto,
 * and a descendant that does so is still a target.
 */
function releaseWrapper(element) {
    const parent = element?.parentElement;
    if (!parent || !parent.style) return;
    const name = typeof parent.className === "string" ? parent.className : "";
    if (name && !name.includes("dom-widget")) return;
    if (parent.style.pointerEvents !== "none") {
        parent.style.pointerEvents = "none";
    }
}

/** Tell the user something went wrong, without a modal if possible. */
function report(message) {
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({
            severity: "error",
            summary: "Multi-Load Cowboy",
            detail: message,
            life: 6000,
        });
        return;
    }
    alert(`Multi-Load Cowboy\n\n${message}`);
}

/** True when ComfyUI is running a light palette. */
function isLightTheme() {
    try {
        const bg = getComputedStyle(document.body).backgroundColor;
        const m = /(\d+),\s*(\d+),\s*(\d+)/.exec(bg || "");
        if (!m) return false;
        const luma = (0.2126 * +m[1] + 0.7152 * +m[2] + 0.0722 * +m[3]) / 255;
        return luma > 0.5;
    } catch (err) {
        return false;
    }
}

/** Split "sub/name.png [input]" into a /view query. */
function viewURL(value) {
    if (!value || value === EMPTY) return null;
    let name = value;
    let type = "input";
    const annotated = /^(.*)\s+\[(\w+)\]$/.exec(name);
    if (annotated) {
        name = annotated[1];
        type = annotated[2];
    }
    const cut = name.lastIndexOf("/");
    const subfolder = cut >= 0 ? name.slice(0, cut) : "";
    const filename = cut >= 0 ? name.slice(cut + 1) : name;
    return api.apiURL(
        `/view?filename=${encodeURIComponent(filename)}` +
        `&type=${encodeURIComponent(type)}` +
        `&subfolder=${encodeURIComponent(subfolder)}` +
        `&rand=${cacheBust}`
    );
}

/** Short label for a cell: the file name without its folder. */
function shortName(value) {
    if (!value || value === EMPTY) return "";
    const stripped = value.replace(/\s+\[\w+\]$/, "");
    const cut = stripped.lastIndexOf("/");
    return cut >= 0 ? stripped.slice(cut + 1) : stripped;
}

function slotWidgets(node) {
    return SLOT_NAMES.map(
        (name) => node.widgets?.find((w) => w.name === name)
    );
}

/** Every file name the six combos know about, minus the empty marker. */
function knownFiles(node) {
    const widget = slotWidgets(node).find((w) => w && w.options);
    const values = widget?.options?.values;
    if (!Array.isArray(values)) return [];
    return values.filter((v) => v && v !== EMPTY);
}

/** Teach every slot about a file that arrived after the page loaded. */
function rememberFile(node, value) {
    if (!value || value === EMPTY) return;
    for (const widget of slotWidgets(node)) {
        const values = widget?.options?.values;
        if (Array.isArray(values) && !values.includes(value)) {
            values.push(value);
        }
    }
}

function getSlot(node, index) {
    const widget = slotWidgets(node)[index];
    return widget ? widget.value : EMPTY;
}

function setSlot(node, index, value) {
    const widget = slotWidgets(node)[index];
    if (!widget) return;
    const next = value || EMPTY;
    if (widget.value === next) return;
    rememberFile(node, next);
    widget.value = next;
    if (widget.callback) widget.callback(next);
    node.graph?.setDirtyCanvas(true, true);
}

/** Ask the server for the current input folder listing. */
async function refreshFileList(node) {
    try {
        const resp = await api.fetchApi(`/object_info/${NODE_NAME}`);
        if (!resp.ok) return;
        const info = await resp.json();
        const spec = info?.[NODE_NAME]?.input?.required?.image_1;
        const values = Array.isArray(spec?.[0]) ? spec[0] : null;
        if (!values || !values.length) return;
        for (const widget of slotWidgets(node)) {
            if (widget?.options) widget.options.values = [...values];
        }
    } catch (err) {
        /* An offline listing is not worth an error dialog. */
    }
}

async function uploadFile(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    const resp = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });
    if (resp.status !== 200) {
        throw new Error(`upload failed: ${resp.status} ${resp.statusText}`);
    }
    const data = await resp.json();
    cacheBust += 1;
    return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function pickFiles(multiple) {
    return new Promise((resolve) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.multiple = !!multiple;
        input.style.display = "none";
        document.body.appendChild(input);
        input.addEventListener("change", () => {
            const files = Array.from(input.files || []);
            input.remove();
            resolve(files);
        });
        input.addEventListener("cancel", () => {
            input.remove();
            resolve([]);
        });
        input.click();
    });
}

/** Grid geometry for a panel of the given width. */
function layout(boxWidth) {
    const box = Math.max(120, boxWidth || 340);
    const cols = box < 250 ? 2 : 3;
    const rows = Math.ceil(SLOTS / cols);
    const inner = box - PAD * 2 - GAP * (cols - 1);
    const cellW = Math.max(34, Math.floor(inner / cols));
    const cellH = Math.max(26, Math.round(cellW * CELL_RATIO));
    const height = HEAD_H + PAD * 2 + rows * cellH + GAP * (rows - 1);
    return { cols, rows, cellW, cellH, height, box };
}

/**
 * Where the panel sits, and how far to pull it up.
 *
 * With no inputs, the band beside the output labels is dead space. The
 * grid widget lives at the end of the widget list, where it cannot
 * disturb saved values, and a lift widget with a negative height moves
 * it up into that band. The height is capped to the band so the grid can
 * never land on top of the settings below it. A node too narrow to share
 * the band keeps the grid where it sits, under everything else.
 *
 * `liftY` is where the lift itself lands and `settingsTop` is where the
 * first real widget sits. Both depend only on the widgets before them,
 * so feeding them back in here cannot oscillate.
 */
function plan(node, elementWidth, liftY, settingsTop) {
    const elemW = elementWidth ||
        Math.max(120, (node.size?.[0] || 380) - DOM_MARGIN * 2);
    const band = (node.outputs?.length || 0) * SLOT_H;
    const ceiling = settingsTop || band + 6;
    /* The widget box is the panel plus the frontend's margins, and it
       has to clear the first setting below it. */
    const roomBeside = Math.max(
        0, ceiling - TOP_INSET - DOM_MARGIN * 2 - ROW_SPACING - CLEARANCE
    );
    const beside = elemW - LABEL_RESERVE >= MIN_GRID_BOX &&
        roomBeside >= MIN_BAND;
    const available = beside ? elemW - LABEL_RESERVE : elemW;
    let geometry = layout(available);
    let height = geometry.height;

    if (beside) {
        /* The band sets the height, so a very wide node would stretch
           the cells into letterboxes. Hold the panel to the width that
           keeps them in proportion and leave the slack to the labels. */
        height = Math.min(height, roomBeside);
        const rows = geometry.rows;
        const cellH = (height - HEAD_H - PAD * 2 - GAP * (rows - 1)) / rows;
        const ideal = Math.round(
            geometry.cols * (cellH / CELL_RATIO) + PAD * 2 +
            GAP * (geometry.cols - 1)
        );
        const box = Math.max(MIN_GRID_BOX, Math.min(available, ideal));
        if (box !== available) geometry = { ...layout(box), cols: geometry.cols };
    }

    /* The lift's own box also gains ROW_SPACING, so ask for that much
       more to land the grid on TOP_INSET. */
    const pull = beside && liftY > 0
        ? Math.max(0, liftY - TOP_INSET + ROW_SPACING)
        : 0;
    return { ...geometry, natural: geometry.height, height, beside, pull, elemW };
}

/* ------------------------------------------------------------------ */
/* Picker popover                                                      */
/* ------------------------------------------------------------------ */

let openPopover = null;

function closePopover() {
    if (!openPopover) return;
    openPopover.remove();
    openPopover = null;
    document.removeEventListener("pointerdown", onDocPointerDown, true);
    document.removeEventListener("keydown", onDocKeyDown, true);
}

function onDocPointerDown(event) {
    if (openPopover && !openPopover.contains(event.target)) closePopover();
}

function onDocKeyDown(event) {
    if (event.key === "Escape") {
        event.stopPropagation();
        closePopover();
    }
}

function showPicker(node, index, anchor, onFill) {
    closePopover();
    injectCSS();

    const current = getSlot(node, index);
    const pop = document.createElement("div");
    pop.className = "mlc-pop" + (isLightTheme() ? " mlc-light" : "");

    const head = document.createElement("div");
    head.className = "mlc-pop-head";
    const title = document.createElement("div");
    title.className = "mlc-pop-title";
    title.textContent = `Slot ${index + 1}`;
    head.appendChild(title);
    const foot = document.createElement("div");
    foot.className = "mlc-pop-foot";
    head.appendChild(foot);

    const row = document.createElement("div");
    row.className = "mlc-pop-row";
    const uploadBtn = document.createElement("button");
    uploadBtn.className = "mlc-btn";
    uploadBtn.textContent = "Upload…";
    row.appendChild(uploadBtn);
    if (current && current !== EMPTY) {
        const clearBtn = document.createElement("button");
        clearBtn.className = "mlc-btn mlc-btn-quiet";
        clearBtn.textContent = "Clear";
        clearBtn.addEventListener("click", () => {
            setSlot(node, index, EMPTY);
            closePopover();
        });
        row.appendChild(clearBtn);
    }

    const search = document.createElement("input");
    search.className = "mlc-search";
    search.type = "text";
    search.placeholder = "Search the input folder…";
    search.spellcheck = false;

    const grid = document.createElement("div");
    grid.className = "mlc-pop-grid";

    pop.append(head, row, search, grid);
    document.body.appendChild(pop);

    /* Keep the panel on screen next to the cell. */
    const box = anchor.getBoundingClientRect();
    const width = 322;
    let left = Math.min(
        Math.max(8, box.left), window.innerWidth - width - 8
    );
    const estimated = 340;
    let top = box.bottom + 8;
    if (top + estimated > window.innerHeight) {
        top = Math.max(8, box.top - estimated - 8);
    }
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;

    const LIMIT = 240;

    function render() {
        const term = search.value.trim().toLowerCase();
        const all = knownFiles(node);
        const hits = term
            ? all.filter((f) => f.toLowerCase().includes(term))
            : all;
        grid.replaceChildren();

        for (const value of hits.slice(0, LIMIT)) {
            const option = document.createElement("div");
            option.className =
                "mlc-opt" + (value === current ? " is-current" : "");
            option.title = value;

            const img = document.createElement("img");
            img.loading = "lazy";
            img.src = viewURL(value);
            img.addEventListener("error", () => {
                img.style.visibility = "hidden";
            });

            const name = document.createElement("div");
            name.className = "mlc-opt-name";
            name.textContent = shortName(value);

            option.append(img, name);
            option.addEventListener("click", () => {
                setSlot(node, index, value);
                closePopover();
            });
            grid.appendChild(option);
        }

        if (!hits.length) {
            foot.textContent = all.length ? "no match" : "input folder empty";
        } else if (hits.length > LIMIT) {
            foot.textContent = `${LIMIT} of ${hits.length}`;
        } else {
            foot.textContent = `${hits.length} file${hits.length === 1 ? "" : "s"}`;
        }
    }

    uploadBtn.addEventListener("click", async () => {
        const files = await pickFiles(true);
        if (!files.length) return;
        uploadBtn.classList.add("is-busy");
        uploadBtn.textContent = "Uploading…";
        try {
            await onFill(index, files);
        } finally {
            closePopover();
        }
    });

    search.addEventListener("input", render);
    render();

    openPopover = pop;
    document.addEventListener("pointerdown", onDocPointerDown, true);
    document.addEventListener("keydown", onDocKeyDown, true);
    setTimeout(() => search.focus(), 0);

    /* A fresh listing may add files uploaded from another node. */
    refreshFileList(node).then(() => {
        if (openPopover === pop) render();
    });
}

/* ------------------------------------------------------------------ */
/* The grid widget                                                     */
/* ------------------------------------------------------------------ */

function buildGrid(node) {
    injectCSS();

    const root = document.createElement("div");
    root.className = "mlc-root";

    const panel = document.createElement("div");
    panel.className = "mlc-panel";
    root.appendChild(panel);

    const head = document.createElement("div");
    head.className = "mlc-head";
    const eyebrow = document.createElement("div");
    eyebrow.className = "mlc-eyebrow";
    eyebrow.textContent = "Multi-Load";
    const headRight = document.createElement("div");
    headRight.className = "mlc-head-right";
    const count = document.createElement("div");
    count.className = "mlc-count";
    const sweep = document.createElement("button");
    sweep.className = "mlc-sweep";
    sweep.title = "Clear every slot";
    sweep.innerHTML = ICON_SWEEP;
    headRight.append(count, sweep);
    head.append(eyebrow, headRight);

    const grid = document.createElement("div");
    grid.className = "mlc-grid";

    panel.append(head, grid);

    const cells = [];
    for (let i = 0; i < SLOTS; i++) {
        const cell = document.createElement("div");
        cell.className = "mlc-cell";
        cell.dataset.index = String(i);

        const thumb = document.createElement("img");
        thumb.className = "mlc-thumb";
        thumb.draggable = false;

        const plus = document.createElement("div");
        plus.className = "mlc-plus";
        plus.innerHTML = ICON_PLUS;

        const warn = document.createElement("div");
        warn.className = "mlc-warn";
        warn.innerHTML = ICON_WARN;

        const num = document.createElement("div");
        num.className = "mlc-num";
        num.textContent = String(i + 1);

        const meta = document.createElement("div");
        meta.className = "mlc-meta";

        const clear = document.createElement("button");
        clear.className = "mlc-clear";
        clear.title = "Clear this slot";
        clear.innerHTML = ICON_X;

        cell.append(thumb, plus, warn, num, meta, clear);
        grid.appendChild(cell);
        cells.push({ cell, thumb, meta, clear });
    }

    return { root, panel, grid, cells, count, sweep };
}

/**
 * Upload files and drop them into slots, starting at `start` and using
 * the empty slots after it. Returns when every file has landed.
 */
async function fillFrom(node, start, files, refresh) {
    let cursor = start;
    for (const file of files) {
        while (cursor < SLOTS && cursor !== start &&
               getSlot(node, cursor) !== EMPTY) {
            cursor += 1;
        }
        if (cursor >= SLOTS) break;
        try {
            const value = await uploadFile(file);
            setSlot(node, cursor, value);
        } catch (err) {
            console.error("[MultiLoadCowboy]", err);
            report(`Could not upload ${file.name}: ${err.message}`);
            break;
        }
        cursor += 1;
        refresh();
    }
    refresh();
}

function attachGrid(node) {
    const ui = buildGrid(node);
    const { root, panel, cells, count, sweep } = ui;

    let dragSource = null;
    let lastLayout = null;
    /* Slots whose file would not load. They stay marked until the value
       changes, so a redraw cannot flip them back to a broken thumbnail. */
    const missing = new Array(SLOTS).fill(null);

    function refresh() {
        let filled = 0;
        cells.forEach((parts, i) => {
            const value = getSlot(node, i);
            const isFilled = !!value && value !== EMPTY;
            if (isFilled) filled += 1;
            if (missing[i] && missing[i] !== value) missing[i] = null;
            const isMissing = isFilled && missing[i] === value;

            parts.cell.classList.toggle("is-filled", isFilled && !isMissing);
            parts.cell.classList.toggle("is-missing", isMissing);
            parts.cell.draggable = isFilled && !isMissing;
            parts.cell.title = isMissing
                ? `${shortName(value)} - file is gone from the input folder`
                : isFilled
                    ? `${shortName(value)} - click to change, drag to reorder`
                    : `Slot ${i + 1} - click to load, or drop a file here`;

            const url = isFilled ? viewURL(value) : null;
            if (url && parts.thumb.dataset.src !== url) {
                parts.thumb.dataset.src = url;
                parts.thumb.dataset.value = value;
                parts.thumb.src = url;
            } else if (!isFilled) {
                parts.thumb.removeAttribute("src");
                delete parts.thumb.dataset.src;
            }
            parts.meta.textContent = isFilled ? shortName(value) : "";
        });

        count.textContent = `${filled} / ${SLOTS}`;
        sweep.disabled = filled === 0;
        const light = isLightTheme();
        root.classList.toggle("mlc-light", light);
        panel.classList.toggle("mlc-light", light);
    }

    /* --- per cell wiring --- */

    cells.forEach((parts, index) => {
        const { cell, thumb, clear } = parts;

        thumb.addEventListener("error", () => {
            const value = thumb.dataset.value;
            if (value && value === getSlot(node, index)) {
                missing[index] = value;
                refresh();
            }
        });
        thumb.addEventListener("load", () => {
            if (missing[index]) {
                missing[index] = null;
                refresh();
            }
        });

        cell.addEventListener("pointerdown", (event) => {
            /* Keep the canvas from starting a node drag under us. */
            event.stopPropagation();
        });

        cell.addEventListener("click", (event) => {
            event.stopPropagation();
            showPicker(node, index, cell, async (start, files) => {
                await fillFrom(node, start, files, refresh);
            });
        });

        clear.addEventListener("click", (event) => {
            event.stopPropagation();
            event.preventDefault();
            setSlot(node, index, EMPTY);
            refresh();
        });
        clear.addEventListener("pointerdown", (e) => e.stopPropagation());

        cell.addEventListener("dragstart", (event) => {
            if (getSlot(node, index) === EMPTY) {
                event.preventDefault();
                return;
            }
            dragSource = index;
            cell.classList.add("is-dragging");
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/mlc-slot", String(index));
        });

        cell.addEventListener("dragend", () => {
            dragSource = null;
            cell.classList.remove("is-dragging");
        });

        cell.addEventListener("dragover", (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect =
                dragSource === null ? "copy" : "move";
            if (dragSource !== index) cell.classList.add("is-drop");
        });

        cell.addEventListener("dragleave", () => {
            cell.classList.remove("is-drop");
        });

        cell.addEventListener("drop", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            cell.classList.remove("is-drop");

            if (dragSource !== null && dragSource !== index) {
                /* Swap two slots. */
                const from = getSlot(node, dragSource);
                const to = getSlot(node, index);
                setSlot(node, index, from);
                setSlot(node, dragSource, to);
                dragSource = null;
                refresh();
                return;
            }

            const files = Array.from(event.dataTransfer?.files || [])
                .filter((f) => f.type.startsWith("image/"));
            if (files.length) {
                await fillFrom(node, index, files, refresh);
            }
        });
    });

    /* --- whole panel --- */

    sweep.addEventListener("pointerdown", (e) => e.stopPropagation());
    sweep.addEventListener("click", (event) => {
        event.stopPropagation();
        for (let i = 0; i < SLOTS; i++) setSlot(node, i, EMPTY);
        refresh();
    });

    root.addEventListener("dragover", (event) => {
        if (dragSource !== null) return;
        event.preventDefault();
        event.stopPropagation();
    });

    root.addEventListener("drop", async (event) => {
        if (dragSource !== null) return;
        event.preventDefault();
        event.stopPropagation();
        const files = Array.from(event.dataTransfer?.files || [])
            .filter((f) => f.type.startsWith("image/"));
        if (!files.length) return;
        let start = 0;
        while (start < SLOTS && getSlot(node, start) !== EMPTY) start += 1;
        if (start >= SLOTS) start = 0;
        await fillFrom(node, start, files, refresh);
    });

    /* --- widget --- */

    /* Where the lift lands naturally: after every real widget. Measured
       once the frontend has laid out, estimated before that. */
    function liftY() {
        if (spacer && typeof spacer.y === "number" && spacer.y > 0) {
            return spacer.y;
        }
        const band = (node.outputs?.length || 0) * SLOT_H;
        const rows = (node.widgets || []).filter(
            (w) => !w.hidden && w !== spacer && w !== widget
        ).length;
        return band + rows * 24 + 6;
    }

    /* Top of the first real widget below the band: the line the grid
       must stay above. Measured once laid out, estimated before that. */
    function settingsTop() {
        let top = Infinity;
        for (const w of node.widgets || []) {
            if (w === spacer || w === widget || w.hidden) continue;
            if (typeof w.y === "number" && w.y > 0) top = Math.min(top, w.y);
        }
        return Number.isFinite(top) ? top : 0;
    }

    /* The panel is measured from the widget box the frontend hands us,
       never from our own guess at its margins, so nothing gets clipped. */
    function currentPlan() {
        return plan(node, root.clientWidth || 0, liftY(), settingsTop());
    }

    /** Height to ask for: the panel plus the frontend's own margins. */
    function requestHeight() {
        return currentPlan().height + DOM_MARGIN * 2;
    }

    function applyLayout() {
        /* The wrappers appear when the elements mount, so keep asking. */
        releaseWrapper(root);
        releaseWrapper(spacerEl);

        const next = currentPlan();
        if (lastLayout &&
            next.cols === lastLayout.cols &&
            next.box === lastLayout.box &&
            next.beside === lastLayout.beside) {
            return next;
        }
        panel.style.width = `${next.box}px`;
        panel.style.marginRight = next.beside ? "auto" : "0";
        panel.style.setProperty("--mlc-cols", String(next.cols));
        panel.style.setProperty("--mlc-rows", String(next.rows));
        lastLayout = next;
        return next;
    }

    let spacer;
    let widget;

    const spacerEl = document.createElement("div");
    spacerEl.className = "mlc-spacer";
    spacerEl.style.pointerEvents = "none";

    /* A negative height moves everything after it up, which lifts the
       grid into the empty band beside the output labels. */
    spacer = node.addDOMWidget("multi_load_lift", "mlc_lift", spacerEl, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => -currentPlan().pull,
        getMaxHeight: () => -currentPlan().pull,
    });
    spacer.serialize = false;
    spacer.computeSize = () => [0, -currentPlan().pull];
    spacer.computeLayoutSize = () => {
        const pull = -currentPlan().pull;
        return { minHeight: pull, maxHeight: pull, minWidth: 0 };
    };

    widget = node.addDOMWidget("multi_load_grid", "mlc_grid", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => requestHeight(),
        getMaxHeight: () => requestHeight(),
    });
    /* addDOMWidget only records the option, and the serialiser reads the
       flag off the widget itself. Core's audio widget sets both. */
    widget.serialize = false;

    /*
     * Both of ours stay at the END of node.widgets, and nothing may be
     * inserted in front of a real widget. The frontend saves widget
     * values by absolute index, leaving a hole for anything with
     * serialize === false, but loads them compacted - it walks the
     * serializable widgets and consumes the array in order. Those two
     * only agree while every non-serializable widget sits last;
     * anywhere else, one reload shifts every value along by one and a
     * number widget ends up holding a string.
     */

    widget.computeSize = function (width) {
        applyLayout();
        return [width, requestHeight()];
    };

    /* Newer Vue node rendering asks for the size a different way. */
    widget.computeLayoutSize = function () {
        applyLayout();
        const height = requestHeight();
        return { minHeight: height, maxHeight: height, minWidth: 0 };
    };

    /**
     * Bottom of the laid-out content, read from the frontend's own
     * numbers rather than re-derived. Falls back to litegraph's estimate
     * before the first layout has run.
     */
    function contentBottom() {
        let bottom = 0;
        let measured = false;
        for (const w of node.widgets || []) {
            if (w.hidden) continue;
            const y = typeof w.y === "number" ? w.y : 0;
            const h = typeof w.computedHeight === "number"
                ? w.computedHeight : 0;
            if (y > 0 || h > 0) measured = true;
            bottom = Math.max(bottom, y + h);
        }
        if (!measured) {
            return node.computeSize ? node.computeSize()[1] : null;
        }
        const band = (node.outputs?.length || 0) * SLOT_H + TOP_INSET;
        return Math.max(bottom, band) + 8;
    }

    /* Trim the node to what the layout actually uses. Lifting the grid
       leaves litegraph's own estimate too tall. */
    function fitHeight() {
        const want = contentBottom();
        if (want && Math.abs((node.size?.[1] || 0) - want) > 2) {
            if (typeof node.setSize === "function") {
                node.setSize([node.size[0], want]);
            } else {
                node.size[1] = want;
            }
            node.graph?.setDirtyCanvas(true, true);
        }
    }

    /* The element only gets a real width once it is mounted; re-measure
       then, and whenever the node is resized. */
    if (typeof ResizeObserver === "function") {
        let lastWidth = 0;
        const observer = new ResizeObserver(() => {
            const width = root.clientWidth;
            if (!width || width === lastWidth) return;
            lastWidth = width;
            applyLayout();
            node.graph?.setDirtyCanvas(true, true);
            requestAnimationFrame(fitHeight);
        });
        observer.observe(root);
        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            observer.disconnect();
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    }

    applyLayout();
    widget.mlcRefresh = refresh;
    node.mlcContentBottom = contentBottom;
    return { widget, spacer, refresh, currentPlan, fitHeight, contentBottom };
}

/**
 * Repair widgets_values saved by a build that put the grid widgets in
 * front of the real ones.
 *
 * Those saves carry a leading null per non-serializable widget, while
 * the loader expects a compacted array, so every value lands one or two
 * widgets late. Dropping the leading nulls until the array matches the
 * widget count puts them back where they belong. Only leading nulls go:
 * none of this node's widgets ever holds a null of its own.
 */
function repairWidgetValues(node, info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values)) return 0;

    const expected = (node.widgets || [])
        .filter((w) => w.serialize !== false).length;
    if (!expected || values.length <= expected) return 0;

    let dropped = 0;
    while (values.length > expected &&
           (values[0] === null || values[0] === undefined)) {
        values.shift();
        dropped += 1;
    }
    if (dropped) {
        console.info(
            `[MultiLoadCowboy] repaired ${dropped} shifted widget ` +
            "value(s) from an older save"
        );
    }
    return dropped;
}

/**
 * What each widget is allowed to hold, read from the node definition.
 */
function collectSpecs(nodeData) {
    const specs = {};
    for (const group of [nodeData?.input?.required, nodeData?.input?.optional]) {
        for (const [name, entry] of Object.entries(group || {})) {
            const [type, config = {}] = Array.isArray(entry) ? entry : [entry, {}];
            specs[name] = {
                kind: Array.isArray(type) ? "combo"
                    : (type === "INT" || type === "FLOAT") ? "number" : "string",
                values: Array.isArray(type) ? type : null,
                fallback: config.default,
            };
        }
    }
    return specs;
}

/**
 * Put back anything holding the wrong kind of value.
 *
 * A build that let saved values shift along could leave a combo string
 * in width or height. A number widget in that state cannot be dragged or
 * typed into at all - the first click turns it into undefined - so a
 * node damaged that way has to be healed rather than left to the user.
 *
 * Image slots are only healed when they hold something that is not a
 * string: a name that is simply missing from the input folder is kept,
 * because the grid already shows that as a missing tile.
 */
function healWidgetValues(node, specs) {
    const healed = [];
    for (const widget of node.widgets || []) {
        const spec = specs[widget.name];
        if (!spec || widget.serialize === false) continue;
        const value = widget.value;

        if (spec.kind === "number") {
            if (typeof value === "number" && Number.isFinite(value)) continue;
            const asNumber = typeof value === "string" && value.trim() !== ""
                ? Number(value) : NaN;
            widget.value = Number.isFinite(asNumber)
                ? asNumber
                : (spec.fallback ?? 0);
        } else if (SLOT_NAMES.includes(widget.name)) {
            if (typeof value === "string" && value !== "") continue;
            widget.value = EMPTY;
        } else if (spec.kind === "combo") {
            const values = Array.isArray(widget.options?.values)
                ? widget.options.values : spec.values;
            if (!values || values.includes(value)) continue;
            widget.value = spec.fallback ?? values[0];
        } else {
            if (typeof value === "string") continue;
            widget.value = spec.fallback ?? "";
        }
        healed.push(`${widget.name}=${JSON.stringify(value)}`);
    }

    if (healed.length) {
        console.info(
            "[MultiLoadCowboy] reset widgets holding the wrong kind of " +
            `value: ${healed.join(", ")}`
        );
    }
    return healed;
}

/* ------------------------------------------------------------------ */
/* Extension                                                           */
/* ------------------------------------------------------------------ */

app.registerExtension({
    name: "TrentNodes.MultiLoadCowboy",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const specs = collectSpecs(nodeData);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;

            const node = this;
            const combos = slotWidgets(node);
            if (combos.some((w) => !w)) {
                console.warn(
                    "[MultiLoadCowboy] slot widgets missing; leaving the " +
                    "stock combos in place"
                );
                return result;
            }

            /* Hide the combos: the grid is their face now. They stay in
               node.widgets so serialisation and the API are untouched. */
            for (const combo of combos) {
                combo.origType = combo.type;
                combo.origComputeSize = combo.computeSize;
                combo.type = "converted-widget-mlc";
                combo.hidden = true;
                combo.computeSize = () => [0, -4];
            }

            const { refresh, fitHeight } = attachGrid(node);
            node.mlcRefresh = refresh;
            node.mlcFitHeight = fitHeight;

            /* Redraw when anything else writes to a slot. */
            for (const combo of combos) {
                const original = combo.callback;
                combo.callback = function (value) {
                    const out = original
                        ? original.apply(this, arguments)
                        : undefined;
                    refresh();
                    return out;
                };
            }

            /* Never let a resize crush the grid out of view. Growing is
               the user's business; shrinking past the content is not. */
            const onResize = node.onResize;
            node.onResize = function (size) {
                const out = onResize ? onResize.apply(this, arguments) : undefined;
                const smallest = this.mlcContentBottom?.();
                if (smallest && size[1] < smallest) size[1] = smallest;
                return out;
            };

            /* Wide enough that the grid and the output labels can share
               the band without the cells getting cramped. */
            node.size[0] = Math.max(node.size[0] || 0, 520);
            node.size[1] = Math.max(node.size[1] || 0, node.computeSize()[1]);
            refresh();
            /* Two frames: one for the element to mount, one for the
               frontend to lay the widgets out. */
            if (typeof requestAnimationFrame === "function") {
                requestAnimationFrame(() => requestAnimationFrame(fitHeight));
            }
            return result;
        };

        /* Litegraph draws hidden widgets when their type is null, so
           blank the marker type for the duration of the draw. */
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const hidden = (this.widgets || []).filter(
                (w) => typeof w.type === "string" &&
                    w.type.includes("converted-widget")
            );
            const saved = hidden.map((w) => w.type);
            hidden.forEach((w) => (w.type = null));
            const result = onDrawForeground
                ? onDrawForeground.apply(this, arguments)
                : undefined;
            hidden.forEach((w, i) => (w.type = saved[i]));
            return result;
        };

        /* configure() applies widgets_values, so the repair has to run
           before it, not in onConfigure which fires afterwards. */
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            try {
                repairWidgetValues(this, info);
            } catch (err) {
                console.error("[MultiLoadCowboy] value repair failed", err);
            }
            const result = configure
                ? configure.apply(this, arguments)
                : undefined;
            try {
                if (healWidgetValues(this, specs).length) {
                    this.mlcRefresh?.();
                }
            } catch (err) {
                console.error("[MultiLoadCowboy] value heal failed", err);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure
                ? onConfigure.apply(this, arguments)
                : undefined;
            /* Widget values land after configure returns. */
            setTimeout(() => {
                try {
                    healWidgetValues(this, specs);
                } catch (err) {
                    console.error("[MultiLoadCowboy] value heal failed", err);
                }
                this.mlcRefresh?.();
                this.mlcFitHeight?.();
            }, 0);
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            closePopover();
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };
    },
});
