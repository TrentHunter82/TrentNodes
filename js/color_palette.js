import { app } from "../../scripts/app.js";

/*
 * Trent Color Palette
 * -------------------
 * A docked ComfyUI sidebar tab holding 7 preset color swatches.
 *   - Left-click a swatch  -> color all currently-selected node(s) AND group(s)
 *   - Right-click a swatch -> pick a new preset color for that slot
 *   - "Reset" chip         -> strip selected node(s)/group(s) back to default
 *   - "🪣 paint" toggle     -> arm a swatch, then click any node/group to paint it
 *   - Drop an image on the panel -> auto-fill the 7 swatches with its dominant
 *                                   colors (one-step "undo import" to revert)
 * Presets are also synced into LiteGraph's right-click "Colors" menu.
 * Show/hide/dock is handled natively by ComfyUI's sidebar (icon rail, or the
 * "🎨 Color Palette" command). Colors persist in localStorage; the UI is built
 * once and re-mounted on each sidebar render so paint/undo state survives toggles.
 */

const TAB_ID = "trent-color-palette";
const LS_COLORS = "TrentPalette.colors";
const LS_PREV = "TrentPalette.colors.prev"; // one-step undo backup for image imports

const N_COLORS = 7;            // palette size (must match DEFAULT_COLORS length)
const SAMPLE_MAX_EDGE = 128;   // downscale dropped images so the longest edge is ~this many px
const ALPHA_MIN = 16;          // ignore near-transparent pixels when quantizing
const IMAGE_RE = /\.(?:png|jpe?g|webp|bmp|gif|avif|tiff?)$/i;

// Default 7 presets (title-bar colors). Right-click any swatch to change them.
const DEFAULT_COLORS = [
    "#e0457b", // pink/red
    "#e08a45", // orange
    "#e0c945", // yellow
    "#5cb85c", // green
    "#45b6e0", // cyan/blue
    "#7c5ce0", // purple
    "#8a939b", // slate/gray
];

// ---- helpers ---------------------------------------------------------------

function loadColors() {
    try {
        const raw = localStorage.getItem(LS_COLORS);
        if (raw) {
            const arr = JSON.parse(raw);
            if (Array.isArray(arr) && arr.length === 7) return arr;
        }
    } catch (e) { /* fall through */ }
    return [...DEFAULT_COLORS];
}

function saveColors(colors) {
    localStorage.setItem(LS_COLORS, JSON.stringify(colors));
}

// Derive a darker body color from the (brighter) title color.
function darken(hex, factor = 0.42) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    const r = Math.round(((n >> 16) & 255) * factor);
    const g = Math.round(((n >> 8) & 255) * factor);
    const b = Math.round((n & 255) * factor);
    return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
}

// Lighten a color by raising HSL lightness `amount` of the way toward white,
// keeping hue and saturation fixed (lighter, not more saturated). Used to give
// group backgrounds a softer tint than the node title color painted alongside.
function lighten(hex, amount = 0.15) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    const { h, s, l } = rgbToHsl([(n >> 16) & 255, (n >> 8) & 255, n & 255]);
    return rgbToHex(hslToRgb(h, s, l + (1 - l) * amount));
}

// ---- color extraction (drag an image -> dominant palette) -------------------

function rgbToHex([r, g, b]) {
    return "#" + [r, g, b].map((v) => Math.max(0, Math.min(255, v)).toString(16).padStart(2, "0")).join("");
}

// Standard RGB -> HSL. h:0..360, s/l:0..1. Used only for pleasant swatch ordering.
function rgbToHsl([r, g, b]) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const l = (max + min) / 2;
    let h = 0, s = 0;
    if (max !== min) {
        const d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h *= 60;
    }
    return { h, s, l };
}

// Inverse of rgbToHsl. h:0..360, s/l:0..1 -> [r,g,b] each 0..255.
function hslToRgb(h, s, l) {
    h = ((h % 360) + 360) % 360 / 360;
    s = Math.max(0, Math.min(1, s));
    l = Math.max(0, Math.min(1, l));
    if (s === 0) { const v = Math.round(l * 255); return [v, v, v]; }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const hue = (t) => {
        if (t < 0) t += 1; else if (t > 1) t -= 1;
        if (t < 1 / 6) return p + (q - p) * 6 * t;
        if (t < 1 / 2) return q;
        if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
        return p;
    };
    return [hue(h + 1 / 3), hue(h), hue(h - 1 / 3)].map((v) => Math.round(v * 255));
}

function dedupe(pixels) {
    const seen = new Set(), out = [];
    for (const p of pixels) {
        const k = (p[0] << 16) | (p[1] << 8) | p[2];
        if (!seen.has(k)) { seen.add(k); out.push(p); }
    }
    return out;
}

function sortSlice(arr, start, end, ch) {
    const sub = arr.slice(start, end).sort((a, b) => a[ch] - b[ch]);
    for (let i = 0; i < sub.length; i++) arr[start + i] = sub[i];
}

/*
 * Median-cut quantization. Repeatedly splits the box with the most pixels
 * along its widest RGB channel at the median, until we have `n` boxes, then
 * averages each box to one representative color. Guards on count/range make
 * near-grayscale, flat-logo and few-color images terminate early (fewer than
 * n boxes -> caller pads up to n). Count-priority gives a balanced "dominant
 * colors" spread; volume-priority would favor vivid minority colors instead.
 */
function medianCut(pixels, n) {
    if (pixels.length <= n) return dedupe(pixels).slice(0, n);

    const makeBox = (start, end) => {
        let rMin = 255, rMax = 0, gMin = 255, gMax = 0, bMin = 255, bMax = 0;
        for (let i = start; i < end; i++) {
            const p = pixels[i];
            if (p[0] < rMin) rMin = p[0]; if (p[0] > rMax) rMax = p[0];
            if (p[1] < gMin) gMin = p[1]; if (p[1] > gMax) gMax = p[1];
            if (p[2] < bMin) bMin = p[2]; if (p[2] > bMax) bMax = p[2];
        }
        return { start, end, ranges: [rMax - rMin, gMax - gMin, bMax - bMin], count: end - start };
    };

    let boxes = [makeBox(0, pixels.length)];

    while (boxes.length < n) {
        let target = null, ti = -1;
        for (let i = 0; i < boxes.length; i++) {
            const b = boxes[i];
            if (b.count < 2) continue;                  // a single pixel can't be split
            if (Math.max(...b.ranges) === 0) continue;  // all identical -> unsplittable
            if (!target || b.count > target.count) { target = b; ti = i; }
        }
        if (!target) break;                             // nothing left to split

        const ch = target.ranges[0] >= target.ranges[1] && target.ranges[0] >= target.ranges[2] ? 0
            : target.ranges[1] >= target.ranges[2] ? 1 : 2;
        sortSlice(pixels, target.start, target.end, ch);
        const mid = target.start + (target.count >> 1);
        boxes.splice(ti, 1, makeBox(target.start, mid), makeBox(mid, target.end));
    }

    return boxes.map((b) => {
        let r = 0, g = 0, bl = 0;
        for (let i = b.start; i < b.end; i++) { r += pixels[i][0]; g += pixels[i][1]; bl += pixels[i][2]; }
        const c = b.count || 1;
        return [Math.round(r / c), Math.round(g / c), Math.round(bl / c)];
    });
}

// Guarantee exactly `n` colors: when the image yields fewer, cycle the found
// colors with slight luminance steps so the row reads as "from this image".
function padColors(colors, n) {
    if (colors.length >= n) return colors.slice(0, n);
    if (colors.length === 0) return DEFAULT_COLORS.map((h) => {
        const m = /^#?([0-9a-f]{6})$/i.exec(h);
        const v = parseInt(m[1], 16);
        return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
    });
    const out = colors.slice();
    let i = 0;
    while (out.length < n) {
        const base = colors[i % colors.length];
        const f = 0.78 + 0.16 * Math.floor(i / colors.length);
        out.push(base.map((v) => Math.max(0, Math.min(255, Math.round(v * f)))));
        i++;
    }
    return out;
}

// Order for a pleasant swatch strip: chromatic colors around the hue wheel,
// near-gray colors (unstable hue) pinned to the end dark->light. This trades
// away dominance ordering for a designed-strip look.
function orderColors(colors) {
    const withHsl = colors.map((c) => ({ c, ...rgbToHsl(c) }));
    const chromatic = withHsl.filter((x) => x.s > 0.12);
    const grays = withHsl.filter((x) => x.s <= 0.12);
    chromatic.sort((a, b) => a.h - b.h);
    grays.sort((a, b) => a.l - b.l);
    return [...chromatic, ...grays].map((x) => x.c);
}

/*
 * Load an image source into a downscaled offscreen canvas and return its RGBA
 * pixel data. Rejects on decode failure or a tainted canvas (cross-origin
 * without CORS headers -> getImageData throws SecurityError) — that single
 * throw site is how the caller decides to keep the current palette.
 */
function loadImagePixels(src, { crossOrigin = false } = {}) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        if (crossOrigin) img.crossOrigin = "anonymous";
        img.onload = () => {
            try {
                const w0 = img.naturalWidth, h0 = img.naturalHeight;
                if (!w0 || !h0) { reject(new Error("empty image")); return; }
                const scale = Math.min(1, SAMPLE_MAX_EDGE / Math.max(w0, h0));
                const w = Math.max(1, Math.round(w0 * scale));
                const h = Math.max(1, Math.round(h0 * scale));
                const cv = document.createElement("canvas");
                cv.width = w; cv.height = h;
                const ctx = cv.getContext("2d", { willReadFrequently: true });
                ctx.drawImage(img, 0, 0, w, h);
                resolve(ctx.getImageData(0, 0, w, h).data);
            } catch (err) {
                reject(err);
            }
        };
        img.onerror = () => reject(new Error("decode failed"));
        img.src = src;
    });
}

function getSelectedNodes() {
    const canvas = app.canvas;
    if (!canvas || !canvas.selected_nodes) return [];
    return Object.values(canvas.selected_nodes);
}

// Collect selected groups robustly across LiteGraph versions (mirrors the
// helper in organize_group_grid.js): selected_groups may be a Set, Array or
// object; selected_group is the legacy single; selectedItems holds both nodes
// and groups, so filter to LGraphGroup instances.
function getSelectedGroups() {
    const canvas = app.canvas;
    if (!canvas) return [];
    const out = [];
    const seen = new Set();
    const push = (g) => { if (g && !seen.has(g)) { seen.add(g); out.push(g); } };

    if (canvas.selected_groups) {
        if (canvas.selected_groups instanceof Set || Array.isArray(canvas.selected_groups)) {
            for (const g of canvas.selected_groups) push(g);
        } else if (typeof canvas.selected_groups === "object") {
            for (const k of Object.keys(canvas.selected_groups)) push(canvas.selected_groups[k]);
        }
    }
    push(canvas.selected_group);

    if (canvas.selectedItems) {
        for (const item of canvas.selectedItems) {
            if (window.LiteGraph && item instanceof window.LiteGraph.LGraphGroup) push(item);
        }
    }
    return out;
}

function applyColorToSelection(hex) {
    const nodes = getSelectedNodes();
    const groups = getSelectedGroups();
    if (!nodes.length && !groups.length) {
        flash("Select a node or group first");
        return;
    }
    const body = darken(hex);
    for (const node of nodes) {
        node.color = hex;
        node.bgcolor = body;
    }
    const groupHex = lighten(hex); // ~15% lighter tint than the node title color
    for (const group of groups) {
        group.color = groupHex; // groups carry a single color (no bgcolor)
    }
    app.graph.change();
}

function resetSelection() {
    const nodes = getSelectedNodes();
    const groups = getSelectedGroups();
    if (!nodes.length && !groups.length) {
        flash("Select a node or group first");
        return;
    }
    for (const node of nodes) {
        delete node.color;
        delete node.bgcolor;
    }
    for (const group of groups) {
        delete group.color;
    }
    app.graph.change();
}

// ---- paint-bucket mode -----------------------------------------------------
// `bucketArmed` = paint mode is on (toggle button); `armedHex` = a swatch color
// has been picked. The canvas patch only paints when BOTH are set. The UI-aware
// disarm is installed by buildPaletteBody; this default just clears state.
let bucketArmed = false;
let armedHex = null;
let disarmBucket = () => { bucketArmed = false; armedHex = null; };

// Permanently wrap LGraphCanvas.processMouseDown. When armed and a plain
// left-click lands on a node or group, paint it and consume the event so no
// drag/box-select/pan starts. When not armed it's a transparent pass-through.
function installBucketPatch() {
    const LGC = window.LGraphCanvas;
    if (!LGC || !LGC.prototype || LGC.prototype.__trentBucketPatched) return;
    const proto = LGC.prototype;
    const orig = proto.processMouseDown;
    proto.processMouseDown = function (e) {
        if (bucketArmed && armedHex && e.button === 0 && !e.ctrlKey && !e.shiftKey && !e.altKey) {
            let x = e.canvasX, y = e.canvasY;
            if (x === undefined || y === undefined) {
                const p = this.convertEventToCanvasOffset(e);
                x = p[0]; y = p[1];
            }
            const graph = this.graph;
            const node = graph && graph.getNodeOnPos ? graph.getNodeOnPos(x, y) : null;
            const group = (!node && graph && graph.getGroupOnPos) ? graph.getGroupOnPos(x, y) : null;
            if (node) {
                node.color = armedHex;
                node.bgcolor = darken(armedHex);
                if (graph) graph.change();
                e.preventDefault(); e.stopPropagation();
                return false;
            }
            if (group) {
                group.color = lighten(armedHex);
                if (graph) graph.change();
                e.preventDefault(); e.stopPropagation();
                return false;
            }
            // armed but clicked empty canvas -> fall through (pan/deselect as normal)
        }
        return orig.apply(this, arguments);
    };
    proto.__trentBucketPatched = true;
}

// One page-lifetime Escape handler that disarms paint mode.
function installEscDisarm() {
    if (window.__trentBucketEsc) return;
    window.addEventListener("keydown", (e) => {
        if (bucketArmed && e.key === "Escape") disarmBucket();
    });
    window.__trentBucketEsc = true;
}

// Sync the 7 presets into the native right-click "Colors" submenu.
function syncContextMenuColors(colors) {
    const LGC = window.LGraphCanvas;
    if (!LGC) return;
    LGC.node_colors = LGC.node_colors || {};
    colors.forEach((hex, i) => {
        LGC.node_colors["Trent " + (i + 1)] = {
            color: hex,
            bgcolor: darken(hex),
            groupcolor: hex,
        };
    });
}

// tiny transient toast on the panel
let flashTimer = null;
function flash(msg) {
    const el = document.getElementById("trent-palette-hint");
    if (!el) return;
    el.textContent = msg;
    el.style.opacity = "1";
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => { el.style.opacity = "0.45"; }, 1400);
}

// ---- panel UI --------------------------------------------------------------

// Build the palette UI once into a detached container and return it. ComfyUI's
// sidebar render() re-parents this same element on every open, so all event
// handlers, the `colors` closures, and paint/undo state survive tab toggles.
function buildPaletteBody() {
    const colors = loadColors();
    syncContextMenuColors(colors);

    const root = document.createElement("div");
    root.id = "trent-palette";
    Object.assign(root.style, {
        boxSizing: "border-box",
        padding: "10px",
        font: "12px sans-serif",
        color: "#cdd6dd",
        userSelect: "none",
        width: "100%",
    });

    // body content mounts directly into root (the sidebar frame supplies the
    // title/border/scroll; no floating chrome, drag handle, or collapse here).
    const body = root;

    const swatchRow = document.createElement("div");
    Object.assign(swatchRow.style, { display: "flex", gap: "6px", flexWrap: "wrap" });

    const swatchEls = []; // swatch divs, parallel to `colors`, for bulk restyling

    function refreshSwatchStyle(sw, hex) {
        sw.style.background = hex;
        sw.title = `Left-click: apply color (or arm it, in paint mode)\nRight-click: edit this color (${hex})`;
    }

    // Outline the armed swatch (idx) in paint mode; pass -1 to clear all.
    function highlightArmedSwatch(idx) {
        swatchEls.forEach((sw, j) => {
            sw.style.outline = (j === idx) ? "2px solid #fff" : "none";
            sw.style.outlineOffset = (j === idx) ? "1px" : "0";
        });
    }

    colors.forEach((hex, i) => {
        const sw = document.createElement("div");
        Object.assign(sw.style, {
            width: "26px", height: "26px", borderRadius: "5px",
            cursor: "pointer", border: "1px solid rgba(255,255,255,0.15)",
            transition: "transform 0.08s",
        });
        refreshSwatchStyle(sw, hex);
        swatchEls.push(sw);
        sw.addEventListener("mouseenter", () => { sw.style.transform = "scale(1.12)"; });
        sw.addEventListener("mouseleave", () => { sw.style.transform = "scale(1)"; });

        // left-click -> in paint mode arm this color, otherwise apply to selection
        sw.addEventListener("click", () => {
            if (bucketArmed) {
                armedHex = colors[i];
                highlightArmedSwatch(i);
                flash("Armed — click a node or group");
            } else {
                applyColorToSelection(colors[i]);
            }
        });

        // right-click -> edit via hidden native color picker
        sw.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            const picker = document.createElement("input");
            picker.type = "color";
            picker.value = colors[i];
            Object.assign(picker.style, { position: "fixed", left: "-9999px" });
            document.body.appendChild(picker);
            picker.addEventListener("input", () => {
                colors[i] = picker.value;
                refreshSwatchStyle(sw, colors[i]);
            });
            picker.addEventListener("change", () => {
                colors[i] = picker.value;
                refreshSwatchStyle(sw, colors[i]);
                saveColors(colors);
                syncContextMenuColors(colors);
                flash(`Swatch ${i + 1} set`);
                picker.remove();
            });
            picker.click();
        });

        swatchRow.appendChild(sw);
    });
    body.appendChild(swatchRow);

    // footer row: reset chip + hint
    const footer = document.createElement("div");
    Object.assign(footer.style, {
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginTop: "8px", gap: "8px", flexWrap: "wrap",
    });
    const resetChip = document.createElement("div");
    resetChip.textContent = "✕ reset";
    Object.assign(resetChip.style, {
        cursor: "pointer", padding: "3px 8px", borderRadius: "5px",
        background: "#1c272e", border: "1px solid #2a3942", fontSize: "11px",
    });
    resetChip.title = "Remove custom color from selected node(s)";
    resetChip.addEventListener("click", resetSelection);
    resetChip.addEventListener("mouseenter", () => { resetChip.style.background = "#26343d"; });
    resetChip.addEventListener("mouseleave", () => { resetChip.style.background = "#1c272e"; });

    // paint-bucket toggle: arm a color, then click nodes/groups to paint them
    const paintChip = document.createElement("div");
    paintChip.textContent = "🪣 paint";
    Object.assign(paintChip.style, {
        cursor: "pointer", padding: "3px 8px", borderRadius: "5px",
        background: "#1c272e", border: "1px solid #2a3942", fontSize: "11px",
    });
    paintChip.title = "Paint mode: click a swatch to arm, then click any node or group to color it (Esc to stop)";

    function setPaintMode(on) {
        bucketArmed = on;
        if (!on) { armedHex = null; highlightArmedSwatch(-1); }
        paintChip.style.background = on ? "#274b57" : "#1c272e";
        paintChip.style.borderColor = on ? "#45b6e0" : "#2a3942";
        const cv = app.canvas && app.canvas.canvas;
        if (cv) cv.style.cursor = on ? "crosshair" : "";
        if (on) flash("Paint mode — click a swatch to arm");
    }
    disarmBucket = () => setPaintMode(false); // let Esc/patch reach the UI-aware version
    paintChip.addEventListener("click", () => setPaintMode(!bucketArmed));
    paintChip.addEventListener("mouseenter", () => { if (!bucketArmed) paintChip.style.background = "#26343d"; });
    paintChip.addEventListener("mouseleave", () => { if (!bucketArmed) paintChip.style.background = "#1c272e"; });

    // undo chip for image imports (hidden until an import happens)
    const undoChip = document.createElement("div");
    undoChip.textContent = "↶ undo import";
    Object.assign(undoChip.style, {
        cursor: "pointer", padding: "3px 8px", borderRadius: "5px",
        background: "#1c272e", border: "1px solid #45b6e0", fontSize: "11px",
        display: "none",
    });
    undoChip.addEventListener("mouseenter", () => { undoChip.style.background = "#26343d"; });
    undoChip.addEventListener("mouseleave", () => { undoChip.style.background = "#1c272e"; });

    const hint = document.createElement("span");
    hint.id = "trent-palette-hint";
    hint.textContent = "right-click swatch = edit";
    Object.assign(hint.style, { fontSize: "10px", opacity: "0.45", transition: "opacity 0.3s", marginLeft: "auto" });

    footer.appendChild(resetChip);
    footer.appendChild(paintChip);
    footer.appendChild(undoChip);
    footer.appendChild(hint);
    body.appendChild(footer);

    // ---- image drop -> extracted palette (with one-step undo) ----
    const dropZone = document.createElement("div");
    dropZone.id = "trent-palette-drop";
    dropZone.textContent = "⤓ drop image to extract palette";
    Object.assign(dropZone.style, {
        marginTop: "8px", padding: "6px 8px", borderRadius: "6px",
        border: "1px dashed #2a3942", textAlign: "center", fontSize: "10px",
        color: "#7c8a93", background: "transparent", transition: "all 0.12s",
    });
    body.appendChild(dropZone);

    let previousColors = null;
    let undoTimer = null;
    function showUndoChip() {
        undoChip.style.display = "block";
        clearTimeout(undoTimer);
        undoTimer = setTimeout(hideUndoChip, 10000);
    }
    function hideUndoChip() {
        clearTimeout(undoTimer);
        undoChip.style.display = "none";
    }
    // Rehydrate a pending undo from a prior session (accidental import + reload).
    try {
        const rawPrev = localStorage.getItem(LS_PREV);
        if (rawPrev) {
            const arr = JSON.parse(rawPrev);
            if (Array.isArray(arr) && arr.length === N_COLORS) { previousColors = arr; showUndoChip(); }
        }
    } catch (e) { /* ignore */ }

    function setPalette(hexes) {
        for (let i = 0; i < N_COLORS; i++) {
            colors[i] = hexes[i];               // mutate in place: swatch handlers close over colors[i]
            refreshSwatchStyle(swatchEls[i], colors[i]);
        }
        saveColors(colors);
        syncContextMenuColors(colors);
    }
    function applyImportedPalette(hexes) {
        previousColors = colors.slice();
        try { localStorage.setItem(LS_PREV, JSON.stringify(previousColors)); } catch (e) {}
        setPalette(hexes);
        showUndoChip();
        flash("Palette imported");
    }
    function restorePalette() {
        if (!previousColors || previousColors.length !== N_COLORS) return;
        setPalette(previousColors.slice());
        previousColors = null;
        try { localStorage.removeItem(LS_PREV); } catch (e) {}
        hideUndoChip();
        flash("Palette restored");
    }
    undoChip.addEventListener("click", restorePalette);

    // Quantize collected pixel data into N colors and apply them.
    function finishExtraction(rgba) {
        const px = [];
        for (let i = 0; i < rgba.length; i += 4) {
            if (rgba[i + 3] < ALPHA_MIN) continue;
            px.push([rgba[i], rgba[i + 1], rgba[i + 2]]);
        }
        if (px.length === 0) { flash("Image is fully transparent"); return; }
        const result = orderColors(padColors(medianCut(px, N_COLORS), N_COLORS));
        applyImportedPalette(result.map(rgbToHex));
    }
    async function extractFromFile(file) {
        const url = URL.createObjectURL(file);
        try {
            finishExtraction(await loadImagePixels(url)); // blob: URL is same-origin, never tainted
        } catch (err) {
            flash("Could not read image");
            console.warn("[TrentPalette]", err);
        } finally {
            URL.revokeObjectURL(url);
        }
    }
    async function extractFromUrl(uri) {
        const sameOrigin = uri.startsWith("data:") || uri.startsWith("blob:") || uri.startsWith(location.origin);
        try {
            finishExtraction(await loadImagePixels(uri, { crossOrigin: !sameOrigin }));
        } catch (err) {
            flash("Image blocked (CORS) — save it and drag the file");
            console.warn("[TrentPalette]", err);
        }
    }

    // True if a drag carries an image file or a URL (file contents/names are
    // hidden during dragover, so we inspect items[].type/kind or types[]).
    function dragHasImage(e) {
        const dt = e.dataTransfer;
        if (!dt) return false;
        if (dt.items && dt.items.length) {
            for (const it of dt.items) {
                // Accept any dragged file (the drop handler filters to images);
                // some browsers don't expose the MIME type mid-drag.
                if (it.kind === "file") return true;
                if (it.kind === "string" && (it.type === "text/uri-list" || it.type === "text/plain")) return true;
            }
            return false;
        }
        return (dt.types || []).some((t) => t === "Files" || t === "text/uri-list" || t === "text/plain");
    }

    let dragDepth = 0;
    function setDropHighlight(on) {
        dropZone.style.borderColor = on ? "#45b6e0" : "#2a3942";
        dropZone.style.background = on ? "rgba(69,182,224,0.12)" : "transparent";
        dropZone.style.color = on ? "#cdd6dd" : "#7c8a93";
    }
    root.addEventListener("dragenter", (e) => {
        if (!dragHasImage(e)) return;
        e.preventDefault();
        dragDepth++;
        setDropHighlight(true);
    });
    root.addEventListener("dragover", (e) => {
        if (!dragHasImage(e)) return;
        e.preventDefault();
        e.stopPropagation(); // keep ComfyUI's global canvas drop from also firing
        e.dataTransfer.dropEffect = "copy";
    });
    root.addEventListener("dragleave", () => {
        dragDepth = Math.max(0, dragDepth - 1);
        if (dragDepth === 0) setDropHighlight(false);
    });
    root.addEventListener("drop", async (e) => {
        e.preventDefault();
        e.stopPropagation(); // critical: block the canvas from spawning a LoadImage node
        dragDepth = 0;
        setDropHighlight(false);
        const dt = e.dataTransfer;
        const file = [...(dt.files || [])].find((f) => /^image\//i.test(f.type) || IMAGE_RE.test(f.name || ""));
        if (file) { await extractFromFile(file); return; }
        const uri = (dt.getData("text/uri-list") || dt.getData("text/plain") || "")
            .split("\n").map((s) => s.trim()).find((s) => /^https?:|^data:|^blob:/.test(s));
        if (uri) { await extractFromUrl(uri); return; }
        flash("No image in drop");
    });

    return root;
}

// ---- sidebar mounting ------------------------------------------------------

// Build the palette body lazily, then keep the one instance for the app's life.
let paletteRoot = null;
function ensurePaletteBuilt() {
    if (!paletteRoot) paletteRoot = buildPaletteBody();
    return paletteRoot;
}

// Inject the 🎨 emoji as the sidebar tab's icon (icons are CSS classes; the
// ::before content trick mirrors how other extensions supply a custom glyph).
function installTabIcon() {
    if (document.getElementById("trent-palette-icon-style")) return;
    const style = document.createElement("style");
    style.id = "trent-palette-icon-style";
    style.textContent = ".trent-palette-icon:before{content:'🎨';font-style:normal;}";
    document.head.appendChild(style);
}

// Open/close the palette sidebar tab. The toggle lives on the sidebar-tab store;
// its exact location has shifted across frontend versions, so probe a few paths.
function toggleSidebar() {
    const em = app.extensionManager;
    if (!em) return;
    const fns = [
        em.toggleSidebarTab,
        em.sidebarTab && em.sidebarTab.toggleSidebarTab,
        em.sidebarTab && em.sidebarTab.value && em.sidebarTab.value.toggleSidebarTab,
    ];
    for (const fn of fns) {
        if (typeof fn === "function") { fn.call(em.sidebarTab || em, TAB_ID); return; }
    }
    console.warn("[TrentPalette] could not toggle sidebar tab (API not found)");
}

app.registerExtension({
    name: "TrentNodes.colorPalette",

    commands: [
        {
            id: "TrentNodes.ToggleColorPalette",
            label: "🎨 Color Palette",
            icon: "pi pi-palette",
            function: toggleSidebar,
        },
    ],

    keybindings: [
        {
            commandId: "TrentNodes.ToggleColorPalette",
            combo: { key: "c", shift: true, alt: true },
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.ToggleColorPalette"],
        },
    ],

    async setup() {
        installTabIcon();
        // Dock the palette as a native sidebar tab. render() fires on every open
        // (sometimes with a fresh container), so re-parent the one persistent
        // body instead of rebuilding it — keeps paint/undo state and closures.
        app.extensionManager.registerSidebarTab({
            id: TAB_ID,
            title: "Colors",
            tooltip: "Trent Color Palette",
            icon: "trent-palette-icon",
            type: "custom",
            render: (el) => { el.appendChild(ensurePaletteBuilt()); },
        });
        // Re-sync presets into the context menu once the canvas/LiteGraph is ready.
        syncContextMenuColors(loadColors());
        // Paint-bucket: install the canvas click patch + Escape-to-disarm once.
        installBucketPatch();
        installEscDisarm();
    },
});
