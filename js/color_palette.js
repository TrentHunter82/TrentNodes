import { app } from "../../scripts/app.js";

/*
 * Trent Color Palette
 * -------------------
 * A tiny floating, draggable palette of 7 preset color swatches.
 *   - Left-click a swatch  -> color all currently-selected node(s)
 *   - Right-click a swatch -> pick a new preset color for that slot
 *   - "Reset" chip         -> strip selected node(s) back to default color
 * Presets are also synced into LiteGraph's right-click "Colors" menu.
 * Everything (colors, panel position, collapsed state) persists in localStorage.
 */

const LS_COLORS = "TrentPalette.colors";
const LS_POS = "TrentPalette.pos";
const LS_COLLAPSED = "TrentPalette.collapsed";

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

function getSelectedNodes() {
    const canvas = app.canvas;
    if (!canvas || !canvas.selected_nodes) return [];
    return Object.values(canvas.selected_nodes);
}

function applyColorToSelection(hex) {
    const nodes = getSelectedNodes();
    if (!nodes.length) {
        flash("Select a node first");
        return;
    }
    const body = darken(hex);
    for (const node of nodes) {
        node.color = hex;
        node.bgcolor = body;
    }
    app.canvas.setDirty(true, true);
}

function resetSelection() {
    const nodes = getSelectedNodes();
    if (!nodes.length) {
        flash("Select a node first");
        return;
    }
    for (const node of nodes) {
        delete node.color;
        delete node.bgcolor;
    }
    app.canvas.setDirty(true, true);
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

function buildPanel() {
    if (document.getElementById("trent-palette")) return;

    const colors = loadColors();
    syncContextMenuColors(colors);

    const panel = document.createElement("div");
    panel.id = "trent-palette";
    Object.assign(panel.style, {
        position: "fixed",
        zIndex: "1000",
        background: "#11181d",
        border: "1px solid #2a3942",
        borderRadius: "8px",
        boxShadow: "0 4px 18px rgba(0,0,0,0.55)",
        font: "12px sans-serif",
        color: "#cdd6dd",
        userSelect: "none",
        width: "auto",
    });

    // restore position
    let pos = { left: "auto", top: "120px", right: "16px" };
    try { pos = Object.assign(pos, JSON.parse(localStorage.getItem(LS_POS) || "{}")); } catch (e) {}
    panel.style.left = pos.left;
    panel.style.top = pos.top;
    panel.style.right = pos.left === "auto" ? pos.right : "auto";

    // header (drag handle + collapse)
    const header = document.createElement("div");
    Object.assign(header.style, {
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "5px 8px", cursor: "move", borderBottom: "1px solid #2a3942",
        gap: "8px",
    });
    const title = document.createElement("span");
    title.textContent = "🎨 Colors";
    title.style.fontWeight = "600";
    title.style.letterSpacing = "0.3px";
    const collapseBtn = document.createElement("span");
    collapseBtn.style.cursor = "pointer";
    collapseBtn.style.padding = "0 4px";
    collapseBtn.style.opacity = "0.7";
    header.appendChild(title);
    header.appendChild(collapseBtn);
    panel.appendChild(header);

    // body
    const body = document.createElement("div");
    body.style.padding = "8px";

    const swatchRow = document.createElement("div");
    Object.assign(swatchRow.style, { display: "flex", gap: "6px", flexWrap: "nowrap" });

    function refreshSwatchStyle(sw, hex) {
        sw.style.background = hex;
        sw.title = `Left-click: color selected node\nRight-click: change this color (${hex})`;
    }

    colors.forEach((hex, i) => {
        const sw = document.createElement("div");
        Object.assign(sw.style, {
            width: "26px", height: "26px", borderRadius: "5px",
            cursor: "pointer", border: "1px solid rgba(255,255,255,0.15)",
            transition: "transform 0.08s",
        });
        refreshSwatchStyle(sw, hex);
        sw.addEventListener("mouseenter", () => { sw.style.transform = "scale(1.12)"; });
        sw.addEventListener("mouseleave", () => { sw.style.transform = "scale(1)"; });

        // left-click -> apply
        sw.addEventListener("click", () => applyColorToSelection(colors[i]));

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
        marginTop: "8px", gap: "8px",
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

    const hint = document.createElement("span");
    hint.id = "trent-palette-hint";
    hint.textContent = "right-click swatch = edit";
    Object.assign(hint.style, { fontSize: "10px", opacity: "0.45", transition: "opacity 0.3s" });

    footer.appendChild(resetChip);
    footer.appendChild(hint);
    body.appendChild(footer);

    panel.appendChild(body);
    document.body.appendChild(panel);

    // ---- collapse behavior ----
    function setCollapsed(collapsed) {
        body.style.display = collapsed ? "none" : "block";
        collapseBtn.textContent = collapsed ? "▸" : "▾";
        localStorage.setItem(LS_COLLAPSED, collapsed ? "1" : "0");
    }
    setCollapsed(localStorage.getItem(LS_COLLAPSED) === "1");
    collapseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        setCollapsed(body.style.display !== "none");
    });

    // ---- dragging ----
    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
    header.addEventListener("mousedown", (e) => {
        if (e.target === collapseBtn) return;
        dragging = true;
        const rect = panel.getBoundingClientRect();
        ox = rect.left; oy = rect.top;
        sx = e.clientX; sy = e.clientY;
        e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const nx = Math.max(0, Math.min(window.innerWidth - 60, ox + (e.clientX - sx)));
        const ny = Math.max(0, Math.min(window.innerHeight - 30, oy + (e.clientY - sy)));
        panel.style.left = nx + "px";
        panel.style.top = ny + "px";
        panel.style.right = "auto";
    });
    window.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        localStorage.setItem(LS_POS, JSON.stringify({
            left: panel.style.left, top: panel.style.top, right: "auto",
        }));
    });
}

app.registerExtension({
    name: "TrentNodes.colorPalette",
    async setup() {
        buildPanel();
        // Re-sync presets into the context menu once the canvas/LiteGraph is ready.
        syncContextMenuColors(loadColors());
    },
});
