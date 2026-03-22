import { app } from "../../scripts/app.js";

/**
 * Grid Paste Extension
 *
 * Paste multiple copies of selected nodes arranged in a grid.
 *
 * Two modes:
 *   Ctrl+Shift+;       - Grid Paste (independent copies)
 *   Ctrl+Shift+Alt+;   - Grid Paste Connected (each copy's
 *                         inputs wired to the original sources,
 *                         same as Ctrl+Shift+V but in bulk)
 *
 * If nodes are selected, auto-copies them first.
 * Otherwise uses whatever is already on the clipboard.
 */

const CLIPBOARD_KEY = "litegrapheditor_clipboard";
const MAX_COPIES = 100;

/**
 * Compute the bounding box of all items in clipboard data.
 * Returns {x, y, width, height} or null if empty.
 */
function getClipboardBBox(parsed) {
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const node of (parsed.nodes || [])) {
        const [x, y] = node.pos;
        const [w, h] = node.size || [200, 100];
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + w);
        maxY = Math.max(maxY, y + h);
    }
    for (const group of (parsed.groups || [])) {
        const [x, y, w, h] = group.bounding;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + w);
        maxY = Math.max(maxY, y + h);
    }
    for (const reroute of (parsed.reroutes || [])) {
        const [x, y] = reroute.pos;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + 10);
        maxY = Math.max(maxY, y + 10);
    }

    if (!isFinite(minX)) return null;

    return {
        x: minX,
        y: minY,
        width: Math.max(maxX - minX, 1),
        height: Math.max(maxY - minY, 1)
    };
}

/**
 * Core grid paste logic.
 * @param {boolean} connectInputs - When true, each copy's
 *   uncopied input sources are wired to the original graph
 *   nodes (same behavior as Ctrl+Shift+V).
 */
function gridPaste(connectInputs = false) {
    const canvas = app.canvas;
    if (!canvas) return;

    // Auto-copy if nodes are currently selected
    const hasSelection = (
        canvas.selectedItems?.size > 0 ||
        (canvas.selected_nodes &&
         Object.keys(canvas.selected_nodes).length > 0)
    );
    if (hasSelection) {
        canvas.copyToClipboard();
    }

    // Read clipboard
    const clipboardStr = localStorage.getItem(CLIPBOARD_KEY);
    if (!clipboardStr) {
        console.warn("[GridPaste] Clipboard empty.");
        return;
    }

    let parsed;
    try {
        parsed = JSON.parse(clipboardStr);
    } catch (e) {
        console.error("[GridPaste] Bad clipboard data:", e);
        return;
    }

    const bbox = getClipboardBBox(parsed);
    if (!bbox) {
        console.warn("[GridPaste] No items in clipboard.");
        return;
    }

    // Prompt for count
    const mode = connectInputs ? "Connected" : "Standard";
    const sizeLabel = Math.ceil(bbox.width)
        + " x " + Math.ceil(bbox.height) + " px";
    const countStr = prompt(
        "Grid Paste (" + mode + ") -- how many copies?\n"
            + "Selection size: " + sizeLabel,
        "4"
    );
    if (!countStr) return;

    const count = parseInt(countStr, 10);
    if (isNaN(count) || count < 1) return;
    if (count > MAX_COPIES) {
        prompt("Maximum " + MAX_COPIES + " copies.", "");
        return;
    }

    // Grid dimensions (roughly square)
    const cols = Math.ceil(Math.sqrt(count));
    const padding = 50;

    // Paste origin: current mouse position on canvas
    const startX = canvas.graph_mouse[0];
    const startY = canvas.graph_mouse[1];

    // Wrap all pastes in one undo transaction.
    // ChangeTracker uses a counter, so nested
    // emitBeforeChange/emitAfterChange calls inside
    // pasteFromClipboard are handled correctly.
    canvas.emitBeforeChange();

    try {
        for (let i = 0; i < count; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = startX + col * (bbox.width + padding);
            const y = startY + row * (bbox.height + padding);
            canvas.pasteFromClipboard({
                position: [x, y],
                connectInputs
            });
        }
    } finally {
        canvas.emitAfterChange();
    }

    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "TrentNodes.GridPaste",

    commands: [
        {
            id: "TrentNodes.GridPaste",
            label: "Grid Paste",
            icon: "pi pi-th-large",
            function: () => gridPaste(false)
        },
        {
            id: "TrentNodes.GridPasteConnected",
            label: "Grid Paste Connected",
            icon: "pi pi-th-large",
            function: () => gridPaste(true)
        }
    ],

    keybindings: [
        {
            commandId: "TrentNodes.GridPaste",
            combo: { key: ";", ctrl: true, shift: true }
        },
        {
            commandId: "TrentNodes.GridPasteConnected",
            combo: {
                key: ";", ctrl: true, shift: true, alt: true
            }
        }
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: [
                "TrentNodes.GridPaste",
                "TrentNodes.GridPasteConnected"
            ]
        }
    ],

    getSelectionToolboxCommands: () => [
        "TrentNodes.GridPaste",
        "TrentNodes.GridPasteConnected"
    ]
});
