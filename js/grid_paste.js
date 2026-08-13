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
 * If nodes are selected, auto-copies them first and lays the
 * grid out around the selection.  Otherwise uses whatever is
 * already on the clipboard and starts the grid at the mouse.
 */

const CLIPBOARD_KEY = "litegrapheditor_clipboard";
const MAX_COPIES = 100;
const PADDING = 50;

function toast(severity, summary, detail) {
    app.extensionManager?.toast?.add({
        severity, summary, detail, life: 5000
    });
}

/**
 * Read [x, y, width, height] out of a litegraph bounding
 * rect.  Nodes and groups expose `boundingRect`; older items
 * only expose `getBounding()`.
 */
function readRect(item) {
    const rect = item.boundingRect ?? item.getBounding?.();
    if (!rect || rect.length < 4) return null;
    const [x, y, w, h] = rect;
    if (![x, y, w, h].every(Number.isFinite)) return null;
    return [x, y, w, h];
}

/**
 * Bounding box of the items currently selected on the canvas.
 *
 * `anchor` is the minimum item *pos*, because that is what
 * pasteFromClipboard() anchors on.  `width`/`height` come from
 * the full bounding rects, which include node title bars, so
 * grid cells never clip a title.
 *
 * Returns null when nothing is selected.
 */
function getSelectionBBox(canvas) {
    const items = canvas.selectedItems;
    if (!items || items.size === 0) return null;

    let anchorX = Infinity, anchorY = Infinity;
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    for (const item of items) {
        const pos = item.pos;
        if (pos && Number.isFinite(pos[0]) && Number.isFinite(pos[1])) {
            anchorX = Math.min(anchorX, pos[0]);
            anchorY = Math.min(anchorY, pos[1]);
        }

        const rect = readRect(item);
        if (rect) {
            minX = Math.min(minX, rect[0]);
            minY = Math.min(minY, rect[1]);
            maxX = Math.max(maxX, rect[0] + rect[2]);
            maxY = Math.max(maxY, rect[1] + rect[3]);
        } else if (pos) {
            minX = Math.min(minX, pos[0]);
            minY = Math.min(minY, pos[1]);
            maxX = Math.max(maxX, pos[0] + 10);
            maxY = Math.max(maxY, pos[1] + 10);
        }
    }

    if (!isFinite(anchorX) || !isFinite(minX)) return null;

    return {
        anchor: [anchorX, anchorY],
        width: Math.max(maxX - minX, 1),
        height: Math.max(maxY - minY, 1)
    };
}

/**
 * Fallback bounding box, computed from the serialized
 * clipboard.  Used when nothing is selected, so the live
 * items are not available.
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
        anchor: [minX, minY],
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

    // Measure the selection before the first paste.  Pasting
    // replaces the selection with the new copies, so this is
    // the only chance to read the source geometry.
    const selectionBBox = getSelectionBBox(canvas);
    if (selectionBBox) {
        canvas.copyToClipboard();
    }

    // Read clipboard
    const clipboardStr = localStorage.getItem(CLIPBOARD_KEY);
    if (!clipboardStr) {
        toast("warn", "Grid Paste", "Clipboard is empty.");
        return;
    }

    let parsed;
    try {
        parsed = JSON.parse(clipboardStr);
    } catch (e) {
        console.error("[GridPaste] Bad clipboard data:", e);
        toast("error", "Grid Paste", "Clipboard data is not valid.");
        return;
    }

    const bbox = selectionBBox ?? getClipboardBBox(parsed);
    if (!bbox) {
        toast("warn", "Grid Paste", "No items in clipboard.");
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
        toast(
            "warn", "Grid Paste",
            "Maximum " + MAX_COPIES + " copies."
        );
        return;
    }

    // Grid origin.
    //
    // With a live selection the grid is anchored on the source
    // items: the source holds cell 0 and the copies fill the
    // cells after it, so no copy lands on top of the source.
    // canvas.graph_mouse is not usable here -- the selection
    // toolbox is a DOM overlay, so the canvas stops receiving
    // pointer moves once the cursor is on a toolbox button and
    // graph_mouse keeps a stale value.
    //
    // With no selection there is no source to anchor on, so
    // the grid starts at the mouse.
    const usingSelection = selectionBBox !== null;
    const firstCell = usingSelection ? 1 : 0;
    const cols = Math.ceil(Math.sqrt(count + firstCell));
    const originX = usingSelection
        ? bbox.anchor[0] : canvas.graph_mouse[0];
    const originY = usingSelection
        ? bbox.anchor[1] : canvas.graph_mouse[1];

    const cellWidth = bbox.width + PADDING;
    const cellHeight = bbox.height + PADDING;

    // Wrap all pastes in one undo transaction.
    // ChangeTracker uses a counter, so nested
    // emitBeforeChange/emitAfterChange calls inside
    // pasteFromClipboard are handled correctly.
    canvas.emitBeforeChange();

    try {
        for (let i = 0; i < count; i++) {
            const cell = firstCell + i;
            const col = cell % cols;
            const row = Math.floor(cell / cols);
            canvas.pasteFromClipboard({
                position: [
                    originX + col * cellWidth,
                    originY + row * cellHeight
                ],
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

    // The keybinding store matches on KeyboardEvent.key, and
    // browsers report Shift + ";" as ":".  Both characters are
    // registered: ":" is the one that fires on a US layout,
    // ";" covers layouts that do not shift the character.
    keybindings: [
        {
            commandId: "TrentNodes.GridPaste",
            combo: { key: ":", ctrl: true, shift: true }
        },
        {
            commandId: "TrentNodes.GridPaste",
            combo: { key: ";", ctrl: true, shift: true }
        },
        {
            commandId: "TrentNodes.GridPasteConnected",
            combo: {
                key: ":", ctrl: true, shift: true, alt: true
            }
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
