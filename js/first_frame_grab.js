import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * First Frame Grab Extension
 *
 * Hotkey: Shift+Alt+F
 *
 * Takes the video from the selected VHS Load Video node (upload or
 * path variant; native LoadVideo also works), asks the TrentNodes
 * server to extract frame 1 into the input folder, then drops a
 * LoadImage node on the canvas next to the source node with that
 * frame already selected and previewed.
 *
 * If nothing is selected but the graph contains exactly one video
 * load node, that node is used.
 */

const LOAD_TYPES = new Set([
    "VHS_LoadVideo",
    "VHS_LoadVideoPath",
    "VHS_LoadVideoFFmpeg",
    "VHS_LoadVideoFFmpegPath",
    "LoadVideo",
]);

// Horizontal gap between the new LoadImage and the source node.
const GAP_X = 60;
// Fallback LoadImage width used for placement before first draw.
const DEFAULT_WIDTH = 315;

function getNodeType(node) {
    return (
        node.comfyClass
        || node.constructor?.comfyClass
        || node.type
        || ""
    );
}

function notify(severity, detail) {
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({
            severity,
            summary: "First Frame Grab",
            detail,
            life: 4000,
        });
    } else if (severity === "error" || severity === "warn") {
        alert(`First Frame Grab: ${detail}`);
    } else {
        console.log(`[FirstFrameGrab] ${detail}`);
    }
}

/**
 * Pick the source node: a selected load node wins; otherwise fall
 * back to the graph's sole load node if there is exactly one.
 */
function findSourceNode() {
    const selected = Object.values(app.canvas.selected_nodes ?? {});
    const pick = selected.find((n) => LOAD_TYPES.has(getNodeType(n)));
    if (pick) return pick;

    const loaders = (app.graph._nodes || []).filter(
        (n) => LOAD_TYPES.has(getNodeType(n))
    );
    return loaders.length === 1 ? loaders[0] : null;
}

/**
 * Create the LoadImage node beside the source node, select the
 * extracted frame in its image widget, and force the preview.
 */
function spawnLoadImage(sourceNode, filename) {
    const canvas = app.canvas;
    canvas.emitBeforeChange();
    try {
        const node = LiteGraph.createNode("LoadImage");
        if (!node) {
            notify("error", "Could not create a LoadImage node.");
            return;
        }
        node.pos = [
            sourceNode.pos[0]
                - (node.size?.[0] ?? DEFAULT_WIDTH)
                - GAP_X,
            sourceNode.pos[1],
        ];
        app.graph.add(node);

        const widget = node.widgets?.find((w) => w.name === "image");
        if (widget) {
            const values = widget.options?.values;
            if (Array.isArray(values) && !values.includes(filename)) {
                values.push(filename);
            }
            widget.value = filename;
            widget.callback?.(filename);
        }

        // Force the preview: the combo's own refresh only knows files
        // that existed at page load, so fetch the image directly.
        const img = new Image();
        img.onload = () => {
            node.imgs = [img];
            app.graph.setDirtyCanvas(true, true);
        };
        img.src = api.apiURL(
            "/view?filename=" + encodeURIComponent(filename)
            + "&type=input&subfolder=&rand=" + Math.random()
        );

        app.graph.setDirtyCanvas(true, true);
    } finally {
        canvas.emitAfterChange();
    }
}

async function grabFirstFrame() {
    const sourceNode = findSourceNode();
    if (!sourceNode) {
        notify(
            "warn",
            "Select a VHS Load Video node first."
        );
        return;
    }

    const widget = sourceNode.widgets?.find(
        (w) => w.name === "video" || w.name === "file"
    );
    const video = widget?.value;
    if (typeof video !== "string" || !video.trim()) {
        notify("warn", "That node has no video chosen.");
        return;
    }

    let resp;
    try {
        resp = await api.fetchApi("/trent/first_frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video }),
        });
    } catch (err) {
        notify("error", `Request failed: ${err}`);
        return;
    }

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.ok) {
        notify("error", data.error || `Server error ${resp.status}`);
        return;
    }

    spawnLoadImage(sourceNode, data.filename);
    notify("success", `Extracted ${data.filename}`);
}

app.registerExtension({
    name: "TrentNodes.FirstFrameGrab",

    commands: [
        {
            id: "TrentNodes.FirstFrameGrab",
            label: "Grab First Frame as Load Image",
            icon: "pi pi-image",
            function: () => grabFirstFrame(),
        },
    ],

    keybindings: [
        {
            commandId: "TrentNodes.FirstFrameGrab",
            combo: { key: "f", shift: true, alt: true },
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.FirstFrameGrab"],
        },
    ],
});
