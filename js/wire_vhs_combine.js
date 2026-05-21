import { app } from "../../scripts/app.js";

/**
 * Wire VHS Combine Extension
 *
 * One-hotkey wiring of a VHS workflow:
 *   1. Adds (or reuses) a VHS_VideoInfo node off the Load Video's
 *      video_info output.
 *   2. Converts the VHS_VideoCombine `frame_rate` widget to an
 *      input slot and connects VideoInfo's `loaded_fps` to it.
 *   3. Sets VideoCombine `format` to `video/h264-mp4` and `crf`
 *      to 13.
 *
 * Scope: if both a VHS Load Video and a VHS_VideoCombine are
 * selected, those are used. Otherwise falls back to the single
 * matching pair in the graph.
 *
 * Hotkey: Shift+Alt+V
 */

const VHS_LOAD_TYPES = new Set([
    "VHS_LoadVideo",
    "VHS_LoadVideoPath",
    "VHS_LoadVideoFFmpeg",
    "VHS_LoadVideoFFmpegPath",
]);
const VHS_VIDEO_INFO = "VHS_VideoInfo";
const VHS_VIDEO_COMBINE = "VHS_VideoCombine";

// VHS_LoadVideo* output 3 = video_info (VHS_VIDEOINFO).
const LOAD_VIDEO_INFO_SLOT = 3;
// VHS_VideoInfo output 5 = loaded_fps (FLOAT).
const INFO_LOADED_FPS_SLOT = 5;

const TARGET_FORMAT = "video/h264-mp4";
const TARGET_CRF = 13;

// Resolved at module load via the official ComfyUI shim, with a
// dynamic-import fallback for builds where window.comfyAPI hasn't
// populated yet.
let convertWidgetToInput = window.comfyAPI?.widgetInputs?.convertToInput;
if (!convertWidgetToInput) {
    import("/extensions/core/widgetInputs.js")
        .then((mod) => {
            convertWidgetToInput = mod.convertToInput;
        })
        .catch((err) => {
            console.warn(
                "[WireVHSCombine] Could not load widgetInputs.js:", err
            );
        });
}

function getNodeType(node) {
    return (
        node.comfyClass
        || node.constructor?.comfyClass
        || node.type
        || ""
    );
}

/**
 * Get the node-definition input config for a given input name.
 * Falls back through several places where ComfyUI stashes nodeData.
 */
function getInputConfig(node, inputName) {
    const sources = [
        node.constructor?.nodeData,
        LiteGraph.registered_node_types?.[node.type]?.nodeData,
        LiteGraph.registered_node_types?.[getNodeType(node)]?.nodeData,
    ];
    for (const nd of sources) {
        const req = nd?.input?.required?.[inputName];
        if (req) return req;
        const opt = nd?.input?.optional?.[inputName];
        if (opt) return opt;
    }
    return null;
}

/**
 * Resolve the working Load + Combine pair using the selection-first
 * pattern: selected nodes if they unambiguously identify one of each,
 * otherwise fall back to the graph if it contains exactly one of each.
 */
function resolvePair() {
    const canvas = app.canvas;
    const graph = app.graph;
    const selected = canvas?.selected_nodes
        ? Object.values(canvas.selected_nodes)
        : [];

    const findIn = (nodes) => {
        const loads = [];
        const combines = [];
        for (const n of nodes) {
            const t = getNodeType(n);
            if (VHS_LOAD_TYPES.has(t)) loads.push(n);
            else if (t === VHS_VIDEO_COMBINE) combines.push(n);
        }
        return { loads, combines };
    };

    let pick = findIn(selected);
    let usingSelection = (
        pick.loads.length > 0 || pick.combines.length > 0
    );

    if (!usingSelection) {
        pick = findIn(graph._nodes || []);
    } else {
        // Fill in whichever side the selection is missing from the graph.
        if (pick.loads.length === 0) {
            pick.loads = (graph._nodes || []).filter(
                (n) => VHS_LOAD_TYPES.has(getNodeType(n))
            );
        }
        if (pick.combines.length === 0) {
            pick.combines = (graph._nodes || []).filter(
                (n) => getNodeType(n) === VHS_VIDEO_COMBINE
            );
        }
    }

    if (pick.loads.length !== 1 || pick.combines.length !== 1) {
        return { error: (
            `Select one VHS Load Video and one VHS Video Combine,`
            + ` then press Shift+Alt+V.\n\n`
            + `Found ${pick.loads.length} Load Video node(s)`
            + ` and ${pick.combines.length} Video Combine node(s)`
            + ` in ${usingSelection ? "your selection" : "the graph"}.`
        )};
    }

    return { loadNode: pick.loads[0], combineNode: pick.combines[0] };
}

/**
 * Find a VHS_VideoInfo node already wired to the Load's video_info
 * output. Returns null if none exists.
 */
function findExistingVideoInfo(loadNode, graph) {
    const out = loadNode.outputs?.[LOAD_VIDEO_INFO_SLOT];
    if (!out || !out.links) return null;
    for (const linkId of out.links) {
        const link = graph.links[linkId];
        if (!link) continue;
        const target = graph.getNodeById(link.target_id);
        if (target && getNodeType(target) === VHS_VIDEO_INFO) {
            return target;
        }
    }
    return null;
}

/**
 * Find the input slot index for a given input name. Returns -1 if
 * the node doesn't have such an input slot (e.g. it's still a widget).
 */
function findInputSlotByName(node, name) {
    if (!node.inputs) return -1;
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name === name) return i;
    }
    return -1;
}

/**
 * Ensure `frame_rate` is an input slot on the combine node, converting
 * the widget if needed. Returns the input slot index, or -1 on failure.
 */
function ensureFrameRateInput(combineNode) {
    const existing = findInputSlotByName(combineNode, "frame_rate");
    if (existing >= 0) return existing;

    const widget = combineNode.widgets?.find(
        (w) => w.name === "frame_rate"
    );
    if (!widget) {
        console.warn(
            "[WireVHSCombine] frame_rate widget not found on Combine."
        );
        return -1;
    }

    if (!convertWidgetToInput) {
        console.warn(
            "[WireVHSCombine] convertToInput helper unavailable."
            + " Cannot convert frame_rate widget to input."
        );
        return -1;
    }

    const config = getInputConfig(combineNode, "frame_rate")
        ?? ["FLOAT", { default: 8, min: 1, step: 1 }];

    try {
        convertWidgetToInput(combineNode, widget, config);
    } catch (err) {
        console.warn(
            "[WireVHSCombine] convertToInput threw:", err
        );
        return -1;
    }

    return findInputSlotByName(combineNode, "frame_rate");
}

/**
 * Set the `format` widget to h264-mp4, firing its callback so VHS
 * adds the format-specific widgets (crf, pix_fmt, etc.). Then set
 * crf to the target value.
 */
function applyH264Settings(combineNode) {
    const formatW = combineNode.widgets?.find((w) => w.name === "format");
    if (!formatW) {
        console.warn(
            "[WireVHSCombine] format widget not found on Combine."
        );
        return;
    }

    if (formatW.value !== TARGET_FORMAT) {
        formatW.value = TARGET_FORMAT;
        if (formatW.callback) formatW.callback(TARGET_FORMAT);
    }

    // After the callback, VHS has appended the format-specific widgets.
    const crfW = combineNode.widgets?.find((w) => w.name === "crf");
    if (crfW) {
        crfW.value = TARGET_CRF;
        if (crfW.callback) crfW.callback(TARGET_CRF);
    } else {
        console.warn(
            "[WireVHSCombine] crf widget did not appear after format"
            + " change. Open the Combine node and set crf manually."
        );
    }

    combineNode.setDirtyCanvas(true, true);
}

function wireVHSCombine() {
    if (
        !LiteGraph.registered_node_types[VHS_VIDEO_INFO]
        || !LiteGraph.registered_node_types[VHS_VIDEO_COMBINE]
    ) {
        alert(
            "VHS (Video Helper Suite) is not installed.\n"
            + "Install it to use the Wire VHS Combine feature."
        );
        return;
    }

    const resolved = resolvePair();
    if (resolved.error) {
        alert(resolved.error);
        return;
    }
    const { loadNode, combineNode } = resolved;

    const canvas = app.canvas;
    const graph = app.graph;

    canvas.emitBeforeChange();
    try {
        // Step 1: ensure a VideoInfo node is wired to the Load.
        let infoNode = findExistingVideoInfo(loadNode, graph);
        if (!infoNode) {
            infoNode = LiteGraph.createNode(VHS_VIDEO_INFO);
            if (!infoNode) {
                console.warn(
                    "[WireVHSCombine] Failed to create VHS_VideoInfo."
                );
                return;
            }
            infoNode.pos = [
                loadNode.pos[0] + (loadNode.size?.[0] || 240) + 40,
                loadNode.pos[1],
            ];
            graph.add(infoNode);
            loadNode.connect(LOAD_VIDEO_INFO_SLOT, infoNode, 0);
        }

        // Step 2: convert frame_rate widget to input on the Combine.
        const frameRateSlot = ensureFrameRateInput(combineNode);
        if (frameRateSlot < 0) return;

        // Step 3: wire VideoInfo loaded_fps -> Combine frame_rate.
        infoNode.connect(
            INFO_LOADED_FPS_SLOT, combineNode, frameRateSlot
        );

        // Step 4: set format=h264-mp4, crf=13.
        applyH264Settings(combineNode);

        graph.setDirtyCanvas(true, true);

        console.log(
            "[WireVHSCombine] wired"
            + ` ${getNodeType(loadNode)} #${loadNode.id}`
            + ` -> VHS_VideoInfo #${infoNode.id}`
            + ` -> VHS_VideoCombine #${combineNode.id}`
            + ` (h264-mp4, crf ${TARGET_CRF})`
        );
    } finally {
        canvas.emitAfterChange();
    }
}

app.registerExtension({
    name: "TrentNodes.WireVHSCombine",

    commands: [
        {
            id: "TrentNodes.WireVHSCombine",
            label: "Wire VHS Combine (h264 CRF 13)",
            icon: "pi pi-link",
            function: wireVHSCombine,
        },
    ],

    keybindings: [
        {
            commandId: "TrentNodes.WireVHSCombine",
            combo: { key: "v", shift: true, alt: true },
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.WireVHSCombine"],
        },
    ],
});
