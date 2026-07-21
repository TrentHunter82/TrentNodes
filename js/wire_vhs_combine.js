import { app } from "../../scripts/app.js";

/**
 * Wire VHS Combine Extension
 *
 * One-hotkey wiring of a VHS output workflow. Two behaviours,
 * chosen automatically:
 *
 *   A. A VHS Load Video is present (feeding, or available to feed,
 *      the Combine):
 *        1. Adds (or reuses) a VHS_VideoInfo node off the Load
 *           Video's video_info output.
 *        2. Converts the VHS_VideoCombine `frame_rate` widget to an
 *           input slot.
 *        3. Runs VideoInfo `loaded_fps` down through a chain of two
 *           Reroute nodes parked at the bottom of the workflow, then
 *           back up into the Combine `frame_rate` input.
 *        4. Sets Combine `format` to `video/h264-mp4` and `crf` 13.
 *
 *   B. No VHS Load Video anywhere in the graph:
 *        Just sets the Combine to `video/h264-mp4` / crf 13 and
 *        defaults its `frame_rate` widget to 24. No VideoInfo,
 *        no reroutes.
 *
 * Scope: if a VHS Load Video and/or a VHS_VideoCombine are selected,
 * those are used. Otherwise falls back to the single matching
 * node(s) in the graph.
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
const REROUTE_TYPE = "Reroute";

// VHS_LoadVideo* output 3 = video_info (VHS_VIDEOINFO).
const LOAD_VIDEO_INFO_SLOT = 3;
// VHS_VideoInfo output 5 = loaded_fps (FLOAT).
const INFO_LOADED_FPS_SLOT = 5;

const TARGET_FORMAT = "video/h264-mp4";
const TARGET_CRF = 13;
// Fallback frame rate used when there is no Load Video to source fps from.
const DEFAULT_FPS = 24;
// Vertical gap below the lowest node where the reroute chain is parked.
const BOTTOM_MARGIN = 120;

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
 * Resolve the working Combine and (optional) Load node using the
 * selection-first pattern: selected nodes take priority, with the
 * graph filling in whichever side the selection is missing.
 *
 * Returns { combineNode, loadNode } where loadNode may be null
 * (scenario B), or { error }.
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
    const usingSelection = (
        pick.loads.length > 0 || pick.combines.length > 0
    );

    if (!usingSelection) {
        pick = findIn(graph._nodes || []);
    } else {
        // Fill in whichever side the selection is missing from the graph.
        if (pick.combines.length === 0) {
            pick.combines = (graph._nodes || []).filter(
                (n) => getNodeType(n) === VHS_VIDEO_COMBINE
            );
        }
        if (pick.loads.length === 0) {
            pick.loads = (graph._nodes || []).filter(
                (n) => VHS_LOAD_TYPES.has(getNodeType(n))
            );
        }
    }

    if (pick.combines.length !== 1) {
        return { error: (
            `Select one VHS Video Combine, then press Shift+Alt+V.\n\n`
            + `Found ${pick.combines.length} Video Combine node(s)`
            + ` in ${usingSelection ? "your selection" : "the graph"}.`
        )};
    }
    if (pick.loads.length > 1) {
        return { error: (
            `Found ${pick.loads.length} VHS Load Video nodes.\n\n`
            + `Select the one to wire (plus the Video Combine),`
            + ` then press Shift+Alt+V.`
        )};
    }

    return {
        combineNode: pick.combines[0],
        loadNode: pick.loads[0] || null,
    };
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
 * Y coordinate just below the lowest node in the graph, where the
 * reroute chain is parked so its wire runs along the bottom edge.
 */
function computeBottomY(graph) {
    let maxY = -Infinity;
    for (const n of graph._nodes || []) {
        const y = (n.pos?.[1] ?? 0) + (n.size?.[1] ?? 0);
        if (y > maxY) maxY = y;
    }
    if (!isFinite(maxY)) maxY = 0;
    return maxY + BOTTOM_MARGIN;
}

/**
 * Route `sourceNode[sourceSlot]` into `combineNode[frameRateSlot]`
 * through a chain of two Reroute nodes parked at the bottom of the
 * workflow. Falls back to a direct connection if Reroute nodes can't
 * be created. Returns the created reroute nodes (or null).
 */
function createRerouteChain(
    graph, sourceNode, sourceSlot, combineNode, frameRateSlot
) {
    const rerouteA = LiteGraph.createNode(REROUTE_TYPE);
    const rerouteB = LiteGraph.createNode(REROUTE_TYPE);
    if (!rerouteA || !rerouteB) {
        console.warn(
            "[WireVHSCombine] Could not create Reroute nodes;"
            + " connecting fps directly."
        );
        sourceNode.connect(sourceSlot, combineNode, frameRateSlot);
        return null;
    }

    const bottomY = computeBottomY(graph);
    // A sits under the source, B under the Combine, so the wire drops
    // down, runs along the bottom, and comes back up into the Combine.
    rerouteA.pos = [sourceNode.pos[0], bottomY];
    rerouteB.pos = [combineNode.pos[0], bottomY];
    graph.add(rerouteA);
    graph.add(rerouteB);

    sourceNode.connect(sourceSlot, rerouteA, 0);
    rerouteA.connect(0, rerouteB, 0);
    rerouteB.connect(0, combineNode, frameRateSlot);

    return [rerouteA, rerouteB];
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

/**
 * Scenario B fallback: default the frame_rate to DEFAULT_FPS. Only
 * works while frame_rate is still a widget (the no-Load-Video path
 * never converts it to an input).
 */
function applyDefaultFps(combineNode) {
    const widget = combineNode.widgets?.find(
        (w) => w.name === "frame_rate"
    );
    if (widget) {
        widget.value = DEFAULT_FPS;
        if (widget.callback) widget.callback(DEFAULT_FPS);
    } else {
        console.warn(
            "[WireVHSCombine] frame_rate is an input slot, not a"
            + " widget; leaving it as-is (no Load Video to source fps)."
        );
    }
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
        // Always: set format=h264-mp4, crf=13.
        applyH264Settings(combineNode);

        // Scenario B: no Load Video -> just default the fps and stop.
        if (!loadNode) {
            applyDefaultFps(combineNode);
            graph.setDirtyCanvas(true, true);
            console.log(
                "[WireVHSCombine] no Load Video found;"
                + ` set VHS_VideoCombine #${combineNode.id} to h264-mp4,`
                + ` crf ${TARGET_CRF}, frame_rate ${DEFAULT_FPS}.`
            );
            return;
        }

        // Scenario A, step 1: ensure a VideoInfo node is wired to Load.
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

        // Step 3: run loaded_fps -> two bottom reroutes -> frame_rate,
        // unless frame_rate is already wired (idempotent re-runs).
        const frInput = combineNode.inputs?.[frameRateSlot];
        if (frInput && frInput.link != null) {
            console.log(
                "[WireVHSCombine] frame_rate already wired;"
                + " leaving existing routing intact."
            );
        } else {
            createRerouteChain(
                graph, infoNode, INFO_LOADED_FPS_SLOT,
                combineNode, frameRateSlot
            );
        }

        graph.setDirtyCanvas(true, true);

        console.log(
            "[WireVHSCombine] wired"
            + ` ${getNodeType(loadNode)} #${loadNode.id}`
            + ` -> VHS_VideoInfo #${infoNode.id}`
            + ` -> [2x Reroute @ bottom]`
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
