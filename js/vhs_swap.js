import { app } from "../../scripts/app.js";

/**
 * VHS Swap Extension
 *
 * Replace native ComfyUI video nodes with VHS equivalents:
 *   LoadVideo       -> VHS_LoadVideo
 *   SaveVideo       -> VHS_VideoCombine
 *
 * Also collapses intermediate adapter nodes that become
 * unnecessary after the swap:
 *   GetVideoComponents  (removed, connections rewired)
 *   CreateVideo         (removed, connections rewired)
 *
 * After swapping, always wires up the VHS_VideoCombine(s):
 *   - Sets `format` to video/h264-mp4 and `crf` to 13.
 *   - Adds/reuses a VHS_VideoInfo off the VHS Load Video and runs its
 *     `loaded_fps` down through two Reroute nodes parked at the bottom
 *     of the workflow, then up into the Combine `frame_rate` input.
 *     (This restores the fps that the swap otherwise drops when it
 *     collapses GetVideoComponents.)
 *   - No VHS Load Video? Defaults `frame_rate` to 24 instead.
 * This wiring runs even when there's nothing to swap (a graph that is
 * already VHS), so Shift+V is also a one-press "wire the Combine" tool.
 *
 * Scope: if nodes are selected, only those are replaced.
 * Otherwise all matching nodes in the graph are replaced.
 *
 * Hotkey: Shift+V
 */

const NATIVE_LOAD = "LoadVideo";
const NATIVE_SAVE = "SaveVideo";
const NATIVE_GET_COMPONENTS = "GetVideoComponents";
const NATIVE_CREATE = "CreateVideo";
const VHS_LOAD = "VHS_LoadVideo";
const VHS_COMBINE = "VHS_VideoCombine";
const VHS_VIDEO_INFO = "VHS_VideoInfo";
const REROUTE_TYPE = "Reroute";

// All VHS Load Video variants (for sourcing fps after a swap).
const VHS_LOAD_TYPES = new Set([
    "VHS_LoadVideo",
    "VHS_LoadVideoPath",
    "VHS_LoadVideoFFmpeg",
    "VHS_LoadVideoFFmpegPath",
]);

// VHS_LoadVideo* output 3 = video_info; VHS_VideoInfo output 5 = loaded_fps.
const LOAD_VIDEO_INFO_SLOT = 3;
const INFO_LOADED_FPS_SLOT = 5;

// Combine output settings applied on every swap.
const TARGET_FORMAT = "video/h264-mp4";
const TARGET_CRF = 13;
// Fallback frame rate when there's no VHS Load Video to source fps from.
const DEFAULT_FPS = 24;
// Vertical gap below the lowest node where the reroute chain is parked.
const BOTTOM_MARGIN = 120;

// convertToInput helper, for turning the Combine's frame_rate widget into
// an input slot. Resolved via the ComfyUI shim, with a dynamic-import
// fallback for builds where window.comfyAPI hasn't populated yet.
let convertWidgetToInput = window.comfyAPI?.widgetInputs?.convertToInput;
if (!convertWidgetToInput) {
    import("/extensions/core/widgetInputs.js")
        .then((mod) => {
            convertWidgetToInput = mod.convertToInput;
        })
        .catch((err) => {
            console.warn(
                "[VHSSwap] Could not load widgetInputs.js:", err
            );
        });
}

/**
 * Collect all input and output connections for a node.
 */
function collectConnections(graph, node) {
    const inputs = [];
    for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
        const inp = node.inputs[i];
        if (inp.link == null) continue;
        const link = graph.links[inp.link];
        if (!link) continue;
        inputs.push({
            slotIndex: i,
            name: inp.name,
            type: inp.type,
            originNodeId: link.origin_id,
            originSlot: link.origin_slot,
        });
    }
    const outputs = [];
    for (let i = 0; i < (node.outputs?.length ?? 0); i++) {
        const out = node.outputs[i];
        for (const linkId of out.links || []) {
            const link = graph.links[linkId];
            if (!link) continue;
            outputs.push({
                slotIndex: i,
                name: out.name,
                type: out.type,
                targetNodeId: link.target_id,
                targetSlot: link.target_slot,
            });
        }
    }
    return { inputs, outputs };
}

/**
 * Transfer a widget value from one node to another by name.
 */
function transferWidgetValue(srcNode, dstNode, srcName, dstName) {
    const srcW = srcNode.widgets?.find((w) => w.name === srcName);
    const dstW = dstNode.widgets?.find((w) => w.name === dstName);
    if (srcW && dstW) {
        dstW.value = srcW.value;
        if (dstW.callback) dstW.callback(dstW.value);
    }
}

/**
 * Get the comfyClass for a node, handling different
 * property locations across ComfyUI versions.
 */
function getNodeType(node) {
    return (
        node.comfyClass
        || node.constructor?.comfyClass
        || node.type
        || ""
    );
}

/**
 * Get working set of nodes: selected if any, otherwise all.
 */
function getTargetNodes() {
    const canvas = app.canvas;
    const selected = canvas.selected_nodes;
    if (selected && Object.keys(selected).length > 0) {
        return Object.values(selected);
    }
    return app.graph._nodes || [];
}

/**
 * Bucket nodes by their native video type.
 */
function categorizeNodes(nodes) {
    const buckets = {
        load: [],
        save: [],
        getComponents: [],
        create: [],
    };
    for (const node of nodes) {
        const t = getNodeType(node);
        if (t === NATIVE_LOAD) buckets.load.push(node);
        else if (t === NATIVE_SAVE) buckets.save.push(node);
        else if (t === NATIVE_GET_COMPONENTS) {
            buckets.getComponents.push(node);
        } else if (t === NATIVE_CREATE) {
            buckets.create.push(node);
        }
    }
    return buckets;
}

/**
 * Main swap function.
 */
function vhsSwap() {
    // -- Step 1: validate VHS installed --
    if (
        !LiteGraph.registered_node_types[VHS_LOAD]
        || !LiteGraph.registered_node_types[VHS_COMBINE]
    ) {
        alert(
            "VHS (Video Helper Suite) is not installed.\n"
            + "Install it to use the VHS Swap feature."
        );
        return;
    }

    const canvas = app.canvas;
    const graph = app.graph;

    // -- Step 2: gather and categorize target nodes --
    const targets = getTargetNodes();
    const buckets = categorizeNodes(targets);
    const totalTargets = (
        buckets.load.length
        + buckets.save.length
        + buckets.getComponents.length
        + buckets.create.length
    );
    // Note: no early return when totalTargets === 0. The swap phases
    // below are no-ops on an empty bucket set, and we still want the
    // fps/mp4 wiring to run on an already-VHS graph.

    // -- Step 3: undo transaction --
    canvas.emitBeforeChange();
    try {
        // Phase A: scan all connections before any mutations
        const connMap = new Map();
        for (const list of Object.values(buckets)) {
            for (const node of list) {
                connMap.set(node.id, collectConnections(graph, node));
            }
        }

        // Build sets of IDs for quick lookup
        const saveIds = new Set(buckets.save.map((n) => n.id));
        const gvcIds = new Set(
            buckets.getComponents.map((n) => n.id)
        );
        const createIds = new Set(
            buckets.create.map((n) => n.id)
        );

        // Phase B: create replacement nodes
        const replacements = new Map(); // oldId -> newNode

        for (const node of buckets.load) {
            const newNode = LiteGraph.createNode(VHS_LOAD);
            if (!newNode) {
                console.warn("VHS Swap: failed to create", VHS_LOAD);
                continue;
            }
            newNode.pos = [...node.pos];
            graph.add(newNode);
            transferWidgetValue(node, newNode, "file", "video");
            replacements.set(node.id, newNode);
        }

        for (const node of buckets.save) {
            const newNode = LiteGraph.createNode(VHS_COMBINE);
            if (!newNode) {
                console.warn(
                    "VHS Swap: failed to create", VHS_COMBINE
                );
                continue;
            }
            newNode.pos = [...node.pos];
            graph.add(newNode);
            transferWidgetValue(
                node, newNode, "filename_prefix", "filename_prefix"
            );
            replacements.set(node.id, newNode);
        }

        // Phase C: reconnect

        // C1: reconnect LoadVideo replacements (VHS_LoadVideo)
        // VHS_LoadVideo outputs:
        //   slot 0 = IMAGE, slot 1 = frame_count (INT),
        //   slot 2 = audio (AUDIO), slot 3 = video_info
        for (const node of buckets.load) {
            const newNode = replacements.get(node.id);
            if (!newNode) continue;
            const conns = connMap.get(node.id);

            // Reconnect inputs (upstream -> new node)
            // Native LoadVideo has no upstream link inputs
            // (only widget: file combo), so nothing to rewire.

            // Reconnect outputs (new node -> downstream)
            // Old node had single output: slot 0 = VIDEO
            for (const out of conns.outputs) {
                const targetNode = graph.getNodeById(out.targetNodeId);
                if (!targetNode) continue;

                // If target is GetVideoComponents being collapsed,
                // skip -- we handle it in Phase C3
                if (gvcIds.has(out.targetNodeId)) continue;

                // If target is SaveVideo being replaced,
                // skip -- we handle it in C2
                if (saveIds.has(out.targetNodeId)) continue;

                // Target expects IMAGE? Connect slot 0
                const targetInput = targetNode.inputs?.[
                    out.targetSlot
                ];
                if (targetInput) {
                    const tType = (targetInput.type || "").toUpperCase();
                    if (
                        tType === "IMAGE"
                        || tType === "*"
                        || tType === ""
                    ) {
                        newNode.connect(0, targetNode, out.targetSlot);
                    } else {
                        console.warn(
                            "VHS Swap: cannot reconnect"
                            + ` VHS_LoadVideo IMAGE output to`
                            + ` ${getNodeType(targetNode)}`
                            + ` input "${targetInput.name}"`
                            + ` (type: ${tType})`
                        );
                    }
                }
            }
        }

        // C2: reconnect SaveVideo replacements (VHS_VideoCombine)
        // VHS_VideoCombine inputs:
        //   slot 0 = images (IMAGE), slot 7 = audio (AUDIO)
        //   (optional, after widgets)
        // Native SaveVideo inputs:
        //   slot 0 = video (VIDEO), slot 1 = filename_prefix,
        //   slot 2 = format, slot 3 = codec
        for (const node of buckets.save) {
            const newNode = replacements.get(node.id);
            if (!newNode) continue;
            const conns = connMap.get(node.id);

            for (const inp of conns.inputs) {
                // Only slot 0 (video) is a link input on SaveVideo
                if (inp.slotIndex !== 0) continue;

                const originNode = graph.getNodeById(inp.originNodeId);
                if (!originNode) continue;

                // If origin is CreateVideo being collapsed,
                // trace through to its sources in Phase C4
                if (createIds.has(inp.originNodeId)) continue;

                // If origin is a replaced LoadVideo
                if (replacements.has(inp.originNodeId)) {
                    const srcNew = replacements.get(inp.originNodeId);
                    // IMAGE output slot 0 -> images input
                    srcNew.connect(0, newNode, 0);
                    // Also connect audio: slot 2 -> audio input
                    connectToNamedInput(srcNew, 2, newNode, "audio");
                    continue;
                }

                // Origin outputs IMAGE? Connect to images (slot 0)
                const originOutput = originNode.outputs?.[
                    inp.originSlot
                ];
                if (originOutput) {
                    const oType = (
                        originOutput.type || ""
                    ).toUpperCase();
                    if (
                        oType === "IMAGE"
                        || oType === "*"
                        || oType === ""
                    ) {
                        originNode.connect(
                            inp.originSlot, newNode, 0
                        );
                    } else {
                        console.warn(
                            "VHS Swap: cannot reconnect"
                            + ` ${getNodeType(originNode)}`
                            + ` output "${originOutput.name}"`
                            + ` (type: ${oType}) to`
                            + ` VHS_VideoCombine images input`
                        );
                    }
                }
            }
        }

        // C3: collapse GetVideoComponents
        // GVC inputs: slot 0 = video (VIDEO)
        // GVC outputs: slot 0 = IMAGE, slot 1 = AUDIO, slot 2 = fps
        for (const node of buckets.getComponents) {
            const conns = connMap.get(node.id);
            // Find the upstream source
            const upstreamConn = conns.inputs.find(
                (c) => c.slotIndex === 0
            );
            let srcNode = null;
            let srcIsReplaced = false;
            if (upstreamConn) {
                if (replacements.has(upstreamConn.originNodeId)) {
                    srcNode = replacements.get(
                        upstreamConn.originNodeId
                    );
                    srcIsReplaced = true;
                } else {
                    srcNode = graph.getNodeById(
                        upstreamConn.originNodeId
                    );
                }
            }

            // Rewire each output
            for (const out of conns.outputs) {
                const targetNode = graph.getNodeById(
                    out.targetNodeId
                );
                if (!targetNode) continue;
                // If target was also replaced, use the replacement
                const finalTarget = replacements.has(
                    out.targetNodeId
                )
                    ? replacements.get(out.targetNodeId)
                    : targetNode;
                const finalSlot = replacements.has(out.targetNodeId)
                    ? findInputSlotByType(finalTarget, out.type)
                    : out.targetSlot;

                if (!srcNode || !srcIsReplaced) {
                    console.warn(
                        "VHS Swap: GetVideoComponents upstream"
                        + " was not replaced; cannot collapse"
                    );
                    continue;
                }

                // Map GVC output slots to VHS_LoadVideo slots:
                //   GVC 0 (IMAGE) -> VHS 0 (IMAGE)
                //   GVC 1 (AUDIO) -> VHS 2 (audio)
                //   GVC 2 (fps)   -> no direct equivalent
                if (out.slotIndex === 0) {
                    srcNode.connect(0, finalTarget, finalSlot);
                } else if (out.slotIndex === 1) {
                    srcNode.connect(2, finalTarget, finalSlot);
                } else if (out.slotIndex === 2) {
                    console.warn(
                        "VHS Swap: fps output from"
                        + " GetVideoComponents has no direct"
                        + " VHS equivalent; connection dropped"
                    );
                }
            }
        }

        // C4: collapse CreateVideo
        // CV inputs: slot 0 = images (IMAGE),
        //            slot 1 = fps (FLOAT),
        //            slot 2 = audio (AUDIO, optional)
        // CV outputs: slot 0 = VIDEO
        for (const node of buckets.create) {
            const conns = connMap.get(node.id);

            // Find upstream sources
            const imgConn = conns.inputs.find(
                (c) => c.slotIndex === 0
            );
            const fpsConn = conns.inputs.find(
                (c) => c.slotIndex === 1
            );
            const audioConn = conns.inputs.find(
                (c) => c.slotIndex === 2
            );

            // Find downstream targets (VIDEO output)
            for (const out of conns.outputs) {
                let finalTarget = graph.getNodeById(
                    out.targetNodeId
                );
                if (!finalTarget) continue;
                // If target was replaced (SaveVideo->VHS_Combine)
                if (replacements.has(out.targetNodeId)) {
                    finalTarget = replacements.get(
                        out.targetNodeId
                    );
                }

                // Connect IMAGE source -> target images input
                if (imgConn) {
                    let imgSrc = replacements.has(
                        imgConn.originNodeId
                    )
                        ? replacements.get(imgConn.originNodeId)
                        : graph.getNodeById(imgConn.originNodeId);
                    let imgSlot = replacements.has(
                        imgConn.originNodeId
                    )
                        ? 0  // VHS_LoadVideo IMAGE = slot 0
                        : imgConn.originSlot;
                    if (imgSrc) {
                        const tSlot = findInputSlotByName(
                            finalTarget, "images"
                        );
                        imgSrc.connect(
                            imgSlot,
                            finalTarget,
                            tSlot >= 0 ? tSlot : out.targetSlot
                        );
                    }
                }

                // Connect AUDIO source -> target audio input
                if (audioConn) {
                    let audioSrc = replacements.has(
                        audioConn.originNodeId
                    )
                        ? replacements.get(audioConn.originNodeId)
                        : graph.getNodeById(
                            audioConn.originNodeId
                        );
                    let audioSlot = replacements.has(
                        audioConn.originNodeId
                    )
                        ? 2  // VHS_LoadVideo audio = slot 2
                        : audioConn.originSlot;
                    if (audioSrc) {
                        connectToNamedInput(
                            audioSrc, audioSlot, finalTarget, "audio"
                        );
                    }
                }

                // Transfer fps widget value if available
                if (fpsConn) {
                    // fps was a link -- find the source value
                    // and set frame_rate widget on target
                    const fpsSrc = graph.getNodeById(
                        fpsConn.originNodeId
                    );
                    if (fpsSrc) {
                        const fpsOut = fpsSrc.outputs?.[
                            fpsConn.originSlot
                        ];
                        if (fpsOut) {
                            // Try to get the value from widgets
                            const fpsW = fpsSrc.widgets?.find(
                                (w) => w.name === "fps"
                                    || w.name === "value"
                            );
                            if (fpsW) {
                                const rateW = finalTarget
                                    .widgets?.find(
                                        (w) => w.name === "frame_rate"
                                    );
                                if (rateW) {
                                    rateW.value = fpsW.value;
                                }
                            }
                        }
                    }
                } else {
                    // fps came from a widget on CreateVideo itself
                    const fpsWidget = node.widgets?.find(
                        (w) => w.name === "fps"
                    );
                    if (fpsWidget) {
                        const rateW = finalTarget.widgets?.find(
                            (w) => w.name === "frame_rate"
                        );
                        if (rateW) {
                            rateW.value = fpsWidget.value;
                        }
                    }
                }
            }
        }

        // Phase D: remove all original nodes
        const toRemove = [
            ...buckets.load,
            ...buckets.save,
            ...buckets.getComponents,
            ...buckets.create,
        ];
        for (const node of toRemove) {
            graph.remove(node);
        }

        // Resize replacement nodes to fit widgets
        for (const newNode of replacements.values()) {
            newNode.size = newNode.computeSize();
        }

        if (totalTargets > 0) {
            console.log(
                `VHS Swap: replaced ${buckets.load.length} LoadVideo,`
                + ` ${buckets.save.length} SaveVideo;`
                + ` collapsed ${buckets.getComponents.length}`
                + ` GetVideoComponents, ${buckets.create.length} CreateVideo`
            );
        }

        // -- Step 4: always wire the VHS_VideoCombine(s): mp4/crf13 and
        // loaded_fps in via a VHS_VideoInfo + bottom reroutes.
        wireCombineFps(graph);

        graph.setDirtyCanvas(true, true);
    } finally {
        canvas.emitAfterChange();
    }
}

/**
 * Connect a source node's output slot to a target node's
 * input found by name.
 */
function connectToNamedInput(srcNode, srcSlot, dstNode, inputName) {
    const idx = findInputSlotByName(dstNode, inputName);
    if (idx >= 0) {
        srcNode.connect(srcSlot, dstNode, idx);
    }
}

/**
 * Find an input slot index by name on a node.
 * Returns -1 if not found.
 */
function findInputSlotByName(node, name) {
    if (!node.inputs) return -1;
    for (let i = 0; i < node.inputs.length; i++) {
        if (node.inputs[i].name === name) return i;
    }
    return -1;
}

/**
 * Find an input slot index by type on a node.
 * Returns the original slot index if not found.
 */
function findInputSlotByType(node, type) {
    if (!node.inputs) return 0;
    const upper = (type || "").toUpperCase();
    for (let i = 0; i < node.inputs.length; i++) {
        const iType = (node.inputs[i].type || "").toUpperCase();
        if (iType === upper || iType === "*") return i;
    }
    return 0;
}

// -- Combine fps / mp4 wiring (runs after every swap) --

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
            "[VHSSwap] frame_rate widget not found on Combine."
        );
        return -1;
    }

    if (!convertWidgetToInput) {
        console.warn(
            "[VHSSwap] convertToInput helper unavailable;"
            + " cannot convert frame_rate widget to input."
        );
        return -1;
    }

    const config = getInputConfig(combineNode, "frame_rate")
        ?? ["FLOAT", { default: 8, min: 1, step: 1 }];

    try {
        convertWidgetToInput(combineNode, widget, config);
    } catch (err) {
        console.warn("[VHSSwap] convertToInput threw:", err);
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
 * workflow. Falls back to a direct connection if Reroutes can't be made.
 */
function createRerouteChain(
    graph, sourceNode, sourceSlot, combineNode, frameRateSlot
) {
    const rerouteA = LiteGraph.createNode(REROUTE_TYPE);
    const rerouteB = LiteGraph.createNode(REROUTE_TYPE);
    if (!rerouteA || !rerouteB) {
        console.warn(
            "[VHSSwap] Could not create Reroute nodes;"
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
 * Set the Combine `format` to h264-mp4 (firing its callback so VHS adds
 * the format-specific widgets), then set crf.
 */
function applyH264Settings(combineNode) {
    const formatW = combineNode.widgets?.find((w) => w.name === "format");
    if (!formatW) {
        console.warn("[VHSSwap] format widget not found on Combine.");
        return;
    }

    if (formatW.value !== TARGET_FORMAT) {
        formatW.value = TARGET_FORMAT;
        if (formatW.callback) formatW.callback(TARGET_FORMAT);
    }

    const crfW = combineNode.widgets?.find((w) => w.name === "crf");
    if (crfW) {
        crfW.value = TARGET_CRF;
        if (crfW.callback) crfW.callback(TARGET_CRF);
    } else {
        console.warn(
            "[VHSSwap] crf widget did not appear after format change."
        );
    }

    combineNode.setDirtyCanvas(true, true);
}

/**
 * Default the frame_rate to DEFAULT_FPS. Only works while frame_rate is
 * still a widget (the no-Load-Video path never converts it to an input).
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
            "[VHSSwap] frame_rate is an input slot, not a widget;"
            + " leaving it as-is (no Load Video to source fps)."
        );
    }
}

/**
 * For every VHS_VideoCombine in the graph: set mp4/crf13, and (if there
 * is exactly one VHS Load Video) run its loaded_fps through a VHS_VideoInfo
 * and two bottom reroutes into the Combine's frame_rate. Otherwise default
 * frame_rate to 24. Idempotent: an existing Reroute feed is left intact; a
 * stale direct loaded_fps->frame_rate link is rebuilt through the reroutes.
 */
function wireCombineFps(graph) {
    const nodes = graph._nodes || [];
    const combines = nodes.filter((n) => getNodeType(n) === VHS_COMBINE);
    if (combines.length === 0) return;

    const loads = nodes.filter((n) => VHS_LOAD_TYPES.has(getNodeType(n)));
    const loadNode = loads.length === 1 ? loads[0] : null;

    for (const combineNode of combines) {
        // Always: h264-mp4 / crf 13.
        applyH264Settings(combineNode);

        // No single Load Video to source fps -> default to 24.
        if (!loadNode) {
            applyDefaultFps(combineNode);
            continue;
        }

        // Ensure a VideoInfo node hangs off the Load Video.
        let infoNode = findExistingVideoInfo(loadNode, graph);
        if (!infoNode) {
            infoNode = LiteGraph.createNode(VHS_VIDEO_INFO);
            if (!infoNode) {
                console.warn("[VHSSwap] Failed to create VHS_VideoInfo.");
                applyDefaultFps(combineNode);
                continue;
            }
            infoNode.pos = [
                loadNode.pos[0] + (loadNode.size?.[0] || 240) + 40,
                loadNode.pos[1],
            ];
            graph.add(infoNode);
            loadNode.connect(LOAD_VIDEO_INFO_SLOT, infoNode, 0);
        }

        // Convert frame_rate to an input.
        const frameRateSlot = ensureFrameRateInput(combineNode);
        if (frameRateSlot < 0) continue;

        // loaded_fps -> two bottom reroutes -> frame_rate (idempotent).
        const frInput = combineNode.inputs?.[frameRateSlot];
        const existingLink = (frInput && frInput.link != null)
            ? graph.links[frInput.link]
            : null;
        const existingSource = existingLink
            ? graph.getNodeById(existingLink.origin_id)
            : null;

        if (
            existingSource
            && getNodeType(existingSource) === REROUTE_TYPE
        ) {
            // Already routed through a Reroute; leave it intact.
            continue;
        }
        if (existingLink) {
            // Drop a stale direct connection before re-routing.
            combineNode.disconnectInput(frameRateSlot);
        }
        createRerouteChain(
            graph, infoNode, INFO_LOADED_FPS_SLOT,
            combineNode, frameRateSlot
        );
    }
}

// -- Extension registration --

app.registerExtension({
    name: "TrentNodes.VHSSwap",

    commands: [
        {
            id: "TrentNodes.VHSSwap",
            label: "Swap Native Video to VHS (+ wire fps/mp4)",
            icon: "pi pi-refresh",
            function: () => vhsSwap(),
        },
    ],

    keybindings: [
        {
            commandId: "TrentNodes.VHSSwap",
            combo: { key: "v", shift: true },
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.VHSSwap"],
        },
    ],
});
