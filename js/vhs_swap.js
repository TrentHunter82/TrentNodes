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
    if (totalTargets === 0) return;

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

        graph.setDirtyCanvas(true, true);

        const counts = {
            load: buckets.load.length,
            save: buckets.save.length,
            gvc: buckets.getComponents.length,
            create: buckets.create.length,
        };
        console.log(
            `VHS Swap: replaced ${counts.load} LoadVideo,`
            + ` ${counts.save} SaveVideo;`
            + ` collapsed ${counts.gvc} GetVideoComponents,`
            + ` ${counts.create} CreateVideo`
        );
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

// -- Extension registration --

app.registerExtension({
    name: "TrentNodes.VHSSwap",

    commands: [
        {
            id: "TrentNodes.VHSSwap",
            label: "Swap Native Video to VHS",
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
