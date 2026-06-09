import { app } from "../../scripts/app.js";

/**
 * Reroute Roundup
 *
 * Shift+Alt+R — converts every native "circle" reroute in the active
 * graph back into the classic legacy `Reroute` node (the little pill
 * with an input dot on the left and an output dot on the right).
 *
 * Native reroutes are visual waypoints stored on real links
 * (graph.reroutes), not nodes. The legacy Reroute is an actual node.
 * For each native reroute we drop a legacy node at the same spot and
 * re-wire the path through it, preserving every connection (including
 * chains and fan-out).
 *
 * How the native model maps to nodes:
 *   - reroute.parentId points UPSTREAM (toward the source output);
 *     a reroute with no parentId is a "root" sitting on an output.
 *   - link.parentId is the LAST reroute before the real input, so a
 *     reroute directly feeds an input iff link.parentId === reroute.id.
 *
 * So each native reroute becomes one legacy node whose:
 *   - input  ← parent reroute's node (chain) or the real source output (root)
 *   - output → every real input it directly feeds, and (implicitly, via
 *              their own input wiring) every child reroute node.
 */

const LEGACY_TYPE = "Reroute";

/** The graph currently shown on the canvas (may be a subgraph). */
function getActiveGraph() {
    return (app.canvas && app.canvas.graph) || app.graph;
}

/** graph.links is a Map proxied to also allow bracket access — use .get. */
function getLink(graph, id) {
    return graph.links.get ? graph.links.get(id) : graph.links[id];
}

/**
 * Capture everything we need about every connected native reroute BEFORE
 * touching the graph. Re-wiring disconnects the original links, which the
 * engine uses to auto-delete native reroutes — so we must not read live
 * reroute state mid-conversion.
 */
function snapshotReroutes(graph) {
    const out = [];

    for (const reroute of graph.reroutes.values()) {
        const linkIds = [...reroute.linkIds];
        // Floating / dangling reroutes carry no real connection to preserve.
        if (linkIds.length === 0) continue;

        const hasParent =
            reroute.parentId != null && graph.reroutes.has(reroute.parentId);

        // Root reroutes connect straight to a real output. Every link through
        // a reroute shares the same origin, so the first link is enough.
        let source = null;
        if (!hasParent) {
            const firstLink = getLink(graph, linkIds[0]);
            if (firstLink) {
                source = {
                    nodeId: firstLink.origin_id,
                    slot: firstLink.origin_slot,
                };
            }
        }

        // Real inputs fed directly by this reroute (it is their last hop).
        const targets = [];
        for (const lid of linkIds) {
            const link = getLink(graph, lid);
            if (link && link.parentId === reroute.id) {
                targets.push({ nodeId: link.target_id, slot: link.target_slot });
            }
        }

        out.push({
            id: reroute.id,
            pos: [reroute.pos[0], reroute.pos[1]],
            parentId: hasParent ? reroute.parentId : null,
            source,
            targets,
        });
    }

    return out;
}

/** Order roots first so link types/colours settle as we wire downstream. */
function orderByDepth(snapshots) {
    const byId = new Map(snapshots.map((s) => [s.id, s]));
    const depth = new Map();

    const resolve = (s) => {
        if (depth.has(s.id)) return depth.get(s.id);
        depth.set(s.id, 0); // cycle guard
        let d = 0;
        if (s.parentId != null && byId.has(s.parentId)) {
            d = resolve(byId.get(s.parentId)) + 1;
        }
        depth.set(s.id, d);
        return d;
    };

    for (const s of snapshots) resolve(s);
    return [...snapshots].sort((a, b) => depth.get(a.id) - depth.get(b.id));
}

function convertNativeReroutesToLegacy() {
    const LiteGraph = window.LiteGraph;
    const graph = getActiveGraph();

    if (!graph || !graph.reroutes || graph.reroutes.size === 0) {
        console.log("[RerouteRoundup] No native reroutes in the active graph.");
        return 0;
    }

    const snapshots = snapshotReroutes(graph);
    if (snapshots.length === 0) {
        console.log("[RerouteRoundup] No connected native reroutes to convert.");
        return 0;
    }

    const canvas = app.canvas;
    canvas?.emitBeforeChange?.();
    try {
        // Phase 1 — create a legacy node for each native reroute, centred on it.
        const nodeByRerouteId = new Map();
        for (const snap of snapshots) {
            const node = LiteGraph.createNode(LEGACY_TYPE);
            if (!node) {
                console.error(
                    "[RerouteRoundup] Could not create a legacy 'Reroute' node — "
                    + "is the core Comfy.RerouteNode extension present?"
                );
                return 0;
            }
            graph.add(node);
            node.pos = [
                snap.pos[0] - node.size[0] / 2,
                snap.pos[1] - node.size[1] / 2,
            ];
            nodeByRerouteId.set(snap.id, node);
        }

        // Phase 2 — wire each legacy node's input and direct outputs.
        for (const snap of orderByDepth(snapshots)) {
            const node = nodeByRerouteId.get(snap.id);

            // Input: chain from the parent reroute's node, or the real source.
            if (snap.parentId != null) {
                nodeByRerouteId.get(snap.parentId)?.connect(0, node, 0);
            } else if (snap.source) {
                const origin = graph.getNodeById(snap.source.nodeId);
                origin?.connect(snap.source.slot, node, 0);
            }

            // Output: every real input this reroute directly feeds. Re-using an
            // already-connected input swaps out the original native-routed link.
            for (const t of snap.targets) {
                const target = graph.getNodeById(t.nodeId);
                if (target) node.connect(0, target, t.slot);
            }
        }

        // Phase 3 — drop any native reroute that survived re-wiring. removeReroute
        // just extracts it from the chain; the real links were already replaced.
        for (const snap of snapshots) {
            if (graph.reroutes.has(snap.id)) graph.removeReroute(snap.id);
        }
    } finally {
        canvas?.emitAfterChange?.();
    }

    app.graph.setDirtyCanvas(true, true);
    console.log(
        `[RerouteRoundup] Converted ${snapshots.length} `
        + `native reroute(s) to legacy Reroute node(s).`
    );
    return snapshots.length;
}

app.registerExtension({
    name: "TrentNodes.RerouteRoundup",

    commands: [
        {
            id: "TrentNodes.RerouteRoundup",
            label: "Convert Circle Reroutes → Legacy Reroute Nodes",
            icon: "pi pi-arrows-h",
            function: convertNativeReroutesToLegacy,
        },
    ],

    keybindings: [
        {
            commandId: "TrentNodes.RerouteRoundup",
            combo: { key: "r", shift: true, alt: true },
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.RerouteRoundup"],
        },
    ],
});
