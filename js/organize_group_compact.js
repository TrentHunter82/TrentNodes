import { app } from "../../scripts/app.js";
import { findSourceBundles, packSourceBundle } from "./source_subgraphs.js";

/**
 * Organize Group Compact
 *
 * Shift+Alt+O — for each selected group, lays out its child
 * nodes left-to-right by connection depth (topological rank),
 * then resizes the group to wrap the result. Direct input nodes in the same group can
 * be packed with their consumer into a subgraph; unconnected inputs
 * are exposed automatically, with sources stacked above the consumer.
 *
 * Nodes follow their downstream socket positions. Shared providers
 * use the middle of their targets, and columns pack around those
 * positions without overlapping measured titles or node bodies.
 * Sinks and disconnected nodes retain their vertical order.
 *
 * Leaf nodes (no in-group predecessors) are pulled rightward to
 * sit one column to the left of their earliest target, so loose
 * inputs stay close to where they plug in instead of piling up
 * in column 0.
 */

// Uniform padding between the group's inner edges and its contents, matching
// ComfyUI's native "Fit Group To Contents" (its default is 10; we use 25).
const PADDING = 25;
const ROW_GAP = 40;
const AUTO_SUBGRAPHS = "TrentNodes.OrganizeGroupCompact.AutoSubgraphs";
const SPACING = "TrentNodes.OrganizeGroupCompact.Spacing";

function getTitleHeight() {
    return (window.LiteGraph && window.LiteGraph.NODE_TITLE_HEIGHT) || 30;
}

function isCollapsed(node) {
    return !!(node.flags && node.flags.collapsed);
}

function getSelectedGroups(canvas) {
    const out = [];
    const seen = new Set();
    const push = (g) => {
        if (g && !seen.has(g)) { seen.add(g); out.push(g); }
    };

    if (canvas.selected_groups) {
        if (canvas.selected_groups instanceof Set
            || Array.isArray(canvas.selected_groups)) {
            for (const g of canvas.selected_groups) push(g);
        } else if (typeof canvas.selected_groups === "object") {
            for (const k of Object.keys(canvas.selected_groups)) {
                push(canvas.selected_groups[k]);
            }
        }
    }
    push(canvas.selected_group);

    if (out.length === 0 && canvas.selectedItems) {
        for (const item of canvas.selectedItems) {
            if (window.LiteGraph
                && item instanceof window.LiteGraph.LGraphGroup) {
                push(item);
            }
        }
    }

    return out;
}

function getLink(graph, id) {
    if (!graph || !graph.links) return null;
    if (typeof graph.links.get === "function") return graph.links.get(id);
    return graph.links[id];
}

/**
 * Build successor / predecessor adjacency restricted to nodes
 * inside the group. Self-loops are ignored.
 */
function buildAdjacency(nodes, graph) {
    const inGroup = new Set(nodes.map((n) => n.id));
    const succ = new Map();
    const pred = new Map();
    for (const n of nodes) {
        succ.set(n.id, new Set());
        pred.set(n.id, new Set());
    }
    for (const n of nodes) {
        if (!n.outputs) continue;
        for (const out of n.outputs) {
            const links = out && out.links;
            if (!links) continue;
            for (const linkId of links) {
                const link = getLink(graph, linkId);
                if (!link) continue;
                const tid = link.target_id;
                if (tid === n.id) continue;
                if (inGroup.has(tid)) {
                    succ.get(n.id).add(tid);
                    pred.get(tid).add(n.id);
                }
            }
        }
    }
    return { succ, pred };
}

/**
 * Assign each node a column index = longest path from any source
 * (a node with no in-group predecessors). Handles DAGs cleanly;
 * cycles are broken by leaving back-edges out of the rank update.
 */
function computeRanks(nodes, succ, pred) {
    const indeg = new Map();
    for (const n of nodes) indeg.set(n.id, pred.get(n.id).size);

    const rank = new Map();
    for (const n of nodes) rank.set(n.id, 0);

    const queue = [];
    for (const n of nodes) {
        if (indeg.get(n.id) === 0) queue.push(n.id);
    }

    const visited = new Set();
    while (queue.length > 0) {
        const id = queue.shift();
        if (visited.has(id)) continue;
        visited.add(id);
        for (const sid of succ.get(id)) {
            const candidate = rank.get(id) + 1;
            if (candidate > rank.get(sid)) rank.set(sid, candidate);
            indeg.set(sid, indeg.get(sid) - 1);
            if (indeg.get(sid) === 0) queue.push(sid);
        }
    }

    // Cycle fallback: place any remaining node one column past its
    // already-visited predecessors (or column 0 if none).
    let safety = nodes.length + 1;
    while (visited.size < nodes.length && safety-- > 0) {
        for (const n of nodes) {
            if (visited.has(n.id)) continue;
            let r = 0;
            for (const pid of pred.get(n.id)) {
                if (visited.has(pid)) r = Math.max(r, rank.get(pid) + 1);
            }
            rank.set(n.id, r);
            visited.add(n.id);
        }
    }

    return rank;
}

function median(values) {
    values.sort((a, b) => a - b);
    const middle = Math.floor(values.length / 2);
    return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

function nodeGeometry(node, ctx) {
    node.updateArea(ctx);
    const [x, y, width, height] = node.boundingRect;
    const inputs = (node.inputs || []).map((_, i) => node.getInputPos(i));
    const outputs = (node.outputs || []).map((_, i) => node.getOutputPos(i));
    return {
        width, height, dx: x - node.pos[0], dy: y - node.pos[1],
        inputs: inputs.map((p) => p[1] - node.pos[1]), inputXs: inputs.map((p) => p[0] - node.pos[0]),
        outputs: outputs.map((p) => p[1] - node.pos[1]), outputXs: outputs.map((p) => p[0] - node.pos[0]),
    };
}

function layoutQuality(nodes, edges, geometry) {
    const segments = edges.map((e) => ({ edge: e,
        a: [e.source.pos[0] + e.outputX, e.source.pos[1] + e.outputY],
        b: [e.target.pos[0] + e.inputX, e.target.pos[1] + e.inputY],
    }));
    const side = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let crossings = 0, length = 0;
    for (const [i, a] of segments.entries()) {
        length += Math.hypot(a.b[0] - a.a[0], a.b[1] - a.a[1]);
        for (let j = i + 1; j < segments.length; j++) {
            const b = segments[j];
            if (a.edge.source === b.edge.source || a.edge.target === b.edge.target) continue;
            if (side(a.a, a.b, b.a) * side(a.a, a.b, b.b) < -0.001
                && side(b.a, b.b, a.a) * side(b.a, b.b, a.b) < -0.001) crossings++;
        }
    }
    const right = Math.max(...nodes.map((n) => n.pos[0] + geometry.get(n).dx + geometry.get(n).width));
    const bottom = Math.max(...nodes.map((n) => n.pos[1] + geometry.get(n).dy + geometry.get(n).height));
    const left = Math.min(...nodes.map((n) => n.pos[0] + geometry.get(n).dx));
    const top = Math.min(...nodes.map((n) => n.pos[1] + geometry.get(n).dy));
    return [crossings, Math.round(length * 100) / 100, (right - left) * (bottom - top)];
}

function placeColumn(column, desired, geometry, top, heightLimit, gap) {
    const offsets = [];
    const blocks = [];
    let height = 0;
    for (const [i, node] of column.entries()) {
        offsets.push(height);
        blocks.push({ start: i, end: i, sum: desired.get(node) - height, count: 1 });
        height += geometry.get(node).height + gap;
        // Pool clashing positions instead of pushing every later node downward.
        while (blocks.length > 1) {
            const b = blocks.at(-1), a = blocks.at(-2);
            if (a.sum / a.count <= b.sum / b.count) break;
            blocks.splice(-2, 2, { start: a.start, end: b.end, sum: a.sum + b.sum, count: a.count + b.count });
        }
    }
    const ceiling = top + Math.max(0, heightLimit - Math.max(0, height - gap));
    for (const block of blocks) {
        const y = Math.max(top, Math.min(ceiling, block.sum / block.count));
        for (let i = block.start; i <= block.end; i++) {
            const node = column[i];
            node.pos[1] = y + offsets[i] - geometry.get(node).dy;
        }
    }
}

function compactColumns(columns, edges, geometry, left, gap, rowGap) {
    const placed = [];
    const parents = new Map();
    for (const edge of edges) {
        if (!parents.has(edge.target)) parents.set(edge.target, new Set());
        parents.get(edge.target).add(edge.source);
    }
    for (const column of columns) {
        for (const node of column) {
            const box = geometry.get(node);
            const top = node.pos[1] + box.dy;
            let x = left;
            for (const other of placed) {
                const otherBox = geometry.get(other);
                const otherTop = other.pos[1] + otherBox.dy;
                const overlapsY = top < otherTop + otherBox.height + rowGap
                    && top + box.height + rowGap > otherTop;
                if (overlapsY || parents.get(node)?.has(other)) {
                    x = Math.max(x, other.pos[0] + otherBox.dx + otherBox.width + gap);
                }
            }
            node.pos[0] = x - box.dx;
        }
        placed.push(...column);
    }
    for (const node of [...placed].reverse()) {
        if (!isCollapsed(node)) continue;
        const targets = edges.filter((e) => e.source === node).map((e) => e.target);
        if (!targets.length) continue;
        const box = geometry.get(node), top = node.pos[1] + box.dy;
        const obstacles = placed.filter((other) => {
            const b = geometry.get(other), y = other.pos[1] + b.dy;
            return other.pos[0] + b.dx > node.pos[0] + box.dx
                && top < y + b.height + rowGap && top + box.height + rowGap > y;
        });
        const right = Math.min(...[...targets, ...obstacles].map((n) => n.pos[0] + geometry.get(n).dx)) - gap;
        node.pos[0] = Math.max(node.pos[0], right - box.width - box.dx);
    }
}

function arrangeGroup(group, graph) {
    if (!group || typeof group.recomputeInsideNodes !== "function") return;
    group.recomputeInsideNodes();

    const LGNode = window.LiteGraph && window.LiteGraph.LGraphNode;
    const children = Array.from(group._children || []);
    const nodes = LGNode
        ? children.filter((c) => c instanceof LGNode)
        : children;
    if (nodes.length === 0) return;

    const ctx = app.canvas && app.canvas.ctx;
    const geometry = new Map(nodes.map((n) => [n, nodeGeometry(n, ctx)]));
    const relaxed = app.extensionManager.setting.get(SPACING) === "Relaxed";
    const colGap = relaxed ? 48 : 28;
    const rowGap = relaxed ? 24 : 10;

    const { succ, pred } = buildAdjacency(nodes, graph);
    const rank = computeRanks(nodes, succ, pred);

    // Pull leaves toward their targets: a node with no in-group
    // predecessors snaps to (earliest_target_rank - 1) instead of
    // sitting at column 0. This is safe because the leaf's
    // contribution to its targets' longest-path rank is unchanged
    // (target_rank >= leaf_rank + 1 still holds).
    for (const n of nodes) {
        if (pred.get(n.id).size > 0) continue;
        const successors = succ.get(n.id);
        if (!successors || successors.size === 0) continue;
        let minTargetRank = Infinity;
        for (const sid of successors) {
            const r = rank.get(sid);
            if (r != null && r < minTargetRank) minTargetRank = r;
        }
        if (minTargetRank > 1 && Number.isFinite(minTargetRank)) {
            rank.set(n.id, minTargetRank - 1);
        }
    }

    const ranks = [...new Set(nodes.map((n) => rank.get(n.id)))].sort((a, b) => a - b);
    const cols = ranks.map((r) => nodes.filter((n) => rank.get(n.id) === r));
    const numCols = cols.length;
    const members = new Set(nodes);
    const edges = nodes.flatMap((source) => (source.outputs || []).flatMap((output, slot) =>
        (output.links || []).flatMap((id) => {
            const link = getLink(graph, id);
            const target = link && graph.getNodeById(link.target_id);
            return members.has(target) && rank.get(source.id) < rank.get(target.id)
                ? [{ source, target, inputSlot: link.target_slot,
                    outputY: geometry.get(source).outputs[slot], inputY: geometry.get(target).inputs[link.target_slot],
                    outputX: geometry.get(source).outputXs[slot], inputX: geometry.get(target).inputXs[link.target_slot] }]
                : [];
        })));

    const titleH = group.titleHeight ?? getTitleHeight();
    const originX = group.pos[0] + PADDING;
    const originY = group.pos[1] + titleH + PADDING;
    const heightLimit = Math.max(...cols.map((col) =>
        col.reduce((height, n) => height + geometry.get(n).height + rowGap, -rowGap)));

    let best, bestQuality;
    const consider = () => {
        const dx = originX - Math.min(...nodes.map((n) => n.pos[0] + geometry.get(n).dx));
        const dy = originY - Math.min(...nodes.map((n) => n.pos[1] + geometry.get(n).dy));
        for (const node of nodes) { node.pos[0] += dx; node.pos[1] += dy; }
        const quality = layoutQuality(nodes, edges, geometry);
        const difference = bestQuality && quality.findIndex((v, i) => Math.abs(v - bestQuality[i]) > 0.001);
        if (!bestQuality || (difference >= 0 && quality[difference] < bestQuality[difference])) {
            bestQuality = quality;
            best = new Map(nodes.map((n) => [n, [...n.pos]]));
        }
    };
    const originals = new Map(nodes.map((n) => [n, [...n.pos]]));

    // Keep a compact version of the original input ordering as a candidate.
    for (let c = numCols - 1; c >= 0; c--) {
        const keys = new Map(cols[c].map((node) => {
            const targets = edges.filter((e) => e.source === node)
                .sort((a, b) => a.target.pos[1] - b.target.pos[1] || a.inputSlot - b.inputSlot);
            return [node, targets.length ? [targets[0].target.pos[1], targets[0].inputSlot] : [c === numCols - 1 ? node.pos[1] : Infinity, 0]];
        }));
        cols[c].sort((a, b) => keys.get(a)[0] - keys.get(b)[0] || keys.get(a)[1] - keys.get(b)[1]
            || (keys.get(a)[0] === Infinity ? a.pos[1] - b.pos[1] : 0) || String(a.id).localeCompare(String(b.id)));
        let y = originY;
        for (const node of cols[c]) {
            node.pos[1] = y - geometry.get(node).dy;
            y += geometry.get(node).height + rowGap + (isCollapsed(node) ? getTitleHeight() : 0);
        }
    }
    let x = originX;
    for (const column of cols) {
        const width = Math.max(...column.map((n) => geometry.get(n).width));
        for (const node of column) {
            const box = geometry.get(node);
            node.pos[0] = x + (isCollapsed(node) ? width - box.width : 0) - box.dx;
        }
        x += width + Math.max(40, colGap);
    }
    consider();
    compactColumns(cols, edges, geometry, originX, colGap, rowGap);
    consider();
    for (const node of nodes) node.pos = [...originals.get(node)];

    // Place the rightmost column first, top-to-bottom by current y.
    // It acts as the anchor that everything to the left orders against.
    if (numCols > 0) {
        const rc = numCols - 1;
        cols[rc].sort((a, b) => a.pos[1] - b.pos[1] || String(a.id).localeCompare(String(b.id)));
        let y = originY;
        for (const node of cols[rc]) {
            node.pos[1] = y - geometry.get(node).dy;
            y += geometry.get(node).height + rowGap;
        }
    }

    // Target socket order controls rows; actual output offsets control alignment.
    for (let c = numCols - 2; c >= 0; c--) {
        const keys = new Map(), desired = new Map();
        for (const node of cols[c]) {
            const links = edges.filter((e) => e.source === node);
            keys.set(node, links.length ? median(links.map((e) => e.target.pos[1] + e.inputY)) : Infinity);
            desired.set(node, links.length
                ? median(links.map((e) => e.target.pos[1] + e.inputY - e.outputY)) + geometry.get(node).dy
                : originY);
        }
        cols[c].sort((a, b) => (keys.get(a) - keys.get(b)) || a.pos[1] - b.pos[1] || String(a.id).localeCompare(String(b.id)));
        let y = originY;
        for (const node of cols[c]) {
            if (keys.get(node) === Infinity) desired.set(node, y);
            y += geometry.get(node).height + rowGap;
        }
        placeColumn(cols[c], desired, geometry, originY, heightLimit, rowGap);
    }
    compactColumns(cols, edges, geometry, originX, colGap, rowGap);
    consider();
    // Receivers also follow their providers; keep the chosen row order fixed.
    for (let c = 1; c < numCols; c++) {
        const desired = new Map(cols[c].map((node) => {
            const links = edges.filter((e) => e.target === node);
            return [node, links.length
                ? median(links.map((e) => e.source.pos[1] + e.outputY - e.inputY)) + geometry.get(node).dy
                : node.pos[1] + geometry.get(node).dy];
        }));
        placeColumn(cols[c], desired, geometry, originY, heightLimit, rowGap);
    }
    compactColumns(cols, edges, geometry, originX, colGap, rowGap);
    consider();
    for (const node of nodes) node.pos = best.get(node);

    // A node's boundingRect is a cached, measured rect that only refreshes on the
    // render pass — assigning node.pos above does NOT update it. resizeTo() reads
    // those cached rects, so without this it would wrap the nodes' pre-move
    // positions, leaving the group too small/offset. Force each moved node to
    // re-measure first, reusing LiteGraph's own title/collapse-aware logic.
    for (const n of nodes) {
        if (typeof n.updateArea === "function") n.updateArea(ctx);
        else if (typeof n.measure === "function") n.measure(n.boundingRect, ctx);
    }

    // Wrap the group around the laid-out nodes with uniform PADDING. Prefer
    // LiteGraph's native resizeTo(): it fits to each node's full bounding box
    // (including the title that renders above node.pos[1]) and accounts for the
    // group's own title height — identical to ComfyUI's "Fit Group To Contents".
    if (typeof group.resizeTo === "function") {
        group.resizeTo(nodes, PADDING);
    } else {
        // Fallback for older litegraph (approximate; ignores node titles).
        let maxBottom = originY;
        for (const n of nodes) {
            const box = geometry.get(n);
            const b = n.pos[1] + box.dy + box.height;
            if (b > maxBottom) maxBottom = b;
        }
        const totalW = Math.max(...nodes.map((n) => n.pos[0] + geometry.get(n).dx + geometry.get(n).width)) - originX;
        group.size = [
            totalW + 2 * PADDING,
            (maxBottom - originY) + 2 * PADDING + titleH
        ];
    }
}

function groupNodes(group) {
    group.recomputeInsideNodes();
    return Array.from(group._children || []).filter((n) =>
        n instanceof window.LiteGraph.LGraphNode);
}

function selectedNodes(canvas) {
    return Array.from(canvas.selectedItems ?? Object.values(canvas.selected_nodes ?? {}))
        .filter((n) => n instanceof window.LiteGraph.LGraphNode);
}

function groupedBundles(nodes, graph) {
    const owners = new Map();
    const scopes = new Map();
    const groups = [...graph._groups].sort((a, b) =>
        a.size[0] * a.size[1] - b.size[0] * b.size[1]);
    for (const group of groups) {
        const members = groupNodes(group);
        const movable = !group.pinned && !members.some((n) => n.pinned);
        scopes.set(group, []);
        for (const n of members) {
            if (owners.has(n)) continue;
            owners.set(n, group);
            if (movable) scopes.get(group).push(n);
        }
    }
    return [...scopes].flatMap(([group, scope]) =>
        findSourceBundles(nodes.filter((n) => owners.get(n) === group), scope, graph));
}

function organizeSelectedGroups() {
    const canvas = app.canvas;
    if (!canvas) return;

    const graph = canvas.graph || app.graph;
    const groups = getSelectedGroups(canvas).filter((g) =>
        graph._groups.includes(g) && !g.pinned && !groupNodes(g).some((n) => n.pinned));
    const nodes = groups.length
        ? [...new Set(groups.flatMap(groupNodes))].filter((n) => !n.pinned)
        : selectedNodes(canvas).filter((n) => !n.pinned);
    if (!nodes.length) {
        app.extensionManager.toast.add({ severity: "info", summary: "Select nodes or a group first", life: 3000 });
        return;
    }

    const autoPack = app.extensionManager.setting.get(AUTO_SUBGRAPHS) ?? true;
    const bundles = autoPack
        ? groupedBundles(nodes, graph)
        : [];
    if (bundles.length && typeof graph.convertToSubgraph !== "function") {
        app.extensionManager.toast.add({ severity: "error", summary: "This frontend cannot create subgraphs", life: 4000 });
        return;
    }

    canvas.emitBeforeChange();
    try {
        const replacements = new Map();
        for (const bundle of bundles) {
            const packed = packSourceBundle(graph, bundle, ROW_GAP, getTitleHeight());
            packed.updateArea(canvas.ctx);
            for (const n of bundle.consumers) replacements.set(n, packed);
            for (const s of bundle.sources) replacements.set(s.node, packed);
        }
        if (groups.length) {
            for (const g of groups) arrangeGroup(g, graph);
        } else {
            const laidOut = [...new Set(nodes.map((n) => replacements.get(n) || n))];
            for (const node of laidOut) node.updateArea(canvas.ctx);
            const origin = [
                Math.min(...laidOut.map((n) => n.boundingRect[0])) - PADDING,
                Math.min(...laidOut.map((n) => n.boundingRect[1])) - PADDING - getTitleHeight(),
            ];
            arrangeGroup({
                pos: origin, _children: laidOut,
                recomputeInsideNodes() {},
            }, graph);
            canvas.selectedItems = new Set(laidOut);
        }
    } catch (error) {
        console.error("[OrganizeGroupCompact]", error);
        app.extensionManager.toast.add({ severity: "error", summary: "Could not organize selection", detail: error.message, life: 5000 });
    } finally {
        canvas.emitAfterChange();
    }

    graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "TrentNodes.OrganizeGroupCompact",

    settings: [
        {
            id: SPACING,
            category: ["TrentNodes", "Group Organizer", "Spacing"],
            name: "Layout spacing",
            type: "combo",
            options: ["Compact", "Relaxed"],
            defaultValue: "Compact",
            tooltip: "Compact shortens gaps around measured node bounds. Relaxed leaves more room between nodes and wires.",
        },
        {
            id: AUTO_SUBGRAPHS,
            category: ["TrentNodes", "Group Organizer", "Auto-create subgraphs"],
            name: "Auto-create subgraphs from direct inputs in the same group",
            type: "boolean",
            defaultValue: true,
            tooltip: "Shift+Alt+O packs consumers with their direct input nodes in the same group and exposes unconnected inputs and controls. Disable for layout only.",
        },
    ],

    commands: [
        {
            id: "TrentNodes.OrganizeGroupCompact",
            label: "Organize Group Compact",
            icon: "pi pi-th-large",
            function: organizeSelectedGroups
        }
    ],

    // Surface the command in the canvas selection toolbox (the popup toolbar)
    // whenever nodes or a group are selected. ComfyUI collects the command id(s)
    // across selected items and renders them as buttons.
    getSelectionToolboxCommands(selectedItem) {
        if (window.LiteGraph && (selectedItem instanceof window.LiteGraph.LGraphGroup
            || selectedItem instanceof window.LiteGraph.LGraphNode)) {
            return ["TrentNodes.OrganizeGroupCompact"];
        }
        return [];
    },

    keybindings: [
        {
            commandId: "TrentNodes.OrganizeGroupCompact",
            combo: { key: "o", shift: true, alt: true },
            targetElementId: "graph-canvas"
        }
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.OrganizeGroupCompact"]
        }
    ]
});
