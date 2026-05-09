import { app } from "../../scripts/app.js";

/**
 * Organize Group as Grid
 *
 * Shift+Alt+A — for each selected group, lays out its child
 * nodes left-to-right by connection depth (topological rank),
 * then resizes the group to wrap the result.
 *
 * Within a column, nodes are ordered to match the input-slot
 * order of their downstream targets — the node feeding the
 * topmost input slot comes first, the one feeding the next slot
 * second, and so on. Spacing is uniform; we don't try to align
 * exact pixel heights. Sinks and disconnected nodes fall back
 * to their current y.
 *
 * Leaf nodes (no in-group predecessors) are pulled rightward to
 * sit one column to the left of their earliest target, so loose
 * inputs stay close to where they plug in instead of piling up
 * in column 0.
 */

const PADDING = 20;
const COL_GAP = 40;
const ROW_GAP = 40;

function getTitleHeight() {
    return (window.LiteGraph && window.LiteGraph.NODE_TITLE_HEIGHT) || 30;
}

function isCollapsed(node) {
    return !!(node.flags && node.flags.collapsed);
}

/**
 * Visual size of a node accounting for collapsed state.
 * Collapsed nodes render as a short pill (NODE_COLLAPSED_WIDTH x
 * title height), but `node.size` keeps the uncollapsed dimensions.
 */
function effectiveSize(node) {
    if (isCollapsed(node)) {
        const LG = window.LiteGraph || {};
        const w = node._collapsed_width || LG.NODE_COLLAPSED_WIDTH || 80;
        const h = LG.NODE_TITLE_HEIGHT || 30;
        return [w, h];
    }
    return [node.size[0], node.size[1]];
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

function arrangeGroup(group, graph) {
    if (!group || typeof group.recomputeInsideNodes !== "function") return;
    group.recomputeInsideNodes();

    const LGNode = window.LiteGraph && window.LiteGraph.LGraphNode;
    const children = Array.from(group._children || []);
    const nodes = LGNode
        ? children.filter((c) => c instanceof LGNode)
        : children;
    if (nodes.length === 0) return;

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

    const numCols = Math.max(...nodes.map((n) => rank.get(n.id))) + 1;
    const cols = Array.from({ length: numCols }, () => []);
    for (const n of nodes) cols[rank.get(n.id)].push(n);

    const titleH = getTitleHeight();
    const colWidths = cols.map((col) =>
        col.length > 0
            ? Math.max(...col.map((n) => effectiveSize(n)[0]))
            : 0
    );

    const colXOffsets = new Array(numCols).fill(0);
    for (let c = 1; c < numCols; c++) {
        colXOffsets[c] = colXOffsets[c - 1] + colWidths[c - 1] + COL_GAP;
    }

    const originX = group.pos[0] + PADDING;
    const originY = group.pos[1] + titleH + PADDING;

    // Collapsed nodes right-align inside their column; full-size
    // nodes left-align.
    const xForNode = (node, c) => {
        if (isCollapsed(node)) {
            const w = effectiveSize(node)[0];
            return originX + colXOffsets[c] + colWidths[c] - w;
        }
        return originX + colXOffsets[c];
    };

    // Place the rightmost column first, top-to-bottom by current y.
    // It acts as the anchor that everything to the left orders against.
    if (numCols > 0) {
        const rc = numCols - 1;
        cols[rc].sort((a, b) => a.pos[1] - b.pos[1]);
        let y = originY;
        for (const node of cols[rc]) {
            node.pos = [xForNode(node, rc), y];
            y += effectiveSize(node)[1] + ROW_GAP;
        }
    }

    // Sort key for `node`: the (target_y, target_slot) of its
    // primary in-group downstream link. Primary = the topmost
    // already-placed target; ties broken by lowest input slot.
    // Falls back to current y for sinks / disconnected nodes.
    const sortKey = (node) => {
        const targets = succ.get(node.id);
        if (!targets || targets.size === 0 || !node.outputs) {
            return [node.pos[1], 0];
        }
        let bestY = Infinity;
        let bestSlot = 0;
        let found = false;
        for (let oi = 0; oi < node.outputs.length; oi++) {
            const out = node.outputs[oi];
            if (!out || !out.links) continue;
            for (const linkId of out.links) {
                const link = getLink(graph, linkId);
                if (!link) continue;
                if (!targets.has(link.target_id)) continue;
                const tgt = graph.getNodeById(link.target_id);
                if (!tgt) continue;
                const ty = tgt.pos[1];
                const ts = link.target_slot || 0;
                if (!found || ty < bestY
                    || (ty === bestY && ts < bestSlot)) {
                    bestY = ty;
                    bestSlot = ts;
                    found = true;
                }
            }
        }
        return found ? [bestY, bestSlot] : [node.pos[1], 0];
    };

    // Place remaining columns right-to-left, ordered by
    // (target_y, target_slot) so the node feeding the topmost input
    // comes first, the next-slot feeder second, and so on.
    for (let c = numCols - 2; c >= 0; c--) {
        cols[c].sort((a, b) => {
            const ka = sortKey(a);
            const kb = sortKey(b);
            if (ka[0] !== kb[0]) return ka[0] - kb[0];
            return ka[1] - kb[1];
        });
        let y = originY;
        for (const node of cols[c]) {
            node.pos = [xForNode(node, c), y];
            y += effectiveSize(node)[1] + ROW_GAP;
        }
    }

    // Group height = wrap the lowest bottom edge across all columns.
    let maxBottom = originY;
    for (const n of nodes) {
        const b = n.pos[1] + effectiveSize(n)[1];
        if (b > maxBottom) maxBottom = b;
    }

    const totalW = numCols > 0
        ? colXOffsets[numCols - 1] + colWidths[numCols - 1]
        : 0;
    group.size = [
        totalW + 2 * PADDING,
        (maxBottom - originY) + 2 * PADDING + titleH
    ];
}

function organizeSelectedGroups() {
    const canvas = app.canvas;
    if (!canvas) return;

    const groups = getSelectedGroups(canvas);
    if (groups.length === 0) {
        console.warn(
            "[OrganizeGroupGrid] No group selected. "
            + "Click a group's title bar first."
        );
        return;
    }

    canvas.emitBeforeChange();
    try {
        for (const g of groups) arrangeGroup(g, app.graph);
    } finally {
        canvas.emitAfterChange();
    }

    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "TrentNodes.OrganizeGroupGrid",

    commands: [
        {
            id: "TrentNodes.OrganizeGroupGrid",
            label: "Organize Group as Grid",
            icon: "pi pi-th-large",
            function: organizeSelectedGroups
        }
    ],

    keybindings: [
        {
            commandId: "TrentNodes.OrganizeGroupGrid",
            combo: { key: "a", shift: true, alt: true }
        }
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.OrganizeGroupGrid"]
        }
    ]
});
