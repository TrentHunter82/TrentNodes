import { app } from "../../scripts/app.js";

/**
 * Organize Group as Grid
 *
 * Shift+Alt+A — for each selected group, lays out its child
 * nodes left-to-right by connection depth (topological rank),
 * then resizes the group to wrap the result.
 *
 * Within a column, nodes are stacked top-to-bottom by the mean
 * y-position of their downstream targets (barycenter heuristic),
 * which keeps wires roughly parallel and minimizes crossings.
 *
 * Falls back to current y-position when a node has no downstream
 * targets inside the group.
 */

const PADDING = 20;
const COL_GAP = 40;
const ROW_GAP = 20;

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

    const numCols = Math.max(...nodes.map((n) => rank.get(n.id))) + 1;
    const cols = Array.from({ length: numCols }, () => []);
    for (const n of nodes) cols[rank.get(n.id)].push(n);

    // Sort each column by mean current y of downstream targets
    // inside the group; fall back to own current y when a node
    // has no in-group successors (sinks / disconnected nodes).
    const meanSuccY = (node) => {
        const s = succ.get(node.id);
        if (!s || s.size === 0) return node.pos[1];
        let total = 0, count = 0;
        for (const sid of s) {
            const sn = graph.getNodeById(sid);
            if (sn) { total += sn.pos[1]; count++; }
        }
        return count > 0 ? total / count : node.pos[1];
    };
    for (let c = 0; c < numCols; c++) {
        cols[c].sort((a, b) => {
            const ay = meanSuccY(a);
            const by = meanSuccY(b);
            if (ay !== by) return ay - by;
            return a.pos[1] - b.pos[1];
        });
    }

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

    let maxColHeight = 0;
    for (let c = 0; c < numCols; c++) {
        let y = 0;
        const colRight = colXOffsets[c] + colWidths[c];
        for (const node of cols[c]) {
            const [w, h] = effectiveSize(node);
            const x = isCollapsed(node)
                ? originX + colRight - w
                : originX + colXOffsets[c];
            node.pos = [x, originY + y];
            y += h + ROW_GAP;
        }
        if (cols[c].length > 0) y -= ROW_GAP;
        if (y > maxColHeight) maxColHeight = y;
    }

    const totalW = numCols > 0
        ? colXOffsets[numCols - 1] + colWidths[numCols - 1]
        : 0;
    group.size = [
        totalW + 2 * PADDING,
        maxColHeight + 2 * PADDING + titleH
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
