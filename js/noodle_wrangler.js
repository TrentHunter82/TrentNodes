import { app } from "../../scripts/app.js";

/*
 * Trent Noodle Wrangler
 * ---------------------
 * Two canvas behaviors, toggled from the Colors sidebar (see color_palette.js)
 * or via the TrentNodes menu commands:
 *
 *   1. Circuit noodles — draw every link as a rounded-corner circuit-board
 *      trace (90/45-degree runs routed around nodes) instead of a spline.
 *      Routing core adapted from niknah/quick-connections (MIT license,
 *      https://github.com/niknah/quick-connections). Only the circuit-board
 *      renderer is ported — the "quick connections" drag helper is not.
 *
 *   2. Magnet — while dragging nodes, nearby nodes attract: edges align,
 *      nodes dock side-by-side/stacked with a uniform gap, and connected
 *      slots pull level so their noodle runs dead straight.
 *      Hold Alt while dragging to suppress the magnet.
 *
 * State persists in localStorage. Other TrentNodes scripts can subscribe to
 * toggle changes via onNoodleChange(fn) to keep their UI in sync.
 */

const LS_CIRCUIT = "TrentNoodles.circuit";
const LS_MAGNET = "TrentNoodles.magnet";

const state = {
    circuit: localStorage.getItem(LS_CIRCUIT) !== "0", // default ON
    magnet: localStorage.getItem(LS_MAGNET) === "1",   // default OFF
};

const listeners = new Set();
function notify() {
    for (const fn of listeners) {
        try { fn({ ...state }); } catch (e) { console.warn("[TrentNoodles]", e); }
    }
}

export function onNoodleChange(fn) { listeners.add(fn); }
export function getCircuit() { return state.circuit; }
export function getMagnet() { return state.magnet; }

function redrawCanvas() {
    const c = app.canvas;
    if (c) c.setDirty(true, true);
}

export function setCircuit(on) {
    state.circuit = !!on;
    localStorage.setItem(LS_CIRCUIT, on ? "1" : "0");
    notify();
    redrawCanvas();
}

export function setMagnet(on) {
    state.magnet = !!on;
    localStorage.setItem(LS_MAGNET, on ? "1" : "0");
    notify();
}

// links_render_mode value that means "user hid all links" — respect it.
function hiddenLinkMode() {
    const LG = window.LiteGraph || {};
    const LGC = window.LGraphCanvas || {};
    return LG.HIDDEN_LINK ?? LGC.HIDDEN_LINK ?? -999;
}

/* ==========================================================================
 * Part 1 — circuit-board link routing
 * Adapted from niknah/quick-connections (MIT).
 * ========================================================================== */

/**
 * @preserve
 * Fast, destructive implementation of the Liang-Barsky line clipping
 * algorithm. Clips a 2D segment by a rectangle.
 * @author Alexander Milevski <info@w8r.name>
 * @license MIT
 */
const EPSILON = 1e-6;
const INSIDE = 1;
const OUTSIDE = 0;

function clipT(num, denom, c) {
    const tE = c[0], tL = c[1];
    if (Math.abs(denom) < EPSILON) return num < 0;
    const t = num / denom;
    if (denom > 0) {
        if (t > tL) return 0;
        if (t > tE) c[0] = t;
    } else {
        if (t < tE) return 0;
        if (t < tL) c[1] = t;
    }
    return 1;
}

// Does segment a->b cross `box` [xmin, ymin, xmax, ymax]? Writes the clipped
// segment into da/db when provided.
function liangBarsky(a, b, box, da, db) {
    const x1 = a[0], y1 = a[1];
    const x2 = b[0], y2 = b[1];
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (da === undefined || db === undefined) {
        da = a; db = b;
    } else {
        da[0] = a[0]; da[1] = a[1];
        db[0] = b[0]; db[1] = b[1];
    }
    if (Math.abs(dx) < EPSILON && Math.abs(dy) < EPSILON &&
        x1 >= box[0] && x1 <= box[2] && y1 >= box[1] && y1 <= box[3]) {
        return INSIDE;
    }
    const c = [0, 1];
    if (clipT(box[0] - x1, dx, c) &&
        clipT(x1 - box[2], -dx, c) &&
        clipT(box[1] - y1, dy, c) &&
        clipT(y1 - box[3], -dy, c)) {
        const tE = c[0], tL = c[1];
        if (tL < 1) {
            db[0] = x1 + tL * dx;
            db[1] = y1 + tL * dy;
        }
        if (tE > 0) {
            da[0] += tE * dx;
            da[1] += tE * dy;
        }
        return INSIDE;
    }
    return OUTSIDE;
}

// Slot-position helpers that work in both canvas and Vue nodes modes.
function slotOutPos(node, slot) {
    if (window.LiteGraph?.vueNodesMode && node.getSlotPosition) {
        return node.getSlotPosition(slot, false);
    }
    if (node.getOutputPos) return node.getOutputPos(slot);
    return node.getConnectionPos(false, slot);
}
function slotInPos(node, slot) {
    if (window.LiteGraph?.vueNodesMode && node.getSlotPosition) {
        return node.getSlotPosition(slot, true);
    }
    if (node.getInputPos) return node.getInputPos(slot);
    return node.getConnectionPos(true, slot);
}

// Plans orthogonal (90/45 degree) paths for every link, routing around nodes.
class MapLinks {
    constructor(canvas, config) {
        this.canvas = canvas;
        this.nodesByRight = [];
        this.nodesById = {};
        this.paths = [];
        this.config = config;
        this.lastCalcTime = 0;
    }

    // find which node is in the way of the output-to-input segment
    findClippedNode(outputXY, inputXY) {
        let closestDistance = Number.MAX_SAFE_INTEGER;
        let closest = null;
        for (let i = 0; i < this.nodesByRight.length; ++i) {
            const node = this.nodesByRight[i];
            const clipA = [-1, -1];
            const clipB = [-1, -1];
            if (liangBarsky(outputXY, inputXY, node.area, clipA, clipB) === INSIDE) {
                const centerX = (node.area[0] + ((node.area[2] - node.area[0]) / 2));
                const centerY = (node.area[1] + ((node.area[3] - node.area[1]) / 2));
                const dist = Math.sqrt(((centerX - outputXY[0]) ** 2) + ((centerY - outputXY[1]) ** 2));
                if (dist < closestDistance) {
                    closest = { start: clipA, end: clipB, node };
                    closestDistance = dist;
                }
            }
        }
        return { clipped: closest, closestDistance };
    }

    testPath(path) {
        const len1 = (path.length - 1);
        for (let p = 0; p < len1; ++p) {
            const { clipped } = this.findClippedNode(path[p], path[p + 1]);
            if (clipped) return clipped;
        }
        return null;
    }

    mapFinalLink(outputXY, inputXY) {
        const { clipped } = this.findClippedNode(outputXY, inputXY);
        if (!clipped) {
            // direct, nothing blocking us
            return { path: [outputXY, inputXY] };
        }

        const horzDistance = inputXY[0] - outputXY[0];
        const vertDistance = inputXY[1] - outputXY[1];
        const horzDistanceAbs = Math.abs(horzDistance);
        const vertDistanceAbs = Math.abs(vertDistance);

        if (horzDistanceAbs > vertDistanceAbs) {
            const goingLeft = inputXY[0] < outputXY[0];
            const pathStraight45 = [
                [outputXY[0], outputXY[1]],
                [inputXY[0] - (goingLeft ? -vertDistanceAbs : vertDistanceAbs), outputXY[1]],
                [inputXY[0], inputXY[1]],
            ];
            if (!this.testPath(pathStraight45)) return { path: pathStraight45 };

            const path45Straight = [
                [outputXY[0], outputXY[1]],
                [outputXY[0] + (goingLeft ? -vertDistanceAbs : vertDistanceAbs), inputXY[1]],
                [inputXY[0], inputXY[1]],
            ];
            if (!this.testPath(path45Straight)) return { path: path45Straight };
        } else {
            const goingUp = inputXY[1] < outputXY[1];
            const pathStraight45 = [
                [outputXY[0], outputXY[1]],
                [outputXY[0], inputXY[1] + (goingUp ? horzDistanceAbs : -horzDistanceAbs)],
                [inputXY[0], inputXY[1]],
            ];
            if (!this.testPath(pathStraight45)) return { path: pathStraight45 };

            const path45Straight = [
                [outputXY[0], outputXY[1]],
                [inputXY[0], outputXY[1] - (goingUp ? horzDistanceAbs : -horzDistanceAbs)],
                [inputXY[0], inputXY[1]],
            ];
            if (!this.testPath(path45Straight)) return { path: path45Straight };
        }

        const path90Straight = [
            [outputXY[0], outputXY[1]],
            [outputXY[0], inputXY[1]],
            [inputXY[0], inputXY[1]],
        ];
        const clippedVert = this.testPath(path90Straight);
        if (!clippedVert) return { path: path90Straight };

        const pathStraight90 = [
            [outputXY[0], outputXY[1]],
            [inputXY[0], outputXY[1]],
            [inputXY[0], inputXY[1]],
        ];
        const clippedHorz = this.testPath(pathStraight90);
        if (!clippedHorz) return { path: pathStraight90 };

        return { clippedHorz, clippedVert };
    }

    mapLink(outputXY, inputXY, targetNodeInfo, isBlocked) {
        const { clippedHorz, clippedVert, path } = this.mapFinalLink(outputXY, inputXY);
        if (path) return path;

        const horzDistance = inputXY[0] - outputXY[0];
        const vertDistance = inputXY[1] - outputXY[1];
        const horzDistanceAbs = Math.abs(horzDistance);
        const vertDistanceAbs = Math.abs(vertDistance);

        let blockedNodeId;
        let pathAvoidNode;
        let lastPathLocation;
        let linesArea;

        if (horzDistanceAbs > vertDistanceAbs) {
            // horizontal first, then vertical around the blocking node
            blockedNodeId = clippedHorz.node.node.id;
            linesArea = clippedHorz.node.linesArea;
            const horzEdge = horzDistance <= 0 ? (linesArea[2]) : (linesArea[0] - 1);
            pathAvoidNode = [
                [outputXY[0], outputXY[1]],
                [horzEdge, outputXY[1]],
            ];
            if (horzDistance <= 0) linesArea[2] += this.config.lineSpace;
            else linesArea[0] -= this.config.lineSpace;

            const vertDistanceViaBlockTop =
                Math.abs(inputXY[1] - linesArea[1]) + Math.abs(linesArea[1] - outputXY[1]);
            const vertDistanceViaBlockBottom =
                Math.abs(inputXY[1] - linesArea[3]) + Math.abs(linesArea[3] - outputXY[1]);

            lastPathLocation = [
                horzEdge,
                vertDistanceViaBlockTop <= vertDistanceViaBlockBottom ? (linesArea[1]) : (linesArea[3]),
            ];
            if (this.testPath([...pathAvoidNode, lastPathLocation])) {
                lastPathLocation = [
                    horzEdge,
                    vertDistanceViaBlockTop > vertDistanceViaBlockBottom ? (linesArea[1]) : (linesArea[3]),
                ];
            }
            if (lastPathLocation[1] < outputXY[1]) {
                linesArea[1] -= this.config.lineSpace;
                lastPathLocation[1] -= 1;
            } else {
                linesArea[3] += this.config.lineSpace;
                lastPathLocation[1] += 1;
            }
        } else {
            // vertical first, then horizontal around the blocking node
            blockedNodeId = clippedVert.node.node.id;
            linesArea = clippedVert.node.linesArea;
            const vertEdge = vertDistance <= 0 ? (linesArea[3] + 1) : (linesArea[1] - 1);
            pathAvoidNode = [
                [outputXY[0], outputXY[1]],
                [outputXY[0], vertEdge],
            ];
            if (vertDistance <= 0) linesArea[3] += this.config.lineSpace;
            else linesArea[1] -= this.config.lineSpace;

            const horzDistanceViaBlockLeft =
                Math.abs(inputXY[0] - linesArea[0]) + Math.abs(linesArea[0] - outputXY[0]);
            const horzDistanceViaBlockRight =
                Math.abs(inputXY[0] - linesArea[2]) + Math.abs(linesArea[2] - outputXY[0]);

            lastPathLocation = [
                horzDistanceViaBlockLeft <= horzDistanceViaBlockRight ? (linesArea[0] - 1) : (linesArea[2]),
                vertEdge,
            ];
            if (this.testPath([...pathAvoidNode, lastPathLocation])) {
                lastPathLocation = [
                    horzDistanceViaBlockLeft > horzDistanceViaBlockRight ? (linesArea[0]) : (linesArea[2]),
                    vertEdge,
                ];
            }
            if (lastPathLocation[0] < outputXY[0]) linesArea[0] -= this.config.lineSpace;
            else linesArea[2] += this.config.lineSpace;
        }

        if (isBlocked[blockedNodeId] > 3) {
            // blocked too many times — give up and draw a direct line
            isBlocked.blocked = true;
            return [outputXY, inputXY];
        }
        isBlocked[blockedNodeId] = (isBlocked[blockedNodeId] || 0) + 1;

        const nextPath = this.mapLink(lastPathLocation, inputXY, targetNodeInfo, isBlocked);
        return [...pathAvoidNode, lastPathLocation, ...nextPath.slice(1)];
    }

    // widen the do-not-draw margin next to the source node when a path leaves vertically
    expandSourceNodeLinesArea(sourceNodeInfo, path) {
        if (path.length < 3) return false;
        if (path[1][0] === path[2][0]) {
            sourceNodeInfo.linesArea[2] += this.config.lineSpace;
        }
        return true;
    }

    // widen the left margin of the target node when a path arrives vertically
    expandTargetNodeLinesArea(targetNodeInfo, path) {
        if (path.length < 2) return false;
        const p = path.length - 2;
        if (path[p - 1][0] === path[p][0]) {
            targetNodeInfo.linesArea[0] -= this.config.lineSpace;
        }
        return true;
    }

    getNodeOnPos(xy) {
        for (let i = 0; i < this.nodesByRight.length; ++i) {
            const nodeI = this.nodesByRight[i];
            const { linesArea } = nodeI;
            if (xy[0] >= linesArea[0] && xy[1] >= linesArea[1] &&
                xy[0] < linesArea[2] && xy[1] < linesArea[3]) {
                return nodeI;
            }
        }
        return null;
    }

    mapLinks(nodesByExecution) {
        const graphLinks = this.canvas.graph.links;
        if (!graphLinks) return;

        const startCalcTime = performance.now();
        this.paths = [];
        this.nodesById = {};
        this.nodesByRight = nodesByExecution.map((node) => {
            const barea = new Float32Array(4);
            node.getBounding(barea);
            const area = [barea[0], barea[1], barea[0] + barea[2], barea[1] + barea[3]];
            const linesArea = Array.from(area);
            linesArea[0] += this.config.nodeSpace[0];
            linesArea[1] += this.config.nodeSpace[1];
            linesArea[2] += this.config.nodeSpace[2];
            linesArea[3] += this.config.nodeSpace[3];
            const obj = { node, area, linesArea };
            this.nodesById[node.id] = obj;
            return obj;
        });

        const nodesByRightId = {};
        for (const x of this.nodesByRight) nodesByRightId[x.node.id] = x.node;

        for (const nodeI of this.nodesByRight) {
            const { node } = nodeI;
            const outputs = node.outputs;
            if (!outputs) continue;
            outputs.forEach((output, slot) => {
                const links = output.links;
                if (!links) return;

                const outputXYConnection = slotOutPos(node, slot);
                const outputNodeInfo = this.nodesById[node.id];
                let outputXY = Array.from(outputXYConnection);
                for (const linkId of links) {
                    outputXY[0] = outputNodeInfo.linesArea[2];
                    const link = graphLinks[linkId];
                    if (!link) continue;
                    let targetNode = this.canvas.graph.getNodeById(link.target_id);
                    if (!targetNode) {
                        // maybe this is the in/out node of a subgraph
                        targetNode = nodesByRightId[link.target_id];
                    }
                    if (!targetNode) continue;

                    const inputXYConnection = slotInPos(targetNode, link.target_slot);
                    const inputXY = Array.from(inputXYConnection);
                    const nodeInfo = this.nodesById[targetNode.id];
                    inputXY[0] = nodeInfo.linesArea[0] - 1;

                    let path = null;
                    if (!this.getNodeOnPos(inputXY) && !this.getNodeOnPos(outputXY)) {
                        const isBlocked = {};
                        const pathFound = this.mapLink(outputXY, inputXY, nodeInfo, isBlocked);
                        if (pathFound) {
                            // drop duplicated trailing points
                            while (pathFound.length >= 2) {
                                const a = pathFound[pathFound.length - 1];
                                const b = pathFound[pathFound.length - 2];
                                if (a[0] === b[0] && a[1] === b[1]) pathFound.pop();
                                else break;
                            }
                        }
                        if (!isBlocked.blocked && pathFound && pathFound.length > 2) {
                            path = [outputXYConnection, ...pathFound, inputXYConnection];
                            this.expandTargetNodeLinesArea(nodeInfo, path);
                        }
                    }
                    if (!path) {
                        path = [outputXYConnection, outputXY, inputXY, inputXYConnection];
                    }
                    this.expandSourceNodeLinesArea(nodeI, path);
                    this.paths.push({ path, node, targetNode, slot });
                    outputXY = [outputXY[0] + this.config.lineSpace, outputXY[1]];
                }
            });
        }
        this.lastCalcTime = Math.min(performance.now() - startCalcTime, 30000);
    }

    drawLinks(ctx) {
        const byType = this.canvas.default_connection_color_byType;
        const defaults = this.canvas.default_connection_color;
        if (!byType || !defaults) return;

        ctx.save();
        const currentNodeIds = this.canvas.selected_nodes || {};
        const lineWidth = this.canvas.connections_width || 3;
        const cornerRadius = this.config.lineSpace;

        for (const pathI of this.paths) {
            const path = pathI.path;
            const connection = pathI.node.outputs[pathI.slot];
            if (path.length <= 1) continue;
            ctx.beginPath();
            const slotColor = byType[connection.type] || defaults.input_on;
            ctx.strokeStyle =
                (currentNodeIds[pathI.node.id] || currentNodeIds[pathI.targetNode.id])
                    ? "white" : slotColor;
            ctx.lineWidth = lineWidth;

            let isPrevDotRound = false;
            for (let p = 0; p < path.length; ++p) {
                const pos = path[p];
                if (p === 0) ctx.moveTo(pos[0], pos[1]);

                const prevPos = pos;
                const cornerPos = path[p + 1];
                const nextPos = path[p + 2];
                let drawn = false;
                if (nextPos) {
                    const xDiffBefore = cornerPos[0] - prevPos[0];
                    const yDiffBefore = cornerPos[1] - prevPos[1];
                    const xDiffAfter = nextPos[0] - cornerPos[0];
                    const yDiffAfter = nextPos[1] - cornerPos[1];
                    const isBeforeStraight = xDiffBefore === 0 || yDiffBefore === 0;
                    const isAfterStraight = xDiffAfter === 0 || yDiffAfter === 0;
                    if (isBeforeStraight || isAfterStraight) {
                        const beforePos = [cornerPos[0], cornerPos[1]];
                        const afterPos = [cornerPos[0], cornerPos[1]];
                        if (isBeforeStraight) {
                            beforePos[0] = cornerPos[0] - cornerRadius * Math.sign(xDiffBefore);
                            beforePos[1] = cornerPos[1] - cornerRadius * Math.sign(yDiffBefore);
                        }
                        if (isAfterStraight) {
                            afterPos[0] = cornerPos[0] + cornerRadius * Math.sign(xDiffAfter);
                            afterPos[1] = cornerPos[1] + cornerRadius * Math.sign(yDiffAfter);
                        }
                        if (isPrevDotRound
                            && Math.abs(isPrevDotRound[0] - beforePos[0]) <= cornerRadius
                            && Math.abs(isPrevDotRound[1] - beforePos[1]) <= cornerRadius) {
                            // two rounded corners too close together — skip
                        } else {
                            ctx.lineTo(beforePos[0], beforePos[1]);
                            ctx.quadraticCurveTo(cornerPos[0], cornerPos[1], afterPos[0], afterPos[1]);
                        }
                        isPrevDotRound = beforePos;
                        drawn = true;
                    }
                }
                if (p > 0 && !drawn) {
                    if (!isPrevDotRound) ctx.lineTo(pos[0], pos[1]);
                    isPrevDotRound = false;
                }
            }
            ctx.stroke();
            ctx.closePath();
        }
        ctx.restore();
    }
}

// Pretend the subgraph in/out slot panels are normal nodes so links to them route.
class SubgraphSlotProxy {
    constructor(slot) { this.slot = slot; }
    get links() { return this.slot.linkIds; }
}

class SubgraphInOutNodeProxy {
    constructor(subgraphNode, isInput) {
        this.subgraphNode = subgraphNode;
        this.isInput = isInput;
        this.slots = [];
        for (const slot of this.subgraphNode.slots) {
            this.slots.push(new SubgraphSlotProxy(slot));
        }
    }

    get id() { return this.subgraphNode.id; }

    get outputs() {
        // the output panel inside a subgraph has no outputs, only inputs
        return this.isInput ? this.slots : [];
    }

    getSlotPosition(slot) { return this.subgraphNode.slots[slot].pos; }
    getInputPos(slot) { return this.getSlotPosition(slot); }
    getOutputPos(slot) { return this.getSlotPosition(slot); }

    getBounding(area) {
        area[0] = this.subgraphNode.boundingRect[0];
        area[1] = this.subgraphNode.boundingRect[1];
        area[2] = this.subgraphNode.boundingRect[2];
        area[3] = this.subgraphNode.boundingRect[3];
        return area;
    }
}

// Owns recalc scheduling: recalc every draw while cheap, debounce when the
// graph gets big enough that path planning is slow.
class CircuitBoard {
    constructor() {
        this.mapLinks = null;
        this.recalcTimeout = null;
        this.canvas = null;
    }

    recalc() {
        const LG = window.LiteGraph || {};
        const lineSpace = Math.max(4, Math.floor((LG.NODE_SLOT_HEIGHT || 20) / 2));
        const prev = this.mapLinks;
        this.mapLinks = new MapLinks(this.canvas, {
            lineSpace,
            nodeSpace: [-8, -4, 12, 4],
        });
        const nodesByExecution = this.canvas.graph.computeExecutionOrder() || [];
        if (this.canvas.subgraph) {
            nodesByExecution.push(new SubgraphInOutNodeProxy(this.canvas.subgraph.inputNode, true));
            nodesByExecution.push(new SubgraphInOutNodeProxy(this.canvas.subgraph.outputNode, false));
        }
        try {
            this.mapLinks.mapLinks(nodesByExecution);
        } catch (e) {
            console.error("[TrentNoodles] mapLinks error", e);
            this.mapLinks = prev;
        }
    }

    draw(canvas, ctx) {
        this.canvas = canvas;
        if (!this.mapLinks || this.mapLinks.lastCalcTime <= 100) {
            this.recalc();
        } else if (!this.recalcTimeout) {
            // big graph: recalc off the draw path, then ask for a redraw
            this.recalcTimeout = setTimeout(() => {
                this.recalcTimeout = null;
                this.recalc();
                redrawCanvas();
            }, this.mapLinks.lastCalcTime * 2);
        }
        if (this.mapLinks) this.mapLinks.drawLinks(ctx);
    }
}

/* ==========================================================================
 * Part 2 — magnetic node snapping
 * ========================================================================== */

const MAGNET = {
    screenSnap: 12,     // screen px within which an edge alignment engages
    screenSlotSnap: 18, // screen px for slot (straight-noodle) alignment
    range: 200,         // canvas px: how close a neighbour must be to pull
    gap: 32,            // canvas px: docking gap between node boxes
};

function nodeBox(node) {
    const b = new Float32Array(4);
    node.getBounding(b);
    return { x1: b[0], y1: b[1], x2: b[0] + b[2], y2: b[1] + b[3] };
}

function boxesNear(a, b, range) {
    return a.x1 - range < b.x2 && a.x2 + range > b.x1 &&
        a.y1 - range < b.y2 && a.y2 + range > b.y1;
}

// Collect the LGraphNodes currently being dragged (selectedItems may also
// contain groups/reroutes — only rigid-shift real nodes).
function draggedNodes(canvas) {
    const LG = window.LiteGraph;
    const out = [];
    if (!canvas.selectedItems) return out;
    for (const item of canvas.selectedItems) {
        if (LG && LG.LGraphNode && item instanceof LG.LGraphNode) out.push(item);
        else if (!LG?.LGraphNode && item.pos && item.size && item.getBounding) out.push(item);
    }
    return out;
}

function applyMagnet(canvas) {
    const graph = canvas.graph;
    if (!graph) return;
    const dragged = draggedNodes(canvas);
    if (!dragged.length) return;

    const draggedIds = new Set(dragged.map((n) => n.id));

    // union box of everything being dragged — the whole selection shifts rigidly
    let ub = null;
    for (const n of dragged) {
        const b = nodeBox(n);
        if (!ub) ub = { ...b };
        else {
            ub.x1 = Math.min(ub.x1, b.x1); ub.y1 = Math.min(ub.y1, b.y1);
            ub.x2 = Math.max(ub.x2, b.x2); ub.y2 = Math.max(ub.y2, b.y2);
        }
    }

    const zoom = canvas.ds?.scale || 1;
    const snap = MAGNET.screenSnap / zoom;
    const slotSnap = MAGNET.screenSlotSnap / zoom;

    let bestX = null; // { delta, score }
    let bestY = null;
    const considerX = (delta, weight, limit) => {
        const score = Math.abs(delta) * weight;
        if (Math.abs(delta) <= limit && (!bestX || score < bestX.score)) bestX = { delta, score };
    };
    const considerY = (delta, weight, limit) => {
        const score = Math.abs(delta) * weight;
        if (Math.abs(delta) <= limit && (!bestY || score < bestY.score)) bestY = { delta, score };
    };

    // 1) edge alignment + gap docking against nearby stationary nodes
    for (const other of graph._nodes || []) {
        if (draggedIds.has(other.id)) continue;
        if (other.flags?.pinned) { /* pinned nodes still attract — fine */ }
        const ob = nodeBox(other);
        if (!boxesNear(ub, ob, MAGNET.range)) continue;

        // edges (weight 1) and centers (weight 1.25, weaker pull)
        considerX(ob.x1 - ub.x1, 1, snap);
        considerX(ob.x2 - ub.x2, 1, snap);
        considerX((ob.x1 + ob.x2) / 2 - (ub.x1 + ub.x2) / 2, 1.25, snap);
        considerY(ob.y1 - ub.y1, 1, snap);
        considerY(ob.y2 - ub.y2, 1, snap);
        considerY((ob.y1 + ob.y2) / 2 - (ub.y1 + ub.y2) / 2, 1.25, snap);

        // docking: sit beside / stack under with a uniform gap (weight 0.9)
        considerX(ob.x2 + MAGNET.gap - ub.x1, 0.9, snap);
        considerX(ob.x1 - MAGNET.gap - ub.x2, 0.9, snap);
        considerY(ob.y2 + MAGNET.gap - ub.y1, 0.9, snap);
        considerY(ob.y1 - MAGNET.gap - ub.y2, 0.9, snap);
    }

    // 2) slot alignment: pull connected slots level so the noodle runs straight
    //    (weight 0.5 — the strongest magnet)
    const links = graph._links?.values ? graph._links.values() : Object.values(graph.links || {});
    for (const link of links) {
        if (!link) continue;
        const outDragged = draggedIds.has(link.origin_id);
        const inDragged = draggedIds.has(link.target_id);
        if (outDragged === inDragged) continue; // both moving or both parked
        const outNode = graph.getNodeById(link.origin_id);
        const inNode = graph.getNodeById(link.target_id);
        if (!outNode || !inNode) continue;
        if (outNode.flags?.collapsed || inNode.flags?.collapsed) continue;
        try {
            const oPos = slotOutPos(outNode, link.origin_slot);
            const iPos = slotInPos(inNode, link.target_slot);
            const delta = outDragged ? (iPos[1] - oPos[1]) : (oPos[1] - iPos[1]);
            considerY(delta, 0.5, slotSnap);
        } catch (e) { /* odd slot — skip */ }
    }

    if (!bestX && !bestY) return;
    const dx = bestX ? bestX.delta : 0;
    const dy = bestY ? bestY.delta : 0;
    for (const n of dragged) {
        n.pos[0] += dx;
        n.pos[1] += dy;
    }
    canvas.dirty_canvas = true;
    canvas.dirty_bgcanvas = true;
}

/* ==========================================================================
 * Wiring
 * ========================================================================== */

function installPatches() {
    const proto = app.canvas?.constructor?.prototype || window.LGraphCanvas?.prototype;
    if (!proto || proto.__trentNoodlesPatched) return;

    // circuit noodles: replace link drawing when enabled
    const origDrawConnections = proto.drawConnections;
    const board = new CircuitBoard();
    proto.drawConnections = function (ctx) {
        if (state.circuit && this.graph && !window.LiteGraph?.vueNodesMode) {
            try {
                // native draw fills renderedPaths for link hover/click hit-testing;
                // clear it so no stale spline hotspots linger under our traces
                this.renderedPaths?.clear?.();
                if (this.links_render_mode === hiddenLinkMode()) return;
                board.draw(this, ctx);
                return;
            } catch (e) {
                console.error("[TrentNoodles] circuit draw failed, falling back", e);
            }
        }
        return origDrawConnections.apply(this, arguments);
    };

    // magnet: nudge dragged nodes toward alignment after each mouse move
    const origProcessMouseMove = proto.processMouseMove;
    proto.processMouseMove = function (e) {
        const r = origProcessMouseMove.apply(this, arguments);
        try {
            if (state.magnet && this.isDragging && !e.altKey && !window.LiteGraph?.vueNodesMode) {
                applyMagnet(this);
            }
        } catch (err) {
            console.warn("[TrentNoodles] magnet error", err);
        }
        return r;
    };

    proto.__trentNoodlesPatched = true;
}

app.registerExtension({
    name: "TrentNodes.noodleWrangler",

    commands: [
        {
            id: "TrentNodes.ToggleCircuitNoodles",
            label: "⚡ Circuit Noodles",
            icon: "pi pi-share-alt",
            function: () => setCircuit(!state.circuit),
        },
        {
            id: "TrentNodes.ToggleMagnet",
            label: "🧲 Magnet Snapping",
            icon: "pi pi-arrows-alt",
            function: () => setMagnet(!state.magnet),
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.ToggleCircuitNoodles", "TrentNodes.ToggleMagnet"],
        },
    ],

    async setup() {
        installPatches();
    },
});
