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
 *   2. Magnet (smart align) — Figma/Excalidraw-style alignment while
 *      dragging nodes: hard snap + dashed guide lines when an edge lines
 *      up, a 32px dock gap is hit, or connected slots sit level (straight
 *      noodle). Instant in, instant out — no attraction physics.
 *      Hold Alt while dragging to suppress it.
 *
 * State persists in localStorage. Other TrentNodes scripts can subscribe to
 * toggle changes via onNoodleChange(fn) to keep their UI in sync.
 */

const LS_CIRCUIT = "TrentNoodles.circuit";
const LS_MAGNET = "TrentNoodles.magnet";

const state = {
    circuit: localStorage.getItem(LS_CIRCUIT) === "1", // default OFF
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
        this.calcTopLeft = {}; // node.id -> top-left at plan time, to spot moves
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
        this.calcTopLeft = {};
        this.nodesByRight = nodesByExecution.map((node) => {
            const barea = new Float32Array(4);
            node.getBounding(barea);
            this.calcTopLeft[node.id] = [barea[0], barea[1]];
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
                    const link = graphLinks.get ? graphLinks.get(linkId) : graphLinks[linkId];
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
                    this.paths.push({ path, node, targetNode, slot, targetSlot: link.target_slot });
                    outputXY = [outputXY[0] + this.config.lineSpace, outputXY[1]];
                }
            });
        }
        // Cap the measurement: one janky frame (GC, VRAM hitch) must not
        // inflate the debounce below into a multi-second path freeze.
        this.lastCalcTime = Math.min(performance.now() - startCalcTime, 2000);
    }

    // Did this node move since the paths were planned? Matters when the
    // plan is debounced (big graph): stale traces must not stay pinned to
    // the node's old position while it is being dragged.
    nodeMovedSincePlan(node, scratch) {
        const p0 = this.calcTopLeft[node.id];
        if (!p0) return false;
        node.getBounding(scratch);
        return Math.abs(scratch[0] - p0[0]) > 0.5 || Math.abs(scratch[1] - p0[1]) > 0.5;
    }

    drawLinks(ctx) {
        const byType = this.canvas.default_connection_color_byType || {};
        const defaults = this.canvas.default_connection_color || {};

        ctx.save();
        const currentNodeIds = this.canvas.selected_nodes || {};
        const lineWidth = this.canvas.connections_width || 3;
        const cornerRadius = this.config.lineSpace;
        const scratch = new Float32Array(4);

        for (const pathI of this.paths) {
            // dynamic nodes (e.g. the Trent Bus) can remove slots between
            // plans — skip dead entries instead of crashing the draw loop
            const connection = pathI.node.outputs?.[pathI.slot];
            const targetInput = pathI.targetNode.inputs?.[pathI.targetSlot];
            if (!connection || !targetInput) continue;

            let path = pathI.path;
            // an endpoint node moved since the (debounced) plan — re-anchor
            // the trace live so it never detaches from the node mid-drag
            if (this.nodeMovedSincePlan(pathI.node, scratch)
                || this.nodeMovedSincePlan(pathI.targetNode, scratch)) {
                try {
                    const o = slotOutPos(pathI.node, pathI.slot);
                    const t = slotInPos(pathI.targetNode, pathI.targetSlot);
                    const midX = (o[0] + t[0]) / 2;
                    path = [
                        [o[0], o[1]],
                        [midX, o[1]],
                        [midX, t[1]],
                        [t[0], t[1]],
                    ];
                } catch (e) {
                    continue; // odd slot state — the pending replan fixes it
                }
            }

            if (path.length <= 1) continue;
            ctx.beginPath();
            const slotColor = byType[connection.type] || defaults.input_on || "#AFA";
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

    // Returns true when circuit traces were drawn; false tells the caller
    // to fall back to native spline rendering for this frame.
    draw(canvas, ctx) {
        this.canvas = canvas;
        if (!this.mapLinks || this.mapLinks.lastCalcTime <= 100) {
            this.recalc();
        } else if (!this.recalcTimeout) {
            // big graph: recalc off the draw path, then ask for a redraw.
            // Hard 1.5s ceiling — stale paths are live-re-anchored in
            // drawLinks, but the full replan must still come soon.
            const delay = Math.min(this.mapLinks.lastCalcTime * 2, 1500);
            this.recalcTimeout = setTimeout(() => {
                this.recalcTimeout = null;
                this.recalc();
                redrawCanvas();
            }, delay);
        }
        if (!this.mapLinks) return false;
        this.mapLinks.drawLinks(ctx);
        return true;
    }
}

/* ==========================================================================
 * Part 2 — magnetic node snapping
 * ========================================================================== */

const MAGNET = {
    snap: 10,      // screen px: edge/dock alignment snaps
    slotSnap: 14,  // screen px: slot (straight-noodle) alignment snaps
    range: 200,    // canvas px: neighbour search radius
    gap: 32,       // canvas px: docking gap between node boxes
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

// Collect what is being dragged, split into nodes and groups (reroutes and
// anything else in selectedItems are ignored).
function draggedItems(canvas) {
    const LG = window.LiteGraph;
    const nodes = [], groups = [];
    if (!canvas.selectedItems) return { nodes, groups };
    for (const item of canvas.selectedItems) {
        if (LG?.LGraphGroup && item instanceof LG.LGraphGroup) groups.push(item);
        else if (LG?.LGraphNode && item instanceof LG.LGraphNode) nodes.push(item);
        else if (!LG?.LGraphNode && item.pos && item.size && item.getBounding) nodes.push(item);
    }
    return { nodes, groups };
}

// A group's frame box (its pos/size ARE the frame; no title offset needed).
function groupBox(g) {
    return { x1: g.pos[0], y1: g.pos[1], x2: g.pos[0] + g.size[0], y2: g.pos[1] + g.size[1] };
}

// Everything nested inside the dragged groups moves with them, so none of
// it may serve as a snap target. Returns { nodeIds, groups } of the nested
// content (recursing through sub-groups).
function nestedContent(groups) {
    const LG = window.LiteGraph;
    const nodeIds = new Set();
    const nested = new Set(groups);
    const stack = [...groups];
    while (stack.length) {
        const g = stack.pop();
        for (const child of g._children || []) {
            if (LG?.LGraphGroup && child instanceof LG.LGraphGroup) {
                if (!nested.has(child)) { nested.add(child); stack.push(child); }
            } else if (child.id !== undefined) {
                nodeIds.add(child.id);
            }
        }
        for (const n of g._nodes || []) nodeIds.add(n.id);
    }
    return { nodeIds, groups: nested };
}

/*
 * Smart align, Excalidraw/Figma style: a pure function of the free drag
 * position, evaluated synchronously after every pointer move. LiteGraph
 * moves dragged items by per-event pointer deltas, so an offset we add on
 * top persists. sim.off tracks that offset:
 *   node.pos = free position (pure mouse) + sim.off
 * Inside the snap threshold the offset is the full correction (hard snap,
 * guide line shown); outside it is zero (node exactly under the mouse).
 * No easing, no attraction, no timers — snapping in and releasing are
 * both instant, which is what makes it feel crisp instead of janky.
 */
const sim = {
    canvas: null,
    off: [0, 0],
    guides: [], // guide segments drawn by drawMagnetGuides
};

function stopSim() {
    sim.off = [0, 0];
    if (sim.guides.length && sim.canvas) {
        sim.canvas.dirty_canvas = true;
        sim.canvas.dirty_bgcanvas = true;
    }
    sim.guides = [];
    sim.canvas = null;
}

// All snap candidates for the current union box, per axis. Each candidate:
// { delta, weight, snapR, guide } — delta shifts the whole box, weight
// scores competing candidates (lower = stronger), snapR is the screen-px
// snap radius, guide is the dashed line drawn while snapped.
function collectCandidates(canvas, ub, draggedIds, excludedGroups, includeSlots) {
    const xs = [], ys = [];
    const graph = canvas.graph;
    const gap = MAGNET.gap;

    // snap targets: parked nodes and parked group frames
    const targetBoxes = [];
    for (const other of graph._nodes || []) {
        if (!draggedIds.has(other.id)) targetBoxes.push(nodeBox(other));
    }
    for (const g of graph._groups || []) {
        if (!excludedGroups.has(g)) targetBoxes.push(groupBox(g));
    }

    for (const ob of targetBoxes) {
        if (!boxesNear(ub, ob, MAGNET.range)) continue;

        const ySpan = [Math.min(ob.y1, ub.y1) - 16, Math.max(ob.y2, ub.y2) + 16];
        const xSpan = [Math.min(ob.x1, ub.x1) - 16, Math.max(ob.x2, ub.x2) + 16];
        const gx = (at) => ({ axis: "x", at, a: ySpan[0], b: ySpan[1] });
        const gy = (at) => ({ axis: "y", at, a: xSpan[0], b: xSpan[1] });

        // edge alignment (weight 1), centers (1.25, weaker)
        xs.push(
            { delta: ob.x1 - ub.x1, weight: 1, snapR: MAGNET.snap, guide: gx(ob.x1) },
            { delta: ob.x2 - ub.x2, weight: 1, snapR: MAGNET.snap, guide: gx(ob.x2) },
            { delta: (ob.x1 + ob.x2) / 2 - (ub.x1 + ub.x2) / 2, weight: 1.25, snapR: MAGNET.snap, guide: gx((ob.x1 + ob.x2) / 2) },
            // docking: sit beside with a uniform gap (0.9, slightly stronger)
            { delta: ob.x2 + gap - ub.x1, weight: 0.9, snapR: MAGNET.snap, guide: gx(ob.x2 + gap / 2) },
            { delta: ob.x1 - gap - ub.x2, weight: 0.9, snapR: MAGNET.snap, guide: gx(ob.x1 - gap / 2) },
        );
        ys.push(
            { delta: ob.y1 - ub.y1, weight: 1, snapR: MAGNET.snap, guide: gy(ob.y1) },
            { delta: ob.y2 - ub.y2, weight: 1, snapR: MAGNET.snap, guide: gy(ob.y2) },
            { delta: (ob.y1 + ob.y2) / 2 - (ub.y1 + ub.y2) / 2, weight: 1.25, snapR: MAGNET.snap, guide: gy((ob.y1 + ob.y2) / 2) },
            { delta: ob.y2 + gap - ub.y1, weight: 0.9, snapR: MAGNET.snap, guide: gy(ob.y2 + gap / 2) },
            { delta: ob.y1 - gap - ub.y2, weight: 0.9, snapR: MAGNET.snap, guide: gy(ob.y1 - gap / 2) },
        );
    }

    // slot alignment: pull connected slots level so the noodle runs straight
    // (weight 0.5 — the strongest magnet, and a wider snap radius).
    // Skipped when a group is being dragged: a group aligns by its frame only.
    if (!includeSlots) return { xs, ys };
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
            const targetY = outDragged ? iPos[1] : oPos[1];
            ys.push({
                delta, weight: 0.5, snapR: MAGNET.slotSnap,
                guide: {
                    axis: "y", at: targetY,
                    a: Math.min(oPos[0], iPos[0]) - 10,
                    b: Math.max(oPos[0], iPos[0]) + 10,
                },
            });
        } catch (e) { /* odd slot — skip */ }
    }
    return { xs, ys };
}

// One axis: pick the best candidate inside its snap radius, measured from
// the FREE position (current pos minus the offset we already applied).
// Returns the full correction, or 0 when nothing is in range.
function pickAxis(cands, axis, zoom) {
    let best = null;
    for (const c of cands) {
        const dFree = c.delta + sim.off[axis]; // candidate delta in the free frame
        const dScr = Math.abs(dFree) * zoom;
        if (dScr > c.snapR) continue;
        const score = dScr * c.weight;
        if (!best || score < best.score) best = { c, dFree, score };
    }
    if (!best) return 0;
    sim.guides.push(best.c.guide);
    return best.dFree;
}

// Runs synchronously after LiteGraph applies each pointer move: hard-snap
// to the best target in range, or return the selection exactly to the
// mouse. Groups align by their frame box; their contents ride along.
// soloDrag mirrors LiteGraph's Ctrl/Meta drag (group frame moves without
// its children), so our correction moves exactly what the drag moves.
function applySmartAlign(canvas, altFree, soloDrag) {
    const { nodes, groups } = draggedItems(canvas);
    if (!nodes.length && !groups.length) return;

    const nested = nestedContent(groups);
    // nodes the user also selected but that already ride inside a dragged
    // group must not be shifted twice
    const looseNodes = soloDrag ? nodes : nodes.filter((n) => !nested.nodeIds.has(n.id));

    const zoom = canvas.ds?.scale || 1;
    const boxes = [...looseNodes.map(nodeBox), ...groups.map(groupBox)];
    let ub = null;
    for (const b of boxes) {
        if (!ub) ub = { ...b };
        else {
            ub.x1 = Math.min(ub.x1, b.x1); ub.y1 = Math.min(ub.y1, b.y1);
            ub.x2 = Math.max(ub.x2, b.x2); ub.y2 = Math.max(ub.y2, b.y2);
        }
    }

    const prevGuides = sim.guides.length;
    sim.guides = [];

    const desired = [0, 0];
    if (!altFree) {
        const draggedIds = new Set(nodes.map((n) => n.id));
        // content moving inside dragged groups can't be a snap target either
        // (unless Ctrl-dragging the frame alone, when contents stay parked)
        if (!soloDrag) for (const id of nested.nodeIds) draggedIds.add(id);
        const excludedGroups = soloDrag ? new Set(groups) : nested.groups;
        const { xs, ys } = collectCandidates(
            canvas, ub, draggedIds, excludedGroups,
            groups.length === 0, // slots only for pure node drags
        );
        desired[0] = pickAxis(xs, 0, zoom);
        desired[1] = pickAxis(ys, 1, zoom);
    }

    const dx = desired[0] - sim.off[0];
    const dy = desired[1] - sim.off[1];
    if (dx || dy) {
        for (const g of groups) g.move(dx, dy, !!soloDrag);
        for (const n of looseNodes) {
            n.pos[0] += dx;
            n.pos[1] += dy;
        }
    }
    sim.off[0] = desired[0];
    sim.off[1] = desired[1];

    if (dx || dy || sim.guides.length !== prevGuides) {
        canvas.dirty_canvas = true;
        canvas.dirty_bgcanvas = true;
    }
}

// dashed accent lines along whatever the magnet is currently locked to;
// chained onto canvas.onDrawForeground (graph space)
function drawMagnetGuides(canvas, ctx) {
    if (!sim.guides.length || sim.canvas !== canvas) return;
    const zoom = canvas.ds?.scale || 1;
    ctx.save();
    ctx.strokeStyle = "rgba(69, 182, 224, 0.85)";
    ctx.lineWidth = 1.5 / zoom;
    ctx.setLineDash([6 / zoom, 4 / zoom]);
    for (const g of sim.guides) {
        ctx.beginPath();
        if (g.axis === "x") {
            ctx.moveTo(g.at, g.a);
            ctx.lineTo(g.at, g.b);
        } else {
            ctx.moveTo(g.a, g.at);
            ctx.lineTo(g.b, g.at);
        }
        ctx.stroke();
    }
    ctx.restore();
}

function chainGuideOverlay(canvas) {
    if (canvas.__trentMagnetOverlay) return;
    const prev = canvas.onDrawForeground;
    canvas.onDrawForeground = function (ctx, area) {
        if (prev) prev.call(this, ctx, area);
        try { drawMagnetGuides(this, ctx); } catch (e) { /* never break drawing */ }
    };
    canvas.__trentMagnetOverlay = true;
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
                if (board.draw(this, ctx)) return;
                // board has no usable plan (first recalc failed) — draw
                // native splines instead of leaving every link invisible
            } catch (e) {
                console.error("[TrentNoodles] circuit draw failed, falling back", e);
            }
        }
        return origDrawConnections.apply(this, arguments);
    };

    // Smart-align event wiring. NOTE: the canvas registers bound COPIES of
    // its pointer handlers at construction (this.processMouseMove.bind(this)),
    // so patching the prototype after startup never intercepts real mouse
    // events. Window listeners are immune to that. This one is BUBBLE phase:
    // the canvas's own pointermove (target) runs first and moves the dragged
    // nodes, then this fires in the same event and applies the snap — so the
    // drawn frame is always consistent, with no oscillation between the two.
    window.addEventListener("pointermove", (e) => {
        try {
            const canvas = app.canvas;
            if (!canvas || window.LiteGraph?.vueNodesMode) return;
            if (state.magnet && canvas.isDragging) {
                sim.canvas = canvas;
                chainGuideOverlay(canvas);
                // Ctrl/Meta = LiteGraph's "move group frame without contents"
                applySmartAlign(canvas, !!e.altKey, !!(e.ctrlKey || e.metaKey));
            } else if (sim.canvas) {
                stopSim();
            }
        } catch (err) {
            console.warn("[TrentNoodles] magnet error", err);
        }
    }, false);

    // the snap offset is already exact on the last move; just reset state
    // BEFORE the canvas handler finalizes the drag (capture = ancestors first)
    window.addEventListener("pointerup", () => {
        if (sim.canvas) stopSim();
    }, true);

    window.addEventListener("pointercancel", () => {
        if (sim.canvas) stopSim();
    }, true);

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
