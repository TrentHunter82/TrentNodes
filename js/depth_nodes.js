import { app } from "../../scripts/app.js";

/*
 * Trent Depth
 * -----------
 * Gives canvas nodes a raised, lit look, toggled from the Trent Tools sidebar
 * (see color_palette.js):
 *   - a deeper drop shadow under each node (palette DEFAULT_SHADOW_COLOR)
 *   - a top-lit sheen over the node body and a light bevel along the top edge
 *   - drop shadows under links (litegraph's render_connections_shadows)
 * Canvas mode only; Nodes 2.0 (Vue nodes) is left alone.
 */

const LS_DEPTH = "TrentDepth.on";

let depthOn = localStorage.getItem(LS_DEPTH) === "1"; // default OFF

export function getDepth() { return depthOn; }

export function setDepth(on) {
    depthOn = !!on;
    localStorage.setItem(LS_DEPTH, on ? "1" : "0");
    if (app.canvas) {
        app.canvas.render_connections_shadows = depthOn;
        app.canvas.setDirty(true, true);
    }
}

function nodePath(ctx, node, x, y, w, h) {
    const LG = window.LiteGraph;
    const shape = node.renderingShape;
    ctx.beginPath();
    if (shape === LG.ROUND_SHAPE || shape === LG.CARD_SHAPE) {
        const r = LG.ROUND_RADIUS;
        ctx.roundRect(x, y, w, h, shape === LG.CARD_SHAPE ? [r, r, 0, 0] : [r]);
    } else {
        ctx.rect(x, y, w, h);
    }
}

function drawSheen(ctx, node) {
    const b = node.boundingRect;
    const x = b[0] - node.pos[0], y = b[1] - node.pos[1], w = b[2], h = b[3];

    ctx.save();
    ctx.shadowColor = "transparent";
    const sheen = ctx.createLinearGradient(0, y, 0, y + h);
    sheen.addColorStop(0, "rgba(255,255,255,0.07)");
    sheen.addColorStop(0.35, "rgba(255,255,255,0)");
    sheen.addColorStop(1, "rgba(0,0,0,0.18)");
    nodePath(ctx, node, x, y, w, h);
    ctx.fillStyle = sheen;
    ctx.fill();

    const bevel = ctx.createLinearGradient(0, y, 0, y + Math.min(h, 40));
    bevel.addColorStop(0, "rgba(255,255,255,0.22)");
    bevel.addColorStop(1, "rgba(255,255,255,0)");
    nodePath(ctx, node, x + 0.5, y + 0.5, w - 1, h - 1);
    ctx.strokeStyle = bevel;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
}

function installPatch() {
    const proto = app.canvas?.constructor?.prototype;
    if (!proto || proto.__trentDepthPatched) return;
    proto.__trentDepthPatched = true;

    const origDrawNodeShape = proto.drawNodeShape;
    proto.drawNodeShape = function (node, ctx, ...rest) {
        const active = depthOn && !this.low_quality && !window.LiteGraph?.vueNodesMode;
        // drawNode() sets a 2px/3px shadow just before this call; the body
        // fill inside origDrawNodeShape is what casts it.
        if (active && ctx.shadowColor !== "transparent" && ctx.shadowColor !== "rgba(0, 0, 0, 0)") {
            const s = this.ds.scale;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 6 * s;
            ctx.shadowBlur = 18 * s;
        }
        origDrawNodeShape.call(this, node, ctx, ...rest);
        if (active) drawSheen(ctx, node);
    };
}

app.registerExtension({
    name: "TrentNodes.depthNodes",
    async setup() {
        installPatch();
        app.canvas.render_connections_shadows = depthOn;
    },
});
