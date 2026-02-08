import { app } from "/scripts/app.js";

/**
 * Frame Ramp Boogie - Dynamic Visibility + Live Curve Preview
 *
 * Features:
 * - Hides irrelevant controls based on current settings
 * - Live easing curve preview canvas that updates in real-time
 * - Bezier control point visualization with handle lines
 */

// ── Widget hide/show helpers ──────────────────────────────

function hideWidget(node, widget) {
    if (!widget || widget._hidden) return;
    widget._hidden = true;
    widget._origType = widget.type;
    widget._origComputeSize = widget.computeSize;
    widget.type = "trent_hidden";
    widget.computeSize = () => [0, -4];
}

function showWidget(node, widget) {
    if (!widget || !widget._hidden) return;
    widget._hidden = false;
    widget.type = widget._origType;
    widget.computeSize = widget._origComputeSize;
}

// ── Easing math (ported from utils/easing.py) ─────────────

const EASING_FNS = {
    linear: (t) => t,
    ease_in: (t) => t * t,
    ease_out: (t) => 1.0 - (1.0 - t) * (1.0 - t),
    ease_in_out: (t) => 3.0 * t * t - 2.0 * t * t * t,
    smooth: (t) => t * t * (3.0 - 2.0 * t),
};

function cubicBezier(t, p1x, p1y, p2x, p2y) {
    // Newton-Raphson: solve B_x(u) = t, then evaluate B_y(u)
    let u = t;
    for (let i = 0; i < 8; i++) {
        const u2 = u * u;
        const u3 = u2 * u;
        const omu = 1.0 - u;
        const omu2 = omu * omu;
        const bx =
            3.0 * p1x * u * omu2 +
            3.0 * p2x * u2 * omu +
            u3;
        let dbx =
            3.0 * p1x * (1.0 - 4.0 * u + 3.0 * u2) +
            3.0 * p2x * (2.0 * u - 3.0 * u2) +
            3.0 * u2;
        dbx = Math.max(dbx, 1e-7);
        u = Math.max(0.0, Math.min(1.0, u - (bx - t) / dbx));
    }
    const u2 = u * u;
    const u3 = u2 * u;
    const omu = 1.0 - u;
    const omu2 = omu * omu;
    const by =
        3.0 * p1y * u * omu2 +
        3.0 * p2y * u2 * omu +
        u3;
    return Math.max(0.0, Math.min(1.0, by));
}

// Must match Python _PRESET_VALUES exactly
const BEZIER_PRESETS = {
    ease: [0.25, 0.1, 0.25, 1.0],
    ease_in: [0.42, 0.0, 1.0, 1.0],
    ease_out: [0.0, 0.0, 0.58, 1.0],
    ease_in_out: [0.42, 0.0, 0.58, 1.0],
    sharp: [0.75, 0.0, 0.25, 1.0],
    gentle: [0.4, 0.0, 0.6, 1.0],
};

function evaluateEasing(t, easing, p1x, p1y, p2x, p2y) {
    if (easing === "cubic_bezier") {
        return cubicBezier(t, p1x, p1y, p2x, p2y);
    }
    const fn = EASING_FNS[easing] || EASING_FNS.linear;
    return fn(t);
}

// ── Curve preview drawing ─────────────────────────────────

function drawCurve(canvas, easing, p1x, p1y, p2x, p2y) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    const pad = 14;
    const gw = w - 2 * pad;
    const gh = h - 2 * pad;

    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = "#0a1218";
    ctx.fillRect(0, 0, w, h);

    // Subtle grid (4x4)
    ctx.strokeStyle = "#1a2530";
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(pad + (gw / 4) * i, pad);
        ctx.lineTo(pad + (gw / 4) * i, pad + gh);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pad, pad + (gh / 4) * i);
        ctx.lineTo(pad + gw, pad + (gh / 4) * i);
        ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "#2a3a48";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad + gh);
    ctx.lineTo(pad + gw, pad + gh);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, pad + gh);
    ctx.stroke();

    // Diagonal reference (linear)
    if (easing !== "linear") {
        ctx.strokeStyle = "#1a2530";
        ctx.lineWidth = 0.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(pad, pad + gh);
        ctx.lineTo(pad + gw, pad);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Bezier control point handles (behind curve)
    if (easing === "cubic_bezier") {
        const sx = pad;
        const sy = pad + gh;
        const ex = pad + gw;
        const ey = pad;
        const cp1x = pad + p1x * gw;
        const cp1y = pad + gh - p1y * gh;
        const cp2x = pad + p2x * gw;
        const cp2y = pad + gh - p2y * gh;

        ctx.strokeStyle = "rgba(255, 110, 64, 0.35)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(cp1x, cp1y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cp2x, cp2y);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Main curve
    ctx.strokeStyle = "#5c9cff";
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    const steps = 80;
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const v = evaluateEasing(t, easing, p1x, p1y, p2x, p2y);
        const x = pad + t * gw;
        const y = pad + gh - v * gh;
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // Bezier control point dots (on top)
    if (easing === "cubic_bezier") {
        const cp1x = pad + p1x * gw;
        const cp1y = pad + gh - p1y * gh;
        const cp2x = pad + p2x * gw;
        const cp2y = pad + gh - p2y * gh;

        ctx.fillStyle = "#ff6e40";
        ctx.beginPath();
        ctx.arc(cp1x, cp1y, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(cp2x, cp2y, 4, 0, 2 * Math.PI);
        ctx.fill();

        // Labels
        ctx.fillStyle = "#ff6e40";
        ctx.font = "bold 8px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("P1", cp1x, cp1y - 7);
        ctx.fillText("P2", cp2x, cp2y - 7);
    }

    // Axis labels
    ctx.fillStyle = "#556677";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("time", pad + gw / 2, pad + gh + 12);
    ctx.textAlign = "right";
    ctx.fillText("0", pad - 3, pad + gh + 3);
    ctx.fillText("1", pad - 3, pad + 5);
    ctx.textAlign = "left";
    ctx.fillText("1", pad + gw + 2, pad + gh + 3);

    // Easing name label
    ctx.fillStyle = "#7a8a99";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "right";
    const label = easing.replace(/_/g, " ");
    ctx.fillText(label, pad + gw, pad - 4);

    // "blend" Y-axis label
    ctx.fillStyle = "#556677";
    ctx.save();
    ctx.translate(5, pad + gh / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.font = "8px sans-serif";
    ctx.fillText("blend", 0, 0);
    ctx.restore();

    ctx.restore();
}

// ── Main extension ────────────────────────────────────────

app.registerExtension({
    name: "Trent.FrameRampBoogie",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "FrameRampBoogie") {
            return;
        }

        const findWidget = (name) =>
            node.widgets?.find((w) => w.name === name);

        // ── Create curve preview widget ───────────────

        const container = document.createElement("div");
        container.style.cssText = `
            width: 100%;
            box-sizing: border-box;
        `;

        const canvas = document.createElement("canvas");
        const dpr = window.devicePixelRatio || 1;
        const canvasW = 220;
        const canvasH = 110;
        canvas.width = canvasW * dpr;
        canvas.height = canvasH * dpr;
        canvas.style.cssText = `
            display: block;
            width: 100%;
            height: ${canvasH}px;
            border-radius: 4px;
        `;
        container.appendChild(canvas);

        const previewWidget = node.addDOMWidget(
            "easing_preview",
            "customCanvas",
            container
        );
        previewWidget.computeSize = () => [220, canvasH + 8];
        previewWidget.serializeValue = () => undefined;

        // ── Hide all optional widgets immediately ────
        // Prevents flash of bezier controls on node
        // creation. updateVisibility will show the
        // correct ones after a short delay.

        const allOptional = [
            "region_size", "bezier_preset",
            "p1_x", "p1_y", "p2_x", "p2_y",
        ];
        for (const name of allOptional) {
            const w = findWidget(name);
            if (w) hideWidget(node, w);
        }

        // ── Curve redraw function ─────────────────────

        const redrawCurve = () => {
            const easing =
                findWidget("easing")?.value || "linear";
            const preset =
                findWidget("bezier_preset")?.value || "ease";

            let p1x = findWidget("p1_x")?.value ?? 0.25;
            let p1y = findWidget("p1_y")?.value ?? 0.1;
            let p2x = findWidget("p2_x")?.value ?? 0.25;
            let p2y = findWidget("p2_y")?.value ?? 1.0;

            // Apply preset if not custom
            if (
                easing === "cubic_bezier" &&
                preset !== "custom" &&
                BEZIER_PRESETS[preset]
            ) {
                [p1x, p1y, p2x, p2y] = BEZIER_PRESETS[preset];
            }

            drawCurve(canvas, easing, p1x, p1y, p2x, p2y);
        };

        // ── Widget visibility logic ───────────────────

        const updateVisibility = () => {
            const easing = findWidget("easing")?.value;
            const region = findWidget("target_region")?.value;
            const preset = findWidget("bezier_preset")?.value;

            const isBezier = easing === "cubic_bezier";
            const isCustom = preset === "custom";
            const isFullBatch = region === "full_batch";

            // bezier_preset: only when easing is cubic_bezier
            const presetW = findWidget("bezier_preset");
            if (isBezier) {
                showWidget(node, presetW);
            } else {
                hideWidget(node, presetW);
            }

            // p1/p2 sliders: only when bezier + custom
            const sliders = ["p1_x", "p1_y", "p2_x", "p2_y"];
            for (const name of sliders) {
                const w = findWidget(name);
                if (isBezier && isCustom) {
                    showWidget(node, w);
                } else {
                    hideWidget(node, w);
                }
            }

            // region_size: only when not full_batch
            const regionW = findWidget("region_size");
            if (isFullBatch) {
                hideWidget(node, regionW);
            } else {
                showWidget(node, regionW);
            }

            // Redraw curve preview
            redrawCurve();

            node.setSize(node.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        // ── Hook widget callbacks ─────────────────────

        const watchWidgets = [
            "easing",
            "target_region",
            "bezier_preset",
        ];
        for (const name of watchWidgets) {
            const widget = findWidget(name);
            if (widget) {
                const orig = widget.callback;
                widget.callback = function (value) {
                    if (orig) orig.apply(this, arguments);
                    setTimeout(updateVisibility, 50);
                };
            }
        }

        // Hook bezier sliders for live curve updates
        const bezierSliders = ["p1_x", "p1_y", "p2_x", "p2_y"];
        for (const name of bezierSliders) {
            const widget = findWidget(name);
            if (widget) {
                const orig = widget.callback;
                widget.callback = function (value) {
                    if (orig) orig.apply(this, arguments);
                    setTimeout(redrawCurve, 10);
                };
            }
        }

        // ── Workflow load restore ─────────────────────

        const origConfigure = node.onConfigure;
        node.onConfigure = function (config) {
            if (origConfigure) {
                origConfigure.apply(this, arguments);
            }
            setTimeout(updateVisibility, 100);
        };

        // ── Initial update ────────────────────────────

        setTimeout(updateVisibility, 100);
    },
});
