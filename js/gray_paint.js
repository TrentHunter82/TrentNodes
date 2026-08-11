/**
 * Gray Paint Tracker Widget for TrentNodes
 *
 * Paint a gray region on the first frame of a video. The painted
 * mask is serialized as a base64 PNG into the hidden `mask_data`
 * widget; the Python node tracks a point and makes the region follow
 * motion across the clip.
 *
 * Built on the point_picker.js pattern (image via onExecuted, display
 * <-> image coord mapping, resize/configure handling) plus:
 *   - an offscreen native-resolution paint canvas (the serialized mask)
 *   - brush size / erase / clear controls
 *   - a "Set Anchor" mode that writes anchor_x / anchor_y + crosshair
 *   - the DA3 hidden-widget pattern from white_to_whatever.js
 */

import { app } from "../../scripts/app.js";

console.log("[TrentNodes] Gray Paint Tracker extension loading...");

app.registerExtension({
    name: "Trent.GrayPaintTracker",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "GrayPaintTracker") {
            return;
        }

        console.log("[TrentNodes] Registering GrayPaintTracker node");

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // ---------------------------------------------------------
            // DOM scaffolding
            // ---------------------------------------------------------
            const container = document.createElement("div");
            container.style.cssText = `
                position: relative;
                width: 100%;
                background: #1a1a1a;
                overflow: hidden;
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                display: flex;
                align-items: center;
                justify-content: center;
            `;

            // --- info / control bar ---
            const bar = document.createElement("div");
            bar.style.cssText = `
                position: absolute;
                top: 4px;
                left: 4px;
                right: 4px;
                z-index: 10;
                display: flex;
                flex-wrap: wrap;
                gap: 4px;
                align-items: center;
                font-family: monospace;
                font-size: 11px;
                pointer-events: none;
            `;
            container.appendChild(bar);

            const mkBtn = (label) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.style.cssText = `
                    padding: 2px 6px;
                    background: rgba(0,0,0,0.7);
                    color: #ddd;
                    border: 1px solid #444;
                    border-radius: 3px;
                    cursor: pointer;
                    font-size: 11px;
                    pointer-events: auto;
                `;
                return b;
            };

            const status = document.createElement("div");
            status.style.cssText = `
                padding: 2px 6px;
                background: rgba(0,0,0,0.7);
                color: #0f0;
                border-radius: 3px;
                pointer-events: none;
            `;
            status.textContent = "Run once to load frame";
            bar.appendChild(status);

            // brush size slider
            const brushWrap = document.createElement("label");
            brushWrap.style.cssText =
                "color:#bbb;display:flex;align-items:center;gap:3px;" +
                "background:rgba(0,0,0,0.6);padding:1px 5px;border-radius:3px;" +
                "pointer-events:auto;";
            brushWrap.appendChild(document.createTextNode("brush"));
            const brush = document.createElement("input");
            brush.type = "range";
            brush.min = "2";
            brush.max = "200";
            brush.value = "40";
            brush.style.width = "70px";
            brushWrap.appendChild(brush);
            bar.appendChild(brushWrap);

            const eraseBtn = mkBtn("Erase: off");
            bar.appendChild(eraseBtn);
            const anchorBtn = mkBtn("Set Anchor: off");
            bar.appendChild(anchorBtn);
            const clearBtn = mkBtn("Clear");
            bar.appendChild(clearBtn);

            // --- visible canvas (image + paint overlay) ---
            const canvas = document.createElement("canvas");
            canvas.width = 400;
            canvas.height = 300;
            canvas.style.cssText = `
                display: block;
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                cursor: crosshair;
                margin: 0 auto;
            `;
            container.appendChild(canvas);
            const ctx = canvas.getContext("2d");

            // --- offscreen paint canvas (the serialized mask) ---
            const paint = document.createElement("canvas");
            paint.width = canvas.width;
            paint.height = canvas.height;
            const pctx = paint.getContext("2d");

            // ---------------------------------------------------------
            // State
            // ---------------------------------------------------------
            const S = {
                image: null,
                imgW: 0,
                imgH: 0,
                brushSize: 40,
                erasing: false,
                anchorMode: false,
                drawing: false,
                lastX: 0,
                lastY: 0,
                anchorX: 0,
                anchorY: 0,
                widgetHeight: 300,
            };
            this.grayPaint = S;

            const widget = this.addDOMWidget(
                "gray_paint_canvas",
                "customCanvas",
                container,
                { serialize: false }
            );
            widget.computeSize = (width) => [width, S.widgetHeight];

            // ---------------------------------------------------------
            // Widget accessors
            // ---------------------------------------------------------
            const getW = (name) => this.widgets?.find((w) => w.name === name);
            const maskWidget = getW("mask_data");

            // Hide the (large) mask_data string widget using the DA3
            // pattern: keep it in this.widgets so it serializes, but
            // collapse it and null its type during draw so it doesn't
            // render.
            if (maskWidget) {
                maskWidget.origType = maskWidget.type;
                maskWidget.origComputeSize = maskWidget.computeSize;
                maskWidget.computeSize = () => [0, -4];
                maskWidget.type = "converted-widget-graypaint";
                maskWidget.hidden = true;

                const origDraw = this.onDrawForeground;
                this.onDrawForeground = function () {
                    const hidden = this.widgets.filter(
                        (w) =>
                            typeof w.type === "string" &&
                            w.type.includes("converted-widget")
                    );
                    const saved = hidden.map((w) => w.type);
                    hidden.forEach((w) => (w.type = null));
                    const r = origDraw
                        ? origDraw.apply(this, arguments)
                        : undefined;
                    hidden.forEach((w, i) => (w.type = saved[i]));
                    return r;
                };
            }

            // ---------------------------------------------------------
            // Drawing helpers
            // ---------------------------------------------------------
            const updateStatus = () => {
                const mode = S.anchorMode
                    ? "ANCHOR"
                    : S.erasing
                    ? "ERASE"
                    : "PAINT";
                status.textContent =
                    `${mode} | brush ${S.brushSize} | ` +
                    `anchor (${S.anchorX},${S.anchorY})`;
            };

            this.redrawGrayPaint = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (S.image) {
                    ctx.drawImage(S.image, 0, 0, canvas.width, canvas.height);
                } else {
                    ctx.fillStyle = "#222";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#666";
                    // the bitmap can be native video resolution (restored
                    // mask, no frame yet) — scale the caption with it so it
                    // stays readable at the displayed size
                    const fs = Math.max(14, Math.round(canvas.width / 28));
                    ctx.font = `${fs}px sans-serif`;
                    ctx.textAlign = "center";
                    ctx.fillText(
                        "Run once to load the first frame",
                        canvas.width / 2,
                        canvas.height / 2 - fs * 0.6
                    );
                    ctx.fillText(
                        "then paint a gray region",
                        canvas.width / 2,
                        canvas.height / 2 + fs
                    );
                }

                // paint overlay (translucent so the frame shows through)
                ctx.save();
                ctx.globalAlpha = 0.55;
                ctx.drawImage(paint, 0, 0, canvas.width, canvas.height);
                ctx.restore();

                // anchor crosshair (in image space -> display space).
                // Only meaningful in manual anchor mode — in centroid mode
                // the backend ignores anchor_x/y, so drawing a mark at the
                // default (0,0) would just be a phantom in the corner.
                if (S.imgW > 0 && getW("anchor_mode")?.value === "manual") {
                    const sx = canvas.width / S.imgW;
                    const sy = canvas.height / S.imgH;
                    const dx = S.anchorX * sx;
                    const dy = S.anchorY * sy;
                    const arm = 12;
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = "#fff";
                    crosshair(ctx, dx, dy, arm);
                    ctx.lineWidth = 1.5;
                    ctx.strokeStyle = "#f00";
                    crosshair(ctx, dx, dy, arm);
                    ctx.fillStyle = "#0f0";
                    ctx.beginPath();
                    ctx.arc(dx, dy, 3, 0, 2 * Math.PI);
                    ctx.fill();
                }
            };

            const crosshair = (c, x, y, arm) => {
                c.beginPath();
                c.moveTo(x - arm, y);
                c.lineTo(x + arm, y);
                c.stroke();
                c.beginPath();
                c.moveTo(x, y - arm);
                c.lineTo(x, y + arm);
                c.stroke();
            };

            // Map a pointer event to paint-canvas (== native frame) pixels.
            const toPaintCoords = (e) => {
                const rect = canvas.getBoundingClientRect();
                // hidden canvas (collapsed node mid-capture) → zero rect →
                // garbage coords that would smear into a mask corner
                if (!rect.width || !rect.height) return null;
                // object-fit: contain can letterbox the bitmap inside the
                // element box — map through the content box, not the rect.
                // Clamp: with pointer capture the pointer can leave the
                // canvas mid-stroke; keep the stroke pinned to the edge.
                const scale = Math.min(
                    rect.width / canvas.width,
                    rect.height / canvas.height
                ) || 1;
                const offX = (rect.width - canvas.width * scale) / 2;
                const offY = (rect.height - canvas.height * scale) / 2;
                const x = Math.max(0, Math.min(canvas.width,
                    (e.clientX - rect.left - offX) / scale));
                const y = Math.max(0, Math.min(canvas.height,
                    (e.clientY - rect.top - offY) / scale));
                return { x, y };
            };

            const applyBrush = () => {
                pctx.lineCap = "round";
                pctx.lineJoin = "round";
                pctx.lineWidth = S.brushSize;
                pctx.strokeStyle = "rgba(128,128,128,1)";
                pctx.fillStyle = "rgba(128,128,128,1)";
                pctx.globalCompositeOperation = S.erasing
                    ? "destination-out"
                    : "source-over";
            };

            const serializeMask = () => {
                if (!maskWidget) return;
                // toDataURL -> "data:image/png;base64,XXXX"; store raw base64.
                const url = paint.toDataURL("image/png");
                const b64 = url.split(",", 2)[1] || "";
                maskWidget.value = b64;
                if (maskWidget.callback) maskWidget.callback(b64);
                app.graph.setDirtyCanvas(true, true);
            };

            // ---------------------------------------------------------
            // Pointer interaction
            // ---------------------------------------------------------
            canvas.addEventListener("pointerdown", (e) => {
                if (!S.imgW) return;
                if (e.button !== 0) return;
                const pc = toPaintCoords(e);
                if (!pc) return;
                const { x, y } = pc;

                if (S.anchorMode) {
                    // Set anchor in native image coords.
                    S.anchorX = Math.max(
                        0,
                        Math.min(S.imgW - 1, Math.round(x))
                    );
                    S.anchorY = Math.max(
                        0,
                        Math.min(S.imgH - 1, Math.round(y))
                    );
                    const ax = getW("anchor_x");
                    const ay = getW("anchor_y");
                    const am = getW("anchor_mode");
                    if (ax) {
                        ax.value = S.anchorX;
                        if (ax.callback) ax.callback(S.anchorX);
                    }
                    if (ay) {
                        ay.value = S.anchorY;
                        if (ay.callback) ay.callback(S.anchorY);
                    }
                    if (am) {
                        am.value = "manual";
                        if (am.callback) am.callback("manual");
                    }
                    updateStatus();
                    this.redrawGrayPaint();
                    app.graph.setDirtyCanvas(true, true);
                    return;
                }

                // Paint mode: begin stroke with a dot. Capture the pointer
                // so the stroke survives crossing the toolbar strip or
                // leaving the canvas — it ends on pointerup/cancel only.
                S.drawing = true;
                try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* ok */ }
                S.lastX = x;
                S.lastY = y;
                applyBrush();
                pctx.beginPath();
                pctx.arc(x, y, S.brushSize / 2, 0, 2 * Math.PI);
                pctx.fill();
                this.redrawGrayPaint();
            });

            canvas.addEventListener("pointermove", (e) => {
                if (!S.drawing) return;
                // lost pointerup (failed capture, chorded buttons): the
                // left button is no longer down — end, don't keep painting
                if (!(e.buttons & 1)) { endStroke(); return; }
                const pc = toPaintCoords(e);
                if (!pc) return;
                const { x, y } = pc;
                applyBrush();
                pctx.beginPath();
                pctx.moveTo(S.lastX, S.lastY);
                pctx.lineTo(x, y);
                pctx.stroke();
                S.lastX = x;
                S.lastY = y;
                this.redrawGrayPaint();
            });

            const endStroke = () => {
                if (!S.drawing) return;
                S.drawing = false;
                serializeMask();
            };
            canvas.addEventListener("pointerup", endStroke);
            canvas.addEventListener("pointercancel", endStroke);
            canvas.addEventListener("lostpointercapture", endStroke);

            // Wheel over canvas adjusts brush size.
            canvas.addEventListener(
                "wheel",
                (e) => {
                    e.preventDefault();
                    const delta = e.deltaY < 0 ? 4 : -4;
                    S.brushSize = Math.max(
                        2,
                        Math.min(200, S.brushSize + delta)
                    );
                    brush.value = S.brushSize;
                    updateStatus();
                },
                { passive: false }
            );

            // ---------------------------------------------------------
            // Controls
            // ---------------------------------------------------------
            brush.addEventListener("input", () => {
                S.brushSize = parseInt(brush.value, 10) || 40;
                updateStatus();
            });
            eraseBtn.addEventListener("click", () => {
                S.erasing = !S.erasing;
                eraseBtn.textContent = `Erase: ${S.erasing ? "on" : "off"}`;
                if (S.erasing) {
                    S.anchorMode = false;
                    anchorBtn.textContent = "Set Anchor: off";
                }
                updateStatus();
            });
            anchorBtn.addEventListener("click", () => {
                S.anchorMode = !S.anchorMode;
                anchorBtn.textContent =
                    `Set Anchor: ${S.anchorMode ? "on" : "off"}`;
                if (S.anchorMode) {
                    S.erasing = false;
                    eraseBtn.textContent = "Erase: off";
                }
                updateStatus();
            });
            clearBtn.addEventListener("click", () => {
                pctx.save();
                pctx.globalCompositeOperation = "source-over";
                pctx.clearRect(0, 0, paint.width, paint.height);
                pctx.restore();
                serializeMask();
                this.redrawGrayPaint();
            });

            // ---------------------------------------------------------
            // Sizing to native frame resolution
            // ---------------------------------------------------------
            // Chrome = title + slots + parameter widgets above the canvas.
            // Derived from computeSize instead of a constant so the math
            // stays right when widgets are added or hidden (the old
            // hardcoded 140 was copied from point_picker's 2-widget stack
            // and left the node fighting every resize).
            const chromeHeight = () =>
                Math.max(0, this.computeSize()[1] - S.widgetHeight);

            const resizeToImage = (w, h, clearMask = true) => {
                // Only resize (which clears) the paint canvas when the
                // frame dimensions actually change, so repeated runs of
                // the same video preserve the painted mask.
                const dimsChanged = S.imgW !== w || S.imgH !== h;
                if (paint.width !== w || paint.height !== h) {
                    paint.width = w;
                    paint.height = h;
                    if (clearMask && maskWidget) {
                        // the bitmap was just wiped — the serialized mask
                        // must not silently keep the old-resolution paint
                        maskWidget.value = "";
                        if (maskWidget.callback) maskWidget.callback("");
                    }
                }
                canvas.width = w;
                canvas.height = h;
                S.imgW = w;
                S.imgH = h;

                // keep the user's node size when the frame is unchanged
                if (!dimsChanged) return;

                if (!clearMask) {
                    // restore path (workflow load): the node carries a saved
                    // size — fit the widget into it instead of resizing the
                    // node to the mask aspect, or every open stomps the size
                    const chrome = chromeHeight();
                    S.widgetHeight = Math.max(160, (this.size[1] || 0) - chrome);
                    container.style.height = S.widgetHeight + "px";
                    return;
                }

                const nodeWidth = this.size[0] || 400;
                const avail = nodeWidth - 20;
                const aspect = h / w;
                const newH = Math.min(420, Math.round(avail * aspect));

                this._gpResizing = true;
                S.widgetHeight = Math.max(160, newH);
                container.style.height = S.widgetHeight + "px";
                this.setSize(this.computeSize());
                setTimeout(() => {
                    this._gpResizing = false;
                }, 50);
            };

            // ---------------------------------------------------------
            // Image arrives from Python
            // ---------------------------------------------------------
            const prevOnExecuted = this.onExecuted;
            this.onExecuted = (message) => {
                if (prevOnExecuted) prevOnExecuted.call(this, message);
                if (message.preview_image && message.preview_image[0]) {
                    const img = new Image();
                    img.onload = () => {
                        S.image = img;
                        resizeToImage(img.width, img.height);
                        updateStatus();
                        this.redrawGrayPaint();
                    };
                    img.src =
                        "data:image/jpeg;base64," + message.preview_image[0];
                }
            };

            // ---------------------------------------------------------
            // Restore painted mask after workflow load
            // ---------------------------------------------------------
            const restoreFromWidgets = () => {
                const ax = getW("anchor_x");
                const ay = getW("anchor_y");
                if (ax) S.anchorX = ax.value || 0;
                if (ay) S.anchorY = ay.value || 0;

                if (maskWidget && maskWidget.value) {
                    const img = new Image();
                    img.onload = () => {
                        // Always route through resizeToImage so paint,
                        // canvas, and S.imgW change together — a partial
                        // resize here desyncs the dims that the wipe/clear
                        // logic compares. clearMask=false keeps mask_data.
                        resizeToImage(img.width, img.height, false);
                        pctx.save();
                        pctx.globalCompositeOperation = "source-over";
                        pctx.clearRect(0, 0, paint.width, paint.height);
                        pctx.drawImage(img, 0, 0);
                        pctx.restore();
                        this.redrawGrayPaint();
                    };
                    img.src = "data:image/png;base64," + maskWidget.value;
                }
                updateStatus();
                this.redrawGrayPaint();
            };

            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function (config) {
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }
                // a configured node has a saved size — the init timer must
                // not stomp it
                this.__gpConfigured = true;
                setTimeout(restoreFromWidgets, 100);
            };

            // ---------------------------------------------------------
            // Node resize
            // ---------------------------------------------------------
            const originalOnResize = this.onResize;
            this.onResize = function (size) {
                if (originalOnResize) {
                    originalOnResize.apply(this, arguments);
                }
                if (this._gpResizing) return;
                this._gpResizing = true;
                const newH = Math.max(160, size[1] - chromeHeight());
                if (Math.abs(newH - S.widgetHeight) > 5) {
                    S.widgetHeight = newH;
                    container.style.height = newH + "px";
                    this.redrawGrayPaint();
                }
                setTimeout(() => {
                    this._gpResizing = false;
                }, 50);
            };

            // ---------------------------------------------------------
            // Initial layout
            // ---------------------------------------------------------
            // Editing anchor_x / anchor_y / anchor_mode must move the
            // crosshair live (point_picker's setupWidgetCallbacks pattern —
            // it was dropped in the copy).
            const hookAnchorWidgets = () => {
                for (const [name, key] of [
                    ["anchor_x", "anchorX"],
                    ["anchor_y", "anchorY"],
                ]) {
                    const w = getW(name);
                    if (!w || w.__gpHooked) continue;
                    w.__gpHooked = true;
                    const prevCb = w.callback;
                    w.callback = (v, ...rest) => {
                        const r = prevCb ? prevCb.call(w, v, ...rest) : undefined;
                        S[key] = Math.round(Number(v)) || 0;
                        updateStatus();
                        this.redrawGrayPaint();
                        return r;
                    };
                }
                const am = getW("anchor_mode");
                if (am && !am.__gpHooked) {
                    am.__gpHooked = true;
                    const prevCb = am.callback;
                    am.callback = (v, ...rest) => {
                        const r = prevCb ? prevCb.call(am, v, ...rest) : undefined;
                        this.redrawGrayPaint();
                        return r;
                    };
                }
            };
            hookAnchorWidgets();

            setTimeout(() => {
                updateStatus();
                this.redrawGrayPaint();
                if (!this.__gpConfigured) {
                    // fresh node: give it a sane default footprint. A
                    // loaded node keeps its saved size (the old forced
                    // [w, 520] discarded it on every workflow open).
                    const nodeWidth = Math.max(380, this.size[0] || 380);
                    container.style.height = S.widgetHeight + "px";
                    this.setSize([nodeWidth, this.computeSize()[1]]);
                }
            }, 100);

            return result;
        };
    },
});
