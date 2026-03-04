/**
 * Video Layer Ho Down - Interactive Canvas Extension
 *
 * Multi-layer compositing with drag-to-position canvas.
 * Dynamic layer inputs (connect one, next appears).
 * Per-layer widgets shown only when connected.
 * All widgets stay visible and serialize normally.
 */

import { app } from "/scripts/app.js";

const MAX_LAYERS = 5;
const BLEND_MODES = [
    "normal", "multiply", "screen", "overlay", "add",
];

function drawCheckerboard(ctx, x, y, w, h, cell) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    const cols = Math.ceil(w / cell);
    const rows = Math.ceil(h / cell);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            ctx.fillStyle =
                (r + c) % 2 === 0 ? "#444" : "#333";
            ctx.fillRect(
                x + c * cell, y + r * cell,
                cell, cell
            );
        }
    }
    ctx.restore();
}

app.registerExtension({
    name: "Trent.VideoLayerHoDown",

    async nodeCreated(node) {
        if (
            node.constructor.comfyClass !==
            "VideoLayerHoDown"
        ) {
            return;
        }

        // =============================================
        // 1. STATE
        // =============================================

        const state = {
            canvas: null,
            ctx: null,
            container: null,
            widgetHeight: 250,
            displayScale: 1.0,
            displayOffsetX: 0,
            displayOffsetY: 0,
            bgImage: null,
            bgW: 0,
            bgH: 0,
            // Per-layer preview data: { idx: {image, w, h} }
            layers: {},
            selectedLayer: null,
            dragging: false,
            dragStartX: 0,
            dragStartY: 0,
            dragOffsetStartX: 0,
            dragOffsetStartY: 0,
        };
        node._layerState = state;

        // =============================================
        // 2. WIDGET HELPERS
        // =============================================

        const findWidget = (name) =>
            node.widgets?.find((w) => w.name === name);

        /**
         * Get layer x value from its widget
         */
        const getLayerX = (idx) => {
            const w = findWidget(`layer_${idx}_x`);
            return w ? w.value : 0;
        };

        const getLayerY = (idx) => {
            const w = findWidget(`layer_${idx}_y`);
            return w ? w.value : 0;
        };

        /**
         * Set layer x/y widget values
         */
        const setLayerXY = (idx, x, y) => {
            const xw = findWidget(`layer_${idx}_x`);
            const yw = findWidget(`layer_${idx}_y`);
            if (xw) {
                xw.value = x;
                if (xw.callback) xw.callback(x);
            }
            if (yw) {
                yw.value = y;
                if (yw.callback) yw.callback(y);
            }
        };

        /**
         * Create per-layer widgets for a given index.
         * Widgets stay visible and serialize normally.
         */
        const ensureLayerWidgets = (idx) => {
            const names = {
                x: `layer_${idx}_x`,
                y: `layer_${idx}_y`,
                scale: `layer_${idx}_scale`,
                opacity: `layer_${idx}_opacity`,
                blend: `layer_${idx}_blend`,
            };

            if (!findWidget(names.x)) {
                node.addWidget(
                    "number", names.x, 0,
                    () => { redrawCanvas(); },
                    {
                        min: -8192, max: 8192,
                        step: 1, precision: 0,
                    }
                );
            }
            if (!findWidget(names.y)) {
                node.addWidget(
                    "number", names.y, 0,
                    () => { redrawCanvas(); },
                    {
                        min: -8192, max: 8192,
                        step: 1, precision: 0,
                    }
                );
            }
            if (!findWidget(names.scale)) {
                node.addWidget(
                    "slider", names.scale, 1.0,
                    () => {},
                    {
                        min: 0.01, max: 10.0,
                        step: 0.01, precision: 2,
                    }
                );
            }
            if (!findWidget(names.opacity)) {
                node.addWidget(
                    "slider", names.opacity, 1.0,
                    () => {},
                    {
                        min: 0.0, max: 1.0,
                        step: 0.01, precision: 2,
                    }
                );
            }
            if (!findWidget(names.blend)) {
                node.addWidget(
                    "combo", names.blend, "normal",
                    () => {},
                    { values: BLEND_MODES }
                );
            }
        };

        /**
         * Remove per-layer widgets
         */
        const removeLayerWidgets = (idx) => {
            const prefixes = [
                `layer_${idx}_x`,
                `layer_${idx}_y`,
                `layer_${idx}_scale`,
                `layer_${idx}_opacity`,
                `layer_${idx}_blend`,
            ];
            node.widgets = node.widgets.filter(
                (w) => !prefixes.includes(w.name)
            );
            // Clear preview data
            delete state.layers[idx];
        };

        /**
         * Sort layer widgets: group by layer number,
         * order: x, y, scale, opacity, blend
         */
        const sortLayerWidgets = () => {
            if (!node.widgets) return;
            const layerW = [];
            const otherW = [];
            for (const w of node.widgets) {
                if (w.name.match(/^layer_\d+_/)) {
                    layerW.push(w);
                } else {
                    otherW.push(w);
                }
            }
            const order = {
                x: 0, y: 1, scale: 2,
                opacity: 3, blend: 4,
            };
            layerW.sort((a, b) => {
                const aIdx = parseInt(
                    a.name.match(/^layer_(\d+)_/)[1]
                );
                const bIdx = parseInt(
                    b.name.match(/^layer_(\d+)_/)[1]
                );
                if (aIdx !== bIdx) return aIdx - bIdx;
                const aKey = a.name.split("_").pop();
                const bKey = b.name.split("_").pop();
                return (order[aKey] ?? 9) -
                    (order[bKey] ?? 9);
            });
            node.widgets.length = 0;
            node.widgets.push(...otherW, ...layerW);
        };

        // =============================================
        // 3. DYNAMIC INPUTS (WanVace pattern)
        // =============================================

        const getLayerIndices = () => {
            const indices = [];
            for (const inp of node.inputs || []) {
                const m = inp.name.match(
                    /^layer_(\d+)$/
                );
                if (m) indices.push(parseInt(m[1]));
            }
            return indices.sort((a, b) => a - b);
        };

        const isLayerConnected = (idx) => {
            const inp = node.inputs?.find(
                (i) => i.name === `layer_${idx}`
            );
            return inp && inp.link !== null;
        };

        const addLayerInput = (idx) => {
            const name = `layer_${idx}`;
            if (node.inputs?.find(
                (i) => i.name === name
            )) {
                return false;
            }
            node.addInput(name, "IMAGE");
            return true;
        };

        const removeLayerInput = (idx) => {
            const name = `layer_${idx}`;
            const i = node.inputs?.findIndex(
                (inp) => inp.name === name
            );
            if (i >= 0) {
                const inp = node.inputs[i];
                if (inp.link !== null) {
                    app.graph.removeLink(inp.link);
                }
                node.removeInput(i);
                return true;
            }
            return false;
        };

        const updateDynamicInputs = () => {
            const indices = getLayerIndices();
            if (indices.length === 0) {
                addLayerInput(1);
                return;
            }

            let maxIdx = Math.max(...indices);

            if (
                isLayerConnected(maxIdx) &&
                maxIdx < MAX_LAYERS
            ) {
                addLayerInput(maxIdx + 1);
            } else {
                while (maxIdx > 1) {
                    const prev = maxIdx - 1;
                    if (
                        !isLayerConnected(maxIdx) &&
                        !isLayerConnected(prev)
                    ) {
                        removeLayerInput(maxIdx);
                        removeLayerWidgets(maxIdx);
                        maxIdx = prev;
                    } else {
                        break;
                    }
                }
            }

            // Show/hide per-layer widgets
            const updated = getLayerIndices();
            for (const idx of updated) {
                if (isLayerConnected(idx)) {
                    ensureLayerWidgets(idx);
                } else {
                    removeLayerWidgets(idx);
                }
            }

            sortLayerWidgets();
            node.setSize(node.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        // =============================================
        // 4. CANVAS WIDGET
        // =============================================

        const container = document.createElement("div");
        container.style.cssText = `
            position: relative;
            width: 100%;
            background: #0a1218;
            overflow: hidden;
            box-sizing: border-box;
        `;
        state.container = container;

        // Info bar
        const infoBar = document.createElement("div");
        infoBar.style.cssText = `
            position: absolute;
            top: 4px; left: 4px; right: 4px;
            z-index: 10;
            display: flex;
            justify-content: space-between;
            align-items: center;
            pointer-events: none;
        `;
        container.appendChild(infoBar);

        const coordsDisplay =
            document.createElement("div");
        coordsDisplay.style.cssText = `
            padding: 3px 7px;
            background: rgba(0,0,0,0.7);
            color: #0f0;
            border-radius: 3px;
            font-size: 10px;
            font-family: monospace;
            pointer-events: none;
        `;
        coordsDisplay.textContent = "No layer selected";
        infoBar.appendChild(coordsDisplay);

        const resetBtn =
            document.createElement("button");
        resetBtn.textContent = "Center";
        resetBtn.style.cssText = `
            padding: 2px 7px;
            background: rgba(255,255,255,0.15);
            color: #ccc;
            border: 1px solid #444;
            border-radius: 3px;
            cursor: pointer;
            font-size: 9px;
            pointer-events: auto;
        `;
        infoBar.appendChild(resetBtn);

        const canvas = document.createElement("canvas");
        canvas.width = 400;
        canvas.height = 250;
        canvas.style.cssText = `
            display: block;
            max-width: 100%;
            cursor: default;
            margin: 0 auto;
        `;
        container.appendChild(canvas);
        const ctx = canvas.getContext("2d");
        state.canvas = canvas;
        state.ctx = ctx;

        const canvasWidget = node.addDOMWidget(
            "layer_canvas", "customCanvas", container
        );
        canvasWidget.computeSize = (width) => {
            return [width, state.widgetHeight];
        };
        canvasWidget.serializeValue = () => undefined;

        // =============================================
        // 5. CANVAS RENDERING
        // =============================================

        const redrawCanvas = () => {
            const s = state;
            const { canvas: c, ctx: cx } = s;

            cx.clearRect(0, 0, c.width, c.height);

            if (!s.bgImage) {
                cx.fillStyle = "#1a1a1a";
                cx.fillRect(
                    0, 0, c.width, c.height
                );
                cx.fillStyle = "#555";
                cx.font = "12px sans-serif";
                cx.textAlign = "center";
                cx.fillText(
                    "Run to load preview",
                    c.width / 2, c.height / 2
                );
                return;
            }

            // Fit bg in canvas
            const scX = c.width / s.bgW;
            const scY = c.height / s.bgH;
            s.displayScale = Math.min(scX, scY);
            const dw = s.bgW * s.displayScale;
            const dh = s.bgH * s.displayScale;
            s.displayOffsetX = (c.width - dw) / 2;
            s.displayOffsetY = (c.height - dh) / 2;

            drawCheckerboard(
                cx, s.displayOffsetX,
                s.displayOffsetY, dw, dh, 8
            );

            cx.drawImage(
                s.bgImage, s.displayOffsetX,
                s.displayOffsetY, dw, dh
            );

            // Draw layers
            const layerIndices = Object.keys(
                s.layers
            ).map(Number).sort((a, b) => a - b);

            for (const idx of layerIndices) {
                const lyr = s.layers[idx];
                if (!lyr || !lyr.image) continue;

                // Read position from widgets
                const lx_img = getLayerX(idx);
                const ly_img = getLayerY(idx);

                const lw = lyr.w * s.displayScale;
                const lh = lyr.h * s.displayScale;
                const lx =
                    s.displayOffsetX +
                    lx_img * s.displayScale;
                const ly =
                    s.displayOffsetY +
                    ly_img * s.displayScale;

                cx.drawImage(
                    lyr.image, lx, ly, lw, lh
                );

                const selected =
                    s.selectedLayer === idx;
                cx.strokeStyle = selected
                    ? "#ff0"
                    : "rgba(255,255,255,0.4)";
                cx.lineWidth = selected ? 2 : 1;
                cx.setLineDash(
                    selected ? [] : [4, 4]
                );
                cx.strokeRect(lx, ly, lw, lh);
                cx.setLineDash([]);

                // Layer label
                cx.fillStyle = selected
                    ? "#ff0"
                    : "rgba(255,255,255,0.6)";
                cx.font = "bold 10px monospace";
                cx.textAlign = "left";
                cx.fillText(
                    `L${idx}`, lx + 3, ly + 12
                );

                // Crosshair during drag
                if (selected && s.dragging) {
                    const cxr = lx + lw / 2;
                    const cyr = ly + lh / 2;
                    cx.strokeStyle =
                        "rgba(255,255,0,0.3)";
                    cx.lineWidth = 1;
                    cx.setLineDash([2, 4]);
                    cx.beginPath();
                    cx.moveTo(
                        cxr, s.displayOffsetY
                    );
                    cx.lineTo(
                        cxr, s.displayOffsetY + dh
                    );
                    cx.stroke();
                    cx.beginPath();
                    cx.moveTo(
                        s.displayOffsetX, cyr
                    );
                    cx.lineTo(
                        s.displayOffsetX + dw, cyr
                    );
                    cx.stroke();
                    cx.setLineDash([]);
                }
            }

            // Coords display
            if (s.selectedLayer !== null) {
                const sx = getLayerX(s.selectedLayer);
                const sy = getLayerY(s.selectedLayer);
                coordsDisplay.textContent =
                    `L${s.selectedLayer}:` +
                    ` (${sx}, ${sy})`;
            } else {
                coordsDisplay.textContent =
                    "No layer selected";
            }
        };

        // =============================================
        // 6. MOUSE INTERACTION
        // =============================================

        const hitTestLayer = (mx, my) => {
            const s = state;
            const indices = Object.keys(
                s.layers
            ).map(Number).sort(
                (a, b) => b - a
            );
            for (const idx of indices) {
                const lyr = s.layers[idx];
                if (!lyr || !lyr.image) continue;
                const lx_img = getLayerX(idx);
                const ly_img = getLayerY(idx);
                const lw = lyr.w * s.displayScale;
                const lh = lyr.h * s.displayScale;
                const lx =
                    s.displayOffsetX +
                    lx_img * s.displayScale;
                const ly =
                    s.displayOffsetY +
                    ly_img * s.displayScale;
                if (
                    mx >= lx && mx <= lx + lw &&
                    my >= ly && my <= ly + lh
                ) {
                    return idx;
                }
            }
            return null;
        };

        canvas.addEventListener("mousedown", (e) => {
            const s = state;
            if (!s.bgImage) return;

            const rect =
                canvas.getBoundingClientRect();
            const mx =
                ((e.clientX - rect.left) /
                    rect.width) * canvas.width;
            const my =
                ((e.clientY - rect.top) /
                    rect.height) * canvas.height;

            const hit = hitTestLayer(mx, my);
            s.selectedLayer = hit;

            if (hit !== null) {
                s.dragging = true;
                s.dragStartX = mx;
                s.dragStartY = my;
                s.dragOffsetStartX = getLayerX(hit);
                s.dragOffsetStartY = getLayerY(hit);
                canvas.style.cursor = "grabbing";
            }

            redrawCanvas();
        });

        canvas.addEventListener("mousemove", (e) => {
            const s = state;
            const rect =
                canvas.getBoundingClientRect();
            const mx =
                ((e.clientX - rect.left) /
                    rect.width) * canvas.width;
            const my =
                ((e.clientY - rect.top) /
                    rect.height) * canvas.height;

            if (s.dragging && s.selectedLayer !== null) {
                const dx =
                    (mx - s.dragStartX) /
                    s.displayScale;
                const dy =
                    (my - s.dragStartY) /
                    s.displayScale;
                const newX = Math.round(
                    s.dragOffsetStartX + dx
                );
                const newY = Math.round(
                    s.dragOffsetStartY + dy
                );
                setLayerXY(
                    s.selectedLayer, newX, newY
                );
                redrawCanvas();
            } else {
                const hit = hitTestLayer(mx, my);
                canvas.style.cursor =
                    hit !== null ? "move" : "default";
            }
        });

        const finalizeDrag = () => {
            const s = state;
            if (!s.dragging) return;
            s.dragging = false;
            canvas.style.cursor = "move";
            redrawCanvas();
            app.graph.setDirtyCanvas(true, true);
        };

        canvas.addEventListener(
            "mouseup", finalizeDrag
        );
        canvas.addEventListener(
            "mouseleave", finalizeDrag
        );

        // Reset button
        resetBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const s = state;
            if (
                s.selectedLayer === null ||
                !s.bgImage
            ) {
                return;
            }
            const lyr = s.layers[s.selectedLayer];
            if (!lyr) return;
            const cx = Math.round(
                (s.bgW - lyr.w) / 2
            );
            const cy = Math.round(
                (s.bgH - lyr.h) / 2
            );
            setLayerXY(s.selectedLayer, cx, cy);
            redrawCanvas();
            app.graph.setDirtyCanvas(true, true);
        });

        // =============================================
        // 7. LIFECYCLE HOOKS
        // =============================================

        node.onExecuted = (message) => {
            const s = state;

            if (!message.bg_preview?.[0]) return;

            const bgImg = new Image();
            bgImg.onload = () => {
                s.bgImage = bgImg;
                s.bgW = message.bg_size[0];
                s.bgH = message.bg_size[1];

                const nodeW = node.size[0] || 450;
                const availW = nodeW - 20;
                const aspect = s.bgH / s.bgW;
                const newH = Math.min(
                    350,
                    Math.round(availW * aspect)
                );

                canvas.width = Math.round(availW);
                canvas.height = newH;
                s.widgetHeight = newH;
                container.style.height =
                    newH + "px";

                // Load layer previews
                const previews =
                    message.layer_previews || [];
                const sizes =
                    message.layer_sizes || [];

                // Map to connected layer indices
                const connectedIndices =
                    getLayerIndices().filter(
                        (i) => isLayerConnected(i)
                    );

                // Clear old layer preview data
                s.layers = {};

                let loaded = 0;
                const total = previews.length;

                if (total === 0) {
                    node._isResizing = true;
                    node.setSize(
                        node.computeSize()
                    );
                    setTimeout(() => {
                        node._isResizing = false;
                    }, 50);
                    redrawCanvas();
                    return;
                }

                for (let i = 0; i < total; i++) {
                    const idx =
                        connectedIndices[i] || (i + 1);
                    const lyrImg = new Image();
                    const capturedIdx = idx;
                    lyrImg.onload = () => {
                        s.layers[capturedIdx] = {
                            image: lyrImg,
                            w: sizes[i]?.[0] ||
                                lyrImg.width,
                            h: sizes[i]?.[1] ||
                                lyrImg.height,
                        };
                        loaded++;
                        if (loaded >= total) {
                            node._isResizing = true;
                            node.setSize(
                                node.computeSize()
                            );
                            setTimeout(() => {
                                node._isResizing =
                                    false;
                            }, 50);
                            redrawCanvas();
                        }
                    };
                    lyrImg.src =
                        "data:image/png;base64," +
                        previews[i];
                }
            };
            bgImg.src =
                "data:image/jpeg;base64," +
                message.bg_preview[0];
        };

        // Connection changes
        const origOnCC = node.onConnectionsChange;
        node.onConnectionsChange = function (
            type, slotIndex, isConnected,
            link, ioSlot
        ) {
            if (origOnCC) {
                origOnCC.apply(this, arguments);
            }
            if (type === 1) {
                setTimeout(updateDynamicInputs, 50);
            }
        };

        // Workflow load
        const origOnCfg = node.onConfigure;
        node.onConfigure = function (config) {
            if (origOnCfg) {
                origOnCfg.apply(this, arguments);
            }
            if (config.inputs) {
                for (const inp of config.inputs) {
                    const m = inp.name.match(
                        /^layer_(\d+)$/
                    );
                    if (m) {
                        addLayerInput(parseInt(m[1]));
                    }
                }
            }
            setTimeout(() => {
                updateDynamicInputs();
                redrawCanvas();
            }, 100);
        };

        // Resize
        const origResize = node.onResize;
        node.onResize = function (size) {
            if (origResize) {
                origResize.apply(this, arguments);
            }
            if (node._isResizing) return;
            node._isResizing = true;
            const newH = Math.max(
                120, size[1] - 200
            );
            if (
                Math.abs(
                    newH - state.widgetHeight
                ) > 5
            ) {
                state.widgetHeight = newH;
                container.style.height =
                    newH + "px";
                redrawCanvas();
            }
            setTimeout(() => {
                node._isResizing = false;
            }, 50);
        };

        const observer = new ResizeObserver(() => {
            redrawCanvas();
        });
        observer.observe(container);

        // =============================================
        // 8. INITIAL SETUP
        // =============================================

        setTimeout(() => {
            // Remove statically-defined layer_1
            // widgets since they get re-added
            // dynamically when layer_1 is connected
            const staticNames = [
                "layer_1_x", "layer_1_y",
                "layer_1_scale", "layer_1_opacity",
                "layer_1_blend",
            ];
            node.widgets = node.widgets.filter(
                (w) => !staticNames.includes(w.name)
            );

            const indices = getLayerIndices();
            if (indices.length === 0) {
                addLayerInput(1);
            }

            updateDynamicInputs();

            container.style.height = "250px";
            node.setSize([
                Math.max(450, node.size[0] || 450),
                node.computeSize()[1],
            ]);
            redrawCanvas();
        }, 100);
    },
});
