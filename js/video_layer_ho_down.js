/**
 * Video Layer Ho Down - Interactive Canvas Extension
 *
 * Multi-layer compositing with drag-to-position canvas.
 * Dynamic layer inputs (connect one, next appears).
 * Per-layer widgets shown only when connected.
 */

import { app } from "/scripts/app.js";

const MAX_LAYERS = 5;
const BLEND_MODES = [
    "normal", "multiply", "screen", "overlay", "add",
];

/**
 * Draw checkerboard pattern (transparency indicator)
 */
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
        // 1. LAYER STATE
        // =============================================

        const layerState = {
            // Canvas
            canvas: null,
            ctx: null,
            container: null,
            widgetHeight: 300,
            // Display transform
            displayScale: 1.0,
            displayOffsetX: 0,
            displayOffsetY: 0,
            // Background
            bgImage: null,
            bgW: 0,
            bgH: 0,
            // Layers (indexed 1-5)
            layers: {},
            // Selection / drag
            selectedLayer: null,
            dragging: false,
            dragStartX: 0,
            dragStartY: 0,
            dragOffsetStartX: 0,
            dragOffsetStartY: 0,
        };
        node.layerState = layerState;

        // Hidden widget refs (DA3 pattern)
        // Stores { 1: {x: widget, y: widget}, ... }
        const hiddenRefs = {};

        // =============================================
        // 2. WIDGET HELPERS
        // =============================================

        const findWidget = (name) =>
            node.widgets?.find((w) => w.name === name);

        /**
         * Hide layer_1_x and layer_1_y (DA3 pattern)
         */
        const hideInitialXY = () => {
            const xw = findWidget("layer_1_x");
            const yw = findWidget("layer_1_y");
            if (xw || yw) {
                hiddenRefs[1] = {
                    x: xw || null,
                    y: yw || null,
                };
                const hideSet = new Set(
                    [xw, yw].filter(Boolean)
                );
                node.widgets = node.widgets.filter(
                    (w) => !hideSet.has(w)
                );
            }
        };

        /**
         * Ensure per-layer visible widgets exist
         */
        const ensureLayerWidgets = (idx) => {
            const scaleName = `layer_${idx}_scale`;
            const opacityName = `layer_${idx}_opacity`;
            const blendName = `layer_${idx}_blend`;
            const xName = `layer_${idx}_x`;
            const yName = `layer_${idx}_y`;

            // Create x/y hidden widgets if needed
            if (!hiddenRefs[idx]) {
                hiddenRefs[idx] = { x: null, y: null };
            }
            if (!hiddenRefs[idx].x) {
                const xw = node.addWidget(
                    "number", xName, 0,
                    () => {},
                    { min: -8192, max: 8192, step: 1 }
                );
                hiddenRefs[idx].x = xw;
                // Immediately hide
                node.widgets = node.widgets.filter(
                    (w) => w !== xw
                );
            }
            if (!hiddenRefs[idx].y) {
                const yw = node.addWidget(
                    "number", yName, 0,
                    () => {},
                    { min: -8192, max: 8192, step: 1 }
                );
                hiddenRefs[idx].y = yw;
                node.widgets = node.widgets.filter(
                    (w) => w !== yw
                );
            }

            // Create visible widgets if missing
            if (!findWidget(scaleName)) {
                node.addWidget(
                    "slider", scaleName, 1.0,
                    () => {},
                    {
                        min: 0.01, max: 10.0,
                        step: 0.01, precision: 2,
                    }
                );
            }
            if (!findWidget(opacityName)) {
                node.addWidget(
                    "slider", opacityName, 1.0,
                    () => {},
                    {
                        min: 0.0, max: 1.0,
                        step: 0.01, precision: 2,
                    }
                );
            }
            if (!findWidget(blendName)) {
                node.addWidget(
                    "combo", blendName, "normal",
                    () => {},
                    { values: BLEND_MODES }
                );
            }
        };

        /**
         * Remove per-layer widgets
         */
        const removeLayerWidgets = (idx) => {
            const names = [
                `layer_${idx}_scale`,
                `layer_${idx}_opacity`,
                `layer_${idx}_blend`,
            ];
            node.widgets = node.widgets.filter(
                (w) => !names.includes(w.name)
            );
            // Clear hidden refs
            if (hiddenRefs[idx]) {
                // Remove hidden x/y widgets too
                const xw = hiddenRefs[idx].x;
                const yw = hiddenRefs[idx].y;
                if (xw) {
                    node.widgets = node.widgets.filter(
                        (w) => w !== xw
                    );
                }
                if (yw) {
                    node.widgets = node.widgets.filter(
                        (w) => w !== yw
                    );
                }
                delete hiddenRefs[idx];
            }
        };

        /**
         * Sort layer widgets in order
         */
        const sortLayerWidgets = () => {
            if (!node.widgets) return;
            const layerWidgets = [];
            const otherWidgets = [];
            for (const w of node.widgets) {
                if (w.name.match(/^layer_\d+_/)) {
                    layerWidgets.push(w);
                } else {
                    otherWidgets.push(w);
                }
            }
            layerWidgets.sort((a, b) => {
                const aNum = parseInt(
                    a.name.match(/^layer_(\d+)_/)[1]
                );
                const bNum = parseInt(
                    b.name.match(/^layer_(\d+)_/)[1]
                );
                if (aNum !== bNum) return aNum - bNum;
                // Within same layer: scale, opacity,
                // blend
                const order = {
                    scale: 0, opacity: 1, blend: 2,
                };
                const aType = a.name.split("_").pop();
                const bType = b.name.split("_").pop();
                return (
                    (order[aType] ?? 99) -
                    (order[bType] ?? 99)
                );
            });
            node.widgets.length = 0;
            node.widgets.push(
                ...otherWidgets, ...layerWidgets
            );
        };

        // =============================================
        // 3. DYNAMIC INPUTS (WanVace pattern)
        // =============================================

        const getLayerIndices = () => {
            const indices = [];
            for (const input of node.inputs || []) {
                const m = input.name.match(
                    /^layer_(\d+)$/
                );
                if (m) indices.push(parseInt(m[1]));
            }
            return indices.sort((a, b) => a - b);
        };

        const isLayerConnected = (idx) => {
            const input = node.inputs?.find(
                (i) => i.name === `layer_${idx}`
            );
            return input && input.link !== null;
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
            const inputIdx = node.inputs?.findIndex(
                (i) => i.name === name
            );
            if (inputIdx >= 0) {
                const input = node.inputs[inputIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(inputIdx);
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

            // If bottom-most is connected, add next
            if (
                isLayerConnected(maxIdx) &&
                maxIdx < MAX_LAYERS
            ) {
                addLayerInput(maxIdx + 1);
            } else {
                // Remove excess empty from bottom
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
            const updatedIndices = getLayerIndices();
            for (const idx of updatedIndices) {
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
        layerState.container = container;

        // Info bar
        const infoBar = document.createElement("div");
        infoBar.style.cssText = `
            position: absolute;
            top: 5px; left: 5px; right: 5px;
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
            padding: 4px 8px;
            background: rgba(0,0,0,0.7);
            color: #0f0;
            border-radius: 3px;
            font-size: 11px;
            font-family: monospace;
            pointer-events: none;
        `;
        coordsDisplay.textContent = "No layer selected";
        infoBar.appendChild(coordsDisplay);

        const resetBtn =
            document.createElement("button");
        resetBtn.textContent = "Center";
        resetBtn.style.cssText = `
            padding: 3px 8px;
            background: rgba(255,255,255,0.15);
            color: #ccc;
            border: 1px solid #444;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
            pointer-events: auto;
        `;
        infoBar.appendChild(resetBtn);

        // Canvas
        const canvas = document.createElement("canvas");
        canvas.width = 400;
        canvas.height = 300;
        canvas.style.cssText = `
            display: block;
            max-width: 100%;
            cursor: default;
            margin: 0 auto;
        `;
        container.appendChild(canvas);
        const ctx = canvas.getContext("2d");
        layerState.canvas = canvas;
        layerState.ctx = ctx;

        // DOM widget
        const canvasWidget = node.addDOMWidget(
            "layer_canvas", "customCanvas", container
        );
        canvasWidget.computeSize = (width) => {
            return [
                width, layerState.widgetHeight,
            ];
        };
        canvasWidget.serializeValue = () => undefined;

        // =============================================
        // 5. CANVAS RENDERING
        // =============================================

        const redrawCanvas = () => {
            const s = layerState;
            const { canvas: c, ctx: cx } = s;

            cx.clearRect(0, 0, c.width, c.height);

            if (!s.bgImage) {
                cx.fillStyle = "#1a1a1a";
                cx.fillRect(0, 0, c.width, c.height);
                cx.fillStyle = "#555";
                cx.font = "13px sans-serif";
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

            // Checkerboard
            drawCheckerboard(
                cx, s.displayOffsetX,
                s.displayOffsetY, dw, dh, 8
            );

            // Background
            cx.drawImage(
                s.bgImage, s.displayOffsetX,
                s.displayOffsetY, dw, dh
            );

            // Draw each layer
            const layerIndices = Object.keys(
                s.layers
            ).map(Number).sort((a, b) => a - b);

            for (const idx of layerIndices) {
                const lyr = s.layers[idx];
                if (!lyr || !lyr.image) continue;

                const lw = lyr.w * s.displayScale;
                const lh = lyr.h * s.displayScale;
                const lx =
                    s.displayOffsetX +
                    lyr.x * s.displayScale;
                const ly =
                    s.displayOffsetY +
                    lyr.y * s.displayScale;

                cx.drawImage(
                    lyr.image, lx, ly, lw, lh
                );

                // Border
                const selected =
                    s.selectedLayer === idx;
                cx.strokeStyle = selected
                    ? "#ff0" : "rgba(255,255,255,0.4)";
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
                cx.font = "bold 11px monospace";
                cx.textAlign = "left";
                cx.fillText(
                    `L${idx}`,
                    lx + 4, ly + 13
                );

                // Crosshair guides when dragging
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

            // Update coords display
            if (s.selectedLayer !== null) {
                const sl = s.layers[s.selectedLayer];
                if (sl) {
                    coordsDisplay.textContent =
                        `L${s.selectedLayer}:` +
                        ` x=${sl.x}, y=${sl.y}`;
                }
            } else {
                coordsDisplay.textContent =
                    "No layer selected";
            }
        };
        node.redrawLayerCanvas = redrawCanvas;

        // =============================================
        // 6. MOUSE INTERACTION
        // =============================================

        /**
         * Hit-test layers top-down
         */
        const hitTestLayer = (mx, my) => {
            const s = layerState;
            const indices = Object.keys(
                s.layers
            ).map(Number).sort(
                (a, b) => b - a
            ); // top layer first
            for (const idx of indices) {
                const lyr = s.layers[idx];
                if (!lyr || !lyr.image) continue;
                const lw = lyr.w * s.displayScale;
                const lh = lyr.h * s.displayScale;
                const lx =
                    s.displayOffsetX +
                    lyr.x * s.displayScale;
                const ly =
                    s.displayOffsetY +
                    lyr.y * s.displayScale;
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
            const s = layerState;
            if (!s.bgImage) return;

            const rect =
                canvas.getBoundingClientRect();
            const mx =
                ((e.clientX - rect.left) /
                    rect.width) *
                canvas.width;
            const my =
                ((e.clientY - rect.top) /
                    rect.height) *
                canvas.height;

            const hit = hitTestLayer(mx, my);
            s.selectedLayer = hit;

            if (hit !== null) {
                const lyr = s.layers[hit];
                s.dragging = true;
                s.dragStartX = mx;
                s.dragStartY = my;
                s.dragOffsetStartX = lyr.x;
                s.dragOffsetStartY = lyr.y;
                canvas.style.cursor = "grabbing";
            }

            redrawCanvas();
        });

        canvas.addEventListener("mousemove", (e) => {
            const s = layerState;

            const rect =
                canvas.getBoundingClientRect();
            const mx =
                ((e.clientX - rect.left) /
                    rect.width) *
                canvas.width;
            const my =
                ((e.clientY - rect.top) /
                    rect.height) *
                canvas.height;

            if (s.dragging && s.selectedLayer !== null) {
                const dx =
                    (mx - s.dragStartX) /
                    s.displayScale;
                const dy =
                    (my - s.dragStartY) /
                    s.displayScale;
                const lyr = s.layers[s.selectedLayer];
                lyr.x = Math.round(
                    s.dragOffsetStartX + dx
                );
                lyr.y = Math.round(
                    s.dragOffsetStartY + dy
                );
                redrawCanvas();
            } else {
                // Hover cursor
                const hit = hitTestLayer(mx, my);
                canvas.style.cursor =
                    hit !== null ? "grab" : "default";
            }
        });

        const finalizeDrag = () => {
            const s = layerState;
            if (!s.dragging) return;
            s.dragging = false;
            canvas.style.cursor = "grab";

            if (s.selectedLayer !== null) {
                const lyr =
                    s.layers[s.selectedLayer];
                const idx = s.selectedLayer;
                const xRef = hiddenRefs[idx]?.x;
                const yRef = hiddenRefs[idx]?.y;
                if (xRef) {
                    xRef.value = lyr.x;
                    if (xRef.callback) {
                        xRef.callback(lyr.x);
                    }
                }
                if (yRef) {
                    yRef.value = lyr.y;
                    if (yRef.callback) {
                        yRef.callback(lyr.y);
                    }
                }
            }
            redrawCanvas();
            app.graph.setDirtyCanvas(true, true);
        };

        canvas.addEventListener("mouseup", finalizeDrag);
        canvas.addEventListener(
            "mouseleave", finalizeDrag
        );

        // Reset button
        resetBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            const s = layerState;
            if (
                s.selectedLayer === null ||
                !s.bgImage
            ) {
                return;
            }
            const lyr = s.layers[s.selectedLayer];
            if (!lyr) return;
            lyr.x = Math.round(
                (s.bgW - lyr.w) / 2
            );
            lyr.y = Math.round(
                (s.bgH - lyr.h) / 2
            );
            const idx = s.selectedLayer;
            if (hiddenRefs[idx]?.x) {
                hiddenRefs[idx].x.value = lyr.x;
            }
            if (hiddenRefs[idx]?.y) {
                hiddenRefs[idx].y.value = lyr.y;
            }
            redrawCanvas();
            app.graph.setDirtyCanvas(true, true);
        });

        // =============================================
        // 7. LIFECYCLE HOOKS
        // =============================================

        node.onExecuted = (message) => {
            const s = layerState;

            if (message.bg_preview?.[0]) {
                const bgImg = new Image();
                bgImg.onload = () => {
                    s.bgImage = bgImg;
                    s.bgW = message.bg_size[0];
                    s.bgH = message.bg_size[1];

                    const nodeW =
                        node.size[0] || 400;
                    const availW = nodeW - 20;
                    const aspect = s.bgH / s.bgW;
                    const newH = Math.min(
                        400,
                        Math.round(availW * aspect)
                    );

                    canvas.width =
                        Math.round(availW);
                    canvas.height = newH;
                    s.widgetHeight = newH;
                    container.style.height =
                        newH + "px";

                    // Load layers
                    const previews =
                        message.layer_previews || [];
                    const sizes =
                        message.layer_sizes || [];
                    const positions =
                        message.layer_positions || [];

                    let loaded = 0;
                    const total = previews.length;

                    if (total === 0) {
                        s.layers = {};
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

                    // Map connected layer indices
                    const connectedIndices =
                        getLayerIndices().filter(
                            (i) => isLayerConnected(i)
                        );

                    for (
                        let i = 0;
                        i < total;
                        i++
                    ) {
                        const idx =
                            connectedIndices[i] || (i + 1);
                        const lyrImg = new Image();
                        const lyrIdx = idx;
                        lyrImg.onload = () => {
                            if (!s.layers[lyrIdx]) {
                                s.layers[lyrIdx] = {};
                            }
                            const sl = s.layers[lyrIdx];
                            sl.image = lyrImg;
                            sl.w = sizes[i]?.[0] ||
                                lyrImg.width;
                            sl.h = sizes[i]?.[1] ||
                                lyrImg.height;
                            // Use position from
                            // backend or keep
                            // existing
                            if (sl.x === undefined) {
                                sl.x =
                                    positions[i]?.[0] ||
                                    0;
                            }
                            if (sl.y === undefined) {
                                sl.y =
                                    positions[i]?.[1] ||
                                    0;
                            }
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
            }
        };

        // Connection changes
        const origOnConnectionsChange =
            node.onConnectionsChange;
        node.onConnectionsChange = function (
            type, slotIndex, isConnected,
            link, ioSlot
        ) {
            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(
                    this, arguments
                );
            }
            if (type === 1) {
                setTimeout(updateDynamicInputs, 50);
            }
        };

        // Workflow load
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (config) {
            if (origOnConfigure) {
                origOnConfigure.apply(
                    this, arguments
                );
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
                // Restore layer positions from
                // hidden widgets
                const s = layerState;
                for (const [k, refs] of
                    Object.entries(hiddenRefs)
                ) {
                    const idx = parseInt(k);
                    if (!s.layers[idx]) {
                        s.layers[idx] = {};
                    }
                    if (refs.x) {
                        s.layers[idx].x = refs.x.value;
                    }
                    if (refs.y) {
                        s.layers[idx].y = refs.y.value;
                    }
                }
                redrawCanvas();
            }, 100);
        };

        // Resize
        const origOnResize = node.onResize;
        node.onResize = function (size) {
            if (origOnResize) {
                origOnResize.apply(this, arguments);
            }
            if (node._isResizing) return;
            node._isResizing = true;
            const newH = Math.max(
                150, size[1] - 200
            );
            if (
                Math.abs(
                    newH - layerState.widgetHeight
                ) > 5
            ) {
                layerState.widgetHeight = newH;
                container.style.height =
                    newH + "px";
                redrawCanvas();
            }
            setTimeout(() => {
                node._isResizing = false;
            }, 50);
        };

        // ResizeObserver
        const observer = new ResizeObserver(() => {
            redrawCanvas();
        });
        observer.observe(container);

        // =============================================
        // 8. INITIAL SETUP
        // =============================================

        setTimeout(() => {
            // Hide layer_1 x/y widgets
            hideInitialXY();

            // Also hide the static layer_1
            // visible widgets if layer_1 not
            // connected
            const indices = getLayerIndices();
            if (indices.length === 0) {
                addLayerInput(1);
            }

            updateDynamicInputs();

            const nodeW = Math.max(
                400, node.size[0] || 400
            );
            container.style.height = "300px";
            node.setSize([nodeW, 550]);
            redrawCanvas();
        }, 100);
    },
});
