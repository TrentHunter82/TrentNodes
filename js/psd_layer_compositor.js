/**
 * PSD Layer Compositor - Text Color Picker Widget
 *
 * Adds a native HTML5 color picker popup for the
 * text_color string widget on PSDLayerCompositor.
 * Mirrors the pattern used in white_to_whatever.js.
 */

import { app } from "../../scripts/app.js";

console.log("[TrentNodes] PSD Layer Compositor extension loading...");

app.registerExtension({
    name: "Trent.PSDLayerCompositor",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PSDLayerCompositor") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const colorWidget = this.widgets?.find(
                (w) => w.name === "text_color"
            );

            if (!colorWidget) {
                console.warn(
                    "[TrentNodes] PSDLayerCompositor:" +
                        " no 'text_color' widget found"
                );
                return result;
            }

            const getHex = () => {
                const v = colorWidget.value;
                if (typeof v === "string" && /^#?[0-9a-f]{3,6}$/i.test(v)) {
                    return v.startsWith("#") ? v : "#" + v;
                }
                return "#ffffff";
            };

            const row = document.createElement("div");
            row.style.cssText = `
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 4px 6px;
                box-sizing: border-box;
                width: 100%;
                height: 100%;
                font-family: sans-serif;
                font-size: 12px;
                color: #e0e0e0;
            `;

            const label = document.createElement("span");
            label.textContent = "text_color";
            label.style.cssText = `
                flex: 0 0 auto;
                opacity: 0.75;
            `;

            const picker = document.createElement("input");
            picker.type = "color";
            picker.value = getHex();
            picker.title = "Click to open color picker";
            picker.style.cssText = `
                flex: 1 1 auto;
                height: 22px;
                min-width: 40px;
                border: 1px solid #333;
                border-radius: 3px;
                padding: 0;
                background: transparent;
                cursor: pointer;
            `;

            const hexLabel = document.createElement("span");
            hexLabel.textContent = getHex().toUpperCase();
            hexLabel.style.cssText = `
                flex: 0 0 auto;
                font-family: monospace;
                font-size: 11px;
                opacity: 0.85;
                min-width: 64px;
                text-align: right;
            `;

            row.appendChild(label);
            row.appendChild(picker);
            row.appendChild(hexLabel);

            picker.addEventListener("input", (e) => {
                const hex = e.target.value;
                colorWidget.value = hex;
                hexLabel.textContent = hex.toUpperCase();
                if (colorWidget.callback) {
                    colorWidget.callback(hex);
                }
                app.graph.setDirtyCanvas(true, true);
            });

            picker.addEventListener("change", (e) => {
                const hex = e.target.value;
                colorWidget.value = hex;
                hexLabel.textContent = hex.toUpperCase();
                if (colorWidget.callback) {
                    colorWidget.callback(hex);
                }
                app.graph.setDirtyCanvas(true, true);
            });

            const domWidget = this.addDOMWidget(
                "text_color_picker",
                "customCanvas",
                row,
                {
                    serialize: false,
                }
            );
            domWidget.computeSize = (width) => [width, 32];

            colorWidget.origType = colorWidget.type;
            colorWidget.origComputeSize = colorWidget.computeSize;
            colorWidget.computeSize = () => [0, -4];
            colorWidget.type = "converted-widget-psdcompositor-text";
            colorWidget.hidden = true;

            const origOnDrawForeground = this.onDrawForeground;
            this.onDrawForeground = function () {
                const hidden = this.widgets.filter(
                    (w) =>
                        typeof w.type === "string" &&
                        w.type.includes("converted-widget")
                );
                const savedTypes = hidden.map((w) => w.type);
                hidden.forEach((w) => (w.type = null));

                const r = origOnDrawForeground
                    ? origOnDrawForeground.apply(this, arguments)
                    : undefined;

                hidden.forEach(
                    (w, i) => (w.type = savedTypes[i])
                );
                return r;
            };

            const origCallback = colorWidget.callback;
            colorWidget.callback = (value) => {
                if (origCallback) origCallback(value);
                if (typeof value === "string") {
                    const hex = value.startsWith("#")
                        ? value
                        : "#" + value;
                    picker.value = hex;
                    hexLabel.textContent = hex.toUpperCase();
                }
            };

            const origOnConfigure = this.onConfigure;
            this.onConfigure = function (config) {
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
                setTimeout(() => {
                    const hex = getHex();
                    picker.value = hex;
                    hexLabel.textContent = hex.toUpperCase();
                }, 50);
            };

            setTimeout(() => {
                this.setSize(this.computeSize());
            }, 50);

            return result;
        };
    },
});
