/**
 * White To Whatever - Color Picker Widget
 *
 * Adds a native HTML5 color picker popup to the
 * WhiteToWhatever node. Clicking the swatch opens
 * the browser's built-in color picker; the chosen
 * hex value is written back into the node's
 * underlying STRING widget so it serializes with
 * the workflow.
 */

import { app } from "../../scripts/app.js";

console.log("[TrentNodes] White To Whatever extension loading...");

app.registerExtension({
    name: "Trent.WhiteToWhatever",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "WhiteToWhatever") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Locate the underlying STRING widget that
            // holds the hex color. We keep it alive
            // (so the value is serialized) but hide it
            // visually by removing it from the widgets
            // array — the DA3 pattern. The reference
            // below lets us read/write its value after
            // removal.
            const colorWidget = this.widgets?.find(
                (w) => w.name === "color"
            );

            if (!colorWidget) {
                console.warn(
                    "[TrentNodes] WhiteToWhatever:" +
                        " no 'color' widget found"
                );
                return result;
            }

            const getHex = () => {
                const v = colorWidget.value;
                if (typeof v === "string" && /^#?[0-9a-f]{3,6}$/i.test(v)) {
                    return v.startsWith("#") ? v : "#" + v;
                }
                return "#ff3366";
            };

            // Build the DOM swatch + label row
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
            label.textContent = "color";
            label.style.cssText = `
                flex: 0 0 auto;
                opacity: 0.75;
            `;

            // Native color input. Styled to fill the row
            // so the whole thing behaves like one big
            // clickable swatch that opens the OS/browser
            // color picker popup.
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

            // Live update while dragging inside the picker
            picker.addEventListener("input", (e) => {
                const hex = e.target.value;
                colorWidget.value = hex;
                hexLabel.textContent = hex.toUpperCase();
                if (colorWidget.callback) {
                    colorWidget.callback(hex);
                }
                app.graph.setDirtyCanvas(true, true);
            });

            // Final commit when picker closes
            picker.addEventListener("change", (e) => {
                const hex = e.target.value;
                colorWidget.value = hex;
                hexLabel.textContent = hex.toUpperCase();
                if (colorWidget.callback) {
                    colorWidget.callback(hex);
                }
                app.graph.setDirtyCanvas(true, true);
            });

            // Add the DOM widget. Use a fixed compact
            // height so it sits cleanly inside the node.
            const domWidget = this.addDOMWidget(
                "color_picker",
                "customCanvas",
                row,
                {
                    serialize: false,
                }
            );
            domWidget.computeSize = (width) => [width, 32];

            // Hide the original STRING widget WITHOUT
            // removing it from this.widgets. ComfyUI's
            // prompt serializer builds widgets_values
            // from that exact array, so splicing it out
            // drops the required 'color' input from the
            // prompt. Use the SAM3 hiding pattern:
            //  - computeSize returns [0, -4] to eat the
            //    automatic litegraph widget gap
            //  - type includes "converted-widget" so we
            //    can recognise it later
            //  - onDrawForeground override below nulls
            //    the type during draw so litegraph's
            //    widget renderer skips it entirely in
            //    the new v1.38+ frontend
            colorWidget.origType = colorWidget.type;
            colorWidget.origComputeSize = colorWidget.computeSize;
            colorWidget.computeSize = () => [0, -4];
            colorWidget.type = "converted-widget-whitetowhatever";
            colorWidget.hidden = true;

            const origOnDrawForeground = this.onDrawForeground;
            this.onDrawForeground = function () {
                // Temporarily null the hidden widget's
                // type during draw so litegraph skips
                // rendering it, then restore so prompt
                // serialization still sees the real
                // value.
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

            // Re-sync picker when the underlying value
            // is changed externally (workflow load,
            // undo/redo, API calls).
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

            // After workflow load, sync the swatch from
            // the restored widget value.
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

            // Shrink node to a sensible default size
            setTimeout(() => {
                this.setSize(this.computeSize());
            }, 50);

            return result;
        };
    },
});
