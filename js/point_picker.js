/**
 * Point Picker Widget for TrentNodes
 * Click-to-pick point coordinates on an image
 */

import { app } from "../../scripts/app.js";

console.log("[TrentNodes] Point Picker extension loading...");

app.registerExtension({
    name: "Trent.PointPicker",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PointPreview") {
            return;
        }

        console.log("[TrentNodes] Registering PointPreview node");

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);

            // Create canvas container
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

            // Create info bar
            const infoBar = document.createElement("div");
            infoBar.style.cssText = `
                position: absolute;
                top: 5px;
                left: 5px;
                right: 5px;
                z-index: 10;
                display: flex;
                justify-content: space-between;
                align-items: center;
            `;
            container.appendChild(infoBar);

            // Create coordinates display
            const coordsDisplay = document.createElement("div");
            coordsDisplay.style.cssText = `
                padding: 5px 10px;
                background: rgba(0,0,0,0.7);
                color: #0f0;
                border-radius: 3px;
                font-size: 12px;
                font-family: monospace;
            `;
            coordsDisplay.textContent = "Click to set point";
            infoBar.appendChild(coordsDisplay);

            // Create canvas
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
            canvas.title = "Click to set the point";
            container.appendChild(canvas);

            const ctx = canvas.getContext("2d");

            // Store state
            this.pointPicker = {
                canvas: canvas,
                ctx: ctx,
                container: container,
                image: null,
                imageWidth: 0,
                imageHeight: 0,
                pointX: 0,
                pointY: 0,
                coordsDisplay: coordsDisplay,
                widgetHeight: 300
            };

            // Add DOM widget
            const widget = this.addDOMWidget(
                "point_canvas",
                "customCanvas",
                container
            );
            this.pointPicker.domWidget = widget;

            widget.computeSize = (width) => {
                return [width, this.pointPicker.widgetHeight];
            };

            // Find x and y widgets
            const getXWidget = () => this.widgets.find(w => w.name === "x");
            const getYWidget = () => this.widgets.find(w => w.name === "y");

            // Initialize point from current widget values
            const initPoint = () => {
                const xWidget = getXWidget();
                const yWidget = getYWidget();
                if (xWidget && yWidget) {
                    this.pointPicker.pointX = xWidget.value || 0;
                    this.pointPicker.pointY = yWidget.value || 0;
                    this.updateCoordsDisplay();
                }
            };

            // Update coordinates display
            this.updateCoordsDisplay = () => {
                const { pointX, pointY, coordsDisplay } = this.pointPicker;
                coordsDisplay.textContent = `Point: (${pointX}, ${pointY})`;
            };

            // Redraw canvas
            this.redrawPointCanvas = () => {
                const { canvas, ctx, image, pointX, pointY } = this.pointPicker;

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (image) {
                    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                } else {
                    // Placeholder
                    ctx.fillStyle = "#222";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#666";
                    ctx.font = "14px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(
                        "Connect image to preview",
                        canvas.width / 2,
                        canvas.height / 2 - 10
                    );
                    ctx.fillText(
                        "Click to set tracking point",
                        canvas.width / 2,
                        canvas.height / 2 + 15
                    );
                }

                // Draw crosshair at current point
                const scaleX = canvas.width / (this.pointPicker.imageWidth || 1);
                const scaleY = canvas.height / (this.pointPicker.imageHeight || 1);
                const displayX = pointX * scaleX;
                const displayY = pointY * scaleY;

                // Only draw if we have valid coordinates
                if (this.pointPicker.imageWidth > 0) {
                    const armLength = 15;
                    const thickness = 2;

                    // White outline for visibility
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = thickness + 2;

                    // Horizontal line
                    ctx.beginPath();
                    ctx.moveTo(displayX - armLength, displayY);
                    ctx.lineTo(displayX + armLength, displayY);
                    ctx.stroke();

                    // Vertical line
                    ctx.beginPath();
                    ctx.moveTo(displayX, displayY - armLength);
                    ctx.lineTo(displayX, displayY + armLength);
                    ctx.stroke();

                    // Red crosshair
                    ctx.strokeStyle = "#f00";
                    ctx.lineWidth = thickness;

                    ctx.beginPath();
                    ctx.moveTo(displayX - armLength, displayY);
                    ctx.lineTo(displayX + armLength, displayY);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(displayX, displayY - armLength);
                    ctx.lineTo(displayX, displayY + armLength);
                    ctx.stroke();

                    // Green center dot
                    ctx.fillStyle = "#0f0";
                    ctx.beginPath();
                    ctx.arc(displayX, displayY, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            };

            // Canvas click handler
            canvas.addEventListener("click", (e) => {
                if (!this.pointPicker.imageWidth) return;

                const rect = canvas.getBoundingClientRect();

                // Map display coordinates to image coordinates
                const displayX = (e.clientX - rect.left);
                const displayY = (e.clientY - rect.top);

                // Scale to actual canvas size
                const canvasX = (displayX / rect.width) * canvas.width;
                const canvasY = (displayY / rect.height) * canvas.height;

                // Map to original image coordinates
                const scaleX = this.pointPicker.imageWidth / canvas.width;
                const scaleY = this.pointPicker.imageHeight / canvas.height;

                const imageX = Math.round(canvasX * scaleX);
                const imageY = Math.round(canvasY * scaleY);

                // Clamp to image bounds
                this.pointPicker.pointX = Math.max(
                    0,
                    Math.min(this.pointPicker.imageWidth - 1, imageX)
                );
                this.pointPicker.pointY = Math.max(
                    0,
                    Math.min(this.pointPicker.imageHeight - 1, imageY)
                );

                // Update widgets
                const xWidget = getXWidget();
                const yWidget = getYWidget();

                if (xWidget) xWidget.value = this.pointPicker.pointX;
                if (yWidget) yWidget.value = this.pointPicker.pointY;

                this.updateCoordsDisplay();
                this.redrawPointCanvas();

                // Trigger widget callbacks if they exist
                if (xWidget && xWidget.callback) {
                    xWidget.callback(this.pointPicker.pointX);
                }
                if (yWidget && yWidget.callback) {
                    yWidget.callback(this.pointPicker.pointY);
                }

                // Mark graph as dirty
                app.graph.setDirtyCanvas(true, true);
            });

            // Widget value change callbacks
            const setupWidgetCallbacks = () => {
                const xWidget = getXWidget();
                const yWidget = getYWidget();

                if (xWidget) {
                    const origCallback = xWidget.callback;
                    xWidget.callback = (value) => {
                        if (origCallback) origCallback(value);
                        this.pointPicker.pointX = value;
                        this.updateCoordsDisplay();
                        this.redrawPointCanvas();
                    };
                }

                if (yWidget) {
                    const origCallback = yWidget.callback;
                    yWidget.callback = (value) => {
                        if (origCallback) origCallback(value);
                        this.pointPicker.pointY = value;
                        this.updateCoordsDisplay();
                        this.redrawPointCanvas();
                    };
                }
            };

            // Handle executed - image from Python
            this.onExecuted = (message) => {
                console.log("[TrentNodes] PointPreview onExecuted:", message);
                if (message.preview_image && message.preview_image[0]) {
                    console.log("[TrentNodes] Loading preview image...");
                    const img = new Image();
                    img.onerror = (e) => {
                        console.error("[TrentNodes] Failed to load image:", e);
                    };
                    img.onload = () => {
                        console.log("[TrentNodes] Image loaded:", img.width, "x", img.height);
                        this.pointPicker.image = img;
                        this.pointPicker.imageWidth = img.width;
                        this.pointPicker.imageHeight = img.height;

                        // Resize canvas to match aspect ratio
                        const nodeWidth = this.size[0] || 400;
                        const availableWidth = nodeWidth - 20;
                        const aspectRatio = img.height / img.width;
                        const newWidgetHeight = Math.min(
                            400,
                            Math.round(availableWidth * aspectRatio)
                        );

                        canvas.width = img.width;
                        canvas.height = img.height;

                        this._isResizing = true;
                        this.pointPicker.widgetHeight = newWidgetHeight;
                        container.style.height = newWidgetHeight + "px";

                        // Recalculate node size
                        this.setSize(this.computeSize());
                        setTimeout(() => { this._isResizing = false; }, 50);

                        this.redrawPointCanvas();
                    };
                    img.src = "data:image/jpeg;base64," + message.preview_image[0];
                } else {
                    console.log("[TrentNodes] No preview_image in message, keys:", Object.keys(message));
                }
            };

            // Handle node resize
            const originalOnResize = this.onResize;
            this.onResize = function(size) {
                if (originalOnResize) {
                    originalOnResize.apply(this, arguments);
                }

                if (this._isResizing) return;
                this._isResizing = true;

                const newWidgetHeight = Math.max(150, size[1] - 120);

                if (Math.abs(newWidgetHeight - this.pointPicker.widgetHeight) > 5) {
                    this.pointPicker.widgetHeight = newWidgetHeight;
                    container.style.height = newWidgetHeight + "px";
                    this.redrawPointCanvas();
                }

                setTimeout(() => { this._isResizing = false; }, 50);
            };

            // Handle configuration load
            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function(config) {
                if (originalOnConfigure) {
                    originalOnConfigure.apply(this, arguments);
                }
                setTimeout(() => {
                    initPoint();
                    setupWidgetCallbacks();
                    this.redrawPointCanvas();
                }, 100);
            };

            // Initial setup
            setTimeout(() => {
                initPoint();
                setupWidgetCallbacks();
                this.redrawPointCanvas();

                const nodeWidth = Math.max(350, this.size[0] || 350);
                container.style.height = "300px";
                this.setSize([nodeWidth, 420]);
            }, 100);

            return result;
        };
    }
});
