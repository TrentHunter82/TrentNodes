import { app } from "/scripts/app.js";

/**
 * Wan Vace Keyframe Builder - Dynamic Input Extension
 * 
 * Automatically adds new image inputs when you connect to the last available one.
 * Each image input comes with a corresponding frame position slider widget.
 * Sliders are only shown when their corresponding image is connected.
 */
app.registerExtension({
    name: "WanVace.KeyframeBuilder",
    
    async nodeCreated(node) {
        // Only apply to our node type
        if (node.constructor.comfyClass !== "WanVaceKeyframeBuilder") {
            return;
        }
        
        /**
         * Get all image input indices currently on the node
         */
        const getImageInputIndices = () => {
            const indices = [];
            for (const input of node.inputs || []) {
                const match = input.name.match(/^image_(\d+)$/);
                if (match) {
                    indices.push(parseInt(match[1]));
                }
            }
            return indices.sort((a, b) => a - b);
        };
        
        /**
         * Check if an image input at given index is connected
         */
        const isInputConnected = (index) => {
            const input = node.inputs?.find(i => i.name === `image_${index}`);
            return input && input.link !== null;
        };
        
        /**
         * Get the frame_count value
         */
        const getFrameCount = () => {
            const widget = node.widgets?.find(w => w.name === "frame_count");
            return widget?.value || 256;
        };
        
        /**
         * Find or create a frame slider widget for given index
         */
        const ensureFrameWidget = (index) => {
            const frameName = `image_${index}_frame`;
            let widget = node.widgets?.find(w => w.name === frameName);
            
            if (!widget) {
                const maxFrame = getFrameCount();
                widget = node.addWidget(
                    "slider",
                    frameName,
                    index,  // default value = index
                    (value) => {},
                    {
                        min: 1,
                        max: maxFrame,
                        step: 1,
                        precision: 0,
                    }
                );
            }
            
            return widget;
        };
        
        /**
         * Add a new image input slot (without creating widget yet)
         */
        const addImageInput = (index) => {
            const inputName = `image_${index}`;
            
            // Check if already exists
            if (node.inputs?.find(i => i.name === inputName)) {
                return false;
            }
            
            // Add the image input slot
            node.addInput(inputName, "IMAGE");
            return true;
        };
        
        /**
         * Remove an image input slot and its frame widget
         */
        const removeImageInput = (index) => {
            const inputName = `image_${index}`;
            const frameName = `image_${index}_frame`;
            
            // Find and remove the input
            const inputIdx = node.inputs?.findIndex(i => i.name === inputName);
            if (inputIdx >= 0) {
                const input = node.inputs[inputIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(inputIdx);
            }
            
            // Find and remove the widget
            const widgetIdx = node.widgets?.findIndex(w => w.name === frameName);
            if (widgetIdx >= 0) {
                node.widgets.splice(widgetIdx, 1);
            }
        };
        
        /**
         * Sort frame widgets so they always appear in numerical order
         */
        const sortFrameWidgets = () => {
            if (!node.widgets) return;

            // Separate frame widgets from other widgets
            const frameWidgets = [];
            const otherWidgets = [];

            for (const widget of node.widgets) {
                if (widget.name.match(/^image_\d+_frame$/)) {
                    frameWidgets.push(widget);
                } else {
                    otherWidgets.push(widget);
                }
            }

            // Sort frame widgets by their index number
            frameWidgets.sort((a, b) => {
                const aNum = parseInt(a.name.match(/^image_(\d+)_frame$/)[1]);
                const bNum = parseInt(b.name.match(/^image_(\d+)_frame$/)[1]);
                return aNum - bNum;
            });

            // Rebuild widgets array: other widgets first, then sorted frame widgets
            node.widgets.length = 0;
            node.widgets.push(...otherWidgets, ...frameWidgets);
        };

        /**
         * Update widget visibility - show only for connected inputs
         */
        const updateWidgetVisibility = () => {
            const indices = getImageInputIndices();

            for (const idx of indices) {
                const frameName = `image_${idx}_frame`;
                const widget = node.widgets?.find(w => w.name === frameName);
                const connected = isInputConnected(idx);

                if (connected) {
                    // Ensure widget exists and is visible
                    if (!widget) {
                        ensureFrameWidget(idx);
                    }
                } else {
                    // Remove widget if exists (hide it)
                    if (widget) {
                        const widgetIdx = node.widgets.indexOf(widget);
                        if (widgetIdx >= 0) {
                            node.widgets.splice(widgetIdx, 1);
                        }
                    }
                }
            }

            // Always sort after updating visibility
            sortFrameWidgets();
        };
        
        /**
         * Main function to update dynamic inputs based on connection state
         */
        const updateDynamicInputs = () => {
            const indices = getImageInputIndices();
            
            if (indices.length === 0) {
                addImageInput(1);
                return;
            }
            
            // Find connected and unconnected inputs
            const connectedIndices = indices.filter(i => isInputConnected(i));
            const unconnectedIndices = indices.filter(i => !isInputConnected(i));
            
            const maxIndex = Math.max(...indices);
            
            // If the highest input is connected, add a new one
            if (isInputConnected(maxIndex)) {
                addImageInput(maxIndex + 1);
            }
            
            // Remove extra unconnected inputs (keep only one empty slot at the end)
            const maxConnectedIndex = connectedIndices.length > 0 ? Math.max(...connectedIndices) : 0;
            const sortedUnconnected = [...unconnectedIndices].sort((a, b) => b - a);
            
            for (let i = 1; i < sortedUnconnected.length; i++) {
                const idx = sortedUnconnected[i];
                if (idx > maxConnectedIndex) {
                    removeImageInput(idx);
                }
            }
            
            // Update widget visibility
            updateWidgetVisibility();
            
            // Resize node
            node.setSize(node.computeSize());
        };
        
        /**
         * Update all frame slider max values when frame_count changes
         */
        const updateFrameSliderMax = () => {
            const maxFrame = getFrameCount();
            
            for (const widget of node.widgets || []) {
                if (widget.name.match(/^image_\d+_frame$/)) {
                    if (widget.options) {
                        widget.options.max = maxFrame;
                    }
                    if (widget.value > maxFrame) {
                        widget.value = maxFrame;
                    }
                }
            }
        };
        
        // Hook into connection changes
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
            
            // Only handle input connections (type 1)
            if (type === 1) {
                setTimeout(updateDynamicInputs, 50);
            }
        };
        
        // Hook into configure for loading saved workflows
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function(config) {
            if (originalOnConfigure) {
                originalOnConfigure.apply(this, arguments);
            }
            
            // Rebuild dynamic inputs from saved config
            if (config.inputs) {
                for (const input of config.inputs) {
                    const match = input.name.match(/^image_(\d+)$/);
                    if (match) {
                        const idx = parseInt(match[1]);
                        addImageInput(idx);
                    }
                }
            }
            
            setTimeout(() => {
                updateDynamicInputs();
                updateFrameSliderMax();
            }, 100);
        };
        
        // Hook into frame_count widget changes
        const frameCountWidget = node.widgets?.find(w => w.name === "frame_count");
        if (frameCountWidget) {
            const originalCallback = frameCountWidget.callback;
            frameCountWidget.callback = function(value) {
                if (originalCallback) {
                    originalCallback.apply(this, arguments);
                }
                updateFrameSliderMax();
            };
        }
        
        // Initial setup - remove the default image_1_frame widget (we'll add it dynamically when connected)
        setTimeout(() => {
            // Remove the statically-defined image_1_frame widget
            const staticWidget = node.widgets?.find(w => w.name === "image_1_frame");
            if (staticWidget) {
                const idx = node.widgets.indexOf(staticWidget);
                if (idx >= 0) {
                    node.widgets.splice(idx, 1);
                }
            }
            
            // Ensure image_1 input exists
            const indices = getImageInputIndices();
            if (indices.length === 0) {
                addImageInput(1);
            }
            
            // Update visibility based on current connections
            updateDynamicInputs();
            updateFrameSliderMax();
        }, 100);
    },
});
