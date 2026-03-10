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
        
        // Cache for frame widget values - survives widget removal/recreation
        const frameValueCache = {};

        /**
         * Save all current frame widget values to cache
         */
        const cacheAllFrameValues = () => {
            for (const widget of node.widgets || []) {
                const match = widget.name.match(/^image_(\d+)_frame$/);
                if (match) {
                    frameValueCache[parseInt(match[1])] = widget.value;
                }
            }
        };

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
         * Get auto_spacing toggle value
         */
        const getAutoSpacing = () => {
            const widget = node.widgets?.find(w => w.name === "auto_spacing");
            return widget?.value || false;
        };

        /**
         * Get spacing_type value
         */
        const getSpacingType = () => {
            const widget = node.widgets?.find(w => w.name === "spacing_type");
            return widget?.value || "linear";
        };

        /**
         * Get min_spacing value
         */
        const getMinSpacing = () => {
            const widget = node.widgets?.find(w => w.name === "min_spacing");
            return widget?.value || 8;
        };

        /**
         * Easing functions - take t in [0,1], return eased value in [0,1]
         */
        const easingFunctions = {
            linear: (t) => t,
            ease_in: (t) => t * t,
            ease_out: (t) => 1 - (1 - t) * (1 - t),
            ease_in_out: (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2,
        };

        /**
         * Calculate auto-spaced frame positions for connected images
         * Enforces minimum spacing between keyframes for VACE compatibility
         */
        const calculateAutoSpacedFrames = () => {
            const indices = getImageInputIndices();
            const connectedIndices = indices.filter(i => isInputConnected(i)).sort((a, b) => a - b);

            if (connectedIndices.length === 0) return {};

            const frameCount = getFrameCount();
            const spacingType = getSpacingType();
            const minSpacing = getMinSpacing();
            const easingFn = easingFunctions[spacingType] || easingFunctions.linear;

            const framePositions = {};
            const numKeyframes = connectedIndices.length;

            if (numKeyframes === 1) {
                // Single image goes to frame 1
                framePositions[connectedIndices[0]] = 1;
            } else {
                // Calculate minimum required frames for this many keyframes
                // Need (numKeyframes - 1) gaps, each at least minSpacing
                const minRequiredFrames = 1 + (numKeyframes - 1) * minSpacing;

                if (frameCount < minRequiredFrames) {
                    // Not enough frames - space evenly with whatever we have
                    // but warn via console
                    console.warn(`[WanVace] Warning: ${frameCount} frames is not enough for ${numKeyframes} keyframes with ${minSpacing} min spacing. Need at least ${minRequiredFrames} frames.`);
                }

                // Apply easing with minimum spacing constraint
                // First, calculate raw eased positions
                const rawPositions = [];
                for (let i = 0; i < numKeyframes; i++) {
                    const t = i / (numKeyframes - 1);
                    const easedT = easingFn(t);
                    rawPositions.push(easedT);
                }

                // Now map to frame positions while enforcing minimum spacing
                // Start from frame 1, end at frameCount
                const availableRange = frameCount - 1; // frames we can distribute across

                for (let i = 0; i < numKeyframes; i++) {
                    let frame;
                    if (i === 0) {
                        frame = 1; // First keyframe always at frame 1
                    } else if (i === numKeyframes - 1) {
                        frame = frameCount; // Last keyframe always at end
                    } else {
                        // Middle keyframes: apply easing but enforce min spacing
                        const rawFrame = Math.round(1 + rawPositions[i] * availableRange);

                        // Ensure at least minSpacing from previous
                        const prevFrame = framePositions[connectedIndices[i - 1]];
                        const minFrame = prevFrame + minSpacing;

                        // Also ensure room for remaining keyframes
                        const remainingKeyframes = numKeyframes - 1 - i;
                        const maxFrame = frameCount - (remainingKeyframes * minSpacing);

                        frame = Math.max(minFrame, Math.min(maxFrame, rawFrame));
                    }
                    framePositions[connectedIndices[i]] = frame;
                }
            }

            return framePositions;
        };

        /**
         * Apply auto-spacing to frame widgets
         */
        const applyAutoSpacing = () => {
            if (!getAutoSpacing()) return;

            const framePositions = calculateAutoSpacedFrames();

            for (const [imgIdx, framePos] of Object.entries(framePositions)) {
                const widget = node.widgets?.find(w => w.name === `image_${imgIdx}_frame`);
                if (widget) {
                    widget.value = framePos;
                }
            }
        };
        
        /**
         * Find or create a frame slider widget for given index
         */
        const ensureFrameWidget = (index) => {
            const frameName = `image_${index}_frame`;
            let widget = node.widgets?.find(w => w.name === frameName);

            if (!widget) {
                const maxFrame = getFrameCount();
                // Restore cached value if available, otherwise default
                const defaultValue = frameValueCache[index] ?? index;
                widget = node.addWidget(
                    "slider",
                    frameName,
                    defaultValue,
                    (value) => {
                        frameValueCache[index] = value;
                    },
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

            // Cache value before removing widget
            const widget = node.widgets?.find(w => w.name === frameName);
            if (widget) {
                frameValueCache[index] = widget.value;
            }

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
            // Cache all current values before any removal
            cacheAllFrameValues();

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
         *
         * Logic:
         * - Slots stay in fixed positions (image_1, image_2, image_3, etc.)
         * - Always keep exactly one empty slot at the bottom
         * - Only add a new slot when the bottom-most slot gets connected
         * - Only remove consecutive empty slots from the bottom (keep one)
         * - Middle slots can be empty - they stay visible without frame widgets
         */
        const updateDynamicInputs = () => {
            const indices = getImageInputIndices();

            if (indices.length === 0) {
                addImageInput(1);
                return;
            }

            let maxIndex = Math.max(...indices);

            // If the bottom-most input is connected, add a new empty slot
            if (isInputConnected(maxIndex)) {
                addImageInput(maxIndex + 1);
            } else {
                // Remove consecutive empty slots from the bottom, keeping one
                // Example: [connected, empty, empty, empty] -> [connected, empty]
                while (maxIndex > 1) {
                    const secondToLast = maxIndex - 1;
                    // If both bottom and second-to-bottom are empty, remove bottom
                    if (!isInputConnected(maxIndex) && !isInputConnected(secondToLast)) {
                        removeImageInput(maxIndex);
                        maxIndex = secondToLast;
                    } else {
                        // Either bottom is connected or second-to-last is connected
                        // Stop removing
                        break;
                    }
                }
            }

            // Update widget visibility
            updateWidgetVisibility();

            // Apply auto-spacing if enabled
            applyAutoSpacing();

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

            // Cache all existing frame widget values before anything
            // can remove them (values were restored by LiteGraph)
            cacheAllFrameValues();

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
                applyAutoSpacing();
            };
        }

        // Hook into auto_spacing toggle changes
        const autoSpacingWidget = node.widgets?.find(w => w.name === "auto_spacing");
        if (autoSpacingWidget) {
            const originalCallback = autoSpacingWidget.callback;
            autoSpacingWidget.callback = function(value) {
                if (originalCallback) {
                    originalCallback.apply(this, arguments);
                }
                applyAutoSpacing();
            };
        }

        // Hook into spacing_type dropdown changes
        const spacingTypeWidget = node.widgets?.find(w => w.name === "spacing_type");
        if (spacingTypeWidget) {
            const originalCallback = spacingTypeWidget.callback;
            spacingTypeWidget.callback = function(value) {
                if (originalCallback) {
                    originalCallback.apply(this, arguments);
                }
                applyAutoSpacing();
            };
        }

        // Hook into min_spacing changes
        const minSpacingWidget = node.widgets?.find(w => w.name === "min_spacing");
        if (minSpacingWidget) {
            const originalCallback = minSpacingWidget.callback;
            minSpacingWidget.callback = function(value) {
                if (originalCallback) {
                    originalCallback.apply(this, arguments);
                }
                applyAutoSpacing();
            };
        }

        // Initial setup - remove the default image_1_frame widget (we'll add it dynamically when connected)
        setTimeout(() => {
            // Cache any existing values first (important for workflow load)
            cacheAllFrameValues();

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
