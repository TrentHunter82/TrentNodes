import { app } from "/scripts/app.js";

/**
 * Cherry Pick Frames - Dynamic Output Extension
 * 
 * Dynamically shows the appropriate number of frame outputs based on mode and settings.
 * 
 * Python always returns 16 outputs (frame_1 through frame_16).
 * JS shows only the relevant ones based on current settings.
 */

const MAX_OUTPUTS = 16;

app.registerExtension({
    name: "Trent.CherryPickFrames",
    
    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "CherryPickFrames") {
            return;
        }
        
        const getMode = () => {
            const widget = node.widgets?.find(w => w.name === "mode");
            return widget?.value || "first_n";
        };
        
        const getNumFrames = () => {
            const widget = node.widgets?.find(w => w.name === "num_frames");
            return widget?.value || 1;
        };
        
        const getFrameIndices = () => {
            const widget = node.widgets?.find(w => w.name === "frame_indices");
            return widget?.value || "0";
        };
        
        /**
         * Count how many outputs should be visible based on current settings
         */
        const getVisibleOutputCount = () => {
            const mode = getMode();
            const numFrames = getNumFrames();
            const frameIndices = getFrameIndices();
            
            if (mode === "first_n" || mode === "last_n" || mode === "every_nth") {
                return Math.min(numFrames, MAX_OUTPUTS);
            } else if (mode === "specific") {
                // Count comma-separated values
                const parts = frameIndices.replace(/\s/g, "").split(",").filter(p => p !== "");
                return Math.min(parts.length, MAX_OUTPUTS) || 1;
            }
            return 1;
        };
        
        /**
         * Update visible outputs
         */
        const updateOutputs = () => {
            const visibleCount = getVisibleOutputCount();
            
            // Save connections by name
            const savedConnections = {};
            for (const output of node.outputs || []) {
                if (output.links && output.links.length > 0) {
                    savedConnections[output.name] = output.links.map(linkId => {
                        const link = app.graph.links[linkId];
                        if (link) {
                            return { targetId: link.target_id, targetSlot: link.target_slot };
                        }
                        return null;
                    }).filter(Boolean);
                }
            }
            
            // Clear outputs
            while (node.outputs && node.outputs.length > 0) {
                node.removeOutput(0);
            }
            
            // Add visible frame outputs
            for (let i = 0; i < visibleCount; i++) {
                node.addOutput(`frame_${i + 1}`, "IMAGE");
            }
            
            // Restore connections
            for (let i = 0; i < node.outputs.length; i++) {
                const output = node.outputs[i];
                const saved = savedConnections[output.name];
                if (saved) {
                    for (const conn of saved) {
                        const targetNode = app.graph.getNodeById(conn.targetId);
                        if (targetNode) {
                            node.connect(i, targetNode, conn.targetSlot);
                        }
                    }
                }
            }
            
            node.setSize(node.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };
        
        /**
         * Setup callbacks for all relevant widgets
         */
        const setupCallbacks = () => {
            const widgetNames = ["mode", "num_frames", "frame_indices", "step"];
            
            for (const widget of node.widgets || []) {
                if (widgetNames.includes(widget.name)) {
                    const originalCallback = widget.callback;
                    widget.callback = function(value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        clearTimeout(node._updateTimeout);
                        node._updateTimeout = setTimeout(updateOutputs, 100);
                    };
                }
            }
        };
        
        // Handle loading
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function(config) {
            if (originalOnConfigure) originalOnConfigure.apply(this, arguments);
            setTimeout(() => {
                setupCallbacks();
                updateOutputs();
            }, 100);
        };
        
        // Initial setup
        setTimeout(() => {
            setupCallbacks();
            updateOutputs();
        }, 100);
    },
});
