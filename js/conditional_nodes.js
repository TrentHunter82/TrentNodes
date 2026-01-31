import { app } from "/scripts/app.js";

/**
 * Conditional Nodes - Dynamic Input Extension
 *
 * Dynamically manages input visibility for conditional flow nodes.
 * Inputs appear as needed up to the maximum of 3.
 */

const NODE_CONFIGS = {
    "ThisThatOrTheOther": {
        inputs: ["this", "that", "the_other"],
    },
    "FirstValid": {
        inputs: ["first", "second", "third"],
    },
};

app.registerExtension({
    name: "Trent.ConditionalNodes",

    async nodeCreated(node) {
        const config = NODE_CONFIGS[node.constructor.comfyClass];
        if (!config) {
            return;
        }

        const { inputs } = config;
        const maxInputs = inputs.length;

        /**
         * Get all managed input names currently on the node
         */
        const getExistingInputNames = () => {
            return (node.inputs || [])
                .filter(input => inputs.includes(input.name))
                .map(input => input.name);
        };

        /**
         * Check if an input is connected
         */
        const isInputConnected = (inputName) => {
            const input = node.inputs?.find(i => i.name === inputName);
            return input && input.link !== null;
        };

        /**
         * Get the index of an input in the config array
         */
        const getInputIndex = (inputName) => {
            return inputs.indexOf(inputName);
        };

        /**
         * Add an input slot by name
         */
        const addInput = (inputName) => {
            if (node.inputs?.find(i => i.name === inputName)) {
                return false;
            }
            node.addInput(inputName, "*");
            return true;
        };

        /**
         * Remove an input slot by name
         */
        const removeInput = (inputName) => {
            const inputIdx = node.inputs?.findIndex(i => i.name === inputName);
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

        /**
         * Update dynamic inputs based on connection state
         */
        const updateDynamicInputs = () => {
            const existing = getExistingInputNames();

            // Ensure at least the first input exists
            if (existing.length === 0) {
                addInput(inputs[0]);
                node.setSize(node.computeSize());
                return;
            }

            // Find the highest index that is connected
            let maxConnectedIndex = -1;
            for (let i = 0; i < maxInputs; i++) {
                if (isInputConnected(inputs[i])) {
                    maxConnectedIndex = i;
                }
            }

            // We need inputs up to maxConnectedIndex + 2 (one after last connected)
            // but capped at maxInputs
            const neededCount = Math.min(maxConnectedIndex + 2, maxInputs);

            // Add inputs up to needed count
            for (let i = 0; i < neededCount; i++) {
                if (!existing.includes(inputs[i])) {
                    addInput(inputs[i]);
                }
            }

            // Remove excess unconnected inputs from the end
            for (let i = maxInputs - 1; i >= neededCount; i--) {
                if (!isInputConnected(inputs[i]) && existing.includes(inputs[i])) {
                    removeInput(inputs[i]);
                }
            }

            // Resize node
            node.setSize(node.computeSize());
            app.graph.setDirtyCanvas(true, true);
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
        node.onConfigure = function(configData) {
            if (originalOnConfigure) {
                originalOnConfigure.apply(this, arguments);
            }

            // Restore inputs from saved config
            if (configData.inputs) {
                for (const savedInput of configData.inputs) {
                    if (inputs.includes(savedInput.name)) {
                        addInput(savedInput.name);
                    }
                }
            }

            setTimeout(updateDynamicInputs, 100);
        };

        // Initial setup - start with just the first input
        setTimeout(() => {
            const existing = getExistingInputNames();

            // Remove any inputs beyond the first that aren't connected
            for (let i = 1; i < maxInputs; i++) {
                if (!isInputConnected(inputs[i]) && existing.includes(inputs[i])) {
                    removeInput(inputs[i]);
                }
            }

            // Ensure first input exists
            if (!existing.includes(inputs[0])) {
                addInput(inputs[0]);
            }

            updateDynamicInputs();
        }, 100);
    },
});
