import { app } from "/scripts/app.js";

/**
 * Cowboy Multi-Line String Concatenate (Dynamic) - Dynamic Input Extension
 *
 * Automatically adds a new string input when you connect to the last
 * available one. Removes empty trailing inputs when disconnected
 * (keeping one empty slot).
 */
app.registerExtension({
    name: "Trent.CowboyMultiLineStringConcatenateDynamic",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "CowboyMultiLineStringConcatenateDynamic") {
            return;
        }

        const getStringInputIndices = () => {
            const indices = [];
            for (const input of node.inputs || []) {
                const match = input.name.match(/^string_(\d+)$/);
                if (match) {
                    indices.push(parseInt(match[1]));
                }
            }
            return indices.sort((a, b) => a - b);
        };

        const isInputConnected = (index) => {
            const input = node.inputs?.find(i => i.name === `string_${index}`);
            return input && input.link !== null;
        };

        const addStringInput = (index) => {
            const inputName = `string_${index}`;

            if (node.inputs?.find(i => i.name === inputName)) {
                return false;
            }

            node.addInput(inputName, "*");
            return true;
        };

        const removeStringInput = (index) => {
            const inputName = `string_${index}`;

            const inputIdx = node.inputs?.findIndex(i => i.name === inputName);
            if (inputIdx >= 0) {
                const input = node.inputs[inputIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(inputIdx);
            }
        };

        const updateDynamicInputs = () => {
            const indices = getStringInputIndices();

            if (indices.length === 0) {
                addStringInput(1);
                return;
            }

            const connectedIndices = indices.filter(i => isInputConnected(i));
            const unconnectedIndices = indices.filter(i => !isInputConnected(i));

            const maxIndex = Math.max(...indices);

            // If the highest input is connected, add a new one
            if (isInputConnected(maxIndex)) {
                addStringInput(maxIndex + 1);
            }

            // Remove extra unconnected inputs (keep only one empty slot at the end)
            const maxConnectedIndex = connectedIndices.length > 0
                ? Math.max(...connectedIndices)
                : 0;
            const sortedUnconnected = [...unconnectedIndices].sort((a, b) => b - a);

            for (let i = 1; i < sortedUnconnected.length; i++) {
                const idx = sortedUnconnected[i];
                if (idx > maxConnectedIndex) {
                    removeStringInput(idx);
                }
            }

            node.setSize(node.computeSize());
        };

        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, slotIndex, isConnected, link, ioSlot) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }

            // Only handle input connections (type 1)
            if (type === 1) {
                setTimeout(updateDynamicInputs, 50);
            }
        };

        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (config) {
            if (originalOnConfigure) {
                originalOnConfigure.apply(this, arguments);
            }

            // Rebuild dynamic inputs from saved config
            if (config.inputs) {
                for (const input of config.inputs) {
                    const match = input.name.match(/^string_(\d+)$/);
                    if (match) {
                        addStringInput(parseInt(match[1]));
                    }
                }
            }

            setTimeout(updateDynamicInputs, 100);
        };

        setTimeout(() => {
            const indices = getStringInputIndices();
            if (indices.length === 0) {
                addStringInput(1);
            }
            updateDynamicInputs();
        }, 100);
    },
});
