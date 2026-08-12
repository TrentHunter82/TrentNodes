import { api } from "./api.js";

export const app = {
    extension: null,
    graphOutput: {},
    toasts: [],

    canvas: { selectedItems: new Set(), selected_nodes: {} },

    extensionManager: {
        toast: { add: (o) => app.toasts.push(o) },
        queueSettings: { batchCount: 1 },
    },

    registerExtension(ext) {
        app.extension = ext;
    },

    // Mirrors the real app: build a prompt, then hand it to api.queuePrompt.
    async queuePrompt(number, batchCount = 1) {
        for (let i = 0; i < batchCount; i++) {
            const prompt = {
                output: structuredClone(app.graphOutput),
                workflow: {},
            };
            await api.queuePrompt(number, prompt, {});
        }
        return true;
    },
};
