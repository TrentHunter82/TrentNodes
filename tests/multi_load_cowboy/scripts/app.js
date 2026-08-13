export const app = {
    extension: null,
    toasts: [],

    extensionManager: {
        toast: { add: (o) => app.toasts.push(o) },
    },

    registerExtension(ext) {
        app.extension = ext;
    },
};
