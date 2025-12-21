import { app } from "/scripts/app.js";

/**
 * VidScribe MiniCPM Beta - Dynamic Widget Extension
 *
 * Shows/hides the custom_system_prompt widget based on the system_prompt
 * dropdown selection. Only visible when "custom" is selected.
 */
app.registerExtension({
    name: "Trent.VidScribeMiniCPM",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "VidScribeMiniCPMBeta") {
            return;
        }

        /**
         * Get current system_prompt dropdown value
         */
        const getSystemPrompt = () => {
            const widget = node.widgets?.find(w => w.name === "system_prompt");
            return widget?.value || "default";
        };

        /**
         * Find the custom_system_prompt widget
         */
        const findCustomWidget = () => {
            return node.widgets?.find(w => w.name === "custom_system_prompt");
        };

        /**
         * Hide a widget by setting its type to hidden and height to 0
         */
        const hideWidget = (widget) => {
            if (!widget) return;
            widget.origType = widget.type;
            widget.origComputeSize = widget.computeSize;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
        };

        /**
         * Show a widget by restoring its original type
         */
        const showWidget = (widget) => {
            if (!widget || !widget.origType) return;
            widget.type = widget.origType;
            widget.computeSize = widget.origComputeSize;
        };

        /**
         * Update visibility of custom_system_prompt based on dropdown
         */
        const updateCustomPromptVisibility = () => {
            const systemPrompt = getSystemPrompt();
            const shouldShow = systemPrompt === "custom";
            const customWidget = findCustomWidget();

            if (!customWidget) return;

            if (shouldShow) {
                showWidget(customWidget);
            } else {
                hideWidget(customWidget);
            }

            // Resize node to fit
            node.setSize(node.computeSize());
            app.graph.setDirtyCanvas(true, true);
        };

        /**
         * Setup callback on system_prompt dropdown
         */
        const setupDropdownCallback = () => {
            const sysPromptWidget = node.widgets?.find(w => w.name === "system_prompt");
            if (sysPromptWidget) {
                const originalCallback = sysPromptWidget.callback;
                sysPromptWidget.callback = function(value) {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    updateCustomPromptVisibility();
                };
            }
        };

        // Handle loading saved workflows
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function(config) {
            if (originalOnConfigure) {
                originalOnConfigure.apply(this, arguments);
            }

            setTimeout(() => {
                setupDropdownCallback();
                updateCustomPromptVisibility();
            }, 100);
        };

        // Initial setup - wait for widgets to be fully created
        setTimeout(() => {
            setupDropdownCallback();
            updateCustomPromptVisibility();
        }, 100);
    },
});
