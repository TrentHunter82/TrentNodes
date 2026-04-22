import { app } from "/scripts/app.js";

/**
 * VidScribe MiniCPM Beta - Dynamic Widget Extension
 *
 * Hides custom_system_prompt unless system_prompt == "custom".
 * Uses DA3 pattern (array removal) matching frame_ramp_boogie.
 */
app.registerExtension({
    name: "Trent.VidScribeMiniCPM",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "VidScribeMiniCPMBeta") {
            return;
        }

        const findWidget = (name) =>
            node.widgets?.find((w) => w.name === name);

        const setup = () => {
            // Capture ref before any removal
            const customWidget = findWidget("custom_system_prompt");
            if (!customWidget) return false;

            // Set of managed widgets (just one for now)
            const optSet = new Set([customWidget]);

            const updateVisibility = () => {
                const sysVal =
                    findWidget("system_prompt")?.value;
                const shouldShow = sysVal === "custom";

                // Build list of widgets to show
                const toShow = [];
                if (shouldShow) {
                    toShow.push(customWidget);
                }

                // Remove ALL optional widgets from array
                node.widgets = node.widgets.filter(
                    (w) => !optSet.has(w)
                );

                // Re-insert visible ones at end (before
                // any preview widget, or just at end)
                const insertAt = node.widgets.length;
                for (let i = 0; i < toShow.length; i++) {
                    node.widgets.splice(
                        insertAt + i, 0, toShow[i]
                    );
                }

                // Resize
                requestAnimationFrame(() => {
                    const sz = node.computeSize();
                    node.setSize([node.size[0], sz[1]]);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);
                });
            };

            // Hook system_prompt dropdown
            const sysWidget = findWidget("system_prompt");
            if (sysWidget) {
                const orig = sysWidget.callback;
                sysWidget.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    updateVisibility();
                };
            }

            // Initial apply
            updateVisibility();
            return true;
        };

        // Poll until widgets exist, then run setup
        const waitForWidgets = (retries) => {
            if (setup()) return;
            if (retries > 0) {
                requestAnimationFrame(
                    () => waitForWidgets(retries - 1)
                );
            }
        };

        // Handle saved workflow restore
        const origConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (origConfigure) {
                origConfigure.apply(this, arguments);
            }
            waitForWidgets(60);
        };

        // Fresh node
        waitForWidgets(60);
    },
});
