import { app } from "/scripts/app.js";

/**
 * VidScribe MiniCPM Beta - Dynamic Widget Extension
 *
 * Hides custom_system_prompt unless system_prompt == "custom".
 *
 * The widget is hidden IN PLACE (hidden flag + zero computeSize + DOM
 * element display), never removed from node.widgets: multiline STRING
 * widgets are DOM overlays whose element survives array removal, and
 * removal also reorders widgets_values so saved workflows reload with
 * values in the wrong widgets.
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
            const customWidget = findWidget("custom_system_prompt");
            const sysWidget = findWidget("system_prompt");
            if (!customWidget || !sysWidget) return false;

            if (customWidget._trentOrigType === undefined) {
                customWidget._trentOrigType = customWidget.type;
                customWidget._trentOrigComputeSize = customWidget.computeSize;
            }

            const updateVisibility = () => {
                const show = sysWidget.value === "custom";

                // New (Vue) frontend respects the hidden flag for both
                // canvas layout and the DOM overlay position.
                customWidget.hidden = !show;

                // Legacy canvas fallback: take no vertical space and
                // skip drawing when hidden.
                customWidget.type = show
                    ? customWidget._trentOrigType
                    : "hidden";
                customWidget.computeSize = show
                    ? customWidget._trentOrigComputeSize
                    : () => [0, -4];

                // Belt and braces for the DOM overlay itself.
                if (customWidget.element) {
                    customWidget.element.style.display = show ? "" : "none";
                }

                requestAnimationFrame(() => {
                    const sz = node.computeSize();
                    node.setSize([node.size[0], sz[1]]);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);
                });
            };

            // Hook system_prompt dropdown
            const orig = sysWidget.callback;
            sysWidget.callback = function () {
                if (orig) orig.apply(this, arguments);
                updateVisibility();
            };

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
