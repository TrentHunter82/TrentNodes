import { app } from "/scripts/app.js";

/**
 * Cowboy Widget Roundup - Dynamic UI Extension
 *
 * node_count drives how many node_N link inputs exist (1..MAX_NODES).
 * widgets_per_node drives how many node_N_widget_M name fields are
 * visible per node (1..MAX_WIDGETS_PER_NODE).
 *
 * All name-field widgets are declared in Python and only hidden here
 * (never removed) because ComfyUI saves widget values positionally --
 * removing widgets scrambles saved workflows. Link inputs serialize by
 * name, so those are safe to add/remove.
 */

const MAX_NODES = 6;
const MAX_WIDGETS_PER_NODE = 4;

app.registerExtension({
    name: "Trent.CowboyWidgetRoundup",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "CowboyWidgetRoundup") {
            return;
        }

        const findWidget = (name) =>
            node.widgets?.find((w) => w.name === name);

        const setWidgetVisible = (widget, visible) => {
            if (!widget) return;
            if (widget._origType === undefined) {
                widget._origType = widget.type;
                widget._origComputeSize = widget.computeSize;
            }
            if (visible) {
                widget.type = widget._origType;
                widget.computeSize = widget._origComputeSize;
            } else {
                widget.type = "hidden";
                // -4 cancels the inter-widget spacing a hidden widget
                // would otherwise still occupy.
                widget.computeSize = () => [0, -4];
            }
        };

        const update = () => {
            const count = findWidget("node_count")?.value ?? 1;
            const perNode = findWidget("widgets_per_node")?.value ?? 1;

            // Link inputs: keep node_1..node_count, drop the rest.
            for (let i = 1; i <= MAX_NODES; i++) {
                const name = `node_${i}`;
                const idx =
                    node.inputs?.findIndex((inp) => inp.name === name) ?? -1;
                if (i <= count && idx < 0) {
                    node.addInput(name, "*");
                } else if (i > count && idx >= 0) {
                    const input = node.inputs[idx];
                    if (input.link !== null) {
                        app.graph.removeLink(input.link);
                    }
                    node.removeInput(idx);
                }
            }

            // Name fields: show only the ones inside both counts.
            for (let i = 1; i <= MAX_NODES; i++) {
                for (let j = 1; j <= MAX_WIDGETS_PER_NODE; j++) {
                    setWidgetVisible(
                        findWidget(`node_${i}_widget_${j}`),
                        i <= count && j <= perNode
                    );
                }
            }

            node.setSize([node.size[0], node.computeSize()[1]]);
            node.setDirtyCanvas(true, true);
        };

        // Re-run the layout whenever either count widget changes.
        for (const name of ["node_count", "widgets_per_node"]) {
            const widget = findWidget(name);
            if (!widget) continue;
            const originalCallback = widget.callback;
            widget.callback = function (...args) {
                originalCallback?.apply(this, args);
                update();
            };
        }

        // After a workflow loads, the saved widget values and inputs
        // are in place; reconcile the layout with them.
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            originalOnConfigure?.apply(this, args);
            setTimeout(update, 50);
        };

        setTimeout(update, 50);
    },
});
