import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Shows the elapsed-time string on the WorkflowTimer node after each run.
app.registerExtension({
    name: "TrentNodes.WorkflowTimer",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WorkflowTimer") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            const text = message?.text?.join?.("") ?? "";
            let w = this.widgets?.find((w) => w.name === "elapsed");
            if (!w) {
                w = ComfyWidgets.STRING(
                    this,
                    "elapsed",
                    ["STRING", { multiline: false }],
                    app
                ).widget;
                if (w.inputEl) w.inputEl.readOnly = true;
                w.serialize = false;
            }
            w.value = text;
            this.setDirtyCanvas(true, false);
        };
    },
});
