// Clear-slot button for Save for Next Time / Take from Last Time.
//
// Adds a "Clear slot" button to both nodes. It deletes every
// numbered entry in the node's slot via /trent/for_next_time/clear.
// Hand-renamed (pinned) entries are kept, matching prune's rules.
//
// A button, not a toggle: clearing is a one-shot action you do at
// the start of a new loop. A boolean widget would be saved with
// the workflow and silently wipe the slot on every queue.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAMES = ["SaveForNextTime", "TakeFromLastTime"];

function toast(severity, summary, detail) {
    app.extensionManager?.toast?.add({
        severity,
        summary,
        detail,
        life: 4000,
    });
}

async function clearSlot(slotName) {
    const response = await api.fetchApi("/trent/for_next_time/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slot_name: slotName }),
    });
    return response.json();
}

app.registerExtension({
    name: "TrentNodes.ForNextTime",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.includes(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const slotWidget = this.widgets?.find(
                (w) => w.name === "slot_name"
            );
            if (!slotWidget) {
                console.warn("[ForNextTime] slot_name widget not found");
                return result;
            }

            const clearWidget = this.addWidget(
                "button",
                "clear_slot",
                "Clear slot",
                async () => {
                    const slot = (slotWidget.value || "").trim();
                    if (!slot) {
                        toast("warn", "Clear slot", "slot_name is empty.");
                        return;
                    }
                    if (!confirm(`Delete all saved entries in slot "${slot}"?`)) {
                        return;
                    }
                    try {
                        const data = await clearSlot(slot);
                        if (!data.ok) {
                            toast("error", "Clear slot", data.error || "Failed.");
                            return;
                        }
                        let detail = `Removed ${data.removed.length} entries from "${slot}".`;
                        if (data.kept_pinned.length) {
                            detail += ` Kept ${data.kept_pinned.length} pinned.`;
                        }
                        toast("success", "Clear slot", detail);
                    } catch (error) {
                        toast("error", "Clear slot", String(error));
                    }
                }
            );
            // serialize=false keeps the button out of saved workflows.
            clearWidget.serialize = false;

            return result;
        };
    },
});
