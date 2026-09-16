import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NAME = "TrentH3CodexPromptor";
const ADVANCED = ["model", "reasoning_effort", "revision", "timeout_seconds", "max_frames", "fps"];
const CONTEXT = ["reference_roles", "context", "dialogue", "source_soundscape", "source_music", "sound_log"];
const widget = (node, name) => node.widgets?.find((entry) => entry.name === name);
const changed = (node) => (node.graph ?? app.graph)?.change?.();

function notify(message) {
    app.extensionManager?.toast?.add({ severity: "info", summary: "H3 Codex Promptor", detail: message, life: 6000 });
}

// Sending only this dependency closure prevents a prompt-writing click from
// starting downstream H3 samplers or unrelated output nodes.
export function promptBranch(output, id) {
    if (!output[String(id)] || output[String(id)].class_type !== NAME) {
        throw new Error("Place the promptor on the main canvas to use its Generate/Refine buttons. Inside a subgraph, use ComfyUI Run.");
    }
    const result = {};
    function visit(key) {
        key = String(key);
        if (result[key]) return;
        const entry = output[key];
        if (!entry) throw new Error(`Missing upstream node ${key}.`);
        result[key] = entry;
        for (const value of Object.values(entry.inputs || {})) {
            if (Array.isArray(value) && value.length === 2 && Number.isInteger(value[1])
                && (typeof value[0] === "string" || typeof value[0] === "number") && output[String(value[0])]) {
                visit(value[0]);
            }
        }
    }
    visit(id);
    return result;
}

function refresh(node) {
    for (const name of ADVANCED) {
        const entry = widget(node, name);
        if (!entry) continue;
        entry.hidden = !node.properties.h3CodexAdvanced;
        entry.options = { ...entry.options, hidden: entry.hidden };
    }
    for (const name of CONTEXT) {
        const entry = widget(node, name);
        if (!entry) continue;
        const connected = node.inputs?.some((input) => input.name === name && input.link != null);
        entry.hidden = !node.properties.h3CodexContext && !connected && !entry.value?.trim();
        entry.options = { ...entry.options, hidden: entry.hidden };
    }
    if (node.setSize && node.computeSize) {
        const minimum = node.computeSize();
        node.setSize([Math.max(node.size[0], 480), Math.max(node.size[1], minimum[1], 760)]);
    }
    node.setDirtyCanvas?.(true, true);
}

async function queueWriting(node, action) {
    if (node._h3Submitting) return;
    if (node.mode === 2 || node.mode === 4) {
        notify("Enable this node before generating a prompt.");
        return;
    }
    if (action === "refine" && (!widget(node, "prompt_text")?.value?.trim() || !widget(node, "refinement")?.value?.trim())) {
        notify("Keep a prompt in the editor and enter a refinement request first.");
        return;
    }
    node._h3Submitting = true;
    const actionWidget = widget(node, "action");
    const oldAction = actionWidget.value;
    actionWidget.value = action;
    node._h3EditorAtQueue = widget(node, "prompt_text").value;
    try {
        const request = await app.graphToPrompt();
        request.output = promptBranch(request.output, node.id);
        await api.queuePrompt(0, request);
        changed(node);
        notify("Prompt writing queued. Downstream H3 generation was not queued.");
    } catch (error) {
        actionWidget.value = oldAction;
        notify(String(error.message || error));
    } finally {
        node._h3Submitting = false;
        refresh(node);
    }
}

app.registerExtension({
    name: "TrentNodes.H3CodexPromptor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NAME) return;
        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previousCreated?.apply(this, args);
            this.properties ||= {};
            const button = (name, callback) => {
                const entry = this.addWidget("button", name, null, callback, { serialize: false });
                entry.serialize = false;
            };
            button("Generate prompt", () => queueWriting(this, "generate"));
            button("Refine prompt", () => queueWriting(this, "refine"));
            button("Lock editor", () => {
                if (!widget(this, "prompt_text").value.trim()) return notify("The prompt editor is empty.");
                widget(this, "action").value = "locked";
                changed(this);
                notify("Locked: H3 runs will use the editor exactly as saved.");
                refresh(this);
            });
            button("New variation", () => {
                widget(this, "revision").value += 1;
                return queueWriting(this, "generate");
            });
            button("Advanced settings", () => {
                this.properties.h3CodexAdvanced = !this.properties.h3CodexAdvanced;
                refresh(this);
                changed(this);
            });
            button("Context & dialogue", () => {
                this.properties.h3CodexContext = !this.properties.h3CodexContext;
                refresh(this);
                changed(this);
            });
            const editor = widget(this, "prompt_text");
            const previousBefore = editor.beforeQueued;
            editor.beforeQueued = (...values) => {
                previousBefore?.apply(editor, values);
                this._h3EditorAtQueue = editor.value;
            };
            this.size[0] = Math.max(this.size[0], 430);
            refresh(this);
            return result;
        };
        const previousConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = previousConfigure?.apply(this, args);
            refresh(this);
            return result;
        };
        const previousExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            previousExecuted?.call(this, message);
            const result = message?.h3_prompt?.[0];
            if (typeof result !== "string") return;
            const editor = widget(this, "prompt_text");
            this.properties.h3CodexLastPrompt = result;
            this.properties.h3CodexReport = message.h3_report?.[0] || "";
            if (this._h3EditorAtQueue === undefined || editor.value === this._h3EditorAtQueue) {
                editor.value = result;
                widget(this, "action").value = "locked";
                notify("Prompt ready and locked. Edit it directly, or enter a request and click Refine.");
            } else {
                notify("Prompt finished; your in-progress editor changes were preserved. The generated prompt is available at the node output.");
            }
            delete this._h3EditorAtQueue;
            changed(this);
            refresh(this);
        };
        const previousMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            previousMenu?.apply(this, arguments);
            options.push({ content: "Show Codex validation report", callback: () => {
                const report = document.createElement("pre");
                report.style.cssText = "white-space:pre-wrap;max-width:70vw;max-height:70vh;overflow:auto";
                report.textContent = this.properties.h3CodexReport || "Generate a prompt first.";
                app.ui.dialog.show(report);
            } });
        };
    },
});
