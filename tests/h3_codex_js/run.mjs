import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "../..");
const stage = mkdtempSync(join(tmpdir(), "trent-h3-codex-js-"));
try {
    writeFileSync(join(stage, "package.json"), '{"type":"module"}');
    mkdirSync(join(stage, "scripts"));
    writeFileSync(join(stage, "scripts/app.js"), `export const app = {
        registerExtension(ext) { this.ext = ext; },
        async graphToPrompt() { return { output: this.output, workflow: {} }; },
        extensionManager: { toast: { add() {} } },
        graph: { changes: 0, change() { this.changes++; } },
    };`);
    writeFileSync(join(stage, "scripts/api.js"), `export const api = {
        async queuePrompt(n, p) { this.last = p; }
    };`);
    writeFileSync(join(stage, "extension.js"), readFileSync(join(root, "js/h3_codex_promptor.js"), "utf8").replaceAll("../../scripts/", "./scripts/"));
    const { app } = await import(pathToFileURL(join(stage, "scripts/app.js")));
    const { api } = await import(pathToFileURL(join(stage, "scripts/api.js")));
    const { promptBranch } = await import(pathToFileURL(join(stage, "extension.js")));
    const graph = {
        1: { class_type: "LoadImage", inputs: {} },
        2: { class_type: "TrentH3CodexPromptor", inputs: { first_frame: ["1", 0] } },
        3: { class_type: "KSampler", inputs: { prompt: ["2", 0] } },
        4: { class_type: "SaveVideo", inputs: { video: ["3", 0] } },
        5: { class_type: "SaveImage", inputs: {} },
    };
    assert.deepEqual(Object.keys(promptBranch(graph, 2)), ["1", "2"]);
    assert.throws(() => promptBranch(graph, 7));
    assert.throws(() => promptBranch(graph, 3));
    class Node {
        constructor() {
            this.id = 2; this.mode = 0; this.size = [300, 400]; this.properties = {}; this.inputs = [];
            this.widgets = ["creative_brief", "duration_seconds", "mode", "action", "prompt_text", "refinement",
                "model", "reasoning_effort", "revision", "timeout_seconds", "max_frames", "fps", "reference_roles",
                "context", "dialogue", "source_soundscape", "source_music", "sound_log"].map(name => ({ name, value: "", options: {} }));
            this.widgets.find(w => w.name === "revision").value = 0;
            this.widgets.find(w => w.name === "action").value = "generate";
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget); return widget;
        }
        setDirtyCanvas() {}
    }
    await app.ext.beforeRegisterNodeDef(Node, { name: "TrentH3CodexPromptor" });
    const node = new Node();
    const original = [...node.widgets];
    node.onNodeCreated();
    const w = name => node.widgets.find(x => x.name === name);
    const assertLockState = locked => {
        const status = locked ? "🔒 Locked" : "🔓 Unlocked";
        assert.equal(w("Lock editor").label, `${status} · click to ${locked ? "unlock" : "lock"}`);
        assert.equal(w("prompt_text").label, `Prompt editor · ${status}`);
    };
    assertLockState(false);
    w("Lock editor").callback();
    assert.equal(w("action").value, "generate", "an empty editor cannot be locked");
    assertLockState(false);
    assert.equal(w("model").hidden, true);
    assert.equal(w("context").hidden, true);
    w("context").value = "Keep the red mug";
    node.onConfigure();
    assert.equal(w("context").hidden, false);
    assert.ok(original.every(x => node.widgets.includes(x)), "hiding preserves positional serialization");
    app.output = graph;
    await w("Generate prompt").callback();
    assert.deepEqual(Object.keys(api.last.output), ["1", "2"]);
    node.onExecuted({ h3_prompt: ["generated"], h3_report: ["checks"] });
    assert.equal(w("prompt_text").value, "generated");
    assert.equal(w("action").value, "locked");
    assertLockState(true);
    assert.ok(app.graph.changes >= 2, "queue state and generated editor are marked changed for persistence");
    w("Lock editor").callback();
    assert.equal(w("action").value, "generate");
    assertLockState(false);
    w("Lock editor").callback();
    assert.equal(w("action").value, "locked");
    assertLockState(true);
    const queued = api.queuePrompt;
    api.queuePrompt = async () => { throw new Error("queue unavailable"); };
    await w("Generate prompt").callback();
    assert.equal(w("action").value, "locked", "a rejected queue restores the lock state");
    assertLockState(true);
    api.queuePrompt = queued;
    w("refinement").value = "Slower";
    await w("Refine prompt").callback();
    assertLockState(false);
    w("prompt_text").value = "manual edit while running";
    node.onExecuted({ h3_prompt: ["revised"], h3_report: ["checks"] });
    assert.equal(w("prompt_text").value, "manual edit while running");
    assert.equal(node.properties.h3CodexLastPrompt, "revised");
    assertLockState(false);
    w("action").value = "locked";
    node.onConfigure();
    assertLockState(true);
    w("action").value = "refine";
    w("action").callback("refine");
    assertLockState(false);
    assert.ok(node.widgets.filter(w => w.type === "button").every(w => w.serialize === false));
    console.log("H3 Codex frontend checks passed: prompt-only queue, lock toggle and status, queue failure restoration, edit preservation, widget serialization, context visibility.");
} finally {
    rmSync(stage, { recursive: true, force: true });
}
