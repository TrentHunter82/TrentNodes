// Checks js/queue_selected_outputs.js against a mocked ComfyUI frontend.
//
//   node tests/queue_selected_outputs/run.mjs
//
// The extension imports "../../scripts/app.js", which only resolves when
// ComfyUI serves it. So we copy the source next to the mocks in ./scripts and
// rewrite that one import, rather than keeping a second copy of the logic.

import { mkdtempSync, readFileSync, rmSync, writeFileSync, cpSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "..", "js", "queue_selected_outputs.js");

const stage = mkdtempSync(join(tmpdir(), "trentnodes-queue-"));
process.on("exit", () => rmSync(stage, { recursive: true, force: true }));

cpSync(join(here, "scripts"), join(stage, "scripts"), { recursive: true });
writeFileSync(
    join(stage, "extension.js"),
    readFileSync(source, "utf8").replaceAll("../../scripts/", "./scripts/"),
);

// ---------------------------------------------------------------------------
// Minimal LiteGraph stand-ins
// ---------------------------------------------------------------------------

class LGraphNode {
    constructor(id, type, { output = false, mode = 0, subgraph = null } = {}) {
        this.id = id;
        this.type = type;
        this.mode = mode;
        this.subgraph = subgraph;
        // The real output_node flag lives on the node class, not the instance.
        Object.defineProperty(this, "constructor", {
            value: { nodeData: { output_node: output } },
        });
    }
    isSubgraphNode() {
        return this.subgraph !== null;
    }
}

class LGraphGroup {
    constructor(children) {
        this.children = children;
    }
}

globalThis.LGraphNode = LGraphNode;
globalThis.LGraphGroup = LGraphGroup;

const { app } = await import(pathToFileURL(join(stage, "scripts", "app.js")));
const { api } = await import(pathToFileURL(join(stage, "scripts", "api.js")));
await import(pathToFileURL(join(stage, "extension.js")));

await app.extension.setup();
const run = app.extension.commands[0].function;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

let failures = 0;

function check(name, actual, expected) {
    const a = JSON.stringify(actual);
    const e = JSON.stringify(expected);
    if (a === e) {
        console.log(`  PASS  ${name}`);
    } else {
        failures++;
        console.log(`  FAIL  ${name}\n        expected ${e}\n        actual   ${a}`);
    }
}

function reset(output, selected) {
    app.graphOutput = output;
    app.canvas.selectedItems = new Set(selected);
    app.canvas.selected_nodes = {};
    app.toasts = [];
    api.calls = [];
}

/** The node ids of every prompt that reached the server, in order. */
function sentKeys() {
    return api.calls.map((c) => Object.keys(c.output).sort());
}

// ---------------------------------------------------------------------------
// A flat graph: 1 -> 2 -> 5 (SaveImage), and 7 -> 9 (SaveImage)
// ---------------------------------------------------------------------------

const FLAT = {
    "1": { class_type: "CheckpointLoader", inputs: { ckpt_name: "x.safetensors" } },
    "2": { class_type: "KSampler", inputs: { model: ["1", 0], seed: 42 } },
    "5": { class_type: "SaveImage", inputs: { images: ["2", 0] } },
    "7": { class_type: "LoadImage", inputs: { image: "y.png" } },
    "9": { class_type: "SaveImage", inputs: { images: ["7", 0] } },
};

const save5 = new LGraphNode(5, "SaveImage", { output: true });
const save9 = new LGraphNode(9, "SaveImage", { output: true });
const sampler2 = new LGraphNode(2, "KSampler");

console.log("Flat graph");

reset(FLAT, [save5]);
await run();
check("one output node pulls only its own branch", sentKeys(), [["1", "2", "5"]]);

reset(FLAT, [save5, save9]);
await run();
check("two output nodes pull both branches", sentKeys(), [["1", "2", "5", "7", "9"]]);

reset(FLAT, [sampler2]);
await run();
check("non-output selection queues nothing", sentKeys(), []);
check("non-output selection warns the user", app.toasts.map((t) => t.summary), ["Nothing to queue"]);

reset(FLAT, []);
await run();
check("empty selection queues nothing", sentKeys(), []);

reset(FLAT, [new LGraphNode(5, "SaveImage", { output: true, mode: 2 })]);
await run();
check("muted output node queues nothing", sentKeys(), []);

reset(FLAT, [new LGraphNode(5, "SaveImage", { output: true, mode: 4 })]);
await run();
check("bypassed output node queues nothing", sentKeys(), []);

console.log("Groups");

reset(FLAT, [new LGraphGroup([sampler2, save9])]);
await run();
check("group selection pulls its output nodes", sentKeys(), [["7", "9"]]);

console.log("Legacy selection API");

app.graphOutput = FLAT;
app.canvas.selectedItems = new Set();
app.canvas.selected_nodes = { 5: save5 };
app.toasts = [];
api.calls = [];
await run();
check("selected_nodes fallback works", sentKeys(), [["1", "2", "5"]]);

// ---------------------------------------------------------------------------
// A graph with a subgraph node 12 wrapping inner nodes 3 -> 5
// ---------------------------------------------------------------------------

console.log("Subgraphs");

const NESTED = {
    "1": { class_type: "LoadImage", inputs: { image: "y.png" } },
    "12:3": { class_type: "ImageScale", inputs: { image: ["1", 0] } },
    "12:5": { class_type: "SaveImage", inputs: { images: ["12:3", 0] } },
    "20": { class_type: "SaveImage", inputs: { images: ["1", 0] } },
};

const innerSave = new LGraphNode(5, "SaveImage", { output: true });
const innerScale = new LGraphNode(3, "ImageScale");
const subgraphNode = new LGraphNode(12, "Subgraph", {
    subgraph: { nodes: [innerScale, innerSave] },
});

reset(NESTED, [subgraphNode]);
await run();
check("selected subgraph node resolves its inner output", sentKeys(), [["1", "12:3", "12:5"]]);

reset(NESTED, [innerSave]);
await run();
check("inner node alone resolves by suffix", sentKeys(), [["1", "12:3", "12:5"]]);

console.log("Batch count");

app.extensionManager.queueSettings.batchCount = 3;
reset(FLAT, [save5]);
await run();
check("batch count queues three pruned prompts", sentKeys(), [
    ["1", "2", "5"], ["1", "2", "5"], ["1", "2", "5"],
]);
app.extensionManager.queueSettings.batchCount = 1;

console.log("Isolation");

reset(FLAT, [save5]);
await run();
api.calls = [];
await app.queuePrompt(0, 1);
check("a normal queue is untouched once the run ends", sentKeys(), [
    ["1", "2", "5", "7", "9"],
]);

console.log(failures ? `\n${failures} failure(s)` : "\nAll checks passed");
process.exit(failures ? 1 : 0);
