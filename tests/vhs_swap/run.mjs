import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../../js/vhs_swap.js", import.meta.url), "utf8");
const warnings = [];
const context = vm.createContext({
    app: { registerExtension() {} },
    window: {},
    LiteGraph: { registered_node_types: {} },
    console: { warn: (...args) => warnings.push(args) },
});
const script = new vm.Script(source.replace(/^import .*;\r?\n/m, ""));
script.runInContext(context);
const ensure = vm.runInContext("ensureFrameRateInput", context);

// Existing modern sockets must retain their position, links, and widget value.
const modern = {
    inputs: [{ name: "images" }, { name: "frame_rate", link: 42 }],
    widgets: [{ name: "frame_rate", value: 30 }],
};
const before = JSON.stringify(modern);
assert.equal(ensure(modern), 1);
assert.equal(JSON.stringify(modern), before);

// The compatibility helper can appear after the extension module loads.
const legacy = { inputs: [], widgets: [{ name: "frame_rate", value: 24 }] };
const config = ["FLOAT", { default: 8, min: 1 }];
legacy.constructor = { nodeData: { input: { required: { frame_rate: config } } } };
let calls = 0;
context.window.comfyAPI = { widgetInputs: {
    convertToInput(node, widget, suppliedConfig) {
        calls++;
        assert.equal(suppliedConfig, config);
        node.inputs.push({ name: widget.name, type: "FLOAT", widget: { name: widget.name } });
    },
} };
assert.equal(ensure(legacy), 0);
assert.equal(calls, 1);
assert.equal(legacy.widgets[0].value, 24);
assert.equal(ensure(legacy), 0);
assert.equal(calls, 1);

// An unsupported frontend fails locally without changing the node.
delete context.window.comfyAPI;
const missing = { inputs: [], widgets: [{ name: "frame_rate", value: 25 }] };
assert.equal(ensure(missing), -1);
assert.equal(missing.inputs.length, 0);
assert.equal(missing.widgets[0].value, 25);
assert.equal(warnings.length, 1);
console.log("VHS Swap checks passed: existing sockets, delayed helper availability, idempotence, and safe failure.");
