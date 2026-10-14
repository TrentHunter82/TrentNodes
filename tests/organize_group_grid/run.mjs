import assert from "node:assert/strict";
import { findSourceBundles, stackSourcesAbove, exposeUnconnectedInputs } from "../../js/source_subgraphs.js";

const source = (id, links = []) => ({
    id, title: `Source ${id}`, pos: [0, 0], size: [200, 100],
    inputs: [{ name: "value", type: "INT", link: null }],
    outputs: [{ links }],
});
const consumer = {
    id: 10, title: "Consumer", pos: [500, 500], size: [300, 200],
    inputs: [{ name: "first", link: 0 }, { name: "last", link: 1 }], outputs: [],
};
const a = source(1, [0]), b = source(2, [1]);
const links = new Map([
    [0, { origin_id: 1, target_id: 10, target_slot: 0 }],
    [1, { origin_id: 2, target_id: 10, target_slot: 1 }],
]);
const nodes = new Map([[1, a], [2, b], [10, consumer]]);
const graph = { getLink: (id) => links.get(id), getNodeById: (id) => nodes.get(id) };
const bundles = () => findSourceBundles([consumer], [a, b], graph);

assert.deepEqual(bundles()[0].sources.map((s) => s.node.id), [1, 2]);
assert.equal(findSourceBundles([consumer], [b], graph)[0].sources.length, 1);
a.inputs[0].link = 0;
assert.deepEqual(bundles()[0].sources.map((s) => s.node.id), [1, 2]);
a.inputs[0].link = null;
a.outputs[0].links.push(2);
links.set(2, { origin_id: 1, target_id: 99 });
assert.deepEqual(bundles()[0].sources.map((s) => s.node.id), [1, 2]);
a.outputs[0].links.pop();
a.pinned = true;
assert.deepEqual(bundles()[0].sources.map((s) => s.node.id), [2]);
a.pinned = false;
consumer.pinned = true;
assert.equal(bundles().length, 0);
consumer.pinned = false;
a.isSubgraphNode = () => true;
assert.deepEqual(bundles()[0].sources.map((s) => s.node.id), [2]);
delete a.isSubgraphNode;
consumer.inputs.push({ name: "duplicate", link: 3 });
links.set(3, { origin_id: 1, target_id: 10, target_slot: 2 });
a.outputs[0].links.push(3);
assert.equal(bundles()[0].sources.length, 2);
assert.equal(bundles()[0].sources[0].inputIndex, 0);

const upstream = source(3, [4]);
nodes.set(3, upstream);
links.set(4, { origin_id: 3, target_id: 1 });
a.inputs[0].link = 4;
const direct = findSourceBundles([consumer], [upstream, a, b, consumer], graph);
assert.deepEqual(direct[0].sources.map((s) => s.node.id), [1, 2], 'Only direct providers belong inside');
const chain = findSourceBundles([a, consumer], [upstream, a, b, consumer], graph);
assert.equal(chain.length, 1, 'Do not capture an already included provider as another consumer');
assert.equal(chain[0].sources.some((s) => s.node === upstream), false);
a.inputs[0].link = null;

const other = { ...consumer, id: 11, inputs: [{ name: 'first', link: 5 }, { name: 'free', link: null }] };
nodes.set(11, other);
links.set(5, { origin_id: 1, target_id: 11 });
const shared = findSourceBundles([consumer, other], [a, b, consumer, other], graph);
assert.equal(shared.length, 1, 'Shared providers must not be moved twice');
assert.equal(shared[0].consumers.length, 2);
assert.deepEqual(shared[0].sources.map((s) => s.node.id), [1, 2]);

stackSourcesAbove(consumer, bundles()[0].sources);
assert.equal(b.pos[1] + b.size[1] + 40, consumer.pos[1] - 30);
assert.equal(a.pos[1] + a.size[1] + 40, b.pos[1] - 30);
b.flags = { collapsed: true };
stackSourcesAbove(consumer, bundles()[0].sources);
assert.equal(b.pos[1] + 40, consumer.pos[1] - 30);
assert.equal(a.pos[1] + a.size[1] + 40, b.pos[1] - 30);

const exposed = [];
const subgraph = {
    inputs: [],
    addInput(name, type) {
        const slot = { name, type, connect(input, node) {
            exposed.push([node.id, input.name, name, type]);
            input.link = 100 + exposed.length;
            return {};
        } };
        this.inputs.push(slot);
        return slot;
    },
    removeInput(slot) { this.inputs.splice(this.inputs.indexOf(slot), 1); },
};
const c = source(3);
c.inputs.push({ name: "connected", type: "INT", link: 0 });
c.inputs.push({ name: "hidden", type: "STRING", link: null });
c.getWidgetFromSlot = (s) => ({ value: 7, hidden: s.name === "hidden" });
exposeUnconnectedInputs(subgraph, [a, c]);
assert.deepEqual(exposed, [[1, "value", "value", "INT"], [3, "value", "value_2", "INT"]]);
assert.equal(c.inputs[1].link, 0);
assert.equal(c.inputs[2].link, null);
exposeUnconnectedInputs(subgraph, [a, c]);
assert.equal(exposed.length, 2);

const failing = { ...subgraph, inputs: [], addInput(name, type) {
    const slot = { name, type, connect() { return undefined; } };
    this.inputs.push(slot); return slot;
} };
assert.throws(() => exposeUnconnectedInputs(failing, [source(4)]), /Cannot expose/);
assert.equal(failing.inputs.length, 0);
console.log("Group organizer: direct providers, scope, shared providers, connected inputs, pins, ordering, collapsed spacing, exposure, and failed connections passed.");

await import("./layout.mjs");
