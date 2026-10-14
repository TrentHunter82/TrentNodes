import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

class Node {
    constructor(id, width = 180, height = 100, y = 0) {
        Object.assign(this, { id, pos: [0, y], size: [width, height], flags: {}, inputs: [], outputs: [], boundingRect: [0, 0, 0, 0] });
    }
    updateArea() {
        this.boundingRect = [this.pos[0], this.pos[1] - 30,
            this.flags.collapsed ? 80 : this.size[0], this.flags.collapsed ? 30 : this.size[1] + 30];
    }
    getInputPos(slot) { return [this.pos[0], this.pos[1] + (this.flags.collapsed ? -15 : this.inputs[slot].offset ?? 14 + slot * 20)]; }
    getOutputPos(slot) { return [this.pos[0] + (this.flags.collapsed ? 80 : this.size[0]), this.pos[1] + (this.flags.collapsed ? -15 : this.outputs[slot].offset ?? 14 + slot * 20)]; }
}
let spacing = 'Compact';
const context = vm.createContext({ window: { LiteGraph: { LGraphNode: Node, NODE_TITLE_HEIGHT: 30 } },
    app: { canvas: {}, extensionManager: { setting: { get: () => spacing } }, registerExtension() {} }, console });
const code = await readFile(new URL('../../js/organize_group_compact.js', import.meta.url), 'utf8');
vm.runInContext(code.replace(/^import .*\r?\n/gm, '') + '\nglobalThis.arrange = arrangeGroup;', context);

function fixture(nodes) {
    const byId = new Map(nodes.map(n => [n.id, n])), links = new Map();
    const graph = { links, getNodeById: id => byId.get(id) };
    const group = { pos: [0, 0], size: [0, 0], _children: nodes, recomputeInsideNodes() {},
        resizeTo(items, padding) {
            const left = Math.min(...items.map(n => n.boundingRect[0]));
            const top = Math.min(...items.map(n => n.boundingRect[1]));
            const right = Math.max(...items.map(n => n.boundingRect[0] + n.boundingRect[2]));
            const bottom = Math.max(...items.map(n => n.boundingRect[1] + n.boundingRect[3]));
            this.pos = [left - padding, top - padding - 30];
            this.size = [right - left + padding * 2, bottom - top + padding * 2 + 30];
        } };
    return { nodes, group, graph,
        connect(a, b, input = 0, output = 0) {
            const id = links.size;
            while (b.inputs.length <= input) b.inputs.push({ link: null });
            while (a.outputs.length <= output) a.outputs.push({ links: [] });
            b.inputs[input].link = id; a.outputs[output].links.push(id);
            links.set(id, { origin_id: a.id, target_id: b.id, origin_slot: output, target_slot: input });
        },
        arrange() { context.arrange(group, graph); },
    };
}
function safeAndStable(f) {
    const originalSizes = f.nodes.map(n => [...n.size]);
    f.arrange();
    for (const a of f.nodes) {
        assert.ok(a.pos.every(Number.isFinite));
        for (const b of f.nodes) {
            if (a === b) continue;
            const [x, y, w, h] = a.boundingRect, [bx, by, bw, bh] = b.boundingRect;
            assert.ok(x + w <= bx || bx + bw <= x || y + h <= by || by + bh <= y, `Nodes ${a.id} and ${b.id} overlap`);
        }
    }
    const first = f.nodes.map(n => [...n.pos]);
    f.arrange();
    f.nodes.forEach((n, i) => n.pos.forEach((value, axis) => assert.ok(Math.abs(value - first[i][axis]) < 0.001, `Node ${n.id} drifts on repeat: axis ${axis}, ${first[i][axis]} -> ${value}`)));
    assert.deepEqual(f.nodes.map(n => [...n.size]), originalSizes, 'Layout resized controls or previews');
}

const source = new Node(1, 180, 80), sink = new Node(2, 180, 300);
const alignment = fixture([source, sink]); alignment.connect(source, sink);
source.outputs[0].offset = 45; sink.inputs[0].offset = 115;
safeAndStable(alignment);
assert.equal(source.getOutputPos(0)[1], sink.getInputPos(0)[1], 'Actual socket offsets must align');

const shared = new Node(1, 100, 40), upper = new Node(2, 180, 240), lower = new Node(3, 180, 240, 800);
const fanout = fixture([shared, upper, lower]); fanout.connect(shared, upper); fanout.connect(shared, lower);
safeAndStable(fanout);
assert.equal(shared.getOutputPos(0)[1], (upper.getInputPos(0)[1] + lower.getInputPos(0)[1]) / 2, 'Shared provider must center between its targets');

const a = new Node(1, 600, 240), b = new Node(2, 120, 60, 400), c = new Node(3, 120, 240), d = new Node(4, 120, 60, 400);
const branches = fixture([a, b, c, d]); branches.connect(a, c); branches.connect(b, d);
safeAndStable(branches);
assert.ok(d.pos[0] < c.pos[0], 'Small branch should fill space beside the wide preview');
assert.ok(branches.group.size[0] < 600 + 120 + 40 + 50);

const feeders = fixture([new Node(1), new Node(2, 300, 260), new Node(3, 300, 260)]);
feeders.nodes[0].flags.collapsed = true;
feeders.connect(feeders.nodes[0], feeders.nodes[2], 0); feeders.connect(feeders.nodes[1], feeders.nodes[2], 1);
safeAndStable(feeders);
assert.ok(feeders.nodes[0].pos[1] < feeders.nodes[1].pos[1], 'Feeders must follow input order');
assert.equal(feeders.nodes[2].pos[0] - (feeders.nodes[0].pos[0] + 80), 28, 'Collapsed node should sit close to its target');

const compactSize = [...fanout.group.size]; spacing = 'Relaxed';
safeAndStable(fanout);
assert.ok(fanout.group.size[0] > compactSize[0] && fanout.group.size[1] > compactSize[1]);
spacing = 'Compact';

let seed = 1234;
const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
for (let trial = 0; trial < 50; trial++) {
    const nodes = Array.from({ length: 20 }, (_, i) => new Node(i, 80 + Math.floor(random() * 450), 40 + Math.floor(random() * 360), random() * 2000));
    const f = fixture(nodes);
    for (let i = 0; i < nodes.length; i++) {
        nodes[i].flags.collapsed = random() < 0.15;
        for (let j = i + 1; j < nodes.length; j++) if (random() < 0.1) f.connect(nodes[i], nodes[j], nodes[j].inputs.length);
    }
    safeAndStable(f);
}
const cycle = fixture([new Node(1), new Node(2), new Node(3)]);
cycle.connect(cycle.nodes[0], cycle.nodes[1]); cycle.connect(cycle.nodes[1], cycle.nodes[2]); cycle.connect(cycle.nodes[2], cycle.nodes[0]);
safeAndStable(cycle);
console.log('Layout: actual sockets, shared providers, compact branches, titles, collapsed nodes, spacing, cycles, and 50 varied graph stability checks passed.');
