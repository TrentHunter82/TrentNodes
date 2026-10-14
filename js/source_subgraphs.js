function getLink(graph, id) {
    return graph.getLink ? graph.getLink(id) : graph.links.get(id);
}

function canMove(node) {
    return node && !node.pinned && !node.isVirtualNode && !node.isSubgraphNode?.();
}

export function findSourceBundles(consumers, sourceScope, graph) {
    const allowed = new Set(sourceScope);
    const incoming = (node) => (node.inputs ?? []).flatMap((input, inputIndex) => {
        const link = input.link != null && getLink(graph, input.link);
        const source = link && graph.getNodeById(link.origin_id);
        return allowed.has(source) && canMove(source) && source !== node
            ? [{ node: source, inputIndex }] : [];
    });
    const depths = new Map();
    function depth(node, visiting = new Set()) {
        if (depths.has(node)) return depths.get(node);
        if (visiting.has(node)) return 0;
        visiting.add(node);
        const value = Math.max(0, ...incoming(node).map((s) => depth(s.node, visiting) + 1));
        visiting.delete(node);
        depths.set(node, value);
        return value;
    }
    for (const node of consumers) depth(node);
    const bundles = [];
    const owners = new Map();
    for (const consumer of [...consumers].sort((a, b) => depths.get(b) - depths.get(a))) {
        if (!canMove(consumer) || owners.has(consumer)) continue;
        const sourceNodes = new Set();
        const sources = incoming(consumer).filter((s) =>
            !sourceNodes.has(s.node) && sourceNodes.add(s.node));
        if (!sources.length) continue;
        const nodes = [...sources.map((s) => s.node), consumer];
        if (!nodes.some((n) => n.inputs?.some((s) => s.link == null))) continue;
        const overlaps = new Set(sources.map((s) => owners.get(s.node)).filter(Boolean));
        // Only merge shared providers, never an earlier bundle's consumer.
        if (sources.some((s) => owners.get(s.node)?.consumers.includes(s.node))) continue;
        const bundle = { consumers: [consumer], sources };
        for (const other of overlaps) {
            bundle.consumers.unshift(...other.consumers);
            bundle.sources.unshift(...other.sources);
            bundles.splice(bundles.indexOf(other), 1);
        }
        const seen = new Set();
        bundle.sources = bundle.sources.filter((s) => !seen.has(s.node) && seen.add(s.node));
        bundles.push(bundle);
        for (const node of [...bundle.consumers, ...bundle.sources.map((s) => s.node)]) owners.set(node, bundle);
    }
    return bundles;
}

export function stackSourcesAbove(consumer, sources, gap = 40, titleHeight = 30) {
    let bottom = consumer.pos[1] - titleHeight - gap;
    for (let i = sources.length - 1; i >= 0; i--) {
        const node = sources[i].node;
        const height = node.flags?.collapsed ? 0 : node.size[1];
        node.pos = [consumer.pos[0], bottom - height];
        bottom = node.pos[1] - titleHeight - gap;
    }
}

function uniqueName(base, slots) {
    const names = new Set(slots.map((s) => s.name));
    let name = base;
    for (let suffix = 2; names.has(name); suffix++) name = `${base}_${suffix}`;
    return name;
}

export function exposeUnconnectedInputs(subgraph, nodes) {
    for (const node of nodes) {
        for (const input of node.inputs ?? []) {
            // Conversion may already have promoted a recommended widget.
            if (input.link != null) continue;
            const widget = node.getWidgetFromSlot?.(input);
            if (widget?.hidden || widget?.options?.hidden) continue;
            const name = uniqueName(input.name, subgraph.inputs);
            const boundary = subgraph.addInput(name, String(input.type));
            boundary.label = input.label;
            if (!boundary.connect(input, node)) {
                subgraph.removeInput(boundary);
                throw new Error(`Cannot expose ${node.title}: ${input.name}`);
            }
        }
    }
}

function exposeConsumerOutputs(subgraph, consumer) {
    for (const [index, output] of (consumer.outputs ?? []).entries()) {
        const exposed = subgraph.outputs.some((s) => (s.linkIds ?? []).some((id) => {
            const link = getLink(subgraph, id);
            return link?.origin_id === consumer.id && link.origin_slot === index;
        }));
        if (exposed) continue;
        const name = uniqueName(output.name || `output_${index + 1}`, subgraph.outputs);
        const boundary = subgraph.addOutput(name, String(output.type));
        if (!boundary.connect(output, consumer)) {
            subgraph.removeOutput(boundary);
            throw new Error(`Cannot expose ${consumer.title}: ${output.name}`);
        }
    }
}

function arrangeBoundaryNodes(subgraph, nodes, titleHeight) {
    const left = Math.min(...nodes.map((n) => n.pos[0]));
    const top = Math.min(...nodes.map((n) => n.pos[1] - titleHeight));
    const right = Math.max(...nodes.map((n) => n.pos[0] + n.size[0]));
    subgraph.inputNode.arrange();
    subgraph.outputNode.arrange();
    subgraph.inputNode.pos = [left - subgraph.inputNode.boundingRect[2] - 60, top];
    subgraph.outputNode.pos = [right + 60, nodes.at(-1).pos[1]];
}

export function packSourceBundle(graph, bundle, gap = 40, titleHeight = 30) {
    const originals = [...bundle.sources.map((s) => s.node), ...bundle.consumers];
    const { subgraph, node } = graph.convertToSubgraph(new Set(originals));
    try {
        const consumers = bundle.consumers.map((n) => subgraph.getNodeById(n.id));
        const consumer = consumers[0];
        const sources = bundle.sources.map((s) => ({
            ...s, node: subgraph.getNodeById(s.node.id),
        }));
        const inside = [...sources.map((s) => s.node), ...consumers];
        exposeUnconnectedInputs(subgraph, inside);
        for (const n of consumers) exposeConsumerOutputs(subgraph, n);
        let left = consumer.pos[0];
        for (const n of consumers) {
            n.pos = [left, consumer.pos[1]];
            left += n.size[0] + gap;
        }
        stackSourcesAbove(consumer, sources, gap, titleHeight);
        arrangeBoundaryNodes(subgraph, inside, titleHeight);
        subgraph.name = consumer.title;
        node.title = consumer.title;
        node.expandToFitContent();
        return node;
    } catch (error) {
        graph.unpackSubgraph(node);
        throw error;
    }
}
