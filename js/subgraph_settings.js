import { app } from "../../scripts/app.js";
import { exposeUnconnectedInputs, exposeUnconnectedOutputs, isEditableWidget, removeUneditablePromotions } from "./source_subgraphs.js";

const COMMAND = "TrentNodes.CreateSettingsSubgraph";
const CONTROLS = "trentLinkedSettings";
const restored = new WeakMap();

export function restoreLinkedSettings(host) {
    if (!host.isSubgraphNode?.()) return;
    for (const node of host.subgraph.nodes) if (node.isSubgraphNode?.()) restoreLinkedSettings(node);
    let linked = restored.get(host);
    if (!linked) { linked = new Set(); restored.set(host, linked); }
    for (const control of host.properties[CONTROLS] ?? []) {
        const key = `${control.nodeId}:${control.widgetName}`;
        if (linked.has(key)) continue;
        const source = host.subgraph.getNodeById(control.nodeId);
        const widget = source?.widgets?.find((w) => w.name === control.widgetName);
        if (!widget) continue;
        const proxy = host.addWidget(widget.type, control.name, widget.value, (value, ...args) => {
            widget.callback?.call(widget, value, args[0] ?? app.canvas, source, ...args.slice(2));
            host.graph?.setDirtyCanvas(true, true);
        }, { ...widget.options, serialize: false });
        proxy.serialize = false;
        Object.defineProperty(proxy, "value", {
            configurable: true,
            get: () => widget.value,
            set: (value) => { widget.value = value; },
        });
        linked.add(key);
    }
    if (linked.size) host.expandToFitContent();
}

function exposeSettings(host, plans) {
    const inside = plans.map((p) => host.subgraph.getNodeById(p.id));
    removeUneditablePromotions(host, plans);
    exposeUnconnectedInputs(host.subgraph, inside);
    exposeUnconnectedOutputs(host.subgraph, inside);
    const controls = host.properties[CONTROLS] ??= [];
    for (const plan of plans) {
        const node = host.subgraph.getNodeById(plan.id);
        for (const name of plan.widgets) {
            const widget = node.widgets.find((w) => w.name === name);
            if (!widget || node.getSlotFromWidget(widget)) continue;
            if (controls.some((c) => c.nodeId === node.id && c.widgetName === name)) continue;
            const label = `${node.title} · ${widget.label || name}`;
            let unique = label;
            for (let i = 2; controls.some((c) => c.name === unique); i++) unique = `${label} ${i}`;
            controls.push({ nodeId: node.id, widgetName: name, name: unique });
        }
    }
    restoreLinkedSettings(host);
    host.subgraph.inputNode.arrange();
    host.subgraph.outputNode.arrange();
    const left = Math.min(...inside.map((n) => n.pos[0]));
    const top = Math.min(...inside.map((n) => n.pos[1])) - window.LiteGraph.NODE_TITLE_HEIGHT;
    const right = Math.max(...inside.map((n) => n.pos[0] + n.size[0]));
    host.subgraph.inputNode.pos = [left - host.subgraph.inputNode.boundingRect[2] - 60, top];
    host.subgraph.outputNode.pos = [right + 60, top];
    host.expandToFitContent();
}

function createSettingsSubgraph() {
    const canvas = app.canvas;
    if (!canvas) return;
    const graph = canvas.graph || app.graph;
    const LG = window.LiteGraph;
    const selected = Array.from(canvas.selectedItems ?? Object.values(canvas.selected_nodes ?? {}));
    const groups = selected.filter((n) => n instanceof LG.LGraphGroup);
    const nodes = new Set(selected.filter((n) => n instanceof LG.LGraphNode));
    for (const group of groups) {
        group.recomputeInsideNodes();
        for (const node of group._children) if (node instanceof LG.LGraphNode) nodes.add(node);
    }
    if (!nodes.size) {
        app.extensionManager.toast.add({ severity: "info", summary: "Select nodes or a group first", life: 3000 });
        return;
    }
    if (groups.some((g) => g.pinned) || [...nodes].some((n) => n.pinned)) {
        app.extensionManager.toast.add({ severity: "info", summary: "Unpin the selection before creating a subgraph", life: 3000 });
        return;
    }
    const existing = nodes.size === 1 && [...nodes][0].isSubgraphNode?.() ? [...nodes][0] : null;
    const sources = existing ? existing.subgraph.nodes : [...nodes];
    if (!sources.length) return;
    const plans = sources.map((node) => ({ id: node.id,
        widgets: (node.widgets ?? []).filter((w) => isEditableWidget(w)
            || (!node.getSlotFromWidget(w) && node.widgets.some((target) => target.linkedWidgets?.includes(w))))
            .filter((w) => !w.hidden && !w.options?.hidden && w.type !== "button")
            .map((w) => w.name),
    }));
    canvas.emitBeforeChange();
    let host = existing;
    try {
        if (!host) {
            host = graph.convertToSubgraph(new Set([...nodes, ...groups])).node;
            if (groups.length === 1) { host.title = groups[0].title; host.subgraph.name = host.title; }
        }
        exposeSettings(host, plans);
        host.updateArea(canvas.ctx);
        canvas.selectedItems = new Set([host]);
    } catch (error) {
        if (host && !existing) graph.unpackSubgraph(host);
        console.error("[CreateSettingsSubgraph]", error);
        app.extensionManager.toast.add({ severity: "error", summary: "Could not expose subgraph settings", detail: error.message, life: 5000 });
    } finally {
        canvas.emitAfterChange();
    }
    graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: COMMAND,
    commands: [{ id: COMMAND, label: "Create Subgraph with Settings", icon: "pi pi-sitemap", function: createSettingsSubgraph }],
    keybindings: [{ commandId: COMMAND, combo: { key: "p", alt: true, shift: true }, targetElementId: "graph-canvas" }],
    menuCommands: [{ path: ["TrentNodes"], commands: [COMMAND] }],
    getSelectionToolboxCommands(item) {
        const LG = window.LiteGraph;
        return item instanceof LG.LGraphNode || item instanceof LG.LGraphGroup ? [COMMAND] : [];
    },
    loadedGraphNode(node) { restoreLinkedSettings(node); },
    nodeCreated(node) { queueMicrotask(() => restoreLinkedSettings(node)); },
});
