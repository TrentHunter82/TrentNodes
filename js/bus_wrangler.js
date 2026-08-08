import { app } from "../../scripts/app.js";

/**
 * Bus Wrangler
 *
 * A frontend-only "Trent Bus" virtual node plus commands that build one
 * automatically from the current selection. The node never reaches the
 * backend — graphToPrompt resolves through it via getInputLink(slot),
 * the same mechanism the legacy Reroute node uses.
 *
 * Two ways to use the node:
 *   - Standalone multi-reroute: sources → pairs → targets. Each pair
 *     behaves exactly like a legacy Reroute for its own link.
 *   - One-noodle bus: connect one bus's `bus` output to another bus's
 *     `bus` input. The downstream bus mirrors the upstream channels and
 *     resolves each output back through the single bus noodle. A local
 *     link plugged into a mirrored input overrides that channel.
 *
 * Slot layout (index-aligned on both sides):
 *   [0]   bus  (TRENT_BUS) — optional chain to another Trent Bus
 *   [1..] pairs             — input i passes straight through to output i
 *   last  "+"               — spare wildcard pair; connect to grow the bus
 *
 * Unplugging a middle pair resets it to a spare but keeps its position,
 * so live channels never shift index under a downstream mirror. The
 * node's right-click menu has "Bus: remove unused pairs" for a hard
 * compaction (that one DOES shift indices).
 *
 * Commands (also in the TrentNodes menu):
 *   Shift+Alt+B — splice ONE bus inline across every link that enters
 *                 the selected node(s) from outside.
 *   Shift+Alt+N — same, but as an A══B pair joined by a single noodle:
 *                 sources → A ══ B → targets. Drag A off toward the
 *                 sources; only the bus noodle crosses the graph.
 *
 * Known limits: connect the `bus` output only to another Trent Bus.
 * Widget-converted inputs ride the bus like any link, but if one
 * misbehaves at queue time, wire it directly.
 */

const NODE_TYPE = "TrentBus";
const BUS_TYPE = "TRENT_BUS";
const SPARE_NAME = "+";
const WILD = "*";

/** The graph currently shown on the canvas (may be a subgraph). */
function getActiveGraph() {
    return (app.canvas && app.canvas.graph) || app.graph;
}

/** graph.links is a Map proxied to also allow bracket access — use .get. */
function getLink(graph, id) {
    if (id == null || !graph) return null;
    return graph.links.get ? graph.links.get(id) : graph.links[id];
}

function toast(severity, detail) {
    try {
        app.extensionManager?.toast?.add?.({
            severity,
            summary: "Bus Wrangler",
            detail,
            life: 4000,
        });
    } catch (e) { /* toast API missing — console below still fires */ }
    console.log(`[BusWrangler] ${detail}`);
}

/**
 * Inbound links that cross the selection boundary, deduped by source
 * output so a fanned-out source becomes one channel with many targets.
 */
function collectBoundaryInbound(graph, nodes) {
    const sel = new Set(nodes);
    const byChannel = new Map();

    for (const node of nodes) {
        (node.inputs ?? []).forEach((inp, slot) => {
            const link = getLink(graph, inp.link);
            if (!link || link.type === BUS_TYPE) return;
            const origin = graph.getNodeById(link.origin_id);
            if (!origin || sel.has(origin)) return;

            const key = `${link.origin_id}:${link.origin_slot}`;
            let ch = byChannel.get(key);
            if (!ch) {
                ch = {
                    origin,
                    originSlot: link.origin_slot,
                    type: link.type ?? inp.type ?? WILD,
                    name: inp.label ?? inp.name
                        ?? String(link.type ?? WILD).toLowerCase(),
                    targets: [],
                };
                byChannel.set(key, ch);
            }
            ch.targets.push({ node, slot });
        });
    }
    return [...byChannel.values()];
}

/** Give a bus exactly: [bus port, one typed pair per channel, spare]. */
function shapeBus(bus, channels) {
    while (bus.inputs.length > 1) bus.removeInput(bus.inputs.length - 1);
    while (bus.outputs.length > 1) bus.removeOutput(bus.outputs.length - 1);
    for (const ch of channels) {
        bus.addInput(ch.name, ch.type);
        bus.addOutput(ch.name, ch.type);
    }
    bus.addInput(SPARE_NAME, WILD);
    bus.addOutput(SPARE_NAME, WILD);
    bus.size = bus.computeSize();
}

/**
 * Splice a bus across every link entering the selection from outside.
 * pair=false: sources → bus → targets (one node, N noodles both sides).
 * pair=true:  sources → A ══ B → targets (one noodle between A and B).
 */
function spliceBus({ pair }) {
    const LiteGraph = window.LiteGraph;
    const graph = getActiveGraph();
    const nodes = Object.values(app.canvas?.selected_nodes ?? {})
        .filter((n) => n.type !== NODE_TYPE);

    if (!nodes.length) {
        toast("warn", "Select at least one node first.");
        return 0;
    }
    nodes.sort((a, b) => a.pos[1] - b.pos[1] || a.pos[0] - b.pos[0]);

    const channels = collectBoundaryInbound(graph, nodes);
    if (!channels.length) {
        toast("warn", "No links enter the selection from outside.");
        return 0;
    }

    app.canvas?.emitBeforeChange?.();
    try {
        let minX = Infinity, minY = Infinity;
        for (const n of nodes) {
            minX = Math.min(minX, n.pos[0]);
            minY = Math.min(minY, n.pos[1]);
        }

        // busB feeds the targets; the collector takes the sources.
        const busB = LiteGraph.createNode(NODE_TYPE);
        graph.add(busB);
        const busA = pair ? LiteGraph.createNode(NODE_TYPE) : null;
        if (busA) graph.add(busA);
        const collector = busA ?? busB;

        collector._wiring = true;
        busB._wiring = true;
        try {
            shapeBus(collector, channels);
            if (busA) shapeBus(busB, channels);

            channels.forEach((ch, k) => {
                const idx = k + 1;
                ch.origin.connect(ch.originSlot, collector, idx);
                for (const t of ch.targets) busB.connect(idx, t.node, t.slot);
            });
            if (busA) busA.connect(0, busB, 0);
        } finally {
            delete collector._wiring;
            delete busB._wiring;
        }

        busB.pos = [minX - busB.size[0] - 70, minY];
        if (busA) busA.pos = [busB.pos[0] - busA.size[0] - 160, minY];

        toast("success", `Bused ${channels.length} link(s) through `
            + (pair ? "a one-noodle bus pair." : "one bus."));
    } finally {
        app.canvas?.emitAfterChange?.();
    }

    graph.setDirtyCanvas(true, true);
    return channels.length;
}

app.registerExtension({
    name: "TrentNodes.BusWrangler",

    registerCustomNodes() {
        const LiteGraph = window.LiteGraph;
        const LGraphCanvas = window.LGraphCanvas ?? LiteGraph.LGraphCanvas;

        if (LGraphCanvas?.link_type_colors) {
            LGraphCanvas.link_type_colors[BUS_TYPE] = "#c8a24b";
        }

        class TrentBusNode extends LiteGraph.LGraphNode {
            constructor(title) {
                super(title ?? "Trent Bus");
                this.isVirtualNode = true;
                this.addInput("bus", BUS_TYPE);
                this.addOutput("bus", BUS_TYPE);
                this.#addSparePair();
                this.size = this.computeSize();
            }

            /* ---------- prompt-time resolution ---------- */

            /**
             * The whole trick. graphToPrompt walks virtual nodes by
             * calling getInputLink(origin_slot) until it hits a real
             * node. Pairs are index-aligned, so output i resolves to
             * input i — locally if plugged, else through the bus
             * noodle into the upstream bus at the same index.
             */
            getInputLink(slot) {
                if (slot === 0) return null; // the bus noodle carries no value
                if (this._resolving) return null; // bus loop guard
                this._resolving = true;
                try {
                    const inp = this.inputs[slot];
                    if (inp && inp.link != null) {
                        return getLink(this.graph, inp.link);
                    }
                    return this.#upstreamBus()?.getInputLink(slot) ?? null;
                } finally {
                    this._resolving = false;
                }
            }

            /* ---------- dynamic slots ---------- */

            onConnectionsChange(kind, index, connected) {
                if (!this.graph || this._wiring) return;
                if (index === 0) {
                    // Bus chain changed — remirror self, then downstream.
                    if (kind === LiteGraph.INPUT) this.#refreshMirror();
                    this.#syncDownstream();
                    return;
                }
                if (connected && kind === LiteGraph.INPUT) this.#stampPair(index);
                if (connected && kind === LiteGraph.OUTPUT) this.#stampPair(index);
                this.#normalize();
                this.#syncDownstream();
            }

            onConfigure() {
                this._wiring = true;
                try {
                    this.#normalize();
                } finally {
                    delete this._wiring;
                }
            }

            getExtraMenuOptions(_, options) {
                options.push({
                    content: "Bus: remove unused pairs",
                    callback: () => this.#compactHard(),
                });
            }

            /* ---------- helpers ---------- */

            #addSparePair() {
                this.addInput(SPARE_NAME, WILD);
                this.addOutput(SPARE_NAME, WILD);
            }

            #pairEmpty(i) {
                const inp = this.inputs[i], out = this.outputs[i];
                return (!inp || inp.link == null)
                    && (!out || !out.links || out.links.length === 0);
            }

            #isSparePair(i) {
                return this.#pairEmpty(i)
                    && this.inputs[i]?.type === WILD
                    && this.outputs[i]?.type === WILD;
            }

            #upstreamBus() {
                const link = getLink(this.graph, this.inputs[0]?.link);
                const node = link && this.graph.getNodeById(link.origin_id);
                return node && node.type === NODE_TYPE ? node : null;
            }

            /** Copy name/type onto pair i from whichever side just linked. */
            #stampPair(i) {
                const inp = this.inputs[i], out = this.outputs[i];
                if (!inp || !out) return;

                let name = null, type = null;
                const inLink = getLink(this.graph, inp.link);
                const outLink = getLink(this.graph, out.links?.[0]);
                if (inLink) {
                    const origin = this.graph.getNodeById(inLink.origin_id);
                    const slot = origin?.outputs?.[inLink.origin_slot];
                    type = inLink.type ?? slot?.type ?? WILD;
                    name = slot?.label ?? slot?.name ?? null;
                } else if (outLink) {
                    const target = this.graph.getNodeById(outLink.target_id);
                    const slot = target?.inputs?.[outLink.target_slot];
                    type = outLink.type ?? slot?.type ?? WILD;
                    name = slot?.label ?? slot?.name ?? null;
                }
                if (type == null || type === WILD) return;

                inp.type = type;
                out.type = type;
                inp.name = out.name = name || String(type).toLowerCase();
                // Links made while the slot was still "*" keep their old
                // type — retype them so noodle colours match.
                if (inLink) inLink.type = type;
                for (const lid of out.links ?? []) {
                    const l = getLink(this.graph, lid);
                    if (l) l.type = type;
                }
            }

            /**
             * Reset fully-unplugged pairs to spares IN PLACE (indices
             * never shift under a downstream mirror), trim empty
             * trailing pairs, keep exactly one spare, remirror.
             */
            #normalize() {
                for (let i = 1; i < Math.min(this.inputs.length, this.outputs.length); i++) {
                    if (this.#pairEmpty(i) && this.inputs[i].type !== WILD) {
                        this.inputs[i].type = this.outputs[i].type = WILD;
                        this.inputs[i].name = this.outputs[i].name = SPARE_NAME;
                        this.inputs[i].label = this.outputs[i].label = undefined;
                    }
                }
                while (this.inputs.length < this.outputs.length) this.addInput(SPARE_NAME, WILD);
                while (this.outputs.length < this.inputs.length) this.addOutput(SPARE_NAME, WILD);
                while (this.inputs.length > 2
                    && this.#isSparePair(this.inputs.length - 1)
                    && this.#isSparePair(this.inputs.length - 2)) {
                    this.removeInput(this.inputs.length - 1);
                    this.removeOutput(this.outputs.length - 1);
                }
                if (!this.#isSparePair(this.inputs.length - 1)) this.#addSparePair();
                this.#refreshMirror();
                this.size = this.computeSize();
            }

            /** Channels this bus carries (index 0 here = pair slot 1). */
            busChannels(seen = new Set()) {
                if (seen.has(this.id)) return [];
                seen.add(this.id);
                const up = this.#upstreamBus()?.busChannels(seen) ?? [];
                const count = Math.max(up.length, this.inputs.length - 1);
                const chans = [];
                for (let k = 0; k < count; k++) {
                    const inp = this.inputs[k + 1];
                    if (inp && inp.type !== WILD) {
                        chans.push({ name: inp.name, type: inp.type });
                    } else if (up[k]) {
                        chans.push(up[k]);
                    } else {
                        chans.push({ name: SPARE_NAME, type: WILD });
                    }
                }
                while (chans.length && chans[chans.length - 1].type === WILD) chans.pop();
                return chans;
            }

            /** Rename/retype own pairs to match the upstream bus. */
            #refreshMirror() {
                const up = this.#upstreamBus();
                if (!up) return;
                const chans = up.busChannels();
                for (let k = 0; k < chans.length; k++) {
                    const idx = k + 1;
                    while (this.inputs.length <= idx) this.addInput(SPARE_NAME, WILD);
                    while (this.outputs.length <= idx) this.addOutput(SPARE_NAME, WILD);
                    const inp = this.inputs[idx], out = this.outputs[idx];
                    if (inp.link != null) continue; // local override wins
                    inp.name = out.name = chans[k].name;
                    inp.type = out.type = chans[k].type;
                }
                // Extra linkless pairs past the upstream shape decay to spares.
                for (let idx = chans.length + 1; idx < this.inputs.length; idx++) {
                    if (this.#pairEmpty(idx)) {
                        this.inputs[idx].type = this.outputs[idx].type = WILD;
                        this.inputs[idx].name = this.outputs[idx].name = SPARE_NAME;
                    }
                }
                this.size = this.computeSize();
            }

            /** Push channel changes through every connected downstream bus. */
            #syncDownstream(seen = new Set()) {
                if (seen.has(this.id)) return;
                seen.add(this.id);
                for (const lid of this.outputs[0]?.links ?? []) {
                    const link = getLink(this.graph, lid);
                    const t = link && this.graph.getNodeById(link.target_id);
                    if (t && t.type === NODE_TYPE) {
                        t.#refreshMirror();
                        t.#syncDownstream(seen);
                    }
                }
            }

            /** Right-click action: drop dead pairs even mid-list. */
            #compactHard() {
                app.canvas?.emitBeforeChange?.();
                try {
                    for (let i = this.inputs.length - 1; i >= 1; i--) {
                        if (this.#pairEmpty(i)) {
                            this.removeInput(i);
                            if (this.outputs[i]) this.removeOutput(i);
                        }
                    }
                    this.#normalize();
                    this.#syncDownstream();
                } finally {
                    app.canvas?.emitAfterChange?.();
                }
                this.graph?.setDirtyCanvas(true, true);
            }
        }

        LiteGraph.registerNodeType(
            NODE_TYPE,
            Object.assign(TrentBusNode, {
                title: "Trent Bus",
                collapsable: true,
            })
        );
        TrentBusNode.category = "Trent/Routing";
    },

    commands: [
        {
            id: "TrentNodes.BusSplice",
            label: "Bus: Splice Selection (multi reroute)",
            icon: "pi pi-sitemap",
            function: () => spliceBus({ pair: false }),
        },
        {
            id: "TrentNodes.BusPair",
            label: "Bus: Wrap Selection (one-noodle bus pair)",
            icon: "pi pi-share-alt",
            function: () => spliceBus({ pair: true }),
        },
    ],

    keybindings: [
        {
            commandId: "TrentNodes.BusSplice",
            combo: { key: "b", shift: true, alt: true },
        },
        {
            commandId: "TrentNodes.BusPair",
            combo: { key: "n", shift: true, alt: true },
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.BusSplice", "TrentNodes.BusPair"],
        },
    ],
});
