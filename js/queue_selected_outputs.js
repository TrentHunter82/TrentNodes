import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Queue Selected Output Nodes Extension
 *
 * Shift+Alt+Q  - Queue only the selected output nodes, plus everything
 *                upstream that feeds them. The rest of the graph is dropped.
 *
 * Selection follows the same rules as the core selection toolbox: selected
 * groups contribute their child nodes, and a selected subgraph node
 * contributes the output nodes inside it.
 *
 * Implementation notes
 * --------------------
 * The prompt is pruned on the client, the same way rgthree does it: we wrap
 * api.queuePrompt, and while a "queue selected" run is in flight we replace
 * prompt.output with the dependency closure of the chosen output nodes.
 *
 * We deliberately do NOT use the core Comfy.QueueSelectedOutputNodes command
 * or the backend partial_execution_targets field. Those send the whole prompt
 * and ask the server to run part of it, so a broken or half-wired branch
 * elsewhere in the graph can still block the run. Pruning on the client drops
 * those nodes from the request entirely.
 *
 * We still go through app.queuePrompt rather than posting to the API
 * ourselves, so widget beforeQueued hooks keep running -- that is what
 * advances control_after_generate seeds.
 */

/** LiteGraph modes that stop a node from producing anything. */
const MUTED = 2;
const BYPASSED = 4;

/** Stops a malformed subgraph from recursing forever. */
const MAX_SUBGRAPH_DEPTH = 16;

/**
 * Targets for the run currently in flight, or null when we are not driving
 * the queue. Read by the api.queuePrompt wrapper below.
 */
let pendingTargets = null;

/** Set by the wrapper so the command can tell whether pruning actually ran. */
let prunedThisRun = 0;

function toast(severity, summary, detail) {
    app.extensionManager?.toast?.add({ severity, summary, detail, life: 5000 });
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

function isNode(item) {
    return typeof LGraphNode !== "undefined"
        ? item instanceof LGraphNode
        : Boolean(item?.constructor?.nodeData);
}

function isGroup(item) {
    return typeof LGraphGroup !== "undefined" && item instanceof LGraphGroup;
}

function getGroupChildren(group) {
    return Array.from(group.children ?? group._children ?? []).filter(isNode);
}

function isOutputNode(node) {
    if (node.mode === MUTED || node.mode === BYPASSED) return false;
    return Boolean(node.constructor?.nodeData?.output_node);
}

/**
 * Selected items, merging both selection APIs. Older frontends keep
 * canvas.selected_nodes (id -> node); newer ones keep canvas.selectedItems,
 * a Set that also holds groups and reroutes.
 */
function getSelectedItems() {
    const canvas = app.canvas;
    if (!canvas) return [];

    const items = new Set();
    for (const item of canvas.selectedItems ?? []) {
        if (item) items.add(item);
    }
    for (const node of Object.values(canvas.selected_nodes ?? {})) {
        if (node) items.add(node);
    }
    return Array.from(items);
}

/**
 * Flatten the selection into output nodes, tagged with the chain of subgraph
 * node ids we walked through to reach them. That chain is what turns a local
 * node id into the colon-joined id the backend uses, e.g. "12:5".
 */
function collectOutputTargets(node, path, found, depth = 0) {
    if (node.mode === MUTED || node.mode === BYPASSED) return;

    if (node.isSubgraphNode?.() && node.subgraph) {
        if (depth >= MAX_SUBGRAPH_DEPTH) {
            console.warn(
                "[TrentNodes] Queue Selected: subgraph nesting is too deep, "
                + "stopped descending at",
                `${node.type}#${node.id}`,
            );
            return;
        }
        const inner = [...path, node.id];
        for (const child of node.subgraph.nodes ?? []) {
            collectOutputTargets(child, inner, found, depth + 1);
        }
        return;
    }

    if (!isOutputNode(node)) return;

    const key = [...path, node.id].join(":");
    if (!found.has(key)) found.set(key, { node, key });
}

/** Every output node the current selection implies, deduped. */
function getSelectedOutputTargets() {
    const found = new Map();

    for (const item of getSelectedItems()) {
        if (isGroup(item)) {
            for (const child of getGroupChildren(item)) {
                collectOutputTargets(child, [], found);
            }
        } else if (isNode(item)) {
            collectOutputTargets(item, [], found);
        }
    }
    return Array.from(found.values());
}

// ---------------------------------------------------------------------------
// Prompt pruning
// ---------------------------------------------------------------------------

/**
 * Map a target onto the keys it owns in prompt.output.
 *
 * The colon-joined key is right whenever we reached the node by walking down
 * from a selected subgraph node. When the user is editing inside a subgraph
 * we never see that path, so fall back to a suffix match. A subgraph that is
 * instantiated more than once can match several keys; we keep them all rather
 * than guess, which over-queues instead of running the wrong instance.
 */
function resolvePromptIds(target, output) {
    if (output[target.key] !== undefined) return [target.key];

    const suffix = `:${target.node.id}`;
    return Object.keys(output).filter((key) => key.endsWith(suffix));
}

/**
 * Copy `id` and everything it depends on from `source` into `dest`.
 * Prompt inputs that are links look like [sourceNodeId, slotIndex].
 */
function collectWithDependencies(id, source, dest) {
    if (dest[id] !== undefined) return dest;

    const node = source[id];
    if (node === undefined) return dest;

    dest[id] = node;
    for (const value of Object.values(node.inputs ?? {})) {
        if (Array.isArray(value)) {
            collectWithDependencies(String(value[0]), source, dest);
        }
    }
    return dest;
}

function pruneOutput(output, targets) {
    const pruned = {};
    const unresolved = [];

    for (const target of targets) {
        const ids = resolvePromptIds(target, output);
        if (!ids.length) {
            unresolved.push(target);
            continue;
        }
        for (const id of ids) collectWithDependencies(id, output, pruned);
    }

    if (unresolved.length) {
        console.warn(
            "[TrentNodes] Queue Selected: these nodes are not in the prompt, "
            + "so they were skipped.",
            unresolved.map((t) => `${t.node.type}#${t.key}`),
        );
    }
    return pruned;
}

// ---------------------------------------------------------------------------
// api.queuePrompt wrapper
// ---------------------------------------------------------------------------

const HOOK_FLAG = "__trentNodesQueueSelectedHook";

function installQueueHook() {
    if (api[HOOK_FLAG]) return;
    api[HOOK_FLAG] = true;

    const originalQueuePrompt = api.queuePrompt;

    api.queuePrompt = async function (number, prompt, ...rest) {
        if (pendingTargets?.length && prompt?.output) {
            const pruned = pruneOutput(prompt.output, pendingTargets);
            const kept = Object.keys(pruned).length;

            if (kept) {
                console.log(
                    `[TrentNodes] Queue Selected: sending ${kept} of `
                    + `${Object.keys(prompt.output).length} node(s).`,
                );
                prompt.output = pruned;
                prunedThisRun++;
            } else {
                console.warn(
                    "[TrentNodes] Queue Selected: nothing resolved to a prompt "
                    + "node, so the full graph was queued instead.",
                );
            }
        }
        return originalQueuePrompt.call(api, number, prompt, ...rest);
    };
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

async function queueSelectedOutputs() {
    const targets = getSelectedOutputTargets();

    if (!targets.length) {
        toast(
            "error",
            "Nothing to queue",
            getSelectedItems().length
                ? "The selection has no active output node. Muted and bypassed nodes do not count."
                : "Select at least one output node first.",
        );
        return;
    }

    const batchCount = app.extensionManager?.queueSettings?.batchCount ?? 1;

    prunedThisRun = 0;
    pendingTargets = targets;
    try {
        await app.queuePrompt(0, batchCount);
        if (!prunedThisRun) {
            console.warn(
                "[TrentNodes] Queue Selected: the prompt was never pruned. "
                + "Another queue submission was probably already in flight.",
            );
        }
    } catch (e) {
        console.error("[TrentNodes] Queue Selected failed.", e);
        toast("error", "Failed to queue", String(e?.message ?? e));
    } finally {
        pendingTargets = null;
    }
}

app.registerExtension({
    name: "TrentNodes.QueueSelectedOutputs",

    async setup() {
        installQueueHook();
    },

    commands: [
        {
            id: "TrentNodes.QueueSelectedOutputs",
            label: "Queue Selected Output Nodes",
            icon: "pi pi-play",
            function: queueSelectedOutputs,
        },
    ],

    keybindings: [
        {
            commandId: "TrentNodes.QueueSelectedOutputs",
            combo: { key: "q", shift: true, alt: true },
        },
    ],

    menuCommands: [
        {
            path: ["TrentNodes"],
            commands: ["TrentNodes.QueueSelectedOutputs"],
        },
    ],
});
