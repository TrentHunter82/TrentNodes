// Delete-time link bridging.
//
// deleteSelected() calls node.connectInputToOutput() before removing the node: each
// connected input whose type matches an output is bridged straight from the input's origin
// to the node's consumers -- with plain connect() calls, so none of the LinkConnector events
// the drag pipeline records provenance from ever fire.
//
// Two consequences for pass-throughs, one cure: a bridged link INHERITS the removed node's
// provenance for the input it came from. If that input was fed via a pin (provenance
// recorded on the node being removed), the bridged wire re-anchors onto the same pin -- the
// deleted node's apparent upstream sibling. If it was fed by an owned primitive with no
// record to inherit, provenance derives from the primitive's ownership -- otherwise the wire
// would draw from a node that is deliberately invisible.
//
// Patch the PROTOTYPE (same rule as the ledger): node instances are per-type subclasses,
// but connectInputToOutput lives on the LGraphNode base.

import { isOwned } from "./primitive.js";
import { recordProvenance, provenanceOf } from "./reanchor.js";
import { nodeById } from "./graph.js";
import { invalidate } from "./ledger.js";
import { isEnabled } from "./enabled.js";

const HOST = "cablemanagement.host";
const HOST_SLOT = "cablemanagement.hostSlot";

let installed = false;

function basePrototype(app) {
  const viaGlobal = window.LiteGraph?.LGraphNode?.prototype;
  if (typeof viaGlobal?.connectInputToOutput === "function") return viaGlobal;
  // Fallback: walk up from a live node until the defining prototype appears.
  let p = app?.graph?.nodes?.[0] ? Object.getPrototypeOf(app.graph.nodes[0]) : null;
  while (p && !Object.prototype.hasOwnProperty.call(p, "connectInputToOutput")) {
    p = Object.getPrototypeOf(p);
  }
  return p;
}

export function install(app) {
  if (installed) return true;
  const proto = basePrototype(app);
  if (!proto || proto.__cablemanagementBridge) return !!proto;
  const orig = proto.connectInputToOutput;
  proto.__cablemanagementBridge = true;

  proto.connectInputToOutput = function (...args) {
    const graph = this.graph;
    if (!graph?.state || !isEnabled()) return orig.apply(this, args);

    // Which provenance should a bridge from each of this node's input origins inherit?
    const inherit = new Map(); // "originId:originSlot" -> [hostId, hostIndex]
    for (const [i, input] of (this.inputs ?? []).entries()) {
      if (input?.link == null) continue;
      const link = graph.getLink ? graph.getLink(input.link) : graph._links?.get?.(input.link);
      if (!link) continue;
      let prov = provenanceOf(this, i, graph);
      if (!prov) {
        const origin = nodeById(graph, link.origin_id);
        if (isOwned(origin)) {
          prov = [origin.properties[HOST], origin.properties[HOST_SLOT]];
        }
      }
      if (prov) inherit.set(`${link.origin_id}:${link.origin_slot}`, [prov[0], prov[1]]);
    }

    const before = Number(graph.state.lastLinkId ?? 0);
    const result = orig.apply(this, args);
    const after = Number(graph.state.lastLinkId ?? 0);

    let bridged = false;
    for (let id = before + 1; id <= after; id++) {
      const link = graph.getLink ? graph.getLink(id) : graph._links?.get?.(id);
      if (!link) continue;
      bridged = true;
      const prov = inherit.get(`${link.origin_id}:${link.origin_slot}`);
      const host = prov && nodeById(graph, prov[0]);
      const target = nodeById(graph, link.target_id);
      if (host && target) {
        recordProvenance(target, link.target_slot, host, Number(prov[1]), link.id);
      }
    }
    if (bridged) {
      // The target's row may gain its link with no watched DOM attribute flipping, and the
      // ledger cache predates the new link ids.
      invalidate();
      window.dispatchEvent(new CustomEvent("cablemanagement:resync"));
    }
    return result;
  };

  installed = true;
  return true;
}
