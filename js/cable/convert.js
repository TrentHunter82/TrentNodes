// Keeping the pass-through picture honest across "Convert to Subgraph".
//
// Two things go wrong without help (both measured):
//
//   1. An owned primitive's HOST gets converted. The primitive is hidden machinery
//      parked in the outer graph; with its host gone the reaper removes it and core
//      cascades every consumer link away -- silent data loss. The primitive IS the
//      host widget's value, so it belongs WITH the host: adopt it into the
//      conversion set. Core's own boundary handling then does the rest -- the
//      primitive lands inside beside its host (mirroring keeps working), each
//      consumer link becomes a boundary crossing, and outside consumers end up fed
//      from a new subgraph output. Live value, no bespoke rewiring.
//
//   2. An outside consumer's provenance names a converted host. The record dies
//      with the host (prune forgets it) and the hidden true link is revealed
//      between the subgraph's siblings. The subgraph node inherits the picture
//      instead: the host's fed input became a subgraph input whose gate slot index
//      equals the node's input index -- remap the record onto the subgraph node's
//      own pin there. Node ids and crossing-link ids both survive conversion, so
//      the mapping is exact and the record's link-id check still holds.

import { isOwned } from "./primitive.js";
import { isEnabled } from "./enabled.js";
import { recordProvenance } from "./reanchor.js";
import { nodeById } from "./graph.js";
import { invalidate } from "./ledger.js";

const FROM = "cablemanagement.from";
const HOST = "cablemanagement.host";

/** Self-guarding; retried from each sync until the prototype exists. */
export function install() {
  const LG = window.LiteGraph?.LGraph;
  if (!LG?.prototype?.convertToSubgraph || LG.prototype.__cablemanagementConvert) return;
  LG.prototype.__cablemanagementConvert = true;
  const orig = LG.prototype.convertToSubgraph;

  LG.prototype.convertToSubgraph = function (items) {
    if (!isEnabled()) return orig.call(this, items);
    const graph = this;
    const snapshot = [];
    try {
      // Core expands a lone group into its members AFTER receiving the set; do it
      // first so adoption sees the real membership. Core's own expansion then
      // skips (it only fires on a single-item set) and ends at the same shape.
      const first = items?.size === 1 ? [...items][0] : null;
      if (first && typeof first.recomputeInsideNodes === "function") {
        first.recomputeInsideNodes();
        items = new Set([first, ...(first.children ?? [])]);
      }

      for (const n of graph.nodes ?? []) {
        if (!isOwned(n) || items.has(n)) continue;
        const host = nodeById(graph, n.properties?.[HOST]);
        if (host && items.has(host)) items.add(n);
      }

      // Provenance held by nodes STAYING outside, naming a host that converts.
      for (const t of graph.nodes ?? []) {
        const map = t.properties?.[FROM];
        if (!map || items.has(t)) continue;
        for (const k of Object.keys(map)) {
          const [hostId, hostIdx, linkId] = map[k];
          const host = nodeById(graph, hostId);
          if (host && items.has(host)) {
            snapshot.push({ t, k: Number(k), hostId: String(hostId), hostIdx, linkId });
          }
        }
      }
    } catch {
      /* fixups are cablemanagement's problem; the conversion itself must proceed untouched */
    }

    const result = orig.call(this, items);

    try {
      const { subgraph, node } = result ?? {};
      for (const s of subgraph && node ? snapshot : []) {
        // Which gate slot inherited the host's fed input? Inner ids are
        // preserved, so the gate's inner link still names (host, slot) exactly.
        // No match means the host's input was fed from INSIDE the set -- the
        // value now emerges from a subgraph output and the honest draw is right.
        let gi = -1;
        (subgraph.inputs ?? []).forEach((si, i) => {
          if (gi >= 0) return;
          for (const lid of si.linkIds ?? []) {
            const l = subgraph.getLink(lid);
            if (l && String(l.target_id) === s.hostId && l.target_slot === s.hostIdx) {
              gi = i;
              return;
            }
          }
        });
        // Only while the consumer still holds the exact link the record names.
        if (gi >= 0 && s.t.inputs?.[s.k]?.link === s.linkId) {
          recordProvenance(s.t, s.k, node, gi, s.linkId);
        }
      }
      if (snapshot.length) invalidate();
    } catch {
      /* same: a cosmetic remap must never break a finished conversion */
    }
    return result;
  };
}
