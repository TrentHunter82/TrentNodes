# H3 Chain Step + Chain Audio — handoff

Written 2026-09-29 for a fresh Claude Code session. The work was done in
Codex Desktop on 2026-09-23, not in Claude Code, so there is no Claude
transcript to resume. Everything below was reconstructed from the Codex
rollout, the committed code, and the Codex task outputs.

## What it is

Two nodes in `nodes/h3_chain_batch.py` (commit `900ba31`, on `main` and
`origin/main`) that let a VHS Meta Batch Manager drive an exact number of
MiniMax H3 video extensions and keep the audio of every extension for
the final mux.

- **H3 Chain Step** (`H3ChainStep`, category `Trent/MiniMax`)
  - Inputs: `meta_batch` (VHS_BatchManager), `chunks`, `seed`.
    Hidden: `prompt`, `unique_id`.
  - Outputs: `step` (0-based extension index), `seed` (base seed + step,
    wrapped at 2^64).
  - Reads the manager's `requeue` counter out of the prompt dict to learn
    which pass this is. Replaces the driver video that VHS normally uses
    for termination: sets `total_frames = chunks * frames_per_batch` and
    `has_closed_inputs` on the last pass.
  - On step 0 it calls `meta_batch.reset()` and re-stamps `unique_id`.
    VHS resets its ID after finalizing, and a cached manager can leak into
    a manually re-queued job with identical widgets.
  - Raises if another VHS loader is attached (`meta_batch.inputs` non-empty),
    if the step exceeds `chunks`, or if the manager identity changed.
- **H3 Chain Audio + Save Barrier** (`H3ChainAudio`, OUTPUT_NODE)
  - Inputs: `audio`, `meta_batch`, `step`, `saved_tail` (STRING, the path
    from Save Latent).
  - Keeps a CPU float32 copy of every chunk in `self.chunks`. On the last
    pass it returns the concatenated track; on earlier passes it returns
    only the latest chunk (VHS ignores audio until the final pass).
  - `saved_tail` exists only as an execution-order barrier: the node
    refuses to run until Save Latent has produced a path, so the tail is
    on disk before VHS re-queues.
  - Raises on lost chunks, sample-rate change, or channel change. It also
    refuses to resume a partial batch.

Both nodes use `IS_CHANGED = nan` so they run on every re-queue.

## Tests

```bash
cd /home/trent/ComfyUI/custom_nodes/TrentNodes
PYTHONPATH=nodes /home/trent/ComfyUI/venv/bin/python -m unittest discover -s tests -p test_h3_chain_batch.py -v
```

8 tests, all pass as of 2026-09-29 (counting, seed wrap, audio order,
CPU ownership, fresh-run reset, cached-manager recovery, incompatible
drivers, missing or inconsistent audio). They use a `SimpleNamespace`
stand-in for the VHS manager, so they do not exercise real VHS.

Codex also ran queue integration tests against real VHS, MMH3TrimAV and
Motion Context save/load (three passes, 306 frames, 400/600/800 Hz tones
in order). Those scripts live in the Codex work folder, not in the repo.

## Workflows

Tested pair (the ones to use):

- `user/default/workflows/H3 Chain Tested/H3 Chain - Primer.json`
- `user/default/workflows/H3 Chain Tested/H3 Chain - Fixed.json`

Older pair from the 2026-08-31 Claude session (pre-Codex, superseded):

- `user/default/workflows/trent-h3-metabatch-chain/H3 Chain 0 - Primer.json`
- `user/default/workflows/trent-h3-metabatch-chain/H3 Chain 1 - MetaBatch Extend.json`

Run order:

1. Queue the Primer once. It writes the first clip plus a 39-frame AV
   latent tail via `MiniMaxH3MotionContextSaveLatent` with prefix
   `h3_context/chain`.
2. Load Fixed. Set the continuation prompt and `chunks`. Queue once. The
   manager re-queues itself `chunks` times.
3. The final `-audio.mp4` from VHS Video Combine has the accumulated
   soundtrack. VHS also writes a silent intermediate MP4. The primer stays
   a separate file; concatenate it with the body for the full video.

Numbers at the supplied settings: 141 frames per pass, 39 context frames
trimmed, 102 new frames = 4.25 s per chunk. Primer = 5.875 s.

Nodes the Fixed workflow depends on, by pack:

- `ComfyUI-H3-Motion-Context-MultiRef`: MotionContextLoadLatent,
  MotionContextSaveLatent, MotionContextTrim, GeneratedAVMaskedContext.
- `ComfyUI-MMH3Tools`: MMH3TrimAV.
- TrentNodes: H3ChainStep, H3ChainAudio, MiniMaxH3RepackAVLatent.
- VideoHelperSuite: VHS_BatchManager, VHS_VideoCombine.
- Core: MiniMaxH3ImageToVideo, SigmaShift, SamplerCustomAdvanced, etc.

Rules baked into the graph: one batch manager, one VHS Video Combine, no
VHS video loader on that manager, keep the step-to-RandomNoise and
saved_tail-to-audio links in place. Use a separate latent prefix per
independent chain and update both Load Latent and Save Latent.

## Render results (from the Codex session)

- 141-frame primer + 3 automatic extensions at 1344x768, 24 fps, 20
  steps: all completed, loop stopped after exactly 3, 306 frames with
  audio through all chunks.
- First version showed a camera cut at the third join. AV masking kept
  the prefix numerically (max video err 2.38e-7) but the camera still
  changed after the prefix.
- Feeding a decoded frame from the start of the copied tail into H3's
  `first_frame` input removed the cut. Mean pixel diff at the join
  dropped from 83.70 to 6.32 (median of nearby frame diffs 6.09). This
  is the approach in `H3 Chain - Fixed.json`.
- Framing still drifts later inside the last extension. Joins are hard
  cuts; no crossfade.
- Longer chains and other scenes were not quality-tested.

Demo video and comparison clips:
`C:\Users\Trent\Documents\Codex\2026-09-23\can-you-look-this-over-to\outputs\`
(`h3-test-complete.mp4` is the 18.625 s demo, `comparisons/` has the
guide-chain and masked-chain variants with join PNGs and JSON reports).

## Open items

Resolved 2026-09-29 (Claude Code session):

- **Docs and README.** `README.md` now has a `Trent/MiniMax` section
  covering H3 Chain Step, H3 Chain Audio + Save Barrier, and H3 Repack
  AV Latent, plus a NEW blurb at the top. Source text was the Codex
  README in the outputs folder.
- **Version bump.** `pyproject.toml` is 1.20.0.
- **Output folders.** The running server uses the default
  `/home/trent/ComfyUI/output`. Save Latent writes through
  `folder_paths.get_save_image_path`, so the Primer creates
  `output/h3_context/chain_00001_.safetensors` on its first run and the
  Fixed workflow's Load Latent (`h3_context`, clip_index 0) picks the
  newest file there. The folder is missing only because the Primer has
  not run on this server instance yet. Both nodes are confirmed loaded
  in the running server via `/object_info`.

Still open:

1. **Next experiment Codex proposed.** Keep first-frame guidance and the
   protected AV context, and add the previous batch's last frame through
   MiniMax H3 Add Guide at frame 38 (the first generated frame after the
   39 context frames). Compare same-seed against the current result. It
   may tighten the handoff but will not stop later drift. Needs GPU
   renders and a visual judgement, so it was not started autonomously.
2. **Crossfade at joins** is not implemented. Note that the masked AV
   context copies the previous tail exactly (max err 2.38e-7), so there
   is no overlap of *generated* content to blend. A crossfade would need
   the design changed so the new pass regenerates some of the previous
   tail instead of copying it.
3. **Cancel mid-chain** leaves the VHS video unfinalized. A fresh queue
   resets encoder and audio state but does not restore an earlier tail.
   Per-clip backups are the only rollback. Load/Save Latent already
   support fixed `clip_index` slots; driving those from `step` would
   give per-clip rollback, but it changes the workflow contract and the
   Primer's Save slot, so it is a design decision for Trent.
4. **Re-queue semantics.** With Load Latent on "newest", a second queue
   of the Fixed workflow continues from the *last extension's* tail, not
   the primer's, while H3 Chain Audio and VHS start a new file. That is
   how the Codex README describes it ("continues from its most recent
   tail"). Decide whether that is the intended behaviour before wiring
   fixed slots.

## 2026-10-02: slots and Ref2VA pair (Claude Code session)

- Review against VHS `BatchManager`, `requeue_workflow` and Video Combine
  found no defects in either node.
- `H3ChainStep` gained two outputs after `seed`: `load_slot` (step + 1)
  and `save_slot` (step + 2). The existing workflows keep working because
  the new outputs come last. Items 3 and 4 above are settled for the
  slot workflows: the primer owns slot 1, every queue restarts from it,
  and pass k's tail stays in slot k + 2 for a manual rollback.
- New workflows in `user/default/workflows/H3 Chain Tested/`, built from
  the tested pair by `build_ref_chain.py` (session scratchpad, not kept):
  - `H3 Ref Chain - Primer.json`: ref2va checkpoint, MiniMax H3 Reference
    to Video with one `<Picture 1>` (`input/charref-ptlbxBsb.png`), Ref2VA
    six-section prompt, saves the tail to slot 1 of
    `output/h3_context/ref_chain/`.
  - `H3 Ref Chain - Extend.json`: same refs; the first frame of the loaded
    tail goes through Add Guide at `frame_idx` 0 (Reference to Video has
    no `first_frame` input); Load/Save `clip_index` linked to the slot
    outputs.
  - Ref2V turbo 8-step LoRA is in place but bypassed. Turbo LoRAs are
    reported to weaken ref mode; test at 20 steps without it first.
- 2026-10-03: `H3ChainStep` gained an optional `primer_pass` BOOLEAN and
  a fifth output `primer`. With `primer_pass` on, passes = chunks + 1,
  `load_slot` = step, `save_slot` = step + 1, and `primer` is true on
  pass 0 only. `H3 Ref Chain - One Queue.json` uses it: a second Reference
  to Video node holds the primer prompt, and three core `ComfySwitchNode`
  (lazy) nodes pick latent, conditioning and trim (PrimitiveInt 0) on pass
  0. The tail loader, masked context, continuation prompt and Add Guide
  only run on pass 1 and later. Output prefix `h3_chain/ref_one_queue`.
  Validates except for the new outputs (server not restarted); the
  validator's MatchType warnings are a checker limitation.
- 2026-10-14: `H3 Ref Chain - One Queue + Refiner.json` adds noEmbryo's
  `H3ClipRefiner` (statistical half only, strength 1.0, context "39")
  after the sampler. Reference = primer tail from slot 1 through a lazy
  switch (nothing on pass 0, so the refiner only measures there). The
  refiner returns its own `_AVStreamPair`, which core VAE Decode and
  MMH3TrimAV reject, so its output goes through core
  `LTXVSeparateAVLatent` -> `LTXVConcatAVLatent` to become a NestedTensor
  again. CPU test with H3-shaped tensors passed; server validation passes
  with 0 errors. Not rendered. Compared against Endless MiniMax H3 v1.6
  (`user/default/workflows/Endless-H3/`): manual one-clip-per-queue,
  Niko conditioning-row context (22 frames), full-clip saves, crossfading
  stitcher.
- Status: Primer validates clean against the server. Extend validates
  clean except for the two slot links, because the server still had the
  two-output node loaded; restart ComfyUI. Neither workflow has been
  rendered.

## Where the original conversation is

Codex Desktop, thread "Can you look this over to...", 2026-09-23.
Rollout file:
`C:\Users\Trent\.codex\sessions\2026\09\23\rollout-2026-09-23T06-00-47-01a0cdec-deca-7a11-a031-335811f28a14.jsonl`
(WSL path `/mnt/c/Users/Trent/.codex/sessions/2026/09/23/`). The other
Sep 23 rollouts in that folder that mention the node are approval-review
side sessions with no new content.

Related memory notes: `h3-metabatch-chain-build.md` (the 2026-08-31
Claude-built predecessor), `h3-context-window-workflows.md`,
`trent-minimax-h3-cache-node.md` (never share a model branch with an H3
cache node).
