# H3 Pika Video Chain + Clip Refiner — handoff

Session 2026-11-07 → 2026-11-11 (Claude Code). Transcript:
`/home/trent/.claude/projects/-home-trent-ComfyUI-custom-nodes/dddc8277-a51a-4b53-bceb-6ca544b8dc7d.jsonl`.
Earlier chain history: `docs/H3_CHAIN_BATCH_HANDOFF.md`. Memory notes: `geiru-chain-integration`,
`fused-turbo-mystic-model-source`, `pika-h3-flex-setup`.

## TL;DR

- Problem: in Trent's 10-clip two-pass chain, colour drifts clip by clip (saturation climbs, image darkens,
  sky shrinks). Baseline saturation rise over 10 clips: **+50**.
- Fix that works: **noEmbryo H3ClipRefiner** (statistical mode) on both passes, reference = **clip 1's latent**.
  Saturation rise drops to **+14**; adding the latent pin gets **+8**.
- Trent watched the outputs: "the one with the clip refiner is by far the best I've seen."
- Deliverable: clean colour-coded workflow
  `user/default/workflows/Pika-H3/H3 Pika Video Chain + Clip Refiner (Trent).json`
  (copy in `C:\Users\Trent\Downloads`). Not committed anywhere — it lives only in the user workflow folder.
- All TrentNodes nodes it uses are already pushed (last chain commit `8e2dc29` Add Pin First Frame From Latent).

## The pipeline (what one queue run does)

One queue = one clip. VHS Meta Batch Manager (`frames_per_batch` 141) re-queues; H3VideoChain counts clips.

1. **Models**: UNET `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot` (MATLOWAI fused turbo
   Mystic, *not* a local merge) → comfy kitchen attention → Pika H3 Flex fl2va LoRA (1.0).
   Pass 2 branch adds Open-H3 2K refiner LoRA (rank 16) + BlockSparseAttention sol-attn.
   TE `qwen3vl_32b_minimax_h3_nvfp4_awq`, video VAE LynnReal light int8, audio VAE fp32,
   latent upscaler `minimax_h3_latent_upscaler_3d_conv_v1_fp16`.
2. **Prompt per clip**: `FIXED PROMPTS` multiline string, 10 Ref2VA prompts separated by `\n=====\n`.
   `ComfyMathExpression` `a - b + 1` (a = chain seed, b = Seed primitive) → clip number →
   `RegexExtract` group_index. **Use the math node's output slot 1 (INT); slot 0 is FLOAT** (bug hit once).
3. **Pass 1** (960x544): Continue From Last Clip (low res) → H3 conditioning → er_sde, Pika 6 sigmas
   `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0` → **Clip Refiner (low res)** →
   Remember Clip Ending (low res).
4. **Upscale**: MiniMaxH3LatentUpscaleWithModel → 1920x1088.
5. **Pass 2** (2K): Continue From Last Clip (high res) → er_sde, sigmas `0.7, 0.494118, 0.0`, seed 42 fixed →
   **Clip Refiner (high res)** → Remember Clip Ending (high res).
6. **Output**: VAE decode + audio decode → per-clip VideoCombine and H3JoinClips → final mp4
   (`h3_chain/pika_chain_clip_refiner`).

Chain settings: H3VideoChain `[10 clips, seed 20261107, fixed, overlap 39, 0, 0]`.
Continue From Last Clip has `keep_look_of_clip_1 = true` (Keystone Graft: clip 1 frame 0 is re-grafted).
Chain names `CR low res` / `CR high res` (separate memories per pass — do not share a name between passes).

### Clip Refiner wiring (per pass, purple group)

```
TakeFromLastTime(slot clip_refiner_ref_low) ─┐
sampler output ──────────────────────────────┼─ ComfySwitchNode(switch = is_first_clip)
                                             │     on_true  = sampler output (clip 1 is its own ref)
                                             │     on_false = TakeFromLastTime.latent
                                             └─→ SaveForNextTime(clip_refiner_ref_low)
ComfySwitch → H3ClipRefiner.reference
sampler output → H3ClipRefiner → LTXVSeparateAVLatent → LTXVConcatAVLatent → Remember Clip Ending / decode
```

- Node id has a space: `H3ClipRefiner -noEmbryo` (URL-quote it for `/object_info`).
  Pack: `custom_nodes/ComfyUI-noEmbryo` (github.com/noembryo/ComfyUI-noEmbryo, v1.9.6, commit `1dc8681`).
- Widgets: strength 1.0, context_length `"39"` (must match the chain overlap), refine_* defaults,
  detail_strength 1.0. **No model connected → statistical mode only**: per-channel mean/std match plus
  high-band gain match to the reference. The refine_denoise/steps/cfg/sampler widgets are unused.
- Its output is not a clean AV latent for downstream nodes; the Separate→Concat pair re-packs it.
- The code also corrects the overlap frames, although its README says the overlap stays raw.
- High-res side uses slot `clip_refiner_ref_high`, same pattern.

## Results (10-clip 2K walking test, hillside two-girls image, seed 20261107)

Measured by `measure_full.py` at each clip end (frame 140 + 102k), frames scaled to 960x544,
relative to the end of clip 1.

| Run | Saturation rise | Brightness | Sky kept (%) | Grain end | Max seam |
|---|---|---|---|---|---|
| Baseline 39-frame overlap (`full_A`) | +50.2 | −11.8 | 53 | 1.30 | 1.2 |
| Pinned latent start frame (`full_A_pin`) | +41.7 | – | 59 | – | – |
| Clip Refiner (`full_CR`) | +14.4 | +1.0 | 81 | 1.02 | 1.5 |
| Clip Refiner + latent pin (`full_CR_pin`) | +8.0 | – | 85 | 1.23 | 1.3 |

Raw numbers: scratchpad `full_results.json`.

Readings:
- The pin barely helps alone: with keep_look_of_clip_1 on, the VAE offset is already constant; the pin shows
  mostly at clip 10.
- Clip Refiner is the main fix. CR + pin is slightly better on colour but slightly grainier; Trent has not yet
  picked between them. The clean workflow is the **CR without pin** version.
- Earlier ranking (memory `geiru-chain-integration`): split unsample = no colour fix; latent pin = small.

## Session 2026-11-11 (second): generative half + strength 0.7

Same image, seed 20261107, 10 clips, prompts, sigmas as `full_CR`. Both runs ~18–20 min.

| Run | Saturation rise | Brightness | Sky kept (%) | Grain start → end | Max seam |
|---|---|---|---|---|---|
| Clip Refiner (`full_CR`, reference) | +14.4 (104 → 118) | +1.0 | 81 | 0.75 → 1.02 | 1.5 |
| CR + detail resample on pass 1 (`full_CR_detail`) | +24.7 (97 → 121) | −2.0 | 76 | 0.61 → 1.21 | 1.4 |
| CR strength 0.7, no model (`full_CR_s07`) | +14.2 (104 → 118) | +1.5 | 79 | 0.75 → 1.00 | 1.3 |

- **Generative half (`full_CR_detail`)**: pass-1 refiner got model (Pika Flex LoRA output, same as the pass-1
  guider), positive (pass-1 H3 conditioning) and negative = ConditioningZeroOut of it; refine_denoise 0.4,
  refine_steps 3, cfg 1, euler/normal, seed 0, detail_strength 1.0. All 10 clips logged "refine resample done"
  (~12 s per clip). **Result: worse.** It makes clip 1 visibly softer and hazier (grain 0.75 → 0.61, sat 104 → 97;
  the resample also runs on clip 1), whites turn blue over the chain (white R−B −4 → −33, CR +16 → +12), and the
  colour drift is larger than plain CR. Reasons in the code (`minimax.py` `_detail_transfer`): at
  detail_strength 1.0 the whole high band (gaussian sigma 2) is REPLACED by the resample's high band, and a
  3-step euler/normal resample of the 6-step turbo model gives a softer high band; also the high-band gain
  match is skipped whenever the resample ran (`if refined_body is None and current_hi_sd is not None`), so the
  texture ratchet is no longer corrected (grain doubles 0.61 → 1.21 vs ×1.36 in CR).
  Not tried: detail_strength < 1, the Pika sigmas/er_sde for the resample, or the resample on pass 2.
- **Strength 0.7 (`full_CR_s07`)**: same as strength 1.0 within noise (sat +14.2 vs +14.4, grain 1.00 vs 1.02,
  seams 1.3 vs 1.5, sky 79 vs 81). Strength 1.0 does not over-flatten; no reason to change it.
- Recommendation unchanged: statistical Clip Refiner, strength 1.0, no model.

Files: workflows `H3-Context-Noise-Test/H3 Full full_CR_detail.json`, `H3 Full full_CR_s07.json`; builders
`make_full_cr_detail.py`, `make_full_cr_s07.py` (edit a copy of `H3 Full full_CR.json`, no server needed) and
`run_full_cr_detail.sh` / `run_full_cr_s07.sh` in `projects/h3_clip_refiner_chain/`. The old scratchpad paths
in that folder are fixed (now point at the folder itself). Outputs `output/h3_chain/full_CR_detail_00001-audio.mp4`,
`full_CR_s07_00001-audio.mp4`. Grid `Downloads/full_grid_detail_titled.mp4` (Baseline / Clip Refiner /
Clip Refiner + pin / Clip Refiner + detail). Numbers: `full_results.json` (previous copy
`full_results_2026-11-10.json`).

## Ref2VA version (2026-11-11, test only)

Trent asked for a proper Ref2VA chain. The fused base model already holds the ref delta, so the change is small
(builder `projects/h3_clip_refiner_chain/make_ref2va_cr.py`, reads the clean workflow, writes
`H3-Context-Noise-Test/H3 Pika Video Chain Ref2VA + Clip Refiner (test).json`):
- both `MiniMaxH3ImageToVideo` → core `MiniMaxH3ReferenceToVideo` (`<Picture 1>` → `ref_image_0`, audio VAE
  connected, `ref_image_size` match); Pika LoRA fl2va → `pika_h3_flex_ref2va_comfy_bf16`;
- `start_image` → `first_frame` wires removed (Continue From Last Clip needs no first_image / vae now);
- 10 walk prompts converted by script to the six-section Ref2VA format (`walk_prompts_ref2va.json`);
- chain names `Ref2VA CR low/high res`, slots `ref2va_clip_refiner_ref_*`, prefix `h3_chain/pika_chain_ref2va_cr`.
Chain nodes and Clip Refiner groups unchanged. 10 clips at 2K took 17 min (not slower than fl2va).

Result (`pika_chain_ref2va_cr`, same seed/image): sat 99 → 107 (**+7.2**, fl2va CR +14.4), brightness +0.6,
sky 87 (81), white R−B +16 → 0, grain 0.78 → 0.99, seams max 1.4. Clip 1 still opens close to the picture.
Faces and wardrobe (ringer cuffs, collar) hold at least as well as fl2va by eye. One seed, one scene.
Side by side: `Downloads/ref2va_vs_fl2va_titled.mp4`; video `Downloads/pika_chain_ref2va_cr.mp4`.
Not yet done: a clean copy in `workflows/Pika-H3/` (note text, group titles still say fl2va wording).

### Ref2VA + latent pin (2026-11-11) — best colour hold so far

Trent asked for the latent pin on the Ref2VA chain. `H3PinFirstFrameFromLatent` only replaced an existing
frame-0 guide, and Ref2VA conditioning has none (KeyError), so the node now ADDS a frame-0 keyframe from the
first locked latent when there is none (`nodes/h3_video_chain.py` + test case, 10 tests pass, **uncommitted**;
fl2va behaviour unchanged; clip 1 passes through, so it stays pure Ref2VA). Workflow
`H3-Context-Noise-Test/H3 Pika Video Chain Ref2VA + Clip Refiner + pin (test).json`
(builder `make_ref2va_cr_pin.py`: pin node between each Ref2VA conditioning and its BasicGuider, both passes).

| Run | Saturation rise | Brightness | Sky kept (%) | White R−B | Grain | Max seam |
|---|---|---|---|---|---|---|
| fl2va CR (`full_CR`) | +14.4 | +1.0 | 81 | +16 → +12 | 0.75 → 1.02 | 1.5 |
| fl2va CR + pin (`full_CR_pin`) | +8.0 | +3.5 | 85 | +15 → +10 | 0.75 → 1.23 | 1.3 |
| Ref2VA CR (`pika_chain_ref2va_cr`) | +7.2 | +0.6 | 87 | +16 → 0 | 0.78 → 0.99 | 1.4 |
| Ref2VA CR + pin (`pika_chain_ref2va_cr_pin`) | **+3.2** | +1.7 | **90** | +16 → −7 | 0.78 → 0.88 | 1.4 |

Per-clip saturation with the pin stays flat from clip 4 (101 102 100 100 100 104 103). Grain grows least of
all runs. Whites lean slightly cool at the end (R−B −7). 17 min for 10 clips. One seed, one scene.
Grid `Downloads/ref2va_pin_grid_titled.mp4` (fl2va CR / fl2va CR + pin / Ref2VA CR / Ref2VA CR + pin);
video `Downloads/pika_chain_ref2va_cr_pin.mp4`.

## Ref2VA + pin, 8 steps on pass 1 (2026-11-11) — not better

Same as the Ref2VA + pin run, only the pass 1 sigmas changed to 8 steps (own list, equal log-SNR spacing between the Pika end values):
`1.0, 0.972973, 0.946396, 0.896382, 0.809174, 0.675192, 0.504951, 0.333333, 0.0`. er_sde, pass 2 unchanged. 18.5 min.

- sat 102.3 -> 104.5 (+2.2), per clip 102 108 106 108 82 94 94 102 103 105; sky kept 100; grain 0.76 -> 0.89; white R-B +15 -> +21.
- Clip 5 made a hard cut to a close-up inside the clip (frame 506), and clip 6 cut back at the join (frame 549, seam 10.9). All other seams 1.0-1.4.
- End-to-end drift is similar to 6 steps, but the per-clip values move more and the shot broke once. One seed; the cut may be chance or a result of 8 steps on the 6-step Pika LoRA.
- Files: `make_ref2va_cr_pin_8step.py`, workflow `H3 Pika Video Chain Ref2VA + Clip Refiner + pin 8step (test).json`, output `pika_chain_ref2va_cr_pin_8step`, grid `ref2va_pin_6_vs_8_titled.mp4` (Downloads).

## Ref2VA + pin, euler on pass 1 (2026-11-11) — not better

Same as the Ref2VA + pin run (6 steps), only the pass 1 sampler changed from er_sde to euler. Pass 2 stays er_sde. 16.5 min.

- sat 102.2 -> 106.7 (+4.4), per clip 102 103 104 102 79 102 103 103 110 107; sky kept 91; grain 0.75 -> 0.86; white R-B +16 -> +12.
- Seams 1.3 1.2 1.3 1.1 9.2 1.1 1.3 1.2 1.3. The clip 1 -> 2 seam is 1.3 (er_sde 1.4): euler does not mend the seams.
- Clip 5 again hard-cut to a close-up and clip 6 cut back, the same break as in the 8-step run. So the break is not chance noise: clip 5 (line "a copy of a copy of a photo") sits near a cut, and only er_sde at 6 steps held the shot.
- Files: `make_ref2va_cr_pin_euler.py`, workflow `... + pin euler (test).json`, output `pika_chain_ref2va_cr_pin_euler`, grid `ref2va_pin_ersde_vs_euler_titled.mp4` (Downloads).

## Ref2VA prompts brought to spec (2026-11-11/12) — cut gone, colour drift up

Run `pika_chain_ref2va_cr_pin_euler_p2`: same as the euler run, only the shot text changed (`make_ref2va_cr_pin_euler_p2.py`, prompts in `walk_prompts_ref2va_p2.json`): no time stamps inside the shot ("Then ...", "After she finishes, in the same shot, ..."), no "End at 00:05.875", camera "pulls out at slow speed with small amplitude" instead of "glides backward on a steady dolly", no instruction text. 16.5 min.

- No hard cut anywhere (old prompts: clip 5 cut at frame 506, seam 9.2). Seams 1.5 1.1 1.2 1.3 1.3 1.2 1.2 1.2 1.2. So the in-shot "At/From MM:SS.mmm" stamps were the probable cause of the clip 5 cut.
- Colour drift is worse: sat 102.7 -> 114.2 (+11.5), per clip 103 105 108 111 110 110 109 112 115 114; sky kept 81; grain 0.72 -> 0.87; white R-B +12 -> +9. Old prompts + euler: +4.4 / sky 91. One seed, so the size of this effect is not confirmed.
- Not tested: the new prompts with er_sde on pass 1; dialogue timing by ear.
- Grid `ref2va_prompt_old_vs_new_titled.mp4` (Downloads).

## To-spec prompts with er_sde on pass 1 (2026-11-12) — no cut, colour drift up again

Run `pika_chain_ref2va_cr_pin_p2` (`make_ref2va_cr_pin_p2.py`): p2 prompts, er_sde on both passes. 16 min.

- No hard cut. Seams 1.3 1.2 1.3 1.1 1.3 1.1 1.0 1.2 1.3.
- sat 101.2 -> 115.6 (+14.5), per clip 101 104 109 110 110 110 114 109 116 116; sky kept 82; grain 0.62 -> 0.97; bright +2.6; white R-B +15 -> +21.
- Old prompts, same setup: +3.2 / sky 90 / grain 0.78 -> 0.88. Both samplers now agree: the p2 prompts lose about 8-11 points of colour hold. Clip 1 also starts softer (grain 0.62 vs 0.78).
- Not separated yet: which p2 edit costs the colour (time stamps removed, or the camera sentence "pulls out at slow speed with small amplitude"). Next test: p2 with the old camera sentence restored.
- Grid `ref2va_ersde_prompt_old_vs_new_titled.mp4` (Downloads).

## p3 prompts: no time stamps + old camera sentence (2026-11-12) — best overall so far

Run `pika_chain_ref2va_cr_pin_p3` (`make_ref2va_cr_pin_p3.py`, prompts `walk_prompts_ref2va_p3.json`): p2 prompts with "The camera glides backward on a steady dolly ahead of them, keeping a medium two-shot ... at the same distance" put back. er_sde on both passes, 6 steps. 16 min.

- No hard cut. Seams 1.5 1.0 1.0 1.0 1.1 1.1 1.2 1.2 1.1 (lowest of all runs after join 1).
- sat 99.7 -> 103.2 (+3.5), per clip 100 101 99 103 103 103 104 103 106 103; sky kept 92; grain 0.77 -> 0.74 (first run with no grain growth); bright +4.4; white R-B +15 -> +1.
- Conclusion: the camera sentence "pulls out at slow speed with small amplitude" caused the colour loss of p2 (+14.5), not the stamp removal. The model needs "at the same distance" to hold the framing. Removing the in-shot time stamps is safe and removes the cut risk.
- Grid `ref2va_prompt_3way_titled.mp4` (Downloads).

## Join measurement (2026-11-12, `measure_joins.py`, `join_results.json`, CPU only)

Step from the last frame of a clip to the first new frame of the next, against the normal frame-to-frame change.

- Join 1 -> 2 is a colour step made by the Clip Refiner: whole-frame R-B -10.6 (fl2va CR, with or without pin), -18.9 (Ref2VA CR, with or without pin), -17.0 (p2); baseline with no refiner -0.2. Saturation -5, Ref2VA also brightness -4. 30-40x the normal frame change. Visible as "cooler and darker in one frame".
- Reason: clip 1 is the reference and passes unchanged; every later clip is matched to the mean of the whole clip 1 (which starts with the seated close shot), not to its walking end.
- Later joins: saturation steps down 2-3 points at each join (drift inside the clip, reset by the refiner); R-B steps about 1.3. Grain and sharpness: no step. Motion: no step.
- Repairs not yet tried: (1) Clip Refiner reference = only the end of clip 1; (2) pixel colour match at the join in Join Clips from the two decodes of the 39 overlap frames.

## Clip Refiner reference = end of clip 1 (2026-11-12) — first join better, drift worse

Run `pika_chain_ref2va_cr_pin_p3_tailref` (`make_ref2va_cr_pin_p3_tailref.py`): run 9 setup, but later clips take only the last 22 latents of clip 1 (about frames 66-140, the walk) as the refiner reference (Take -> LTXVSeparateAVLatent -> LatentCut t -22/22 -> LTXVConcatAVLatent -> switch). 16 min.

- Join 1 -> 2: R-B step -20.8 -> -8.5, saturation step -4.4 -> -2.8, sharpness step -0.52 -> -0.25, seam 1.5 -> 1.1. Still visible, but less than half.
- Drift is worse: sat +9.2 (run 9: +3.5), per clip 99 105 103 105 104 109 105 112 109 109; sky 87 (92); grain 0.77 -> 1.07 (0.77 -> 0.74); white R-B +15 -> +22.
- CORRECTION (the first reading of the noEmbryo code was wrong): the refiner DOES correct head and tail. The body is matched to the reference; head and tail are then each matched to the corrected body's channel mean/std (minimax.py `_correct_region`, ~L1190). Only clip 1 passes fully unchanged (its body is its own reference). So the join 1 -> 2 step is: natural end of clip 1 -> clip 2 forced to the mean look of the clip 1 body. Later joins step a little because head/body/tail are each forced to the same statistics whatever their content.
- Consequence used by the Join Clips colour match: the copied frames at the head of a new clip are the same picture as the end of the clip before, but with the refiner's correction, so the two decodes differ by exactly the colour step.
- Ideas not yet tried: correct the tail too; ramp the correction over the first second; pixel colour match of the overlap frames in Join Clips; start the chain reference at a walking clip; correct clip 1 too with a ramp.
- Grid `ref2va_p3_ref_all_vs_tail_titled.mp4` (Downloads).

## Join Clips colour match, first version (2026-11-12, TrentNodes ea62fb8) — fixes R-B, overshoots saturation

Run `pika_chain_ref2va_cr_pin_p3_joinmatch` (`make_ref2va_cr_pin_p3_joinmatch.py`): run 9 setup + Join Clips `match_colour_frames` 102. The match is per RGB channel mean + spread, measured on the 39 copied frames against the saved look of the end of the clip before. 17.5 min. Output 1920x1088, 24 fps, 1059 frames.

- Drift unchanged from run 9, as expected: sat +3.6, sky 92, grain 0.77 -> 0.74. Seams 1.1-1.2 (join 1: 1.5 -> 1.2).
- Join 1 -> 2: R-B step -20.8 -> -1.9, but saturation step -4.4 -> +8.9 and brightness -3.0 -> -3.6. The first new frame is now too orange: a different jump, not a clean fix. Join 2 -> 3 also R-B +5.4.
- Reasons: (1) the refiner forces the copied head to the body statistics separately, so the head does not carry the same correction as the new frames; (2) the refiner's change is not a per-channel RGB shift, so a per-channel match cannot undo it.
- CPU simulation on the run 9 video (480x272): measure on the k frames each side of the join and match with a full 3x3 colour matrix (Monge-Kantorovich linear transfer) with a fade. k=4, fade 48: join 1 luma +1.2 / sat +1.6 / R-B 0.0; later joins sat step 2.1 -> 0.7, R-B 0.7 -> 1.1; clip-end saturation identical to run 9. Per-channel mean (or mean+std) on the same frames still leaves sat +5 to +8.
- Next: change Join Clips to this boundary + colour-matrix match (save the last k emitted frames' mean + covariance instead of the 39-frame look), one GPU run to confirm.
- Grid `ref2va_p3_joinmatch_titled.mp4` (Downloads).

## Join Clips colour match, second version (2026-11-12, TrentNodes e851489) — BEST, joins clean

Run `pika_chain_ref2va_cr_pin_p3_joinmatch2` (`make_ref2va_cr_pin_p3_joinmatch2.py`): run 9 setup + Join Clips `match_colour_frames` 48. The node compares the last 4 emitted frames of the clip before with the first 4 new frames and removes the step with a Monge-Kantorovich 3x3 colour matrix, faded to zero over 48 frames. 16.5 min, 1920x1088.

| Join 1 -> 2 | luma | sat | R-B | seam |
|---|---|---|---|---|
| Hard join (run 9) | -3.0 | -4.4 | -20.8 | 1.5 |
| Match v1 (per channel, copied frames) | -3.6 | +8.9 | -1.9 | 1.2 |
| Match v2 (colour matrix across the join) | +1.9 | +0.4 | -0.1 | 1.1 |

- All joins, mean step: luma 1.0 (run 9: 1.3), sat 0.8 (2.3), R-B 1.0 (2.9). Seams 1.0-1.2.
- Drift identical to run 9: sat +3.5, sky 92, grain 0.77 -> 0.74, white R-B +15 -> +1.
- A small sharpness step stays at join 1 (-0.27, run 9 -0.52): clip 1 is a little sharper than the later clips.
- Recipe now: Ref2VA + Pika ref2va LoRA 6 steps er_sde + Clip Refiner (statistical, reference = all of clip 1) + Pin First Frame From Latent on both passes + p3 prompts + Join Clips match_colour_frames 48.
- Grid `ref2va_p3_joinmatch_3way_titled.mp4` (Downloads). One seed, one scene.

## Files

Test workflows (`user/default/workflows/H3-Context-Noise-Test/`):
`H3 Full full_A.json`, `full_A_pin`, `full_CR`, `full_CR_pin`.
Source of all four: `C:\Users\Trent\Downloads\Unsaved Workflow (2) (1).json` (Trent's original chain) with the
song and AnimateDiff LoRA removed (nodes 30–34, 47 dropped), audio decode added, walk prompts added.

Outputs (`output/h3_chain/`): `full_A_00001-audio.mp4`, `full_A_pin_…`, `full_CR_…`, `full_CR_pin_…`
(1920x1088, 1059 frames, all upscaled through pass 2).

Grids in Downloads: `full_grid3_titled.mp4` (3-up) and `full_grid4_titled.mp4` (2x2, titles
"Baseline 39-frame overlap", "Pinned latent start frame", "Clip Refiner", "Clip Refiner + latent pin",
red light, clip counter). Also `full_CR_clip_refiner.mp4`, `full_CR_pin_clip_refiner_plus_pin.mp4`.

Scripts: permanent copy in `/home/trent/ComfyUI/projects/h3_clip_refiner_chain/` (the session scratchpad
`/tmp/claude-1000/-home-trent-ComfyUI-custom-nodes/dddc8277-…/scratchpad/` will not survive a reboot).
The scripts still hard-code that scratchpad path in a few places (`grep -l scratchpad`) — change those
paths before re-running them from the new folder:
- `make_full.py` — `build(tag, pin)` from the original.
- `make_full_cr.py`, `make_full_cr_pin.py` — add the refiner groups (has its own `fresh()` helper; do **not**
  import `make_ctxwin`, its main runs on import and rewrites the context-window test workflows).
- `make_clean_cr.py` — builds the clean workflow.
- `measure_full.py <tags…>` — the table above.
- `run_full*.sh` + `wait_idle.sh` — queue via `/home/trent/comfy-mcp-venv/bin/comfy run --workflow …`.

## Clean workflow layout

Built on Trent's original groups/positions (he likes his layout; keep it). Groups: Models, Inputs, Clip & VAE,
Pass 1 Conditioning, Pass 1 — Pika Flex draft, Upscale + Pass 2 — 2K refiner, Pass 2 Conditioning, Batch,
Preview and Shift, Output, plus new: "Prompts - one per clip (split by =====)" and
"Seed + clip number -> prompt" (#3f789e), "Clip Refiner - pass 1/2 (match to clip 1)" (#a1309b, purple nodes).
A `Seed` PrimitiveInt (20261107) feeds both H3VideoChain seed and the math node `b`, so changing the seed
keeps the prompt index right. Batch manager count is 6 in the saved file; set it ≥ clips needed (10).

## Gotchas

- Do not launch/restart ComfyUI; Trent does. Tell him before any GPU run (one 10-clip 2K run ≈ 15–35 min).
- `fresh()` in the builders must insert `'fixed'` after seed widgets (control_after_generate value).
- The WSL disk filled on 2026-11-10 (vhdx on F:, 1.5 TB cap). Now 227 GB free. If command output vanishes
  and writes come out empty, the disk is full again — redirect output to `/mnt/c/Users/Trent/AppData/Local/Temp/`.
- Generated workflow JSON: VHS widgets_values must be a dict (see memory `generated-workflow-json-format`).

## Open items (offered, not approved — ask first)

1. ~~Refiner strength sweep~~ DONE 2026-11-11: 0.7 = 1.0 within noise.
2. Decide CR vs CR + pin; optionally add a bypassable pin group to the clean workflow.
3. Try a different image/scene and seed to confirm the gain is not specific to the hillside walk.
4. ~~Test generative mode~~ DONE 2026-11-11: worse at defaults (soft clip 1, blue whites). Untested: detail_strength < 1.
5. Add the clean workflow to TrentNodes `example_workflows` and push (needs Trent's OK; credit noEmbryo,
   NebSH two-pass recipe, Pika, Open-H3 refiner, MATLOWAI model in the note).
6. Hermes backup cron is broken since `~/hermes-backup-repo` was deleted (`~/.hermes/scripts/hermes-backup.sh`
   REPO_DIR) — re-point or disable.
7. Uncommitted dial files in TrentNodes (`h3_dial_*`, `h3_style_tokens.py`, `H3_IMAGE_DIAL_HANDOFF.md`) belong
   to separate work — do not touch or commit them with this.

## 2026-11-13 - Dance control tests (android picture + "Heads Will Roll" dance + "Came From Space")

Goal: make the Ref2VA chain follow a dance video. All runs: 3 clips, seed 20261107, graph of Trent's last run
(`projects/h3_clip_refiner_chain/run_ctl_test.py`, strips with `cmp_strip.sh`). Outputs in `output/h3_chain/ctl*_00001-audio.mp4`.

| Run | Motion input | Result |
|---|---|---|
| Trent | Fun ControlNet 2.0, thin skeleton video with the audience, strength 1.3 / 0.9 | loose; rows of extra androids (audience sentence in the prompt) |
| A | Fun ControlNet 2.0, strength 1.0 both passes, 3-dancer DWPose video (body + hands + face, 960x544), no audience sentence | poses follow well; facing and camera distance not exact |
| C | Guide Latents (real dance video in Add Guide, both passes) + Motion Transfer v1.0 LoRA, `motion_transfer:` | motion exact; picture fades to white fog, studio ceiling leaks |
| D | as C, pass 1 only | same as C (the fog comes from pass 1) |
| E | Guide Latents pass 1 + Motion Transfer FFP LoRA, `vfx_edit:`, picture as guide frame 0 in clip 1 | motion exact, look holds; hair of the real dancers leaks |
| **F** | **as E, guide = the 3-dancer DWPose skeleton video** | **BEST: motion, camera and timing exact, bald androids, white studio. Light green tint, one arm fault at the end of clip 1** |

Facts:
- The prompt must not contradict the guide (camera, who moves). Prompts: `make_android_dance_prompts.py` (moves read from DWPose keypoints of all 1059 frames).
- Pose is the weakest condition of Fun ControlNet Union 2.0; Guide Latents (frame N on frame N) hold motion and camera exactly.
- A real video as guide leaks hair, clothes and room. A skeleton video does not. The FFP LoRA contains the SolarRenderer control LoRA (trained on structure videos).
- Guide and motion LoRA on pass 1 only; pass 2 stays the plain chain.
- Tools: `pose_extract.py` (DWPose ONNX on CPU, ~1 fps), `pose_describe.py`, `pose_render.py` (tracks the 3 largest people; loses dancers in the floor section, 378 dancer-frames of 3177).
- Workflow: `user/default/workflows/H3-Context-Noise-Test/H3 Pika Ref2VA Video Chain + Motion Guide FFP (test).json` (validates, not rendered as a UI file; run F used the same graph through the API).
- Open: full 10 clips not rendered; floor section tracking; green tint; lip sync not judged.

### Same day, later: Sapiens guide and tracker repair

- `pose_track.py`: Hungarian match of the 3 dancers together, motion prediction, recovery of lost tracks, gap fill up to 12 frames. Rear dancer B: clips 7-10 found 141 / 141 / 112 / 127 of 141 frames (first tracker: 102 / 17 / 0 / 0).
- `sapiens_extract.py`: Sapiens 2 1B pose (308 keypoints) on the tracked boxes, as a separate script next to ComfyUI (~15 min for 1059 frames x 3 dancers). `sapiens_render.py openpose|raw308` draws the guide.
- Run G (run F recipe, Sapiens joints, OpenPose drawing) and run H (raw 308 drawing): both follow exactly, the wrong arm of run F at the end of clip 1 is gone. No clear difference between G and H in clips 1-3. The floor section (clips 9-10) is still not rendered.
- The workflow file now uses `heads_sapiens_openpose_960x544.mp4`. Research page: https://claude.ai/artifact/EtQH8WiZE5pAxDg2g3zxvx

### Same day, evening: run I (guide + FFP + control net)

Run I = run G plus Fun ControlNet Union 2.0 on pass 1 at 1.0, same Sapiens OpenPose skeleton as control video. 3 clips.
Result: worse than G. Motion follows the same as G (no visible gain in clips 1-3), but the skeleton drawing leaks into
the look: bodies turn thin and stick-like, dark lines run down the limbs, coloured joint dots show at knees, and from
clip 3 the heads become magenta blobs (the face keypoint colours). Do not combine the control net with the guide.
Wiring trap: control net node 79 takes its model from the Pika LoRA (node 3), so with switch 75 on the FFP LoRA branch
is bypassed. `run_ctl_test.py` now feeds node 79 from the FFP LoRA; the editor test workflow still has the trap.
Also seen in run G clip 3: green tint and a see-through torso, so the look drifts even without the control net.

### Same day, evening: full 10-clip run with the run G recipe (ctlG10)

`output/h3_chain/ctlG10_00001-audio.mp4`, 1059 frames, 44.1 s, about 37 min. Recipe = run G (Sapiens OpenPose skeleton on
Add Guide pass 1, FFP LoRA 1.0 pass 1, `vfx_edit:`, control net off), seed 20261107.
- Motion follows the real dance in all 10 clips, floor section (clips 9-10) included. Compared by eye, 5 frames per clip.
- Where the rear dancer has no skeleton (clips 9-10) the model keeps her in a plausible pose; she does not vanish.
- Colour is stable over the chain: per-clip mean Y 194.6-198.3, saturation 4.8-5.4, U/V within 0.5 (ffmpeg signalstats).
  The green tint is there from clip 1 and does not grow.
- Faults: light haze behind the dancers in the middle clips; one extra crouched android for a moment in clip 7, where
  the skeleton shows a small figure at the left edge that looks like a seated audience member (tracker error, not verified
  frame by frame); stray long hand lines in one skeleton frame of clip 7 with no visible effect.
- Not checked: joins at full speed, lip sync, a second seed.

### Same day, night: clean skeleton (ctlK10) and original music (ctlM10)

- `sapiens_render.py` now drops stray hand points (a finger point far from the median of its own hand, or a whole hand whose wrist is far from the finger group). 163 of 1059 frames changed. Limb-length rules were rejected: the long arm in the floor section is real perspective. New guide: `input/heads_sapiens_openpose_clean_960x544.mp4` (the old file is kept).
- `run_ctl_test.py` got `audio=` (node 68) and `lyrics=off` (removes the Came From Space `<d>` lines and the "plays without a voice" sentence, adds one neutral sing-along sentence).
- The guide video starts at source frame 445 (18.54 s) of the full Heads Will Roll video. `input/heads_will_roll_guide_audio.wav` is the matching 44.125 s of its music.
- ctlK10 = run G recipe + clean guide + Came From Space. ctlM10 = same + original music, `lyrics=off`. Both 10 clips, about 30 to 37 min each, success. Outputs `output/h3_chain/ctlK10_00001-audio.mp4`, `ctlM10_00001-audio.mp4`.
- Result: at the frames that had stray lines (600, 672, 941) G10, K10 and M10 look the same. The stray lines had no visible effect; the clean guide is still the better default. Colour per clip is the same as G10.
- Clip 7 fault CONFIRMED and still present in all three runs: near frames 704 to 718 the left dancer is hidden behind the front dancer and her track jumps to a seated audience member at the left edge. The model draws a crouched android there (K10 and M10 also draw the real left dancer, so 4 androids). Fix = keep the tracker off the audience (pose_track.py), then run sapiens_extract and render again.
- Not checked: lip sync and beat match of ctlM10 (needs ears), joins at speed, a second seed.

## 14 November 2026: character swap LoRA on the skeleton guide (swapA3, swapB3)

Alisson suggested the Character Swap LoRA with a pose map. Two 3-clip tests (345 frames), same seed, clean skeleton guide, dance-video music, pass 2 unchanged.

- **swapA3**: `minimax_h3_character_swap_v1.1_ref2va_fro` at 1.0 on pass 1 in place of the Motion Transfer FFP LoRA, `body_swap:` in front of our six-section prompts, skeleton video into Add Guide with no picture on guide frame 1 (the picture goes in through the reference input only). Result: works. All three skeletons become androids, white studio from the picture, no skeleton lines, poses match ctlM10 in six sampled frames, bodies a little more solid than ctlM10. Room differs (ceiling light, less fog).
- **swapB3**: same, but Alisson's fixed one-line `body_swap` template as the prompt of each clip. Result: worse. After clip 1 the background goes dark grey (the template tells the model to keep the guide's background, which is black), the androids turn glassy and lose the face.
- Rule: with a pose-map guide keep our scene-describing prompts and use only the `body_swap:` trigger. Do not use the fixed template.
- Not checked: audio, lip sync, joins at full speed, 10 clips, a second seed.
- Script: `run_ctl_test.py swapA3 clips=3 guide=heads_sapiens_openpose_clean_960x544.mp4 lora=minimax_h3_character_swap_v1.1_ref2va_fro.safetensors passes=1 trigger=body_swap: audio=heads_will_roll_guide_audio.wav lyrics=off` (`prompt=swap` gives the template).
- Clean test workflow of the FFP recipe: `user/default/workflows/H3-Context-Noise-Test/H3 Pika Ref2VA Video Chain + Motion Guide (clean).json`, built by `projects/h3_clip_refiner_chain/make_motion_guide_clean_workflow.py`.

## 15 November 2026: roycho's brightness blocks + P-H3 VAE check

Sources: roycho00087 (Banodoco minimax_h3_resources, 2026-11-15 04:16): attention boost on blocks **31-33, 36-44, 47**
(found by dropping blocks one by one and measuring brightness), keystone only, 30 clips, "all but 9% of the colour loss
removed". Built on ckinpdx `MMH3CarryAttention` (ComfyUI-MMH3Tools, pulled 82ea5fa -> e11c05e; tests pass). That node adds
a bias to attention scores of token spans: `carry_bias` needs ckinpdx's Looping Sampler (`carry="mask"`, key
`mmh3_carried_latents`), so it does nothing in our chain; `reference_bias` works under any sampler.

Runs (run 16 graph, walk test, seed 20261107, p3 prompts, both Clip Refiners BYPASSED; builder `make_nocr_refbias.py`,
runner `run_nocr_refbias.sh`; ~17 min each):

| Run | Sat rise | Bright | Sky kept | White R-B | Grain | Seams |
|---|---|---|---|---|---|---|
| Run 16 (`..._p3_joinmatch2`, with CR) | +3.5 | +4.2 | 92 | +15 -> +1 | 0.77 -> 0.74 | 1.2 |
| No CR (`pika_chain_ref2va_nocr`) | +26.4 | -14.5 | 64 | +15 -> +27 | 0.77 -> 0.78 | 1.3 |
| No CR + reference_bias 1.0 on roycho blocks, pass 1 (`..._nocr_refbias`) | +26.9 | -15.7 | 65 | +13 -> +31 | 0.72 -> 0.78 | 1.2 |

- Reference bias on those blocks = no colour effect. Clip 1 changed slightly (sat 101.5 vs 99.6, grain 0.72 vs 0.77), so
  the patch ran. NOT a test of roycho's exact method: his bias is on the carried/keystone frames, ours on the Ref2VA
  picture tokens. A faithful test needs a positive bias on the 39 copied latents (or frame 0) - would need the layout
  span of the overlap latents passed in (`mmh3_carried_latents`) or ckinpdx's Looping Sampler.
- New fact: on Ref2VA + pin + p3 + join match, the Clip Refiner alone takes the drift from +26.4 to +3.5. It is still the
  core of the recipe. Ref2VA without CR is far worse than fl2va + CR (+14.4).
- Grid `Downloads/refbias_grid_titled.mp4` (run 16 / no CR / no CR + ref bias / fl2va baseline).

P-H3 VAE (CPU-style round trip of `two_girls_hillside.png`, `output/vaetest/`): `p-h3-image_vae_bf16` has the SAME encoder
as `minimax_h3_video_vae_fp16` (max weight diff 3e-8) - only the decoder is tuned. So "encode the conditioning with the
P-H3 VAE" = encoding with the official video VAE. Round-trip shifts (sat / grain): video fp16 +5.6 / -0.37, LynnReal light
(our chain) +0.4 / -0.49, P-H3 image +0.8 / -0.18, H3 image fp16 +3.1 / -0.07. Untested idea: P-H3 decoder for the final
decode only, to keep grain (not checked on video latents).
