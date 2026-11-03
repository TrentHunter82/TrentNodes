# H3 Image Dial: Research Handoff

Last updated 2026-11-01. Written for the next research session. You do not need the old chat.

## Goal

Find whether images can steer the style of MiniMax H3 video the way ComfyUI-Dials text pairs do, with no training.

Trent's priority is the image route. A text dial already exists from someone else (matlowai), so text results are a baseline to beat, not the goal.

**Decision rule:** the best result wins. Between true ties, ship the simplest. Trent would like an elegant, simple solution, but the best result always comes first.

## Current answer

**Images can steer style, but only so far. Best image recipe on Trent's workflow:**

1. Stylize the start image with an image editor, keeping pose and framing.
2. Make a matched pair dial: stylized image minus original image.
3. Use the stylized image as the start image, and apply the dial at about +12.

Result: strong ink lines and a painted illustration feel, mostly bald, on prompt, with about 24% less motion than the same start image with no dial. It still has some 3D form. This was Trent's idea, and it is the only image method that holds the subject on his workflow.

**The bar to beat:** style-only text dial `h3_text_style` at +3, regular start image. It is flatter than anything the image route has made, with about 29% less motion. These two have not been compared directly on the same start image.

**Every result is one seed.** Treat them as leads.

## How a dial works

ComfyUI-Dials (matlowai, installed at commit `75d825e`):

- `dial = mean(B tokens) - mean(A tokens)`, measured after `condition_proj` and `token_refiner`.
- Apply Dial adds `strength * dial` to every prompt token after `token_refiner`, through an object patch on `token_refiner.forward`.
- The file has one key: `diffusion_model.token_refiner.dials_shift`, 5376 values, bf16.
- Apply Dial stops at +20. For higher values, save a scaled copy of the dial. `h3_img_pair_x2` is one.
- H3 runs the text path once per sampling, through `preprocess_text_embeds` in `comfy/ldm/minimax/model.py`.

Verified on the int8 fused model: the shift matches `strength * vec` within bf16 rounding, and the base model returns to its exact state after a dial run.

## Results

**Unless noted:** chain workflow, neutral prompt, regular start image, seed `636283492403563`, 243 frames, AnimateDiff LoRA on. Control: flow 1.59, sharpness 782, bald, rather 3D, thin rings.

Flow counts background motion, and sharpness rises with ink lines. Neither measures quality.

### Image methods

| Method | Result | Verdict |
|---|---|---|
| **Pair dial +3, +7, +12** | Steady change toward bold bands and lines, stays bald. Flow 1.56, 1.44, 1.42. Sharpness 870, 944, 1057. | Works, not flat yet |
| **Pair dial +18, +25, +35** | Becomes a bearded man with dark hair. +35 is a skull-faced costumed figure. | Breaks above about +12 |
| **Stylized start image, no dial** | Flat first frame, returns to glossy 3D by about 2.5 s. Flow 1.65. | Does not hold alone |
| **Stylized start image + pair dial +12** | **Best image result.** Ink lines and swirls. Flow 1.25, sharpness 2337. Side swirls can read as hair. | Current recipe |
| **Stylized start image + pair dial +25** | Bearded man with hair. | Breaks |
| Tile dial vs photos, +7, beach T2V | Photo became a 2D ink cartoon, not an exact match to the poster set. | Works with no style words and no start image |
| Tile dial, chain, +3, glossy prompt | Bolder shapes, still glossy. | Weak |
| Tile dial, chain, +7, neutral prompt | Glossy cartoon doll with big rainbow hair. Breaks "no hair". | Fails |
| Whole images vs grey | Carries subjects and colours, such as hair, sunglasses, bikini. No style. | Fails |
| Image minus caption +3 | More photoreal skin, patterned headscarf, flow 0.71. | Wrong direction at this sign and strength |

### Methods that do not shift all tokens

| Method | Result |
|---|---|
| Appended tile tokens, unlabelled, x1 and x2 | No style. Changes about like a new seed. x2 looks like x1. |
| Style images as `<Picture 2>` onward, with a sentence | No style. |
| `<Picture 1>` vision tokens swapped to the stylized image | No style. Softer colours, earrings gone. |

**Conclusion:** changing vision tokens failed three ways, while shifting all tokens has worked. Do not spend more time on token appending or swapping unless there is a new reason. Note that a trained adapter could behave differently, because it forces attention onto the image, and H3 has none for this.

### Text baselines

| Method | Result |
|---|---|
| **Style-only text dial `h3_text_style`, +1** | Flatter rings, face still 3D. Flow 1.58. |
| **`h3_text_style`, +2** | Graphic ink look, some depth. Hair-like side shapes. Flow 1.07. |
| **`h3_text_style`, +3** | **Flattest result.** Hard shapes, no gloss, mostly bald. Some letter-like shapes, skin turns black in places. Flow 1.13, sharpness 1329. |
| Text dial with the word "poster", +2 | Strong ink lines. Flow 1.14. |
| Text dial with the word "poster", +5 | A woman holding a sign with random letters. The word was taken literally. |
| Style words in the prompt, no dial | Poster colours, matte, printed paper edges, bald. Flow 1.49. |

### Fun dials made during testing

- Motion +0.5: flow +43%, face paint drops.
- `h3_liquid_paint` +1: flow +25%, sharpness +8%. With motion +0.5: flow +78%, sharpness -9%, paint restored.
- `h3_psychedelic`: +1.5 gives fractal flowers. +5 and +10 lose the woman completely and become abstract mandalas.
- `h3_woman_potato`: -1.5 gives glamour and long hair. +1.5 gives a rounder face and no potato.

## Step 1 results (2026-11-01)

Stylized start image, neutral prompt, AnimateDiff LoRA on, seed `636283492403563` unless noted. Run files: `user/image_dial_research/runs/step1_runs/`. Stack script: `scripts/stack_step1.py`.

**Measured:**

| Row | Flow | Sharpness |
|---|---|---|
| No dial (earlier run `pair2/s0`) | 1.654 | 833 |
| Pair +8 | 1.194 | 2058 |
| Pair +10 | 1.163 | 2364 |
| Pair +12 (earlier run `pair2/s12`) | 1.250 | 2337 |
| Pair +14 | 1.388 | 1951 |
| Text style +3 | 1.135 | 1415 |
| Seed `20261101`, no dial (added control) | 1.569 | 786 |
| Seed `20261101`, pair +12 | 1.276 | 2014 |

**Seen in frame sheets (Claude's reading, six frames per row, not a playback review):**

- Text +3 is much flatter than every pair row: hard graphic shapes with no gloss. It loses the most subject: the face changes through the clip, teeth show, the rings break into letter-like fragments.
- Pair +8 and +10 keep a woman, with ink bands, but keep gloss and 3D form. Both add hoop earrings that the control does not have.
- Pair +12 has the strongest ink look that still holds a woman.
- Pair +14 starts the break: the face reads as male by the last third of the clip. Side swirls also read more like hair.
- Seed 2 pair +12 repeats the look of seed 1: swirls, ink lines, bald woman, earrings. Style and hold repeated on this second seed.

**Verdict:** pair +12 is the better clip, and text +3 is the stronger style. The pair dial has not beaten text on flatness. The flow numbers do not decide motion: pair +14 has more flow than +12, and graphic edges can change the number. Trent has not judged these yet.

**Stacks:** `step1/stacks/step1_seed1_stack.mp4`, `step1/stacks/step1_seed2_stack.mp4`.

## Step 2 state

- `nodes/h3_dial_block_range.py` is written, uncommitted and experimental. It uses a `DIFFUSION_MODEL` wrapper, because sparse attention overwrites all 50 `double_block` slots on pass 2. A plain `set_model_patch_replace` at the dial position would give pass 2 no dial. The wrapper wraps the patch it finds in the slot.
- Fake-model unit test passed: the shift enters at the first block, leaves after the last block, and keeps existing patches.
- **Real-model proof passed** (pair +12, same seed, same server; a real render with the model compiler and prefetch path active). Mean pixel difference out of 255: hook 0-49 vs Apply Dial 1.637; Apply Dial vs earlier Apply Dial (rerun noise) 1.636; Apply Dial vs no dial 67.4. The log shows the hook on 6 pass 1 steps (`other patch False`, 1413 text rows) and 2 pass 2 steps (`other patch True`, 2445 text rows). Sparse attention still ran its slot on pass 2.
- Run files: `runs/step2_runs/`, built by `scripts/build_step2.py`. Output: `output/h3_dial_test/step2/`. Stack script: `scripts/stack_step2.py`.

**Range results (stylized start, neutral prompt, LoRA on, seed `636283492403563`, one seed):**

| Range | Pair +12 flow / sharpness | Text +3 flow / sharpness |
|---|---|---|
| No dial | 1.654 / 833 | 1.654 / 833 |
| All blocks | 1.250 / 2337 | 1.135 / 1415 |
| 0-19 | 1.245 / 2267 | 1.115 / 1426 |
| 20-34 | 1.575 / 853 | 1.658 / 857 |
| 35-49 | 1.671 / 856 | 1.679 / 832 |
| 20-49 | 1.614 / 850 | 1.631 / 840 |

**Seen in frame sheets:** 0-19 looks like the all-blocks dial for both dials, including the motion cost and the subject drift. 20-34, 35-49 and 20-49 look like no dial: glossy rings, no style.

**The prediction failed.** I expected style to live in late blocks, so that a late shift would keep motion and subject. At these strengths, late shifts do almost nothing, and the early range carries the style and the side effects. This does not show that late blocks have no style role. It tests only a text shift of this size at those blocks, and rows have already been changed by earlier blocks. The higher late strengths are untested.

**Stacks:** `step2/stacks/step2_pair_ranges_stack.mp4`, `step2/stacks/step2_text_ranges_stack.mp4`. Waiting for Trent to review before further tests.

**Early split, pair +12 only (same settings, one seed):**

| Range | Flow | Sharpness | Seen in three sampled frames |
|---|---|---|---|
| No dial | 1.654 | 833 | Glossy rings |
| 0-19 | 1.245 | 2267 | Ink swirls |
| **0-4** | **1.257** | **2193** | Almost the same as 0-19, including earrings |
| 5-9 | 1.581 | 769 | Close to no dial, small flow drop |
| 10-14 | 1.642 | 842 | Close to no dial |
| 15-19 | 1.663 | 844 | Close to no dial |

**Result:** blocks 0-4 carry nearly the whole dial effect, with its flow drop. Style and flow drop did not separate at any tested range. Claude's decision: stop block mapping and go to Step 3. Not tested: single blocks 0-4, stronger late shifts, other seeds, playback review by Trent. Grid: `step2/stacks/step2_pair_early_grid.mp4`.

**Single blocks, pair +12 only (shift added at the block input, removed at the next block input; same settings, one seed):**

| Block | Flow | Sharpness | Seen in two sampled frames |
|---|---|---|---|
| No dial | 1.654 | 833 | Glossy rings |
| All blocks | 1.250 | 2337 | Ink swirls |
| 0-4 | 1.257 | 2193 | Ink swirls |
| **0 only** | **1.340** | **2351** | Ink swirls, earrings, as strong as all blocks |
| 1 only | 1.611 | 820 | Close to no dial |
| 2 only | 1.626 | 774 | Close to no dial |
| 3 only | 1.599 | 848 | Close to no dial |
| 4 only | 1.664 | 840 | Close to no dial |

**Result:** nearly the whole effect needs the shift at block 0. This corrects the earlier reading of "the first few blocks". Lead: block 0 only keeps the sharpness of all blocks with 7% more flow (1.340 vs 1.250). Apply Dial repeated at 1.250 on two renders, so the gap is not rerun noise on this seed, but it could be a scene change and not more real motion. Cause not tested. Next check: block 0 only on seed `20261101` against `step1/seed2_p12`. Grid: `step2/stacks/step2_pair_single_blocks_grid.mp4`.

**Block 0 only on seed `20261101`:** no dial flow 1.569 / sharpness 786, all blocks 1.276 / 2014, block 0 only 1.312 / 2124. Same look in four sampled frames, earrings stay. The flow gain fell from 7% to 3%. Same direction on two seeds, but small. **Decision: stop block mapping, go to Step 3.** Grid: `step2/stacks/step2_seed2_block0_grid.mp4`.

## Step 3 results: multi-subject pair dial (2026-11-01)

**Dial:** `h3_img_pair_multi`, six pairs: station, pilot, robot, mouse, superboy, pizza. Made by `scripts/stylize_step3.py` and `scripts/make_multi_pair_dial.py`, images in `output/image_dial_step3/`. Style reference `image_dial_style_poster.png`. Agreement 0.70, norm 23.6, cosine 0.68 with `h3_img_pair`. The mannequin and the indoor portrait are in no pair. The dragon pair was dropped, because Qwen pasted the poster woman over it. Pilot gained an orange suit and planet, and mouse gained a grey background. These two stayed in the dial. Do not use the adult images in `input/` as dial material.

**Measured (seed `636283492403563`, LoRA on, one seed):**

| Row | Flow | Sharpness |
|---|---|---|
| Styled start, no dial | 1.654 | 833 |
| Styled start, old pair +12 | 1.250 | 2337 |
| Styled start, multi +12 | 1.347 | 1294 |
| Styled start, multi +16 | 1.629 | 1077 |
| Regular start, old pair +12 | 1.422 | 1057 |
| Regular start, multi +12 | 1.378 | 596 |
| Portrait, no dial | 0.580 | 530 |
| Portrait, old pair +12 | 1.121 | 523 |
| Portrait, multi +12 | 0.908 | 563 |

**Seen in three sampled frames per row:**

- Styled start, multi +12: large flat colour panels with thick black outlines. Bald woman, no earrings in these frames. It looks flatter than old pair +12, with less gloss, but it turns the prompted rings and swirls into wide bands.
- Styled start, multi +16: the flattest pair look so far, but dark hair appears and the face reads male. Breaks.
- Regular start, multi +12: panels and outlines appear from an unstylized start, with some 3D kept and a small earring early.
- Portrait (a new photo and a new prompt): **no style change from either dial.** It stays a photo. Flow rises and the room changes. The dial failed here.

Sharpness fell while the look got flatter, so sharpness does not rank these. Multi +12 is an image-route lead on Trent's clip, not proven. Its transfer to the mannequin may depend on that clip already having bold paint and a colour-band prompt.

**Grids:** `step3/stacks/step3_multi_grid.mp4`, `step3/stacks/step3_portrait_grid.mp4`. Next: multi +12 on seed `20261101`, multi +10, and text +3 on the portrait as the baseline, after Trent's review.

## AnimateDiff LoRA at 0.25 (Trent's idea, 2026-11-02)

Node 53 at 0.25 instead of 0.69, all else the same. Run files `runs/step3_runs/ad025_*`. Grid `step3/stacks/step3_ad025_grid.mp4`. One seed. Compare with 0.69 by flow and sharpness:

| Row | Flow 0.69 → 0.25 | Sharpness 0.69 → 0.25 |
|---|---|---|
| Styled, no dial | 1.654 → 1.073 | 833 → 2305 |
| Styled, old pair +12 | 1.250 → 0.636 | 2337 → 3458 |
| Styled, multi +12 | 1.347 → 0.736 | 1294 → 3382 |
| Styled, text +3 | 1.135 → 0.736 | 1415 → 2509 |
| Regular, multi +12 | 1.378 → 0.876 | 596 → 1158 |
| Portrait, multi +12 | 0.908 → 0.489 | 563 → 497 |

**Seen in three sampled frames per row:**

- Styled, no dial: holds fine ink-and-colour illustration lines to the end of the clip, with no dial. Eyes and lips stay partly realistic. At 0.69 this same row returned to glossy 3D.
- Styled, multi +12: the most illustrated look of the image rows. Thick outlines, flat colour, face holds, rings become wide bands.
- Styled, old pair +12: between those two.
- Styled, text +3: flat graphic shapes, but the face breaks into symbols by mid-clip.
- Regular, multi +12: glossy 3D with lines. The styled start matters.
- Portrait, multi +12: still a photo, with harder colour steps. Not a style change.
- Earrings appear in every mannequin row here, including no dial, so at 0.25 they do not come from the dial.

**Result:** the LoRA strength is a large part of the 3D pull-back (unproven item 3). Measured cost: flow falls 35-49% from the same row at 0.69. Flow counts background changes, so Trent must judge the real motion. Next: LoRA 0.4 and 0.55, styled start, no dial and multi +12, to find the trade-off.

## Fast motion text dial (Trent's idea, 2026-11-02)

**Dial:** `h3_fast_motion`, made with Make Dial (`runs/mk_fast.json`): five slow-motion vs fast-motion pairs over six subjects, 30 pairs, agreement 0.704, |dial| 45.3, debias on, calibrated. No style words. Cosine 0.55 with `h3_motion`, 0.02 with `h3_img_pair_multi`.

**Test:** styled start, AnimateDiff 0.25, multi +12 plus fast +s, saved as one combined file `t_multi12_fast{s}` (= 12 x multi + s x fast, applied at strength 1; test leftovers). One seed. Grid `step3/stacks/step3_fast_grid.mp4`.

| Row | Flow | Sharpness | Seen in three sampled frames |
|---|---|---|---|
| Old recipe: no dial, AD 0.69 | 1.654 | 833 | Glossy 3D |
| Multi +12, fast 0 | 0.736 | 3382 | Illustrated, outlines |
| Fast +1 | 1.167 | 2981 | Same look, closer framing |
| **Fast +2** | **2.388** | 2940 | Same illustrated look, face holds, bald. Camera pushes in close. Flatter background |
| Fast +4 | 3.062 | 3232 | Look holds, but an angry face, gritted teeth and speed lines |
| Fast +8 | 2.369 | 1723 | Breaks: football players in helmets, ends as a photoreal man |

**Result:** the fast dial restores flow above the old recipe at +2 while the illustrated look stays (one seed). Part of the flow may be camera push-in, not subject motion. Earrings appear in every row at AD 0.25. +4 changes mood, +8 swaps subject (likely from "crowd of people" and sport-like speed). Candidate recipe: styled start + multi +12 + fast +2 + AD 0.25. Needs Trent's playback review and a second seed.

## AnimateDiff 0.4 with fast and morph dials (2026-11-02)

**Morph dial:** `h3_morph`, Make Dial (`runs/mk_morph.json`): five "frozen still frame" vs "constantly morphing / melting / reshaping" pairs over six subjects (no crowd), 30 pairs, agreement 0.714, |dial| 56.1. Cosine 0.68 with `h3_liquid_paint`, 0.17 with `h3_fast_motion`, 0.005 with multi. Combined test files `t_multi12_morph{1,2,4}`.

**Test:** styled start, AnimateDiff 0.4, one seed. Grid `step3/stacks/step3_ad040_grid.mp4`.

| Row | Flow | Seen in three sampled frames |
|---|---|---|
| Old recipe, AD 0.69, no dial | 1.654 | Glossy 3D |
| Multi +12 | 0.889 | Flat with outlines; 0.4 still holds the look |
| Multi +12, fast +2 | 3.070 | Flattest and most graphic, bold shapes, wide eyes, stern brow, camera close |
| Multi +12, morph +1 | 1.594 | Flat face, background bands now wave and flow |
| Multi +12, morph +2 | 1.917 | More flowing; a hand rises to her head early, drips in the background |
| Multi +12, morph +4 | 2.283 | Everything melts; looser, more painterly, less outline, mouth wide open |

**Result:** AD 0.4 keeps the flat look with these dials. Morph gives flowing, melting motion in the scene rather than camera push. Fast gives the most flow but more of it looks like framing and expression. Leads for Trent's playback: multi +12 + morph +2 + AD 0.4, and multi +12 + fast +2 + AD 0.4. Next: a second seed for the chosen one, and morph +2 with fast +1 stacked.

## Animation and 2D text dials (2026-11-02)

**Dials:** `h3_animation` (photoreal live action vs animated cartoon, 30 pairs, agreement 0.719, |dial| 49.0) and `h3_2d` (3D sculpted vs flat 2D drawing, 30 pairs, agreement 0.654, |dial| 41.2), via `runs/mk_h3_animation.json` and `runs/mk_h3_2d.json`. Cosine anim/2D 0.385, each about 0.53 with `h3_text_style`, 2D vs morph -0.168. Test files `t_multi12_morph2_anim2d{1,2,4}` and `t_morph2_anim2d2`.

**Test:** styled start, AD 0.4, one seed. Grid `step3/stacks/step3_anim2d_grid.mp4`.

| Row | Flow | Seen in three sampled frames |
|---|---|---|
| Old recipe | 1.654 | Glossy 3D |
| Multi +12, morph +2 | 1.917 | Flat, outlines, flowing bands |
| + anim and 2D +1 | 1.514 | Flatter, more cartoon face, arm raised |
| + anim and 2D +2 | 0.960 | Round cartoon shapes, big grin, simple fills |
| + anim and 2D +4 | 1.304 | Full cartoon character: big eyes, grin, peace sign. Ends as a smiley swirl, subject lost |
| **No image dial: morph +2, anim and 2D +2** | **2.121** | Flat, liquid, cartoon-like; face holds; most flow of the set |

**Result:** the two text dials push toward real 2D animation. +4 loses the subject. The text-only row (no image dial) kept a flat look with the most flow, so the image dial may not be needed once the LoRA is at 0.4 with a stylized start. One seed. Next: text-only recipe on seed 2, and with the regular start image.

## What is still unproven

1. **Does the pair dial hold on more seeds?** It held on one second seed. See Step 1 results.
2. **Does it work beyond its own image?** Partly. The six-subject dial changed the mannequin clip, but neither pair dial changed a photo portrait at +12. See Step 3 results.
3. **What causes the 3D pull-back?** The AnimateDiff LoRA strength is a large part of it: at 0.25 the stylized start holds with no dial (one seed). The fused model and refiner pass are untested. **Do not remove the LoRA from any final recipe. Trent wants its motion.** A one-off test that turns it off is fine for diagnosis only.
4. **Why do strong dials turn the subject into a man?** The pair dial has near-zero cosine with the woman dial (-0.015), and that did not predict it. Do not use one cosine as evidence of a clean dial.
5. **Audio.** Saved videos use the forced song, so these tests do not show dial effects on model audio or lip sync.
6. **Full chain.** No dial has been rendered across a multi-clip chain.

## Next research, in order

### 1. Compare on equal terms (about 25 minutes)

With the stylized start image, neutral prompt and LoRA on:

- Pair dial at +8, +10 and +14.
- Pair dial at +12 on a second seed.
- `h3_text_style` at +3.

### 2. Map blocks with a dial (Trent's current interest)

**Why:** dials shift the text that all 50 blocks read, which may be why style dials cost motion and why subjects break. If style lives in later blocks, a shift on those blocks only could keep motion and the subject. This is an inference, and nobody has reported testing it.

**What others report (no full H3 map found, 2026-11-01):**

- Ada: "0-19 / 50 are the more motion heavy blocks". No test shown.
- JohnDopamine: train blocks 20+ on images to reduce motion loss. Common practice, and he also questions it.
- David Snow: image LoRAs on all blocks cost about 50% motion, every second block about 10-15%. His estimates. He also says targeting blocks should work in theory but "minimax H3 doesn't work this way".
- aimatlow: H3 DyRoPE "timing fidelity" preset uses blocks 30-49.
- fredbliss: blocks 45, 48 and 49 are sensitive to int8. This is numeric error, not function.
- alcaitiff: general rule, not specific to H3: composition early, lighting, style and detail late.
- TenStrip's 10Eros Max page mentions "block analysis and maps". Check that page.
- Wan 2.1 paper (arXiv 2504.10317): one layer strongly affects structure. No paper found that isolates a style block in a video model.

**The hook is possible.** `comfy/ldm/minimax/model.py:760-772` checks `("double_block", i)` in `patches_replace["dit"]`. The patch receives `img` (`h`), `layout` and `original_block`. `layout.segments` gives the text rows, and text comes first in `h`. Set it with `ModelPatcher.set_model_patch_replace(fn, "dit", "double_block", i)`. Nodes must go through the model patcher.

**Plan:**

1. Add `strength * vec` to the text rows at the input of block N, and optionally subtract it at the end of a range. Blocks also change text rows, so a range is not perfectly clean.
2. First check that this path matches a normal dial when N = 0, as far as it can. The normal dial shifts after the refiner, so block 0 input is the same point.
3. Check that it runs with the model compiler and prefetch path, and on the sparse attention pass 2.
4. Test ranges 0-19, 20-34, 35-49 and 20-49, using `h3_text_style` +3 and the pair dial +12, on one seed. Score style, motion and subject.

### 3. Improve the pair dial

- More pairs across different subjects. Stylize 5-10 images, then test on unrelated prompts.
- Debias, as Make Dial does. The image dial has none.
- PCA on the differences, keeping several directions.
- Matched pairs give token-to-token alignment, so a per-token difference is now possible. But vision-token shifts have failed, so a per-token map onto prompt tokens is the only form left, and it needs a good reason.

### 4. If it works, make it simple

Update `H3MakeImageDial` to take `original_images` and `stylized_images`, matched by order. This could replace tile mode. Keep the node small, and remove what does not change results.

## Files

| What | Where |
|---|---|
| Pushed node | `custom_nodes/TrentNodes/nodes/h3_image_dial.py` (commit `3d80a9d`, on main) |
| **Uncommitted, failed experiments** | `custom_nodes/TrentNodes/nodes/h3_style_tokens.py`: three token nodes. Trent has not decided. I recommend deleting it. |
| **Uncommitted** | This handoff |
| **Uncommitted, experiment** | `custom_nodes/TrentNodes/nodes/h3_dial_block_range.py`: block range test node |
| Scripts | `user/image_dial_research/scripts/` (paths still point to an old scratchpad, so update them) |
| Run files | `user/image_dial_research/runs/` |
| Pair dial script | `user/image_dial_research/scripts/make_pair_dial.py` |
| Stylized images | `output/image_dial_pair/stylized_0000{1,2,3}_.png`. `input/image_dial_stylized_start.png` is a copy of the first. |
| Style images | `input/image_dial_style_*.png` |
| Dials | `models/loras/dials/`. `t_ctrl` and `my_style_dial` are test leftovers. |
| Stacks | `output/h3_dial_test/*/stacks/` |
| Shareable T2V workflow | `user/default/workflows/Pika-H3/H3 Image Style Dial (Trent).json` (tile method), copy in Downloads |
| Shareable chain workflow | `user/default/workflows/Pika-H3/Pika H3 Video Chain + Image Style Dial (Trent).json` (tile method, set to +3, 4-clip run not rendered), copy in Downloads |
| Trent's chain workflow | `user/default/workflows/Pika-H3/Pika H3 Flex 2-Pass Upscale - Video Chain (Trent).json`. Not changed. |

**Key stacks:**

- `pair2/stacks/pair_styled_start_stack.mp4`: best image result.
- `pair2/stacks/pair_higher_stack.mp4`: where it breaks.
- `pair/stacks/pair_dial_stack.mp4`: pair dial +3 to +12.
- `compare/stacks/text_style_only_stack.mp4`: text baseline.
- `share/stacks/image_style_stack.mp4`: tile dial on the beach prompt.

**How to stylize:** Qwen Image Edit 2511 bf16 + 8-step Lightning LoRA, CFG 1, euler, simple, shift 3, `TextEncodeQwenImageEditPlus` with image1 = input and image2 = style reference. The prompt asks to redraw image 1 in the style of image 2, and keep the subject, pose and framing, with no hair and no objects from image 2.

## How to test

- **Neutral prompt:** in `runs/cmp_runs` or `runs/pair_runs`. Word-check it for style terms. The only hit should be "ink" in "hot pink".
- **Test files:** one clip, `control_after_generate` fixed, ending names `dial test low` and `dial test high`. **Never use `low res` or `high res` in a test, because clip 1 clears that folder.**
- **Same-graph control:** strength 0 returns the model unchanged. A re-rendered control differed from the first by a mean of 1.8 out of 255, so runs repeat closely, not exactly.
- **Review:** stack rows top to bottom, with settings written on each row, and look at frames from the whole clip. Trent likes sets of three. Full playback is his call, because frame sheets cannot judge motion.
- **Time:** about 5 minutes per chain clip, about 1 minute per beach clip after the first load.

## Server and rules

- ComfyUI runs detached with `--disable-cuda-malloc`. Log: `user/comfy-restart.log`. **It was started by Claude, not in Trent's terminal.**
- Find it with `pgrep -f '^venv/bin/python main.py'`. A plain `pgrep -f "main.py"` matches your own shell, and `kill` then stops your own command.
- New or changed nodes need a restart. Trent normally starts ComfyUI himself, so ask before you restart, and check that the queue is empty.
- MCP `restart_comfyui` refuses a server it did not start. Run workflows with MCP `run_workflow(wait=False)`.
- Commit and push only when Trent asks. The TrentNodes branch is `main`.
- Keep the AnimateDiff LoRA on in any recipe for Trent's flow.
- Do not change the real `<Picture 1>` without his okay.
- Say so plainly when a test fails or a prediction is wrong.
- Trent posts on Banodoco as Flipping Sigmas.
- Check the MiniMax H3 Community License territory terms before paid work. Dial files fall under it.

## Credit

ComfyUI-Dials: matlowai (aimatlow). The study that started Dials used David Snow's every-second-block finding. The matched pair idea and the image dial direction are Trent's. Concat reference for H3 tags: blepping's `ConditioningConcatH3` gist. InstantStyle and IP-Adapter informed the ideas tested here.

## Dial style workflow (2026-11-02)

`user/default/workflows/Pika-H3/Pika H3 Flex Dial Style - Video Chain (Trent).json`, built by
`user/image_dial_research/scripts/build_dial_workflow.py` from the original Pika 2-pass chain.
Five chained Apply Dial nodes after the AnimateDiff LoRA (0.4), feeding both passes:
h3_img_pair_multi +12, h3_morph +2, h3_animation +1, h3_2d +1, h3_fast_motion 0 (off).
Stylized start image, neutral test prompt, prefix `h3_chain/dial_style_chain`.

Check (one clip, seed 636283492403563, same as ad040_anim2d1): the API prompts differ only in
five chained dials vs one combo file at 1.0. Pixel diff vs ad040_anim2d1 = 32/255 average, but
pose, look, timing and the end smile match; only swirl stripe placement differs. Inferred, not
tested: bf16 rounding from adding five shifts one by one vs one summed vector moves fine detail.
Same recipe, not bit-identical. Frame strip: Downloads/dial_chain_vs_combo.jpg.

## H3 Dial Operator node (2026-11-02, uncommitted)

`nodes/h3_dial_operator.py` + `js/h3_dial_operator.js` + two routes in `server.py` +
`tests/test_h3_dial_operator.py` (7 pass). Slots = saved dial file or A/B terms (terms expand over
4 templates x subjects via Dials Make Dial); knob -20..+20; all active dials summed in fp32 and applied
once with Apply Dial. Text dials are cached by a term key; a saved file stores the key and terms, so
matching terms load the file with no clip. Save mix writes the summed vector as one dial file.

Render check (same seed as ad040_anim2d1, slots = the four dial files at 12/2/1/1):
operator vs combo file 1.67/255 (rerun-noise level); operator vs 5-node chain 32.16. So the chain drift
comes from applying dials one by one (sequential bf16 adds), not from the dials. Prefer one summed apply.
A strength-0 text slot "slow motion" -> "fast motion" with save saved
`dials/slow-motion_vs_fast-motion.safetensors` with terms and pairs. Template-made text dials are untested
for look. Routes 404 until a full server restart (hot reload cannot add aiohttp routes).
