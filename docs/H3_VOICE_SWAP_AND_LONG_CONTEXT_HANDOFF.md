# H3 voice swap + long-context handoff (15 to 21 November 2026)

Read this first in the next session. Older history: `docs/H3_CLIP_REFINER_CHAIN_HANDOFF.md` (the chain colour-drift work up to
run 16) and the artifacts "Long Video Colour Hold" (https://claude.ai/artifact/DrU8tCy4nCNhF9Ez8pRbwC) and
"H3 Dance Chain Findings" (https://claude.ai/artifact/JCEdpADCNT7yUw79FduUrf).

## Rules from Trent (keep)

- Do not launch or restart ComfyUI. Trent does it. Check the server with comfy-mcp `server_info`.
- Tell Trent before every GPU run, with the expected time. Queue runs one at a time.
- Ask before deleting anything and before installing a node pack. Check free disk space before long runs (WSL disk, ~170 GB free).
- Workflows I build: keep the default node titles. Instructions go in rgthree `Label (rgthree)` nodes (white 12 px Helvetica on
  black). Memory `workflow-node-titles-labels`.
- `comfy run` (CLI) rejects rgthree Labels as unknown nodes: queue a copy with the Labels removed. The browser ignores them.
- Commit or push only when Trent asks. Never add the uncommitted `nodes/h3_style_tokens.py` (separate dial work).
- New node files need a ComfyUI restart. Hot reload did NOT pick up input changes of the OmniVoice node either.

## 1. Chain colour hold: state of the art in our tests

Best recipe (run 16, 12 Nov): Ref2VA + Pika ref2va LoRA 6 steps er_sde + Clip Refiner (statistical, reference = clip 1) + Pin First
Frame From Latent (both passes) + p3 prompts + Join Clips `match_colour_frames 48`. Walk test, 10 clips: saturation +3.5, sky 92 %.

New this week (15 Nov, `projects/h3_clip_refiner_chain/make_nocr_refbias.py`):

| Run | Sat rise | Bright | Sky |
|---|---|---|---|
| Run 16 (with Clip Refiner) | +3.5 | +4.2 | 92 |
| Run 16 without Clip Refiner (`pika_chain_ref2va_nocr`) | +26.4 | -14.5 | 64 |
| No CR + MMH3CarryAttention reference_bias 1.0 on blocks 31-33,36-44,47, pass 1 | +26.9 | -15.7 | 65 |

- The Clip Refiner is the core of the recipe (+26 -> +3.5).
- The reference-bias test was NOT roycho's method: he biases the keystone (clean frame 0 of clip 1), we biased the Ref2VA picture tokens.
- `p-h3-image_vae_bf16` has the same encoder as `minimax_h3_video_vae_fp16` (max weight diff 3e-8). Encoding conditioning with it
  changes nothing. Its decoder keeps more grain (round trip grain -0.18 vs LynnReal -0.49); untested as a final decoder.
- ckinpdx ComfyUI-MMH3Tools pulled 82ea5fa -> e11c05e (Carry Attention node). `carry_bias` only works with his Looping Sampler.

## 2. Long-context news (Banodoco, 15 to 21 November) and what to implement

1. **Keystone Attention (roycho, geiru_context_suite, GPL-3, not installed).** Positive attention bias toward the keystone frame
   (clean frame 0 of clip 1) in chosen blocks only. Block map (roycho, 18 Nov): brightness 31-33, 36-38, 40-41, 44, 47; saturation
   30-32, 34-37, 39, 42, 44. Reported: 6-9 % brightness loss over 30 clips, 50-clip chains. MATLOWAI repeated it
   (https://gist.github.com/matlowai/755f1b19ddc8b04dd4cceae8b026ea21): blocks 30-39 give +12-16 %, 40-49 ~+3 %, 0-29 nothing;
   contrast, sharpness and tint go through the same blocks. roycho: "keystone alone doesn't do it; keystone attention + keystone keeps
   ~95 % of degradation away" and it pulls brightness toward the source clip, so switch the keystone at a scene change.
   **Implement (top priority):** our own small TrentNodes node (MIT, own code; read his for the idea only) that adds a bias to the
   attention scores of the pinned frame-0 keyframe tokens. In core, those rows are the `cond` segment in
   `transformer_options["minimax_h3_layout"].segments` (see `comfy/ldm/minimax/model.py` ~L405-426), and `block_index` is set per block.
   Use `optimized_attention_override` and chain any previous override, like MMH3CarryAttention does. Inputs: model, bias, blocks
   string. Test: run 16 graph WITHOUT the Clip Refiner + this node (bias on `cond`, blocks 30-39), against `pika_chain_ref2va_nocr`
   (+26.4) and run 16 (+3.5); then with the refiner. ~17 min per 10-clip run.
2. **Hybrid windows (roycho):** native context windows + `switch_step` (windows sample in sequence with locked overlap before the
   step, then all blend). Claims 2-3 min videos with motion and no burn. Worth one A/B against our chain after item 1.
3. **Temporal latent upscaler / de-rope** (Al Bashar, https://github.com/HyRo34/Minimax-H3-Temporal-Latent-Upscaler; weights George0667
   v2 temporal, CC BY-NC; needs MATLOWAI MAINodes, GPL). Flags fast-motion 17-frame blocks, inserts 2-3x latent frames, redraws only
   those, exact-recovers the frame count. Trent wants a single-clip dance test first (ask before installing MAINodes + pack + weights).
   Chain risk: the pass-2 latent becomes longer, so exact recover must run before Remember Clip Ending. Also: Alex240z autoregressive
   temporal resampler, crinklypaper H3-FastDerope (anime-tuned).
4. Watch: NikoDemon H3-Disk-Upscale (decode to disk, renoise the blend when chaining upscaled clips; unreleased); ckinpdx LongCat-style
   continuation; Chandler note: core context-window fuse accumulators sit full-length on GPU (CPU hosting is identical output).

## 3. OmniVoice Performance Transfer node (new, pushed)

TrentNodes `nodes/omnivoice_perform.py` + `utils/omnivoice/`. Commits 345892f (nodes), 3a60340 (segments, gate, room tone),
d37d64a (gate fix). Model folder `models/omnivoice/OmniVoice` (3.1 GB copy of k2-fsa/OmniVoice; weights CC-BY-NC).
Method by Nathan (SanDiegoDude), OmniVoice Manager (F:\OmniVoice-Manager, Apache-2.0): pin codebook layer 0 of the performance's
token grid (timing) at a stride, generate the rest with the target voice clone prompt.

- `strength` = pin stride in voice mode: 3 -> every 3rd token, 5 -> 6th, 6 -> 8th (6 is mine, untuned). 25 tokens/s, so stride 8 =
  one anchor per 0.32 s: the voice is freest but fast speech drifts late (up to 0.7 s).
- `max_segment_seconds` (5): cuts at the quietest point between words, Whisper per piece, exact-length join. Short pieces hold the
  target voice better.
- `gate_silence`: mutes the output where the performance is silent (threshold max(10 % of p99, 3x p10 noise floor), opens 0.15 s
  before speech, holds 0.6 s after).
- `room_tone`: the MelBandRoFormer `instruments` stem; voice is set to the original vocal level (performance minus room).
- My fix vs Nathan: no 0.1 s edge pad, so the output has the exact input length.
- Trent's best: strength 6, guidance 5.5, 128 steps, 5 s segments. Where lip timing matters, use strength 3 for that piece.

Key lesson: H3 Ref2VA audio references cannot separate words from timbre (v1-v3 failed). With a real guide video H3 copies the mouth
from the guide frames; locked audio barely moves the lips (re-render diff 2-3/255). So the **audio** must match the original timing:
OmniVoice strength is the sync control.

## 4. Character swap project (goggins-lotus.mp4, frames 1379-1927 at 24 fps)

- Shot 1 (segment frames 0-457): Jensen (`jenson.webp` sheet, first-frame swap `edit_00005_.png`), voice `jenson.wav` 4.46 s + 4.5 s.
- Cut at segment 19.083 s (frame 458). Shot 2 (458-548): long-haired character (`input/longhair_character_sheet.png`), voice
  `RAM_Altman.wav` 16:35 + 10 s. Next source cut 116 frames after 1837, so shot 2 runs are 107 frames.
- **Final:** `Downloads/character_swap_FINAL_v3_girls_comp.mp4` = v4 run frames 0-344 + clips 4-5 re-render (345-457, audio with
  "girls" at strength 3) + shot 2 run (458-548). Audio: comp voice `input/jensen_revoice_comp.wav` + shot 2 mix, -18 LUFS.
- Workflows (`user/default/workflows/H3-Context-Noise-Test/` + Downloads): v4 `H3 Character Swap Chain + OmniVoice Audio Guide v4 (test).json`
  (chain, OmniVoice runs once and is cached, per-clip trim at Video Chain song_start, Add Guide audio both passes, Join Clips gets
  the same audio); v5 `H3 Character Swap Shot 2 + OmniVoice Audio Guide v5 (test).json` (1 clip). Builders in
  `projects/h3_clip_refiner_chain/make_character_swap_voice_v4.py`, `make_shot2_swap_v5.py`.
- Resume trick: H3VideoChain `clips_already_made = N` + `clips = extra clips` re-renders from the saved ending of clip N
  (`clips` counts NEW clips). Backups of the overwritten clip 4/5 endings are in the old session scratchpad (gone after reboot).
- Shot 2 is a little softer than the 2K render; no colour match across the shot cut yet.

## 5. Open items (offer, do not start without Trent's OK)

1. Keystone Attention node (section 2.1) + walk-chain A/B. Top priority.
2. Temporal latent upscaler single-clip dance test (installs needed).
3. OmniVoice per-segment strength (auto: tighter stride where speech is fast), and a node that does the room-tone split itself.
4. H3 partial audio pin (designed, not built): pin the target audio latent by step range / time stride / strength via the audio
   noise mask, as Continue From Last Clip does for the overlap.
5. Shot-aware chain: detect cuts in the guide video (Cut Detective) and split runs at them automatically.
6. Hybrid windows A/B; second scene and seed for the run 16 recipe (still one scene, one seed).
