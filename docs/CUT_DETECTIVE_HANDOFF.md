# Cut Detective — handoff

**Status:** shipped and pushed, commit `e77aa95` on `main`. Everything below is
verified unless a line says otherwise. Written 2026-08-12.

Two pieces of work, coupled:

1. **Cut Detective** (`Trent/Video`) — neural shot-boundary detection with a
   film-strip preview.
2. **H3 Auto Prompt Generator** gained three inputs that change what it writes:
   `cut_times`, `first_frame_alignment`, `music_video`.

---

## 1. Read this before touching anything

Two environment facts cost real time to rediscover.

### OmniShotCut installs only with `--no-deps`

```bash
pip install --no-deps git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git
```

Its `requirements.txt` pins `transformers==4.57.3` and its `pyproject` lists
`torch`/`torchvision`. A plain install can downgrade a working ComfyUI
environment. The package itself imports neither transformers nor anything the
venv lacks — `--no-deps` is safe, and was verified by diffing `pip freeze`
before and after (one added line, nothing else).

It hardcodes `.to("cuda")`. There is no CPU path. Weights are 164 MB from
HuggingFace (`uva-cv-lab/OmniShotCut`) on first `omnishotcut.load()`.

It is **not** an active line in `requirements.txt`, deliberately — see the
comment block there.

### TransNetV2 must stay on CPU

`detectors.py` loads it with `device="cpu"` on purpose. On torch 2.11 + cu130
its dilated 3D convolutions dispatch to `aten::slow_conv3d_forward`, which has
no CUDA kernel, and it raises `NotImplementedError`. A bare dilated
`nn.Conv3d` on CUDA works fine, so this is specific to the model's conv config,
not a general torch regression. On CPU it runs at roughly 300 frames/second —
the same wall-clock as OmniShotCut on GPU for a 1800-frame clip. **Do not
"fix" this to cuda.**

---

## 2. File map

| File | What lives there |
|---|---|
| `nodes/cut_detective.py` | The node: widgets, VIDEO/IMAGE resolution, output packing |
| `utils/cut_detect/detectors.py` | The three backends, `Shot`/`ShotList`, span folding, runt merging |
| `utils/cut_detect/formats.py` | String serializers **and** the tolerant `parse_cut_times` reader |
| `utils/cut_detect/filmstrip.py` | Contact-sheet renderer (PIL) |
| `nodes/h3_auto_prompt.py` | `_resolve_cut_list`, `_resolve_alignment`, mode wiring |
| `utils/h3_prompt/prompts.py` | System prompt, task-context builder, music-video and alignment blocks |
| `utils/h3_prompt/assembler.py` | `_apply_known_times`, `enforce_alignment`, `enforce_music_video` |
| `utils/h3_prompt/keyframes.py` | `select_keyframes(known_boundaries=...)` |
| `tools/cut_detective_dev_run.py` | Manual runner, no ComfyUI server |
| `tests/test_cut_detect.py` | Model-free: formats, parser, span folding, film strip |
| `tests/test_cut_detect_models.py` | Gated on `TRENT_TEST_CUT_MODELS=1`; loads real weights |

---

## 3. Dev loop

Everything runs from the ComfyUI root without starting the server.

```bash
# whole suite
for t in cut_detect h3_assembler h3_keyframes h3_node h3_audio h3_video; do
  venv/bin/python custom_nodes/TrentNodes/tests/test_$t.py
done

# the model-backed ones (downloads weights, needs CUDA for omnishotcut)
TRENT_TEST_CUT_MODELS=1 \
  venv/bin/python custom_nodes/TrentNodes/tests/test_cut_detect_models.py

# see it on a real clip; --detector all is the fastest way to tell whether a
# disagreement is the model or the wiring
venv/bin/python custom_nodes/TrentNodes/tools/cut_detective_dev_run.py \
  --video input/clip.mp4 --detector all
```

`tests/test_h3_node.py` drives the whole H3 node against a `FakeBackend`, so
prompt-shape changes are testable without spending an API call. **Its
`first_user_text` vs `last_user_text` distinction matters:** the retry message
quotes the validator's errors back, so a test asserting on task-context wording
must read `first_user_text` or it will match on error text instead. That bit me
once already.

---

## 4. Decisions already made — don't re-litigate

- **OmniShotCut over AutoShot.** AutoShot's weights are on Baidu Cloud. Range
  F1 0.883 vs 0.814 for both TransNetV2 and AutoShot on the paper's benchmark.
- **pip install, not vendoring.** Considered and rejected: 16 files to own and
  hand-merge. Trade-off is that a venv rebuild loses it.
- **ChopCuts left alone.** It still splits a clip into per-scene MP4s. Cut
  Detective only *detects*. Any workflow using ChopCuts still works.
- **`New_Start` mid-video reads as `hard cut`.** OmniShotCut runs on
  overlapping 100-frame windows, so each window's first span carries
  `New_Start`; mid-clip it means "no relation to the previous shot could be
  read", which is still an instantaneous boundary. The untouched model labels
  stay in `Shot.raw_labels`.
- **Transition spans are boundaries, not shots.** A 5-frame dissolve folds into
  the *following* shot as its `entry` kind plus `transition_frames`, so
  `len(shots)` is the number of real shots. The shot starts where the effect
  ends.
- **A hyphen is a range separator, never a minus sign** in `parse_cut_times`,
  so the legacy H3 `[0.000s-3.250s]` form still reads. Negative times are
  therefore not expressible; that is fine.
- **The alignment hook is prepended by the assembler**, not written by the VLM,
  because `strip_wrapper` deletes everything above `subject_definitions:`.
- **Music video overrides `enable_audio_prompt`.** A silent music video is
  incoherent; nobody enables the mode wanting blanked audio sections.

---

## 5. Refinement candidates, ranked

Nothing here is broken. These are the places I would look first.

### a. `music_is_reference` is inferred, not declared

`h3_auto_prompt.py` derives "the song is fed to H3 as `<Audio 1>`" from the
`audio` input being connected. The reasoning: if you hand the node the actual
song, your H3 graph is being fed it too. That is the common case, not a
guarantee. If it guesses wrong you get a prompt referencing an `<Audio 1>` that
H3 never receives.

**Fix if it bites:** promote it to an explicit widget, or a three-way
`music_video` enum (`off` / `generate_score` / `reuse_audio_1`).

### b. Prose alignment repair is regex-based

`enforce_alignment` strips sentences denying `<Picture 1>` its framing,
background or lighting. It catches the phrasings the system prompt's own worked
example produces. An unusual phrasing from a different VLM could slip through
and leave the prompt self-contradictory. The task-context instruction is the
first line of defence; the regex is the second.

**Fix if it bites:** widen `_ALIGNMENT_NEGATION`, or check the assembled prompt
for a contradiction and raise a retry error.

### c. `sensitivity` does nothing for OmniShotCut

It predicts shot ranges directly, with no threshold to turn. The widget says
so, but a user turning the knob and seeing no change is a reasonable
complaint. `min_shot_frames` is the only post-hoc control on that path.

**Options:** hide the widget when `detector == "omnishotcut"` (needs JS), or
expose the model's `overlap` instead, which does change results at window
edges.

### d. Film strip is one thumbnail per shot

A 60-shot clip gives a 60-thumbnail sheet, which is fine, but a long shot shows
only its first frame. Multiple thumbnails per shot were designed for and not
built. `render_film_strip` would need a `thumbs_per_shot` argument and a
layout pass.

### e. Real-provider end-to-end is untested

Every H3 change is verified against `FakeBackend`. The measured-cut hand-off,
the alignment hook and music-video mode have **never been run against a real
VLM**, and no H3 generation has been run from a prompt these modes produced.
That is the biggest untested surface. `tools/h3_prompt_dev_run.py` drives a
real provider.

### f. Dissolves vs the exclusion block

The stock exclusions include "No morphing, compositing seams, dissolves, or
crossfades." When Cut Detective measures a dissolve, the system prompt tells
the model to write the incoming shot plainly and never describe the effect, so
the two are consistent. It has not been checked against a real generation
though — worth a look if dissolve-heavy source material behaves oddly.

---

## 6. Measured behaviour, for regression comparison

60s clip with 6 dissolves and 2 wipes (`osc/__assets__/demo_video7.mp4`):

| Detector | Shots | Outcome |
|---|---|---|
| omnishotcut | 14 | typed every dissolve and wipe |
| transnetv2 | 11 | missed the gradual boundaries |
| classic | 14 | wrong 14 — false 0.13s shots, missed two real cuts |

On synthetic hard cuts all three agree exactly. On a real clip
(`input/11_blade_runner_eye_scene.mp4`) OmniShotCut found a genuine cut at
0.391s that eyeballing the film strip confirmed.

Speed, 1800 frames: OmniShotCut ~3-8s on GPU, TransNetV2 ~5.8s on CPU, classic
~3s. OmniShotCut peaks at ~0.45 GB VRAM.

---

## 7. Related memory

`cut-detective-shot-detection` and `h3-auto-prompt-node` in
`~/.claude/projects/-home-trent-ComfyUI/memory/` carry the environment gotchas
and the H3 prompt-format findings.
