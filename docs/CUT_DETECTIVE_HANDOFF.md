# Cut Detective — handoff

**Status:** shipped. Cut Detective and the three new H3 inputs landed in
`e77aa95`; this doc and the dev runner in `f25fb00`. A second pass then fixed six
bugs and brought the emitted prompt in line with MiniMax's published format —
that work is §5 below. Everything here is verified unless a line says otherwise.
Written 2026-08-12, revised 2026-08-13.

Two pieces of work, coupled:

1. **Cut Detective** (`Trent/Video`) — neural shot-boundary detection with a
   film-strip preview.
2. **H3 Auto Prompt Generator** gained three inputs that change what it writes:
   `cut_times`, `first_frame_alignment`, `music_video`.

---

## 1. Read this before touching anything

Three facts cost real time to rediscover.

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

### The official H3 format is a document, not a memory

Two files, and they are the authority for everything the assembler enforces:

- `huggingface.co/MiniMaxAI/MiniMax-H3` → `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`
  (REF2VA — the mode this node writes)
- the same repo → `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`
  (shot labels, camera grammar, the alignment instruction forms)

Fetch them with plain `curl`; the HF MCP tools returned Cloudflare 502s. A
third-party "H3 PROMPT DIRECTOR" system prompt circulates in
`/mnt/c/Users/Trent/Downloads/SYSTEM_PROMPT.txt` — it reproduces the section
skeleton and the retention vocabulary correctly, but it is **not** an official
document and it invents things the real guides never say. Check the guides.

---

## 2. File map

| File | What lives there |
|---|---|
| `nodes/cut_detective.py` | The node: widgets, VIDEO/IMAGE resolution, output packing |
| `utils/cut_detect/detectors.py` | The three backends, `Shot`/`ShotList`, span folding, runt merging, the fallback policy |
| `utils/cut_detect/formats.py` | String serializers **and** the tolerant `parse_cut_times` reader |
| `utils/cut_detect/filmstrip.py` | Contact-sheet renderer (PIL): card planning, ribbon segments |
| `nodes/h3_auto_prompt.py` | `_resolve_cut_list`, `_resolve_alignment`, mode wiring |
| `utils/h3_prompt/prompts.py` | System prompt, task-context builder, the official constants |
| `utils/h3_prompt/assembler.py` | `_apply_known_times`, `enforce_alignment`, `enforce_music_video`, `enforce_task_type`, `enforce_retention_labels` |
| `utils/h3_prompt/keyframes.py` | `select_keyframes(known_boundaries=...)` |
| `tools/cut_detective_dev_run.py` | Manual runner, no ComfyUI server |
| `tools/h3_prompt_dev_run.py` | Manual runner against a **real** VLM provider |
| `tests/test_cut_detect.py` | Model-free: formats, parser, span folding, film strip, the node |
| `tests/test_cut_detect_models.py` | Gated on `TRENT_TEST_CUT_MODELS=1`; loads real weights |
| `tests/test_h3_format.py` | Conformance against the published format, rule by quoted rule |

---

## 3. Dev loop

Everything runs from the ComfyUI root without starting the server.

```bash
# whole suite
for t in cut_detect h3_assembler h3_format h3_keyframes h3_node h3_audio h3_video; do
  venv/bin/python custom_nodes/TrentNodes/tests/test_$t.py
done

# the model-backed ones (downloads weights, needs CUDA for omnishotcut)
TRENT_TEST_CUT_MODELS=1 \
  venv/bin/python custom_nodes/TrentNodes/tests/test_cut_detect_models.py

# see it on a real clip; --detector all is the fastest way to tell whether a
# disagreement is the model or the wiring
venv/bin/python custom_nodes/TrentNodes/tools/cut_detective_dev_run.py \
  --video input/clip.mp4 --detector all --thumbs-per-shot 3

# the whole H3 hand-off against a real provider, including the three new
# inputs. --profile both_ab prints two prompts side by side.
venv/bin/python custom_nodes/TrentNodes/tools/h3_prompt_dev_run.py \
  --video input/clip.mp4 --reference input/face.png \
  --cut-times "0.0, 1.5, 3.0" --first-frame-alignment --profile both_ab

# the hybrid graph: --reference stays <Picture 1> (identity only),
# --first-frame becomes <Picture 2> and is what Shot 1 opens on
venv/bin/python custom_nodes/TrentNodes/tools/h3_prompt_dev_run.py \
  --video input/clip.mp4 --reference input/character_sheet.png \
  --first-frame input/frame0_swapped.png --first-frame-alignment
```

`tests/test_h3_node.py` drives the whole H3 node against a `FakeBackend`, so
prompt-shape changes are testable without spending an API call. **Its
`first_user_text` vs `last_user_text` distinction matters:** the retry message
quotes the validator's errors back, so a test asserting on task-context wording
must read `first_user_text` or it will match on error text instead. That bit me
once already.

`tools/cut_detective_dev_run.py` re-parses its own `cut_times` output and warns
if the round trip breaks. That check is free and it is the fastest way to catch
a format drift before a paid VLM call sees it.

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
  stay in `Shot.raw_labels` — `intra`/`inter` are the shot's own, and a shot
  entered through a folded transition also carries `transition_intra` and
  `transition_inter`.
- **Transition spans are boundaries, not shots.** A 5-frame dissolve folds into
  the *following* shot as its `entry` kind plus `transition_frames`, so
  `len(shots)` is the number of real shots. The shot starts where the effect
  ends. A consequence worth remembering: shot durations therefore do **not**
  tile the clip, which is why the ribbon draws `_ribbon_segments` rather than
  shot ranges.
- **A hyphen is a range separator, never a minus sign** in `parse_cut_times`,
  so the legacy H3 `[0.000s-3.250s]` form still reads. Negative times are
  therefore not expressible; that is fine.
- **Valid JSON is never re-read as text.** `parse_cut_times` answers from the
  JSON alone, empty or not. Falling through used to scrape `"fps": 24.0` out of
  a shot-less `cuts_json` and report a cut at 24 seconds.
- **List-vs-row is decided per line**, from that line's own shape. The old
  "the whole input is one line" test silently dropped every token but the first
  once a hand-typed list wrapped.
- **New widgets go at the end of `optional`.** ComfyUI's `configure` assigns
  `widgets_values` by index, so inserting anywhere else shifts every later
  widget's saved value in existing workflows. Same for outputs: append only.
- **The alignment hook is prepended by the assembler**, not written by the VLM,
  because `strip_wrapper` deletes everything above `subject_definitions:`.
- **Music video overrides `enable_audio_prompt`.** A silent music video is
  incoherent; nobody enables the mode wanting blanked audio sections.
- **`sensitivity` is not hidden by JS on the OmniShotCut path.** It would need
  a new `js/cut_detective.js`, and canvas widget-hiding is exactly the surface
  the Vue migration keeps moving. Exposing `omnishotcut_overlap` and emitting a
  note gets the same information in Python, testable, with no frontend coupling.
- **`music_video` stayed a BOOLEAN.** Converting it to a three-way enum would
  land a saved `false` on a combo widget and fail backend validation. The
  appended `music_source` combo is equally expressive at zero migration cost.

---

## 5. What the second pass changed

### Six bugs, each reproduced before it was fixed

| Bug | Symptom |
|---|---|
| `parse_cut_times` scraped valid JSON as text | a shot-less `cuts_json` reported a phantom cut at **24.0s** — the `"fps"` field |
| list-vs-row keyed off `len(lines) == 1` | `"0.0, 1.5\n3.0, 4.5"` silently became `[0.0, 3.0]` |
| No-sentence peeling could empty a section | a legitimate `non_diegetic_music: No score is present.` was eaten into the exclusions and the prompt shipped with **five** of six mandatory sections |
| a module import ran before its own error handler | a missing `transnetv2-pytorch` raised a bare `ModuleNotFoundError`; the crafted pip hint was unreachable |
| `pending_raw or {...}` | a shot entered through a transition lost its own `raw_labels`, against this module's own docstring |
| the trim ladder ran after the music validator | it could set `non_diegetic_music = "N/A"` immediately after the validator rejected exactly that |

Two more, smaller: `min_shot_frames` did not fold runts on the classic path, so
it meant something weaker there than on the two neural ones; and
`MAX_SHEET_WIDTH` was honoured only when fitting columns automatically, so an
explicit `columns=32` at `thumb_width=640` rendered a 20,838 px sheet.

### The prompt now matches the published format

The skeleton was already right — section names, order, headers, `[Shot N] At
MM:SS.mmm,` labels, and the four tag spellings were byte-for-byte correct. The
content conventions inside three sections were not:

- **`summary` had no task-type prefix.** The guide: *"It begins with a
  square-bracketed task-type prefix"*. Nothing in the repo wrote one.
  `build_task_type()` now derives it — `reference generation` base case, plus
  `keyframe completion` under alignment and `audio reuse` when the song is
  reused — and `enforce_task_type` repairs a missing or invented one.
- **Both structural sections were paragraphs.** The guide wants one line per
  reference label in `subject_definitions` and `retention_analysis`. The worked
  example taught the wrong shape, which matters more than the rules do.
- **Retention entries had no scope parenthetical** and used the subject's name
  where the scope belongs. Official is `<Subject 1> (appears in [Shot 1],
  [Shot 3]): fully_preserved - ...`. `enforce_retention_labels` also checks the
  eight fixed markers and rejects an audio marker on a visible tag, and strips
  `(Sx)` IDs, which the guide forbids there.
- **The trailing `No ...` block is not in the format.** `grep -ic exclusion`
  across all four official documents returns **0**, and no example writes
  anything after `non_diegetic_music`. It is now off by default and lives
  behind `append_exclusions`. The irony this resolved: `UPGRADE_OVERRIDES` A
  already dropped the block, so the profile named `upgraded` was closer to
  official than the one named `official`.
- **The alignment hook was a hybrid.** The guide fixes the I2VA sentence at
  `0.00 seconds` / `[Shot 1]` and parameterizes a *different* sentence, L2VA,
  for any other moment. The old code substituted a time into the I2VA wording,
  producing a sentence in neither form. Now 0.00 emits I2VA verbatim and
  anything else emits L2VA.
- **`<Picture 1>` got a standalone entry it should not have had.** The guide:
  an image that only defines a character is cited inside `<Subject N>`, not
  given its own line. That is right when alignment is off. With alignment on it
  *is* a concrete frame anchor, which is exactly when a standalone entry and a
  `([Shot N] first frame)` retention line are correct — so the rule is now
  conditional on the toggle that already existed.

### The hybrid graph: two pictures, two jobs

A workflow can feed H3 a character reference **and** an injected opening frame.
The two are different images doing different jobs, and the node conflated them:
`first_frame_alignment` declared `<Picture 1>` to BE the opening frame, so a
multi-angle character sheet in slot 1 told H3 to open the video on a contact
sheet.

`first_frame_image` fixes it. Connect it and it becomes `<Picture 2>` — which
is how the guide's own standalone-picture example numbers it
(`<Picture 2> is the first frame of [Shot 1], showing ...`). Then:

- The alignment sentence names `<Picture 2>`, and only that image is pinned to
  a moment on the timeline.
- `<Picture 1>` keeps the normal identity-only rule: cited inside the
  `<Subject 1>` line, no standalone entry, no retention line, and no shot ever
  opens on it. The task context says this outright, because it is the mistake
  worth preventing.
- The VLM receives the injected frame as its second image, labelled, so Shot 1
  is described from the frame that actually opens the video rather than
  inferred from the sampled stills.
- **The prose repair follows the aligned tag.** This is the subtle part: under a
  single-picture hook, `<Picture 1> supplies identity only` is a contradiction
  and gets stripped. In a hybrid it is *true* and must survive. `_conflicts_with_alignment`
  takes the picture tag, so both runs are right, and a pronoun back-reference
  cannot reach past a mention of the other picture.

`alignment_picture` on `AssemblyContext` carries this. It defaults to
`"Picture 1"`, so every single-picture path is unchanged.
- **`[English]` was hardcoded** in five places while the guide asks for the
  lyric's own language; ref-mode dialogue keeps the visual label,
  `<Subject N> (Sx)`; and the camera and cut vocabularies now use the official
  fixed phrases (`with small amplitude`, `at slow speed`, `the camera cuts to`).
- **`MAX_PROMPT_CHARS = 7000` was reported to users as "H3 limit 7000".** No
  MiniMax document states any character limit. It is a TrentNodes budget and
  now says so. The word counts (350-500) *are* official and were already right.

`tests/test_h3_format.py` asserts each of these against the quoted rule, so a
future change has to argue with the spec rather than with a preference.

---

## 6. Refinement candidates, ranked

Nothing here is broken. These are the places I would look first.

### a. Real-provider end-to-end is still the biggest untested surface

Every H3 change is verified against `FakeBackend`. The measured-cut hand-off,
the alignment hook and music-video mode have **never been run against a real
VLM**, and no H3 generation has been run from a prompt these modes produced.

`tools/h3_prompt_dev_run.py` can now drive all three — that was the blocker,
and it is gone. The first thing to A/B is `--append-exclusions` with
`--profile both_ab`: the spec has no exclusions block, but H3 has no
negative-prompt field either, and only a real generation settles which wins.

### b. `<scenetrans>` and `<cutoff>` are not implemented

The guide defines both for dialogue crossing a cut and speech truncated by the
video ending. Neither appears anywhere in TrentNodes. Real spec features; they
need dialogue-heavy source material to exercise, which is why they were left.

### c. Prose alignment repair is still regex-based

`enforce_alignment` now scopes the negation to the clause it governs, reads
passive denials backwards, and chains a pronoun back-reference across one
sentence. That kills the old `\bignore\b` false positive. It is still a regex
over prose, and an unusual phrasing from a different VLM could slip through.
The task-context instruction is the first line of defence.

**Fix if it bites:** check the assembled prompt for a contradiction and raise a
retry error. Costs an API call, so it is not the default.

### d. The retry reuses the same seed

`_run_variant` passes the same `seed` on the second call, so a deterministic
provider can return byte-identical output and burn the retry. Cheap to fix;
never observed, because no provider has been driven for real yet. See (a).

### e. Backends are chosen by a private attribute

`nodes/h3_auto_prompt.py` reads `getattr(backend, "_video_upload", None)` to
pick the size cap. Only `OpenAICompatibleBackend` sets it, so Gemini — which
does have a Files API — is capped at the 18 MB inline limit rather than 100 MB.

---

## 7. Measured behaviour, for regression comparison

60s clip with 6 dissolves and 2 wipes (`osc/__assets__/demo_video7.mp4`):

| Detector | Shots | Outcome |
|---|---|---|
| omnishotcut | 14 | typed every dissolve and wipe |
| transnetv2 | 11 | missed the gradual boundaries |
| classic | 14 | wrong 14 — false 0.13s shots, missed two real cuts |

On synthetic hard cuts all three agree exactly. On a real clip
(`input/11_blade_runner_eye_scene.mp4`, 230 frames at 23 fps) all three agree on
a single genuine cut at 0.391s, re-confirmed 2026-08-13 after the parser and
detector changes; OmniShotCut ran in 5.5s on GPU, TransNetV2 in 0.94s on CPU,
classic in 0.27s.

Speed, 1800 frames: OmniShotCut ~3-8s on GPU, TransNetV2 ~5.8s on CPU, classic
~3s. OmniShotCut peaks at ~0.45 GB VRAM.

---

## 8. Related memory

`cut-detective-shot-detection` and `h3-auto-prompt-node` in
`~/.claude/projects/-home-trent-ComfyUI/memory/` carry the environment gotchas
and the H3 prompt-format findings.
