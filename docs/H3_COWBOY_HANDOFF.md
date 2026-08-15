# Ultimate H3 Cowboy Promptor — handoff

**Status:** Phases 0–3, 5 and 6 built, tested and working. Phase 4 (extraction
and tooling) is **not built** — see §7. Written 2026-08-14, on top of
`0cded8a`; Phases 5 and 6 landed the same day. Everything below is verified
unless a line says otherwise — and nothing here has yet driven a real H3
generation.

The older **H3 Auto Prompt Generator** is untouched and still installed. Both
nodes are meant to coexist; §2 says which to use when.

---

## 1. Why this node exists

`H3AutoPromptGenerator` writes one job: replace the performer in a video with
the person in a reference image. That is baked in at four levels — the system
prompt's opening thesis, three `required` widgets (`reference_image`,
`subject_name`, `subject_wardrobe`), per-shot enforcement hardcoding the literal
`<Subject 1>`, and a `build_task_type()` reaching 3 of the spec's 6 task types.

The evidence that settled the design: I ran **MiniMax's own complete reference
example** (`guide_ref` §7 — a coffee-shop *environment*, a Samoyed, two people,
one audio reference) through the old assembler.

```
subject_name='Aria Voss'     retry_errors=3  warnings=1
      RETRY: Shot 1 of detailed_description does not mention Aria Voss <Subject 1>.
      RETRY: Shot 2 of detailed_description does not mention Aria Voss <Subject 1>.
      RETRY: Shot 3 of detailed_description does not mention Aria Voss <Subject 1>.
```

The node would spend a real API call retrying a prompt the spec's authors wrote.
Through the new one, that same text produces **zero errors, zero warnings, and
round-trips byte-identically** (`test_minimax_own_example_passes_clean`).

---

## 2. Which node to use

| | |
|---|---|
| **H3 Auto Prompt Generator** | Character replacement you already have dialled in. Frozen; will not change. |
| **Ultimate H3 Cowboy Promptor** | Everything else: objects, environments, styles, actions, several subjects at once, video editing and continuation — and, in the `base_*` modes, a video written from text with no reference assets at all. |

Byte parity between them is an **explicit non-goal**. You cannot have a
subject-agnostic system prompt and byte-identical output from the same VLM, and
a byte-identical clone would be a second copy of a node you already have.

---

## 3. Read this before touching anything

### The official format is a document, not a memory

Two files are the authority for everything the validators enforce:

- `huggingface.co/MiniMaxAI/MiniMax-H3` → `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`
- same repo → `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`

Fetch with plain `curl`; the HF MCP tools return Cloudflare 502s. A third-party
"H3 PROMPT DIRECTOR" file circulates in `/mnt/c/Users/Trent/Downloads/` — it
reproduces the skeleton correctly but invents things the guides never say. Check
the guides.

`utils/h3_cowboy/spec.py` carries every constant with its citation. **If the
guide does not say it, that file does not assert it** — judgement calls are
labelled as such.

### Two constraints freeze `utils/h3_prompt/`

These are why this is a separate package rather than a generalization in place:

- `tests/test_h3_format.py` slices `prompts.SYSTEM_PROMPT` on `"## WORKED
  EXAMPLE"`. The old person-swap example is a **tested contract**.
- `tests/test_h3_format.py:191` calls `prompts.build_task_type(reuse, align)`
  **positionally**. Its first two parameters are frozen.

`test_h3_cowboy_ref.py::test_the_old_package_is_still_ref_only_and_unchanged`
asserts both, so an accidental generalization fails in the file the person doing
it is already reading.

### The golden-hash lock

`tests/test_h3_assembler.py::test_the_old_assembler_output_is_byte_stable`
hashes two full prompts from the old node. This package **shares** the old
assembler's subject-agnostic helpers rather than forking them, and Phase 4 moves
those helpers to a common core. The hashes turn "I think that refactor was
mechanical" into a fact. I verified the lock bites: reordering `SECTION_ORDER`
or changing the header join both fail it.

Regenerate the hashes **only** when the old node's behaviour is deliberately
changed, and say so in the commit.

---

## 4. File map

| File | What lives there |
|---|---|
| `utils/h3_cowboy/spec.py` | The format as data: both skeletons, subject kinds and their feature cards, asset roles → task types, the editing fixed sentences, the base instruction lines and `render_instruction_line()`, all six official worked examples |
| `utils/h3_cowboy/subjects.py` | `SubjectSpec`, the DSL parser, `bind_images`, and the node face's rows (`SubjectRow`, `subjects_from_rows`, `merge_text_subjects`) |
| `utils/h3_cowboy/wiring.py` | The sampler's arithmetic as plain functions: `snap_length`, `canvas_for`, `trim_reference_frames`, `build_label_map` |
| `utils/h3_cowboy/prompts_ref.py` | Composed ref system prompt + task context |
| `utils/h3_cowboy/prompts_base.py` | Composed base system prompt + task context, per sub-mode |
| `utils/h3_cowboy/assembler.py` | `CowboyContext`, `_process_ref`, `_process_base`, the R1–R5 budget, `enforce_task_type_set`, `enforce_video_edit`, the base parse/emit |
| `nodes/ultimate_h3_cowboy_promptor.py` | The node: 6 subject rows and image slots, roles, music video, both skeletons, retry loop, 18 outputs |
| `js/h3_cowboy.js` | Which widgets are visible: rows grow as you fill them, music appears with `music_video`, each mode hides the other's settings |
| `tests/test_h3_cowboy_subjects.py` | DSL parsing and image binding (18) |
| `tests/test_h3_cowboy_ref.py` | Ref conformance, anchored on both official examples (32) |
| `tests/test_h3_cowboy_node.py` | End-to-end against a fake backend (16) |
| `tests/test_h3_cowboy_base.py` | Base conformance, anchored on all four official cases (41) |
| `tests/test_h3_wiring.py` | The sampler arithmetic, checked against a re-implementation of the original (20) |
| `tests/h3_cowboy_js/run.mjs` | `js/h3_cowboy.js` against a mocked frontend (`node tests/h3_cowboy_js/run.mjs`) |

**Reused unchanged** (verified subject-agnostic): `utils/h3_prompt/keyframes.py`,
`backends.py`, `imaging.py`, `audio_io.py`, and
`utils/cut_detect/formats.py::parse_cut_times`.

**Deliberately not reused:** `enforce_subject_per_shot`, `enforce_wardrobe`, the
whole `_ALIGNMENT_*` / `enforce_alignment` block, `finalize_exclusions`,
`STOCK_EXCLUSIONS`, `SYSTEM_PROMPT`, `build_alignment_hook`, and — after it
caused a real bug, see §6 — `enforce_task_type`.

---

## 5. Dev loop

```bash
cd /home/trent/ComfyUI
for t in cut_detect h3_assembler h3_format h3_keyframes h3_node h3_audio h3_video \
         h3_cowboy_subjects h3_cowboy_ref h3_cowboy_node h3_cowboy_base \
         h3_wiring; do
  venv/bin/python custom_nodes/TrentNodes/tests/test_$t.py
done
node custom_nodes/TrentNodes/tests/h3_cowboy_js/run.mjs
```

The anchor tests are the ones that matter. All of them run MiniMax's own text
through the validator and demand zero retry errors:

- `test_minimax_own_example_passes_clean` — `guide_ref` §7, four subjects of
  three kinds
- `test_the_official_editing_example_passes_clean` — the README's `case-Ref2VA`
  H3-Context-IR output, the only official example of the editing shape
- `test_the_four_official_cases_pass_clean` and
  `test_the_four_official_cases_round_trip_byte_for_byte` — `guide_base` §5,
  all four base sub-modes, instruction line included

If any of them starts failing, the validator is wrong, not the example.

---

## 6. Decisions already made — don't re-litigate

- **`subject_N_image` IS `<Picture N>`.** One rule, no exceptions. The obvious
  alternative — number wired slots 1..M in wiring order — reads fine until
  someone also writes an explicit `@Picture 1`, and then two subjects cite one
  tag for two different images. I hit exactly that during development. A gap in
  the slots leaves a gap in the numbering and warns.
- **A label is DEFINED when it opens a line, not when it is mentioned.** MiniMax's
  own example cites four pictures and two videos inside subject lines and gives
  none of them an entry, per `guide_ref` §2. My first cut required a retention
  line per *mentioned* tag and the official example caught it.
- **Task types are derived from declared asset roles, never typed.** `guide_ref`
  §3: "The mere presence of video or audio does not automatically create a
  corresponding task type." The same wired video can be an edit source, a
  continuation source, a rhythm reference, or just where a subject came from.
- **The task-type prefix is compared as a SET.** The guide states no ordering
  rule and its own output writes `[video editing + audio reference + audio
  reuse]` — out of its own table order. The old package's `enforce_task_type`
  compares the joined string and "corrects" a legal permutation; that is why this
  package has its own `enforce_task_type_set`. We still *emit* one fixed order,
  purely so the same graph diffs cleanly.
- **The three video-editing requirements are injected, not hoped for.** They are
  fixed strings with no content to invent, so a retry over them would be a wasted
  API call.
- **No exclusions block, and no `append_exclusions` widget.** The word
  "exclusion" appears in none of the guides, and nothing follows
  `non_diegetic_music` in any official example. `constraint_notes` replaces it:
  the text is folded into the task context as a positive assertion and never
  becomes output. The guide's own mechanism for "do not change X" is the
  retention marker.
- **No alignment prose repair.** The old package's ~150 lines of negation
  regexes exist to fight its own worked example and its own stock exclusion.
  Delete the cause, and the cure goes with it. Ref mode also has **no instruction
  line above the sections** — both official ref examples start at
  `subject_definitions:`; the alignment sentence is a base-mode construct.
- **A later shot OPENS with its cut.** `guide_base` §4.2 gives the form as a
  template — `[Shot 2] At 00:03.500, the camera cuts to...` — and `guide_ref`
  §5.1 defers to it. Shot label, timestamp, cut phrase, then what the new shot
  shows, one sentence. All six official worked examples do it and none writes
  the cut after the description. Both system prompts state the order and show
  the template, and `_check_cut_phrasing` warns in **both** modes — separately
  for "no cut phrase" and for "a cut phrase that does not open the shot",
  because those need different fixes. The five phrases are lowercase prose; a
  screenplay slug (`CUT TO:`) is off-vocabulary. Details in
  `docs/H3_COWBOY_BASE_MODE_HANDOFF.md` §7.1.
- **Five aggregated retry classes, one message each.** The old node emits a
  retry error per shot, twice, so a six-shot clip can produce twelve. Retries
  cost money.
- **The better attempt wins.** The old loop overwrites unconditionally, so a
  worse retry can replace a good first pass.

---

## 7. What is built, and what is not

### Phase 3 — base mode (T2VA / I2VA / FL2VA / L2VA) — **built**

All four modes run. `prompts_base.py` composes a per-sub-mode system prompt,
`_process_base()` parses and emits the three-field skeleton with its
same-line headers, the instruction line is rendered from the finished body,
and everything on the node is optional so T2VA needs no clip and no images.
All four of `guide_base` §5's cases pass clean and round-trip byte-identically.

`docs/H3_COWBOY_BASE_MODE_HANDOFF.md` is the record: what shipped, the four
things that look like typos and are not, what must not be validated, and the
open risk that none of it has faced a real H3 generation.

The one thing base mode still defers is `<scenetrans>` / `<cutoff>` — see the
non-goals below.

### Phase 5 — reference wiring — **built**

The promptor names its assets `<Picture 1>`, `<Video 1>`, `<Audio 1>`.
`MiniMaxH3ReferenceToVideo` assigns the same tags **independently**, from the
order its own sockets are filled, and nothing checked that the two agreed. A
disagreement produced a correct-looking prompt over the wrong image, with no
artefact to inspect afterwards.

The node now hands every asset back out: `ref_image_1..6`, `ref_video` (as
IMAGE frames), `ref_video_audio`, `ref_audio`, plus `width`, `height`, `length`
and a `label_map` you can read against the prompt. They are appended after the
original five outputs, which never move. It also snaps the duration to H3's
17k+5 frame grid **before** writing the prompt, so a 2.00-second request stops
claiming 2.00 when it really renders 2.33.

`docs/H3_REFERENCE_WIRING_HANDOFF.md` is the record, every sampler fact cited
to `file:line`, with the build log and the four deliberate differences in §10.
The standalone `nodes/h3_reference_wiring.py` is deleted; its arithmetic lives
on in `utils/h3_cowboy/wiring.py`.

### Phase 6 — the node face — **built**

The subjects field used to be a typed DSL and nothing else, which is a lot to
learn before the first prompt. There are now six **rows**, one per picture
slot, each a kind, a name and a description:

| Row widget | Is |
|---|---|
| `subject_N_kind` | what row N is, in plain words. `character` and `environment` sit on top and canonicalise to the guide's `person` and `scene`; every other kind keeps its own name |
| `subject_N_name` | optional, and often blank — MiniMax's own example names nobody |
| `subject_N_description` | the features to keep. Typing here is what makes row N a subject |

The numbering promise is unchanged and now runs the whole way through:
**row N is `<Subject N>` is `subject_N_image` is `<Picture N>`.** A skipped row
leaves a hole rather than closing it, and warns — closing it would point the
prompt at a different image, which is exactly the failure Phase 5 exists to
stop.

Three consequences worth knowing:

- **A row with no image is still a subject.** That is how you ask for a style
  or an action, which no picture can supply.
- **An empty row is not a subject**, whatever the row count says. That is what
  lets a workflow saved before rows existed behave exactly as it did, and what
  lets the count sit anywhere without inventing subjects.
- **The typed `subjects` field survives as the escape hatch**, marked advanced.
  It is for what a row cannot say: a seventh subject, or one citing two
  pictures at once. Typed lines are numbered *after* the filled rows.

`music_video` is ported from the older node and rewritten for several subjects
of any kind: the singer is whichever subject can actually sing rather than an
assumed `<Subject 1>`, and a reused track goes through `audio_role` so it
reaches the task type by the same path a reused clip track does. `lyrics`,
`music_description` and `music_source` come with it. Two new rules run in the
assembler: `R6 MUSIC` retries a music video scored `N/A`, and lyrics repeated
in an audio section are stripped rather than argued with.

The rest is visibility, in `js/h3_cowboy.js` and in `"advanced": True` on the
widgets nobody touches twice. Nothing is hidden that changes an outcome
silently — Python reads all six rows whatever the row count says, and every
ignored widget still warns.

### Phase 4 — extraction and tooling — **not built**

- Move genuinely shared helpers to `utils/h3_prompt/core.py`, re-export from
  `assembler.py`, verify against the golden hashes.
- `tools/h3_cowboy_dev_run.py` on the `tools/h3_prompt_dev_run.py` model.
- A node card. The README entry landed with Phase 6.

### Deliberate non-goals

- Byte parity with the old node (§2).
- `<scenetrans>` / `<cutoff>` — real spec features, but **no worked example
  exists in any official source**, so there is nothing to check an
  implementation against. They tempted us during base mode, and `guide_base`
  Case 1 settles it: it writes the continuity phrasing *without* the tag
  ("the baker's final words carry over from the previous shot"). Still
  deferred.
- A companion `H3SubjectCowboy` node. `parse_subjects()` returns a plain list of
  dataclasses so it costs nothing later; do not build it now.
- Dynamic JS **sockets** for subjects. Six fixed optional slots, no exceptions —
  a workflow saved with 5 subjects and loaded where the JS failed would silently
  lose inputs. `js/h3_cowboy.js` (Phase 6) does not break this: it only sets
  `hidden` on widgets that are always present. Nothing is added, removed or
  reordered, so a failed extension leaves a node that is ugly and complete
  rather than tidy and lossy.
- Deleting `AssemblyContext.music_is_reference`. It **is** dead code — set by the
  old node, documented as stripping a dangling `<Audio 1>`, read by nothing in
  `assembler.py`. Removing it is a change to the frozen node for zero user
  benefit. Recorded here; leave it.

---

## 8. What must NOT be validated

Each of these would fight the spec or manufacture false retries. Several are
things the old node *does* check.

- **Per-shot subject presence.** `guide_ref` §7's own example has `<Subject 2>`
  absent from Shot 3, stated outright in its retention line.
- **Subject proper names.** Not a spec concept; §7 uses none.
- **Frame position for scene, style, action, expression or interface subjects.**
  A scene *is* the frame. Even for spatial kinds this is a warning, at first
  appearance only — the position regex false-negatives on legal phrasing.
- **The 350–500 word range for editing runs**, explicitly exempted by §5.2.
- **Task-type prefix ordering** — compare sets (§6).
- **Shot count against a measured list outside declared `video editing`.** The
  target's structure is not bound to a reference video's.
- **`non_diegetic_music: N/A`** — legal per `guide_base` §4.7, and correct for a
  diegetic on-camera performance.
- **Subject count against asset count**, in either direction. §2.1: one subject
  may come from several assets, and one asset may provide several subjects.
- **"Every wired video gets a `<Video N>` line."** Flatly wrong — §7 cites two
  videos and gives neither a line.
- **A character count.** No MiniMax document states one. The 7000-char budget is
  a TrentNodes number and the warning says so.

---

## 9. Measured behaviour, for regression comparison

| Input | Old node | This node |
|---|---|---|
| `guide_ref` §7 complete example | 3 retry errors, 1 warning | **0 errors, 0 warnings**, byte-identical round trip |
| README `case-Ref2VA` editing output | not applicable (no editing path) | **0 retry errors**, no injections needed |
| `guide_base` §5 Cases 1–4 | not applicable (no base path) | **0 errors, 0 warnings** each, byte-identical round trip including the instruction line |

Composed ref system prompt with three declared kinds: ~11,100 characters, one
worked example, three feature cards. Composed base system prompt: 7,000–7,900
characters depending on sub-mode, one worked example, one body-shape block.

---

## 10. Related memory

`h3-auto-prompt-node` and `cut-detective-shot-detection` in
`~/.claude/projects/-home-trent-ComfyUI/memory/` carry the environment gotchas
and the older node's prompt-format findings.

**Nothing here has been verified against a real H3 generation.** The format is
checked against MiniMax's published guides and locked by their own examples, but
"matches the spec" and "produces a better video" are different claims and only
the first is established. That remains the largest untested surface.
