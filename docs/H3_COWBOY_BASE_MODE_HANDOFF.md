# Ultimate H3 Cowboy Promptor — Phase 3: base mode

**Status:** **built, tested and working.** Designed 2026-08-14, built the
same day. 41 base-mode tests pass; 300 pass across the repo, up from 258.
Everything here is verified against `guide_base` unless a line says
otherwise, and **nothing has been checked against a real H3 generation** —
see §9.

Read `docs/H3_COWBOY_HANDOFF.md` first — it covers the package, the frozen
older node, and the golden-hash lock. This document is only about base mode.

`h3_mode` now offers the four base values and they run. This document was
written as a design and has been rewritten as a record: every section says
what shipped, and the traps are kept because they are still the way to get
this wrong.

---

## 1. What base mode is

Ref mode rewrites a job that has reference assets. Base mode writes a
video from text, optionally anchored to one or two frames. It is **not a
flag on ref mode** — it is a different skeleton, a different system prompt,
and a different H3 checkpoint (`H3-Base-FL2VA` vs `H3-Base-Ref2VA`).

Four sub-modes, from `guide_base` §1:

| Mode | Anchor |
|---|---|
| **T2VA** | none — build the whole timeline from text |
| **I2VA** | `<Picture 1>` is the **first** frame |
| **FL2VA** | `Picture 1` is the first frame, `Picture 2` the last |
| **L2VA** | `<Picture 1>` is the **last** frame |

There are no reference labels: no `<Subject N>`, no `<Video N>`, no
`<Audio N>`, no task-type prefix, no retention analysis. Only `<Picture N>`
survives, and only as a frame anchor.

---

## 2. The skeleton, and the trap in it

Three fields, `guide_base` §2.2:

```
{instruction line}          <- omitted for T2VA
                            <- exactly ONE blank line
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames...

overall_soundscape: Steady rain taps against the café windows...

non_diegetic_music: Sparse piano notes at a slow tempo...
```

### **The trap: base headers put content on the SAME line.**

Ref mode is `subject_definitions:\n<Subject 1> is...`. Base mode is
`integrated_multimodal_description: [Shot 1] Live-action...` — header, colon,
space, content, all one line. I counted it in the guide: 5 same-line
`integrated_multimodal_description:` and 6 same-line `overall_soundscape:`
against 0 newline-separated ones; ref mode is the reverse.

**Consequence:** `reassemble()` in `utils/h3_prompt/assembler.py` does

```python
parts.append(f"{key}:\n{content}")
```

so it **cannot be reused for base mode**. Neither can `parse_sections()`,
which walks `SECTION_ORDER` and expects a bare header line. Base mode has
its own `_reassemble_base()` and `_parse_base_sections()`. This is the
single most likely way to ship a subtly wrong base prompt, because it will
look right in a diff — `test_the_headers_carry_their_content_on_the_same_line`
is what catches it.

A model that writes the ref shape anyway is repaired, not retried:
`_parse_base_sections` pulls the header back onto one line and records the
fix.

---

## 3. The four instruction lines

`guide_base` §2.1. These are **fixed strings the guide says are "always
used"** — not paraphrases of a pattern. They live in `spec.py` as
`INSTRUCTION_T2VA` / `_I2VA` / `_FL2VA` / `_L2VA`, and
`spec.render_instruction_line()` renders one.

### 3.1 When each fires

Base mode has no way to express a style or subject reference, so **every
wired picture is a frame anchor**:

| Pictures wired | Sub-mode | Line |
|---|---|---|
| 0 | T2VA | *(none)* |
| 1, declared first frame | I2VA | fixed, zero parameters |
| 1, declared last frame | L2VA | takes `N` and `S.SS` |
| 2 | FL2VA | takes `N` and `S.SS` |

`h3_mode` is where the user says which, and it is authoritative.
`base_picture_role` (`first_frame` / `last_frame`) exists as the
cross-check the design asked for: a disagreement warns, names the mode
that would match, and follows `h3_mode`. Its default doubles as "unset",
so `base_L2VA` does not warn about a widget nobody touched.

### 3.2 Four things that look like typos

Every one is in MiniMax's own text. `spec.py` carries this warning inline
and `tests/test_h3_cowboy_base.py` locks all four.

1. **The separator is an em dash `—` (U+2014), not a hyphen.** The older
   package ships `" - "` in `utils/h3_prompt/prompts.py` — that is a real
   deviation, already noted in the older handoff.
   `test_the_separator_is_an_em_dash_and_never_a_hyphen` also pins that
   I2VA's line has no dash of any kind.

2. **FL2VA writes bare `Picture 1` and `Shot 1`** — no angle brackets, no
   square brackets — while I2VA and L2VA bracket both. It is systematic, not
   a one-off: it holds in the instruction line *and* in the body of Case 3
   ("begins in the position and framing established by Picture 1"), while
   Case 4's body writes `<Picture 1>`.

   We reproduce it. The failure mode of guessing wrong is a silently worse
   generation no validator can detect, so `fl2va_normalize_picture_tags`
   (default **off**) rewrites both tags to the bracketed forms for an A/B,
   and the setting is recorded in `analysis_json`.

   **Implementation note:** `fix_reference_tags` only rewrites
   already-delimited forms (`[Picture 1]`, `(Picture 1)`), so it leaves bare
   mentions alone, and nothing was added to normalise them. The FL2VA
   validators match both forms through `_PICTURE_ANY_RE`.

3. **`N` and `S.SS` are only knowable after the body exists.**
   `guide_base` §2.1: "`N` is the index of the actual final shot, and
   `S.SS` is the effective video duration formatted to exactly two decimal
   places."

   So the pipeline is: call the model **without** the line → parse the body →
   count `[Shot N]` labels → render the line → prepend. The system prompt
   tells the model not to write it, and any line it writes anyway is
   stripped (the worked examples contain one, so imitation is likely).
   `test_a_model_written_instruction_line_is_stripped_and_replaced` feeds it
   a line claiming `[Shot 9]` and `99.00-second` and demands the real
   numbers back.

   Case 3 is a single-shot 8-second clip and renders `(from Shot 1)` for
   **both** pictures; Case 4 is a single-shot 6-second clip and renders
   `(from [Shot 1])` with `6.00-second`. Both confirm N is the *final* shot,
   which happens to be 1 there.

   The older node renders its hook before the call, which is wrong whenever
   the model writes a different shot count than assumed. It was not copied.

4. **One blank line between the instruction and the body**, per §2.1: "The
   instruction must be the first line of the final prompt, followed by one
   blank line before the core fields."

### 3.3 Per-mode body shape

`guide_base` §3, carried into the system prompt as `prompts_base.BODY_SHAPE`
— one block per sub-mode, and only that mode's block ships.

- **I2VA** — first-frame anchor → action onset → continuous development →
  result or reaction. "`<Picture 1>` is the actual first frame of the video at
  0.00 seconds and belongs to `[Shot 1]`."
- **FL2VA** — first-frame state → observable intermediate changes →
  progressively narrowing differences → last-frame state. Plus a hard
  preference: "FL2VA generally favors a single shot… Use multiple shots only
  when they are explicitly specified. The last frame must be reached by the
  final `[Shot N]` at the end of the video."
- **L2VA** — plausible preceding state → explicit action and transition path →
  gradual convergence in the final shot → last-frame landing. And the trap:
  "`<Picture 1>` is the final frame of the video and belongs to the last
  `[Shot N]`; **it does not inherently belong to Shot 1**."
- **T2VA** — no anchor; "You may add scene, character, action, and sound
  details that remain consistent with the user's intent."

---

## 4. Body rules that differ from ref mode

Most of `guide_base` §4 is shared with ref mode. These are the deltas, and
they are written fresh in `prompts_base.SHARED_BODY` rather than imported
from the ref prompt:

- **Style goes INSIDE `[Shot 1]`**, on the same line after the label:
  "`[Shot 1] Live-action, cinematic, a medium-wide shot frames...`". Ref mode
  puts it in one or two sentences *before* `[Shot 1]`; `guide_ref` §5.2 tables
  the difference explicitly. A model that writes the ref habit is repaired
  deterministically by `_force_style_inside_shot_one`. Common styles the
  guide names: `Cinematic`, `live-action`, `2D-animated`, `3D CG`,
  `claymation`, `watercolor`, `vintage film`.
- **Camera amplitude and speed are optional.** §4.3: "Add amplitude and speed
  only when they are meaningful; medium amplitude and normal speed are usually
  omitted." The ref system prompt tells the model to qualify every move —
  slightly over-strict, and deliberately not copied.
- **Voiceover has an exact required phrase**: `says in an off-screen
  voiceover`, and immediately after every voiceover `<d>` block, state that the
  on-screen character's lips remain closed.
- **On-screen text goes in English double quotes, verbatim, untranslated** —
  §4.5, e.g. `A red neon sign reading "营业中" glows above the doorway.`
- **Compound speaker IDs** `(S1,S2)` when several already-numbered speakers
  vocalise together. Characters who never vocalise get no ID.
- **`overall_soundscape`**: 1–4 sentences, one paragraph. `N/A` **only** when
  the user explicitly requests complete silence.
- **`non_diegetic_music`**: 1–3 sentences, instrumentation/speed/rhythm/
  dynamics. No abstract mood words. `N/A` when there is no non-diegetic music.

---

## 5. The four official worked examples

One ships per run, selected by sub-mode, verbatim — same rule as ref mode.
They live in `spec.py` as `EXAMPLE_BASE_T2VA` … `EXAMPLE_BASE_L2VA`, keyed
by `EXAMPLE_FOR_BASE_MODE`, and they were extracted mechanically rather
than transcribed:

```python
# from the scratchpad copy of guide_base_en.md
text = open("guide_base_en.md").read()
cases = text.split("## 5. Cases", 1)[1]
for name in ("Case 1: T2VA", "Case 2: I2VA", "Case 3: FL2VA", "Case 4: L2VA"):
    body = cases.split(f"### {name}", 1)[1].split("```text", 1)[1]
    print(name, "->", repr(body.split("```", 1)[0].strip()[:60]))
```

Each is shipped whole, **instruction line included** — the line teaches
where it sits — and the output contract then tells the model not to write
one. `test_the_embedded_examples_are_the_guides_text` pins the shape of all
four so a tidy-up cannot quietly change what they teach.

| Case | Proves |
|---|---|
| 1 T2VA | no instruction line at all; same-line headers |
| 2 I2VA | the I2VA line verbatim; `<Picture 1>` cited inside `[Shot 1]` |
| 3 FL2VA | bare `Picture 1` / `Shot 1` in both the line and the body; single shot; `non_diegetic_music: N/A` |
| 4 L2VA | bracketed `<Picture 1>` / `[Shot 1]`; the picture landing at the END |

---

## 6. What was built

```
utils/h3_cowboy/prompts_base.py       system prompt, task context,
                                      re-export of render_instruction_line
utils/h3_cowboy/spec.py               + the four examples, + example_for("base_*"),
                                      + render_instruction_line()
utils/h3_cowboy/assembler.py          + _process_base(), + base parse/emit,
                                      + the base validators
nodes/ultimate_h3_cowboy_promptor.py  the four modes unlocked
tests/test_h3_cowboy_base.py          new (40)
```

`render_instruction_line()` ended up in `spec.py`, beside the four
templates and the comment that explains their em dash and their bare tags,
and is re-exported from `prompts_base` so the import path the design named
also works.

### Signatures

```python
# prompts_base.py
def build_system_prompt(sub_mode: str) -> str: ...
def build_user_context(sub_mode, target_description, duration_seconds, fps,
                       frame_timestamps=None, dialogue_text="",
                       constraint_notes="") -> str: ...

# spec.py  (re-exported from prompts_base)
def render_instruction_line(sub_mode: str, final_shot: int = 1,
                            duration_seconds: float = 0.0) -> str: ...

# assembler.py
def _process_base(raw_text: str, ctx: CowboyContext) -> CowboyResult: ...
def _parse_base_sections(text, fixes, warnings) -> Dict[str, str]: ...
def _reassemble_base(sections, instruction: str = "") -> str: ...
```

`_parse_base_sections` and `_reassemble_base` exist **because of §2's trap**.
The ref versions were not parameterised — the header-join differs, the
section list differs, and `strip_wrapper`'s `_PREAMBLE_RE` hardcodes
`subject_definitions`, so base mode has `_strip_base_wrapper` too.

`process()` branches at the top on `ctx.mode`. If base output ran the ref
path, `parse_sections` would report four missing sections and synthesize
`overall_soundscape` and `non_diegetic_music` on top of ones that already
exist.

### Node changes

- The four `base_*` values run; the `RuntimeError` is gone.
- `base_picture_role` and `fl2va_normalize_picture_tags` are appended last
  in `optional`, so no saved graph's widget order shifts.
- **Everything is optional.** `_resolve_frames` took a `required` parameter
  rather than a loosened error, because ref mode genuinely cannot run
  without a clip and that message is worth keeping sharp. With no clip,
  duration comes from `duration_override`; with neither, the node uses 5
  seconds and warns, naming the widget to set — that number reaches the
  instruction line as `S.SS`.
- Ref-only widgets **warn, never error**: `subjects`, `video_role`,
  `audio_role`, a wired video, a wired audio. A wired clip still supplies
  duration and context frames, labelled "NOT a frame of the target video
  and not citable".
- The one hard error is a mode with too few anchors — `base_FL2VA` with
  fewer than two pictures, `base_I2VA` or `base_L2VA` with none — and it
  names the slot to connect.
- `subject_N_image` **IS** `<Picture N>` here too: slot 1 is the anchor,
  slot 2 is FL2VA's last frame, and slots past what the mode uses warn.
- `cut_times` has no source clip to measure in base mode, so it reads as
  the shot structure being *asked for*. More than one entry is
  `guide_base` §3.2's "explicitly specified", which is what turns FL2VA's
  single-shot warning off.
- `h3_checkpoint_hint` already emitted the right string per mode.

---

## 7. Validation

The R1–R5 budget and the rule that **warnings never trigger a retry** are
unchanged.

**Retry (R1):** anything other than exactly three field headers, in order,
with content; any ref-mode label or header (`<Subject N>`, `<Video N>`,
`<Audio N>`, `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`); a `<Picture N>` the sub-mode does not have — which
is every picture in T2VA, since `guide_ref` §5.2 says T2VA "does not use
full-reference labels". Every instance lands in one message.

A preamble carrying ref sections is **not** stripped, on purpose: deleting
it would hide the one thing worth retrying over behind a prompt that looks
almost right.

**Retry (R3):** user-supplied dialogue absent from any `<d>`. `_check_verbatim`
is shared with ref mode; base mode passes it a different body key.

**Retry (R5):** no usable text.

**Deterministic fix:** the style sentence written before `[Shot 1]` (moved
inside); a header written on its own line (pulled back); `[Shot 1]`
carrying a timestamp (stripped); later shots not `At MM:SS.mmm,` or not
strictly increasing (repaired by `normalize_shot_labels`); a model-written
instruction line (stripped).

**Warning:** a later shot that does not **open** with its cut, in two
separate messages — no cut phrase at all, or a cut phrase buried after the
description (see §7.1). I2VA `<Picture 1>` not cited inside `[Shot 1]`. L2VA
`<Picture 1>` not cited in the final shot. FL2VA missing `Picture 1` in the
opening shot or `Picture 2` in the final one. FL2VA more than one shot when
the user did not ask. `overall_soundscape` outside 1–4 sentences,
`non_diegetic_music` outside 1–3. Over the 7000-character TrentNodes budget,
after shortening the score to its first sentence.

### 7.1 The cut opens the shot — and it is shared with ref mode

`guide_base` §4.2 does not merely list five phrases; it gives the form as a
template, and `guide_ref` §5.1 defers to it for the body rules:

```text
[Shot 2] At 00:03.500, the camera cuts to...
```

Shot label, timestamp, **cut phrase, then what the new shot shows**, all one
sentence. Every later shot in **all six** official worked examples across both
guides follows it, with no exceptions:

| Source | Opening |
|---|---|
| `guide_base` §4.2 template | `At 00:03.500, the camera cuts to...` |
| `guide_base` Case 1 | `At 00:05.000, the camera cuts to a close-up of steam...` |
| `guide_ref` §5.1 template | `At 00:09.000, the shot cuts to an extreme close-up...` |
| `guide_ref` §7 Shot 2 | `At 00:03.000, the shot cuts to a close-up of <Subject 4>...` |
| `guide_ref` §7 Shot 3 | `At 00:05.000, the shot cuts to a close-up of <Subject 3>...` |

Not one of them describes the new shot and mentions the cut afterwards. Both
system prompts now state the order explicitly and show the template, and
`_check_cut_phrasing` lives in the shared warnings section and runs in **both**
modes. It separates two mistakes, because they need different fixes:

- no cut phrase at all — which is also where a legitimately requested
  dissolve, fade or wipe lands, so it can only warn;
- a cut phrase present but not opening the shot — the likelier of the two,
  because it reads perfectly well and is still the wrong shape.

The literal words matter: the five phrases are lowercase prose that run into
the description. A screenplay slug (`CUT TO:`) is outside the guide's
vocabulary, and both system prompts say so.

`test_minimax_own_example_passes_clean` still reports zero warnings with the
check switched on in ref mode, which is the evidence that the rule is
MiniMax's and not ours.

### What must NOT be validated

- **`non_diegetic_music: N/A`.** Legal, and Case 3 uses it.
- **Mood words in `non_diegetic_music`.** The rule is real (§4.7) but
  MiniMax's own README I2VA output breaks it ("enhances the cozy, nostalgic,
  and joyful atmosphere"). Advisory at most.
- **A character count.** No MiniMax document states one; the README's I2VA
  body field alone is ~4,000 characters. The 7000 budget is a TrentNodes
  number and the warning says so.
- **Camera-move stacking.** §4.3's table is headed "Available Expression", not
  a closed grammar. (`check_camera_moves` still warns past two moves in one
  shot, exactly as in ref mode, and never retries.)
- **A word count.** The 350–500 range is a ref-mode number, `guide_ref` §5.2.
  Base mode records the count in `analysis_json` and asserts nothing.
- **`<scenetrans>` / `<cutoff>`.** Defined in §4.4 but **no worked example
  exists in any official source**, so there is nothing to check an
  implementation against. Case 1 shows the continuity phrasing *without* the
  tag ("the baker's final words carry over from the previous shot"). Still
  deferred.

---

## 8. Tests

`tests/test_h3_cowboy_base.py`, 40 of them, same plain-script harness as the
rest. The anchors are the first two:

1. **The four official cases pass with zero retry errors and zero warnings.**
2. **All four round-trip byte-identically** — instruction line included, which
   means the count-the-shots-then-render path agrees with the guide on all
   three of N, S.SS and the blank line.
3. Each embedded example matches the guide's text at every load-bearing point.
4. Same-line headers, in both directions: emitted with content on the line,
   and a ref-shaped reply pulled back onto one.
5. The FL2VA line has no brackets; the I2VA and L2VA lines have both.
6. Em dash in the two lines that have one, and none in I2VA's.
7. `render_instruction_line("base_L2VA", final_shot=3, duration_seconds=7.0)`
   yields `(from [Shot 3])` and `7.00-second`, and a three-shot body drives
   the same result end to end.
8. Base mode never emits `subject_definitions`, `summary` or
   `retention_analysis`, and a reply containing them is exactly one R1 error.
9. T2VA runs with no images, no video and no frames.

Run them with the loop in `docs/H3_COWBOY_HANDOFF.md` §5, which now includes
`h3_cowboy_base`.

---

## 9. Risks

- **No prior art.** Nothing in this repo has produced a base-mode prompt
  against a real generation. The format is locked to MiniMax's own examples,
  but "matches the spec" and "produces a better video" are different claims
  and only the first is established. Budget for the first real-provider run
  to contradict a guess — particularly around FL2VA's bare tags, which is the
  one place we deliberately reproduce something that looks like a mistake.
  `fl2va_normalize_picture_tags` exists to settle it by running it.
- **Base mode needs a different checkpoint.** Verify you actually have an
  `H3-Base-*` deployment before spending time on generation testing; the
  `h3_checkpoint_hint` output exists to make that obvious rather than tribal.
- **The same-line header trap (§2)** is still the way to break this. It looks
  correct in a diff and produces a malformed prompt. Two tests stand on it.
- **The default duration.** With no clip and no `duration_override`, 5.00
  seconds reaches the instruction line as fact. It warns, but a warning in a
  long list is easy to miss.

---

## 10. Reference

Source of truth, fetched with plain `curl` (the HF MCP tools return Cloudflare
502s):

```
https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
```

Section numbers used above: §1 task overview, §2.1 instruction lines, §2.2 the
three fields, §3.1–3.3 per-mode body shape, §4.1 timeline and style, §4.2 shots
and cuts, §4.3 camera, §4.4 speakers and dialogue, §4.5 on-screen text, §4.6
soundscape, §4.7 non-diegetic music, §5 the four cases.
