# Ultimate H3 Cowboy Promptor — Phase 6: the node face

**Status:** **built, tested, pushed.** `14a6391` on `main`, 2026-08-15.
341 Python tests pass across the H3 and cut-detect suites, plus 26 JS checks.
Everything here is verified against the code and the frontend's own source
**except the two things §4 lists** — the canvas and a real H3 generation.

Read `docs/H3_COWBOY_HANDOFF.md` first for the package. This document covers
what changed on the node face: subject rows, music video, and the visibility
extension. `docs/H3_REFERENCE_WIRING_HANDOFF.md` §10 covers the pass-through
outputs, which landed the same day.

---

## 1. The model to hold in your head

One chain, and everything on the face serves it:

```
row N  →  <Subject N>  →  subject_N_image  →  <Picture N>  →  ref_image_N
```

Left to right: what you type, what the prompt calls it, where the picture
goes in, what the sampler calls it, what comes back out. **A gap anywhere
stays a gap**, and warns. Closing it would silently point the prompt at a
different image — the failure the whole Phase 5 pass-through exists to stop.

Three things follow from that and are worth saying out loud:

- **An empty row is not a subject**, whatever `subject_rows` says. A row
  counts when its name or description holds text.
- **A row needs no image.** That is how you ask for a style or an action,
  which no picture can supply.
- **`subject_rows` is a display setting only.** Python reads all six rows and
  ignores the count, then warns if a filled row sits above it. A value that
  quietly stops counting when a widget hides is the worst kind of surprise.

---

## 2. Where everything lives

| Piece | Where |
|---|---|
| The row dataclass and the two builders | `utils/h3_cowboy/subjects.py`: `SubjectRow`, `subjects_from_rows`, `merge_text_subjects` |
| Plain-word kinds | `utils/h3_cowboy/spec.py`: `KIND_CHOICES`, `KIND_ALIASES`, `canonical_kind()` |
| Reading the rows off the node | `nodes/ultimate_h3_cowboy_promptor.py::_read_rows` |
| Music: what the run actually does | same file, `_resolve_music()` |
| Music: what the model is told | `utils/h3_cowboy/prompts_ref.py`: the `MUSIC_VIDEO_*` constants and `_music_lines()` |
| Music: what the reply must satisfy | `utils/h3_cowboy/assembler.py::enforce_music_video` (`R6 MUSIC`) |
| Which widgets are visible | `js/h3_cowboy.js` |
| JS tests, against a mocked frontend | `tests/h3_cowboy_js/run.mjs` |

`character` → `person` and `environment` → `scene` happen in `canonical_kind()`,
so the guide's own word is what reaches every kind card, rule and validator.
The friendly spellings work in the typed DSL too.

---

## 3. Two rules the next change must not break

### 3.1 Nothing is ever removed from `node.widgets`

A workflow serialises widget values **positionally**. A widget missing from
the array at save time hands its neighbours the wrong values on reload — the
bug Multi-Load Cowboy was fixed for twice (`8a4c8e2`, `7d53f56`).

So `js/h3_cowboy.js` only sets flags. It writes **both** spellings, because
the two renderers read different ones:

```js
widget.hidden = true;          // canvas renderer: isWidgetVisible()
widget.options.hidden = true;  // Vue renderer: isWidgetVisible(options, ...)
```

Frontend 1.48.7 does the rest itself: `isWidgetVisible(w)` is
`!(collapsed || w.hidden || w.advanced && !showAdvanced)`, `getLayoutWidgets()`
filters `!w.hidden` so the height follows, and `BaseDOMWidgetImpl.isVisible()`
checks the same flag, which is why the multiline boxes hide properly.

`"advanced": True` in an `INPUT_TYPES` options dict is read straight from
Python and puts a widget behind the node's own **Show advanced** toggle. Eight
widgets use it today. No JS is involved.

**Autogrow cannot help here.** `comfy_api/latest/_io.py:1093` forces
`force_input = True` on any `WidgetInput` in a template, so autogrown rows of
text and combo widgets would become sockets you must wire. That is why the
rows are six fixed widget triples and not a dynamic group.

### 3.2 Inputs and outputs are append-only

New optional inputs go on the end; new outputs go on the end. Two tests exist
solely for this: `test_widgets_are_only_ever_appended` pins the first 22
optional inputs, and `test_the_first_five_outputs_never_move` pins the outputs
every saved graph is wired to.

---

## 4. What is NOT verified — start here

Two different strengths of evidence, and they should not be blurred.

**The canvas.** `js/h3_cowboy.js` is checked against a *mock* of the frontend
(26 assertions) and against the real frontend's *source*. It has never been
loaded in a browser. First session on the canvas should check, in order:

1. Drop a fresh node. Rows 1 and 2 show; rows 3-6 do not.
2. Type in row 2's description. Row 3 appears.
3. Wire an image into `subject_4_image`. Rows 1-4 are all there.
4. Turn on `music_video`. `music_source`, `lyrics` and `music_description`
   appear.
5. Switch `h3_mode` to `base_FL2VA`. The rows and the music toggle go; the
   two base settings arrive.
6. Wire `frames`, then `audio`. `fps`, `video_role` and `audio_role` appear as
   their sockets fill.
7. Save, reload, and confirm every widget still holds its own value. This is
   the one that matters — see §3.1.
8. Open an OLD saved graph. Its typed `subjects` text must still be there and
   the run must behave as it did.

If the extension fails to load, the node is ugly but complete: every widget
shows, and every value still works. Nothing is lost.

**A real generation.** No prompt from this node has been fed to H3. The label
map, the canvas arithmetic and the frame grid are verified against
`comfy_extras/nodes_minimax_h3.py`; the SPONG layout is verified against a
configuration that made real videos. Those are not the same thing. Read
`label_map` against the prompt's own tags before spending VRAM.

---

## 5. Decisions settled — do not re-litigate

- **Rows are read regardless of `subject_rows`.** §1. The alternative hides a
  subject nobody can see or edit.
- **A gap in the rows stays a gap.** §1, and `docs/H3_REFERENCE_WIRING_HANDOFF.md`
  §3.1 for what the sampler does with a compacted one.
- **The typed `subjects` field stays**, marked advanced. Every workflow saved
  before rows existed uses only it, and it says things a row cannot: a seventh
  subject, or one citing two pictures at once. Typed lines are numbered
  **after** the filled rows.
- **`character` and `environment` are labels, not new kinds.** The guide's
  eleven kinds are unchanged; `KIND_ALIASES` is a spelling layer.
- **A reused song goes through `audio_role`.** It reaches the task type by the
  same path a reused clip track does, rather than a second mechanism beside it.
- **`music_source: auto` assumes** the file wired here is the file the H3 graph
  is fed. That is the common case, not a guarantee, which is why
  `reuse_audio_1` and `generate_score` exist to state it outright.
- **Base mode has no music video.** `guide_base` has no reference labels at
  all, so there is no `<Audio 1>` to reuse. It warns and ignores.

---

## 6. Open work, in the order it is worth doing

1. **The canvas checklist in §4.** Everything else is downstream of it.
2. **A real H3 generation**, and the SPONG A/B this node was built for — the
   hand-written REF2VA prompt against this node's output on the same clip. See
   the `spong-h3-remaster-project` memory and `projects/spong_h3/`.
3. ~~**Phase 4 — extraction and tooling**~~ Done 2026-08-15: the shared
   helpers live in `utils/h3_prompt/core.py`, the golden hashes never moved,
   and `tools/h3_cowboy_dev_run.py` exists (its `--reply @file` mode runs the
   pipeline on canned text with no API spend). See
   `docs/H3_COWBOY_HANDOFF.md` §7.
4. **A node card** — screenshot for the README, like the other nodes have.
5. **Possible, not decided:** hide the unused `subject_N_image` sockets above
   the visible rows. Sockets are riskier than widgets — a hidden socket with a
   link is a lost link — so this needs the link check written first, and it
   only pays off if the six sockets actually feel like clutter on the canvas.

---

## 7. Dev loop

```bash
cd /home/trent/ComfyUI
for t in cut_detect h3_assembler h3_format h3_keyframes h3_node h3_audio \
         h3_video h3_cowboy_subjects h3_cowboy_ref h3_cowboy_node \
         h3_cowboy_base h3_wiring; do
  venv/bin/python custom_nodes/TrentNodes/tests/test_$t.py
done
node custom_nodes/TrentNodes/tests/h3_cowboy_js/run.mjs
```

The anchor tests are the ones that matter, and all of them run MiniMax's own
text through the validator demanding zero errors:
`test_minimax_own_example_passes_clean`,
`test_the_official_editing_example_passes_clean`,
`test_the_four_official_cases_pass_clean`, and
`test_the_four_official_cases_round_trip_byte_for_byte`. If one starts
failing, the validator is wrong, not the example.

Do not start the ComfyUI server yourself — Trent runs it.

---

## 8. Traps met while building this

- **`_NA_ONLY_RE` is whole-string**, unlike the older node's prefix `_NA_RE`.
  A score reading "N/A. Then the pad enters." is not treated as silent. That
  is deliberate and stricter.
- **The `subject_rows` Python default is `NUM_SUBJECT_SLOTS`, not the widget's
  2.** Omitted means "no opinion", so a direct call or an API payload that
  never mentions rows does not warn about hiding one.
- **`FPS_WARN_TOLERANCE = 0.5`** exists so 23.976 material stays quiet. It
  drifts by one frame in a thousand against H3's hardcoded 24; 25 and 30 fps
  do not, and still warn.
- **Snapping changed an existing output's value.** `duration_seconds` now
  carries the snapped number. `analysis_json` records
  `requested_duration_seconds` and `snapped_duration_seconds` so any past run
  can still be explained.
- **`comfy_extras` is not an API.** Everything the wiring copies comes from a
  file upstream can change without notice. Re-read `nodes_minimax_h3.py` after
  any ComfyUI update that touches MiniMax; `tests/test_h3_wiring.py` is what
  catches the drift.
