# H3 Reference Wiring — pass the assets through the promptor

**Status:** **BUILT 2026-08-14.** Sections 4 to 7 are done and tested; §10 is
the build log and lists the four places the build differs from this plan.
Nothing has been checked against a real generation yet — see §8.

Sections 2 and 3 are **verified** against `comfy_extras/nodes_minimax_h3.py` at
the lines cited, and they stand whatever gets built. Sections 4 onward are the
plan.

**What changed, and why this document was rewritten.** I first built a separate
node, `H3 Reference Wiring`. That was the wrong shape. You already plug the
images, the video and the audio into the Ultimate H3 Cowboy Promptor, so it
should simply hand them back out — no second node, no second wire-up. It also
turns out to be the *easier* place to fix the timing (§4.2), because the
promptor controls the order in which things happen and a separate node cannot.

**The standalone node is deleted.** Trent settled that. Its arithmetic survives
in `utils/h3_cowboy/wiring.py`, tested, and gets reused as is.

Read `docs/H3_COWBOY_HANDOFF.md` first for the package. This document is only
about getting assets into the sampler in the order the prompt already claims
they are in.

---

## 1. Why this exists

The promptor writes a prompt that names its assets: `<Picture 1>`,
`<Picture 2>`, `<Video 1>`, `<Audio 1>`. `MiniMaxH3ReferenceToVideo` assigns
those same tags **independently**, from the order its own sockets are filled.
Nothing checks that the two agree.

When they disagree, nothing errors. H3 generates a video that follows the wrong
image, and the prompt reads perfectly. That is the worst failure mode in this
pipeline, because there is no artefact to inspect afterwards — the prompt says
`<Picture 2>` and the model was handed a different picture under that name.

Today you also wire every asset **twice**: once into the promptor so it can see
them, once into the sampler so H3 can use them. The second wire-up is where the
two drift apart. Pass-through outputs delete it.

---

## 2. What `MiniMaxH3ReferenceToVideo` actually does

`comfy_extras/nodes_minimax_h3.py:154`. Its own docstring (`:155-161`) states
the contract:

> References enter the presentation in fixed order: images, then videos (each
> soundtrack's `<Audio j>` label right before its `<Video k>`), then standalone
> audio. Ordinals are 1-based per type, so the prompt refers to them as
> `<Picture i>` / `<Video k>` / `<Audio j>`.

Four **Autogrow** input groups (`:180-195`):

| Group | Socket prefix | Max | Becomes |
|---|---|---|---|
| `ref_images` | `ref_image_` | 9 | `<Picture i>` |
| `ref_videos` | `ref_video_` | 3 | `<Video k>` |
| `ref_video_audios` | `ref_video_audio_` | 3 | `<Audio j>`, the soundtrack of the **same-numbered** `ref_video_` |
| `ref_audios` | `ref_audio_` | 3 | `<Audio j>`, standalone |

Plus `clip`, `vae`, `audio_vae`, `prompt`, `width`, `height`, `length`, and
`ref_image_size` (`match` / `max`).

**Your own proven wiring** is already recorded in
`projects/spong_h3/build_prompts.py:11-17`, from the SPONG remaster:

```
ref_image_1..N     = rendered character refs, in manifest "pictures" order
ref_video_1        = the clip/chunk itself  -> <Video 1>
ref_video_audio_1  = the clip's own audio   -> <Audio 1>
```

giving tags in the order `<Picture 1..N>`, `<Audio 1>`, `<Video 1>`. That is
the layout to reproduce, and it is the one configuration known to have made
real videos.

---

## 3. Six traps, all verified in the source

### 3.1 Ordinals come from arrival order, never from the socket number

`:218-220`:

```python
for img in (ref_images or {}).values():
    if img is None:
        continue
```

A `None` in the middle is **skipped, and the ordinals close up behind it**.
Wire images into sockets 1 and 3 and the second image becomes `<Picture 2>`,
not `<Picture 3>`.

This collides head-on with the promptor's rule, which is deliberate and
documented: **`subject_N_image` IS `<Picture N>`, and a gap leaves a gap in the
numbering.** The promptor warns about the gap and keeps writing `<Picture 3>`.
The sampler silently renumbers. Reconciling those two is the main job of this
node — see §5.1.

`RefFolderCowboy` already dodges this by filling densely in natural-sort order
(`nodes/ref_folder_cowboy.py:28-34`), which is why it has never bitten there.

### 3.2 An Autogrow group cannot be fed by one wire

`comfy_api/latest/_io.py:1117-1127`: `TemplatePrefix` materialises the group as
individual sockets named `f"{prefix}{i}" for i in range(max)`. Each is an
ordinary `IMAGE` / `AUDIO` input. There is no list type that fills the group in
one connection.

So this node must emit **one output socket per reference slot**, and they are
wired individually — the `RefFolderCowboy` shape (`RETURN_TYPES = ["IMAGE"] * NUM_SLOTS + ...`).

Note the socket suffix is **0-based** in the schema (`ref_image_0` … `ref_image_8`).
Do not build anything on the suffix; build on arrival order and no gaps.

### 3.3 A video's soundtrack takes an `<Audio j>` number *before* any standalone audio

`:255-264`, in the video loop, the soundtrack is appended to `ref_items` **before**
its video; standalone audios are appended afterwards at `:269-274`. So with one
video soundtrack and one standalone reference, the soundtrack is `<Audio 1>` and
the standalone is `<Audio 2>`.

The promptor's `audio_role` already carries the distinction and decides the
destination socket:

| `audio_role` | Meaning | Socket |
|---|---|---|
| `reuse` | the clip's own track is reused | `ref_video_audio_` paired to the video |
| `reference` | only timbre/beat/style is followed | `ref_audio_` standalone |

The pairing is by numeric suffix (`:239`):
`soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])`.
A soundtrack in a slot whose video is empty is silently dropped.

### 3.4 `length` snaps, and the snap changes the duration the prompt claims

`:33-36`:

```python
def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n
```

`length` is a **frame count at 24 fps**, snapped **up** to the 17k+5 grid. The
snap adds up to 16 frames, and at short durations that is a large fraction.
Measured, not estimated:

| You ask for | `length` | You actually get | Drift |
|---|---|---|---|
| 2.00 s | 56 | 2.333 s | **+17%** |
| 5.00 s | 124 | 5.167 s | +3.3% |
| 8.00 s | 192 | 8.000 s | none |
| 10.00 s | 243 | 10.125 s | +1.3% |

That matters because the promptor writes the duration into the prompt as fact:
FL2VA and L2VA put it in the instruction line as `S.SS`, and every mode places
later shots at times inside it. Prompting `2.00-second` and generating 2.33
seconds is a mismatch the model cannot see — and at that length it is a third
of a second of unaccounted video.

Note 8.00 s lands exactly. `guide_base` Case 3 is an eight-second clip, which
makes 192 a good default to reach for.

**So the snap has to happen before the prompt is written, not after.** That
decides where this node sits in the graph — see §5.2.

### 3.5 24 fps is not your clip's fps

`:29` — `FPS = 24`, hardcoded. The reference video tooltip (`:186`) says
"Reference video frames at 24 fps (2-15s)".

Your SPONG material is 23.976 (`build_prompts.py:27` uses `24000/1001`), and
the promptor happily reports whatever the source clip's rate is. Frames handed
to `ref_video_` are *assumed* to be 24 fps; nothing resamples them. A 30 fps
clip passed straight through is a reference video played 25% slow.

### 3.6 Reference video: `IMAGE` frames, minimum 5, trimmed **down** to 17k+5

The promptor takes a `VIDEO`; the sampler takes `IMAGE` frames. Conversion is
on us. Then, at `:246-253`:

- frames longer than the generation's `frame_count` are truncated;
- fewer than 5 frames raises `ValueError`;
- the count is trimmed **downwards** (`n -= 1`) until `n % 17 == 5`.

Qwen only sees the reference video at 2 fps (`:261`, every 12th frame), so the
useful signal is coarse — worth knowing before spending VRAM on a long one.

---


## 4. What to build

Add pass-through outputs to `nodes/ultimate_h3_cowboy_promptor.py`. No new
node. The assets are already arguments to `generate()`; they just never come
back out.

### 4.1 The outputs

Today the promptor returns five:

```python
RETURN_NAMES = ("h3_prompt", "duration_seconds", "fps", "analysis_json",
                "h3_checkpoint_hint")
```

**Append** the new ones. Never insert. A saved workflow stores links by output
**index**, so inserting an output in the middle silently re-points every wire
after it in every graph the user has.

```python
RETURN_NAMES = (
    "h3_prompt", "duration_seconds", "fps", "analysis_json",
    "h3_checkpoint_hint",
    # --- pass-through, appended 2026-08-xx; append only, never insert ---
    "ref_image_1", "ref_image_2", "ref_image_3",
    "ref_image_4", "ref_image_5", "ref_image_6",
    "ref_video", "ref_video_audio", "ref_audio",
    "width", "height", "length", "label_map",
)
```

| Output | Type | Is |
|---|---|---|
| `ref_image_1..6` | IMAGE | exactly what you plugged into `subject_N_image`, untouched |
| `ref_video` | IMAGE | the video as frames — the sampler's `ref_video_` socket takes IMAGE, not VIDEO |
| `ref_video_audio` | AUDIO | the wired audio |
| `ref_audio` | AUDIO | the same object again |
| `width`, `height` | INT | `canvas_for()` on the video, else on the first picture |
| `length` | INT | the frame count on H3's grid (§4.2) |
| `label_map` | STRING | what each tag will refer to, for checking against the prompt |

Two deliberate non-decisions, both because the node should not guess:

- **The audio comes out of both audio sockets**, carrying the same object.
  Connect `ref_video_audio` when the clip's own track is reused, `ref_audio`
  when only its timbre or beat is referenced. Leave the other unconnected.
- **A gap passes through untouched.** `ref_image_3` always carries what you
  put in `subject_3_image`. The promptor already warns about the gap; §3.1 is
  why that warning matters. Do not compact — that is the bug this is for.

A `None` on an IMAGE output is fine and is already how `RefFolderCowboy`
works: the sampler's optional sockets treat it as not connected
(`nodes/ref_folder_cowboy.py:28-34`).

### 4.2 The timing fix — easier here than anywhere else

§3.4: H3 rounds `length` **up** to a 17k+5 frame grid at 24 fps, so a
2.00-second request really renders 2.33 seconds. The promptor writes the
duration into the prompt as fact — base mode puts it in the instruction line as
`S.SS`, every mode places shot times inside it — so the prompt describes a
shorter video than the one H3 makes.

Inside the promptor this is easy, and that is the real argument for the user's
design. The promptor computes `duration` and *then* writes the prompt. Snap the
duration first, and every downstream number is true:

```python
# right after duration/duration_override is resolved, BEFORE the prompt
length, duration = snap_length(duration)
```

A separate node cannot do this. It would have to feed the promptor and be fed
by it, and ComfyUI graphs are acyclic. That was §5.2 of the old plan and it is
now moot.

Add a `snap_duration_to_h3_grid` BOOLEAN so the change is visible and
reversible. See §5.1 for the default.

---

## 5. Three decisions — all settled

### 5.1 Default for `snap_duration_to_h3_grid` — **SETTLED: on**

Trent chose **default on**. The prompt states the duration as fact, so a prompt
that misstates it is a defect, not a preference.

Consequences to handle when building:

- `duration_seconds` (existing output 2) now carries the **snapped** value, not
  the requested one. That is a value change on an existing output; it is
  deliberate, and `analysis_json` must record both numbers so a past run can
  still be explained.
- The prompt uses the snapped duration everywhere: base mode's instruction line
  `S.SS`, and the ceiling every mode's shot times sit under.
- The toggle still exists, so a run can be reproduced with the old behaviour.
- A run with no clip and no override already defaults to 5.00 s, which snaps to
  5.167 s.

### 5.2 The standalone node — **SETTLED: deleted**

`nodes/h3_reference_wiring.py` is gone, along with its registration and its
node-level tests. `utils/h3_cowboy/wiring.py` and the arithmetic tests in
`tests/test_h3_wiring.py` stay — that is the part worth keeping.

Four behaviours went with it and have to land again on the promptor. The two
warning texts are worth reusing verbatim, because each names the consequence
rather than the symptom:

1. **VIDEO to frames.** The sampler's `ref_video_` socket takes IMAGE. The
   promptor's own `_resolve_frames` already produces the frame tensor, so the
   pass-through just returns it.
2. **The 24 fps warning**, which the promptor does not have today:

   > the video is {fps} fps. H3 reads reference frames as 24 fps and nothing
   > resamples them, so the reference plays at the wrong speed. Re-time the
   > clip to 24 fps first.

3. **The gap consequence.** The promptor already warns about a gap in the
   slots (`subjects.py::bind_images`), but only about its own numbering. Add
   what it costs downstream:

   > The sampler numbers the images it receives, so every tag after the gap
   > shifts down and will not match the prompt. Fill the slots in order.

4. **Canvas selection**: the video's shape wins over the first picture's,
   because it carries the shot's real framing. `canvas_for()` does the rest.

### 5.3 Base mode returns the same shape — **SETTLED: one shared helper**

`generate()` (ref) and `_generate_base()` return separately. ComfyUI reads
outputs by position, so both must return the **same count in the same order**
or the node breaks the moment `h3_mode` changes. Two hand-maintained return
statements drift the first time someone adds an output to one and not the
other.

So exactly one place knows the shape:

```python
def _outputs(
    self, prompt: str, duration: float, fps: float, analysis: dict,
    mode: str, *, images: Optional[dict] = None, frames=None,
    audio=None, width: int = 0, height: int = 0, length: int = 0,
    label_map: str = "",
) -> Tuple:
    """The node's whole return tuple, built once for both modes."""
    images = images or {}
    return (
        prompt, round(duration, 3), int(round(fps)),
        json.dumps(analysis, indent=2),
        spec.CHECKPOINT_FOR_MODE.get(mode, ""),
        *(images.get(f"subject_{slot}_image")
          for slot in range(1, NUM_SUBJECT_SLOTS + 1)),
        frames, audio, audio,
        width, height, length, label_map,
    )
```

Both paths end in `return self._outputs(...)` and nothing else. Test 2 in §7
runs a ref job and a base job and compares the lengths, so a future divergence
fails immediately rather than on the user's canvas.

**Base mode's pass-throughs are mostly `None`, and that is correct.** The base
format has no `<Video 1>` and no `<Audio 1>` at all, and T2VA has no pictures
either. Empty is the honest answer, not a gap to fill. Say so in the output
tooltips, because it looks like a fault and is not.

## 6. What already exists — do not rewrite it

`utils/h3_cowboy/wiring.py`, with 9 tests in `tests/test_h3_wiring.py`. Pure
functions, no torch, no ComfyUI. Every one is a deliberate copy of sampler code
with the source lines named, so an upstream change fails a test here instead of
mis-sizing a real generation.

| Function | Does |
|---|---|
| `snap_length(seconds)` | `(frames on the 17k+5 grid, the duration that really produces)` |
| `trim_reference_frames(count)` | what the sampler keeps from a reference video; `0` below 5 frames |
| `canvas_for(w, h)` | the sampler's `adapt_canvas` — 768 short edge, area cap, /32 |
| `build_label_map(pictures, has_video, has_audio)` | the tag → asset list, in the sampler's presentation order |

Three of its tests are anchors and should keep passing untouched:
`snap_length` and `canvas_for` are each checked against a re-implementation of
the original across their whole range, and the documented drift numbers
(2.00 → 2.33, 8.00 → exact) are pinned.

The frame conversion, the gap warning and the 24 fps warning were written in
the standalone node and went with it. §5.2 carries the two warning texts
verbatim; the rest is a few lines the promptor mostly has already.

---

## 7. Tests

Extend `tests/test_h3_cowboy_node.py`, which already runs the promptor end to
end against a fake backend.

1. **The five original outputs keep their index.** Assert
   `RETURN_NAMES[:5]` explicitly. This is the test that protects saved graphs.
2. **`len(RETURN_TYPES) == len(RETURN_NAMES)`**, and both paths return that
   many values — run ref mode and a base mode and compare lengths.
3. **What goes in comes out**: `subject_3_image` is the same object as
   `ref_image_3`, by identity.
4. **A gap passes through untouched**: slots 1 and 3 wired leaves
   `ref_image_2` as `None` and `ref_image_3` intact.
5. **The audio reaches both audio outputs.**
6. **A wired VIDEO comes out as frames**, not as a VIDEO object.
7. **Snapping makes the prompt agree with `length`**: run `base_L2VA` with
   `duration_override=2.0`, and assert the instruction line says
   `2.33-second` and `length` is 56.
8. **Snapping off leaves the old behaviour**, so the toggle is real.
9. **T2VA with nothing wired** returns the full tuple with `None` in every
   pass-through — the shape holds when there is nothing to pass.

---

## 8. Risks

- **Output index stability is the one that hurts.** Appending is safe;
  inserting silently re-points wires in saved graphs. Test 1 exists for this.
- **Two return paths.** §5.3. A shared helper, or this breaks quietly.
- **`comfy_extras` is not an API.** Everything in §2 and §3 comes from a file
  upstream can change without notice. The mitigations are the copied functions,
  the arithmetic tests, and `file:line` citations throughout. Re-read
  `nodes_minimax_h3.py` after any ComfyUI update that touches MiniMax.
- **Nothing here has been checked against a real generation.** The label map is
  verified against the *code*; the SPONG layout is verified against a
  configuration that produced real videos. Those are different strengths of
  evidence and should not be blurred.
- **Autogrow is newer than most of this repo.** Per-slot outputs stay correct
  because they are ordinary sockets, but nothing should depend on socket
  suffixes. Build on arrival order.

---

## 9. Reference

| Fact | Source |
|---|---|
| Node class, ordering docstring | `comfy_extras/nodes_minimax_h3.py:154-161` |
| The four Autogrow groups and their maxima | same, `:180-195` |
| Image loop, gap collapse | same, `:218-232` |
| Video loop, soundtrack pairing, 5-frame minimum, 17k+5 trim, 2 fps sampling | same, `:234-267` |
| Standalone audio loop | same, `:269-274` |
| `align_frame_count`, `FPS = 24`, `adapt_canvas`, `REF_IMAGE_SHORT_EDGE` | same, `:33-36`, `:29`, `:49-60`, `:28` |
| Autogrow socket naming (0-based) | `comfy_api/latest/_io.py:1117-1127` |
| Proven SPONG wiring | `projects/spong_h3/build_prompts.py:11-17` |
| `None` on an IMAGE output is fine | `custom_nodes/TrentNodes/nodes/ref_folder_cowboy.py:28-75` |
| The promptor's current five outputs | `custom_nodes/TrentNodes/nodes/ultimate_h3_cowboy_promptor.py` |

---

## 10. Build log — 2026-08-14

Everything in §4 to §7 is in `nodes/ultimate_h3_cowboy_promptor.py` and its two
test files. The node now returns **18 outputs**: the original five, then
`ref_image_1..6`, `ref_video`, `ref_video_audio`, `ref_audio`, `width`,
`height`, `length`, `label_map`. `snap_duration_to_h3_grid` is the last widget
in `optional`, default **on**.

| Piece | Where |
|---|---|
| The one return shape both modes use | `_outputs()` |
| The timing fix | `_snap()`, called before the prompt is written in both modes |
| Canvas, video wins over the first picture | `_canvas()` |
| The gap consequence | `_warn_gap_consequence()` |
| The 24 fps warning and the 5-frame floor | `_warn_reference_video()` |

**Four differences from the plan, all deliberate:**

1. **`length` is always on the grid, even with snapping off.** The toggle
   decides what the *prompt says*, not what the sampler is handed — an
   off-grid `length` is not a legal input, so it is never emitted. With the
   toggle off the mismatch is a warning instead, naming both numbers.
2. **The 24 fps warning has a 0.5 fps tolerance** (`FPS_WARN_TOLERANCE`).
   23.976 material drifts by one frame in a thousand, which is not worth
   saying; 25 and 30 fps are, and still warn.
3. **A fifth behaviour came back with the four in §5.2**: a clip under 5
   frames warns, because `trim_reference_frames()` says the sampler raises
   below that. Cheaper than a `ValueError` mid-run.
4. **The base-mode tests live in `tests/test_h3_cowboy_base.py`**, not in
   `test_h3_cowboy_node.py` as §7 assumed. That file already has a correct
   base reply fixture; the node file's canned reply is ref-shaped and fails
   base validation, so it cannot assert an instruction line.

**What the snap changed in the existing tests.** `duration_seconds` is now the
snapped value, so `test_node_end_to_end` expects 2.333 rather than 2.000 and
the base tests compare against `snap_length()`. That is the §5.1 consequence
landing, not a regression. `analysis_json` carries
`requested_duration_seconds`, `snapped_duration_seconds`,
`snap_duration_to_h3_grid` and `h3_length_frames`, so any past run can still be
explained.

**Tests added** — all nine of §7 plus three more:

```
tests/test_h3_cowboy_node.py   the first five outputs never move; both modes
                               return the same shape; identity pass-through;
                               a gap passes through untouched; audio on both
                               sockets; VIDEO out as frames; canvas + length;
                               the off-speed clip warning; the label map order
tests/test_h3_cowboy_base.py   snapping makes the instruction line agree with
                               length; snapping off keeps the old number and
                               warns; T2VA returns the whole tuple with None
                               in every pass-through; a base anchor picture
                               reaches its own slot
```

Run them with:

```
venv/bin/python custom_nodes/TrentNodes/tests/test_h3_cowboy_node.py
venv/bin/python custom_nodes/TrentNodes/tests/test_h3_cowboy_base.py
venv/bin/python custom_nodes/TrentNodes/tests/test_h3_wiring.py
```

**Still open.** The wiring has never driven a real generation. The first run
should read `label_map` against the prompt's own tags before spending VRAM,
and confirm on the canvas that an existing saved workflow still finds its five
original wires.
