---
name: h3-prompting
description: Teach and apply the official MiniMax H3 video prompt formats (Ref2VA six-section and Base three-field). Use when writing, reviewing, or debugging an H3 prompt, or when the user asks how to prompt MiniMax H3.
---

# MiniMax H3 prompting

Everything here traces to MiniMax's own docs on huggingface.co/MiniMaxAI/MiniMax-H3:
`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` (guide_ref), `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` (guide_base), and the repo README. The battle-tested local implementation lives in TrentNodes: `utils/h3_prompt/prompts.py`, `utils/h3_cowboy/{spec,prompts_ref,prompts_base}.py` — read those for worked examples and validators before inventing anything new.

## Step 0: pick the mode (they are different checkpoints, not flags)

| Mode | Checkpoint | Skeleton |
|---|---|---|
| Ref2VA (full-reference: identity swap, editing, continuation, structure reference) | MiniMax-H3-Base-Ref2VA | SIX sections |
| Base T2VA / I2VA / FL2VA / L2VA (text or frame-anchored generation) | MiniMax-H3-Base-FL2VA | THREE fields |

Never mix the skeletons. Base mode has no `<Subject N>`, no `<Video N>`, no `<Audio N>`, no task-type prefix, no retention_analysis.

## Ref2VA format (six sections)

Exactly six lowercase headers, in this order, each with a trailing colon **on its own line**, content on the next line. Output starts at `subject_definitions:` and ends with the `non_diegetic_music` section. Nothing before, nothing after — no markdown, no title, no duration/fps/aspect-ratio claims (those are node/UI parameters), and no trailing "No ..." exclusion list (off-spec practitioner habit; H3 has no negative-prompt field — express constraints as positive assertions instead: not "no wardrobe changes" but "the charcoal jacket remains exactly as defined through every shot").

```
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

### Reference labels
- `<Subject N>` = any reusable **visible** content: person, animal, object, scene, wardrobe, interface, effect, style, action, expression, pose. Not just people.
- `<Picture N>` = reference image, `<Video N>` = reference video, `<Audio N>` = audio signal.
- One line per label in subject_definitions and retention_analysis. Never renumber; never invent a label for an asset that was not given; no new labels in summary.
- An image/video that only shows where a subject came from is cited INSIDE that subject's line and gets **no line of its own**. A `<Picture N>` earns a standalone line only when it is a concrete frame of the target video (first/key/last frame or composition anchor).
- `<Video N>` is for whole-video relationships only: editing it, continuing it, or referencing its camera movement/cuts/rhythm. A person or action reused from a video is still a `<Subject N>`.
- Multi-source subject: combine in one line and say what each asset provides — "<Subject 1> is the woman whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>".

### summary
One short paragraph. MUST begin with a square-bracketed task-type prefix. Legal values, combinable with " + ", never repeated:
`keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference`
Presence of an asset does not imply a type — a video supplying only camera/cuts/rhythm is still `reference generation`; use `video editing`/`video continuation` only when the video is directly edited or continued. An editing summary opens (after the prefix) with the fixed sentence "The target video is an edited version of <Video 1>."

### retention_analysis
One line per defined label, shaped `<Tag N> (scope): marker - explanation`. Scope says where it applies: "(appears in [Shot 1], [Shot 3])", "([Shot 1] first frame)", "(cut and pacing structure)". Markers are fixed values and the two sets never mix:
- visible content: `fully_preserved, partially_preserved, attribute_transfer, weak_reference`
- audio: `fully_copy, partially_copy, reference, weak_reference`
New actions/backgrounds in the target are NOT losses of fidelity (an edited source whose mouth is newly animated is still `fully_preserved`). Never write an `(Sx)` speaker ID here.

### detailed_description
- Open with 1-2 sentences of overall visual style, then the shot timeline. Normally 350-500 English words for generation tasks. A tighter 120-300-word profile also works well in practice — compress environment first, never the central action, camera work, or ending.
- `[Shot 1]` has no timestamp. Later shots: `[Shot N] At MM:SS.mmm,` with strictly increasing times inside the clip duration.
- Every later shot OPENS with its cut, running straight into the new content, using exactly one of five phrases: "the camera cuts to", "the shot cuts to", "the shot transitions to", "the shot changes to", "the shot switches to". Never "CUT TO:", never cut-mentioned-afterwards. Never describe a dissolve/fade/wipe unless the user asked for one — H3 renders what you describe.
- Camera is prose from the official vocabulary only: static, push in, pull out, zoom in/out, pan left/right, tilt up/down, truck left/right, pedestal up/down, arc, roll clockwise/counterclockwise, tracking shot, POV, shake slightly/strongly. Qualify with the fixed phrases "with small amplitude" / "with large amplitude" and "at slow speed" / "at fast speed". ONE dominant move per shot.
- Observable behavior only. No emotions or intentions. Weak: "she is furious". Strong: "her arm snaps forward and she shoves the crate off the table".
- At a subject's first clear appearance describe its referenced features and (for spatial kinds) its position in frame; later shots reuse the label without redefining. A subject need not appear in every shot.
- Dialogue: `<Subject N> (Sx) says, <d>[Language] Exact words.</d>` — visual label stays with the speaker ID; `[Language]` is the language of the words themselves; preserve supplied words verbatim, never translate. Voiceover uses the exact phrase "says in an off-screen voiceover" plus a statement that lips stay closed. On-screen text goes in double quotes, verbatim.
- Identity insurance for swaps: name the subject + tag in every shot, state frame position, and when the subject is distant/back-turned/occluded say so and state that identity and wardrobe remain locked to the reference picture.

### Audio sections
- `overall_soundscape`: 1-4 sentences of diegetic sound tied to visible events. N/A only for requested silence.
- `non_diegetic_music`: 1-3 sentences on instrumentation, tempo, and score development, or exactly `N/A`. Music the characters can hear is diegetic → body, not here. Never repeat dialogue or lyrics in either audio section.
- Music-video mode inverts the balance: non_diegetic_music leads (genre, instruments, tempo/BPM, arrangement vs the shot list) and never `N/A`; overall_soundscape stays thin; cuts follow the beat. Lyrics go inside `<d>` in the shot where sung, original language, split across cuts if needed. A reused track is `<Audio 1>` with a `fully_copy` retention line; a voice that exists only inside the track belongs to `<Audio 1>` and gets no `(Sx)` ID.

## Base format (three fields)

```
integrated_multimodal_description: [Shot 1] Live-action, cinematic, ...
overall_soundscape: ...
non_diegetic_music: ...
```

Differences from ref mode that look like typos and are not:
- Content sits **on the same line** as the header, one space after the colon. Blank line between fields.
- The style goes INSIDE `[Shot 1]`, right after the label — no style sentence before it.
- The alignment instruction line above the fields is written by the pipeline AFTER the body exists (final shot index N and effective duration S.SS are not knowable earlier). The separator in it is an EM DASH (—), not a hyphen. T2VA has no line at all. I2VA's line is fixed verbatim: "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
- FL2VA writes BARE `Picture 1` / `Shot 1` — no angle or square brackets — in the line AND the body. I2VA and L2VA bracket both.
- Per-mode anchors: I2VA — picture is the first frame, open on it, develop forward. FL2VA — write the motion path between the two frames; strongly favor a SINGLE shot so the model can interpolate. L2VA — picture is the LAST frame; infer a plausible earlier state and converge on it; a prompt that opens on the picture has the mode backwards.
- Shot/cut/camera/dialogue/audio rules match ref mode. Speakers take stable `(S1)`/`(S2)` IDs; compound `(S1,S2)` when they vocalise together; non-vocalising content gets no ID. A cut must introduce new information — if only distance/angle changes, move the camera instead.

## Review checklist (run on any H3 prompt)

1. Right skeleton for the mode; headers exact, lowercase, correct line placement.
2. Nothing outside the sections; no negatives-as-exclusion-list; no duration/fps/AR claims.
3. Labels: one line each, no invented or renumbered tags, subject-source assets cited inline not standalone.
4. summary prefix present, legal values only, matches what the assets are actually FOR.
5. Retention markers from the correct fixed set; scopes present; no (Sx) there.
6. Shots: [Shot 1] untimed, later timestamps strictly increasing and inside the duration; each cut uses one of the five phrases and opens its shot. If a measured shot list exists, shot count and times match it exactly.
7. One dominant camera move per shot, official vocabulary + fixed amplitude/speed phrases.
8. Observable-only language; dialogue format exact; supplied words verbatim.
9. Audio sections within sentence budgets; `N/A` used correctly; no lyric/dialogue repetition.
10. Ending resolved: final pose, position, framing, and whether the last image holds.
