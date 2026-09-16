# H3 Codex Promptor

Write, inspect, edit and refine prompts for local MiniMax H3 generation using
your Codex ChatGPT subscription. The node appears under **Trent / VLM**.
Existing H3 nodes and workflows are unchanged.

## Setup

Install [Codex CLI](https://learn.chatgpt.com/docs/cli) in the same environment
and user account as ComfyUI, and run `codex login`. For WSL ComfyUI, do this
inside WSL. Restart ComfyUI after updating TrentNodes, then refresh the browser.
The integration was tested with Codex CLI 0.135.0 on Ubuntu 24.04 / WSL.

The CLI must be on ComfyUI's PATH or at `~/.local/bin/codex`. To choose another
executable, set `TRENT_CODEX_BIN` in the environment that starts ComfyUI.
Leave **model** empty to use your Codex configured model, or enter an available
subscription model under **Advanced settings**. No API key is needed. The
node requires ChatGPT sign-in and will not fall back to API-key billing.

## Use

1. Open `example_workflows/H3_Codex_Promptor.json`, or add the node to your graph.
2. Describe the clip and match **duration_seconds** to your H3 generation.
3. Connect optional references and choose **Generate prompt**. This button
   queues the promptor and its upstream inputs only. Downstream video samplers
   and unrelated output nodes are excluded. Connected upstream generators can
   still run if their outputs are needed.
4. The result appears in the editable **prompt_text** box and is automatically
   locked. Connect **h3_prompt** to your H3 text input. Use the returned
   **checkpoint_hint** to confirm the matching FL2VA or Ref2VA checkpoint.
5. Edit the prompt directly, or type a **refinement** and click **Refine prompt**.
   Use **New variation** for a fresh draft of the original brief.

**Context & dialogue** reveals reference roles, extra context, exact dialogue,
and measured audio descriptions. Nonempty fields remain visible. Right-click
the node and select **Show Codex validation report** for diagnostics, inspected
frame times, assumptions, model and skill revision. The report is also a STRING
output. The two audio-section outputs support workflows with split text inputs.

The Generate/Refine buttons currently work on the main canvas. Inside a
subgraph, set **action** and use ComfyUI Run. Normal Run follows **action**:
`generate`, `refine`, or `locked`. Lock uses the editor exactly as saved,
without calling Codex. A standalone node is an output node and can run alone.

## References and modes

| Inputs in auto mode | Mode |
| --- | --- |
| Text only | T2VA |
| first_frame | I2VA |
| last_frame | L2VA |
| first_frame + last_frame | FL2VA |
| reference_images, video_frames or audio | Ref2VA |

Override **mode** to match your workflow. Base modes require their corresponding
picture count; use Ref2VA for arbitrary references. Picture labels follow this
order: **first_frame, last_frame, then reference_images** in batch order. Supply
the same references in the same order to H3 itself; this node produces text,
not H3 conditioning. Reference images are never silently reclassified as video.

For **video_frames**, connect a VHS loader's IMAGE output and set its actual
fps. The node reuses TrentNodes' scene/motion-aware keyframe selection. Codex
receives sampled JPEG frames with timestamps; this is not full-video or audio
inspection. Images are resized to a maximum side of 1344px for still references
or 1024px for sampled video frames before transmission.

**audio** supplies duration metadata and the availability of `<Audio 1>` in
Ref2VA. The audio itself is not sent to or heard by Codex. Connect H3 Audio
Soundscaper's `overall_soundscape`, `non_diegetic_music` and `sound_log` outputs
to **source_soundscape**, **source_music** and **sound_log**, or enter your own
descriptions/transcript. Explain its intended use in **reference_roles**.

## Skill, checks and caching

The first unlocked request downloads MiniMax's official
[`h3-prompt-writing` skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/d21241f0a4b3acbb34c97dae47fa417b7065e438/skills/h3-prompt-writing)
and both reference guides at commit `d21241f0a4b3acbb34c97dae47fa417b7065e438`.
Every file is SHA-256 verified and cached locally. These upstream documents are
downloaded from MiniMax, not redistributed as TrentNodes-authored instructions.
No global Codex skills, credentials or configuration are changed.

The model drafts and reviews against the selected official guide. Local H3
checks then inspect formatting, shot timing and available reference labels.
At most one corrective turn follows a completed draft with diagnostics.
Remaining diagnostics are reported, not silently rewritten or discarded.
Base keyframe alignment lines use the existing deterministic H3 renderer.
Passing these checks is not a guarantee of generation quality.

Successful results are stored under
`ComfyUI/user/default/trentnodes/h3_codex/results`; the verified skill is in
the sibling `skills` folder. The cache includes prompt context, inspected images,
mode, duration, model selection, effort, skill version and **revision**. Changing
only a downstream H3 seed does not generate another prompt. **New variation**
increments revision; use it after changing the default model in Codex settings
too. Locked prompts are saved in the workflow and work without Codex.

Briefs, context and inspected images go to Codex and consume your subscription's
usage. Local H3 rendering still uses your GPU. The bridge uses a private stdio
app-server process, ephemeral read-only sessions, disabled external integrations
and no shell/environment access. ComfyUI cancellation and the timeout stop that
process; transport failures are not automatically resubmitted.

## Troubleshooting

- **Login expired:** run `codex login` as the ComfyUI user. A cached
  `codex login status` result alone does not prove the token can refresh.
- **Model/request rejected:** check the installed CLI version, subscription
  limits and the requested model/effort combination. Leave model and effort at
  their defaults to follow Codex settings.
- **Prompt did not change:** unlock/generate or use New variation. In locked
  mode, edits to the brief or references intentionally do not rewrite the editor.
- **Skill download failed:** connect to GitHub once. Later requests use the
  verified local copy; locked mode does not need the skill download.

## Verification

```bash
python tests/test_h3_codex.py
node tests/h3_codex_js/run.mjs
```

Run Python tests in the ComfyUI environment. They cover subscription-only auth,
protocol event ordering, cancellations, timeouts, refusal of interactive actions,
cache/lock/refinement, reference labels, mode inference and official-example
calibration. Frontend checks cover branch-only queuing, positional widget
serialization, automatic locking and preserving edits made during generation.
