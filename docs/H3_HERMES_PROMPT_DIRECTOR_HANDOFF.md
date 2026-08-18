# H3 Hermes Prompt Director — implementation handoff

**Status:** design and next-session implementation brief. No Hermes-backed node has been implemented yet.

**Prepared:** 2026-08-15 CDT.

**Primary goal for the next session:** add a new ComfyUI node named **H3 Hermes Prompt Director** beside the working `Ultimate H3 Cowboy Promptor`. The new node must submit a structured H3 prompt job to Hermes Agent's local API server, allow Hermes to use bounded research/media/delegation tools, validate the returned prompt locally, and preserve the existing H3 reference pass-through contract.

**Do not modify or replace the working V1 node.** New code may import its pure helpers, but the implementation must not change V1's inputs, outputs, behavior, registration key, widget order, or tests.

Read these before editing:

1. This handoff.
2. `.hermes/plans/2026-08-15_033023-h3-prompt-director-v2.md` — the full semantic architecture and render-evaluation plan.
3. `docs/H3_NODE_FACE_HANDOFF.md` — widget serialization, append-only inputs/outputs, reference gaps, and current UI rules.
4. `docs/H3_COWBOY_HANDOFF.md` — current H3 package architecture.
5. `nodes/ultimate_h3_cowboy_promptor.py` — compatibility baseline only; do not edit it.

---

## 1. Executive decision

The Hermes version is feasible and should use Hermes Agent's official local **API Server**, specifically the asynchronous Runs API:

```text
POST /v1/runs
GET  /v1/runs/{run_id}
POST /v1/runs/{run_id}/stop
```

The API server runs the complete Hermes agent loop with tools, skills, memory policy, and optional subagents. The Runs API returns a run ID, supports polling, exposes progress events, and supports cancellation; this is a real Hermes invocation rather than a one-shot call to Hermes's underlying LLM.[1]

This will **not** inject a message into the currently open CLI conversation. Every ComfyUI execution gets a fresh Hermes run. That is deliberate: it prevents unrelated chat history from contaminating production prompts and gives the node a stable runtime boundary.

Use the current default Hermes profile for the first vertical slice. Keep the transport/profile abstraction explicit so a dedicated restricted `h3-director` profile can be introduced later without changing the node contract.

### Build a thin Hermes sibling first

Do not wait for every task in the large V2 plan before proving the integration. The next session should build a complete vertical slice:

```text
ComfyUI inputs
  -> local media staging
  -> versioned Hermes request
  -> Hermes Runs API
  -> strict JSON response parsing
  -> existing deterministic H3 assembly/validation
  -> familiar H3 outputs and reference pass-throughs
```

The semantic contract should match the V2 architecture, but the first node does not need the complete render benchmark, face-metric harness, hosted Context-IR baseline, or every future IR type.

---

## 2. Verified starting state

Verified on 2026-08-15 at handoff time:

- Repository: `/home/trent/ComfyUI/custom_nodes/TrentNodes`
- Branch: `main`, tracking `origin/main`
- Git status before this handoff: only `.hermes/` was untracked.
- Hermes Agent: `v0.20.0 (2026.8.3)`
- Hermes gateway: active/running.
- Gateway service definition: reported as outdated and should be refreshed during setup.
- Hermes API server: not enabled; nothing was listening on port `8642`.
- Webhook platform: disabled. It is not needed for this node.
- Existing H3 baseline from the completed planning pass: 281 Python checks + 26 JavaScript checks = 307 total.
- Current LM Studio host was unreachable from WSL (`No route to host`). Do not make it the default fallback until connectivity is restored.
- No Hermes node, client, request schema, response schema, temp-asset manager, or H3 Hermes skill exists yet.
- Existing V1 source has not been changed for this project direction.

The API server is the supported programmatic surface. Do not import private Hermes Python modules from TrentNodes and do not scrape an interactive TUI process. The HTTP boundary is versionable, cancellable, and independently testable.[1][2]

---

## 3. Definition of done for the next session

The next session is complete only when all of the following are true:

1. A new `H3 Hermes Prompt Director` node registers beside V1.
2. A text-only Base/T2VA job reaches the local Hermes Runs API and returns a valid H3 prompt.
3. At least one image-reference job stages a real local image, lets Hermes inspect it, and returns a prompt with correct physical reference numbering.
4. If practical in the same session, a short real video is staged and inspected with the video tool. If video-tool availability blocks this, record the exact tool/runtime blocker and leave the image path working.
5. The node polls the run with a hard timeout and sends `/stop` when ComfyUI execution is interrupted.
6. The API key never appears as a node widget, workflow value, prompt, analysis JSON, console log, test fixture, or committed file.
7. Hermes output is strict JSON and is validated locally before any prompt reaches node outputs.
8. The selected prompt passes the existing H3 assembler/validator with no unresolved hard errors.
9. `analysis_json` identifies the Hermes run ID, terminal status, quality mode, elapsed time, warnings, local validation, selected candidate, and whether any fallback ran.
10. V1's established test baseline still passes.
11. New client, schema, staging, timeout, cancellation, malformed-output, and node-registration tests pass.
12. A real live API smoke result is recorded honestly. Mock-only tests are not sufficient for completion.

A real MiniMax H3 render is **not** required to prove the bridge. It remains required before claiming this node produces better videos than V1.

---

## 4. Non-goals for the first implementation

Do not spend the next session on these before the vertical slice works:

- Replacing or rewriting `UltimateH3CowboyPromptor`.
- Completing the entire render-backed V2 benchmark.
- Exposing the API server outside `127.0.0.1`.
- Using a webhook as the request/response transport.
- Resuming this exact CLI conversation from ComfyUI.
- Storing API credentials in workflow JSON.
- Giving API-originated prompt jobs unrestricted messaging, cron, smart-home, repository-write, or credential access.
- Automatically starting a new Hermes gateway from every node execution.
- Automatically starting ComfyUI. Trent controls the live ComfyUI process.
- Claiming a model-reported list of tool calls is verified provenance. Verified provenance requires collecting actual run/SSE events.
- Claiming “perfect prompt.” H3 generation is stochastic and quality is multi-objective.

---

## 5. Runtime architecture

```text
H3 Hermes Prompt Director node
|
|-- validates widgets and physical reference slots locally
|-- resolves source frames, fps, duration, and H3 grid length locally
|-- stages only the required media under a unique job directory
|-- writes no credentials into that directory
|-- builds h3_hermes_request/1.0
|-- POSTs to Hermes /v1/runs
|-- polls /v1/runs/{id}
|   |-- checks ComfyUI interruption each poll
|   |-- POSTs /stop on interruption or timeout
|   `-- never retries by accidentally creating duplicate runs
|-- parses h3_hermes_result/1.0 from the final output
|-- checks request ID/schema/asset bindings
|-- runs the existing deterministic H3 assembler and validator
|-- rejects or repairs bounded structural errors locally
|-- returns prompt + diagnostics + untouched reference pass-throughs
`-- retains or cleans staged assets according to an explicit policy
```

### Why Runs API

The API server's Runs API is intended for long-form tool-using agent work. It supports polling, terminal states (`completed`, `failed`, `cancelled`), SSE progress events, and a stop endpoint.[1] That makes it safer than holding one synchronous chat-completion connection open while Hermes performs vision, web research, or subagent work.

Use polling for the first implementation. Add SSE progress collection after the request/response path is stable.

### Fresh run per generation

Generate a unique external session ID such as:

```text
comfyui:h3:<prompt-id-or-uuid>
```

Send it as `session_id` for correlation, but do not use a named persistent conversation and do not reuse `previous_response_id`. Each H3 request should be semantically self-contained.

Do not set `X-Hermes-Session-Key` for the initial node. That avoids unintentional long-term-memory coupling between jobs. If H3-specific memory is enabled later, use a dedicated stable key and document exactly what may be remembered.

---

## 6. Hermes API setup

Hermes's API server is disabled by default and requires bearer authentication even on loopback.[1]

### One-time setup

The next session should first verify the live installation and current docs, then enable:

```bash
hermes config set API_SERVER_ENABLED true
hermes config set API_SERVER_KEY '<strong random local secret>'
hermes gateway restart
```

`hermes config set` routes the secret to Hermes's environment file; do not commit or print it.

Keep the default host and port:

```text
host = 127.0.0.1
port = 8642
```

Do not enable CORS. ComfyUI's Python process calls Hermes server-to-server and does not need browser CORS.

Because the service is currently reported outdated, refresh it if a normal restart does not pick up the API settings:

```bash
hermes gateway install
hermes gateway restart
```

### Required preflight

Run these before implementing the client:

```bash
curl -fsS http://127.0.0.1:8642/health
curl -fsS \
  -H "Authorization: Bearer $HERMES_AGENT_API_KEY" \
  http://127.0.0.1:8642/v1/capabilities
curl -fsS \
  -H "Authorization: Bearer $HERMES_AGENT_API_KEY" \
  http://127.0.0.1:8642/v1/toolsets
curl -fsS \
  -H "Authorization: Bearer $HERMES_AGENT_API_KEY" \
  http://127.0.0.1:8642/v1/skills
```

Expected capabilities must include run submission, run status, and run stop. Do not assume endpoint support from version text alone; discover it from `/v1/capabilities`.[1]

### ComfyUI credential source

TrentNodes should resolve the client bearer token in this order:

1. `HERMES_AGENT_API_KEY` in the environment that starts ComfyUI.
2. No second source in the MVP.

Do **not** reuse V1's `api_key` widget. Do not automatically parse `~/.hermes/.env` from node code; that silently couples workflow execution to another application's private config and makes portable deployment harder.

If the key is absent, raise an actionable error:

```text
Hermes API key missing. Export HERMES_AGENT_API_KEY in the environment that starts ComfyUI, then restart ComfyUI. Do not put the key in a workflow widget.
```

The API server's key and `HERMES_AGENT_API_KEY` must contain the same secret, but they deliberately use different variable names so the node never depends on Hermes internals.

---

## 7. Tool and security posture

Hermes documents that the default API-server platform can expose powerful tools, including terminal commands.[1] The H3 node should use the smallest platform toolset that still supports evidence-based prompting.

For the first safe profile, prefer:

- `safe` — web search/extract plus vision.
- `video` — only when configured and verified.
- `skills` — only if the job contract requires a skill.
- `delegation` — balanced/hero only.

Toolsets are configurable per platform; inspect and change the `api_server` platform specifically, not the CLI platform.[3]

Useful commands:

```bash
hermes tools list --platform api_server
hermes tools enable safe video skills delegation --platform api_server
```

Explicitly disable capabilities the H3 job does not need, after inspecting the live list:

```bash
hermes tools disable terminal file cronjob messaging homeassistant spotify --platform api_server
```

Do not blindly run the disable line if a toolset name is absent. Confirm with:

```bash
hermes tools list --platform api_server
```

Important limitations:

- `file` contains both read and write operations; it is not a read-only bundle.[3]
- `skills` includes management operations as well as viewing. The task instructions must say not to create, edit, or delete skills during a prompt job.
- Per-request “max tool calls” and “max subagents” are not documented as hard Runs API controls. Quality-mode budgets in the request are therefore **soft instructions** in the MVP.
- Hard controls in the MVP are localhost binding, bearer auth, platform toolset restriction, request/asset size limits, wall-clock timeout, cancellation, output-size limit, and local validation.
- A dedicated `h3-director` Hermes profile is the recommended production hardening step if stronger isolation is needed.

### Media privacy

- Stage files only under the configured ComfyUI temp root.
- Resolve every path and verify it is inside the request's own UUID directory.
- Reject symlinks escaping the job directory.
- Do not send private media to remote URLs unless the user explicitly selects a remote Hermes deployment and accepts its retention policy.
- Never include credentials, home-directory listings, unrelated files, or repository contents in the task package.
- Default cleanup: delete successful job assets after the result is parsed; retain failed job assets for a bounded 24-hour debugging window, with paths listed in diagnostics.
- Provide one cleanup function and tests; do not scatter ad hoc deletion across the node.

---

## 8. Node interface

### Registration

Recommended class and display names:

```python
NODE_CLASS_MAPPINGS["TrentH3HermesPromptDirector"] = H3HermesPromptDirector
NODE_DISPLAY_NAME_MAPPINGS["TrentH3HermesPromptDirector"] = (
    "H3 Hermes Prompt Director"
)
```

Category:

```text
Trent/VLM
```

### Input strategy

The node should feel familiar to a V1 user. Reuse V1's vocabulary and reference semantics, but implement a new class.

Core inputs:

- `h3_mode`: `ref`, `base_T2VA`, `base_I2VA`, `base_FL2VA`, `base_L2VA`
- `target_description`
- `quality_mode`: `fast`, `balanced`, `hero`; default `balanced`
- `research_policy`: `never`, `when_uncertain`, `always`; default `when_uncertain`
- subject rows and typed `subjects`
- `video` / `frames` / `fps`
- `audio`
- `video_role` / `audio_role`
- `cut_times`
- exact `dialogue`
- `constraint_notes`
- `duration_override`
- Base picture controls
- music-video controls
- ordered subject image sockets

Hermes-specific advanced inputs:

- `hermes_base_url`: default `http://127.0.0.1:8642`
- `timeout_seconds`: default `900`, bounded range
- `poll_interval_seconds`: default `1.0`, bounded range
- `cleanup_policy`: `delete_on_success`, `retain_24h`, `retain`
- `fallback_policy`: `fail_closed`, `existing_provider`; default `fail_closed`
- `fallback_provider` and `fallback_model`, used only when fallback is explicitly selected
- optional provider/model route fields for Hermes; blank means gateway default

Do not add an API-key widget.

Do not dynamically remove widgets from `node.widgets`. If JavaScript hides controls, use the same `widget.hidden` and `widget.options.hidden` approach documented in `H3_NODE_FACE_HANDOFF.md`. Inputs and outputs are append-only after release.

### Outputs

Keep the same first five outputs as V1 so the new node is easy to compare and swap:

1. `h3_prompt` — `STRING`
2. `duration_seconds` — `FLOAT`
3. `fps` — `INT`
4. `analysis_json` — `STRING`
5. `h3_checkpoint_hint` — `STRING`

Then use the same ordered H3 pass-through outputs as V1:

- `ref_image_1` ... `ref_image_6` initially
- `ref_video`
- `ref_video_audio`
- `ref_audio`
- `width`
- `height`
- `length`
- `label_map`

A later compatibility phase may increase to the official H3 limits of 9 images, 3 videos, and 3 audio clips. Do not silently reorder V1-compatible slots during the MVP.

Place candidate prompts, score vectors, run ID, and tool diagnostics inside `analysis_json` rather than adding a wide row of rarely used output sockets.

---

## 9. Media staging contract

Create one job directory per execution:

```text
<ComfyUI temp>/h3_hermes/<request_uuid>/
```

Recommended names:

```text
picture_01.jpg
picture_02.jpg
reference_video_01.mp4
reference_audio_01.wav
manifest.json
```

`manifest.json` contains no secret and no base64 media. It records:

```json
{
  "schema_version": "h3_asset_manifest/1.0",
  "request_id": "uuid",
  "assets": [
    {
      "asset_id": "picture_01",
      "h3_label": "<Picture 1>",
      "kind": "image",
      "path": "/absolute/allowlisted/path/picture_01.jpg",
      "intended_jobs": ["identity"],
      "prohibited_transfers": ["pose", "motion", "audio"],
      "sha256": "...",
      "bytes": 12345
    }
  ]
}
```

Use existing TrentNodes encoders where possible:

- `utils/h3_prompt/imaging.py::tensor_to_jpeg_b64`
- `utils/h3_prompt/video_io.py::prepare_video`
- `utils/h3_prompt/audio_io.py::audio_to_wav_b64`

Decode their base64/byte results into the job directory; do not add a second independent media-conversion stack.

Limits to enforce locally before submission:

- maximum asset count
- maximum bytes per asset
- maximum total staged bytes
- maximum request-text size
- valid media extensions and MIME types
- no path outside the UUID job directory
- no reference-slot gaps in strict mode
- H3 duration policy and frame-grid normalization

Hash calculations should be used only for manifest identity and debugging; never log raw media or base64.

---

## 10. Request contract

The Runs API accepts a string `input` plus optional `instructions` and session fields.[1] Embed the versioned request object as JSON in `input`; keep stable agent behavior in `instructions`.

### `h3_hermes_request/1.0`

Minimum request shape:

```json
{
  "schema_version": "h3_hermes_request/1.0",
  "request_id": "uuid",
  "target_model": "MiniMax H3",
  "h3_mode": "ref",
  "quality_mode": "balanced",
  "research_policy": "when_uncertain",
  "creative_brief": "...",
  "exact_literals": {
    "dialogue": "...",
    "lyrics": "...",
    "visible_text": []
  },
  "generation": {
    "requested_duration_seconds": 8.0,
    "snapped_duration_seconds": 8.041,
    "fps": 24.0,
    "width": 768,
    "height": 432,
    "length": 197
  },
  "task": {
    "task_types": ["reference video editing"],
    "video_role": "edit_source",
    "audio_role": "none",
    "constraints": [],
    "cut_timestamps": []
  },
  "subjects": [],
  "assets": [],
  "budgets": {
    "candidate_count": 2,
    "max_repairs": 1,
    "tool_call_target": 18,
    "subagent_target": 1,
    "wall_clock_timeout_seconds": 900
  },
  "required_response_schema": "h3_hermes_result/1.0"
}
```

The task instructions must require Hermes to:

1. Treat local asset paths as the only allowed private-media scope.
2. Inspect relevant assets with actual tools when available.
3. Separate observations from assumptions.
4. Use official/current H3 evidence first; label community evidence separately.
5. Build a canonical intent plan before drafting.
6. Preserve exact dialogue, lyrics, visible text, reference bindings, and user-locked constraints.
7. Produce deliberately different candidates in `hero`, not paraphrases.
8. Critique candidates against the typed intent, not generic prose beauty.
9. Return JSON only.
10. Never write/delete files, change configuration, manage skills, send messages, schedule jobs, or modify repositories.

Quality-mode intent:

- `fast`: one evidence/writing path, no delegation, one candidate.
- `balanced`: analyze/plan, draft two candidates, independent critique, at most one repair.
- `hero`: three policies — literal/minimal, continuity/identity, temporal/audiovisual — bounded independent review and ranking.

The node's timeout remains authoritative even if Hermes ignores a soft budget.

---

## 11. Response contract

### `h3_hermes_result/1.0`

Hermes must return one JSON object and nothing else:

```json
{
  "schema_version": "h3_hermes_result/1.0",
  "request_id": "uuid",
  "status": "ok",
  "evidence": {
    "observations": [],
    "assumptions": [],
    "uninspected_assets": []
  },
  "intent_ir": {
    "required_atoms": [],
    "preferred_atoms": [],
    "optional_atoms": [],
    "reference_jobs": []
  },
  "candidates": [
    {
      "candidate_id": "balanced_1",
      "policy": "literal_minimal",
      "prompt": "...",
      "score_vector": {
        "required_intent_coverage": 1.0,
        "unsupported_additions": 0,
        "contradictions": 0,
        "reference_fidelity": 1.0,
        "temporal_av_feasibility": 1.0,
        "prompt_economy": 0.9
      },
      "critic_findings": []
    }
  ],
  "selected_candidate_id": "balanced_1",
  "h3_prompt": "...",
  "repairs": [],
  "quality_report": {
    "hard_errors": [],
    "warnings": [],
    "unresolved_ambiguities": [],
    "reported_tools": [],
    "reported_sources": []
  }
}
```

Local parser requirements:

- Extract one JSON object from a raw final string, tolerating one surrounding Markdown fence only.
- Reject missing/unknown incompatible schema versions.
- Require matching request IDs.
- Require `status == "ok"`.
- Require a non-empty `h3_prompt`.
- Require every selected reference label to exist in the submitted manifest.
- Cap response bytes and candidate count.
- Do not trust model-supplied score values as hard validation.
- Run the selected prompt through local deterministic H3 validation.
- Keep the raw Hermes final response only in bounded debug diagnostics; redact the bearer token and never include media base64.

`reported_tools` is model-reported until actual SSE/tool events are collected. Name it accordingly.

---

## 12. Local deterministic authority

Hermes is the reasoning/orchestration engine, not the final syntax authority.

Local TrentNodes code remains responsible for:

- H3 mode and checkpoint compatibility.
- Required Base/Ref section order.
- Task-type line and legal labels.
- Reference numbering and physical slot identity.
- Dialogue/lyrics/visible-text exact preservation.
- Duration and shot bounds.
- 7,000-character maximum.
- H3 frame-grid snapping.
- Reference gap policy.
- Hard validation and bounded structural repair.
- Pass-through asset order.

Reuse, do not duplicate, the existing H3 modules where possible:

- `utils/h3_cowboy/spec.py`
- `utils/h3_cowboy/assembler.py`
- `utils/h3_cowboy/subjects.py`
- existing reference-wiring helpers
- existing H3 canvas/frame-grid helpers

If Hermes returns invalid prose:

1. Attempt strict JSON extraction.
2. If a candidate prompt exists, process it through the existing assembler.
3. Apply only existing deterministic structural fixes.
4. If hard errors remain, do not return it as success.
5. If `fallback_policy == existing_provider`, invoke the explicitly configured existing provider and mark the fallback.
6. Otherwise fail closed with actionable diagnostics.

Never silently substitute a plausible-looking prompt after a failed Hermes run.

---

## 13. Failure, retry, cancellation, and fallback behavior

### Transport errors

- Connection refused / DNS / TLS / authentication: fail immediately with actionable setup guidance.
- HTTP 429: bounded exponential backoff on the **submission** request, with a maximum retry count. Respect the server's concurrent-run cap.[1]
- HTTP 5xx: at most one bounded submission retry using an idempotency key if supported; otherwise do not create a duplicate run blindly.
- Polling transient error after a run ID exists: retry polling the same run ID; never resubmit.

### Timeout

When wall-clock timeout expires:

1. `POST /v1/runs/{run_id}/stop`.
2. Poll briefly for `cancelled` or terminal state.
3. Raise a timeout error containing the run ID but no secret.
4. Apply fallback only when explicitly enabled.

### ComfyUI interruption

In every polling loop call:

```python
comfy.model_management.throw_exception_if_processing_interrupted()
```

If interrupted, send `/stop` in `finally`, then re-raise the ComfyUI interruption exception. Never convert user cancellation into fallback generation.

### Hermes terminal states

Handle explicitly:

- `completed`: parse and validate output.
- `failed`: surface server error and optional explicit fallback.
- `cancelled`: raise cancellation; no fallback.
- `stopping`: keep polling only during a short cancellation grace period.
- unknown status: fail safely and retain diagnostics.

### Fallback

Default: `fail_closed`.

Optional: `existing_provider`, with provider/model selected on the node and credentials resolved from environment variables only. The analysis must record:

```json
{
  "engine_requested": "hermes_agent",
  "engine_used": "existing_provider",
  "fallback_used": true,
  "hermes_error": "sanitized summary"
}
```

Do not default to LM Studio while it remains unreachable from WSL.

---

## 14. Proposed files

Create:

```text
utils/h3_hermes/__init__.py
utils/h3_hermes/client.py
utils/h3_hermes/assets.py
utils/h3_hermes/contract.py
utils/h3_hermes/schema.py
nodes/h3_hermes_prompt_director.py
js/h3_hermes_prompt_director.js              # only if dynamic visibility is needed
tests/test_h3_hermes_client.py
tests/test_h3_hermes_assets.py
tests/test_h3_hermes_contract.py
tests/test_h3_hermes_node.py
tests/h3_hermes_js/run.mjs                    # only if JS is added
tools/h3_hermes_api_smoke.py
```

Modify only:

```text
__init__.py                                  # register the new node
```

Do not modify for the MVP:

```text
nodes/ultimate_h3_cowboy_promptor.py
utils/h3_cowboy/*
js/h3_cowboy.js
tests/h3_cowboy_js/run.mjs
```

Import existing helpers from those modules instead.

If implementing the new node without duplicating large V1 orchestration code proves impossible, stop and document the exact seam needed. Prefer a new shared pure helper module plus behavior-equivalence tests; do not casually refactor V1 during the bridge implementation.

---

## 15. Implementation sequence

### Task 1 — runtime preflight

1. Read the current API Server docs, not only this handoff.
2. Enable the API server and refresh the gateway service if needed.
3. Verify `/health`, `/v1/capabilities`, `/v1/toolsets`, and `/v1/skills`.
4. Submit a manual text-only `/v1/runs` request.
5. Poll it to `completed` and save the sanitized response shape.
6. Prove `/stop` on a deliberately long harmless run.

Do not write node code until the live transport is proven.

### Task 2 — client tests first

Write failing tests for:

- bearer header present but never logged
- capabilities preflight
- run submission
- poll to completion
- transient poll retry using the same run ID
- 401/403 guidance
- 429 bounded retry
- timeout -> stop
- interruption -> stop and re-raise
- failed/cancelled/unknown states
- output-size cap

Use a local fake HTTP server or mocked transport; no live gateway dependency in normal tests.

Then implement `utils/h3_hermes/client.py` with a small typed result object.

### Task 3 — asset staging tests first

Write failing tests for:

- image staging and manifest labels
- MP4 staging using the existing encoder
- WAV staging using the existing encoder
- hashes and byte counts
- total-byte and asset-count limits
- path containment
- symlink escape rejection
- cleanup policy
- no secret fields in manifest

Then implement `utils/h3_hermes/assets.py`.

### Task 4 — request/response contract tests first

Write failing tests for:

- deterministic request serialization
- exact literal preservation
- reference jobs and prohibited transfers
- quality-mode budgets
- strict schema version
- matching request ID
- fenced JSON extraction
- malformed/truncated output
- too many candidates
- missing selected candidate
- invalid reference labels
- response-size cap

Then implement `contract.py` and `schema.py` without adding a heavy schema dependency unless one is already available.

### Task 5 — new node, text-only vertical slice

1. Register `TrentH3HermesPromptDirector`.
2. Implement Base/T2VA first; no media.
3. Submit a real run through `HermesRunsClient`.
4. Validate the returned prompt locally.
5. Populate `analysis_json`.
6. Run registration and text-only node tests.

This proves the agent boundary before media complexity.

### Task 6 — image references and pass-throughs

1. Add ordered subject image staging.
2. Preserve physical slot numbers and fail closed on gaps.
3. Ensure Hermes receives each intended job and prohibited transfer.
4. Validate returned labels against the manifest.
5. Return images untouched on the V1-compatible output slots.
6. Run one real image-reference smoke.

### Task 7 — video/audio and cancellation

1. Stage the original source MP4 when safely reusable; otherwise use `prepare_video`.
2. Stage WAV through the existing audio encoder.
3. Verify API-server `video` tool availability before claiming video was inspected.
4. Poll with ComfyUI interruption checks.
5. Prove a cancelled node causes Hermes `/stop`.
6. If audio inspection is unavailable, return an explicit `uninspected_assets` warning instead of pretending it was heard.

### Task 8 — UI behavior

If reusing V1's visible/hidden controls requires JavaScript:

1. Add a new extension file targeting only the new class key.
2. Never remove widgets from `node.widgets`.
3. Add a dedicated mocked frontend harness.
4. Verify old workflow serialization is untouched.
5. Perform the canvas checklist after Trent restarts ComfyUI.

### Task 9 — full verification

Run targeted tests, then the complete existing H3 baseline, then a live API smoke. Inspect `git diff` and verify V1 files are unchanged.

Do not commit unless Trent asks for a commit.

---

## 16. Test commands

Use the ComfyUI environment:

```bash
cd /home/trent/ComfyUI/custom_nodes/TrentNodes
/home/trent/ComfyUI/venv/bin/python tests/test_h3_hermes_client.py
/home/trent/ComfyUI/venv/bin/python tests/test_h3_hermes_assets.py
/home/trent/ComfyUI/venv/bin/python tests/test_h3_hermes_contract.py
/home/trent/ComfyUI/venv/bin/python tests/test_h3_hermes_node.py
```

If conventional pytest tests are added, invoke them with the same interpreter and prove the actual command used.

Existing H3 regression gate:

```bash
cd /home/trent/ComfyUI/custom_nodes/TrentNodes
for t in h3_assembler h3_format h3_keyframes h3_node h3_audio h3_video \
         h3_cowboy_subjects h3_cowboy_ref h3_cowboy_node \
         h3_cowboy_base h3_wiring; do
  /home/trent/ComfyUI/venv/bin/python "tests/test_${t}.py"
done
node tests/h3_cowboy_js/run.mjs
```

If the current repository has added H3 tests since this handoff, discover and run those too. Do not report only the historical 307 total if the live suite now contains more checks.

Live API smoke:

```bash
/home/trent/ComfyUI/venv/bin/python tools/h3_hermes_api_smoke.py \
  --base-url http://127.0.0.1:8642 \
  --mode base_T2VA \
  --brief "A locked-off five-second shot of rain forming concentric rings in a black ceramic bowl"
```

The smoke tool must read the key from `HERMES_AGENT_API_KEY`, sanitize output, submit once, poll, validate the final schema, and exit non-zero on any failure.

---

## 17. `analysis_json` requirements

At minimum:

```json
{
  "engine_requested": "hermes_agent",
  "engine_used": "hermes_agent",
  "fallback_used": false,
  "hermes": {
    "base_url": "http://127.0.0.1:8642",
    "run_id": "run_...",
    "status": "completed",
    "elapsed_seconds": 0.0,
    "quality_mode": "balanced",
    "research_policy": "when_uncertain",
    "usage": {},
    "reported_tools": [],
    "verified_tool_events": []
  },
  "request": {
    "schema_version": "h3_hermes_request/1.0",
    "request_id": "uuid",
    "asset_count": 0,
    "staged_bytes": 0
  },
  "selection": {
    "selected_candidate_id": "balanced_1",
    "candidate_count": 2,
    "score_vector": {}
  },
  "validation": {
    "hard_errors": [],
    "applied_fixes": [],
    "warnings": [],
    "char_count": 0
  },
  "uninspected_assets": [],
  "cleanup": {
    "policy": "delete_on_success",
    "retained_path": null
  }
}
```

Never include:

- Authorization headers
- API keys
- environment dumps
- base64 media
- raw audio/video bytes
- unrelated filesystem paths
- full model/provider credential configuration

---

## 18. Acceptance matrix

| Case | Expected result |
|---|---|
| API server disabled | Immediate setup error; no fallback unless explicitly selected |
| API key missing | Immediate environment guidance; never request a widget secret |
| Wrong API key | Sanitized authentication error |
| `/v1/capabilities` lacks Runs API | Clear unsupported-runtime error; do not guess endpoints |
| Text-only T2VA | Valid Base prompt and local validation report |
| Ordered image refs | Correct `<Picture N>` labels and untouched pass-throughs |
| Image slot gap | Fail closed in strict mode |
| Video tool unavailable | Prompt may proceed from keyframes only if policy allows; diagnostics say uninspected video |
| Audio tool unavailable | Never claim audio analysis; preserve exact supplied audio/lyrics metadata |
| Malformed Hermes JSON | Local parse failure, bounded recovery, then explicit fallback or failure |
| Valid JSON / invalid H3 | Existing assembler fixes bounded structure; hard leftovers fail |
| Timeout | Stop endpoint called; run ID reported; optional explicit fallback |
| User cancellation | Stop endpoint called; no fallback |
| Hermes 429 | Bounded backoff; no request flood |
| Hermes restart during poll | Continue polling same run when possible; never blind duplicate |
| Successful run | Valid prompt, diagnostics, cleanup per policy |
| V1 regression | Release blocked |

---

## 19. Deferred production work

After the bridge is real and stable:

1. Promote the request/response contract into the full typed V2 `H3PromptSpec` architecture.
2. Add SSE event collection for verified tool/delegation provenance and UI progress.
3. Create a dedicated `h3-director` Hermes profile with restricted tools, isolated memory, dedicated API key, and optionally a sandboxed terminal.
4. Add a project-controlled H3 Prompt Director skill only if it improves maintainability; do not make the node depend on mutable global skill state for hard syntax.
5. Expand reference capacity to official H3 limits while preserving slot identity.
6. Separate analyzer, writer, and critic model routes.
7. Add deterministic literal/minimal fallback compilation.
8. Run matched-seed V1 vs Hermes fast/balanced/hero renders.
9. Evaluate identity/reference fidelity, action completion, temporal consistency, dialogue/ASR, OCR, lip sync, AV timing, prompt economy, latency, cost, and blinded human preference.
10. Promote the Hermes version only after the render-backed release gate passes.

---

## 20. Known traps

- A ComfyUI node cannot message this exact interactive chat without an explicit API/session boundary.
- The Runs API `input` is text. Use local allowlisted paths for video/audio rather than pretending it accepts arbitrary file uploads.[1]
- API-server `instructions` layer on Hermes's core system prompt; they do not replace the agent or remove tools.[1]
- Tool budget fields in the H3 request are soft until enforced by a profile/runtime layer.
- The default API-server platform is powerful. Loopback + auth is necessary but not sufficient; restrict platform tools.
- V1 currently has six image sockets, while official H3 supports more. Do not solve this by silently compacting gaps.
- Workflow widget values are positional. Never remove a widget from `node.widgets`.
- The current V1 provider path samples keyframes. Do not accidentally claim full-video inspection unless the new Hermes path stages and analyzes the video.
- A model's `reported_tools` is not proof. Capture SSE events before calling them verified.
- Do not let retries create multiple expensive Hermes runs.
- Do not fall back after user cancellation.
- Do not default to the currently unreachable LM Studio route.
- Do not mutate global Hermes skills/config during an H3 prompt run.
- Do not claim the prompt is better until H3 renders prove it.

---

## 21. First message for the next session

Paste or paraphrase this:

> Implement the H3 Hermes Prompt Director vertical slice described in `docs/H3_HERMES_PROMPT_DIRECTOR_HANDOFF.md`. Read that file, `.hermes/plans/2026-08-15_033023-h3-prompt-director-v2.md`, and `docs/H3_NODE_FACE_HANDOFF.md` first. Use Hermes's official API Server Runs API, prove the live transport before writing node code, and add a new node beside V1. Do not modify `nodes/ultimate_h3_cowboy_promptor.py` or V1's JS/tests. Keep API credentials out of widgets and source control. Work test-first, run the full live H3 regression gate, and finish with a real text-only plus image-reference Hermes API smoke. Do not claim full video/audio inspection unless actual tool evidence proves it.

---

## Sources

[1] https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server — Hermes Agent, “API Server”; official endpoint, authentication, Runs API, cancellation, capability discovery, system-prompt layering, and security documentation.

[2] https://hermes-agent.nousresearch.com/docs/developer-guide/programmatic-integration — Hermes Agent, “Programmatic Integration”; official guidance for external callers.

[3] https://hermes-agent.nousresearch.com/docs/reference/toolsets-reference — Hermes Agent, “Toolsets Reference”; official per-platform toolset and capability definitions.
