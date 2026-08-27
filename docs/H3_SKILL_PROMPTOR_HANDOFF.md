# H3 Skill Promptor — handoff

## RESOLVED 2026-08-26 — reasoning_effort medium/xhigh empty prompt (v1.15.1)

**Was:** llama-server's `max_tokens` counts reasoning AND content
together; at `xhigh` the model spent all 3072 tokens thinking
(`finish_reason: "length"`, content empty) → VALIDATION FAILED with an
empty h3_prompt.

**Fix (all five plan items done):**
1. `THINKING_ALLOWANCE = {low: 2048, medium: 3072, xhigh: 7168}` lives
   in `utils/h3_skill/client.py` (shared with the dev CLI). The request
   asks for `max_tokens + allowance` AND caps thinking at the allowance
   via per-request `reasoning_budget_tokens` — this build (b10630) has
   it; verified live: the server force-closes thinking at the cap and
   injects `BUDGET_MESSAGE` so the model is not cut mid-word. Content
   therefore always keeps ≥ max_tokens of room; the widget now means
   "prompt-text budget" (tooltip updated). Servers without the feature
   ignore the extra keys and still get the inflated max_tokens.
2. `client.chat` puts `finish_reason` in the usage dict. The node's
   `_check_finish` raises an actionable error on empty content +
   `length` ("raise max_tokens or lower reasoning_effort") BEFORE any
   retry (a retry starves the same way), and warns in the report when
   non-empty content hit the limit. finish_reason is in the report's
   latency line.
3. Loopback fakes accept `(text, finish_reason)` tuples; new tests
   cover budget scaling per effort, the starvation error (no retry
   burned), truncation warnings, and the budget keys on the wire.
4. `tools/h3_skill_dev_run.py` applies the same policy (it builds its
   own chat kwargs — keep it in sync via the client constants).
5. Soundscaper audit: the omni Instruct model does not think, so its
   1500 is pure reply budget — confirmed fine; it now warns in the
   report when a reply is truncated at max_tokens.

Live E2E after the fix: ref2va at xhigh, 6314 completion tokens
(thinking alone exceeded the old 3072 ceiling), finish "stop",
checklist PASS first try.

**Session state 2026-08-26:** repo at `bfd6a53` on `main`
(v1.15.0 + README model links) — both nodes, per-port server manager,
5 offline suites green, live E2Es done. Servers likely still resident:
8735 Qwen3.8 (shell-spawned; ComfyUI attaches to it as a foreign
server), 8736 omni (spawned inside ComfyUI's process on first
Soundscaper run). Models in `models/LLM/` (Qwen3.8 V3 quant, omni
pair, audio-flamingo-3-hf [16 GB, deletable - superseded by AF-Next],
audio-flamingo-next-captioner-hf). V1 Qwen3.8 quant still on
F:\Models. Demo workflow: Downloads/H3_Skill_Promptor.json; A/B
results: Downloads/H3_audio_captioner_AB.md.

> **2026-08-26 addendum — H3 Audio Soundscaper.** New sibling node
> `nodes/h3_audio_soundscaper.py` + `utils/h3_skill/{audio_io,audio_prompts}.py`:
> hears a clip's audio with Qwen3-Omni-30B (A/B winner, see
> Downloads/H3_audio_captioner_AB.md) and outputs
> overall_soundscape / non_diegetic_music / dialogue / sound_log.
> The server manager became **per-port multi-slot** (`_slots` dict) so
> the omni server (8736) coexists with the text VLM (8735);
> `stop_server()` stops all slots, `stop_server(port=...)` one.
> `ServerSpec.reasoning_effort=""` omits `--chat-template-kwargs`
> (the omni template has no such variable) and `client.chat`'s
> `reasoning_effort=None` omits the per-request kwarg likewise.
> Audio rides as OpenAI `input_audio` wav-base64 parts.
> Same day, later: the promptor gained optional `source_soundscape` /
> `source_music` / `sound_log` inputs (appended after max_tokens -
> widget order is append-only). They carry the Soundscaper's measured
> analysis into `build_user_context` as a "MEASURED AUDIO" block that
> anchors the prompt's audio sections. Also: `reference_images` batches
> > 8 frames auto-route to `video_frames` when that socket is free, and
> `_stem()` strips only a literal .gguf (splitext truncated dotted model
> names - the "serving X, not X" bug).

Built 2026-08-25. Standalone MiniMax H3 prompt generation with a local
GGUF vision LLM (Qwen3.8-27B-UD-Q4_K_XL + mmproj-F16) under a managed
`llama-server`. Deliberately independent of the H3AutoPromptGenerator /
Cowboy pipeline.

## Files

| Path | Role |
|---|---|
| `nodes/h3_skill_promptor.py` | `H3SkillPromptor` + `H3LocalLLMStop` nodes; `llm_gguf` folder category |
| `utils/llamacpp_server.py` | llama-server lifecycle: find/spawn/health/attach/stop, single slot, VRAM gate |
| `utils/h3_skill/skill_loader.py` | loads the h3-prompting skill (live → vendored), builds system prompt + user context |
| `utils/h3_skill/checklist.py` | deterministic read-only validator; `assemble_final` adds the base alignment line |
| `utils/h3_skill/client.py` | OpenAI-protocol chat with data-URI images and `chat_template_kwargs` |
| `utils/h3_skill/assets/h3_prompting_skill.md` | vendored skill snapshot (refresh when the skill changes materially) |
| `tools/h3_skill_dev_run.py` | E2E CLI without ComfyUI |
| `tests/test_llamacpp_server.py`, `tests/test_h3_skill_checklist.py`, `tests/test_h3_skill_promptor.py` | offline suites |

## Design decisions

- **The skill IS the system prompt.** `~/.claude/skills/h3-prompting/SKILL.md`
  is read at run time; the vendored snapshot only backs it up. One official
  MiniMax example (from `utils/h3_cowboy/spec.py`) rides along per mode; base
  examples get their alignment line split off so the model never learns to
  write it.
- **No silent rewrites.** `checklist.validate()` returns errors; the node
  retries ONCE with the numbered list; remaining violations go into
  `validation_report`. Only transport artifacts (leading `<think>`, a
  wrapping code fence) are stripped, and each strip is reported.
- **Validator is official-example-calibrated.** Anything MiniMax's own
  examples do is legal: retention `(scope)` is optional, dialogue verbs vary
  (`says`/`exclaims`/`replies`), speaker IDs sit anywhere earlier in the
  clause. The deterministic invariants kept: exact headers, task-type
  vocabulary, non-mixing retention markers, five cut phrases opening each
  later shot, `[Shot 1]` untimed, strictly increasing timestamps inside the
  duration, `<d>[Language]...</d>` shape, no exclusion lists, no
  duration/fps/AR claims.
- **Base alignment line** is rendered AFTER generation via
  `h3_cowboy.spec.render_instruction_line` (final shot parsed from the body,
  duration from the widget) and prepended by `assemble_final`. T2VA has none.
- **Server policy: resident.** One managed slot; same `ServerSpec` reuses,
  a changed spec stops-and-respawns (`reasoning_effort` is excluded from
  spec equality — it rides per request via `chat_template_kwargs`, so
  flipping the widget never reloads the model). No idle daemon —
  `H3LocalLLMStop` frees the ~20 GiB when a video job needs it. `atexit`
  stops it with ComfyUI; `stop_orphan()` reaps a server left by a hard
  crash (matched by binary name + exact port).
- **VRAM gate**: spawn refuses when free VRAM < model+mmproj file size
  + 4 GiB headroom, unless `free_vram_first` unloads ComfyUI models first.
- **Locking**: `_lock` guards slot state only; the startup health poll
  runs OUTSIDE it and honors ComfyUI's Cancel
  (`throw_exception_if_processing_interrupted`), so a cold load never
  wedges shutdown or the queue.
- **Stop node ordering**: an OUTPUT_NODE with no inputs is scheduled
  FIRST by ComfyUI's picker — exactly backwards for "stop after the
  work". `H3LocalLLMStop` therefore has an `after` wildcard input;
  wire any upstream output into it to sequence the stop.
- **Attach path**: `attach()` falls back to `/v1/models` when the server
  has no root `/health` (LM Studio, vLLM), captures the served alias
  (used as the request model id) and the `multimodal` capability (vision
  preflight for attached servers). Alias comparison normalizes
  path/extension so hand-started servers match.

## llama.cpp build on this box (zero sudo)

There are NO official Linux CUDA prebuilts (checked 2026-08-25) and no
`nvcc`/`cmake` on the system. The working recipe pulls the whole CUDA 13
toolchain from pip wheels (the modern UNSUFFIXED packages — the `-cu13`
names are deprecation tombstones):

```bash
python3 -m venv ~/llamacpp-toolchain/venv
~/llamacpp-toolchain/venv/bin/pip install cmake ninja \
    nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cublas nvidia-cuda-cccl
# wheels unpack into ONE unified tree: site-packages/nvidia/cu13/{bin,include,lib,nvvm}
# symlink-farm it to ~/llamacpp-toolchain/cuda, then add:
#   - unversioned lib symlinks (libcudart.so -> libcudart.so.13, ...);
#     wheels ship only versioned names and FindCUDAToolkit needs the dev names
#   - lib64 -> lib, and lib64/stubs/libcuda.so -> /usr/lib/wsl/lib/libcuda.so.1
cd ~/llama.cpp && git checkout b10630   # >= b10450 required for qwen35
CUDACXX=~/llamacpp-toolchain/cuda/bin/nvcc cmake -B build -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120a-real" \
  -DCUDAToolkit_ROOT=~/llamacpp-toolchain/cuda -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_RPATH=~/llamacpp-toolchain/cuda/lib64 \
  -DCMAKE_INSTALL_RPATH=~/llamacpp-toolchain/cuda/lib64 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

The rpath makes the binary find the wheel-provided libcudart/libcublas
without any environment setup. Binary: `~/llama.cpp/build/bin/llama-server`.

## Model facts (verified on this machine)

- GGUF header arch: `qwen35` (48 Gated-DeltaNet + 16 full-attention
  layers). KV cache is small (~2 GiB @ 32k) because only the 16
  full-attention layers hold KV. n_ctx_train 262144.
- `--jinja` is REQUIRED; without it the template breaks. Thinking output
  goes to `reasoning_content` (the content field stays clean).
  `reasoning_effort` rides `chat_template_kwargs` and the template
  accepts ONLY `low` / `medium` / `xhigh` (xhigh is the template
  default; `none` and `high` raise a Jinja exception — verified live).
  There is no off switch; `low` still spends ~1-1.5k thinking tokens.
- Sampling defaults baked into `build_command`: temp 0.7, top-p 0.80,
  top-k 20, min-p 0, presence 1.5 (Unsloth instruct numbers; the
  GGUF-embedded defaults are thinking-mode and are overridden).
- Measured full-offload (-ngl 99, -c 32768): ~25 tok/s generation,
  ~20 GiB VRAM, health OK in ~8 s warm / ~22 s cold.
- Vision through `--mmproj` works on sm_120 (verified). If it ever
  misbehaves, pass `--no-mmproj-offload` via `extra_args`.
- Files are the PRE-V3.0 Unsloth upload (Aug 14/15). Unsloth re-uploaded
  "UD V3.0" quants on Aug 19 with claimed accuracy gains — an optional
  re-download of `Qwen3.8-27B-UD-Q4_K_XL.gguf`.

## Gotchas

- **ComfyUI's `/free` does not release custom-node caches.** With
  `--highvram` and module-level model caches (VRGDG, hybrid-tail, etc.)
  most of the 79 GiB stayed resident until the queue worker cycled.
  `free_vram_first` calls `unload_all_models()` but cannot reach those
  caches either — the honest fix is stopping the offending nodes or
  restarting ComfyUI.
- Port 8735 chosen clear of 8188 (ComfyUI), 8642 (Hermes), 8080, 11434.
- Server log: `/tmp/trentnodes-llamacpp-<port>.log` — every spawn error
  includes its tail.
- `pkill -f llama-server` from a script whose own command line contains
  the pattern kills the script; use `H3LocalLLMStop` or `kill <pid>`.
- Model must live on native ext4 (`models/LLM`); a `/mnt/*` path logs a
  drvfs warning (cold loads there run minutes, not seconds).

## Tests

```bash
cd /home/trent/ComfyUI
venv/bin/python custom_nodes/TrentNodes/tests/test_llamacpp_server.py
venv/bin/python custom_nodes/TrentNodes/tests/test_h3_skill_checklist.py
venv/bin/python custom_nodes/TrentNodes/tests/test_h3_skill_promptor.py
# E2E against a live server:
venv/bin/python custom_nodes/TrentNodes/tools/h3_skill_dev_run.py \
  --mode ref2va --brief "..." --duration 6 --image input/some_ref.png
```

The checklist PASS fixtures are MiniMax's own worked examples — if the
validator rejects an official example, fix the validator, not the example.
