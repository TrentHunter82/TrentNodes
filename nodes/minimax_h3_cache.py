"""
Trent MiniMax H3 Cache - TeaCache-style step caching for MiniMax H3.

Ported from lihaoyun6/ComfyUI-MiniMaxH3-Cache (GPL-3.0), with fixes:

- Audio slope fix: upstream's copied _forward always applies the
  d(sigma_a)/d(sigma_v) scaling to the audio velocity. On ComfyUI branches
  where forward() already applies it (Kijai PR #15243 / SDE-fix), audio got
  scaled twice. This port inspects the installed core and only scales when
  the core does not.
- The core class patch is applied lazily on first node execution instead of
  at import time, so merely having the pack installed never alters MiniMax
  behavior.
- Removed unreachable duplicated code; renamed resuse_threshold ->
  reuse_threshold.

The patched _forward is a copy of comfy.ldm.minimax.model.MiniMaxH3Model._forward
with the 50-block loop factored into _run_blocks behind a ("block_loop", 0)
patches_replace hook. If core _forward changes upstream, re-sync this copy.
Last synced 2026-08-07 against drozbay per-token latent noise mask core
(PR #15375, commit 3c474997): denoise_mask / audio_denoise_mask params and
the per-row timestep pinning they drive. On cores without mask_row_targets
the masks are ignored, matching those cores' own behavior.

Do not install alongside the original ComfyUI-MiniMaxH3-Cache extension:
both replace MiniMaxH3Model._forward and the import order decides which wins.
"""
import inspect

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension

try:
    import comfy.model_prefetch as model_prefetch
except ImportError:
    model_prefetch = None

try:
    import comfy.ldm.minimax.model as mm
except ImportError:
    mm = None


# =====================================================================
# Core patch: add a ("block_loop", 0) hook to MiniMaxH3Model._forward
# =====================================================================

def _apply_core_patch():
    """Patch MiniMaxH3Model with _run_blocks and a block_loop-aware _forward.

    Idempotent; raises RuntimeError when the core layout is unsupported.
    """
    if mm is None or model_prefetch is None:
        raise RuntimeError(
            "[TrentMiniMaxH3Cache] comfy.ldm.minimax.model or comfy.model_prefetch "
            "not found - ComfyUI too old for this node.")

    model_cls = getattr(mm, "MiniMaxH3Model", None)
    if model_cls is None:
        raise RuntimeError(
            '[TrentMiniMaxH3Cache] "MiniMaxH3Model" not found in comfy.ldm.minimax.model.')

    if getattr(model_cls, "_trent_minimax_cache_patch", False):
        return

    # Newer cores (Kijai PR #15243 / SDE fix) handle the audio-schedule
    # conversion in forward() — first via a time_shift_slope velocity scale,
    # then the "audio_stream_scale" carried-variable draft, now the
    # ModelSamplingAV "audio_scale" payload scheme (2026-08-06 PR head).
    # Older cores expect _forward to apply the slope itself. Scaling in both
    # places breaks audio, so detect which core this is.
    forward_src = inspect.getsource(model_cls.forward)
    core_scales_audio = ("time_shift_slope" in forward_src
                         or "audio_stream_scale" in forward_src
                         or "audio_scale" in forward_src)

    def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        end = len(self.blocks) if end is None else end
        prefetch_queue = model_prefetch.make_prefetch_queue(list(self.blocks[start:end]), h.device, transformer_options)
        for i in range(start, end):
            block = self.blocks[i]
            model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"])}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, None)
        return h

    def patched_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None,
                        denoise_mask=None, audio_denoise_mask=None, **kwargs):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype  # compute dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]

        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = mm.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                     keyframes=payload.get("keyframes"),
                                     refs=payload.get("refs"),
                                     frame_count=payload.get("frame_count"))

        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - mm.time_shift_sigma(sigma_v, shift_v, shift_a))

        vis_aug = float(payload.get("visual_cond_noise_aug", mm.VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", mm.AUDIO_COND_TIMESTEP))
        seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "ref_audio": max(t_a, aud_aug)}

        # rows that are preserved by the noise mask run at the cond timestep
        t_pin_v = max(t_v, mm.VISUAL_COND_TIMESTEP)
        t_pin_a = max(t_a, mm.AUDIO_COND_TIMESTEP)
        video_w = None
        audio_w = None
        mask_row_targets = getattr(mm, "mask_row_targets", None)
        if denoise_mask is not None and mask_row_targets is not None:
            targets = mask_row_targets(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
            if targets is not None:
                if bool(targets.any()):
                    video_w = targets.to(torch.float32).unsqueeze(1)  # [n, 1], 1 = generate
                else:
                    seg_t["video"] = t_pin_v
        if audio_denoise_mask is not None and mask_row_targets is not None:
            targets = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1) >= 0.5
            if not bool(targets.all()):
                if bool(targets.any()):
                    audio_w = targets.to(torch.float32).unsqueeze(1)
                else:
                    seg_t["audio"] = t_pin_a

        unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                          | ({t_pin_v} if video_w is not None else set())
                          | ({t_pin_a} if audio_w is not None else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            elif kind == "video" and video_w is not None:
                mod_segments.append((a, b, (row_base + seg_tag[kind], t_row[t_pin_v] * 3 + seg_tag[kind], video_w)))
            elif kind == "audio" and audio_w is not None:
                mod_segments.append((a, b, (row_base + seg_tag[kind], t_row[t_pin_a] * 3 + seg_tag[kind], audio_w)))
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = mm.patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = mm.pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states),
                                             transformer_options=transformer_options)

        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # ref_audio / audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        rope_freqs = mm.rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

        # --- the reason this copy exists: a hookable block loop ---
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        cache_ranges = [(a, b) for a, b, kind in layout.segments if kind in ("audio", "video")]
        if ("block_loop", 0) in blocks_replace:
            def block_loop_wrap(args):
                return {"img": self._run_blocks(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                                args["transformer_options"], args.get("start", 0), args.get("end"))}
            h = blocks_replace[("block_loop", 0)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options, "cache_ranges": cache_ranges,
                 "block_count": len(self.blocks)},
                {"original_block": block_loop_wrap})["img"]
        else:
            h = self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options)
        # ----------------------------------------------------------

        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
        if video_w is not None:
            video_seg = (va, vb, (t_row[seg_t["video"]], t_row[t_pin_v], video_w))
        else:
            video_seg = (va, vb, t_row[seg_t["video"]])
        if audio_w is not None:
            audio_seg = (aa, ab, (t_row[seg_t["audio"]], t_row[t_pin_a], audio_w))
        else:
            audio_seg = (aa, ab, t_row[seg_t["audio"]])
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg)

        video_out = mm.unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = mm.unpack_audio(a)

        if core_scales_audio:
            # forward() converts the audio stream; return raw velocity
            return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
        # legacy core: _forward owns the d(sigma_a)/d(sigma_v) velocity scale.
        # time_shift_slope was removed from newer cores, so compute it inline.
        slope_fn = getattr(mm, "time_shift_slope", None)
        if slope_fn is not None:
            slope_a = slope_fn(sigma_v, shift_v, shift_a)
        else:
            base = sigma_v / (shift_v + sigma_v * (1.0 - shift_v))
            slope_a = (shift_a * (1.0 + (shift_v - 1.0) * base) ** 2) / (shift_v * (1.0 + (shift_a - 1.0) * base) ** 2)
        slope_a = slope_a.to(audio_out.dtype)
        return [-video_out.to(video_x.dtype), (-slope_a) * audio_out.to(audio_x.dtype)]

    model_cls._run_blocks = _run_blocks
    model_cls._forward = patched_forward
    model_cls._trent_minimax_cache_patch = True
    print('[TrentMiniMaxH3Cache] Patched MiniMaxH3Model._forward '
          f'(core scales audio in forward: {core_scales_audio})')


# =====================================================================
# Cache: skip block-loop runs when hidden-state drift stays below threshold
# =====================================================================

class _MiniMaxH3CacheState:
    def __init__(self, reuse_threshold=0.1, start_percent=0.0, end_percent=1.0,
                 max_steps=2, device="auto", verbose=False):
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_steps = max_steps
        self.device = device
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.cached_residual = None
        self.run_count = 0
        self.skip_count = 0
        self.consecutive_skips = 0
        self.step_counter = 0
        self.accumulated_rel_l1 = 0.0
        self.previous_feature_sig = None
        self.cached_layout_sig = None
        self.last_seen_timestep = None
        self.total_steps = 20

    def _signature(self, hidden_states, cache_ranges):
        """Energy fingerprint over the audio/video token ranges only, so text
        and reference tokens do not dilute the step-to-step drift measure."""
        signatures = []
        for start, end in cache_ranges:
            length = end - start
            if length > 0:
                stride = max(1, length // 100)
                sliced = torch.abs(hidden_states[start:end:stride, :64])
                signatures.append(sliced.mean(dim=-1))
        if not signatures:
            stride = max(1, hidden_states.shape[0] // 100)
            return torch.abs(hidden_states[::stride, :64]).mean(dim=-1).detach().clone()
        return torch.cat(signatures, dim=0).detach().clone()

    def _step_scalar(self, timestep):
        if isinstance(timestep, torch.Tensor):
            return timestep.flatten()[0].item()
        if isinstance(timestep, (int, float)):
            return float(timestep)
        return None

    def _store_residual(self, residual_tensor):
        residual = residual_tensor.detach().clone()
        try:
            if self.device == "cpu":
                residual = residual.to("cpu", non_blocking=True)
            self.cached_residual = residual
        except torch.cuda.OutOfMemoryError:
            self.cached_residual = residual.to("cpu", non_blocking=True)

    def _apply_residual(self, hidden_states):
        if self.cached_residual is None:
            return hidden_states
        residual = self.cached_residual
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        return hidden_states + residual

    def __call__(self, args, kwargs):
        original_block = kwargs.get("original_block")

        img = args["img"]
        timestep = args.get("t_emb") if args.get("t_emb") is not None else args.get("timestep")
        cache_ranges = args.get("cache_ranges") or []
        block_count = args.get("block_count")
        transformer_options = args.get("transformer_options", {})
        if timestep is None:
            timestep = transformer_options.get("timestep")

        # reset when the resolution, dtype, or device changes mid-session
        current_layout_sig = (img.shape, img.dtype, img.device, block_count)
        if self.cached_layout_sig != current_layout_sig:
            self.reset()
            self.cached_layout_sig = current_layout_sig

        t = self._step_scalar(timestep)
        if t is None:
            return original_block(args)

        # count a step only when the timestep changes, so CFG cond/uncond
        # passes within one step do not inflate the counter
        if self.last_seen_timestep != t:
            self.last_seen_timestep = t
            self.step_counter += 1

        progress = self.step_counter / max(1, self.total_steps)
        in_accelerate_range = (self.start_percent <= progress <= self.end_percent)

        if self.cached_residual is not None and self.previous_feature_sig is not None:
            current_feature_sig = self._signature(img.detach(), cache_ranges)

            diff = torch.abs(current_feature_sig - self.previous_feature_sig).mean().item()
            denom = torch.abs(self.previous_feature_sig).mean().item() + 1e-6
            self.accumulated_rel_l1 += diff / denom

            is_below_thresh = (self.accumulated_rel_l1 < self.reuse_threshold)
            is_under_mcs = (self.consecutive_skips < self.max_steps)

            if is_below_thresh and is_under_mcs and in_accelerate_range:
                self.skip_count += 1
                self.consecutive_skips += 1
                if self.verbose:
                    print(f"[TrentMiniMaxH3Cache] Step {self.step_counter} [SKIP] "
                          f"(rel_l1: {self.accumulated_rel_l1:.4f} < thresh: {self.reuse_threshold:.4f})")
                img_output = self._apply_residual(img)
                args["img"] = img_output
                return {"img": img_output}

            run_reason = []
            if not is_below_thresh:
                run_reason.append(f"rel_l1({self.accumulated_rel_l1:.4f}) >= thresh({self.reuse_threshold:.4f})")
            if not is_under_mcs:
                run_reason.append(f"reached max_steps({self.max_steps})")
            if not in_accelerate_range:
                run_reason.append(f"progress({progress * 100:.0f}%) outside "
                                  f"[{self.start_percent * 100:.0f}%, {self.end_percent * 100:.0f}%]")
        else:
            run_reason = ["initial step (no cache)"]

        if self.verbose:
            print(f"[TrentMiniMaxH3Cache] Step {self.step_counter} [RUN] - {', '.join(run_reason)}")

        self.run_count += 1
        self.consecutive_skips = 0
        self.cached_residual = None
        self.previous_feature_sig = self._signature(img.detach(), cache_ranges)

        start_img = img.clone()
        res = original_block(args)
        output_img = res["img"] if isinstance(res, dict) else res
        self._store_residual(output_img - start_img)
        self.accumulated_rel_l1 = 0.0
        return res

    def finish(self):
        total = self.run_count + self.skip_count
        if total > 0:
            ran = max(1, total - self.skip_count)
            print(f"[TrentMiniMaxH3Cache] Skipped {self.skip_count}/{total} steps "
                  f"({total / ran:.2f}x speedup).")
        self.reset()


class _SamplingScope:
    """OUTER_SAMPLE wrapper: grabs the sigma schedule for total_steps and
    resets/reports the cache around each sampling run."""

    def __init__(self, cache):
        self.cache = cache

    def __call__(self, sample_fn, *args, **kwargs):
        self.cache.reset()

        sigmas = kwargs.get("sigmas")
        if sigmas is None:
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.ndim == 1 and len(arg) > 1:
                    sigmas = arg
                    break

        if sigmas is not None:
            self.cache.total_steps = max(1, len(sigmas) - 1)
        else:
            print("[TrentMiniMaxH3Cache] Warning: could not find sigmas; "
                  "assuming 20 steps for start/end percent.")

        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


# =====================================================================
# Node
# =====================================================================

class TrentMiniMaxH3Cache:
    """TeaCache-style residual caching for MiniMax H3 (video+audio DiT)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reuse_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.00, "max": 1.00, "step": 0.01,
                    "tooltip": "Skip a step when accumulated hidden-state drift stays below this. Higher = faster but lossier."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.15, "min": 0.00, "max": 1.00, "step": 0.01,
                    "tooltip": "Do not skip before this fraction of the sampling run."
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.90, "min": 0.00, "max": 1.00, "step": 0.01,
                    "tooltip": "Do not skip after this fraction of the sampling run."
                }),
                "max_steps": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Maximum consecutive skipped steps before forcing a real run."
                }),
                "device": (["auto", "cpu"], {
                    "default": "auto",
                    "tooltip": "Where to keep the cached residual. auto = GPU with CPU fallback on OOM."
                }),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "main"
    CATEGORY = "Trent/Optimization"
    DISPLAY_NAME = "Trent MiniMax H3 Cache"

    def main(self, model, reuse_threshold, start_percent, end_percent, max_steps, device, verbose):
        _apply_core_patch()

        m = model.clone()
        diffusion_model = m.model.diffusion_model
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model":
            raise ValueError(f"Model is {diffusion_model.__class__.__name__}, not MiniMaxH3Model.")

        cache = _MiniMaxH3CacheState(
            reuse_threshold=reuse_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            max_steps=max_steps,
            device=device,
            verbose=verbose,
        )
        scope = _SamplingScope(cache)

        m.set_model_patch_replace(cache, "dit", "block_loop", 0)
        m.add_wrapper(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, scope)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "TrentMiniMaxH3Cache": TrentMiniMaxH3Cache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentMiniMaxH3Cache": "Trent MiniMax H3 Cache",
}
