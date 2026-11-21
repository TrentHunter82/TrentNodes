# Performance transfer (V2V) for OmniVoice.
# Ported from OmniVoice Manager `manager/perform.py` by Nathan (SanDiegoDude), Apache License 2.0
# (https://github.com/SanDiegoDude/OmniVoice-Manager). The OmniVoice model code in this folder is from
# k2-fsa/OmniVoice, Apache License 2.0 (see LICENSE).
#
# OmniVoice is a masked-diffusion model over an (8 codebooks x T) token grid, and its unmasking loop never
# overwrites a token that is already set. The performance take is encoded into that grid, codebook layer 0
# (timing / prosody) is pinned, and the rest is generated with the target voice's clone prompt.
#   character: pin all of layer 0, release it partway through the schedule (the target's own delivery takes over)
#   voice:     pin layer 0 at a token stride and never release (the source delivery in the target timbre)
import math

import numpy as np
import torch
import torchaudio

from .model import OmniVoiceGenerationConfig, _get_time_steps, _gumbel_sample

# UI strength -> knob value. 1..5 calibrated by ear by the original author; 6 is an extra, looser step.
CHARACTER_RELEASE = {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.25, 5: 0.10, 6: 0.05}
VOICE_STRIDE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 8}
TOKENS_PER_SECOND = 25


def normalize_active(wav, sr, target_rms=0.1):
    # Quiet takes seed near-silence tokens, so level the active (non-silent) frames first.
    n = int(sr * 0.05)
    k = len(wav) // n
    if k == 0:
        return wav
    rms = np.sqrt((wav[: k * n].reshape(k, n) ** 2).mean(axis=1))
    active = rms[rms > max(float(rms.max()) * 0.1, 1e-5)]
    if len(active) == 0:
        return wav
    return np.clip(wav * (target_rms / float(np.sqrt((active ** 2).mean()))), -1.0, 1.0).astype(np.float32)


def build_keep_mask(C, T, mode, strength):
    keep = torch.zeros(C, T, dtype=torch.bool)
    if mode == "voice":
        keep[0, :: VOICE_STRIDE[strength]] = True
        return keep, None
    keep[0] = True
    return keep, CHARACTER_RELEASE[strength]


def _generate_seeded(model, text, vc_prompt, seed_tokens, keep_mask, cfg, language, release_at):
    # Mirror of OmniVoice._generate_iterative (batch 1 + CFG pair) with a pre-seeded target grid.
    dev = model.device
    C = model.config.num_audio_codebook
    MASK = model.config.audio_mask_id
    T = int(seed_tokens.shape[1])

    inputs = model._prepare_inference_inputs(
        text=text, num_target_tokens=T, ref_text=vc_prompt.ref_text, ref_audio_tokens=vc_prompt.ref_audio_tokens,
        lang=language, instruct=None, denoise=cfg.denoise)
    input_ids = inputs["input_ids"].to(dev)
    audio_mask = inputs["audio_mask"].to(dev)
    c_len = input_ids.size(2)

    keep = keep_mask.to(dev)
    tokens = torch.where(keep, seed_tokens.to(dev), torch.full_like(seed_tokens.to(dev), MASK)).unsqueeze(0)

    bi = torch.full((2, C, c_len), MASK, dtype=torch.long, device=dev)
    bm = torch.zeros((2, c_len), dtype=torch.bool, device=dev)
    ba = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=dev)
    bi[0] = input_ids[0]
    bm[0] = audio_mask[0]
    ba[0] = True
    bi[1, :, :T] = input_ids[0, :, -T:]
    bm[1, :T] = audio_mask[0, -T:]
    ba[1, :, :T, :T] = True
    if c_len > T:
        diag = torch.arange(T, c_len, device=dev)
        ba[1, :, diag, diag] = True
    bi[0, :, c_len - T:] = tokens[0]
    bi[1, :, :T] = tokens[0]

    total_mask = int((tokens[0] == MASK).sum())
    if total_mask == 0:
        return tokens[0]
    ts = _get_time_steps(0.0, 1.0, cfg.num_step, cfg.t_shift).tolist()
    sched, rem = [], total_mask
    for step in range(cfg.num_step):
        num = rem if step == cfg.num_step - 1 else min(math.ceil(total_mask * (ts[step + 1] - ts[step])), rem)
        sched.append(num)
        rem -= num

    release_step = None
    if release_at is not None:
        release_step = max(1, min(cfg.num_step - 1, round(cfg.num_step * release_at)))

    layer_ids = torch.arange(C, device=dev).view(1, -1, 1)
    sample = tokens[0:1]
    for step in range(cfg.num_step):
        if step == release_step:
            extra = int(keep.sum())
            sample.masked_fill_(keep.unsqueeze(0), MASK)
            bi[0:1, :, c_len - T:] = sample
            bi[1:2, :, :T] = sample
            rem_steps = cfg.num_step - step
            for j in range(step, cfg.num_step):
                sched[j] += extra // rem_steps
            sched[-1] += extra % rem_steps
        k = sched[step]
        if k <= 0:
            continue
        logits = model(input_ids=bi, audio_mask=bm, attention_mask=ba).logits.to(torch.float32)
        pred, scores = model._predict_tokens_with_scoring(logits[0:1, :, c_len - T:, :], logits[1:2, :, :T, :], cfg)
        scores = scores - layer_ids * cfg.layer_penalty_factor
        if cfg.position_temperature > 0.0:
            scores = _gumbel_sample(scores, cfg.position_temperature)
        scores.masked_fill_(sample != MASK, -float("inf"))
        topk = torch.topk(scores.flatten(), k).indices
        flat = sample.flatten()
        flat[topk] = pred.flatten()[topk]
        sample.copy_(flat.view_as(sample))
        bi[0:1, :, c_len - T:] = sample
        bi[1:2, :, :T] = sample
    return sample[0]


def perform_transfer(model, text, vc_prompt, perf_wav, perf_sr, mode, strength, seed, language, num_step, guidance_scale):
    """Render `text` in the vc_prompt voice, riding the performance. Output length == performance length."""
    sr = model.sampling_rate
    wav = torch.from_numpy(np.ascontiguousarray(perf_wav, dtype=np.float32))
    if perf_sr != sr:
        wav = torchaudio.functional.resample(wav, perf_sr, sr)
    wav = normalize_active(wav.numpy(), sr)

    hop = sr // TOKENS_PER_SECOND
    length = len(wav)
    wav = wav[: length - length % hop]
    codes = model.audio_tokenizer.encode(torch.from_numpy(wav).view(1, 1, -1).to(model.audio_tokenizer.device))
    tokens = codes.audio_codes.squeeze(0).long()
    keep, release_at = build_keep_mask(tokens.shape[0], tokens.shape[1], mode, strength)

    cfg = OmniVoiceGenerationConfig(num_step=num_step, guidance_scale=guidance_scale)
    torch.manual_seed(seed % (2 ** 31))
    out = _generate_seeded(model, text, vc_prompt, tokens, keep, cfg, language, release_at)
    # Decode without OmniVoice's post-process: its 0.1 s edge pad would shift the line late. The timing is the product.
    audio = model.audio_tokenizer.decode(out.to(model.audio_tokenizer.device).unsqueeze(0)).audio_values[0, 0].float().cpu().numpy()
    if vc_prompt.ref_rms < 0.1:
        audio = audio * vc_prompt.ref_rms / 0.1
    return np.pad(audio, (0, max(0, length - len(audio))))[:length]
