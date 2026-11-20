# Inference-only port of VoxMel (https://github.com/VoxDance-Lab/VoxMel, Apache-2.0, see LICENSE in this folder)
# with the Vocos decoder (https://github.com/gemelo-ai/vocos, MIT).
import math

import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model

import comfy.model_management
import comfy.utils


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sinusoidal_pos_emb(length, dim, device, dtype):
    half = dim // 2
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half)
    args = pos * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)[None]


class FlowMatchingHead(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden, depth, time_embed_dim=128):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.in_proj = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(depth)
        ])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, data_dim))

    def sample(self, cond, steps, generator, cfg_text=0.0, cfg_spk=0.0):
        # cond: [branches, N, cond_dim], branches ordered (full, text dropped, text and speaker dropped)
        x = torch.randn(cond.shape[1], self.data_dim, device=cond.device, dtype=cond.dtype, generator=generator)
        cond_hidden = self.cond_proj(cond)
        t = torch.arange(steps, device=cond.device, dtype=torch.float32) / steps
        time_hidden = self.time_mlp(timestep_embedding(t, self.time_embed_dim).to(cond.dtype))
        for i in range(steps):
            h = self.in_proj(x) + cond_hidden + time_hidden[i]
            for blk in self.blocks:
                h = h + blk(h)
            v = self.out(h)
            if v.shape[0] == 3:
                v = v[2] + cfg_text * (v[0] - v[1]) + cfg_spk * (v[1] - v[2])
            elif v.shape[0] == 2:
                v = v[1] + cfg_text * (v[0] - v[1])
            else:
                v = v[0]
            x = x.add(v, alpha=1.0 / steps)
        return x


class BidirAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, attn_mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
        return x + self.mlp(self.norm2(x))


class UpsampleTier(nn.Module):
    def __init__(self, n_mels, dim, num_heads, n_layers, flow_hidden, flow_depth, detail_norm, attn_window):
        super().__init__()
        self.dim = dim
        self.attn_window = attn_window
        self.detail_norm = detail_norm
        self.in_proj = nn.Linear(n_mels, dim)
        self.blocks = nn.ModuleList([BidirAttention(dim, num_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.flow = FlowMatchingHead(n_mels, dim, flow_hidden, flow_depth)
        self.register_buffer("detail_mean", torch.empty(n_mels))
        self.register_buffer("detail_var", torch.empty(n_mels))

    def sample(self, x, steps, generator):
        length = x.shape[1]
        h = self.in_proj(x) + sinusoidal_pos_emb(length, self.dim, x.device, x.dtype)
        mask = None
        if self.attn_window >= 0:
            idx = torch.arange(length, device=x.device)
            mask = (idx[None, :] - idx[:, None]).abs() > self.attn_window
        for blk in self.blocks:
            h = blk(h, mask)
        y = self.flow.sample(self.norm(h), steps, generator)
        if self.detail_norm:
            y = y * self.detail_var.clamp_min(1e-8).sqrt() + self.detail_mean
        return y.unsqueeze(0)


def haar_split(x, axis):
    if axis == "t":
        e, o = x[:, 0::2], x[:, 1::2]
    else:
        e, o = x[..., 0::2], x[..., 1::2]
    return (e + o) / 2, (e - o) / 2


def haar_merge(avg, diff, axis):
    if axis == "t":
        return torch.stack([avg + diff, avg - diff], dim=2).flatten(1, 2)
    return torch.stack([avg + diff, avg - diff], dim=-1).flatten(-2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.empty(dim))

    def forward(self, x):
        h = self.dwconv(x).transpose(1, 2)
        h = self.pwconv2(self.act(self.pwconv1(self.norm(h))))
        return x + (self.gamma * h).transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, input_channels, dim, intermediate_dim, num_layers):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([ConvNeXtBlock(dim, intermediate_dim) for _ in range(num_layers)])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class ISTFTHead(nn.Module):
    def __init__(self, dim, n_fft, hop_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.empty(n_fft))

    def forward(self, x):
        mag, phase = self.out(x).transpose(1, 2).chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        return torch.istft(spec, self.n_fft, self.hop_length, self.n_fft, self.window, center=True)


class Vocos(nn.Module):
    def __init__(self, n_mels, dim, intermediate_dim, num_layers, n_fft, hop_length):
        super().__init__()
        self.backbone = VocosBackbone(n_mels, dim, intermediate_dim, num_layers)
        self.head = ISTFTHead(dim, n_fft, hop_length)

    def forward(self, mel):
        return self.head(self.backbone(mel))


class VoxMel(nn.Module):
    def __init__(self, n_mels, dim, num_heads, n_layers, split_axes, flow_hidden, flow_depth, detail_norm,
                 tier_attn_window, sample_rate, hop_length, n_fft, llm, vocoder):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.split_axes = split_axes
        self.coarse_fps = sample_rate / hop_length / 2 ** split_axes.count("t")

        tier_mels = []
        coarse_mels = n_mels
        for axis in reversed(split_axes):
            if axis == "f":
                coarse_mels //= 2
            tier_mels.insert(0, coarse_mels)

        llm_dim = llm["hidden_size"]
        self.special_embedding = nn.Embedding(50, llm_dim)
        self.null_text = nn.Parameter(torch.empty(1, llm_dim))
        self.coarse_frame_proj = nn.Linear(coarse_mels, llm_dim)
        self.llm = Qwen2Model(Qwen2Config(**llm))
        self.coarse_flow = FlowMatchingHead(coarse_mels, llm_dim, flow_hidden, flow_depth)
        self.stop_token_head = nn.Linear(llm_dim, 1, bias=False)
        self.up_tiers = nn.ModuleList([
            UpsampleTier(tier_mels[i], dim, num_heads, n_layers[i + 1], flow_hidden, flow_depth, detail_norm, tier_attn_window[i])
            for i in range(len(split_axes))
        ])
        self.register_buffer("mel_mean", torch.empty(n_mels))
        self.register_buffer("mel_std", torch.empty(n_mels))
        self.vocoder = Vocos(n_mels, hop_length=hop_length, n_fft=n_fft, **vocoder)

    def log_mel(self, wav):
        # wav: [1, samples] at self.sample_rate. Matches torchaudio MelSpectrogram(power=1, norm=None, htk).
        window = torch.hann_window(self.n_fft, device=wav.device, dtype=wav.dtype)
        spec = torch.stft(wav, self.n_fft, self.hop_length, window=window, center=True, pad_mode="reflect", return_complex=True)
        mag = torch.view_as_real(spec).norm(dim=-1).transpose(1, 2)
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_fft // 2 + 1, device=wav.device, dtype=wav.dtype)
        m_max = 2595.0 * math.log10(1.0 + (self.sample_rate / 2) / 700.0)
        f_pts = 700.0 * (10 ** (torch.linspace(0, m_max, self.n_mels + 2, device=wav.device, dtype=wav.dtype) / 2595.0) - 1.0)
        slopes = f_pts[None, :] - all_freqs[:, None]
        f_diff = f_pts[1:] - f_pts[:-1]
        fb = torch.minimum(-slopes[:, :-2] / f_diff[:-1], slopes[:, 2:] / f_diff[1:]).clamp_min(0)
        return (mag @ fb).clamp_min(1e-5).log()

    def generate(self, text_tokens, generator, steps, up_steps, cfg_text, cfg_spk, min_frames, max_frames, prompt_wav=None):
        text_emb = self.llm.embed_tokens(text_tokens)
        sos = self.special_embedding.weight[0].reshape(1, 1, -1)
        task = self.special_embedding.weight[1].reshape(1, 1, -1)
        null_text = self.null_text.reshape(1, 1, -1).expand(1, text_emb.shape[1], -1)

        if prompt_wav is None:
            primes = [torch.cat([sos, text_emb, task], dim=1)]
            if cfg_text != 0:
                primes.append(torch.cat([sos, null_text, task], dim=1))
            prompt_known = [None] * len(self.split_axes)
        else:
            mel = self.log_mel(prompt_wav)
            grid = 2 ** self.split_axes.count("t")
            mel = (mel[:, :mel.shape[1] // grid * grid] - self.mel_mean) / self.mel_std
            prompt_known = []
            for axis in reversed(self.split_axes):
                mel = haar_split(mel, axis)[0]
                prompt_known.insert(0, mel)
            prompt_emb = self.coarse_frame_proj(mel)
            primes = [torch.cat([sos, text_emb, task, prompt_emb], dim=1)]
            if cfg_text != 0 or cfg_spk != 0:
                primes.append(torch.cat([sos, null_text, task, prompt_emb], dim=1))
            if cfg_spk != 0:
                primes.append(torch.cat([sos, null_text, task, torch.zeros_like(prompt_emb)], dim=1))

        cur = torch.cat(primes)
        cache = None
        frames = []
        pbar = comfy.utils.ProgressBar(max_frames)
        for i in range(max_frames):
            comfy.model_management.throw_exception_if_processing_interrupted()
            out = self.llm(inputs_embeds=cur, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last = out.last_hidden_state[:, -1]
            if i > min_frames and self.stop_token_head(last[0]).item() > 0:
                break
            frame = self.coarse_flow.sample(last.unsqueeze(1), steps, generator, cfg_text, cfg_spk)
            frames.append(frame)
            cur = self.coarse_frame_proj(frame).unsqueeze(0).expand(len(primes), -1, -1)
            pbar.update(1)

        x = torch.stack(frames, dim=1)
        for k, axis in enumerate(self.split_axes):
            known = x if prompt_known[k] is None else torch.cat([prompt_known[k], x], dim=1)
            detail = self.up_tiers[k].sample(known, up_steps, generator)[:, -x.shape[1]:]
            x = haar_merge(x, detail, axis)

        mel = x * self.mel_std + self.mel_mean
        return self.vocoder(mel.transpose(1, 2))
