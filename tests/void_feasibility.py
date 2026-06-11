"""
VOID-on-stills feasibility: run ComfyUI core's VOID 2-pass inpainting on a
single frame via the replicate-x5 trick, saving fills for visual A/B against
LaMa. Standalone (no server). Run from the ComfyUI root:

    cd /home/trent/ComfyUI && venv/bin/python \
        custom_nodes/TrentNodes/tests/void_feasibility.py
"""

import os
import subprocess
import sys
import time

ROOT = "/home/trent/ComfyUI"
OUT_DIR = os.path.join(
    ROOT, "custom_nodes", "TrentNodes", "tests", "_void_feasibility"
)
FRAME = os.path.join(ROOT, "input", "ep014_drop_in_basket_front_first_frame.png")
W, H, T = 672, 384, 5
SEED = 43
STEPS = 30
CFG = 6.0

sys.path.insert(0, ROOT)

# Refuse to fight a busy GPU (the ComfyUI server may be rendering)
free_mb = int(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
).decode().split()[0])
if free_mb < 40_000:
    print(f"Only {free_mb} MB VRAM free - aborting to not disturb the server.")
    sys.exit(2)

from utils.extra_config import load_extra_path_config  # noqa: E402
if os.path.exists(os.path.join(ROOT, "extra_model_paths.yaml")):
    load_extra_path_config(os.path.join(ROOT, "extra_model_paths.yaml"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402
from comfy.sd import CLIPType  # noqa: E402
from comfy_extras.nodes_custom_sampler import (  # noqa: E402
    BasicScheduler, CFGGuider, RandomNoise, SamplerCustomAdvanced,
)
from comfy_extras.nodes_void import (  # noqa: E402
    OpticalFlowLoader, VOIDInpaintConditioning, VOIDSampler,
    VOIDWarpedNoise, VOIDWarpedNoiseSource,
)

os.makedirs(OUT_DIR, exist_ok=True)

def flush_vram():
    import gc
    gc.collect()
    import comfy.model_management as _mm
    _mm.unload_all_models()
    _mm.soft_empty_cache()
    torch.cuda.empty_cache()




def save(tensor_hwc, name):
    arr = (tensor_hwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT_DIR, name))
    print("  saved", name)


def unwrap(node_output):
    return node_output.args


t0 = time.time()
print("[1/6] Loading frame + building masks...")
img = Image.open(FRAME).convert("RGB").resize((W, H), Image.LANCZOS)
frame = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)

yy = torch.arange(H).view(-1, 1).float()
xx = torch.arange(W).view(1, -1).float()


def ellipse(cy, cx, ry, rx):
    return ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1).float()


# Mask A: red bowl region (fill should reconstruct gridded floor)
# Mask B: robot arm column (fill should reconstruct wall + floor)
masks = {
    "bowl": ellipse(H * 0.66, W * 0.335, H * 0.115, W * 0.085),
    "arm": (ellipse(H * 0.42, W * 0.515, H * 0.40, W * 0.045)).clamp(0, 1),
}

save(frame, "00_original.png")
for name, m in masks.items():
    save(frame * (1 - m.unsqueeze(-1)) + m.unsqueeze(-1) * 0.5, f"00_masked_{name}.png")

print(f"[2/6] Loading VOID models (slow /mnt/d mount, be patient)... t={time.time()-t0:.0f}s")
t5_path = folder_paths.get_full_path_or_raise("text_encoders", "t5xxl_fp16.safetensors")
clip = comfy.sd.load_clip(
    ckpt_paths=[t5_path],
    embedding_directory=folder_paths.get_folder_paths("embeddings"),
    clip_type=CLIPType.COGVIDEOX,
)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
    folder_paths.get_full_path_or_raise("vae", "cogvideox_vae.safetensors")
))
model1 = comfy.sd.load_diffusion_model(
    folder_paths.get_full_path_or_raise("diffusion_models", "void_pass1.safetensors")
)
model2 = comfy.sd.load_diffusion_model(
    folder_paths.get_full_path_or_raise("diffusion_models", "void_pass2.safetensors")
)
flow = unwrap(OpticalFlowLoader.execute("raft_large_C_T_SKHT_V2-ff5fadd5.safetensors"))[0]
print(f"    models loaded t={time.time()-t0:.0f}s")

torch.set_grad_enabled(False)

tokens = clip.tokenize("")
empty_cond = clip.encode_from_tokens_scheduled(tokens)

sampler = unwrap(VOIDSampler.execute())[0]

for name, mask in masks.items():
    print(f"[3/6] {name}: conditioning...  t={time.time()-t0:.0f}s")
    video = frame.unsqueeze(0).repeat(T, 1, 1, 1)        # (T, H, W, 3)
    quadmask = mask.unsqueeze(0).repeat(T, 1, 1)         # (T, H, W), 1=remove

    pos, neg, latent = unwrap(VOIDInpaintConditioning.execute(
        empty_cond, empty_cond, vae, video, quadmask, W, H, T, 1
    ))

    flush_vram()
    print(f"[4/6] {name}: pass 1 ({STEPS} steps)...")
    guider1 = CFGGuider().get_guider(model1, pos, neg, CFG)[0]
    sigmas = BasicScheduler().get_sigmas(model1, "simple", STEPS, 1.0)[0]
    noise1 = RandomNoise().get_noise(SEED)[0]
    out1, _ = SamplerCustomAdvanced().sample(noise1, guider1, sampler, sigmas, latent)
    video1 = vae.decode(out1["samples"])
    if video1.ndim == 5:
        video1 = video1.reshape(-1, *video1.shape[-3:])
    save(video1[T // 2], f"10_void_pass1_{name}.png")

    del guider1
    flush_vram()
    print(f"[5/6] {name}: pass 2 (warped noise + {STEPS} steps)... t={time.time()-t0:.0f}s")
    warped = unwrap(VOIDWarpedNoise.execute(flow, video1, W, H, T, 1))[0]
    noise2 = unwrap(VOIDWarpedNoiseSource.execute(warped))[0]
    guider2 = CFGGuider().get_guider(model2, pos, neg, CFG)[0]
    sigmas2 = BasicScheduler().get_sigmas(model2, "simple", STEPS, 1.0)[0]
    out2, _ = SamplerCustomAdvanced().sample(noise2, guider2, sampler, sigmas2, latent)
    video2 = vae.decode(out2["samples"])
    if video2.ndim == 5:
        video2 = video2.reshape(-1, *video2.shape[-3:])
    mid = video2[T // 2]
    save(mid, f"20_void_pass2_{name}.png")

    # Composite: only masked region from the fill (how the node would use it)
    m3 = mask.unsqueeze(-1).to(mid.device)
    save(frame.to(mid.device) * (1 - m3) + mid * m3, f"30_void_composite_{name}.png")
    del guider2
    flush_vram()

print(f"[6/6] done t={time.time()-t0:.0f}s -> {OUT_DIR}")
