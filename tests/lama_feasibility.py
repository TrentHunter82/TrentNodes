"""
LaMa side of the inpaint A/B: run big-lama (TorchScript) on the same
frame + masks as void_feasibility.py. CPU or GPU. Run from ComfyUI root:

    cd /home/trent/ComfyUI && venv/bin/python \
        custom_nodes/TrentNodes/tests/lama_feasibility.py
"""

import os
import sys
import time

ROOT = "/home/trent/ComfyUI"
OUT_DIR = os.path.join(
    ROOT, "custom_nodes", "TrentNodes", "tests", "_void_feasibility"
)
FRAME = os.path.join(ROOT, "input", "ep014_drop_in_basket_front_first_frame.png")
MODEL = os.path.join(ROOT, "models", "inpaint", "big-lama.pt")
W, H = 768, 432

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = Image.open(FRAME).convert("RGB").resize((W, H), Image.LANCZOS)
frame = torch.from_numpy(np.array(img)).float() / 255.0

yy = torch.arange(H).view(-1, 1).float()
xx = torch.arange(W).view(1, -1).float()


def ellipse(cy, cx, ry, rx):
    return ((((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1).float()


masks = {
    "bowl": ellipse(H * 0.66, W * 0.335, H * 0.115, W * 0.085),
    "arm": ellipse(H * 0.42, W * 0.515, H * 0.40, W * 0.045).clamp(0, 1),
}

print("loading big-lama TorchScript...")
model = torch.jit.load(MODEL, map_location=device).eval()

for name, mask in masks.items():
    t0 = time.time()
    # LaMa convention: image (1,3,H,W) in [0,1], mask (1,1,H,W) binary,
    # dims divisible by 8 (768x432 ok)
    im = frame.permute(2, 0, 1).unsqueeze(0).to(device)
    m = (mask > 0.5).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(im, m)  # (1,3,H,W) in [0,1]
    fill = out[0].permute(1, 2, 0).clamp(0, 1).cpu()
    dt = time.time() - t0
    comp = frame * (1 - mask.unsqueeze(-1)) + fill * mask.unsqueeze(-1)
    arr = (comp.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT_DIR, f"40_lama_composite_{name}.png"))
    print(f"  {name}: {dt:.2f}s on {device} -> 40_lama_composite_{name}.png")
print("done")
