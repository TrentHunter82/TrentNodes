"""
Deep Denoise node for ComfyUI.

Neural denoising for single images and frame batches, built on the
inference recipe the NTIRE 2026 denoising challenge leaders share:
a strong restoration backbone, model ensembling, flip/rotate
self-ensemble and overlapped feathered tiles.

Models (KAIR release, Apache-2.0, loaded through spandrel):
- SCUNet real  : blind, trained on real camera / compression noise
- DRUNet       : non-blind, takes a noise level, so strength is tunable
"""

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from spandrel import ModelLoader

import comfy.model_management as mm
import comfy.model_patcher
import comfy.storage
import comfy.utils
import folder_paths
from comfy_extras.nodes_void import OpticalFlowLoader
from comfy_extras.void_noise_warp import RaftOpticalFlow

from ..utils import denoise
from ..utils.model_cache import _robust_download

KAIR_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/"

SCUNET_CLEAN = ("scunet_color_real_psnr.pth", 71982841)
SCUNET_DETAIL = ("scunet_color_real_gan.pth", 71982835)
DRUNET = ("drunet_color.pth", 130579305)

MODELS = {
    "SCUNet real (clean)": (SCUNET_CLEAN,),
    "SCUNet real (detail)": (SCUNET_DETAIL,),
    "DRUNet (tunable)": (DRUNET,),
    "SCUNet + DRUNet ensemble": (SCUNET_CLEAN, DRUNET),
}

RAFT_FILE = "raft_large_C_T_SKHT_V2-ff5fadd5.safetensors"
RAFT_URL = (
    "https://huggingface.co/Comfy-Org/void-model/resolve/main/"
    "optical_flow/" + RAFT_FILE
)
RAFT_SIZE = 21060840
# Longest side RAFT sees. Flow is smooth, so a reduced-size estimate
# scaled back up is accurate enough and far cheaper.
FLOW_SIDE = 768

TTA_FOLDS = {"off": 1, "x2": 2, "x4": 4, "x8": 8}

_loaded = {}


def load_denoiser(filename: str, size: int):
    """Load a KAIR denoiser through spandrel (cached), downloading once."""
    if filename in _loaded:
        return _loaded[filename]

    path = folder_paths.get_full_path("upscale_models", filename)
    if path is None:
        path = os.path.join(
            folder_paths.get_folder_paths("upscale_models")[0], filename
        )
        print(f"[TrentNodes] Downloading {filename} to {path} ...")
        _robust_download(KAIR_URL + filename, path, size)

    sd = comfy.utils.load_torch_file(path, safe_load=True)
    model = ModelLoader().load_from_state_dict(sd).eval()
    model.patcher = comfy.model_patcher.CoreModelPatcher(
        model.model,
        load_device=mm.get_torch_device(),
        offload_device=mm.unet_offload_device(),
        fast_disk=comfy.storage.state_dict_fast_disk(sd),
    )
    _loaded[filename] = model
    return model


def load_flow_model():
    """Load RAFT-large through the core loader (cached), downloading once."""
    if RAFT_FILE in _loaded:
        return _loaded[RAFT_FILE]

    if folder_paths.get_full_path("optical_flow", RAFT_FILE) is None:
        path = os.path.join(
            folder_paths.get_folder_paths("optical_flow")[0], RAFT_FILE
        )
        print(f"[TrentNodes] Downloading {RAFT_FILE} to {path} ...")
        _robust_download(RAFT_URL, path, RAFT_SIZE)

    patcher = OpticalFlowLoader.execute(RAFT_FILE).args[0]
    _loaded[RAFT_FILE] = patcher
    return patcher


def make_flow_function(raft: RaftOpticalFlow):
    """Flow between two (1, 3, H, W) frames, estimated at reduced size."""
    def flow_function(center, neighbour):
        h, w = center.shape[2:]
        scale = min(1.0, FLOW_SIDE / max(h, w))
        if scale < 1.0:
            size = (round(h * scale), round(w * scale))
            center = F.interpolate(center, size, mode="area")
            neighbour = F.interpolate(neighbour, size, mode="area")
        flow = raft(center[0], neighbour[0])
        if scale < 1.0:
            flow = F.interpolate(
                flow[None], (h, w), mode="bilinear", align_corners=False
            )[0]
            flow[0] *= w / size[1]
            flow[1] *= h / size[0]
        return flow
    return flow_function


def run_drunet(model, image: torch.Tensor, sigma: float) -> torch.Tensor:
    """DRUNet takes the noise level as a fourth, constant input channel."""
    padded, h, w = denoise.pad_to_multiple(image, 8)
    noise_map = torch.full_like(padded[:, :1], sigma)
    out = model.model(torch.cat([padded, noise_map], dim=1))
    return out[:, :, :h, :w]


class DeepDenoise:
    """Neural denoiser for images and frame batches."""

    CATEGORY = "Trent/Image"
    DISPLAY_NAME = "Deep Denoise"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "noise")
    OUTPUT_TOOLTIPS = (
        "Denoised images.",
        "The noise that was removed, centred on mid grey. Blend it "
        "back (or feed it to a grain node) to re-grain.",
    )
    FUNCTION = "denoise"
    DESCRIPTION = (
        "Neural denoising for images and frame batches. SCUNet handles "
        "real camera / compression noise blind; DRUNet gives a tunable "
        "noise level. Optional self-ensemble and temporal smoothing."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image or batch of frames to denoise.",
                }),
                "model": (list(MODELS), {
                    "default": "SCUNet real (clean)",
                    "tooltip": (
                        "clean: smoothest, best for real photo / video "
                        "noise. detail: keeps more texture, can invent "
                        "some. DRUNet: Gaussian-style noise with a "
                        "noise_level you control. ensemble: averages "
                        "SCUNet clean and DRUNet. Models download once "
                        "to models/upscale_models."
                    ),
                }),
                "noise_level": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": (
                        "DRUNet only: noise sigma on a 0-255 scale. "
                        "0 = measure it from each image. Higher removes "
                        "more."
                    ),
                }),
                "luma_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "How much brightness noise to remove. Lower it "
                        "to keep some natural grain."
                    ),
                }),
                "chroma_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "How much colour noise to remove. Colour "
                        "speckle is rarely wanted, so this usually "
                        "stays at 1."
                    ),
                }),
                "self_ensemble": (list(TTA_FOLDS), {
                    "default": "off",
                    "tooltip": (
                        "Denoise flipped / rotated copies and average "
                        "them. Slightly cleaner and more accurate; x8 "
                        "takes 8 times as long."
                    ),
                }),
                "temporal_radius": ("INT", {
                    "default": 0, "min": 0, "max": 4,
                    "tooltip": (
                        "Video batches: fuse each frame with this many "
                        "neighbours on each side to remove shimmer. "
                        "Moving areas are left alone. 0 = treat every "
                        "image independently."
                    ),
                }),
                "motion_compensation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Align neighbour frames with optical flow "
                        "(RAFT) before fusing, so pans and moving "
                        "subjects are smoothed too. Off = only static "
                        "areas are fused (faster)."
                    ),
                }),
                "temporal_tolerance": ("FLOAT", {
                    "default": 0.015, "min": 0.005, "max": 0.25,
                    "step": 0.005,
                    "tooltip": (
                        "How different a neighbour frame's pixel may "
                        "be and still be fused. Raise for stronger "
                        "smoothing; too high ghosts motion."
                    ),
                }),
                "tile_size": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": (
                        "Images larger than this are processed in "
                        "overlapping tiles. Halves automatically on "
                        "out-of-memory."
                    ),
                }),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 0, "max": 256, "step": 8,
                    "tooltip": "Feathered overlap between tiles.",
                }),
            },
        }

    def denoise(
        self,
        images: torch.Tensor,
        model: str,
        noise_level: float,
        luma_strength: float,
        chroma_strength: float,
        self_ensemble: str,
        temporal_radius: int,
        motion_compensation: bool,
        temporal_tolerance: float,
        tile_size: int,
        tile_overlap: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = images[..., 3:] if images.shape[-1] == 4 else None
        noisy = images[..., :3].movedim(-1, 1)

        nets = [load_denoiser(*spec) for spec in MODELS[model]]
        device = nets[0].patcher.load_device
        output_device = mm.intermediate_device()
        folds = TTA_FOLDS[self_ensemble]

        memory_required = tile_size * tile_size * 3 * 4 * 384.0
        mm.load_models_gpu(
            [net.patcher for net in nets],
            memory_required=memory_required,
            force_full_load=True,
        )

        def run_nets(x, sigma):
            out = None
            for net in nets:
                if net.architecture.id == "DRUNet":
                    y = run_drunet(net, x, sigma)
                else:
                    y = net(x)
                out = y if out is None else out + y
            return out / len(nets)

        needs_sigma = any(n.architecture.id == "DRUNet" for n in nets)
        batch = noisy.shape[0]
        tile = tile_size
        while True:
            try:
                overlap = min(tile_overlap, tile // 4)
                pbar = comfy.utils.ProgressBar(
                    batch * comfy.utils.get_tiled_scale_steps(
                        noisy.shape[3], noisy.shape[2],
                        tile_x=tile, tile_y=tile, overlap=overlap,
                    )
                )
                denoised = torch.empty(noisy.shape, device=output_device)
                for b in range(batch):
                    frame = noisy[b:b + 1].to(device, torch.float32)
                    sigma = noise_level / 255.0
                    if needs_sigma and noise_level == 0.0:
                        sigma = denoise.clamp_sigma(
                            denoise.estimate_sigma(frame)
                        )
                        if b == 0:
                            print(
                                "[TrentNodes] Deep Denoise measured "
                                f"noise {denoise.sigma_label(sigma)}"
                            )
                    denoised[b:b + 1] = comfy.utils.tiled_scale(
                        frame,
                        lambda x: denoise.self_ensemble(
                            lambda t: run_nets(t, sigma), x, folds
                        ),
                        tile_x=tile, tile_y=tile, overlap=overlap,
                        upscale_amount=1, pbar=pbar,
                        output_device=output_device,
                    )
                break
            except Exception as e:
                mm.raise_non_oom(e)
                tile //= 2
                if tile < 128:
                    raise e

        denoised = denoised.clamp_(0.0, 1.0)
        if temporal_radius > 0 and batch > 1:
            flow_function = None
            if motion_compensation:
                flow_model = load_flow_model()
                mm.load_model_gpu(flow_model)
                flow_function = make_flow_function(
                    RaftOpticalFlow(flow_model.model, device=device)
                )
            denoised = denoise.temporal_fuse(
                denoised, temporal_radius, temporal_tolerance, device,
                flow_function,
            )

        noisy = noisy.to(denoised)
        result = denoise.blend_luma_chroma(
            noisy, denoised, luma_strength, chroma_strength
        ).clamp_(0.0, 1.0)
        noise = (noisy - result + 0.5).clamp_(0.0, 1.0)

        dtype = mm.intermediate_dtype()
        result = result.movedim(1, -1).to(dtype)
        if alpha is not None:
            result = torch.cat((result, alpha.to(result)), dim=-1)
        return (result, noise.movedim(1, -1).to(dtype))
