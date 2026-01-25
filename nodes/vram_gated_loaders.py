"""
VRAM-Gated Loaders for TrentNodes.

These loaders wait for a VRAM signal before loading models,
ensuring VidScribe MiniCPM completes and unloads first.
"""

import torch

import folder_paths
import comfy.sd
import comfy.utils


class VRAMGatedCheckpointLoader:
    """
    Load checkpoint only after receiving VRAM cleared signal.

    Connect vram_cleared from VidScribe to ensure proper execution order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_signal": ("STRING", {
                    "tooltip": "Connect vram_cleared from VidScribe"
                }),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "Loads checkpoint after VidScribe clears VRAM. "
        "Connect vram_cleared output to vram_signal input."
    )

    def load(self, vram_signal, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise(
            "checkpoints", ckpt_name
        )
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        return out[:3]


class VRAMGatedVAELoader:
    """Load VAE only after receiving VRAM cleared signal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_signal": ("STRING", {
                    "tooltip": "Connect vram_cleared from VidScribe"
                }),
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = "Load VAE after VidScribe clears VRAM."

    def load(self, vram_signal, vae_name):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        return (comfy.sd.VAE(sd_path=vae_path),)


class VRAMGatedUNETLoader:
    """Load Diffusion Model (UNET) only after receiving VRAM cleared signal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_signal": ("STRING", {
                    "tooltip": "Connect vram_cleared from VidScribe"
                }),
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                ),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = "Load Diffusion Model after VidScribe clears VRAM."

    def load(self, vram_signal, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", unet_name
        )
        model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options
        )
        return (model,)


class VRAMGatedLoraLoaderModelOnly:
    """Load LoRA (model only) after receiving VRAM cleared signal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vram_signal": ("STRING", {
                    "tooltip": "Connect vram_cleared from VidScribe"
                }),
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = "Load LoRA after VidScribe clears VRAM."

    def load(self, vram_signal, model, lora_name, strength_model):
        if strength_model == 0:
            return (model,)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(
            model, None, lora, strength_model, 0
        )
        return (model_lora,)


NODE_CLASS_MAPPINGS = {
    "VRAMGatedCheckpointLoader": VRAMGatedCheckpointLoader,
    "VRAMGatedVAELoader": VRAMGatedVAELoader,
    "VRAMGatedUNETLoader": VRAMGatedUNETLoader,
    "VRAMGatedLoraLoaderModelOnly": VRAMGatedLoraLoaderModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMGatedCheckpointLoader": "VRAM Gated Checkpoint Loader",
    "VRAMGatedVAELoader": "VRAM Gated VAE Loader",
    "VRAMGatedUNETLoader": "VRAM Gated Diffusion Model Loader",
    "VRAMGatedLoraLoaderModelOnly": "VRAM Gated LoRA Loader (Model Only)",
}
