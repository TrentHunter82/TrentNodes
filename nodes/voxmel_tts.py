import json
import math
import os

import torch
from transformers import AutoTokenizer

import comfy.audio
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ..utils.voxmel.model import VoxMel

folder_paths.add_model_folder_path("voxmel", os.path.join(folder_paths.models_dir, "voxmel"))

TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "voxmel", "tokenizer")

# Token ids the checkpoints were trained with depend on this exact list and order.
SPECIAL_TOKENS = {
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": [
        "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
        "[breath]", "<strong>", "</strong>", "[noise]",
        "[laughter]", "[cough]", "[clucking]", "[accent]",
        "[quick_breath]", "[chuckle]", "[clearing_throat]",
        "[cry]", "[gasp]", "[heavy_breath]", "[scream]",
        "[sneer]", "[sob]",
        "<laughter>", "</laughter>",
        "[hissing]", "[sigh]", "[vocalized-noise]",
        "[lipsmack]", "[mn]", "[sneeze]", "[inhale]", "[sniffle]", "[exhale]", "[giggle]", "[yawn]",
        "<sys_begin>", "<sys_end>", "<prompt_begin>", "<prompt_end>", "<|BOU|>", "<|BOA|>", "<eos>",
        "[TEXT]", "[SPEECH]", "[beginofspeech]", "[endofspeech]",
    ],
}

VoxMelModel = io.Custom("VOXMEL_MODEL")


class VoxMelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxMelLoader",
            display_name="Load VoxMel Model",
            category="Trent/Audio",
            inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("voxmel"))],
            outputs=[VoxMelModel.Output()],
        )

    @classmethod
    def execute(cls, model_name):
        path = folder_paths.get_full_path_or_raise("voxmel", model_name)
        sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        model = VoxMel(**json.loads(metadata["voxmel_config"])).eval()
        model.load_state_dict(sd)
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        return io.NodeOutput((patcher, tokenizer))


class VoxMelGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VoxMelGenerate",
            display_name="VoxMel Generate Speech",
            category="Trent/Audio",
            inputs=[
                VoxMelModel.Input("model"),
                io.String.Input("text", multiline=True),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=32, min=1, max=256, tooltip="Flow steps for each Coarse-Mel frame."),
                io.Int.Input("residual_steps", default=32, min=1, max=256, tooltip="Flow steps for the temporal and frequency residual tiers."),
                io.Float.Input("cfg_text", default=3.5, min=0.0, max=20.0, step=0.1),
                io.Float.Input("cfg_speaker", default=3.0, min=0.0, max=20.0, step=0.1, tooltip="Used only with reference audio."),
                io.Float.Input("max_seconds", default=30.0, min=0.1, max=600.0, step=0.1),
                io.Audio.Input("reference_audio", optional=True, tooltip="Voice to clone. Without it the voice is random."),
                io.String.Input("reference_text", multiline=True, optional=True, tooltip="Exact transcript of the reference audio."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, model, text, seed, steps, residual_steps, cfg_text, cfg_speaker, max_seconds, reference_audio=None, reference_text=""):
        patcher, tokenizer = model
        comfy.model_management.load_models_gpu([patcher])
        voxmel = patcher.model
        device = patcher.load_device

        prompt_wav = None
        tokens = []
        if reference_audio is not None:
            prompt_wav = reference_audio["waveform"][:1].to(device).mean(dim=1)
            prompt_wav = comfy.audio.resample(prompt_wav, reference_audio["sample_rate"], voxmel.sample_rate)
            tokens = tokenizer(reference_text)["input_ids"]
            # The model continues the reference, so latin text needs a word break after the transcript.
            if not ("一" <= text.lstrip()[:1] <= "鿿"):
                text = " " + text

        tokens = torch.tensor([tokens + tokenizer(text)["input_ids"]], dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(seed)
        waveform = voxmel.generate(
            tokens, generator, steps, residual_steps, cfg_text, cfg_speaker,
            min_frames=math.ceil(0.3 * voxmel.coarse_fps), max_frames=math.ceil(max_seconds * voxmel.coarse_fps),
            prompt_wav=prompt_wav)
        return io.NodeOutput({"waveform": waveform.unsqueeze(1).cpu(), "sample_rate": voxmel.sample_rate})


NODE_CLASS_MAPPINGS = {"VoxMelLoader": VoxMelLoader, "VoxMelGenerate": VoxMelGenerate}
