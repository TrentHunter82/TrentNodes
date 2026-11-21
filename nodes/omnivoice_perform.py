# OmniVoice performance transfer: re-voice a recorded line in a cloned voice, keeping its timing and delivery.
# Method by Nathan (SanDiegoDude), OmniVoice Manager, Apache License 2.0. Model: k2-fsa/OmniVoice.
import os

import torch
from huggingface_hub import snapshot_download
from transformers import pipeline

import comfy.model_management
import comfy.model_patcher
import folder_paths
from comfy_api.latest import io

from ..utils.omnivoice.model import OmniVoice
from ..utils.omnivoice.perform import perform_transfer

folder_paths.add_model_folder_path("omnivoice", os.path.join(folder_paths.models_dir, "omnivoice"))

ASR_MODEL = "openai/whisper-large-v3"
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

OmniVoiceModel = io.Custom("OMNIVOICE_MODEL")


class OmniVoiceHolder(torch.nn.Module):
    # ModelPatcher writes model.device, which is a read-only property on a transformers model.
    def __init__(self, omni):
        super().__init__()
        self.omni = omni


def _model_folders():
    # An OmniVoice model is a folder (LLM + text tokenizer + audio_tokenizer/), e.g. models/omnivoice/OmniVoice.
    names = []
    for base in folder_paths.get_folder_paths("omnivoice"):
        if os.path.isdir(base):
            names += [d for d in os.listdir(base) if os.path.isfile(os.path.join(base, d, "config.json"))]
    return sorted(names)


def _mono(audio):
    return audio["waveform"][0].float().mean(dim=0).cpu().numpy(), audio["sample_rate"]


class TrentOmniVoiceLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TrentOmniVoiceLoader",
            display_name="Load OmniVoice Model",
            category="Trent/Audio",
            inputs=[
                io.Combo.Input("model_name", options=_model_folders(), tooltip="Model folder in models/omnivoice."),
                io.Combo.Input("dtype", options=list(DTYPES), default="float16"),
            ],
            outputs=[OmniVoiceModel.Output()],
        )

    @classmethod
    def execute(cls, model_name, dtype):
        if model_name not in _model_folders():
            raise ValueError(f"OmniVoice model folder not found: {model_name}")
        path = next(os.path.join(b, model_name) for b in folder_paths.get_folder_paths("omnivoice") if os.path.isdir(os.path.join(b, model_name)))
        model = OmniVoice.from_pretrained(path, dtype=DTYPES[dtype]).eval()
        patcher = comfy.model_patcher.ModelPatcher(OmniVoiceHolder(model), load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        return io.NodeOutput(patcher)


class TrentOmniVoicePerformanceTransfer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TrentOmniVoicePerformanceTransfer",
            display_name="OmniVoice Performance Transfer",
            category="Trent/Audio",
            description="Re-voice a recorded line in the voice of a reference clip. The output has the exact length and timing of the performance.",
            inputs=[
                OmniVoiceModel.Input("model"),
                io.Audio.Input("performance", tooltip="The line as performed: its timing, pauses and emphasis are kept. Clean speech works best."),
                io.Audio.Input("voice", tooltip="The target voice. 3 to 10 s of clean speech."),
                io.Combo.Input("mode", options=["voice", "character"], default="voice",
                               tooltip="voice: the exact source delivery in the target timbre. character: the source timing, then the target's own delivery takes over."),
                io.Int.Input("strength", default=3, min=1, max=6,
                             tooltip="voice: 1 = full pin to the source ... 5 = loosest (6 = extra loose, untuned). character: 1 = barely changed ... 4 = default ... 6 = most character."),
                io.Int.Input("steps", default=32, min=4, max=128),
                io.Float.Input("guidance", default=2.0, min=0.0, max=10.0, step=0.1),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.String.Input("language", default="en", tooltip="Language code of the words, e.g. en, de, ja."),
                io.String.Input("text", multiline=True, optional=True, tooltip="The words of the performance. Empty: Whisper transcribes them."),
                io.String.Input("voice_text", multiline=True, optional=True, tooltip="The words of the voice clip. Empty: Whisper transcribes them."),
            ],
            outputs=[io.Audio.Output()],
        )

    @classmethod
    def execute(cls, model, performance, voice, mode, strength, steps, guidance, seed, language, text="", voice_text="") -> io.NodeOutput:
        comfy.model_management.load_models_gpu([model])
        omni = model.model.omni
        perf_wav, perf_sr = _mono(performance)
        voice_wav, voice_sr = _mono(voice)

        text, voice_text = text.strip(), voice_text.strip()
        if not text or not voice_text:
            asr = pipeline("automatic-speech-recognition", model=snapshot_download(ASR_MODEL, local_files_only=True),
                           dtype=torch.float16 if model.load_device.type == "cuda" else torch.float32, device=model.load_device)
            if not text:
                text = asr({"array": perf_wav, "sampling_rate": perf_sr})["text"].strip()
            if not voice_text:
                voice_text = asr({"array": voice_wav, "sampling_rate": voice_sr})["text"].strip()
            del asr

        vc_prompt = omni.create_voice_clone_prompt((torch.from_numpy(voice_wav), voice_sr), ref_text=voice_text)
        out = perform_transfer(omni, text, vc_prompt, perf_wav, perf_sr, mode, strength, seed, language, steps, guidance)
        return io.NodeOutput({"waveform": torch.from_numpy(out).float().view(1, 1, -1), "sample_rate": omni.sampling_rate})


NODE_CLASS_MAPPINGS = {"TrentOmniVoiceLoader": TrentOmniVoiceLoader, "TrentOmniVoicePerformanceTransfer": TrentOmniVoicePerformanceTransfer}
