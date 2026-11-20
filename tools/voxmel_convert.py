"""Repackage a VoxMel checkpoint and the Vocos vocoder as one .safetensors file.

python tools/voxmel_convert.py --checkpoint model_L_final.pt --config config.yaml \
    --llm-config qwen_tokenizer/config.json --vocos pytorch_model.bin --output VoxMel-0.6B.safetensors
"""
import argparse
import json

import torch
import yaml
from safetensors.torch import save_file

LLM_KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size",
            "vocab_size", "max_position_embeddings", "rope_theta", "rms_norm_eps", "hidden_act")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="model_L_final.pt or model_M_final.pt")
parser.add_argument("--config", required=True, help="config.yaml from the same Hugging Face repo")
parser.add_argument("--llm-config", required=True, help="Qwen2 config.json named by llm_config in config.yaml")
parser.add_argument("--vocos", required=True, help="pytorch_model.bin from charactr/vocos-mel-24khz")
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.config) as f:
    conf = yaml.safe_load(f)["model_conf"]
with open(args.llm_config) as f:
    llm = json.load(f)

window = conf.get("tier_attn_window", {})
config = {
    "n_mels": conf["n_mels"],
    "dim": conf["dim"],
    "num_heads": conf["num_heads"],
    "n_layers": conf["n_layers"],
    "split_axes": conf["split_axes"],
    "flow_hidden": conf["flow_hidden"],
    "flow_depth": conf["flow_depth"],
    "detail_norm": conf["detail_norm"],
    "tier_attn_window": [window.get(axis, -1) for axis in conf["split_axes"]],
    "sample_rate": 24000,
    "hop_length": conf["hop_length"],
    "n_fft": 1024,
    "llm": {k: llm[k] for k in LLM_KEYS},
    "vocoder": {"dim": 512, "intermediate_dim": 1536, "num_layers": 8},
}

out = {}
for k, v in torch.load(args.checkpoint, map_location="cpu", weights_only=True).items():
    if k == "hist_mask_emb" or k.startswith("llm.model.lm_head."):
        continue
    out[k.replace("llm.model.model.", "llm.")] = v
for k, v in torch.load(args.vocos, map_location="cpu", weights_only=True).items():
    if k.startswith("feature_extractor."):
        continue
    out["vocoder." + k.replace("head.istft.window", "head.window")] = v

save_file(out, args.output, metadata={"voxmel_config": json.dumps(config)})
