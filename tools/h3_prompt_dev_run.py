"""
Manual end-to-end runner for the H3 Auto Prompt Generator - no ComfyUI
server needed. Loads a video with cv2, runs the full node pipeline
against a real VLM provider, prints the prompt and the analysis JSON.

Usage (from the ComfyUI root):

    venv/bin/python custom_nodes/TrentNodes/tools/h3_prompt_dev_run.py \
        --video input/clip.mp4 --reference input/ref.png \
        --subject "Aria Voss" \
        --wardrobe "charcoal utility jacket, black cargo pants" \
        --provider anthropic

API keys come from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY)
or --api-key.
"""

import argparse
import os
import sys
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils", "utils.h3_prompt"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from TrentNodes.nodes.h3_auto_prompt import H3AutoPromptGenerator  # noqa: E402


def load_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"No frames decoded from: {path}")
    tensor = torch.from_numpy(
        np.stack(frames).astype(np.float32) / 255.0
    )
    return tensor, float(fps)


def load_audio(path: str, video_path: str):
    """Load an audio file (or extract the video's track) as ComfyUI AUDIO."""
    import torchaudio

    if path == "video":
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn",
             "-ac", "1", "-ar", "16000", wav_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                "ffmpeg could not extract audio:\n"
                + result.stderr.decode()[-400:]
            )
        path = wav_path

    waveform, sample_rate = torchaudio.load(path)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def load_image(path: str) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--subject", default="Aria Voss")
    parser.add_argument(
        "--wardrobe",
        default="charcoal utility jacket, black cargo pants, combat boots",
    )
    parser.add_argument(
        "--style",
        default="gritty handheld action thriller, desaturated color grade",
    )
    parser.add_argument(
        "--soundscape", default="ambient",
        choices=["fight", "ambient", "dialogue"],
    )
    parser.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "gemini", "openai", "kimi", "glm", "qwen_api",
                 "qwen_local", "minicpm_local", "magevl_local", "ollama"],
        help="magevl_local needs the ComfyUI environment (its wrapper "
             "imports comfy modules); use it from the node, not here",
    )
    parser.add_argument(
        "--audio", default="",
        help="path to an audio file to describe (gemini only). Use "
             "'video' to pull the source clip's own track via ffmpeg.",
    )
    parser.add_argument("--model", default="auto")
    parser.add_argument(
        "--profile", default="official",
        choices=["official", "upgraded", "both_ab"],
        help="both_ab = two calls; prints the official and upgraded "
             "prompts for A/B testing",
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--dialogue", default="")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument(
        "--json", action="store_true",
        help="also print the frame_analysis_json",
    )
    args = parser.parse_args()

    frames, fps = load_video(args.video)
    reference = load_image(args.reference)
    print(f"Loaded {frames.shape[0]} frames @ {fps:.3f} fps")

    audio = None
    if args.audio:
        audio = load_audio(args.audio, args.video)
        samples = audio["waveform"].shape[-1]
        print(
            f"Loaded {samples / audio['sample_rate']:.2f}s of audio "
            f"@ {audio['sample_rate']} Hz"
        )

    node = H3AutoPromptGenerator()
    prompt, prompt_b, duration, out_fps, analysis = node.generate(
        reference_image=reference,
        subject_name=args.subject,
        subject_wardrobe=args.wardrobe,
        scene_style=args.style,
        soundscape_type=args.soundscape,
        vlm_provider=args.provider,
        model=args.model,
        max_frames_to_analyze=args.max_frames,
        enable_audio_prompt=not args.no_audio,
        prompt_profile=args.profile,
        frames=frames,
        fps=fps,
        audio=audio,
        api_key=args.api_key,
        dialogue=args.dialogue,
    )

    label_a = "official" if args.profile in ("official", "both_ab") else args.profile
    print("\n" + "=" * 72)
    print(f"--- {label_a} ({len(prompt)} chars) ---")
    print(prompt)
    if prompt_b:
        print("\n" + "-" * 72)
        print(f"--- upgraded ({len(prompt_b)} chars) ---")
        print(prompt_b)
    print("=" * 72)
    print(f"\nduration={duration}s fps={out_fps}")
    if args.json:
        print("\n" + analysis)


if __name__ == "__main__":
    main()
