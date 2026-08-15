"""
Manual end-to-end runner for the Ultimate H3 Cowboy Promptor - no
ComfyUI server needed. Loads media with cv2/torchaudio, runs the full
node pipeline against a real VLM provider (or a canned reply with
--reply, spending nothing), prints the prompt, the label map and the
sampler numbers.

Usage (from the ComfyUI root):

    venv/bin/python custom_nodes/TrentNodes/tools/h3_cowboy_dev_run.py \
        --video input/clip.mp4 \
        --row 1 "person|the courier|short dark hair, olive rain jacket" \
        --subject-image 1 input/ref.png \
        --target "the courier ducks under a roller shutter" \
        --provider anthropic

    # Base mode needs no video at all:
    venv/bin/python custom_nodes/TrentNodes/tools/h3_cowboy_dev_run.py \
        --mode base_T2VA --target "a lighthouse in a storm at dusk"

    # Re-run the validators on a hand-written reply, offline:
    ... --reply @reply.txt --video input/clip.mp4

API keys come from the environment (ANTHROPIC_API_KEY / OPENAI_API_KEY)
or --api-key.
"""

import argparse
import json
import os
import sys
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils", "utils.h3_prompt", "utils.h3_cowboy",
                "utils.cut_detect"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from TrentNodes.nodes import ultimate_h3_cowboy_promptor as node_mod  # noqa: E402
from TrentNodes.utils.h3_cowboy import spec  # noqa: E402
from TrentNodes.utils.h3_prompt.backends import (  # noqa: E402
    DEFAULT_MODELS,
    VLMResult,
)


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
        if not video_path:
            raise SystemExit("--audio video needs --video to pull from")
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


def _maybe_file(value: str) -> str:
    """Read '@path' as a file; anything else is the literal value."""
    if value.startswith("@"):
        with open(value[1:], "r", encoding="utf-8") as handle:
            return handle.read().strip()
    return value


class CannedBackend:
    """Return one fixed reply, so the validators run without an API call."""

    name = "canned"
    supports_audio = False
    supports_video = False

    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None, video=None):
        return VLMResult(text=self.reply, usage={"model": "canned"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default="ref", choices=list(spec.MODES),
        help="ref writes the six-section reference format; the base_* "
             "modes write the three-field base format",
    )
    parser.add_argument(
        "--target", default="the courier ducks under a roller shutter",
        help="what the target video should show. @path also works",
    )
    parser.add_argument(
        "--row", nargs=2, action="append", default=[],
        metavar=("N", "KIND|NAME|FEATURES"),
        help="fill subject row N, like the node face. Three fields "
             "split on '|'; kind and name may be empty: "
             "--row 2 'scene|the loading bay|wet concrete, sodium light'",
    )
    parser.add_argument(
        "--subject-image", nargs=2, action="append", default=[],
        metavar=("N", "PATH"),
        help="wire an image into subject_N_image",
    )
    parser.add_argument(
        "--subjects", default="",
        help="EXTRA typed subjects (the advanced field), one DSL line "
             "each, numbered after the filled rows. @path also works",
    )
    parser.add_argument("--video", default="")
    parser.add_argument(
        "--video-role", default="subject_source",
        help="what <Video 1> contributes (see the node's tooltip)",
    )
    parser.add_argument(
        "--audio", default="",
        help="path to an audio file. Use 'video' to pull the source "
             "clip's own track via ffmpeg.",
    )
    parser.add_argument("--audio-role", default="none")
    parser.add_argument(
        "--provider", default="anthropic", choices=list(DEFAULT_MODELS),
    )
    parser.add_argument("--model", default="auto")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--reply", default="",
        help="skip the VLM: run the pipeline on this text (@path works). "
             "Retry errors print instead of costing a second call.",
    )
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cut-times", default="",
        help="measured shot list, in any form Cut Detective emits. "
             "@path also works",
    )
    parser.add_argument("--dialogue", default="", help="@path also works")
    parser.add_argument("--constraint-notes", default="")
    parser.add_argument("--duration-override", type=float, default=0.0)
    parser.add_argument(
        "--no-snap", action="store_true",
        help="do not snap the duration onto H3's 17k+5 frame grid",
    )
    parser.add_argument(
        "--base-picture-role", default=node_mod.DEFAULT_PICTURE_ROLE,
        help="base modes only: what the wired picture is",
    )
    parser.add_argument("--music-video", action="store_true")
    parser.add_argument(
        "--music-source", default="auto",
        choices=["auto", "generate_score", "reuse_audio_1"],
    )
    parser.add_argument("--lyrics", default="", help="@path also works")
    parser.add_argument("--music-description", default="")
    parser.add_argument(
        "--json", action="store_true",
        help="also print the analysis_json",
    )
    args = parser.parse_args()

    args.target = _maybe_file(args.target)
    args.subjects = _maybe_file(args.subjects)
    args.cut_times = _maybe_file(args.cut_times)
    args.dialogue = _maybe_file(args.dialogue)
    args.lyrics = _maybe_file(args.lyrics)

    subject_fields = {}
    for slot, packed in args.row:
        parts = (packed.split("|") + ["", "", ""])[:3]
        kind, name, features = (p.strip() for p in parts)
        if kind:
            subject_fields[f"subject_{slot}_kind"] = kind
        subject_fields[f"subject_{slot}_name"] = name
        subject_fields[f"subject_{slot}_description"] = features
    for slot, path in args.subject_image:
        subject_fields[f"subject_{slot}_image"] = load_image(path)

    frames, fps = (None, 24.0)
    if args.video:
        frames, fps = load_video(args.video)
        print(f"Loaded {frames.shape[0]} frames @ {fps:.3f} fps")

    audio = None
    if args.audio:
        audio = load_audio(args.audio, args.video)
        samples = audio["waveform"].shape[-1]
        print(
            f"Loaded {samples / audio['sample_rate']:.2f}s of audio "
            f"@ {audio['sample_rate']} Hz"
        )

    if args.reply:
        canned = CannedBackend(_maybe_file(args.reply))
        node_mod.get_backend = lambda *a, **k: canned
        print("Using the canned reply; no API call is made.")

    node = node_mod.UltimateH3CowboyPromptor()
    outputs = node.generate(
        h3_mode=args.mode,
        subjects=args.subjects,
        target_description=args.target,
        vlm_provider=args.provider,
        model=args.model,
        frames=frames,
        fps=fps,
        audio=audio,
        api_key=args.api_key,
        video_role=args.video_role,
        audio_role=args.audio_role,
        cut_times=args.cut_times,
        dialogue=args.dialogue,
        constraint_notes=args.constraint_notes,
        duration_override=args.duration_override,
        max_frames_to_analyze=args.max_frames,
        seed=args.seed,
        base_picture_role=args.base_picture_role,
        snap_duration_to_h3_grid=not args.no_snap,
        music_video=args.music_video,
        music_source=args.music_source,
        lyrics=args.lyrics,
        music_description=args.music_description,
        **subject_fields,
    )
    named = dict(zip(node.RETURN_NAMES, outputs))

    print("\n" + "=" * 72)
    print(f"--- h3_prompt ({len(named['h3_prompt'])} chars) ---")
    print(named["h3_prompt"])
    print("=" * 72)
    print(
        f"\nduration={named['duration_seconds']}s fps={named['fps']} "
        f"width={named['width']} height={named['height']} "
        f"length={named['length']}"
    )
    print(f"checkpoint: {named['h3_checkpoint_hint']}")
    if named["label_map"].strip():
        print("\nlabel_map:\n" + named["label_map"])

    # unresolved_errors are already folded into warnings by the node,
    # so printing both would say everything twice.
    analysis = json.loads(named["analysis_json"])
    for line in analysis.get("warnings", []) or []:
        print(f"WARNING: {line}")
    if args.json:
        print("\n" + named["analysis_json"])


if __name__ == "__main__":
    main()
