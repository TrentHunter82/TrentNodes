"""
Manual runner for Cut Detective - no ComfyUI server needed. Decodes a
video with cv2, runs the detector(s), prints the report, and writes the
film strip to a PNG you can open.

Usage (from the ComfyUI root):

    venv/bin/python custom_nodes/TrentNodes/tools/cut_detective_dev_run.py \
        --video input/clip.mp4

    # compare every detector on the same clip - the fastest way to see
    # whether a disagreement is the model or the wiring
    venv/bin/python custom_nodes/TrentNodes/tools/cut_detective_dev_run.py \
        --video input/clip.mp4 --detector all

omnishotcut needs CUDA and the omnishotcut package; see the handoff at
docs/CUT_DETECTIVE_HANDOFF.md. transnetv2 and classic always work.
"""

import argparse
import os
import sys
import time
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils"):
        module = types.ModuleType(f"TrentNodes.{sub}")
        module.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = module

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from TrentNodes.utils.cut_detect import (  # noqa: E402
    detect_shots,
    format_cut_times,
    format_report,
    parse_cut_times,
    render_film_strip,
    shots_to_json,
)


def load_frames(path, max_side=640, limit=0):
    """Decode a video to a (B, H, W, C) [0,1] tensor, like a VIDEO input."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(
            f"could not open {path}. OpenCV cannot decode AV1; re-encode "
            "to H.264 first."
        )
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0

    frames = []
    while not limit or len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if max_side and max(frame.shape[:2]) > max_side:
            scale = max_side / max(frame.shape[:2])
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            )
        frames.append(frame)
    capture.release()

    if not frames:
        raise SystemExit(f"decoded 0 frames from {path}")
    images = torch.from_numpy(np.stack(frames)).float() / 255.0
    return images, fps


def run(images, fps, detector, args):
    started = time.time()
    shots = detect_shots(
        images, fps=fps, detector=detector,
        sensitivity=args.sensitivity, min_shot_frames=args.min_shot_frames,
        overlap=args.overlap, fallback_policy=args.fallback_policy,
    )
    elapsed = time.time() - started

    print(f"\n{'=' * 62}\n{detector}  ({elapsed:.2f}s)\n{'=' * 62}")
    if shots.fallback:
        print(f"FALLBACK: asked for {shots.requested}, ran {shots.detector}")
    print(format_report(shots))
    print(f"\ncut_times: {format_cut_times(shots)}")

    strip = render_film_strip(
        images, shots, thumb_width=args.thumb_width,
        thumbs_per_shot=args.thumbs_per_shot,
        title=f"{os.path.basename(args.video)} - {shots.detector}",
    )
    out = os.path.join(
        args.out_dir, f"cut_detective_{shots.detector}.png"
    )
    Image.fromarray(
        (strip[0].numpy() * 255).astype(np.uint8)
    ).save(out)
    print(f"film strip -> {out}")

    if args.json:
        print(shots_to_json(shots))

    # The H3 hand-off is only as good as the round trip, so check it here
    # rather than discovering a format drift inside a paid VLM call.
    parsed = [c.time for c in parse_cut_times(format_cut_times(shots))]
    if parsed != shots.cut_times:
        print(f"WARNING: cut_times did not round-trip: {parsed}")
    return shots


def main():
    parser = argparse.ArgumentParser(description="Cut Detective dev runner")
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--detector", default="auto",
        choices=["auto", "omnishotcut", "transnetv2", "classic", "all"],
    )
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--min-shot-frames", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument(
        "--thumbs-per-shot", type=int, default=1,
        help="thumbnails sampled from each shot; 1 shows only the "
             "frame after the cut",
    )
    parser.add_argument(
        "--overlap", type=int, default=20,
        help="OmniShotCut inference-window overlap in frames. The knob "
             "that actually moves its results; sensitivity does not",
    )
    parser.add_argument(
        "--fallback-policy", default="cascade",
        choices=["cascade", "neural_only", "strict"],
        help="how far 'auto' may fall when a detector is unavailable",
    )
    parser.add_argument(
        "--max-side", type=int, default=640,
        help="downscale on decode; detection resizes internally anyway",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N frames (0 = all)"
    )
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images, fps = load_frames(args.video, args.max_side, args.limit)
    print(f"decoded {tuple(images.shape)} @ {fps:.3f} fps")

    detectors = (
        ["omnishotcut", "transnetv2", "classic"] if args.detector == "all"
        else [args.detector]
    )
    results = {}
    for name in detectors:
        try:
            results[name] = run(images, fps, name, args)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{name}: FAILED - {type(exc).__name__}: {exc}")

    if len(results) > 1:
        print(f"\n{'=' * 62}\ncomparison\n{'=' * 62}")
        for name, shots in results.items():
            kinds = {}
            for shot in shots.shots[1:]:
                kinds[shot.entry] = kinds.get(shot.entry, 0) + 1
            ran = "" if shots.detector == name else f" (ran {shots.detector})"
            print(
                f"{name:13s}{ran} {len(shots.shots):3d} shots  "
                + (", ".join(f"{n}x {k}" for k, n in sorted(kinds.items()))
                   or "no cuts")
            )


if __name__ == "__main__":
    main()
