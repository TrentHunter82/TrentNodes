#!/usr/bin/env python
"""
Drive the H3 Skill Promptor stack without ComfyUI.

Attaches to a running llama-server by default (the node's managed one on
port 8735), or spawns one with --model. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tools/h3_skill_dev_run.py \
        --mode ref2va --brief "a knight sharpens a sword by firelight" \
        --duration 6 --image /path/to/reference.png

Same checklist-retry flow as the node, minus the tensor plumbing.
"""

import argparse
import base64
import io
import os
import sys
import types

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("utils", "utils.h3_prompt", "utils.h3_cowboy", "utils.h3_skill"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.utils import llamacpp_server  # noqa: E402
from TrentNodes.utils.h3_skill.checklist import assemble_final, validate  # noqa: E402
from TrentNodes.utils.h3_skill.client import build_user_message, chat  # noqa: E402
from TrentNodes.utils.h3_skill.skill_loader import (  # noqa: E402
    CHECKPOINT_FOR_MODE,
    MODES,
    build_system_prompt,
    build_user_context,
)


def _load_image_b64(path: str, max_side: int = 1344) -> str:
    from PIL import Image
    pil = Image.open(path)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    scale = max_side / max(pil.size)
    if scale < 1.0:
        pil = pil.resize(
            (max(1, int(pil.size[0] * scale)), max(1, int(pil.size[1] * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _strip(text: str):
    notes = []
    cleaned = text.strip()
    if cleaned.startswith("<think>") and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
        notes.append("stripped <think>")
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        notes.append("stripped fence")
    return cleaned, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="ref2va")
    parser.add_argument("--brief", required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dialogue", default="")
    parser.add_argument("--image", action="append", default=[],
                        help="reference picture path; repeatable, in order")
    parser.add_argument("--base-url", default="http://127.0.0.1:8735",
                        help="attach to a running server (default)")
    parser.add_argument("--model", default="",
                        help=".gguf path: spawn a managed server instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reasoning-effort", default="low",
                        choices=["low", "medium", "xhigh"])
    parser.add_argument("--max-tokens", type=int, default=3072)
    args = parser.parse_args()

    if args.model:
        spec = llamacpp_server.ServerSpec(
            model_path=os.path.abspath(args.model),
            mmproj_path=llamacpp_server.find_mmproj_for(args.model),
            reasoning_effort=args.reasoning_effort,
        )
        handle = llamacpp_server.ensure_server(spec)
    else:
        handle = llamacpp_server.attach(args.base_url)
    print(f"server: {handle.base_url}", file=sys.stderr)

    bracket = args.mode in ("ref2va", "i2va", "l2va")
    pairs, lines = [], []
    for index, path in enumerate(args.image, start=1):
        tag = f"<Picture {index}>" if bracket else f"Picture {index}"
        label = f"{tag} (reference picture {index})"
        pairs.append((label, _load_image_b64(path)))
        lines.append(label)

    system, source = build_system_prompt(args.mode)
    print(f"skill source: {source}", file=sys.stderr)
    context = build_user_context(
        args.mode, args.brief, args.duration,
        dialogue=args.dialogue, image_lines=lines,
    )
    messages = [
        {"role": "system", "content": system},
        build_user_message(context, pairs),
    ]
    kwargs = dict(
        seed=args.seed, temperature=args.temperature,
        max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
    )

    raw, usage = chat(handle.base_url, messages, **kwargs)
    body, notes = _strip(raw)
    print(f"first pass: {usage}", file=sys.stderr)
    errors = validate(body, args.mode, args.duration)
    if errors:
        print(f"retrying on {len(errors)} checklist violation(s):",
              file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        numbered = "\n".join(
            f"{i}. {e}" for i, e in enumerate(errors, 1)
        )
        messages.append({"role": "assistant", "content": body})
        messages.append({"role": "user", "content": (
            "Your prompt violates the skill checklist:\n" + numbered +
            "\nFix ONLY these violations and return the full corrected "
            "prompt. Same output contract: prompt text only."
        )})
        raw, usage = chat(handle.base_url, messages, **kwargs)
        body, notes = _strip(raw)
        print(f"retry pass: {usage}", file=sys.stderr)
        errors = validate(body, args.mode, args.duration)

    print("=" * 72, file=sys.stderr)
    print(assemble_final(body, args.mode, args.duration))
    print("=" * 72, file=sys.stderr)
    print(f"checkpoint: {CHECKPOINT_FOR_MODE[args.mode]}", file=sys.stderr)
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("validation: PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
