"""H3 request construction, reference checks and small on-disk result cache."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from ..h3_prompt.imaging import tensor_to_jpeg_b64
from ..h3_prompt.keyframes import select_keyframes
from ..h3_skill.checklist import assemble_final, validate
from ..h3_skill.skill_loader import CHECKPOINT_FOR_MODE
from .client import CodexSession, CodexError
from .skill import REVISION, load_official_skill, instructions

MODES = ("auto", "t2va", "i2va", "fl2va", "l2va", "ref2va")
SCHEMA = {
    "type": "object",
    "properties": {
        "prompt_body": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompt_body", "assumptions"],
    "additionalProperties": False,
}
BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REF_FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
              "overall_soundscape", "non_diegetic_music")
HEADER = re.compile(r"(?m)^(" + "|".join(dict.fromkeys(BASE_FIELDS + REF_FIELDS)) + r"):\s*")
LABEL = re.compile(r"<(Picture|Video|Audio) (\d+)>")


def sections(text):
    matches = list(HEADER.finditer(text))
    return {m.group(1): text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
            for i, m in enumerate(matches)}


def resolve_mode(mode, first_frame=None, last_frame=None, reference_images=None,
                 video_frames=None, audio=None):
    if mode not in MODES:
        raise ValueError("Choose a supported H3 mode.")
    if mode != "auto":
        return mode
    if reference_images is not None or video_frames is not None or audio is not None:
        return "ref2va"
    if first_frame is not None and last_frame is not None:
        return "fl2va"
    if first_frame is not None:
        return "i2va"
    return "l2va" if last_frame is not None else "t2va"


def prepare_references(mode, first_frame, last_frame, reference_images,
                       video_frames, fps, max_frames, audio, interrupt):
    images, descriptions, notes = [], [], []
    counts = {"Picture": 0, "Video": 0, "Audio": 0}
    for batch, role in ((first_frame, "first frame"), (last_frame, "last frame"),
                        (reference_images, "reference image")):
        if batch is None:
            continue
        for frame in batch:
            interrupt()
            counts["Picture"] += 1
            label = f"<Picture {counts['Picture']}>"
            description = f"{label}: {role}"
            images.append((description, tensor_to_jpeg_b64(frame, max_side=1344)))
            descriptions.append(description)
    expected = {"t2va": 0, "i2va": 1, "l2va": 1, "fl2va": 2}.get(mode)
    if expected is not None and counts["Picture"] != expected:
        raise ValueError(f"{mode} needs {expected} picture(s); got {counts['Picture']}. Use ref2va for arbitrary references.")
    if video_frames is not None and len(video_frames):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video fps must be positive.")
        frames = select_keyframes(video_frames, fps, max_frames=max_frames)
        counts["Video"] = 1 if mode == "ref2va" else 0
        prefix = "<Video 1>" if mode == "ref2va" else "Visual context clip (not an H3 reference)"
        descriptions.append(f"{prefix}: {len(video_frames)} frames at {fps:g} fps; only sampled stills are inspected.")
        for index, timestamp in zip(frames.indices, frames.timestamps):
            interrupt()
            label = f"{prefix}, sampled frame {index} at {timestamp:.3f}s"
            images.append((label, tensor_to_jpeg_b64(video_frames[index])))
        notes.append(f"Video inspection: {len(frames.indices)} of {len(video_frames)} frames, at "
                     + ", ".join(f"{t:.3f}s" for t in frames.timestamps) + ". Audio was not inspected.")
    if audio is not None:
        duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        counts["Audio"] = 1 if mode == "ref2va" else 0
        prefix = "<Audio 1>" if mode == "ref2va" else "Source audio (context only)"
        descriptions.append(f"{prefix}: duration {duration:.3f}s. Not uploaded or heard by Codex. Use supplied audio analysis and exact dialogue only.")
        notes.append("Audio: metadata only. Connect H3 Audio Soundscaper outputs or supply a transcript/description for content-aware prompting.")
    return images, descriptions, notes, counts


def check_body(body, mode, duration, counts):
    issues = validate(body, mode, duration)
    for kind, number in sorted(set(LABEL.findall(body))):
        if int(number) < 1 or int(number) > counts[kind]:
            issues.append(f"Unknown reference <{kind} {number}>; supplied count is {counts[kind]}.")
    # FL2VA official prose may spell Picture 1 without angle brackets.
    for number in re.findall(r"(?<!<)\bPicture (\d+)\b", body):
        if not 1 <= int(number) <= counts["Picture"]:
            issues.append(f"Unknown reference Picture {number}.")
    return list(dict.fromkeys(issues))


def _decode_result(result):
    if not isinstance(result, dict) or not isinstance(result.get("prompt_body"), str) or not result["prompt_body"].strip():
        raise CodexError("Codex returned no usable prompt body.")
    if not isinstance(result.get("assumptions"), list) or any(not isinstance(x, str) for x in result["assumptions"]):
        raise CodexError("Codex returned malformed assumptions.")
    return result["prompt_body"].strip(), result["assumptions"]


def generate(cache_dir, *, brief, mode, duration, action, prompt_text, refinement,
             context, dialogue, reference_roles, source_soundscape, source_music,
             sound_log, model, effort, revision, timeout, first_frame=None,
             last_frame=None, reference_images=None, video_frames=None, fps=24.0,
             max_frames=8, audio=None, interrupt=lambda: None, session_factory=CodexSession):
    if action not in ("generate", "refine", "locked"):
        raise ValueError("Choose generate, refine or locked.")
    if action == "locked":
        if not prompt_text.strip():
            raise ValueError("The prompt editor is empty. Generate a prompt or paste one before locking.")
        fields = sections(prompt_text)
        inferred = "ref2va" if "subject_definitions" in fields else "t2va"
        if mode in ("auto", "ref2va"):
            checkpoint = CHECKPOINT_FOR_MODE[inferred]
        else:
            checkpoint = CHECKPOINT_FOR_MODE[mode]
        return {"prompt": prompt_text, "checkpoint": checkpoint,
                "report": "Locked: using the prompt editor exactly as saved. No Codex request.",
                "soundscape": fields.get("overall_soundscape", ""),
                "music": fields.get("non_diegetic_music", "")}
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be a positive number.")
    if action == "refine" and (not prompt_text.strip() or not refinement.strip()):
        raise ValueError("Refine needs a prompt in the editor and a revision request.")
    mode = resolve_mode(mode, first_frame, last_frame, reference_images, video_frames, audio)
    images, labels, notes, counts = prepare_references(
        mode, first_frame, last_frame, reference_images, video_frames, fps,
        max_frames, audio, interrupt,
    )
    data = {
        "mode": mode, "duration_seconds": duration, "creative_brief": brief,
        "reference_inventory": labels, "reference_roles": reference_roles,
        "additional_context": context, "exact_dialogue": dialogue,
        "supplied_soundscape_analysis": source_soundscape,
        "supplied_music_analysis": source_music, "supplied_sound_log": sound_log,
    }
    if action == "refine":
        data.update(previous_prompt=prompt_text, revision_request=refinement)
    if not brief.strip() and not images and not context.strip() and action != "refine":
        raise ValueError("Describe your idea or connect visual references before generating.")
    # Every prompt-affecting input participates; unrelated downstream H3 seeds do not.
    fingerprint = json.dumps({"contract": 1, "skill": REVISION, "data": data,
                              "images": images, "model": model, "effort": effort,
                              "revision": revision}, sort_keys=True, ensure_ascii=False)
    key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir) / "results" / f"{key}.json"
    if cache_path.is_file():
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        result["report"] += "\nReused saved result; no Codex request. Increase revision to regenerate."
        return result
    texts = load_official_skill(cache_dir, interrupt)
    inputs = [{"type": "text", "text": "Write the H3 prompt for this request:\n" + json.dumps(data, ensure_ascii=False)}]
    for label, encoded in images:
        inputs.extend([{"type": "text", "text": label},
                       {"type": "image", "url": "data:image/jpeg;base64," + encoded}])
    with tempfile.TemporaryDirectory(prefix="trent-h3-codex-") as directory:
        with session_factory(directory, instructions(texts, mode), model=model,
                             effort=effort, timeout=timeout, interrupt=interrupt) as session:
            body, assumptions = _decode_result(session.turn(inputs, SCHEMA))
            issues = check_body(body, mode, duration, counts)
            passes = 1
            if issues:
                interrupt()
                repair = ("Review the completed draft against the official guide and this diagnostic list. "
                          "Correct applicable issues while preserving intent and exact dialogue. "
                          "The official guide takes precedence over an over-strict diagnostic. Return the whole result.\n"
                          + "\n".join(issues))
                body, assumptions = _decode_result(session.turn([{"type": "text", "text": repair}], SCHEMA))
                issues = check_body(body, mode, duration, counts)
                passes = 2
            selected_model = session.model_name
    prompt = assemble_final(body, mode, duration)
    fields = sections(body)
    report = [f"Mode: {mode}; checkpoint: {CHECKPOINT_FOR_MODE[mode]}",
              f"Codex model: {selected_model}; completed passes: {passes}",
              f"Official MiniMax skill: {REVISION[:12]}"]
    report += notes
    report += ["Picture order: first_frame, last_frame, then reference_images. Use the same order in H3."] if counts["Picture"] else []
    report += ["Checks: " + ("review the diagnostics below" if issues else "passed (format and references; render quality still needs testing)")]
    report += ["- " + issue for issue in issues]
    report += ["Assumption: " + item for item in assumptions]
    result = {"prompt": prompt, "checkpoint": CHECKPOINT_FOR_MODE[mode],
              "report": "\n".join(report), "soundscape": fields.get("overall_soundscape", ""),
              "music": fields.get("non_diegetic_music", "")}
    interrupt()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_path.parent, delete=False) as handle:
        json.dump(result, handle, ensure_ascii=False)
        temporary = handle.name
    os.replace(temporary, cache_path)
    return result
