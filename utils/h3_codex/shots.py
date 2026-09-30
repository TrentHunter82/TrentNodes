"""Cut Detective timing and per-shot evidence for the Codex promptor."""

import math
import re

from ..cut_detect.formats import ParsedCut, format_timecode, parse_cut_times
from ..h3_prompt.keyframes import select_keyframes

SHOT_MARKER = re.compile(r"\[Shot (\d+)\](?: At (\d{1,3}):(\d{2})\.(\d{1,3}),)?")


def shot_timeline(text, duration):
    if not text.strip():
        return [], []
    notes = []
    try:
        cuts = parse_cut_times(text)
    except (ValueError, TypeError, OverflowError):
        cuts = []
    if not cuts:
        notes.append("No readable cut times; using one shot. Check Cut Detective's cut_times or cuts_json output.")
    outside = [cut for cut in cuts if not math.isfinite(cut.time) or cut.time > duration + 0.001]
    if outside:
        notes.append("Cut times outside duration_seconds were omitted: "
                     + ", ".join(str(cut.time) for cut in outside)
                     + ". Match duration_seconds to the source; times are never rescaled.")
    cuts = [ParsedCut(0.0, "start")] + [cut for cut in cuts if 0 < cut.time < duration]
    shots = []
    for i, cut in enumerate(cuts):
        end = cuts[i + 1].time if i + 1 < len(cuts) else duration
        marker = "[Shot 1]" if i == 0 else f"[Shot {i + 1}] At {format_timecode(cut.time)},"
        shots.append({"shot": i + 1, "start_seconds": cut.time, "end_seconds": end,
                      "entry": cut.kind, "prompt_marker": marker})
    return shots, notes


def sample_shots(video_frames, fps, max_frames, shots, interrupt):
    """Keep opening/middle/ending evidence even when the total budget is small."""
    samples, notes = [], []
    budget = max(int(max_frames), 3 * len(shots))
    if budget > max_frames:
        notes.append(f"Shot coverage raises the frame budget from {max_frames} to {budget} "
                     "so every shot can include opening, middle and ending samples.")
    for i, shot in enumerate(shots):
        interrupt()
        # Cut Detective rounds seconds to milliseconds. Snap back to source frames.
        first = max(0, round(shot["start_seconds"] * fps))
        stop = min(len(video_frames), round(shot["end_seconds"] * fps))
        if stop <= first:
            notes.append(f"Shot {shot['shot']}: no source frames in this interval; action timing needs supplied context.")
            continue
        count = stop - first
        target = min(count, budget // len(shots) + (i < budget % len(shots)))
        indices = {0, (count - 1) // 2, count - 1}
        if target > len(indices):
            selected = select_keyframes(video_frames[first:stop], fps, max_frames=target,
                                        known_boundaries=[0])
            for index in selected.indices:
                if len(indices) >= target:
                    break
                indices.add(index)
        for index in sorted(indices):
            absolute = first + index
            samples.append((absolute, absolute / fps, shot["shot"]))
        notes.append(f"Shot {shot['shot']} ({shot['start_seconds']:.3f}–{shot['end_seconds']:.3f}s): "
                     + ", ".join(f"{(first + index) / fps:.3f}s" for index in sorted(indices))
                     + ("; fewer than three source frames, motion evidence is limited." if count < 3 else "."))
    return samples, notes


def check_shot_times(description, shots):
    matches = list(SHOT_MARKER.finditer(description))
    expected = [shot["shot"] for shot in shots]
    found = [int(match[1]) for match in matches]
    if found != expected:
        return [f"Cut timeline: describe exactly shots {expected} in order; found {found}. Do not add or omit cuts."]
    issues = []
    for match, shot in zip(matches, shots):
        if shot["shot"] == 1:
            continue  # The shared official-format validator checks the untimed first marker.
        actual = None if match[2] is None else int(match[2]) * 60 + int(match[3]) + float("0." + match[4])
        if actual is None or abs(actual - shot["start_seconds"]) > 0.0005:
            issues.append(f"Cut timeline: use '{shot['prompt_marker']}' exactly; the supplied cut time must not move.")
    return issues
