"""
Serializing and re-reading a detected cut list.

Cut Detective emits several strings; the H3 Auto Prompt Generator has to
read whichever one gets wired into it, so parse_cut_times() is written
to accept all of them plus whatever a person types by hand. Pure string
work - no torch, no ComfyUI - so it is testable offline.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from .detectors import ShotList

# Boundary kinds a parsed line may name. Order matters: the longest and
# most specific spellings are tried first so "hard cut" never matches as
# a bare "cut".
KIND_WORDS = [
    ("sudden jump", "sudden jump"),
    ("hard cut", "hard cut"),
    ("dissolve", "dissolve"),
    ("crossfade", "dissolve"),
    ("doorway", "doorway"),
    ("wipe", "wipe"),
    ("fade", "fade"),
    ("push", "push"),
    ("slide", "slide"),
    ("zoom", "zoom"),
    ("whip", "slide"),
    ("start", "start"),
    ("cut", "hard cut"),
]

_SHOT_PREFIX_RE = re.compile(r"\[?\s*shot\s+\d+\s*\]?", re.IGNORECASE)
_AT_RE = re.compile(r"\bat\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\d{1,3}:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?")


@dataclass
class ParsedCut:
    time: float
    kind: str = "hard cut"


def format_timecode(seconds: float) -> str:
    """Seconds -> MM:SS.mmm, the shot-time form H3 prompts use."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def format_cut_times(shots: ShotList, include_first: bool = True) -> str:
    """
    Shot start times as plain comma-separated seconds.

    The compact form, for wiring straight into another node. Shot 1
    starts at 0.000; drop it with include_first=False when the consumer
    wants boundaries only.
    """
    times = shots.cut_times
    if not include_first:
        times = [t for t in times if t > 0.0]
    return ", ".join(f"{t:.3f}" for t in times)


def format_shot_table(shots: ShotList) -> str:
    """
    One line per shot, readable by a person and by parse_cut_times().

        Shot 1 | 00:00.000 | 2.500s | start
        Shot 2 | 00:02.500 | 2.583s | hard cut
        Shot 3 | 00:05.083 | 1.917s | dissolve over 0.208s
    """
    lines = []
    for shot in shots.shots:
        entry = shot.entry
        if shot.transition_frames:
            span = shot.transition_frames
            length = (span[1] - span[0]) / shots.fps
            entry = f"{entry} over {length:.3f}s"
        lines.append(
            f"Shot {shot.index} | {format_timecode(shot.start_time)} | "
            f"{shot.duration:.3f}s | {entry}"
        )
    return "\n".join(lines)


def format_report(shots: ShotList) -> str:
    """Full human-readable summary, for a text preview or the console."""
    header = [
        "Cut Detective - shot boundary report",
        "=" * 38,
        f"Detector:  {shots.detector}",
        f"Duration:  {shots.duration:.3f}s at {shots.fps:.3f} fps "
        f"({shots.total_frames} frames)",
        f"Shots:     {len(shots.shots)}  ({shots.num_cuts} cuts)",
    ]

    kinds = {}
    for shot in shots.shots[1:]:
        kinds[shot.entry] = kinds.get(shot.entry, 0) + 1
    if kinds:
        header.append(
            "Cut kinds: "
            + ", ".join(f"{n}x {k}" for k, n in sorted(kinds.items()))
        )

    body = ["", format_shot_table(shots)]
    if shots.notes:
        body += ["", "Notes:"] + [f"  - {n}" for n in shots.notes]
    return "\n".join(header + body)


def shots_to_json(shots: ShotList) -> str:
    """Structured dump, including the detector's untouched labels."""
    payload = {
        "detector": shots.detector,
        "fps": round(shots.fps, 6),
        "total_frames": shots.total_frames,
        "duration_seconds": shots.duration,
        "num_shots": len(shots.shots),
        "num_cuts": shots.num_cuts,
        "cut_times": shots.cut_times,
        "hard_cut_times": shots.hard_cut_times,
        "shots": [
            {
                "index": s.index,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
                "entry": s.entry,
                "transition_frames": (
                    list(s.transition_frames) if s.transition_frames else None
                ),
                "raw_labels": s.raw_labels,
            }
            for s in shots.shots
        ],
        "notes": shots.notes,
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Reading a cut list back in
# ---------------------------------------------------------------------------

def _parse_time_token(token: str) -> Optional[float]:
    """'00:02.500' or '2.5' -> seconds. None if it is not a time."""
    try:
        if ":" in token:
            minutes, secs = token.split(":", 1)
            return int(minutes) * 60 + float(secs)
        return float(token)
    except ValueError:
        return None


def _kind_in(text: str) -> Optional[str]:
    lowered = text.lower()
    for word, kind in KIND_WORDS:
        if word in lowered:
            return kind
    return None


def _from_json(text: str) -> Optional[List[ParsedCut]]:
    """Read Cut Detective's JSON, a bare list of numbers, or neither."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None

    if isinstance(data, dict):
        if isinstance(data.get("shots"), list):
            cuts = []
            for entry in data["shots"]:
                if not isinstance(entry, dict):
                    continue
                time = entry.get("start_time")
                if time is None:
                    continue
                cuts.append(ParsedCut(
                    float(time), str(entry.get("entry") or "hard cut")
                ))
            return cuts
        for key in ("cut_times", "hard_cut_times", "cuts", "times"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            return None

    if not isinstance(data, list):
        return None

    cuts = []
    for entry in data:
        if isinstance(entry, (int, float)):
            cuts.append(ParsedCut(float(entry)))
        elif isinstance(entry, str):
            value = _parse_time_token(entry.strip().rstrip("s"))
            if value is not None:
                cuts.append(ParsedCut(value))
        elif isinstance(entry, dict):
            for key in ("start_time", "time", "start", "seconds"):
                if key in entry:
                    cuts.append(ParsedCut(
                        float(entry[key]),
                        str(entry.get("entry") or entry.get("kind")
                            or "hard cut"),
                    ))
                    break
    return cuts


def parse_cut_times(text: str) -> List[ParsedCut]:
    """
    Read a cut list out of any format Cut Detective emits, or a hand-typed
    one.

    Accepted:
        "0.0, 2.5, 5.083"                       comma or space separated
        "00:00.000  00:02.500"                  MM:SS.mmm timecodes
        "Shot 2 | 00:02.500 | 2.583s | dissolve"  the shot table
        "[Shot 2] At 00:02.500, ..."            H3 prompt shot labels
        the cuts_json output, or a JSON list of numbers or objects

    Returns cuts sorted by time, deduplicated to 1 ms, each carrying its
    boundary kind ("hard cut" when the text does not say). An empty or
    unreadable string returns [].

    A hyphen is a range separator, never a minus sign, so the legacy H3
    "[0.000s-3.250s]" form reads as two boundaries. Times past the end
    of the clip are the caller's to drop - this function has no duration
    to check them against.
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()
    if stripped[0] in "[{":
        parsed = _from_json(stripped)
        if parsed:
            return _tidy(parsed)

    cuts: List[ParsedCut] = []
    lines = [ln for ln in stripped.splitlines() if ln.strip()]

    for line in lines:
        # "[Shot 3] At 00:05.083" - the shot number is not a timestamp.
        cleaned = _SHOT_PREFIX_RE.sub(" ", line)
        cleaned = _AT_RE.sub(" ", cleaned)
        tokens = _TIME_RE.findall(cleaned)
        if not tokens:
            continue

        kind = _kind_in(cleaned) or "hard cut"
        # A pipe marks a shot-table row, where only the first time is the
        # shot start and the rest are its duration and transition length.
        # Any other lone line is a list, so every token on it is a cut.
        if len(lines) == 1 and "|" not in line:
            cuts.extend(
                ParsedCut(value, kind)
                for value in (_parse_time_token(t) for t in tokens)
                if value is not None
            )
        else:
            value = _parse_time_token(tokens[0])
            if value is not None:
                cuts.append(ParsedCut(value, kind))

    return _tidy(cuts)


def _tidy(cuts: List[ParsedCut]) -> List[ParsedCut]:
    """Sort, drop negatives, and collapse cuts within 1 ms of each other."""
    ordered = sorted((c for c in cuts if c.time >= 0.0), key=lambda c: c.time)
    tidied: List[ParsedCut] = []
    for cut in ordered:
        if tidied and abs(cut.time - tidied[-1].time) < 0.001:
            continue
        tidied.append(ParsedCut(round(cut.time, 3), cut.kind))
    return tidied
