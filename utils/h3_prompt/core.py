"""
Subject-agnostic helpers shared by both H3 prompt packages.

Extracted verbatim from assembler.py (Phase 4). Everything here treats
the prompt as structure - sections, shot labels, reference tags, the
character budget - and knows nothing about any particular subject,
wardrobe, or task. The subject-specific enforcement stays in
assembler.py, which re-exports every name moved here so nothing that
imported from there breaks; utils/h3_cowboy/assembler.py imports from
here directly.

The golden hashes in tests/test_h3_assembler.py pin the extraction:
byte-identical output or the move was not mechanical.
"""

import re
from typing import Dict, List, Optional, Tuple

from .prompts import (
    MAX_PROMPT_CHARS,
    MIN_EXCLUSIONS,
    RETENTION_MARKERS_AUDIO,
    RETENTION_MARKERS_VISUAL,
    SECTION_ORDER,
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def format_shot_time(seconds: float) -> str:
    """Seconds -> official 'MM:SS.mmm' shot-time form."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def parse_shot_time(text: str) -> float:
    """'MM:SS.mmm' (or bare seconds like '3.25s') -> seconds."""
    text = text.strip().rstrip("s")
    if ":" in text:
        minutes, secs = text.split(":", 1)
        return int(minutes) * 60 + float(secs)
    return float(text)


# ---------------------------------------------------------------------------
# Wrapper / markdown stripping
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*$", re.MULTILINE)
_PREAMBLE_RE = re.compile(
    r"^.*?(?=^\s*(?:#+\s*)?(?:\*\*)?subject[_\s]definitions?)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def strip_wrapper(text: str, fixes: List[str]) -> str:
    """Remove code fences and any commentary before subject_definitions."""
    if _FENCE_RE.search(text):
        text = _FENCE_RE.sub("", text)
        fixes.append("stripped code fences")

    match = _PREAMBLE_RE.match(text)
    if match and match.group(0).strip():
        text = text[match.end():]
        fixes.append("stripped preamble before subject_definitions")
    return text.strip()


def strip_markdown(text: str, fixes: List[str]) -> str:
    """Remove markdown emphasis/headers/bullets, keep sentence text."""
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    cleaned = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.MULTILINE)
    if cleaned != text:
        fixes.append("stripped markdown formatting")
    return cleaned


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

_OFFICIAL_PATTERNS = {
    "subject_definitions": r"subject[_\s]definitions?",
    "summary": r"summary",
    "retention_analysis": r"retention[_\s]analysis",
    "detailed_description": r"detailed[_\s]description",
    "overall_soundscape": r"overall[_\s]soundscape",
    "non_diegetic_music": r"non[-_\s]diegetic[_\s]music",
}
# Legacy top-level blocks from the pre-official template; their content
# gets folded rather than kept as sections.
_LEGACY_PATTERNS = {
    "camera": r"camera",
    "dialogue": r"dialogue",
    "visible_text": r"visible[_\s]text",
    "exclusions": r"exclusions?",
}


def _header_key(line: str) -> str:
    """Return the section key if `line` is a (possibly miscased) header."""
    stripped = line.strip().strip("*").strip()
    stripped = re.sub(r"^#+\s*", "", stripped).rstrip(":").strip()
    for key, pattern in {**_OFFICIAL_PATTERNS, **_LEGACY_PATTERNS}.items():
        if re.fullmatch(pattern, stripped, re.IGNORECASE):
            return key
    return ""


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CHATTER_RE = re.compile(
    r"(?i)^\s*(here('s| is| are)|let me know|hope (this|that|it)|"
    r"feel free|is there anything|would you like|i('ve| have) "
    r"(followed|written|created))"
)


def _split_no_sentences(text: str) -> Tuple[str, List[str]]:
    """
    Split trailing "No ..." exclusion sentences off a section's tail.
    Trailing assistant chatter ("Let me know if...") below the block
    is dropped. Returns (remaining_text, exclusions).
    """
    exclusions: List[str] = []
    lines = text.split("\n")
    # Drop chatter lines from the bottom before collecting exclusions
    while lines and (
        not lines[-1].strip() or _CHATTER_RE.match(lines[-1])
    ):
        lines.pop()
    kept_lines: List[str] = []
    in_tail = True
    for line in reversed(lines):
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(line.strip()) if s.strip()]
        if in_tail and sentences and all(
            re.match(r"(?i)^no[\s,]", s) or re.match(r"(?i)^n0\b", s)
            for s in sentences
        ):
            exclusions = sentences + exclusions
        else:
            in_tail = False
            kept_lines.append(line)
    kept_lines.reverse()
    return "\n".join(kept_lines).strip(), exclusions


def parse_sections(
    text: str, fixes: List[str], warnings: List[str]
) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse VLM text into {official_section: content} plus the exclusions
    list. Miscased/uppercase headers are recognized and recased; legacy
    CAMERA/DIALOGUE/VISIBLE TEXT blocks fold into detailed_description;
    a legacy EXCLUSIONS block feeds the exclusions list.
    """
    sections: Dict[str, List[str]] = {}
    exclusions: List[str] = []
    current = ""
    folded_legacy = []

    for line in text.split("\n"):
        key = _header_key(line)
        if key:
            if key in _LEGACY_PATTERNS:
                folded_legacy.append(key)
                current = "exclusions_legacy" if key == "exclusions" else (
                    "detailed_description__" + key
                )
            else:
                if line.strip() != f"{key}:":
                    fixes.append(f"recased header to {key}:")
                current = key
            continue
        if not current:
            continue
        sections.setdefault(current, []).append(line)

    result: Dict[str, str] = {}
    for key, lines in sections.items():
        content = "\n".join(lines).strip()
        if not content:
            continue
        if key == "exclusions_legacy":
            for sentence in _SENTENCE_SPLIT_RE.split(content.replace("\n", " ")):
                sentence = sentence.strip()
                if sentence:
                    exclusions.append(sentence)
            continue
        if key.startswith("detailed_description__"):
            sub = key.split("__", 1)[1]
            if sub == "dialogue" and re.fullmatch(
                r"(?i)none\.?", content.strip()
            ):
                continue
            if sub == "visible_text" and "none" in content.lower():
                continue
            prefix = {"camera": "Camera: ", "dialogue": "", "visible_text": ""}[sub]
            result["detailed_description"] = (
                result.get("detailed_description", "") + "\n" + prefix + content
            ).strip()
            continue
        result[key] = (result.get(key, "") + "\n" + content).strip()

    if folded_legacy:
        fixes.append(
            "folded legacy sections into official format: "
            + ", ".join(sorted(set(folded_legacy)))
        )

    # Trailing headerless "No ..." sentences live at the end of the last
    # populated section (officially after non_diegetic_music).
    for key in reversed(SECTION_ORDER):
        if key in result:
            remaining, tail = _split_no_sentences(result[key])
            if tail and remaining:
                result[key] = remaining
                exclusions.extend(tail)
            elif tail:
                # Peeling would empty the section, so those sentences
                # ARE the section. H3 requires all six; a legitimate
                # "non_diegetic_music: No score is present." used to be
                # eaten here and the prompt shipped with five.
                fixes.append(
                    f"kept a 'No ...' sentence as {key} rather than "
                    "reading it as a trailing exclusion"
                )
            break

    missing = [k for k in SECTION_ORDER if k not in result]
    for key in missing:
        if key == "non_diegetic_music":
            result[key] = "N/A"
            fixes.append("synthesized non_diegetic_music: N/A")
        elif key == "overall_soundscape":
            result[key] = (
                "Natural environmental ambience matching the visible scene."
            )
            fixes.append("synthesized minimal overall_soundscape")
        else:
            warnings.append(f"section '{key}' missing from VLM output")

    return result, exclusions


_TAG_FIX_RE = re.compile(
    r"[<\[\(]\s*(picture|image|photo|video|subject|audio)\s*(\d+)\s*[>\]\)]",
    re.IGNORECASE,
)
_TAG_CANONICAL = {
    "picture": "Picture", "image": "Picture", "photo": "Picture",
    "video": "Video", "subject": "Subject", "audio": "Audio",
}


def fix_reference_tags(text: str, fixes: List[str]) -> str:
    """Normalize tag variants to <Picture 1> / <Video 1> / <Subject 1>."""
    def repl(m: re.Match) -> str:
        return f"<{_TAG_CANONICAL[m.group(1).lower()]} {m.group(2)}>"

    fixed = _TAG_FIX_RE.sub(repl, text)
    if fixed != text:
        fixes.append("normalized reference tags")
    return fixed


_SHOT_LABEL_RE = re.compile(
    r"\[\s*Shot\s+(\d+)\s*\](?:\s*At\s+([0-9:.]+s?)\s*,?)?", re.IGNORECASE
)
_LEGACY_RANGE_RE = re.compile(
    r"\[\s*(\d+(?:\.\d+)?)\s*s?\s*[–\-—]\s*(\d+(?:\.\d+)?)\s*s?\s*\]"
)


def _apply_known_times(
    times: List[float],
    known: List[float],
    fixes: List[str],
    warnings: List[str],
    retry_errors: List[str],
) -> List[float]:
    """
    Force model shot times onto a measured cut list.

    Matching counts means the model found the right shots and only its
    timings drift, so the measured times replace them outright. A count
    mismatch is a real content error - a merged or invented shot - so it
    becomes a retry error; the times are still snapped to the nearest
    measured cut so an unfixed retry at least lands on real boundaries,
    and the snap is abandoned if it would break monotonicity.
    """
    if len(times) == len(known):
        if any(abs(a - b) > 0.001 for a, b in zip(times, known)):
            fixes.append(
                f"snapped {len(known)} shot times to the measured cut list"
            )
        return list(known)

    labels = f"{len(times)} [Shot N] label{'' if len(times) == 1 else 's'}"
    if len(known) == 1:
        # A single measured shot is a clip with no cuts at all. Naming a
        # count and a list of one time reads as an off-by-one to fix;
        # what the model has to do is delete shots, so say that.
        retry_errors.append(
            f"detailed_description has {labels}, but the shot-boundary "
            "detector found no cuts: the clip is one continuous shot. "
            "Write [Shot 1] once, with no timestamp, and delete every "
            "later [Shot N] label. Describe the whole clip inside that "
            "one shot."
        )
    else:
        retry_errors.append(
            f"detailed_description has {labels} but the measured shot "
            f"list has {len(known)} shots. Write exactly {len(known)} "
            "shots, starting at these times in seconds: "
            + ", ".join(f"{t:.3f}" for t in known)
            + "."
        )

    snapped = [
        min(known, key=lambda k: abs(k - t)) if t == t else t
        for t in times
    ]
    if any(t != t for t in snapped) or any(
        b <= a for a, b in zip(snapped, snapped[1:])
    ):
        warnings.append(
            "shot count disagrees with the measured cut list and the "
            "times could not be snapped without collapsing two shots; "
            "kept the model's own times"
        )
        return times

    warnings.append(
        f"shot count disagrees with the measured cut list "
        f"({len(times)} written vs {len(known)} measured); snapped each "
        "written shot to its nearest measured cut"
    )
    return snapped


def normalize_shot_labels(
    detailed: str, duration: float, fixes: List[str], warnings: List[str],
    known_times: Optional[List[float]] = None,
    retry_errors: Optional[List[str]] = None,
) -> Tuple[str, List[float]]:
    """
    Normalize shot labels to the official form ([Shot 1] bare, then
    [Shot N] At MM:SS.mmm) and repair their times.

    known_times, when given, is a measured cut list that overrides the
    model's timings - see _apply_known_times.

    Returns (fixed_text, shot_start_times) where times[0] is always 0.0.
    """
    text = detailed
    retry_errors = retry_errors if retry_errors is not None else []

    # Legacy "[0.000s-1.250s]" ranges -> official labels using range starts
    if _LEGACY_RANGE_RE.search(text):
        counter = [0]

        def range_repl(m: re.Match) -> str:
            counter[0] += 1
            if counter[0] == 1:
                return "[Shot 1]"
            return f"[Shot {counter[0]}] At {format_shot_time(float(m.group(1)))},"

        text = _LEGACY_RANGE_RE.sub(range_repl, text)
        fixes.append("converted legacy timestamp ranges to [Shot N] labels")

    labels = list(_SHOT_LABEL_RE.finditer(text))
    if not labels:
        if duration > 4.0:
            warnings.append("no [Shot N] labels found in detailed_description")
        return text, [0.0]

    # Collect start times; shot 1 is implicitly 0.
    times: List[float] = []
    for i, m in enumerate(labels):
        if i == 0:
            times.append(0.0)
            continue
        raw = m.group(2)
        try:
            times.append(parse_shot_time(raw) if raw else float("nan"))
        except ValueError:
            times.append(float("nan"))

    # Repair: NaN, non-monotonic, or out-of-range times get proportional
    # placement; otherwise keep the model's values.
    def monotonic_in_range(ts: List[float]) -> bool:
        prev = 0.0
        for t in ts[1:]:
            if not (prev < t < duration):
                return False
            prev = t
        return True

    if known_times:
        times = _apply_known_times(
            times, list(known_times), fixes, warnings, retry_errors
        )
    elif not monotonic_in_range(times):
        n = len(times)
        repaired = [round(duration * i / n, 3) for i in range(n)]
        if any(t != t for t in times):  # NaN present
            fixes.append("filled missing shot times proportionally")
        else:
            warnings.append(
                "shot times were non-monotonic or out of range; "
                "rescaled proportionally onto the clip duration"
            )
        times = repaired

    # Re-emit canonical labels with the (possibly repaired) times.
    counter = [0]

    def label_repl(m: re.Match) -> str:
        counter[0] += 1
        idx = counter[0]
        if idx == 1:
            return "[Shot 1]"
        return f"[Shot {idx}] At {format_shot_time(times[idx - 1])},"

    text = _SHOT_LABEL_RE.sub(label_repl, text)
    return text, times


def _shot_bodies(detailed: str) -> List[str]:
    """Split detailed_description into per-shot text bodies."""
    parts = _SHOT_LABEL_RE.split(detailed)
    # split() yields [pre, num, time, body, num, time, body, ...]
    bodies = []
    for i in range(3, len(parts), 3):
        bodies.append(parts[i] or "")
    return bodies


# The official move vocabulary (guide_base_en section 4.3), plus the
# common non-official spellings a VLM reaches for, so stacking is still
# detected when the model ignores the term list.
_CAMERA_MOVES = re.compile(
    r"(?i)\b(push(?:es|ing)? in|pull(?:s|ing)? (?:out|back)|pans?|panning|"
    r"tilts?|tilting|trucks?|trucking|pedestals?|arcs?|arcing|orbits?|"
    r"rolls?|rolling|zooms?|zooming|dollies|dolly|cranes?|whip[- ]pan|"
    r"tracking shot|shakes? (?:slightly|strongly))\b"
)


def check_camera_moves(detailed: str, warnings: List[str]) -> None:
    for i, body in enumerate(_shot_bodies(detailed) or [detailed], start=1):
        moves = {m.group(1).lower() for m in _CAMERA_MOVES.finditer(body)}
        if len(moves) > 2:
            warnings.append(
                f"shot {i} stacks {len(moves)} camera moves "
                f"({', '.join(sorted(moves))}); one dominant move is safer"
            )


# "<Tag N> (scope): marker - text", the official retention entry shape.
_RETENTION_ENTRY_RE = re.compile(
    r"^\s*<(Subject|Picture|Video|Audio)\s+(\d+)>\s*(\([^)]*\))?\s*:\s*"
    r"([A-Za-z_]+)",
)


def enforce_retention_labels(
    sections: Dict[str, str], fixes: List[str], warnings: List[str],
) -> Dict[str, str]:
    """
    Check every retention_analysis line against the fixed marker sets.

    The guide calls these "fixed English values", and splits them: audio
    tags take fully_copy / partially_copy / reference / weak_reference,
    visible content takes fully_preserved / partially_preserved /
    attribute_transfer / weak_reference. A marker from the wrong set is
    the mistake worth catching - it reads as valid but means nothing for
    that tag. Also drops any (Sx) ID, which the guide forbids here.
    """
    content = sections.get("retention_analysis", "")
    if not content.strip():
        return sections

    seen_bad: List[str] = []
    for line in content.split("\n"):
        found = _RETENTION_ENTRY_RE.match(line)
        if not found:
            continue
        tag, marker = found.group(1).lower(), found.group(4)
        allowed = (
            RETENTION_MARKERS_AUDIO if tag == "audio"
            else RETENTION_MARKERS_VISUAL
        )
        if marker not in allowed:
            seen_bad.append(f"<{found.group(1)} {found.group(2)}>: {marker}")
    if seen_bad:
        warnings.append(
            "retention_analysis used a marker outside the official set "
            f"for its tag: {'; '.join(seen_bad)}. Allowed are "
            f"{', '.join(RETENTION_MARKERS_VISUAL)} for visible content "
            f"and {', '.join(RETENTION_MARKERS_AUDIO)} for audio."
        )

    # "Do not write (Sx) in retention_analysis" - guide section 5.4.
    stripped = re.sub(r"\s*\(S\d+(?:,\s*S\d+)*\)", "", content)
    if stripped != content:
        sections["retention_analysis"] = stripped
        fixes.append("removed (Sx) speaker IDs from retention_analysis")
    return sections


# ---------------------------------------------------------------------------
# Reassembly + trim
# ---------------------------------------------------------------------------

def reassemble(
    sections: Dict[str, str], exclusions: List[str], prefix: str = ""
) -> str:
    """
    Join the sections back into one prompt.

    `prefix` is the first-frame alignment hook, which sits above
    subject_definitions. It is counted here rather than glued on later
    so the trim ladder measures the real prompt length.
    """
    parts = []
    if prefix:
        parts.append(prefix.strip())
    for key in SECTION_ORDER:
        content = sections.get(key, "").strip()
        if not content:
            continue
        parts.append(f"{key}:\n{content}")
    if exclusions:
        parts.append(" ".join(exclusions))
    return "\n\n".join(parts)


def apply_trim_ladder(
    sections: Dict[str, str], exclusions: List[str],
    fixes: List[str], retry_errors: List[str], prefix: str = "",
    music_video: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Reduce the prompt below MAX_PROMPT_CHARS, cheapest cuts first.

    MAX_PROMPT_CHARS is a TrentNodes budget, not a model limit; see
    prompts.py. The first rung shortens the score rather than blanking
    it: "N/A" is the guide's value for "there is no non-diegetic music",
    so writing it to save characters asserts silence that is not true.
    """
    if len(reassemble(sections, exclusions, prefix)) <= MAX_PROMPT_CHARS:
        return sections, exclusions

    music = sections.get("non_diegetic_music", "")
    if len(music) > 60 and not music_video:
        # Keep the first sentence, which carries the instrumentation.
        first = _SENTENCE_SPLIT_RE.split(music.strip())[0].strip()
        if first and len(first) < len(music):
            sections["non_diegetic_music"] = first
            fixes.append(
                "shortened non_diegetic_music to its first sentence "
                "(over char budget)"
            )
    if len(reassemble(sections, exclusions, prefix)) <= MAX_PROMPT_CHARS:
        return sections, exclusions

    if len(exclusions) > MIN_EXCLUSIONS:
        exclusions = sorted(exclusions, key=len)[:MIN_EXCLUSIONS]
        fixes.append("kept only the 8 shortest exclusions (over char cap)")
    if len(reassemble(sections, exclusions, prefix)) <= MAX_PROMPT_CHARS:
        return sections, exclusions

    retry_errors.append(
        "The prompt exceeds 7000 characters. Shorten "
        "detailed_description while keeping every shot."
    )
    return sections, exclusions
