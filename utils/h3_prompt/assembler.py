"""
Assembler / validator for H3 Auto Prompt Generator output.

Pure string functions - no torch, no ComfyUI - so the whole module is
unit-testable offline. Takes raw VLM text, normalizes it to the official
MiniMax H3 REF2VA format, applies deterministic repairs, and reports
what it could not fix as retry errors (fed back to the VLM once) or
warnings (surfaced to the user).

Official format: six lowercase sections, one line per reference label,
a square-bracketed task-type prefix on summary, fixed retention markers,
and nothing after non_diegetic_music. See the prompts.py module
docstring for the source documents and for why the trailing "No ..."
block is opt-in rather than emitted.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .prompts import (
    MAX_PROMPT_CHARS,
    MIN_EXCLUSIONS,
    RETENTION_MARKERS_AUDIO,
    RETENTION_MARKERS_VISUAL,
    SECTION_ORDER,
    STOCK_EXCLUSIONS,
    TASK_TYPES,
    DETAILED_DESCRIPTION_WORDS,
    UPGRADED_DESCRIPTION_WORDS,
)


@dataclass
class AssemblyContext:
    subject_name: str
    subject_wardrobe: str
    duration_seconds: float
    enable_audio_prompt: bool = True
    # "official": trailing "No..." block padded to MIN_EXCLUSIONS.
    # "upgraded": positive assertions inline; no padding, no block target.
    profile: str = "official"
    # Shot start times in seconds from a real shot-boundary detector,
    # starting at 0.0. When present they are ground truth: shot times are
    # forced onto them rather than repaired proportionally, and a shot
    # count that disagrees becomes a retry error. Empty = none measured.
    known_shot_times: List[float] = field(default_factory=list)
    # First-frame alignment sentence (prompts.build_alignment_hook).
    # Prepended above subject_definitions, and it suppresses any
    # exclusion that would forbid using <Picture 1>'s background, pose,
    # or lighting - the hook declares that they ARE the target frame.
    # Empty = no alignment hook.
    alignment_hook: str = ""
    # Music-video mode: non_diegetic_music leads instead of being N/A,
    # and lyrics belong in <d> inside detailed_description only.
    music_video: bool = False
    # Exact sung words, when supplied. Their absence from the output is
    # a retry error in music-video mode.
    lyrics: str = ""
    # Whether the song reaches H3 as <Audio 1>. When music_video is on
    # and this is off, an <Audio 1> the generation never receives is a
    # dangling reference, so it is stripped rather than shipped.
    music_is_reference: bool = False
    # Which picture the alignment hook declares to BE the target frame.
    # "Picture 2" in a hybrid graph, where <Picture 1> stays the identity
    # reference and only the injected first frame is pinned to a moment.
    # The prose repair follows this tag: with two pictures, "<Picture 1>
    # supplies identity only" is correct and must survive.
    alignment_picture: str = "Picture 1"
    # The mandatory summary task-type prefix, without its brackets
    # (prompts.build_task_type). "" disables the check, which only the
    # older tests do.
    task_type: str = "reference generation"
    # Emit the off-spec trailing "No ..." block. Off by default: no
    # official example writes anything after non_diegetic_music. See the
    # prompts module docstring.
    append_exclusions: bool = False


@dataclass
class AssemblyResult:
    prompt: str = ""
    retry_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    applied_fixes: List[str] = field(default_factory=list)
    char_count: int = 0
    detailed_word_count: int = 0


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


# ---------------------------------------------------------------------------
# Tag + subject fixers
# ---------------------------------------------------------------------------

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

    retry_errors.append(
        f"detailed_description has {len(times)} [Shot N] labels but the "
        f"measured shot list has {len(known)} shots. Write exactly "
        f"{len(known)} shots, starting at these times in seconds: "
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


_POSITION_WORDS = re.compile(
    r"(?i)\b(center|centre|left|right|upper|lower|foreground|midground|"
    r"background|quadrant|third|frame[- ]?(?:left|right|center)|occupies|"
    r"positioned|edge of frame|fills the frame|close[- ]up|distance)\b"
)


def enforce_subject_per_shot(
    detailed: str, subject_name: str, fixes: List[str],
    retry_errors: List[str], warnings: List[str],
) -> str:
    """
    Every shot must contain '<name> <Subject 1>' and a frame-position
    cue. Bare name mentions get tagged; a missing name or position is a
    retry error (never fabricated).
    """
    bodies = _shot_bodies(detailed)
    if not bodies:
        bodies = [detailed]

    name_re = re.compile(re.escape(subject_name), re.IGNORECASE)
    tagged_re = re.compile(
        re.escape(subject_name) + r"\s*<Subject 1>", re.IGNORECASE
    )

    fixed_detail = detailed
    for i, body in enumerate(bodies, start=1):
        if not tagged_re.search(body):
            if name_re.search(body):
                new_body = name_re.sub(
                    f"{subject_name} <Subject 1>", body, count=1
                )
                fixed_detail = fixed_detail.replace(body, new_body, 1)
                fixes.append(f"tagged bare subject name in shot {i}")
            else:
                retry_errors.append(
                    f"Shot {i} of detailed_description does not mention "
                    f"{subject_name} <Subject 1>."
                )
        if not _POSITION_WORDS.search(body):
            retry_errors.append(
                f"Shot {i} of detailed_description does not state "
                f"{subject_name}'s position in the frame."
            )
    return fixed_detail


def _wardrobe_items(wardrobe: str) -> List[str]:
    items = [w.strip() for w in re.split(r"[,;]| and | over ", wardrobe)]
    return [i for i in items if len(i) > 2]


def enforce_wardrobe(
    sections: Dict[str, str], wardrobe: str, fixes: List[str],
    warnings: List[str],
) -> None:
    """
    The wardrobe must be named in subject_definitions, in
    retention_analysis, and at least once elsewhere. Inject the full
    phrase into the two structural sections when absent, onto the
    <Subject 1> line - the wardrobe belongs to the subject, and both
    sections are one line per reference label.
    """
    items = _wardrobe_items(wardrobe)
    if not items:
        return

    def mentions(text: str) -> int:
        low = text.lower()
        return sum(1 for item in items if item.lower() in low)

    for key, sentence in (
        ("subject_definitions",
         f" The exact wardrobe from <Picture 1> is: {wardrobe}."),
        ("retention_analysis",
         f" Also fully preserve the wardrobe from <Picture 1>: {wardrobe}."),
    ):
        if key in sections and mentions(sections[key]) == 0:
            sections[key] = _append_to_subject_line(sections[key], sentence)
            fixes.append(f"injected wardrobe into {key}")

    total = sum(mentions(v) for v in sections.values())
    if total < 3:
        warnings.append(
            f"wardrobe items are named only {total} time(s) across the "
            "prompt (target: 3+)"
        )


def _end_sentence(text: str) -> str:
    """Close a sentence before another is appended to the same line."""
    text = text.rstrip()
    if not text or text.endswith((".", "!", "?", ":", ";")):
        return text
    return text + "."


def _append_to_subject_line(content: str, sentence: str) -> str:
    """
    Add a sentence to the <Subject 1> line rather than the section tail.

    Both structural sections are one line per reference label, so
    appending to the end attaches the subject's wardrobe to whichever
    label happens to be last - usually <Video 1>.
    """
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "<Subject 1>" in line:
            lines[i] = _end_sentence(line) + sentence
            return "\n".join(lines)
    lines[-1] = _end_sentence(lines[-1]) + sentence
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------

# An exclusion forbidding <Picture 1>'s background, pose, or lighting
# is correct for character replacement and flatly wrong under a
# first-frame alignment hook, which declares that those ARE the target
# frame. One of the stock lines says exactly this, so it can arrive
# through padding even when the model never wrote it.
_PICTURE_SCENE_WORDS = re.compile(
    r"(?i)\b(background|pose|lighting|framing|camera angle)\b"
)


def _contradicts_alignment(
    sentence: str, picture: str = "Picture 1"
) -> bool:
    return (
        f"<{picture}>" in sentence
        and bool(_PICTURE_SCENE_WORDS.search(sentence))
    )


# The same contradiction reaches the prose sections, and from a stronger
# source: the system prompt's own worked example writes "Do not copy the
# background, pose, or lighting from <Picture 1>" into
# subject_definitions. A task-context instruction is a weak counter to a
# worked example, so these sentences are removed rather than argued with.
# A verb list alone is not enough. "Ignore any camera shake; keep the
# framing from <Picture 1>" matched the old pattern and lost a correct
# sentence, so the negation has to GOVERN the scene word: only the
# clause it opens is searched, and a clause ends at these breaks.
_ALIGNMENT_NEGATION = re.compile(
    r"\b(?:do(?:es)? not|don'?t|never|no|avoid|without)\s+"
    r"(?:copy|copying|use|using|reuse|reusing|take|taking|replicate|"
    r"replicating|reproduce|reproducing|inherit|inheriting|transfer|"
    r"transferring|carry over|carrying over)\b"
    r"|\b(?:ignore|ignoring|disregard|disregarding|discard|discarding)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK_RE = re.compile(
    r"(?i);|\bwhile\b|\bbut\b|\bwhereas\b|\binstead\b|\bhowever\b"
)
# Passive denials read backwards: the scene word comes before the verb.
_ALIGNMENT_PASSIVE = re.compile(
    r"(?i)\b(is|are|must|should|shall)\s+(not|never)\s+(be\s+)?"
    r"(copied|used|reused|taken|replicated|reproduced|inherited|"
    r"transferred|retained|carried over)\b"
)
_ALIGNMENT_ONLY = re.compile(
    r"\b(?:defines?|supplies|supplied|provides?|gives?|contributes?|"
    r"is used)\s+(?:for\s+)?(?:only|solely|just|nothing but)\b"
    r"|\b(?:only|solely|just)\s+(?:defines?|supplies|provides?|gives?|"
    r"contributes?)\b",
    re.IGNORECASE,
)
# A following sentence can carry the denial with a pronoun instead of
# the tag: "<Picture 1> is the identity reference. Its background must
# not be reproduced." The second sentence names no tag at all.
_PICTURE_BACKREF = re.compile(
    r"(?i)\b(its|it|that image|that frame|this image|the reference "
    r"image|the reference)\b"
)
_OTHER_TAG_RE = re.compile(r"<(?:Video|Audio|Subject)\s+\d+>")
_ALIGNMENT_POSITIVE_TEMPLATE = (
    "<{picture}> is the target video frame declared above: that shot "
    "opens on exactly that image, keeping its framing, background, and "
    "lighting."
)


def _conflicts_with_alignment(
    sentence: str, carried: bool = False, picture: str = "Picture 1"
) -> bool:
    """
    True for prose denying the ALIGNED picture its scene contribution.

    `carried` says the previous sentence was about that picture, which
    lets a pronoun follow-on ("Its background must not be copied.") be
    caught. A sentence naming another tag breaks the chain.

    Only the aligned tag is protected. In a hybrid graph <Picture 1> is
    the identity reference, so "<Picture 1> supplies identity only" is
    true and has to survive - stripping it would be the same conflation
    this whole path exists to prevent.
    """
    tag = f"<{picture}>"
    if tag not in sentence:
        if not (carried and _PICTURE_BACKREF.search(sentence)):
            return False
        if _OTHER_TAG_RE.search(sentence):
            return False
        # A back-reference cannot reach past a mention of the other
        # picture: that sentence is about a different image.
        if re.search(r"<Picture \d+>", sentence):
            return False

    if _ALIGNMENT_ONLY.search(sentence):
        return True
    if _ALIGNMENT_PASSIVE.search(sentence):
        # "The background of <Picture 1> is not copied" - the scene word
        # sits before the verb.
        head = sentence[:_ALIGNMENT_PASSIVE.search(sentence).start()]
        if _PICTURE_SCENE_WORDS.search(head):
            return True

    found = _ALIGNMENT_NEGATION.search(sentence)
    if not found:
        return False
    # Only the clause the negation opens counts, so an unrelated
    # "Ignore any camera shake; keep the framing from <Picture 1>"
    # survives instead of being deleted.
    clause = sentence[found.end():]
    stop = _CLAUSE_BREAK_RE.search(clause)
    if stop:
        clause = clause[:stop.start()]
    return bool(_PICTURE_SCENE_WORDS.search(clause))


def enforce_alignment(
    sections: Dict[str, str], ctx: AssemblyContext, fixes: List[str]
) -> Dict[str, str]:
    """
    Make the prose agree with the alignment hook.

    Drops sentences saying the ALIGNED picture contributes identity
    alone or that its background, pose and lighting must not be copied,
    then states the opposite once. A no-op without a hook, and
    byte-identical when nothing conflicts.

    Which picture is aligned comes from ctx.alignment_picture. In a
    hybrid graph that is <Picture 2>, and the identity-only language
    about <Picture 1> is correct and left alone.
    """
    if not ctx.alignment_hook:
        return sections

    picture = ctx.alignment_picture
    tag = f"<{picture}>"
    removed = 0
    for key in ("subject_definitions", "retention_analysis"):
        content = sections.get(key, "")
        if not content:
            continue
        # Split by line first. Both sections are one line per reference
        # label in the official format, and joining everything with " "
        # collapsed that into a paragraph every time a sentence went.
        out_lines = []
        dropped_here = 0
        for line in content.split("\n"):
            sentences = _SENTENCE_SPLIT_RE.split(line)
            kept = []
            carried = False
            for sentence in sentences:
                if _conflicts_with_alignment(sentence, carried, picture):
                    dropped_here += 1
                    carried = True     # a pronoun may follow the denial
                    continue
                kept.append(sentence)
                carried = tag in sentence
            joined = " ".join(s.strip() for s in kept).strip()
            # A line that emptied out was nothing but denials; dropping
            # it is right, and keeps a blank line out of the section.
            if joined:
                out_lines.append(joined)
        if dropped_here:
            removed += dropped_here
            sections[key] = "\n".join(out_lines).strip()

    if removed:
        fixes.append(
            f"removed {removed} sentence(s) denying {tag} its "
            "framing, background, or lighting (alignment hook is on)"
        )

    positive = _ALIGNMENT_POSITIVE_TEMPLATE.format(picture=picture)
    content = sections.get("subject_definitions", "")
    if content and positive not in content:
        lines = content.split("\n")
        owner = next(
            (i for i, line in enumerate(lines) if line.startswith(tag)), None
        )
        if owner is None:
            # The aligned picture has no line yet, and it is a label of
            # its own - one line per label, so it gets one rather than
            # being tacked onto whichever label happens to be last.
            lines.append(positive)
        else:
            lines[owner] = _end_sentence(lines[owner]) + " " + positive
        sections["subject_definitions"] = "\n".join(lines)
        fixes.append(f"stated the {tag} frame alignment positively")
    return sections


# ---------------------------------------------------------------------------
# Official summary prefix and retention markers
# ---------------------------------------------------------------------------

_TASK_PREFIX_RE = re.compile(r"^\s*\[([^\]]+)\]")


def enforce_task_type(
    sections: Dict[str, str], ctx: AssemblyContext,
    fixes: List[str], warnings: List[str],
) -> Dict[str, str]:
    """
    Give summary its mandatory square-bracketed task-type prefix.

    The guide is explicit that summary "begins with a square-bracketed
    task-type prefix", and the prefix is drawn from a fixed vocabulary.
    The model is told to copy the one in the task context; this repairs
    the run where it did not, because a missing prefix is a format
    error, not a wording preference.
    """
    if not ctx.task_type:
        return sections
    summary = sections.get("summary", "").strip()
    if not summary:
        return sections

    wanted = f"[{ctx.task_type}]"
    found = _TASK_PREFIX_RE.match(summary)
    if not found:
        sections["summary"] = f"{wanted} {summary}"
        fixes.append(f"added the mandatory summary task-type prefix {wanted}")
        return sections

    written = [t.strip() for t in found.group(1).split("+")]
    if [t.lower() for t in written] == ctx.task_type.split(" + "):
        return sections

    # The node derives the prefix from what the run actually uses - a
    # picture pinned to a frame, a reused song - so it knows things the
    # model cannot see. Its value wins even when the model wrote a
    # different but individually-legal one.
    unknown = [t for t in written if t.lower() not in TASK_TYPES]
    if unknown:
        warnings.append(
            "summary task-type prefix used a value outside the official "
            f"list: {', '.join(unknown)}; replaced with {wanted}"
        )
    else:
        fixes.append(
            f"corrected the summary task-type prefix to {wanted} "
            f"(model wrote [{found.group(1)}])"
        )
    sections["summary"] = f"{wanted} {summary[found.end():].strip()}".strip()
    return sections


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


_DIALOGUE_RE = re.compile(r"<d>.*?</d>", re.DOTALL)
_NA_RE = re.compile(r"(?i)^\s*n/?a\b")


def enforce_music_video(
    sections: Dict[str, str], ctx: AssemblyContext,
    fixes: List[str], retry_errors: List[str], warnings: List[str],
) -> Dict[str, str]:
    """
    Hold music-video mode to the official guide's audio rules.

    Three checks, all from VIDEO_PROMPT_WRITING_GUIDE_ref_en section 6:
    a music video's score is never "N/A"; supplied lyrics must actually
    reach a <d> block; and lyrics are written only inside <d> in
    detailed_description, never repeated in the two audio sections.
    """
    if not ctx.music_video:
        return sections

    music = sections.get("non_diegetic_music", "").strip()
    if not music or _NA_RE.match(music):
        retry_errors.append(
            "Music video mode: non_diegetic_music is 'N/A' but a music "
            "video is scored wall to wall. Describe the track - genre, "
            "instrumentation, tempo, and how it develops across the "
            "clip - or state that the supplied score is reused directly."
        )

    detailed = sections.get("detailed_description", "")
    if ctx.lyrics.strip() and not _DIALOGUE_RE.search(detailed):
        retry_errors.append(
            "Music video mode: lyrics were supplied but "
            "detailed_description contains no <d>[English] ...</d> "
            "block. Put the sung words inside <d> in the shot where "
            "they are heard."
        )

    # Lyrics repeated in the audio sections: strip, do not argue.
    stripped = 0
    for key in ("overall_soundscape", "non_diegetic_music"):
        content = sections.get(key, "")
        if not content or not _DIALOGUE_RE.search(content):
            continue
        cleaned = _DIALOGUE_RE.sub("", content)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if cleaned:
            sections[key] = cleaned
            stripped += 1
        else:
            warnings.append(
                f"section '{key}' was nothing but lyrics; left as written"
            )
    if stripped:
        fixes.append(
            f"removed lyrics repeated in {stripped} audio section(s); "
            "they belong only inside <d> in detailed_description"
        )
    return sections


def finalize_exclusions(
    exclusions: List[str], ctx: AssemblyContext, fixes: List[str]
) -> List[str]:
    """
    Clean and dedupe the exclusion sentences. The official profile pads
    to MIN_EXCLUSIONS from the stock pool; the upgraded profile keeps
    whatever the model wrote (constraints live inline as positive
    assertions there).

    With an alignment hook in play, exclusions that would forbid copying
    <Picture 1>'s background, pose, or lighting are dropped and never
    padded back in.
    """
    aligning = bool(ctx.alignment_hook)
    dropped = 0

    cleaned: List[str] = []
    seen = set()
    for raw in exclusions:
        sentence = re.sub(r"^\s*[-*•]\s*", "", raw).strip()
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence += "."
        if aligning and _contradicts_alignment(
                sentence, ctx.alignment_picture):
            dropped += 1
            continue
        key = re.sub(r"\W+", "", sentence.lower())
        if key not in seen:
            seen.add(key)
            cleaned.append(sentence)

    if dropped:
        fixes.append(
            f"dropped {dropped} exclusion(s) that contradicted the "
            "first-frame alignment hook"
        )

    if ctx.profile == "upgraded":
        return cleaned

    if len(cleaned) < MIN_EXCLUSIONS:
        for stock in STOCK_EXCLUSIONS:
            sentence = stock.replace("the subject", ctx.subject_name)
            if aligning and _contradicts_alignment(
                    sentence, ctx.alignment_picture):
                continue
            key = re.sub(r"\W+", "", sentence.lower())
            # Skip stock lines that overlap an existing exclusion's topic
            if key in seen:
                continue
            cleaned.append(sentence)
            seen.add(key)
            if len(cleaned) >= MIN_EXCLUSIONS:
                break
        fixes.append("padded exclusions from stock pool")
    return cleaned


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


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def process(raw_text: str, ctx: AssemblyContext) -> AssemblyResult:
    """Run the full normalize -> fix -> validate pipeline on VLM text."""
    result = AssemblyResult()
    fixes, warnings, retry_errors = (
        result.applied_fixes, result.warnings, result.retry_errors
    )

    # A silent music video is a contradiction. Music-video mode wins:
    # nobody turns it on wanting the audio sections blanked.
    silent = not ctx.enable_audio_prompt and not ctx.music_video
    if not ctx.enable_audio_prompt and ctx.music_video:
        warnings.append(
            "music video mode overrides enable_audio_prompt; the audio "
            "sections are written rather than blanked"
        )

    text = strip_wrapper(raw_text or "", fixes)
    text = strip_markdown(text, fixes)
    text = fix_reference_tags(text, fixes)

    if not text or "subject" not in text.lower():
        retry_errors.append(
            "Output did not contain a subject_definitions section."
        )
        result.prompt = text
        result.char_count = len(text)
        return result

    sections, exclusions = parse_sections(text, fixes, warnings)
    for key in ("summary", "retention_analysis", "detailed_description"):
        if key not in sections:
            retry_errors.append(f"Required section '{key}' is missing.")

    if "detailed_description" in sections:
        detailed, _times = normalize_shot_labels(
            sections["detailed_description"], ctx.duration_seconds,
            fixes, warnings,
            known_times=ctx.known_shot_times,
            retry_errors=retry_errors,
        )
        detailed = enforce_subject_per_shot(
            detailed, ctx.subject_name, fixes, retry_errors, warnings
        )
        check_camera_moves(detailed, warnings)
        if silent:
            stripped = re.sub(r"<d>.*?</d>", "", detailed, flags=re.DOTALL)
            if stripped != detailed:
                fixes.append("removed dialogue lines (audio prompt disabled)")
            detailed = stripped
        sections["detailed_description"] = detailed

        words = len(sections["detailed_description"].split())
        result.detailed_word_count = words
        lo, hi = (
            UPGRADED_DESCRIPTION_WORDS if ctx.profile == "upgraded"
            else DETAILED_DESCRIPTION_WORDS
        )
        if words and (words < lo * 0.6 or words > hi * 1.4):
            warnings.append(
                f"detailed_description is {words} words "
                f"({ctx.profile} target {lo}-{hi})"
            )

    if silent:
        sections["overall_soundscape"] = (
            "Quiet natural ambience only. No dialogue is spoken."
        )
        sections["non_diegetic_music"] = "N/A"
        fixes.append("forced minimal audio sections (audio prompt disabled)")

    enforce_wardrobe(sections, ctx.subject_wardrobe, fixes, warnings)
    sections = enforce_alignment(sections, ctx, fixes)
    sections = enforce_task_type(sections, ctx, fixes, warnings)
    sections = enforce_retention_labels(sections, fixes, warnings)

    # The upgraded profile tells the model to write constraints as
    # positive assertions, so a No-block means it ignored that.
    if ctx.profile == "upgraded" and len(exclusions) > 3:
        warnings.append(
            f"upgraded profile expects inline positive assertions but "
            f"the model wrote {len(exclusions)} trailing 'No...' sentences"
        )

    if ctx.append_exclusions:
        exclusions = finalize_exclusions(exclusions, ctx, fixes)
    elif exclusions:
        # Off-spec: nothing follows non_diegetic_music in any official
        # example. Dropped from the prompt, not from the record -
        # applied_fixes says how many.
        fixes.append(
            f"dropped {len(exclusions)} trailing 'No ...' sentence(s); "
            "the official format ends at non_diegetic_music "
            "(set append_exclusions to keep them)"
        )
        exclusions = []

    # The ladder runs before the music-video validator so the validator
    # judges the final text. The other order let the ladder re-introduce
    # exactly the state the validator had just rejected.
    sections, exclusions = apply_trim_ladder(
        sections, exclusions, fixes, retry_errors, ctx.alignment_hook,
        music_video=ctx.music_video,
    )
    sections = enforce_music_video(
        sections, ctx, fixes, retry_errors, warnings
    )

    result.prompt = reassemble(sections, exclusions, ctx.alignment_hook)
    result.char_count = len(result.prompt)
    if result.char_count > MAX_PROMPT_CHARS:
        warnings.append(
            f"prompt is {result.char_count} chars (TrentNodes budget "
            f"{MAX_PROMPT_CHARS}; MiniMax states no character limit)"
        )
    return result
