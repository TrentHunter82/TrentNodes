"""
Deterministic, read-only validation of H3 prompts against the
h3-prompting skill's review checklist.

validate() returns human-readable error strings and NEVER mutates the
prompt - the node feeds failures back to the LLM for one corrective
pass instead of silently rewriting (the old assembler's approach).

Constants come from the existing spec modules so one canonical copy of
each fixed vocabulary exists; only constants are imported, no assembler
or backend code runs.
"""

import re
from typing import List, Optional, Tuple

from ..h3_prompt.prompts import (
    RETENTION_MARKERS_AUDIO,
    RETENTION_MARKERS_VISUAL,
    SECTION_ORDER,
    TASK_TYPES,
)
from ..h3_cowboy.assembler import CUT_PHRASES
from ..h3_cowboy.spec import EDIT_SUMMARY_OPENER, render_instruction_line
from .skill_loader import COWBOY_MODE

BASE_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

LABEL_RE = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")
SHOT_RE = re.compile(r"\[Shot (\d+)\](?:( At (\d{1,2}):(\d{2})\.(\d{1,3})),)?")
# The (scope) is optional: MiniMax's own ref example writes
# "<Audio 1>: reference - its vocal timbre guides the dialogue".
RETENTION_LINE_RE = re.compile(
    r"^<(Subject|Picture|Video|Audio) (\d+)>\s*(?:\(([^)]*)\)\s*)?:\s*(\w+)\s*-\s*(.+)$"
)
SPEAKER_RE = re.compile(r"\(S\d+(?:,\s*S\d+)*\)")
# Loose by design: MiniMax's own base example writes "(S1) places a
# fresh loaf on the wooden counter and says: <d>[English] ..." - the
# speaker ID sits earlier in the clause and "says" takes a colon. So
# the check only requires: an (Sx) ID, then "says", then <d>[Language]
# within the same clause.
DIALOGUE_LINE_RE = re.compile(
    r"\(S\d+(?:,\s*S\d+)*\)[^<]*\bsays\b[^<]*<d>\[[^\]]+\]"
)

# Words in the prompt body that reveal an off-spec claim.
BANNED_WORD_RE = re.compile(r"\b(fps|frames per second|aspect ratio|resolution)\b", re.I)
EXCLUSION_LIST_RE = re.compile(r"\bNo [a-z][^.]*,\s*no ", re.I)
# The pipeline-rendered alignment line, specifically - NOT ordinary prose
# like "her pose aligns with the window frame".
ALIGNMENT_LINE_RE = re.compile(
    r"aligns with the \d+(?:\.\d+)?-second mark|is fully referenced"
)
AMPLITUDE_RE = re.compile(r"with (\w+) amplitude")
SPEED_RE = re.compile(r"at (\w+) speed")


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()])


def _shot_time(match: re.Match) -> Optional[float]:
    if match.group(2) is None:
        return None
    minutes, seconds, millis = match.group(3), match.group(4), match.group(5)
    return int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000.0


def final_shot_index(text: str) -> int:
    numbers = [int(m.group(1)) for m in SHOT_RE.finditer(text)]
    return max(numbers) if numbers else 1


def _check_shots(body: str, duration_s: Optional[float], errors: List[str]) -> None:
    matches = list(SHOT_RE.finditer(body))
    if not matches:
        errors.append("checklist 6: no [Shot 1] label found in the description.")
        return
    numbers = [int(m.group(1)) for m in matches]
    if numbers[0] != 1:
        errors.append("checklist 6: the first shot label must be [Shot 1].")
    if numbers != list(range(1, len(numbers) + 1)):
        errors.append(
            f"checklist 6: shot numbers must run 1..N without gaps or "
            f"repeats (found {numbers})."
        )
    previous_time = -1.0
    for match in matches:
        number = int(match.group(1))
        time_s = _shot_time(match)
        if number == 1:
            if time_s is not None:
                errors.append("checklist 6: [Shot 1] must carry no timestamp.")
            continue
        if time_s is None:
            errors.append(
                f"checklist 6: [Shot {number}] must be written as "
                f"'[Shot {number}] At MM:SS.mmm,'."
            )
            continue
        if time_s <= previous_time:
            errors.append(
                f"checklist 6: [Shot {number}] timestamp must be strictly "
                f"after the previous shot's."
            )
        previous_time = max(previous_time, time_s)
        if duration_s is not None and time_s >= duration_s:
            errors.append(
                f"checklist 6: [Shot {number}] starts at {time_s:.3f} s, "
                f"outside the {duration_s:.2f} s clip."
            )
        following = body[match.end():].lstrip()
        if not any(
            following.lower().startswith(phrase) for phrase in CUT_PHRASES
        ):
            errors.append(
                f"checklist 6: [Shot {number}] must OPEN with one of the "
                f"five cut phrases ({', '.join(CUT_PHRASES)})."
            )
    if "CUT TO:" in body:
        errors.append("checklist 6: 'CUT TO:' is banned; use the five cut phrases.")


def _check_common(text: str, errors: List[str]) -> None:
    if "```" in text:
        errors.append("checklist 2: markdown code fences are not allowed.")
    if re.search(r"^\s*#", text, re.M):
        errors.append("checklist 2: markdown headings are not allowed.")
    if "**" in text:
        errors.append("checklist 2: markdown bold markers are not allowed.")
    if BANNED_WORD_RE.search(text):
        errors.append(
            "checklist 2: duration/fps/aspect-ratio/resolution claims do "
            "not belong in the prompt."
        )
    if EXCLUSION_LIST_RE.search(text) or re.search(r"\bwatermark", text, re.I):
        errors.append(
            "checklist 2: no trailing exclusion list; state constraints as "
            "positive assertions instead."
        )
    for match in AMPLITUDE_RE.finditer(text):
        if match.group(1) not in ("small", "large"):
            errors.append(
                f"checklist 7: '{match.group(0)}' - the fixed phrases are "
                "'with small amplitude' / 'with large amplitude'."
            )
    for match in SPEED_RE.finditer(text):
        if match.group(1) not in ("slow", "fast"):
            errors.append(
                f"checklist 7: '{match.group(0)}' - the fixed phrases are "
                "'at slow speed' / 'at fast speed'."
            )
    for line in text.splitlines():
        # Minimal invariants only. The official example varies the speech
        # verb (says / exclaims / replies) and interleaves other tags, so
        # the deterministic check is: an (Sx) ID earlier on the line, a
        # [Language] tag right after <d>, and a closing </d>.
        for match in re.finditer(r"<d>", line):
            before = line[: match.start()]
            after = line[match.start():]
            # A voice living only inside a reused track belongs to
            # <Audio N> and gets no (Sx) - lyrics are legal without one.
            if not SPEAKER_RE.search(before) and "<Audio" not in before:
                errors.append(
                    "checklist 8: dialogue needs an (Sx) speaker ID (or an "
                    "<Audio N> attribution for sung lyrics) before the <d> "
                    f"tag (offending line: {line.strip()[:80]})"
                )
                break
            if not re.match(r"<d>\[[^\]]+\]", after):
                errors.append(
                    "checklist 8: <d> must open with a [Language] tag, "
                    "e.g. <d>[English] ..."
                )
                break
        if "<d>" in line and "</d>" not in line:
            errors.append("checklist 8: a <d> tag is not closed on its line.")


def _split_ref_sections(text: str) -> Tuple[dict, List[str]]:
    errors: List[str] = []
    lines = text.strip().splitlines()
    header_set = set(SECTION_ORDER)
    positions = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in header_set:
            positions.append((stripped[:-1], index))
    found_order = [name for name, _ in positions]
    if found_order != list(SECTION_ORDER):
        errors.append(
            "checklist 1: the six lowercase headers must each sit alone on "
            f"their line, in order {', '.join(SECTION_ORDER)} "
            f"(found: {', '.join(found_order) or 'none'})."
        )
        return {}, errors
    if positions[0][1] != 0:
        errors.append(
            "checklist 2: nothing may appear before 'subject_definitions:'."
        )
    sections = {}
    for (name, start), (_, end) in zip(
        positions, positions[1:] + [("", len(lines))]
    ):
        sections[name] = "\n".join(lines[start + 1 : end]).strip()
    return sections, errors


def _validate_ref(text: str, duration_s: Optional[float]) -> List[str]:
    sections, errors = _split_ref_sections(text)
    _check_common(text, errors)
    if not sections:
        return errors

    defined = []
    for line in sections["subject_definitions"].splitlines():
        line = line.strip()
        if not line:
            continue
        match = LABEL_RE.match(line)
        if match is None:
            errors.append(
                "checklist 3: every subject_definitions line must start "
                f"with a label like <Subject 1> (offending: {line[:60]})"
            )
        else:
            defined.append(match.group(0))

    referenced = {m.group(0) for m in LABEL_RE.finditer(text)}
    undefined = sorted(
        label for label in referenced
        if label not in defined
        and not any(label == m.group(0) for m in LABEL_RE.finditer(
            sections["subject_definitions"]))
    )
    if undefined:
        errors.append(
            "checklist 3: labels used but never defined in "
            f"subject_definitions: {', '.join(undefined)}."
        )

    summary = sections["summary"]
    if not summary.startswith("["):
        errors.append(
            "checklist 4: summary must begin with a bracketed task-type "
            "prefix, e.g. [reference generation]."
        )
    else:
        prefix = summary[1 : summary.find("]")] if "]" in summary else ""
        tokens = prefix.split(" + ")  # the separator IS " + ", exactly
        illegal = [token for token in tokens if token not in TASK_TYPES]
        if illegal:
            errors.append(
                f"checklist 4: illegal task type(s) {', '.join(illegal)}; "
                f"legal values: {', '.join(TASK_TYPES)}, combined with "
                "' + ' (space, plus, space)."
            )
        if len(tokens) != len(set(tokens)):
            errors.append("checklist 4: task types must not repeat.")
        if "video editing" in tokens:
            rest = summary[summary.find("]") + 1:].strip()
            if not rest.startswith(EDIT_SUMMARY_OPENER):
                errors.append(
                    "checklist 4: an editing summary opens (after the "
                    f"prefix) with the fixed sentence '{EDIT_SUMMARY_OPENER}'"
                )

    retention_labels = []
    for line in sections["retention_analysis"].splitlines():
        line = line.strip()
        if not line:
            continue
        match = RETENTION_LINE_RE.match(line)
        if match is None:
            errors.append(
                "checklist 5: retention lines must be shaped "
                "'<Tag N> (scope): marker - explanation' "
                f"(offending: {line[:60]})"
            )
            continue
        retention_labels.append(f"<{match.group(1)} {match.group(2)}>")
        # Visual tags carry a (scope); only the official Audio example
        # omits it, so audio stays optional.
        if match.group(1) != "Audio" and match.group(3) is None:
            errors.append(
                f"checklist 5: <{match.group(1)} {match.group(2)}> needs a "
                "(scope) before the colon, e.g. (appears in [Shot 1])."
            )
        marker = match.group(4)
        legal = (
            RETENTION_MARKERS_AUDIO
            if match.group(1) == "Audio"
            else RETENTION_MARKERS_VISUAL
        )
        if marker not in legal:
            errors.append(
                f"checklist 5: '{marker}' is not a legal "
                f"{'audio' if match.group(1) == 'Audio' else 'visual'} "
                f"marker for {match.group(0)[:20]}; legal: {', '.join(legal)}."
            )
    if SPEAKER_RE.search(sections["retention_analysis"]):
        errors.append("checklist 5: no (Sx) speaker IDs in retention_analysis.")
    missing = [label for label in defined if label not in retention_labels]
    extra = [label for label in retention_labels if label not in defined]
    if missing:
        errors.append(
            f"checklist 5: defined labels missing a retention line: "
            f"{', '.join(missing)}."
        )
    if extra:
        errors.append(
            f"checklist 5: retention lines for undefined labels: "
            f"{', '.join(extra)}."
        )

    _check_shots(sections["detailed_description"], duration_s, errors)
    words = len(sections["detailed_description"].split())
    if words < 90 or words > 560:
        errors.append(
            f"checklist 6: detailed_description is {words} words; target "
            "350-500 (a tight 120-300 profile is acceptable)."
        )

    soundscape = sections["overall_soundscape"]
    if soundscape != "N/A" and not 1 <= _sentences(soundscape) <= 5:
        errors.append("checklist 9: overall_soundscape budget is 1-4 sentences.")
    music = sections["non_diegetic_music"]
    if music != "N/A" and not 1 <= _sentences(music) <= 4:
        errors.append(
            "checklist 9: non_diegetic_music budget is 1-3 sentences or "
            "exactly N/A."
        )
    return errors


def _validate_base(text: str, mode: str, duration_s: Optional[float]) -> List[str]:
    errors: List[str] = []
    _check_common(text, errors)
    body = text.strip()

    if not body.startswith(BASE_FIELDS[0] + ": "):
        errors.append(
            "checklist 1: base output must start with "
            "'integrated_multimodal_description: ' with content on the "
            "SAME line."
        )
    for field in BASE_FIELDS:
        occurrences = len(
            re.findall(rf"^{field}: \S", body, re.M)
        )
        if occurrences != 1:
            errors.append(
                f"checklist 1: exactly one '{field}: ' field with content "
                f"on the same line is required (found {occurrences})."
            )
    order = [
        field for field in re.findall(
            rf"^({'|'.join(BASE_FIELDS)}):", body, re.M
        )
    ]
    if order != list(BASE_FIELDS):
        errors.append(
            "checklist 1: base fields must appear in the order "
            + ", ".join(BASE_FIELDS) + "."
        )
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if index > 0 and re.match(
            rf"^({BASE_FIELDS[1]}|{BASE_FIELDS[2]}): ", line
        ) and lines[index - 1].strip():
            errors.append(
                f"checklist 1: a blank line must separate the fields "
                f"(missing before '{line.split(':')[0]}:')."
            )

    for banned, message in (
        ("<Subject", "checklist 3: base mode has no <Subject N> labels."),
        ("<Video", "checklist 3: base mode has no <Video N> labels."),
        ("<Audio", "checklist 3: base mode has no <Audio N> labels."),
        ("subject_definitions:", "checklist 1: base mode has no ref sections."),
        ("retention_analysis:", "checklist 1: base mode has no ref sections."),
    ):
        if banned in body:
            errors.append(message)

    if ALIGNMENT_LINE_RE.search(body):
        errors.append(
            "checklist 2: never write the picture-alignment line; the "
            "pipeline prepends it."
        )

    if mode == "t2va" and re.search(r"<?Picture \d", body):
        errors.append("checklist 3: t2va has no reference pictures.")
    if mode == "fl2va" and "<Picture" in body:
        errors.append(
            "checklist 3: fl2va writes BARE 'Picture 1' - no angle brackets."
        )
    if mode in ("i2va", "l2va") and re.search(r"(?<!<)\bPicture \d+\b", body):
        errors.append(
            f"checklist 3: {mode} brackets its picture references: "
            "'<Picture 1>'."
        )

    imd_match = re.search(
        rf"^{BASE_FIELDS[0]}: (.*?)(?=^\w+:|\Z)", body, re.M | re.S
    )
    imd = imd_match.group(1).strip() if imd_match else body
    _check_shots(imd, duration_s, errors)

    music_match = re.search(rf"^{BASE_FIELDS[2]}: (.+)$", body, re.M)
    if music_match:
        music = music_match.group(1).strip()
        if music != "N/A" and not 1 <= _sentences(music) <= 4:
            errors.append(
                "checklist 9: non_diegetic_music budget is 1-3 sentences "
                "or exactly N/A."
            )
    return errors


def validate(text: str, mode: str, duration_s: Optional[float] = None) -> List[str]:
    """All checklist violations in `text` for `mode`; empty = pass."""
    if not text or not text.strip():
        return ["the model returned an empty prompt."]
    if mode == "ref2va":
        return _validate_ref(text, duration_s)
    return _validate_base(text, mode, duration_s)


def assemble_final(text: str, mode: str, duration_s: float) -> str:
    """
    Pipeline-side finish: prepend the mode's alignment instruction line
    (rendered AFTER the body exists, per guide_base 2.1). The body text
    itself is never modified.
    """
    body = text.strip()
    if mode == "ref2va":
        return body
    line = render_instruction_line(
        COWBOY_MODE[mode],
        final_shot=final_shot_index(body),
        duration_seconds=duration_s,
    )
    if not line:  # t2va
        return body
    return line + "\n\n" + body
