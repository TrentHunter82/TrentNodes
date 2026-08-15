"""
Assembler / validator for Ultimate H3 Cowboy Promptor output.

Takes raw VLM text, normalizes it to the official format, applies
deterministic repairs, and reports what it could not fix.

Two formats, and process() branches on ctx.mode at the top. Ref mode is
six sections with a header on its own line; base mode is three fields
with the content on the SAME line as the header, and no reference labels
at all. They share the retry budget, the shot-label repair and the
verbatim-dialogue check, and nothing else - see the base block at the
bottom of this file for why the parse and the emit could not be shared.

The headline difference from utils/h3_prompt/assembler.py is the retry
budget. That module emits a retry error PER SHOT, twice - a missing
subject mention and a missing frame position - so a six-shot clip can
produce twelve, and a retry costs a real API call. Worse, most of those
checks contradict the spec: MiniMax's own complete example
(guide_ref section 7) produces three of them, because its <Subject 2>
legitimately does not appear in every shot.

Here there are five aggregated classes, one message each, every instance
enumerated inside it:

    R1 FORMAT     unparseable, or a required section is missing
    R2 LABELS     every dangling declaration, undeclared use, missing
                  retention line and nonexistent asset index, in one place
    R3 VERBATIM   supplied dialogue or lyrics absent from any <d>
    R4 STRUCTURE  measured shot count mismatch - only when the run
                  declares video editing (Phase 2)
    R5 EMPTY      no usable text at all

Everything else is a warning, and warnings never trigger a retry. See
docs/H3_COWBOY_HANDOFF.md for the list of things deliberately NOT
validated, each with the guide line that forbids checking it.

The subject-agnostic machinery is imported from the older package rather
than copied, so a fix lands once. Phase 4 moves those helpers to a shared
core; tests/test_h3_assembler.py holds golden hashes that prove the move
changed nothing.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..h3_prompt.core import (
    _FENCE_RE,
    _SENTENCE_SPLIT_RE,
    _SHOT_LABEL_RE,
    _shot_bodies,
    apply_trim_ladder,
    check_camera_moves,
    enforce_retention_labels,
    fix_reference_tags,
    normalize_shot_labels,
    parse_sections,
    reassemble,
    strip_markdown,
    strip_wrapper,
)
from ..h3_prompt.prompts import MAX_PROMPT_CHARS
from .spec import (
    MODES,
    TASK_TYPES,
    BASE_SECTION_ORDER,
    EDIT_SUMMARY_OPENER,
    EDIT_VIDEO_DEFINITION,
    EDIT_VIDEO_RETENTION,
    REF_SECTION_ORDER,
    render_instruction_line,
)
from .subjects import SubjectSpec

# A subject's position in the frame, asked for at its first clear
# appearance (guide_ref 5.3) - and only for kinds that HAVE a position.
_POSITION_WORDS = re.compile(
    r"(?i)\b(center|centre|left|right|foreground|background|midground|"
    r"upper|lower|top|bottom|third|quadrant|frame|beside|behind|"
    r"in front of|near|edge|corner|off-?screen)\b"
)
_ANY_TAG_RE = re.compile(r"<(Subject|Picture|Video|Audio)\s+(\d+)>")
_LINE_LABEL_RE = re.compile(r"^\s*<(Subject|Picture|Video|Audio)\s+(\d+)>")

BASE_MODES = tuple(m for m in MODES if m != "ref")


def _line_opening_labels(section: str) -> set:
    """Labels that OPEN a line - i.e. that the section actually defines."""
    return {
        f"{m.group(1)} {m.group(2)}"
        for line in section.split("\n")
        for m in [_LINE_LABEL_RE.match(line)] if m
    }


@dataclass
class CowboyContext:
    """Everything the assembler needs that is not in the raw text."""
    subjects: List[SubjectSpec] = field(default_factory=list)
    duration_seconds: float = 5.0
    task_type: str = "reference generation"
    mode: str = "ref"
    # Shot start times from a real detector, starting at 0.0. Ground
    # truth for the labels; a count mismatch is only a retry when the run
    # edits the source video (Phase 2), because the target's structure is
    # not otherwise bound to a reference video's.
    known_shot_times: List[float] = field(default_factory=list)
    is_editing: bool = False
    # Exact words the user supplied. Their absence from any <d> is R3.
    dialogue_text: str = ""
    lyrics: str = ""
    # Music-video mode (guide_ref 6). The score leads instead of being
    # "N/A", and a lyric repeated in an audio section is stripped rather
    # than argued with - it belongs only inside <d>.
    music_video: bool = False
    # How many images were actually wired, so a citation of <Picture 9>
    # can be caught.
    wired_pictures: int = 0
    has_video: bool = False
    has_audio: bool = False
    # -- base mode only ---------------------------------------------------
    # guide_base 3.2: "FL2VA generally favors a single shot... Use
    # multiple shots only when they are explicitly specified." True when
    # the user did specify them, which turns the multi-shot warning off.
    multi_shot_requested: bool = False
    # Rewrite FL2VA's bare `Picture 1` / `Shot 1` into the bracketed
    # forms. OFF by default, because bare is what MiniMax's own guide
    # writes in both the instruction line and Case 3's body. It exists
    # only so the two spellings can be A/B'd against real generations -
    # nothing in the format can tell us which is better. See
    # docs/H3_COWBOY_BASE_MODE_HANDOFF.md section 3.2.
    fl2va_normalize_picture_tags: bool = False


@dataclass
class CowboyResult:
    prompt: str = ""
    retry_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    applied_fixes: List[str] = field(default_factory=list)
    char_count: int = 0
    description_word_count: int = 0


def process(raw_text: str, ctx: CowboyContext) -> CowboyResult:
    """
    Normalize, repair and validate one VLM response.

    Branches at the top, and has to. Base output run through the ref path
    would be reported as four missing sections, and parse_sections would
    synthesize overall_soundscape and non_diegetic_music on top of the
    ones already there.
    """
    if ctx.mode == "ref":
        return _process_ref(raw_text, ctx)
    if ctx.mode in BASE_MODES:
        return _process_base(raw_text, ctx)
    raise ValueError(
        f"unknown h3 mode '{ctx.mode}'; legal values are ref, "
        + ", ".join(BASE_MODES)
    )


def _process_ref(raw_text: str, ctx: CowboyContext) -> CowboyResult:
    result = CowboyResult()
    fixes, warnings, errors = (
        result.applied_fixes, result.warnings, result.retry_errors
    )

    text = strip_wrapper(raw_text or "", fixes)
    text = strip_markdown(text, fixes)
    text = fix_reference_tags(text, fixes)

    if not text.strip():
        errors.append("R5 EMPTY: the response contained no usable text.")
        return result
    if "subject" not in text.lower():
        errors.append(
            "R1 FORMAT: the output has no subject_definitions section. "
            "Write all six sections, starting with subject_definitions:."
        )
        result.prompt = text
        result.char_count = len(text)
        return result

    sections, exclusions = parse_sections(text, fixes, warnings)

    missing = [
        key for key in ("summary", "retention_analysis", "detailed_description")
        if key not in sections
    ]
    if missing:
        errors.append(
            "R1 FORMAT: these required sections are missing: "
            + ", ".join(missing)
            + ". Write all six, in order: "
            + ", ".join(REF_SECTION_ORDER) + "."
        )

    if "detailed_description" in sections:
        detailed, _times = normalize_shot_labels(
            sections["detailed_description"], ctx.duration_seconds,
            fixes, warnings,
            known_times=ctx.known_shot_times if ctx.is_editing else [],
            retry_errors=errors if ctx.is_editing else [],
        )
        # Outside an edit the measured list is a hint, not a contract:
        # the target video's structure is not bound to a reference
        # video's. Snap the times, but never retry over the count.
        if ctx.known_shot_times and not ctx.is_editing:
            detailed, _t = normalize_shot_labels(
                detailed, ctx.duration_seconds, fixes, warnings,
                known_times=ctx.known_shot_times, retry_errors=[],
            )
        check_camera_moves(detailed, warnings)
        _check_cut_phrasing(detailed, warnings)
        sections["detailed_description"] = detailed
        result.description_word_count = len(detailed.split())

    sections = enforce_task_type_set(sections, ctx, fixes, warnings)
    sections = enforce_retention_labels(sections, fixes, warnings)
    sections = enforce_video_edit(sections, ctx, fixes)

    _check_labels(sections, ctx, errors, warnings)
    _check_positions(sections, ctx, warnings)
    _check_features_echo(sections, ctx, warnings)
    _check_verbatim(sections, ctx, errors)
    sections = enforce_music_video(sections, ctx, fixes, errors, warnings)

    if exclusions:
        # Nothing follows non_diegetic_music in any official example, and
        # the word "exclusion" appears in none of the guides.
        fixes.append(
            f"dropped {len(exclusions)} trailing 'No ...' sentence(s); the "
            "official format ends at non_diegetic_music"
        )

    sections, _ = apply_trim_ladder(sections, [], fixes, warnings, "")
    result.prompt = reassemble(sections, [], "")
    result.char_count = len(result.prompt)
    return result


# ---------------------------------------------------------------------------
# The summary task-type prefix, compared as a SET
# ---------------------------------------------------------------------------

_TASK_PREFIX_RE = re.compile(r"^\s*\[([^\]]+)\]")


def enforce_task_type_set(
    sections: Dict[str, str], ctx: CowboyContext,
    fixes: List[str], warnings: List[str],
) -> Dict[str, str]:
    """
    Require the prefix, but accept any ORDER of the same types.

    The older package's version compares the joined string, so a model
    writing a legal permutation gets "corrected" - and MiniMax's own
    output writes "[video editing + audio reference + audio reuse]",
    which is out of its own table order. The guide states no ordering
    rule anywhere, so order cannot be a correctness criterion.

    We still emit one fixed order, for determinism: the same graph
    should always produce the same prefix so prompts diff cleanly.
    """
    if not ctx.task_type:
        return sections
    summary = sections.get("summary", "").strip()
    if not summary:
        return sections

    wanted_set = {t.strip().lower() for t in ctx.task_type.split(" + ")}
    found = _TASK_PREFIX_RE.match(summary)
    if not found:
        sections["summary"] = f"[{ctx.task_type}] {summary}"
        fixes.append(
            f"added the mandatory summary task-type prefix [{ctx.task_type}]"
        )
        return sections

    written = [t.strip() for t in found.group(1).split("+")]
    written_set = {t.lower() for t in written}
    if written_set == wanted_set:
        return sections                      # any order is correct

    unknown = [t for t in written if t.lower() not in TASK_TYPES]
    if unknown:
        warnings.append(
            "summary task-type prefix used a value outside the official "
            f"list: {', '.join(unknown)}; replaced with [{ctx.task_type}]"
        )
    else:
        fixes.append(
            f"corrected the summary task-type prefix to [{ctx.task_type}] "
            f"(model wrote [{found.group(1)}])"
        )
    rest = summary[found.end():].strip()
    sections["summary"] = f"[{ctx.task_type}] {rest}".strip()
    return sections


# ---------------------------------------------------------------------------
# Video editing - three fixed sentences, injected not hoped for
# ---------------------------------------------------------------------------

def enforce_video_edit(
    sections: Dict[str, str], ctx: CowboyContext, fixes: List[str],
) -> Dict[str, str]:
    """
    Give an editing job the three things guide_ref 3 and 2.3 require.

    All three are fixed strings with no content to invent, so they are
    written deterministically rather than left to the model and then
    retried over. Same reasoning as the task-type prefix: a fixed string
    cannot drift, and a retry that only adds a known sentence is a wasted
    API call.
    """
    if not ctx.is_editing:
        return sections

    summary = sections.get("summary", "").strip()
    if summary and EDIT_SUMMARY_OPENER not in summary:
        # It goes AFTER the bracketed prefix, not before it.
        prefix, _, rest = summary.partition("]")
        if prefix.startswith("[") and rest:
            sections["summary"] = (
                f"{prefix}] {EDIT_SUMMARY_OPENER} {rest.strip()}"
            )
        else:
            sections["summary"] = f"{EDIT_SUMMARY_OPENER} {summary}"
        fixes.append("added the mandatory video-editing summary opener")

    defined = sections.get("subject_definitions", "")
    if defined and not _line_opening_labels(defined) & {"Video 1"}:
        sections["subject_definitions"] = (
            defined.rstrip() + "\n" + EDIT_VIDEO_DEFINITION
        )
        fixes.append("added the <Video 1> source-video definition line")

    retention = sections.get("retention_analysis", "")
    if retention and not _line_opening_labels(retention) & {"Video 1"}:
        sections["retention_analysis"] = (
            retention.rstrip() + "\n" + EDIT_VIDEO_RETENTION
        )
        fixes.append("added the <Video 1> retention line")
    return sections


# ---------------------------------------------------------------------------
# R2 LABELS - one message, every instance
# ---------------------------------------------------------------------------

def _check_labels(
    sections: Dict[str, str], ctx: CowboyContext,
    errors: List[str], warnings: List[str],
) -> None:
    """
    Declared labels are used, used labels are declared, each has a
    retention line, and no citation points at an asset nobody sent.

    All four land in ONE retry message. They share a cause - the model
    lost track of the label set - and a single message listing every
    instance is what a retry can actually act on.
    """
    defined = sections.get("subject_definitions", "")
    retention = sections.get("retention_analysis", "")
    body = sections.get("detailed_description", "")
    if not defined:
        return

    # A label is DEFINED when it opens a line, not when it is mentioned.
    # guide_ref 2: "If <Picture N> or <Video N> only identifies the source
    # of another referenced item and will not be analyzed or used
    # separately later, cite it inside that item's definition without
    # adding a separate line." The official example cites four pictures
    # and two videos inside subject lines and gives none of them a line -
    # so a mention must not create an obligation.
    defined_labels = _line_opening_labels(defined)
    retention_labels = _line_opening_labels(retention)
    cited_labels = {
        f"{m.group(1)} {m.group(2)}" for m in _ANY_TAG_RE.finditer(defined)
    }
    body_labels = {
        f"{m.group(1)} {m.group(2)}" for m in _ANY_TAG_RE.finditer(body)
    }

    problems = []

    # Every subject we asked for must be defined and must appear.
    for subject in ctx.subjects:
        label = f"Subject {subject.index}"
        if label not in defined_labels:
            problems.append(
                f"<{label}> ({subject.kind}) was requested but has no line "
                "in subject_definitions"
            )
        elif label not in body_labels:
            problems.append(
                f"<{label}> is defined but never appears in "
                "detailed_description"
            )

    # guide_ref 4: "Use one line for each reference label".
    for label in sorted(defined_labels):
        if label not in retention_labels:
            problems.append(
                f"<{label}> is defined but has no line in retention_analysis"
            )

    # guide_ref 2: a label keeps its meaning across sections. A tag used
    # in the body is fine if it opens a line OR is cited inside one -
    # both mean subject_definitions established what it denotes.
    for label in sorted(body_labels - defined_labels - cited_labels):
        problems.append(
            f"<{label}> is used in detailed_description but never defined "
            "in subject_definitions"
        )

    # A citation of an image that was never attached points at nothing.
    for label in sorted(defined_labels | cited_labels | body_labels):
        word, _, number = label.partition(" ")
        if word == "Picture" and int(number) > ctx.wired_pictures:
            problems.append(
                f"<{label}> is cited but only {ctx.wired_pictures} "
                f"image(s) were attached"
            )
        if word == "Video" and not ctx.has_video:
            problems.append(f"<{label}> is cited but no video was attached")
        if word == "Audio" and not ctx.has_audio:
            problems.append(f"<{label}> is cited but no audio was attached")

    if problems:
        errors.append(
            "R2 LABELS: the reference labels do not line up. Fix all of "
            "these:\n" + "\n".join(f"  - {p}" for p in problems)
        )


# ---------------------------------------------------------------------------
# Warnings - shared by both modes
# ---------------------------------------------------------------------------

# guide_base 4.2, and guide_ref 5.1 defers to it for the body rules.
# Dissolve, fade and wipe are also legal "when explicitly requested by
# the user", which is why anything outside this list can only ever warn.
CUT_PHRASES = (
    "the camera cuts to",
    "the shot cuts to",
    "the shot transitions to",
    "the shot changes to",
    "the shot switches to",
)


def _check_cut_phrasing(body: str, warnings: List[str]) -> None:
    """
    A later shot OPENS with its cut, and the cut leads into the new shot.

    guide_base 4.2 gives the form as a template - "[Shot 2] At 00:03.500,
    the camera cuts to..." - and every later shot in all six official
    worked examples across both guides follows it: timestamp, cut phrase,
    then what the new shot shows, in one sentence. None of them describes
    the new shot first and mentions the cut afterwards.

    So there are two distinct mistakes worth naming separately. No cut
    phrase at all is one; having one but burying it mid-shot is the
    other, and it is the more likely of the two because it reads fine.
    Both warn and neither retries: a dissolve, fade or wipe the user
    asked for is legal and lands in the first case.
    """
    for i, shot in enumerate(_shot_bodies(body)[1:], start=2):
        opening = shot.lstrip().lower()
        if any(opening.startswith(phrase) for phrase in CUT_PHRASES):
            continue
        if any(phrase in opening for phrase in CUT_PHRASES):
            warnings.append(
                f"shot {i} has a cut phrase but does not open with it. The "
                "guide's form is \"[Shot N] At MM:SS.mmm, the shot cuts to "
                "<what the new shot shows>\" - the cut comes first and "
                "leads straight into the new shot"
            )
        else:
            warnings.append(
                f"shot {i} does not open with one of the five listed cut "
                "phrases (\"the camera cuts to\", \"the shot cuts to\", "
                "\"the shot transitions to\", \"the shot changes to\", "
                "\"the shot switches to\"); that is only correct if you "
                "asked for a dissolve, fade or wipe"
            )


def _check_positions(
    sections: Dict[str, str], ctx: CowboyContext, warnings: List[str],
) -> None:
    """
    A spatial subject should say where it is, once, at first appearance.

    A warning rather than a retry, for two reasons. The regex
    false-negatives on legal phrasing ("she stands beside the pallet
    stack" has no listed word until "beside" - and plenty of correct
    sentences have none at all), and a false retry costs an API call.
    Non-spatial kinds are skipped entirely: a scene IS the frame, and a
    style or an action has no position to state.
    """
    body = sections.get("detailed_description", "")
    if not body:
        return
    bodies = _shot_bodies(body) or [body]

    for subject in ctx.subjects:
        if not subject.is_spatial:
            continue
        tag = subject.tag
        first = next((b for b in bodies if tag in b), None)
        if first is None:
            continue          # already an R2 problem; do not double-report
        if not _POSITION_WORDS.search(first):
            warnings.append(
                f"{tag} ({subject.kind}) does not obviously state where it "
                "sits in the frame at its first appearance; the guide asks "
                "for that once, at first clear appearance"
            )


def _check_features_echo(
    sections: Dict[str, str], ctx: CowboyContext, warnings: List[str],
) -> None:
    """
    A subject's stated features should survive into its definition line.

    Kind-neutral, replacing the old node's wardrobe machinery - which
    injected "The exact wardrobe from <Picture 1> is: ..." and is
    nonsense on a lens grade or an environment. Loose keyword overlap
    only; the model is entitled to rephrase.
    """
    defined = sections.get("subject_definitions", "")
    if not defined:
        return
    lines = {
        m.group(0): line
        for line in defined.split("\n")
        for m in [_ANY_TAG_RE.match(line.strip())] if m
    }
    for subject in ctx.subjects:
        if not subject.features.strip():
            continue
        line = lines.get(subject.tag, "")
        if not line:
            continue
        words = {
            w for w in re.findall(r"[a-z]{4,}", subject.features.lower())
        }
        if words and not (words & set(re.findall(r"[a-z]{4,}", line.lower()))):
            warnings.append(
                f"{subject.tag} ({subject.kind}) is defined without any of "
                "the features you listed; check the definition line names "
                "what you actually want followed"
            )


def _check_verbatim(
    sections: Dict[str, str], ctx: CowboyContext, errors: List[str],
    body_key: str = "detailed_description",
) -> None:
    """
    Supplied words must reach a <d> block, verbatim.

    guide_ref 5.4 and guide_base 4.4: "Preserve every original word and
    punctuation mark verbatim; do not translate or rewrite them." This is
    a retry because the words cannot be synthesised - only the model can
    place them in the right shot.

    body_key is the only thing base mode changes here: the rule and the
    message are identical, only the field holding the timeline is named
    integrated_multimodal_description instead.
    """
    supplied = [t.strip() for t in (ctx.dialogue_text, ctx.lyrics) if t.strip()]
    if not supplied:
        return
    body = sections.get(body_key, "")
    blocks = " ".join(re.findall(r"<d>(.*?)</d>", body, re.DOTALL))
    missing = [
        text for text in supplied
        if not _appears_in(text, blocks)
    ]
    if missing:
        errors.append(
            "R3 VERBATIM: these exact words were supplied but do not appear "
            "inside any <d> block:\n"
            + "\n".join(f"  - {t[:120]}" for t in missing)
            + "\nPut them inside <d>[Language] ...</d> in the shot where "
            "they are heard, word for word."
        )


def _appears_in(needle: str, haystack: str) -> bool:
    """Whitespace- and case-insensitive containment."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    return norm(needle) in norm(haystack)


# ===========================================================================
# Base mode - T2VA / I2VA / FL2VA / L2VA
# ===========================================================================
#
# Three fields, no reference labels, and headers that carry their content
# on the SAME line:
#
#     integrated_multimodal_description: [Shot 1] Live-action, cinema...
#
# Ref mode writes "subject_definitions:\n<Subject 1> is...". That one
# difference is why base mode has its own parse and its own emit rather
# than a parameterised version of ref's: reassemble() joins with
# f"{key}:\n{content}", parse_sections() walks REF_SECTION_ORDER and
# expects a bare header line, and strip_wrapper's preamble regex
# hardcodes subject_definitions. A parameterised version of all three
# would still look right in a diff while emitting a malformed prompt.

_BASE_HEADER_PATTERNS = {
    "integrated_multimodal_description":
        r"integrated[_\s]multimodal[_\s]description",
    "overall_soundscape": r"overall[_\s]soundscape",
    "non_diegetic_music": r"non[-_\s]diegetic[_\s]music",
}
_BASE_HEADER_RE = re.compile(
    r"^[ \t]*(?:\*\*)?("
    + "|".join(_BASE_HEADER_PATTERNS.values())
    + r")(?:\*\*)?[ \t]*:[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
# Ref-mode headers have no meaning in base mode; seeing one means the
# model wrote the wrong format entirely.
_REF_HEADER_RE = re.compile(
    r"^[ \t]*(?:\*\*)?(subject[_\s]definitions?|summary|"
    r"retention[_\s]analysis|detailed[_\s]description)(?:\*\*)?[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_REF_LABEL_RE = re.compile(r"<(Subject|Video|Audio)\s+(\d+)>")
# Both spellings: <Picture 2> in I2VA and L2VA, bare Picture 2 in FL2VA.
# The FL2VA validators have to match either, because which one is right
# is exactly the question fl2va_normalize_picture_tags exists to answer.
_PICTURE_ANY_RE = re.compile(r"<?\bPicture\s+(\d+)\b>?")
_SHOT_ONE_RE = re.compile(r"\[\s*Shot\s+1\s*\]", re.IGNORECASE)
_NA_ONLY_RE = re.compile(r"(?i)^\s*n/?a\s*\.?\s*$")

# How many pictures each sub-mode has, by definition (guide_base 1).
PICTURES_FOR_MODE = {
    "base_T2VA": 0, "base_I2VA": 1, "base_FL2VA": 2, "base_L2VA": 1,
}

# guide_base 4.6 and 4.7.
SOUNDSCAPE_SENTENCES = (1, 4)
MUSIC_SENTENCES = (1, 3)


def _process_base(raw_text: str, ctx: CowboyContext) -> CowboyResult:
    """
    Normalize, repair and validate one base-mode reply.

    The order matters in one place: the instruction line is rendered
    LAST, from the shot count parsed out of the finished body, because
    guide_base 2.1 defines its N as "the index of the actual final shot".
    Anything that renders it earlier is guessing.
    """
    result = CowboyResult()
    fixes, warnings, errors = (
        result.applied_fixes, result.warnings, result.retry_errors
    )

    text = _strip_base_wrapper(raw_text or "", fixes)
    text = strip_markdown(text, fixes)
    text = fix_reference_tags(text, fixes)

    if not text.strip():
        errors.append("R5 EMPTY: the response contained no usable text.")
        return result

    sections = _parse_base_sections(text, fixes, warnings)
    _check_base_format(text, sections, ctx, errors)

    body = sections.get("integrated_multimodal_description", "")
    final_shot = 1
    if body:
        body = _force_style_inside_shot_one(body, fixes)
        body, _times = normalize_shot_labels(
            body, ctx.duration_seconds, fixes, warnings,
            known_times=ctx.known_shot_times, retry_errors=[],
        )
        check_camera_moves(body, warnings)
        sections["integrated_multimodal_description"] = body
        result.description_word_count = len(body.split())
        final_shot = max(1, len(_SHOT_LABEL_RE.findall(body)))

        _check_cut_phrasing(body, warnings)
        _check_picture_placement(body, ctx, warnings)
        if ctx.mode == "base_FL2VA" and final_shot > 1 and \
                not ctx.multi_shot_requested:
            warnings.append(
                f"FL2VA wrote {final_shot} shots and you did not ask for "
                "more than one; the guide says FL2VA \"generally favors a "
                "single shot so the model can interpolate continuously\""
            )

    _check_audio_sections(sections, warnings)
    _check_verbatim(
        sections, ctx, errors,
        body_key="integrated_multimodal_description",
    )

    instruction = render_instruction_line(
        ctx.mode, final_shot=final_shot,
        duration_seconds=ctx.duration_seconds,
    )
    sections = _apply_base_trim(sections, instruction, fixes, warnings)
    result.prompt = _reassemble_base(sections, instruction)
    if ctx.mode == "base_FL2VA" and ctx.fl2va_normalize_picture_tags:
        result.prompt = _normalize_fl2va_tags(result.prompt, fixes)
    result.char_count = len(result.prompt)
    return result


# ---------------------------------------------------------------------------
# Base parse and emit
# ---------------------------------------------------------------------------

def _strip_base_wrapper(text: str, fixes: List[str]) -> str:
    """
    Drop code fences and everything above the first base header.

    That preamble is usually the model imitating the worked example's
    instruction line. It is stripped rather than argued with: its N and
    S.SS cannot be right yet, since the body it would describe is in the
    same reply. The node writes the real one afterwards.

    A preamble carrying ref-mode sections or labels is left where it is,
    on purpose. Deleting it would hide the one thing worth retrying over
    - the model wrote the six-section format - behind a prompt that looks
    almost right.
    """
    if _FENCE_RE.search(text):
        text = _FENCE_RE.sub("", text)
        fixes.append("stripped code fences")

    found = _BASE_HEADER_RE.search(text)
    if found and text[:found.start()].strip():
        preamble = text[:found.start()].strip()
        if _REF_HEADER_RE.search(preamble) or _REF_LABEL_RE.search(preamble):
            return text.strip()
        text = text[found.start():]
        looks_like_instruction = (
            "aligns with the" in preamble or "is fully referenced" in preamble
        )
        fixes.append(
            "stripped the model's own alignment instruction line; it is "
            "rendered from the finished body instead"
            if looks_like_instruction else
            "stripped preamble before integrated_multimodal_description"
        )
    return text.strip()


def _base_header_key(written: str) -> str:
    """Canonical field name for a header the model may have miscased."""
    for key, pattern in _BASE_HEADER_PATTERNS.items():
        if re.fullmatch(pattern, written, re.IGNORECASE):
            return key
    return ""


def _parse_base_sections(
    text: str, fixes: List[str], warnings: List[str]
) -> Dict[str, str]:
    """
    Split base-mode text into {field: content}.

    Separate from parse_sections() because a base header is followed by
    its content on the same line, not by a newline - see the block
    comment above. Nothing is synthesized here either: a missing field is
    an R1 error, not something to invent, since all three are the whole
    format.
    """
    sections: Dict[str, str] = {}
    matches = list(_BASE_HEADER_RE.finditer(text))
    joined = 0

    for i, found in enumerate(matches):
        key = _base_header_key(found.group(1))
        if not key:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[found.end():end]
        if chunk[:1] == "\n" and chunk.strip():
            joined += 1
        content = chunk.strip()
        if not content:
            continue
        if found.group(0).strip() != f"{key}:":
            fixes.append(f"recased header to {key}:")
        sections[key] = (
            (sections[key] + "\n" + content).strip() if key in sections
            else content
        )

    if joined:
        fixes.append(
            f"pulled {joined} header(s) back onto one line with their "
            "content; base mode writes 'field: content', not 'field:' then "
            "a new line"
        )
    return sections


def _reassemble_base(sections: Dict[str, str], instruction: str = "") -> str:
    """
    Join the three fields back into one prompt.

    Two differences from reassemble(): the header and its content share a
    line, and the instruction goes on top with exactly one blank line
    under it (guide_base 2.1: "The instruction must be the first line of
    the final prompt, followed by one blank line before the core
    fields").
    """
    parts = []
    if instruction.strip():
        parts.append(instruction.strip())
    for key in BASE_SECTION_ORDER:
        content = sections.get(key, "").strip()
        if not content:
            continue
        parts.append(f"{key}: {content}")
    return "\n\n".join(parts)


def _normalize_fl2va_tags(text: str, fixes: List[str]) -> str:
    """
    Bracket FL2VA's bare `Picture 1` and `Shot 1`.

    Off by default. MiniMax writes them bare in the FL2VA instruction
    line AND in Case 3's body, while I2VA and L2VA bracket both, so the
    bare spelling is systematic rather than a slip - and no validator can
    tell which one generates a better video. This switch exists so that
    question can be settled by running it, and the setting is recorded in
    the analysis JSON so a result can be attributed.
    """
    fixed = re.sub(r"(?<![<\w])Picture (\d+)(?![>\w])", r"<Picture \1>", text)
    fixed = re.sub(r"(?<![\[\w])Shot (\d+)(?![\]\w])", r"[Shot \1]", fixed)
    if fixed != text:
        fixes.append(
            "fl2va_normalize_picture_tags is on: bracketed the bare "
            "Picture N and Shot N mentions the guide writes bare"
        )
    return fixed


# ---------------------------------------------------------------------------
# R1 FORMAT - one message, every instance
# ---------------------------------------------------------------------------

def _check_base_format(
    text: str, sections: Dict[str, str], ctx: CowboyContext,
    errors: List[str],
) -> None:
    """
    Three fields, in order, nothing else, and no ref-mode furniture.

    All of it lands in ONE retry message. Every instance has the same
    cause - the model wrote a format other than the one it was asked for
    - and one message listing them is what a retry can act on.
    """
    problems = []

    written = [
        _base_header_key(m.group(1)) for m in _BASE_HEADER_RE.finditer(text)
    ]
    if written != BASE_SECTION_ORDER:
        missing = [k for k in BASE_SECTION_ORDER if k not in written]
        repeated = sorted({k for k in written if written.count(k) > 1})
        if missing:
            problems.append(
                "these fields are missing: " + ", ".join(missing)
            )
        if repeated:
            problems.append(
                "these fields are written more than once: "
                + ", ".join(repeated)
            )
        if not missing and not repeated:
            problems.append(
                "the fields are out of order (" + ", ".join(written)
                + "); write them as " + ", ".join(BASE_SECTION_ORDER)
            )
    for key in BASE_SECTION_ORDER:
        if key in written and key not in sections:
            problems.append(f"{key} has a header but no content")

    for found in _REF_HEADER_RE.finditer(text):
        problems.append(
            f"'{found.group(1)}' is a full-reference section and does not "
            "exist in base mode"
        )
    ref_labels = sorted({
        f"<{m.group(1)} {m.group(2)}>" for m in _REF_LABEL_RE.finditer(text)
    })
    if ref_labels:
        problems.append(
            "these are full-reference labels and do not exist in base "
            "mode: " + ", ".join(ref_labels)
        )

    # guide_base 1: the sub-mode fixes how many pictures there are. A
    # citation past that count points at an image H3 never receives.
    allowed = PICTURES_FOR_MODE.get(ctx.mode, 0)
    cited = sorted({int(m.group(1)) for m in _PICTURE_ANY_RE.finditer(text)})
    stray = [n for n in cited if n > allowed]
    if stray:
        named = ", ".join(f"Picture {n}" for n in stray)
        sub_mode = ctx.mode.replace("base_", "")
        problems.append(
            f"{named} " + ("is" if len(stray) == 1 else "are") + " cited, but "
            + (
                f"{sub_mode} has no reference picture at all"
                if allowed == 0 else
                f"{sub_mode} has only {allowed} picture(s)"
            )
        )

    if problems:
        errors.append(
            "R1 FORMAT: this is the base format, three fields and nothing "
            "else. Fix all of these:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


# ---------------------------------------------------------------------------
# Deterministic body repair
# ---------------------------------------------------------------------------

def _force_style_inside_shot_one(body: str, fixes: List[str]) -> str:
    """
    Put the style back inside [Shot 1], where base mode wants it.

    guide_base 4.1: "At the beginning of [Shot 1], state the overall
    style and initial composition", and its own example writes
    "[Shot 1] Live-action, cinematic, a medium-wide shot frames...". Ref
    mode puts that sentence BEFORE the first label, and guide_ref 5.2
    tables the difference outright - so a model that has seen either
    format will produce this, and it is deterministic to repair.
    """
    found = _SHOT_ONE_RE.search(body)
    if found is None:
        return "[Shot 1] " + body.strip() if body.strip() else body
    head = body[:found.start()].strip()
    if not head:
        return body
    rest = body[found.end():].strip()
    fixes.append(
        "moved the style sentence inside [Shot 1]; base mode writes it "
        "after the label, not before it"
    )
    return f"[Shot 1] {head} {rest}".strip()


# ---------------------------------------------------------------------------
# Warnings - none of these ever costs an API call
# ---------------------------------------------------------------------------

def _picture_in(text: str, index: int) -> bool:
    """True when Picture <index> is cited, bracketed or bare."""
    return any(
        int(m.group(1)) == index for m in _PICTURE_ANY_RE.finditer(text)
    )


def _check_picture_placement(
    body: str, ctx: CowboyContext, warnings: List[str],
) -> None:
    """
    Each anchor frame should be cited where the video actually reaches it.

    Warnings, because the guide describes where the picture belongs
    without fixing the words that put it there, and a false retry costs
    a real API call. The L2VA case is the one worth reading: guide_base
    3.3 says <Picture 1> "belongs to the last [Shot N]; it does not
    inherently belong to Shot 1", and a run that opens on it has the
    whole mode backwards.
    """
    bodies = _shot_bodies(body) or [body]
    first, last = bodies[0], bodies[-1]

    if ctx.mode == "base_I2VA" and not _picture_in(first, 1):
        warnings.append(
            "<Picture 1> is the first frame but is not cited inside "
            "[Shot 1]; guide_base 3.1 puts it at 0.00 seconds in [Shot 1]"
        )
    elif ctx.mode == "base_L2VA" and not _picture_in(last, 1):
        where = "in [Shot 1]" if _picture_in(first, 1) else "nowhere"
        warnings.append(
            f"<Picture 1> is the LAST frame but is cited {where} rather "
            "than in the final shot; guide_base 3.3: it \"does not "
            "inherently belong to Shot 1\""
        )
    elif ctx.mode == "base_FL2VA":
        if not _picture_in(first, 1):
            warnings.append(
                "Picture 1 is the first frame but is not cited in the "
                "opening shot"
            )
        if not _picture_in(last, 2):
            warnings.append(
                "Picture 2 is the last frame but is not cited in the final "
                "shot; guide_base 3.2: \"The last frame must be reached by "
                "the final [Shot N] at the end of the video\""
            )


def _sentence_count(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()])


_D_BLOCK_RE = re.compile(r"<d>.*?</d>", re.DOTALL)


def enforce_music_video(
    sections: Dict[str, str], ctx: CowboyContext, fixes: List[str],
    errors: List[str], warnings: List[str],
) -> Dict[str, str]:
    """
    Hold music-video mode to the audio rules in guide_ref 6.

    Two of them, and they are opposite kinds of problem. A score written
    as "N/A" is a music video with no music, which only the model can
    fix, so it retries. A lyric repeated in an audio section is a
    mechanical duplicate - the words already sit inside <d> where they
    belong - so it is stripped without spending an API call.
    """
    if not ctx.music_video:
        return sections

    music = sections.get("non_diegetic_music", "").strip()
    if not music or _NA_ONLY_RE.match(music):
        errors.append(
            "R6 MUSIC: non_diegetic_music is 'N/A', but a music video is "
            "scored wall to wall. Describe the track - genre, "
            "instrumentation, tempo, and how it develops across the clip - "
            "or say the supplied score is reused directly."
        )

    stripped = 0
    for key in ("overall_soundscape", "non_diegetic_music"):
        content = sections.get(key, "")
        if not content or not _D_BLOCK_RE.search(content):
            continue
        cleaned = re.sub(r"\s{2,}", " ", _D_BLOCK_RE.sub("", content)).strip()
        if cleaned:
            sections[key] = cleaned
            stripped += 1
        else:
            warnings.append(
                f"section '{key}' was nothing but lyrics; left as written"
            )
    if stripped:
        fixes.append(
            f"removed lyrics repeated in {stripped} audio section(s); they "
            "belong only inside <d> in detailed_description"
        )
    return sections


def _check_audio_sections(
    sections: Dict[str, str], warnings: List[str],
) -> None:
    """
    Sentence budgets for the two audio fields (guide_base 4.6 and 4.7).

    "N/A" is skipped in both. It is legal in non_diegetic_music whenever
    there is no audience-only music - Case 3 uses it - and legal in
    overall_soundscape for requested silence.
    """
    for key, (lo, hi) in (
        ("overall_soundscape", SOUNDSCAPE_SENTENCES),
        ("non_diegetic_music", MUSIC_SENTENCES),
    ):
        content = sections.get(key, "").strip()
        if not content or _NA_ONLY_RE.match(content):
            continue
        count = _sentence_count(content)
        if not lo <= count <= hi:
            warnings.append(
                f"{key} is {count} sentence(s); the guide asks for "
                f"{lo}-{hi}"
            )


def _apply_base_trim(
    sections: Dict[str, str], instruction: str,
    fixes: List[str], warnings: List[str],
) -> Dict[str, str]:
    """
    Hold the prompt under the character budget, cheapest cut first.

    MAX_PROMPT_CHARS is a TrentNodes number, not a MiniMax one - no
    official document states a limit - so overshooting warns rather than
    retries. Same first rung as the ref ladder: shorten the score instead
    of blanking it, because "N/A" asserts a silence that is not true.
    """
    if len(_reassemble_base(sections, instruction)) <= MAX_PROMPT_CHARS:
        return sections

    music = sections.get("non_diegetic_music", "")
    if len(music) > 60:
        first = _SENTENCE_SPLIT_RE.split(music.strip())[0].strip()
        if first and len(first) < len(music):
            sections["non_diegetic_music"] = first
            fixes.append(
                "shortened non_diegetic_music to its first sentence "
                "(over char budget)"
            )

    size = len(_reassemble_base(sections, instruction))
    if size > MAX_PROMPT_CHARS:
        warnings.append(
            f"prompt is {size} chars (TrentNodes budget "
            f"{MAX_PROMPT_CHARS}; MiniMax states no character limit)"
        )
    return sections
