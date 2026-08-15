"""
The subject DSL: one line per <Subject N>, kind first.

    <kind> [name] [@Tag N ...] -- <features>

    person  Aria Voss     @Picture 1             -- dark shoulder-length hair, charcoal utility jacket
    scene   the warehouse @Picture 2             -- corrugated steel walls, sodium toplight
    animal  the Samoyed   @Picture 3 @Picture 4  -- thick white fur, pointed ears, curved tail
    style                 @Video 2               -- 16mm grain, halation, teal-shifted shadows
    action  the vault     @Video 1               -- plant left hand, tuck, shoulder roll

Kind first because it is the only field that is always required and always
from a fixed set - which makes a mistyped line detectable instead of
silently becoming a subject named "person".

Everything else is optional. A name is optional because the guide's own
complete example uses none (its subjects are "the young blonde woman",
not "Aria"). Sources are optional because a style or an action often has
no image at all, and a description-only line is the only way to express
one. Features are optional but nearly always wanted - they are the
guide's "main features to follow".

<Subject N> numbers by declaration order, 1-based. Pinning an explicit
number is deliberately NOT supported: the guide's own example numbers
sequentially, and pinning buys collision handling nobody asked for.

Pure string and dataclass work - no torch, no ComfyUI - so it is testable
offline.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .spec import KIND_CHOICES, SPATIAL_KINDS, SUBJECT_KINDS, canonical_kind

# "@Picture 1", "@picture1", "@Video 2" - tolerant on case and on the
# space, because both get typed.
_SOURCE_RE = re.compile(
    r"@\s*(picture|video|audio|image|photo)\s*(\d+)", re.IGNORECASE
)
# Canonical spelling for each source word the guide recognises. "image"
# and "photo" are here because people type them; the guide's tag is
# <Picture N>.
_SOURCE_CANONICAL = {
    "picture": "Picture", "image": "Picture", "photo": "Picture",
    "video": "Video", "audio": "Audio",
}
# The features separator. Required to sit on whitespace so a hyphenated
# feature ("teal-shifted") and an em-dash never split a line.
_FEATURES_RE = re.compile(r"\s+--\s+|\s+--$")


@dataclass
class SubjectSpec:
    """One <Subject N> and everything the prompt needs to define it."""
    index: int                                   # N in <Subject N>
    kind: str                                    # one of SUBJECT_KINDS
    name: str = ""                               # optional, often blank
    sources: List[str] = field(default_factory=list)   # ["Picture 1", ...]
    features: str = ""
    slot: Optional[int] = None                   # subject_N_image that fed it

    @property
    def tag(self) -> str:
        return f"<Subject {self.index}>"

    @property
    def is_spatial(self) -> bool:
        """True when 'where is it in frame' is a meaningful question."""
        return self.kind in SPATIAL_KINDS

    def source_phrase(self) -> str:
        """'<Picture 1>' / '<Picture 3>, <Picture 4>' / '' when none."""
        return ", ".join(f"<{s}>" for s in self.sources)

    def describe(self) -> str:
        """One human-readable line, for the task context and the log."""
        bits = [self.kind]
        if self.name:
            bits.append(self.name)
        if self.sources:
            bits.append(f"from {self.source_phrase()}")
        line = f"{self.tag}: " + " ".join(bits)
        return f"{line} -- {self.features}" if self.features else line


def parse_subjects(text: str, warnings: List[str]) -> List[SubjectSpec]:
    """
    Read the subjects field. Never raises; a bad line warns and is skipped.

    Skipping rather than guessing is deliberate. Defaulting an
    unrecognised kind would inject the wrong feature guidance - a person
    card on a lens grade - and the model would follow it. A warning that
    names the line and the legal kinds is more useful than a subject that
    silently means something else.
    """
    subjects: List[SubjectSpec] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        head, features = _split_features(line)
        sources, head = _take_sources(head)
        kind, name = _take_kind(head)

        if kind is None:
            first = head.split()[0] if head.split() else raw.strip()
            warnings.append(
                f"subjects line {lineno}: '{first}' is not a subject kind, "
                f"so the line was skipped. Start the line with one of: "
                f"{', '.join(SUBJECT_KINDS)}."
            )
            continue
        if not features and not sources:
            warnings.append(
                f"subjects line {lineno} ({kind}) has no features and no "
                "source, so the model has nothing to work from; add "
                "\"-- <features>\" or wire an image."
            )

        subjects.append(SubjectSpec(
            index=len(subjects) + 1, kind=kind, name=name,
            sources=sources, features=features,
        ))
    return subjects


def _split_features(line: str):
    """Split '<head> -- <features>'. No separator means no features."""
    parts = _FEATURES_RE.split(line, maxsplit=1)
    if len(parts) == 1:
        return line.strip(), ""
    return parts[0].strip(), (parts[1] or "").strip()


def _take_sources(head: str):
    """Pull every '@Tag N' out of the head, keeping declaration order."""
    sources = []
    for found in _SOURCE_RE.finditer(head):
        word = _SOURCE_CANONICAL[found.group(1).lower()]
        tag = f"{word} {int(found.group(2))}"
        if tag not in sources:
            sources.append(tag)
    return sources, _SOURCE_RE.sub(" ", head).strip()


def _take_kind(head: str):
    """First token as the kind; the rest is the name. (None, head) if not."""
    tokens = head.split()
    if not tokens:
        return None, ""
    kind = canonical_kind(tokens[0])
    if not kind:
        return None, head
    return kind, " ".join(tokens[1:]).strip()


# ---------------------------------------------------------------------------
# The rows on the node face
# ---------------------------------------------------------------------------

@dataclass
class SubjectRow:
    """One row of the node's subject list: kind, name, description."""
    slot: int
    kind: str = "character"
    name: str = ""
    description: str = ""

    @property
    def is_filled(self) -> bool:
        """A row nobody typed in is not a subject, whatever it holds."""
        return bool(self.name.strip() or self.description.strip())


def subjects_from_rows(
    rows: List[SubjectRow], warnings: List[str],
) -> List[SubjectSpec]:
    """
    Turn the filled rows into subjects, keeping row N as <Subject N>.

    The index is the ROW NUMBER, never the position in this list. That is
    the same promise the images make - subject_N_image IS <Picture N> -
    and the two only line up if a skipped row leaves a hole rather than
    closing it. bind_images() then finds row N for slot N with no
    bookkeeping between them.

    A row with neither a name nor a description is not a subject. That is
    what lets the row count sit at a comfortable number without inventing
    empty subjects, and what keeps a workflow saved before rows existed
    behaving exactly as it did.
    """
    subjects: List[SubjectSpec] = []
    for row in rows:
        if not row.is_filled:
            continue
        kind = canonical_kind(row.kind)
        if not kind:
            warnings.append(
                f"subject row {row.slot}: '{row.kind}' is not a kind, so "
                f"the row was read as a {KIND_CHOICES[0]}. Pick one of: "
                + ", ".join(KIND_CHOICES) + "."
            )
            kind = canonical_kind(KIND_CHOICES[0])
        subjects.append(SubjectSpec(
            index=row.slot, kind=kind, name=row.name.strip(),
            features=row.description.strip(),
        ))

    filled = [s.index for s in subjects]
    if filled and filled != list(range(1, len(filled) + 1)):
        warnings.append(
            "subject rows are filled with a gap ("
            + ", ".join(f"row {i}" for i in filled)
            + "), so the picture numbers skip too. Fill the rows in order "
            "from row 1 unless you meant the gap."
        )
    return subjects


def merge_text_subjects(
    rows: List[SubjectSpec], text: str, warnings: List[str],
) -> List[SubjectSpec]:
    """
    Rows first, then any typed lines, renumbered to follow them.

    The typed field is the escape hatch: more than six subjects, or a
    subject that cites two pictures at once, which no row can express. It
    never renumbers a row, because a row number is a picture number.
    """
    typed = parse_subjects(text, warnings) if text.strip() else []
    if not typed:
        return list(rows)
    if not rows:
        return typed

    start = max(s.index for s in rows)
    for offset, subject in enumerate(typed, start=1):
        subject.index = start + offset
    warnings.append(
        f"{len(typed)} typed subject line(s) follow the {len(rows)} filled "
        "row(s), as <Subject "
        + ">, <Subject ".join(str(s.index) for s in typed)
        + ">. Those numbers are picture numbers too, so a typed line that "
        "needs an image wants subject_" + str(typed[0].index) + "_image."
    )
    return list(rows) + typed


def bind_images(
    subjects: List[SubjectSpec], wired_slots: List[int],
    warnings: List[str],
) -> List[str]:
    """
    Turn wired subject_N_image slots into <Picture N> citations.

    ONE rule, and it is worth stating plainly on the node face:

        subject_N_image IS <Picture N>.

    Slot 2 is always <Picture 2>, whether or not slot 1 is wired. The
    obvious alternative - number the wired slots 1..M in wiring order -
    reads fine until someone also writes an explicit "@Picture 1", and
    then two subjects cite one tag for two different images. Making the
    slot number the picture number is the only scheme where an explicit
    pin and a wired slot cannot disagree.

    The cost is that a gap leaves a gap in the numbering, so this warns
    and asks for the slots to be filled in order. Returns the picture tag
    per wired slot, in slot order, for labelling the upload.
    """
    wired = sorted(set(wired_slots))
    by_index = {s.index: s for s in subjects}

    if wired and wired != list(range(1, len(wired) + 1)):
        warnings.append(
            "subject image slots are wired with a gap "
            f"({', '.join(f'subject_{s}_image' for s in wired)}), so the "
            "picture numbers skip too. Wire them in order from slot 1 "
            "unless you meant the gap."
        )

    for slot in wired:
        tag = f"Picture {slot}"
        subject = by_index.get(slot)
        if subject is None:
            warnings.append(
                f"subject_{slot}_image is wired but row {slot} is empty. The "
                f"image is still sent as <{tag}> and the model is told to "
                f"describe it, but filling in subject_{slot}_kind and "
                f"subject_{slot}_description works far better."
            )
            continue
        if subject.sources and tag not in subject.sources:
            warnings.append(
                f"{subject.tag} cites {subject.source_phrase()} but its own "
                f"slot subject_{slot}_image is wired; <{tag}> was added so "
                "the wired image is not silently unused."
            )
            subject.sources.append(tag)
        elif not subject.sources:
            subject.sources = [tag]
        subject.slot = slot

    # A citation of a picture nobody wired points at nothing H3 receives.
    for subject in subjects:
        for source in subject.sources:
            word, _, number = source.partition(" ")
            if word == "Picture" and int(number) not in wired:
                warnings.append(
                    f"{subject.tag} cites <{source}> but "
                    f"subject_{number}_image is not wired, so that image is "
                    "never sent"
                )
        if not subject.sources and not subject.features:
            warnings.append(
                f"{subject.tag} ({subject.kind}) has neither a source nor "
                "features and cannot be defined"
            )
    return [f"Picture {slot}" for slot in wired]


def declared_kinds(subjects: List[SubjectSpec]) -> List[str]:
    """Kinds actually used, in first-declaration order.

    Only these get a feature card in the system prompt. A run with two
    people and a scene should never read the style card.
    """
    seen = []
    for subject in subjects:
        if subject.kind not in seen:
            seen.append(subject.kind)
    return seen
