"""
Ultimate H3 Cowboy Promptor - the general MiniMax H3 prompt writer.

Where utils/h3_prompt/ writes one job (replace the performer in a video
with the person in a reference image), this package writes any job the
H3 spec describes: subjects of any kind, several at once, all six task
types, and both the ref-mode six-section skeleton and the base-mode
three-field one.

The two packages coexist on purpose. utils/h3_prompt/ is frozen - the
older node keeps working while this is built - and this package imports
its subject-agnostic helpers rather than copying them. See
docs/H3_COWBOY_HANDOFF.md for what is shared, what is deliberately not,
and the two constraints that made a fork the right call.

Light imports only: no torch, no provider SDKs, so a test can read the
format without a GPU.
"""

from .spec import (
    BASE_SECTION_ORDER,
    CHECKPOINT_FOR_MODE,
    KIND_CARDS,
    MODES,
    REF_SECTION_ORDER,
    SPATIAL_KINDS,
    SUBJECT_KINDS,
    TASK_TYPES,
)
from .subjects import (
    SubjectSpec,
    bind_images,
    declared_kinds,
    parse_subjects,
)

__all__ = [
    "BASE_SECTION_ORDER",
    "CHECKPOINT_FOR_MODE",
    "KIND_CARDS",
    "MODES",
    "REF_SECTION_ORDER",
    "SPATIAL_KINDS",
    "SUBJECT_KINDS",
    "SubjectSpec",
    "TASK_TYPES",
    "bind_images",
    "declared_kinds",
    "parse_subjects",
]
