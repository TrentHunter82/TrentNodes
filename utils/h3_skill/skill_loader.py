"""
Load the h3-prompting skill text and build the system prompt from it.

The skill file IS the spec the node enforces. Loading it live from
~/.claude/skills keeps the node and the skill from drifting apart; the
vendored snapshot under assets/ makes the pack self-contained when the
live file is absent (other machines, CI).
"""

import os
from typing import Optional, Tuple

MODES = ("ref2va", "t2va", "i2va", "fl2va", "l2va")

# Node-facing mode id -> h3_cowboy/spec.py mode id (for example lookup
# and the instruction-line renderer).
COWBOY_MODE = {
    "ref2va": "ref",
    "t2va": "base_T2VA",
    "i2va": "base_I2VA",
    "fl2va": "base_FL2VA",
    "l2va": "base_L2VA",
}

# Skill "Step 0" table: Ref2VA and Base are different checkpoints.
CHECKPOINT_FOR_MODE = {
    mode: ("MiniMax-H3-Base-Ref2VA" if mode == "ref2va" else "MiniMax-H3-Base-FL2VA")
    for mode in MODES
}

LIVE_SKILL_PATH = os.path.join(
    os.path.expanduser("~"), ".claude", "skills", "h3-prompting", "SKILL.md"
)
VENDORED_SKILL_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "h3_prompting_skill.md"
)

_REF_OPENER = "subject_definitions:"
_BASE_OPENER = "integrated_multimodal_description:"


def load_skill_text() -> Tuple[str, str]:
    """Return (skill_text, source) where source names what was read."""
    for path, source in (
        (LIVE_SKILL_PATH, "live skill file"),
        (VENDORED_SKILL_PATH, "vendored snapshot"),
    ):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        if text.strip():
            return text, source
    raise RuntimeError(
        "The h3-prompting skill text was not found - neither "
        f"{LIVE_SKILL_PATH} nor the vendored snapshot exists."
    )


def _official_example(mode: str) -> Tuple[str, str]:
    """
    Return (example_body, pipeline_alignment_line) for the mode.

    Always MiniMax's own example, imported from h3_cowboy/spec.py (the
    pack's policy is to never write a worked example ourselves). Base
    examples embed their alignment line above the fields; that line is
    pipeline-rendered in this package too, so it is split off and shown
    separately rather than teaching the model to write it.
    """
    try:
        from ..h3_cowboy import spec as cowboy_spec
    except ImportError:
        return "", ""
    if mode == "ref2va":
        return cowboy_spec.EXAMPLE_REF_GENERATION.strip(), ""
    example = cowboy_spec.EXAMPLE_FOR_BASE_MODE[COWBOY_MODE[mode]].strip()
    if example.startswith(_BASE_OPENER):
        return example, ""
    head, _, rest = example.partition("\n\n")
    return rest.strip(), head.strip()


def build_system_prompt(mode: str, skill_text: Optional[str] = None) -> Tuple[str, str]:
    """
    Assemble the system prompt for `mode`. Returns (prompt, skill_source).
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Legal: {', '.join(MODES)}")
    if skill_text is None:
        skill_text, source = load_skill_text()
    else:
        source = "caller-supplied"

    opener = _REF_OPENER if mode == "ref2va" else _BASE_OPENER
    skeleton = "Ref2VA six-section" if mode == "ref2va" else "Base three-field"

    parts = [
        "You are an expert MiniMax H3 video prompt writer. The complete "
        "specification you must follow is the skill document below. It "
        "overrides any habit you have from other video models.",
        "\n\n===== SKILL DOCUMENT =====\n\n" + skill_text.strip(),
        "\n\n===== THIS RUN =====\n\n"
        f"Mode: {mode} -> use the {skeleton} skeleton.\n"
        "Output contract:\n"
        f"- Output ONLY the prompt text. The first characters of your reply "
        f"must be `{opener}`.\n"
        "- No markdown, no code fences, no headings, no commentary, no "
        "reasoning text, nothing before or after the prompt.\n"
        "- Never state duration, fps, or aspect ratio inside the prompt.",
    ]
    if mode != "ref2va":
        parts.append(
            "- Do NOT write the picture-alignment instruction line above the "
            "fields. The pipeline renders and prepends it after you answer, "
            "because the final shot index and effective duration are only "
            "known then."
        )

    example, alignment = _official_example(mode)
    if example:
        parts.append(
            "\n\n===== OFFICIAL EXAMPLE (MiniMax's own, matching this mode) "
            "=====\n\n" + example
        )
        if alignment:
            parts.append(
                "\n\n(For context only - the pipeline will prepend an "
                "alignment line like this to your output afterwards; you "
                "never write it:\n" + alignment + ")"
            )
    return "".join(parts), source


def build_user_context(
    mode: str,
    creative_brief: str,
    duration_seconds: float,
    dialogue: str = "",
    image_lines: Optional[list] = None,
    source_soundscape: str = "",
    source_music: str = "",
    sound_log: str = "",
) -> str:
    """The user-message text that precedes the attached images.

    source_soundscape / source_music / sound_log carry the H3 Audio
    Soundscaper's measured analysis of the SOURCE clip's real audio;
    when present they anchor the prompt's audio sections."""
    lines = [
        "TASK CONTEXT",
        f"- mode: {mode}",
        (
            f"- target clip duration: {duration_seconds:.2f} seconds. Shot "
            "timestamps must stay strictly inside this budget. Never "
            "mention the duration itself in the prompt."
        ),
    ]
    brief = (creative_brief or "").strip()
    lines.append("- creative brief:\n" + (brief if brief else "(none given)"))
    if dialogue.strip():
        lines.append(
            "- dialogue (use these words verbatim inside <d> tags, never "
            "translate):\n" + dialogue.strip()
        )
    if source_soundscape.strip() or source_music.strip() or sound_log.strip():
        lines.append(
            "- MEASURED AUDIO of the source clip (from real audio "
            "analysis). Base the prompt's overall_soundscape and "
            "non_diegetic_music on it, adapted to the brief - keep this "
            "sound design unless the brief changes it. Never quote "
            "dialogue words or lyrics inside the audio sections."
        )
        if source_soundscape.strip():
            lines.append("  measured diegetic soundscape:\n  "
                         + source_soundscape.strip())
        if source_music.strip():
            lines.append("  measured score (non-diegetic):\n  "
                         + source_music.strip())
        if sound_log.strip():
            lines.append("  timestamped sound log:\n  " + sound_log.strip())
    if image_lines:
        lines.append("- attached images, in order:")
        lines.extend(f"  {line}" for line in image_lines)
        if mode == "i2va":
            lines.append("  (the first attached picture is the FIRST frame)")
        elif mode == "l2va":
            lines.append("  (the first attached picture is the LAST frame)")
        elif mode == "fl2va":
            lines.append(
                "  (picture 1 is the FIRST frame, picture 2 is the LAST "
                "frame; write the motion path between them)"
            )
    elif mode in ("i2va", "l2va", "fl2va"):
        lines.append(
            "- WARNING: this mode is frame-anchored but no image was "
            "attached; describe the anchor from the brief."
        )
    lines.append(
        "Write the H3 prompt now, following the skill document and the "
        "output contract exactly."
    )
    return "\n".join(lines)
