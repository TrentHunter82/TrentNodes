"""
Prompts + parsing for the H3 Audio Soundscaper.

Two modes share one output contract (the four labeled sections below),
so parsing, the corrective retry, and the node outputs are identical:

- listening (SYSTEM_PROMPT): the model hears a clip's audio track and
  transcribes what is there.
- design (DESIGN_SYSTEM_PROMPT): no audio exists; the model reads a
  video prompt and invents the soundtrack that fits it.

Every rule below is the h3-prompting skill's audio spec:
overall_soundscape 1-4 sentences of diegetic sound; non_diegetic_music
1-3 sentences or exactly N/A (music-video mode inverts the balance and
is never N/A); dialogue never repeated inside the audio sections;
diegetic music (music the characters can hear) belongs in the
soundscape, not the music section.
"""

import re
from typing import Dict, List, Tuple

SECTIONS = ("sound_log", "overall_soundscape", "non_diegetic_music", "dialogue")

SYSTEM_PROMPT = """\
You are a film sound analyst feeding a MiniMax H3 video-prompt pipeline.
You hear ONLY the audio track of a clip. Answer with exactly these four
lowercase section headers, each alone on its line, in this order, with
the content on the following lines:

sound_log:
overall_soundscape:
non_diegetic_music:
dialogue:

Section rules:
- sound_log: a film sound designer's log - every audible event in order
  with rough MM:SS timing, texture, and a category (Foley / SFX /
  Ambience / Music / Speech).
- overall_soundscape: 1-4 sentences of DIEGETIC sound only (sound that
  exists in the scene's world: foley, ambience, effects, and any music
  the characters could hear). Never quote dialogue words or lyrics.
- non_diegetic_music: 1-3 sentences on the score's instrumentation,
  tempo, and development - or exactly N/A when there is no score.
  Music that plays inside the scene is diegetic and belongs in
  overall_soundscape instead.
- dialogue: each clearly spoken line, one per line, as
  (S1) [Language] Exact words.
  Number speakers (S1), (S2) in order of first appearance. For sung
  lyrics write (lyrics) instead of a speaker ID. Never translate.
  Write N/A when nothing is spoken or sung.

Hard rules:
- The user message states the clip's exact duration. NEVER describe
  anything past it - analysis windows may pad the clip with silence,
  and events you think you hear after the stated end are artifacts.
- Describe only what is audible. Never invent visual content.
- No markdown emphasis, no bullet symbols other than the log lines,
  nothing before the first header or after the last section."""


DESIGN_SYSTEM_PROMPT = """\
You are a film sound designer feeding a MiniMax H3 video-prompt
pipeline. You will read the text of a video prompt or scene
description. No audio exists yet: your job is to DESIGN the soundtrack
that best serves the described visuals. Answer with exactly these four
lowercase section headers, each alone on its line, in this order, with
the content on the following lines:

sound_log:
overall_soundscape:
non_diegetic_music:
dialogue:

Section rules:
- sound_log: a sound designer's cue sheet for the described video -
  every planned audible event in order, keyed to the prompt's shots
  ([Shot 1], [Shot 2] ...) or rough MM:SS timing, each with texture
  and a category (Foley / SFX / Ambience / Music / Speech). Ground
  every cue in something the prompt shows on screen.
- overall_soundscape: 1-4 sentences of DIEGETIC sound only (sound that
  exists in the scene's world: foley, ambience, effects, and any music
  the characters could hear), each tied to a visible event from the
  prompt. Never quote dialogue words or lyrics.
- non_diegetic_music: 1-3 sentences on the score's instrumentation,
  tempo, and how it develops across the shots - or exactly N/A when a
  score would not serve the scene. Music that plays inside the scene
  is diegetic and belongs in overall_soundscape instead. If the prompt
  is a music video, the music leads: give genre, instruments, tempo,
  and arrangement against the shot list, never N/A, and keep
  overall_soundscape thin.
- dialogue: copy each spoken or sung line the video prompt already
  contains, verbatim, one per line, as
  (S1) [Language] Exact words.
  Number speakers (S1), (S2) in order of first appearance. For sung
  lyrics write (lyrics) instead of a speaker ID. Never translate, and
  NEVER invent lines the prompt does not contain. Write N/A when the
  prompt has no dialogue.

Hard rules:
- Design sound only for what the prompt describes. Never add events,
  props, or characters the prompt does not show.
- If the prompt has shot timestamps, keep every cue inside them and in
  order.
- No markdown emphasis, no bullet symbols other than the log lines,
  nothing before the first header or after the last section."""


def build_design_context(video_prompt: str) -> str:
    return (
        "Here is the video prompt. Design its soundtrack now, "
        "following the section rules exactly.\n\n" + video_prompt.strip()
    )


def build_user_context(duration_s: float, scene_context: str = "",
                       truncated: bool = False) -> str:
    lines = [
        f"The attached clip is exactly {duration_s:.2f} seconds long. "
        "Describe nothing past that point.",
    ]
    if truncated:
        lines.append(
            "(The clip was truncated to this length from a longer video.)"
        )
    context = (scene_context or "").strip()
    if context:
        lines.append(
            "Visual context from the editor (use it ONLY to decide what "
            "is diegetic; describe only what you hear):\n" + context
        )
    lines.append("Analyze the audio now, following the section rules exactly.")
    return "\n".join(lines)


def parse_response(text: str) -> Tuple[Dict[str, str], List[str]]:
    """Split the reply into the four sections; returns (sections, errors)."""
    errors: List[str] = []
    lines = text.strip().splitlines()
    positions = []
    for index, line in enumerate(lines):
        stripped = line.strip().rstrip(":")
        if line.strip().endswith(":") and stripped in SECTIONS:
            positions.append((stripped, index))
    found = [name for name, _ in positions]
    if found != list(SECTIONS):
        errors.append(
            "reply must contain exactly the four headers "
            f"{', '.join(SECTIONS)} in order, each alone on its line "
            f"(found: {', '.join(found) or 'none'})."
        )
        return {}, errors
    sections = {}
    for (name, start), (_, end) in zip(positions, positions[1:] + [("", len(lines))]):
        sections[name] = "\n".join(lines[start + 1: end]).strip()

    def _sentences(value: str) -> int:
        return len([s for s in re.split(r"[.!?]+(?:\s|$)", value) if s.strip()])

    soundscape = sections["overall_soundscape"]
    if soundscape != "N/A" and not 1 <= _sentences(soundscape) <= 5:
        errors.append("overall_soundscape must be 1-4 sentences.")
    music = sections["non_diegetic_music"]
    if music != "N/A" and not 1 <= _sentences(music) <= 4:
        errors.append("non_diegetic_music must be 1-3 sentences or exactly N/A.")
    for line in sections["dialogue"].splitlines():
        line = line.strip()
        if line and line != "N/A" and not re.match(
            r"^\((S\d+|lyrics)\) \[[^\]]+\] ", line
        ):
            errors.append(
                "dialogue lines must look like '(S1) [English] Exact "
                f"words.' or N/A (offending: {line[:60]})"
            )
            break
    return sections, errors


def build_retry_message(errors: List[str]) -> str:
    numbered = "\n".join(f"{i}. {e}" for i, e in enumerate(errors, 1))
    return (
        "Your reply violates the output contract:\n" + numbered +
        "\nReturn the full corrected reply - four sections, same rules."
    )
