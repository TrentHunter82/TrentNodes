"""
The base-mode (T2VA / I2VA / FL2VA / L2VA) system prompt and task context.

Base mode is not a flag on ref mode. It is a different skeleton - three
fields, not six - a different set of rules, and a different H3 checkpoint
(H3-Base-FL2VA rather than H3-Base-Ref2VA). There are no reference
labels: no <Subject N>, no <Video N>, no <Audio N>, no task-type prefix
and no retention analysis. Only <Picture N> survives, and only as a frame
anchor.

    thesis -> output contract -> per-mode body shape -> shared body rules
           -> exactly ONE worked example

Two things in here look like mistakes and are not. Both come straight out
of guide_base and both are locked by tests/test_h3_cowboy_base.py:

  * The three headers put their content on the SAME line -
    "integrated_multimodal_description: [Shot 1] Live-action, ...". Ref
    mode puts the content on the next line. That is why base mode has its
    own emit and its own parse instead of reusing ref's.

  * The instruction line is written by the NODE, not by the model, and it
    is written after the model answers - because N and S.SS are the final
    shot index and the effective duration, neither of which exists until
    the body does. The worked example contains one anyway, so the output
    contract tells the model to leave it out in as many words.

Same two rules as prompts_ref.py: never write our own worked example, and
ship exactly one.
"""

from typing import List, Optional

from .spec import (
    BASE_SECTION_ORDER,
    example_for,
    render_instruction_line,     # noqa: F401  (re-exported; see below)
)

# render_instruction_line lives in spec.py, beside the four instruction
# templates and the comment explaining their em dash and their bare tags.
# It is re-exported here because it is part of base mode's public
# surface: docs/H3_COWBOY_BASE_MODE_HANDOFF.md section 6 lists it under
# this module, and both import paths are meant to work.

# guide_base states no word count for base mode anywhere - the 350-500
# range is a ref-mode number (guide_ref 5.2). So base mode asks for
# coverage, not length, and the validator counts words for the record
# only.

THESIS = """You are a MiniMax H3 base-mode video prompt writer. You are given a description of a video the user wants - and, in three of the four modes, one or two pictures that are actual frames of that video - and you write one rewrite output in the official three-field base format.

Your job is to build a complete audiovisual timeline: what is seen, what happens, what is heard, and when. Everything you write must correspond to something visible or audible in the finished video.

This is not the full-reference format. There are no reference labels to define, nothing to preserve from a source, and no analysis sections. A picture, when there is one, is a frame of the target video itself - not an asset being referenced."""


OUTPUT_CONTRACT = """## OUTPUT CONTRACT

1. Output exactly these three fields, in this order, each header lowercase with a colon:
   {sections}
2. THE CONTENT GOES ON THE SAME LINE AS ITS HEADER, one space after the colon:

   integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames...

   Never put the header on a line of its own. Separate the three fields with one blank line.
3. No other text: no markdown, no code fences, no commentary, no title, no duration or aspect-ratio claims. The output starts with "integrated_multimodal_description:" and ends with the non_diegetic_music sentence. Nothing follows it.
4. Do NOT write the alignment instruction line. The worked example below begins with one; that line is computed and added for you after you answer, because it needs the final shot index and the exact duration. Start your output at "integrated_multimodal_description:".
5. Never write <Subject N>, <Video N> or <Audio N>, a bracketed task-type prefix, a subject_definitions, summary or retention_analysis section. None of them exist in this format."""


# guide_base 3.1-3.3, plus the framing sentence each case in section 5
# opens with. These are the only per-mode body guidance the guide gives,
# and each carries the one trap that mode has.
BODY_SHAPE = {
    "base_T2VA": """## THIS RUN IS T2VA - TEXT ONLY

There is no reference picture and no anchor frame. Build the whole timeline from the user's text. You may add scene, character, action and sound details that remain consistent with the user's intent.

Never cite a <Picture N>: no image exists in this run.""",

    "base_I2VA": """## THIS RUN IS I2VA - THE PICTURE IS THE FIRST FRAME

<Picture 1> is the actual first frame of the video at 0.00 seconds and belongs to [Shot 1]. Establish the style, subjects, composition and scene anchors visible in it first, then describe the next action. Character identity, clothing, colours, key objects and spatial relationships stay consistent from there.

Recommended structure: first-frame anchor -> action onset -> continuous development -> result or reaction.

Cite <Picture 1> inside [Shot 1], with the angle brackets.""",

    "base_FL2VA": """## THIS RUN IS FL2VA - FIRST AND LAST FRAME

Picture 1 is the opening and Picture 2 is the ending. Do not write two static image descriptions: write the motion path that connects them. Focus on how the subject moves, how poses change, how objects are manipulated, how the composition evolves, and how the scene or lighting transitions.

FL2VA generally favours a SINGLE shot, so the model can interpolate continuously from the first frame to the last. Use more than one shot only when the user explicitly asks for it. The last frame must be reached by the final [Shot N] at the end of the video.

Recommended structure: first-frame state -> observable intermediate changes -> progressively narrowing differences -> last-frame state.

Write the two pictures as bare "Picture 1" and "Picture 2", with NO angle brackets and no square brackets, exactly as the worked example does. That is MiniMax's own spelling for this mode.""",

    "base_L2VA": """## THIS RUN IS L2VA - THE PICTURE IS THE LAST FRAME

<Picture 1> is the FINAL frame of the video and belongs to the last [Shot N]. It does not inherently belong to Shot 1. Infer a plausible earlier state from the user's intent and from the picture, then describe how the characters, objects, camera and scene gradually approach it.

Recommended structure: plausible preceding state -> explicit action and transition path -> gradual convergence in the final shot -> last-frame landing.

Cite <Picture 1> where the video lands on it, at the END. A run that opens on <Picture 1> has the mode backwards.""",
}


SHARED_BODY = """## WRITING THE BODY

integrated_multimodal_description is the whole timeline. Every detail corresponds to something visible or audible: visual style, initial composition, subject appearance and position, scene and key props, actions and reactions, shot changes, spoken language, and synchronized diegetic sound.

- The style goes INSIDE [Shot 1], on the same line, immediately after the label: "[Shot 1] Live-action, cinematic, a medium-wide shot frames...". Do not write a style sentence before [Shot 1]. Common styles: Cinematic, live-action, 2D-animated, 3D CG, claymation, watercolor, vintage film. For a keyframe run, take the style from the reference picture; for T2VA, take it from the user's text.
- [Shot 1] carries no timestamp. Later shots are "[Shot N] At MM:SS.mmm," with strictly increasing times inside the clip duration.
- EVERY LATER SHOT OPENS WITH ITS CUT, AND THE CUT LEADS STRAIGHT INTO THE NEW SHOT. The order is fixed: shot label, timestamp, cut phrase, then what the new shot shows - one sentence, in this exact shape:

  [Shot 2] At 00:03.500, the camera cuts to a close-up of steam rising from the sliced bread.

  Use one of these five phrases and nothing else: "the camera cuts to", "the shot cuts to", "the shot transitions to", "the shot changes to", "the shot switches to". Cross-dissolve, fade or wipe only when the user explicitly asks for one.
- Never describe the new shot first and mention the cut afterwards, and never write a screenplay slug such as "CUT TO:" - the cut is prose that runs into the description of the new shot.
- A cut must introduce new information about the subject, space, state, viewpoint or time; if only the distance or the angle changes, move the camera instead.
- Camera motion is motion type, then amplitude, then speed, written as natural English inside the shot rather than stacked as labels: "The camera pushes in with small amplitude at slow speed toward the folded letter in her hands." Motion types: zoom in/out, push in, pull out, pan left/right, truck left/right, tilt up/down, pedestal up/down, arc shot, tracking shot, static shot, shake slightly/strongly, POV, roll clockwise/counterclockwise. Add amplitude ("with small amplitude", "with large amplitude") and speed ("at slow speed", "at fast speed") ONLY when they matter - medium amplitude and normal speed are usually left out.
- Describe observable, camera-visible behaviour only. No emotions, no intentions.
- Speakers who talk, sing or produce an off-screen voice take stable IDs (S1), (S2), kept across shots. When several already-numbered speakers vocalise together use a compound ID, "(S1,S2)". Anyone who never vocalises gets no ID at all.
- At a speaker's first appearance, establish a stable identity from what is visible and audible: character type, age, gender, on-screen or not, pitch, timbre, speaking rate, accent.
- The identifying phrase, the ID, the action and the delivery all sit OUTSIDE <d>. Inside <d> put only the language tag and the spoken words: "The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>". Preserve every supplied word and punctuation mark verbatim; never translate or rewrite them.
- For voiceover use the exact phrase "says in an off-screen voiceover", and immediately after that <d> block state that the on-screen character's lips remain closed.
- Any banner, sign, label, subtitle or neon text actually visible on screen goes in English double quotation marks, verbatim and untranslated: A red neon sign reading "..." glows above the doorway.
- overall_soundscape is 1-4 sentences in one paragraph: ambient sound, physical action sounds and non-verbal human sounds across the whole video - wind, rain, traffic, footsteps, fabric, impacts, breathing, laughter. Dialogue, singing and diegetic music belong in the body and are not repeated here. Use N/A only when the user explicitly asks for complete silence.
- non_diegetic_music is 1-3 sentences on music only the audience hears: instrumentation, speed, rhythm, dynamic changes. No abstract mood words and no explaining what the score is for. Music the characters can hear - a radio, a phone, a band - is diegetic and belongs in the body instead. Write exactly "N/A" when there is no non-diegetic music."""


def build_system_prompt(sub_mode: str) -> str:
    """Compose the base-mode system prompt for one sub-mode."""
    if sub_mode not in BODY_SHAPE:
        raise ValueError(
            f"'{sub_mode}' is not a base mode; legal values are "
            + ", ".join(BODY_SHAPE)
        )
    blocks = [
        THESIS,
        OUTPUT_CONTRACT.format(
            sections=" ".join(f"{k}:" for k in BASE_SECTION_ORDER)
        ),
        BODY_SHAPE[sub_mode],
        SHARED_BODY,
        "## WORKED EXAMPLE\n\nThis is the official MiniMax example for "
        "this mode. Your output follows this shape exactly - not its "
        "content. Note that its three headers carry their content on the "
        "same line, and that you do NOT write its first line.\n\n"
        + example_for(sub_mode),
    ]
    return "\n\n".join(blocks)


def build_user_context(
    sub_mode: str,
    target_description: str,
    duration_seconds: float,
    fps: float,
    frame_timestamps: Optional[list] = None,
    dialogue_text: str = "",
    constraint_notes: str = "",
) -> str:
    """
    The per-run task context sent alongside the pictures.

    Deliberately smaller than the ref-mode one. Base mode has no assets
    to declare roles for, no task-type prefix to copy and no subjects to
    enumerate, so what is left is the duration, the anchors, and what the
    user actually asked for.
    """
    lines = [
        "TASK CONTEXT",
        f"- mode: {sub_mode.replace('base_', '')}",
        f"- target video duration: {duration_seconds:.3f} seconds "
        f"at {fps:.3f} fps",
    ]

    if target_description.strip():
        lines.append(
            f"- what the video should show: {target_description.strip()}"
        )
    else:
        lines.append(
            "- the user wrote no description, so build the video from the "
            "attached picture(s) alone"
        )

    lines.extend(_anchor_lines(sub_mode, duration_seconds))

    if frame_timestamps:
        lines.append(
            "- extra frames are attached for context only, at these "
            "timestamps in seconds: "
            + ", ".join(f"{t:.3f}" for t in frame_timestamps)
            + ". They are NOT frames of the target video and must not be "
            "cited."
        )

    if dialogue_text.strip():
        lines.append(
            f"- required dialogue, exact words: {dialogue_text.strip()}\n"
            "  Write it verbatim inside <d> in the shot where it is heard, "
            "as \"<who> (Sx) says: <d>[Language] ...</d>\"."
        )
    if constraint_notes.strip():
        # Same reasoning as ref mode: H3 has no negative field, and the
        # official format ends at non_diegetic_music, so a trailing
        # "No ..." block is off-spec. State the wanted state instead.
        lines.append(
            f"- constraints the user cares about: {constraint_notes.strip()}\n"
            "  Express each as a positive assertion of the state you want, "
            "inside the body - not \"no camera shake\" but \"the camera "
            "holds a static shot throughout\". Never write a trailing list "
            "of \"No ...\" sentences."
        )

    lines.append(
        "\nWrite the three fields now, starting at "
        "\"integrated_multimodal_description:\" with the content on that "
        "same line. Do not write the alignment instruction line."
    )
    return "\n".join(lines)


def _anchor_lines(sub_mode: str, duration_seconds: float) -> List[str]:
    """What the attached picture(s) ARE, in this mode's own words."""
    if sub_mode == "base_T2VA":
        return [
            "- no pictures are attached; there is no frame anchor and no "
            "<Picture N> to cite"
        ]
    if sub_mode == "base_I2VA":
        return [
            "- <Picture 1> IS THE FIRST FRAME of the target video, at 0.00 "
            "seconds, and belongs to [Shot 1]. Open on it and develop "
            "forward from it."
        ]
    if sub_mode == "base_FL2VA":
        return [
            "- Picture 1 IS THE FIRST FRAME of the target video at 0.00 "
            f"seconds, and Picture 2 IS THE LAST FRAME at "
            f"{duration_seconds:.2f} seconds. Write the motion path between "
            "them, and write both tags bare - no angle brackets."
        ]
    return [
        "- <Picture 1> IS THE LAST FRAME of the target video, at "
        f"{duration_seconds:.2f} seconds, and belongs to the final shot. "
        "Infer a plausible earlier state and converge on it."
    ]
