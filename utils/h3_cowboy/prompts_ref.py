"""
The ref-mode (Ref2VA) system prompt and task context, composed per run.

Composed rather than one big constant, because the guides themselves are
composed: guide_ref 5.1 says "The basic format follows the Video Prompt
Writing Guide (T2VA / I2VA / FL2VA / L2VA)" and its header says it
"focuses on the reference labels, analysis sections, and format
differences specific to full-reference mode".

    thesis -> output contract -> label rules -> task-type rules
           -> retention rules -> shared body -> kind cards (declared only)
           -> exactly ONE worked example

Two rules govern this file:

  * Kind cards are injected only for kinds this run declares. A job with
    two people and a scene never reads the style card. That is what keeps
    a general prompt from bloating into uselessness.

  * The worked example is always MiniMax's own, and there is always
    exactly one. An example teaches shape far more strongly than a
    numbered rule does, so a wrong one is worse than a missing one, and
    two of different shapes get blended.
"""

from typing import List, Optional

from .spec import (
    KIND_CARDS,
    MULTI_SOURCE_RULE,
    REF_SECTION_ORDER,
    RETENTION_MARKERS_AUDIO,
    RETENTION_MARKERS_VISUAL,
    TASK_TYPES,
    example_for,
)
from .subjects import SubjectSpec, declared_kinds

# guide_ref 5.2: "For generation tasks, detailed_description is normally
# 350-500 English words." Editing runs are exempt (Phase 2).
DESCRIPTION_WORDS = (350, 500)

# ---------------------------------------------------------------------------
# Music-video mode
# ---------------------------------------------------------------------------
# guide_ref sections 5.4 and 6: lyrics live in <d> inside
# detailed_description and are never repeated in the audio sections; a
# reused track is <Audio N> with a fully_copy retention line; and verbal
# content that exists only inside a reused track is attributed to
# <Audio N> instead of being given a new (Sx) speaker ID.
#
# Carried over from the older node, which had this and this node did not.
# Rewritten for several subjects of any kind: the singer is whichever
# subject the run names, not an assumed <Subject 1>.

MUSIC_VIDEO_GUIDANCE = (
    "- MUSIC VIDEO MODE IS ON. The target is a music video, so the audio "
    "balance inverts from the usual documentary one:\n"
    "  * non_diegetic_music is the primary audio section and must never "
    "be \"N/A\". Give genre, instrumentation, tempo (a BPM when you can "
    "hear one), and how the track develops across the clip - intro, "
    "verse, chorus, drop, outro - and where those land against the shot "
    "list.\n"
    "  * overall_soundscape stays thin. A music video is scored wall to "
    "wall, so write only what is genuinely audible under the track: an "
    "instrument played on camera, a crowd, a room tone. Do not write "
    "footsteps, cloth rustle and impact sounds shot by shot; the music "
    "covers them.\n"
    "  * Cuts are musical. Where the shot list supports it, say the cut "
    "rhythm follows the beat rather than describing each cut as a "
    "narrative event.\n"
    "  * Performance is the action. Describe what the performer does to "
    "camera - the performance energy, the movement on the beat, where "
    "they look - with the same physical, observable language as any "
    "other shot."
)

MUSIC_VIDEO_TRACK = (
    "- the track, as the user describes it (build non_diegetic_music on "
    "this): {description}"
)

MUSIC_VIDEO_LYRICS = (
    "- LYRICS, exact words: {lyrics}\n"
    "  Write them inside the shot where they are sung, as "
    "<d>[Language] ...</d>, exactly as given and in their original "
    "language. Put the actual language of these words in the brackets - "
    "it is not always English. State that the performer's mouth movement "
    "matches the words. Split them across shots if the clip cuts "
    "mid-line. NEVER repeat lyrics in overall_soundscape or "
    "non_diegetic_music - they belong only inside <d> in "
    "detailed_description."
)

MUSIC_VIDEO_SINGER = (
    "  {tag} is the vocal source, so give it a speaker ID and keep the "
    "visual label with it: write \"{tag} (S1) sings, <d>[Language] "
    "...</d>\"."
)

MUSIC_VIDEO_NO_LYRICS = (
    "- lyrics: none supplied. Do not invent words. If a performer's "
    "mouth moves to the track, say the mouth moves in time with the "
    "music without intelligible lyrics, and write no <d> lines."
)

MUSIC_VIDEO_AUDIO_REUSE = (
    "- THE SONG IS SUPPLIED TO THE GENERATION AS <Audio 1>. For this run "
    "<Audio 1> is a valid reference tag alongside <Picture N>, <Video N> "
    "and <Subject N>. Define it on its own line in subject_definitions, "
    "give it the line \"<Audio 1>: fully_copy - <Audio 1> is reused as "
    "the target video's complete audience-only score.\" in "
    "retention_analysis, and build non_diegetic_music on that track "
    "rather than inventing one. Sung words that exist only inside "
    "<Audio 1> belong to <Audio 1>: do not give the track a new (Sx) "
    "speaker ID."
)

MUSIC_VIDEO_HEARD = (
    "- THE TRACK IS ATTACHED FOR YOU TO HEAR. Build non_diegetic_music "
    "on what it actually sounds like, not on what the frames imply: the "
    "real genre, the instruments you can pick out, the tempo, and where "
    "the arrangement changes against the shot list. Do not transcribe "
    "sung words into <d> unless the lyrics are supplied above."
)


THESIS = """You are a MiniMax H3 full-reference (Ref2VA) rewrite writer. You are given reference assets - images, video, audio - and a description of the target video the user wants, and you write one rewrite output in the official six-section full-reference format.

Your job is to state what the target video contains and, for every piece of referenced content, exactly how much of it survives into that video.

Referenced content is anything reusable and visible: a person, an animal, an object or prop, a scene or environment, a garment, an interface, a visual effect, a visual style, an action, an expression, or a pose. Every one of those is handled identically - it gets a <Subject N> label, one definition line, one retention line, and a described appearance in the body. Nothing about this task is specific to people."""


OUTPUT_CONTRACT = """## OUTPUT CONTRACT

1. Output exactly six sections, in this order, each header lowercase with a trailing colon on its own line:
   {sections}
2. No other text: no markdown, no code fences, no commentary, no title, no duration or aspect-ratio claims. The output starts with "subject_definitions:" and ends with the non_diegetic_music section. Nothing follows it - no trailing notes, no "No ..." sentences.
3. In subject_definitions and retention_analysis, give every reference label its own line. One line per label, never a run-on paragraph."""


LABEL_RULES = """## REFERENCE LABELS

<Subject N> is visible content reused in the target video. <Picture N> is a reference image, <Video N> a reference video, <Audio N> an audio signal.

- Once a label is assigned it keeps the same meaning in every section. Never renumber and never invent a label for an asset you were not given.
- Define each label on its own line: what it denotes, its reference role, and the main features to follow. Name its source asset inside that line.
- If a <Picture N> or <Video N> only identifies where a subject came from and is not used separately, cite it INSIDE that subject's line and give it no line of its own.
- Give a <Picture N> its own line only when the image itself is a concrete frame of the target video - a first frame, keyframe, last frame, or composition anchor - or a storyboard reference.
- <Video N> is for whole-video relationships only: editing a source, continuing from one, or referencing its camera movement, cuts, rhythm or temporal structure. A person, object, scene or action reused from a video still belongs under <Subject N>.
- {multi_source}
- Do not introduce new labels in summary."""


TASK_TYPE_RULES = """## SUMMARY TASK-TYPE PREFIX

summary is one short paragraph and MUST begin with a square-bracketed task-type prefix, copied exactly as the task context gives it. The legal values are: {types}. Several combine with " + " and are never repeated."""


RETENTION_RULES = """## RETENTION MARKERS

retention_analysis has one line per label defined in subject_definitions, shaped "<Tag N> (scope): marker - explanation". The scope parenthetical says where the label applies, e.g. "(appears in [Shot 1], [Shot 3])" for a subject, "([Shot 1] first frame)" for a picture, "(cut and pacing structure)" for a video.

The marker is one of these fixed values and nothing else:
- visible content (<Subject N>, <Picture N>, <Video N>): {visual}
- audio (<Audio N>): {audio}

Choose the marker within the role the label already has in subject_definitions. New actions, backgrounds or events in the target video are not losses of fidelity. Never write an (Sx) speaker ID in retention_analysis."""


SHARED_BODY = """## WRITING THE BODY

detailed_description opens with one or two sentences of overall style, BEFORE the first [Shot 1]. Then the shot timeline.

- Label the opening shot [Shot 1] with no timestamp. Later shots are "[Shot N] At MM:SS.mmm," with strictly increasing times inside the clip duration.
- EVERY LATER SHOT OPENS WITH ITS CUT, AND THE CUT LEADS STRAIGHT INTO THE NEW SHOT. The order is fixed: shot label, timestamp, cut phrase, then what the new shot shows - one sentence, in this exact shape:

  [Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1.

  Use one of these five phrases and nothing else: "the camera cuts to", "the shot cuts to", "the shot transitions to", "the shot changes to", "the shot switches to". Cross-dissolve, fade or wipe only when the user explicitly asks for one. Never describe the new shot first and mention the cut afterwards, and never write a screenplay slug such as "CUT TO:".
- At a subject's first clear appearance, describe its referenced characteristics, where it sits in the frame when that is meaningful, and what it is doing. In later shots reuse the same label without redefining it. A subject does not have to appear in every shot.
- Camera direction is prose inside the shot, from the official vocabulary: static, push in, pull out, pan left/right, tilt up/down, truck left/right, pedestal up/down, arc, roll clockwise/counterclockwise, tracking shot, POV, shake slightly/strongly. Qualify with the fixed phrases "with small amplitude" or "with large amplitude" and "at slow speed" or "at fast speed". One dominant move per shot.
- Describe observable, camera-visible behaviour only. No emotions, no intentions.
- Dialogue and lyrics go inside the shot as: <Subject N> (Sx) says, <d>[Language] Exact words.</d> Keep the visual label with the speaker ID. [Language] is the language of the words themselves. Assign (Sx) in the order vocal events occur; content that never vocalises gets no ID. Preserve supplied words verbatim - never translate or rewrite them.
- overall_soundscape is 1-4 sentences of ambient and physical sound tied to visible events. Use N/A only for requested silence.
- non_diegetic_music is 1-3 sentences on instrumentation, tempo and how the score develops, or exactly "N/A" when there is genuinely no audience-only music. Never repeat dialogue or lyrics in either audio section.
- detailed_description is normally {lo}-{hi} words. Prioritise action and camera accuracy over decoration."""


def build_system_prompt(
    subjects: List[SubjectSpec], task_types: Optional[List[str]] = None
) -> str:
    """Compose the ref-mode system prompt for this run's declared kinds."""
    blocks = [
        THESIS,
        OUTPUT_CONTRACT.format(
            sections=" ".join(f"{k}:" for k in REF_SECTION_ORDER)
        ),
        LABEL_RULES.format(multi_source=MULTI_SOURCE_RULE),
        TASK_TYPE_RULES.format(types=", ".join(TASK_TYPES)),
        RETENTION_RULES.format(
            visual=", ".join(RETENTION_MARKERS_VISUAL),
            audio=", ".join(RETENTION_MARKERS_AUDIO),
        ),
        SHARED_BODY.format(lo=DESCRIPTION_WORDS[0], hi=DESCRIPTION_WORDS[1]),
    ]

    kinds = declared_kinds(subjects)
    if kinds:
        cards = "\n".join(f"- {KIND_CARDS[k]}" for k in kinds)
        blocks.append(
            "## FEATURES TO FOLLOW, BY SUBJECT KIND\n\n"
            "These are the kinds this run actually uses. For each subject, "
            "the definition line names the features listed here that are "
            "visible in its source.\n\n" + cards
        )

    blocks.append(
        "## WORKED EXAMPLE\n\nThis is an official MiniMax example. Your "
        "output follows this shape exactly - not its content.\n\n"
        + example_for("ref", task_types or [])
    )
    return "\n\n".join(blocks)


def build_user_context(
    subjects: List[SubjectSpec],
    target_description: str,
    duration_seconds: float,
    fps: float,
    task_type: str,
    frame_timestamps: Optional[list] = None,
    cut_timestamps: Optional[list] = None,
    cut_source: str = "local",
    cut_kinds: Optional[list] = None,
    full_clip: bool = False,
    has_video: bool = False,
    audio_available: bool = False,
    dialogue_text: str = "",
    constraint_notes: str = "",
    image_tags: Optional[List[str]] = None,
    video_role: str = "subject_source",
    music_video: bool = False,
    lyrics: str = "",
    music_description: str = "",
    music_is_reference: bool = False,
) -> str:
    """
    The per-run task context sent alongside the images.

    cut_source is where cut_timestamps came from: "measured" for a real
    shot-boundary detector (Cut Detective), "local" for the node's own
    frame-difference guess. A measured list is stated as fact; a local
    one stays a hint the model may override from the frames.
    """
    lines = [
        "TASK CONTEXT",
        f"- clip duration: {duration_seconds:.3f} seconds at {fps:.3f} fps",
        f"- summary MUST begin with this exact prefix: [{task_type}] "
        "Copy it character for character, brackets included, then continue "
        "the paragraph.",
    ]

    if target_description.strip():
        lines.append(
            f"- what the target video should show: {target_description.strip()}"
        )

    lines.append("- the subjects to define, in this order:")
    for subject in subjects:
        lines.append(f"    {subject.describe()}")
        if not subject.is_spatial:
            lines.append(
                f"      ({subject.kind} has no position in the frame - do "
                "not write one for it)"
            )

    if image_tags:
        lines.append(
            "- images attached, in order: "
            + ", ".join(f"<{t}>" for t in image_tags)
        )
    if has_video:
        lines.append(_video_role_line(video_role))

    lines.extend(_cut_lines(
        cut_timestamps or [], cut_source, cut_kinds, duration_seconds,
        full_clip, frame_timestamps or [],
    ))

    if dialogue_text.strip():
        lines.append(
            f"- required dialogue, exact words: {dialogue_text.strip()}\n"
            "  Write it verbatim inside <d> in the shot where it is heard, "
            "as \"<Subject N> (Sx) says, <d>[Language] ...</d>\"."
        )

    if music_video:
        lines.extend(_music_lines(
            subjects, lyrics, music_description, music_is_reference,
            audio_available,
        ))
    elif audio_available:
        lines.append(
            "- the clip's audio track is attached. Describe what is "
            "genuinely audible; do not transcribe speech into a spoken line "
            "unless the words are supplied above."
        )
    if constraint_notes.strip():
        # The old node appended a trailing "No ..." block for this, which
        # is off-spec - nothing follows non_diegetic_music in any official
        # example. The guide's own mechanism for "do not change X" is the
        # retention marker, so these are folded in positively instead.
        lines.append(
            f"- constraints the user cares about: {constraint_notes.strip()}\n"
            "  Express each as a positive assertion of the state you want, "
            "inside retention_analysis or the body - not \"no wardrobe "
            "changes\" but \"the charcoal utility jacket remains exactly as "
            "defined through every shot\". Never write a trailing list of "
            "\"No ...\" sentences."
        )

    lines.append(
        "\nWrite the complete H3 Ref2VA prompt now, following every rule "
        "in the output contract."
    )
    return "\n".join(lines)


def _music_lines(
    subjects: List[SubjectSpec], lyrics: str, music_description: str,
    music_is_reference: bool, audio_available: bool,
) -> List[str]:
    """
    The music-video block, in the order the model reads best.

    Mode first, then the track, then the words. The singer is whichever
    subject can actually sing - the first person or animal declared - and
    when nothing in the run can, no speaker ID is asked for at all.
    """
    lines = [MUSIC_VIDEO_GUIDANCE]
    if music_description.strip():
        lines.append(MUSIC_VIDEO_TRACK.format(
            description=music_description.strip()
        ))
    if music_is_reference:
        lines.append(MUSIC_VIDEO_AUDIO_REUSE)
    if lyrics.strip():
        lines.append(MUSIC_VIDEO_LYRICS.format(lyrics=lyrics.strip()))
        singer = next(
            (s for s in subjects if s.kind in ("person", "animal")), None
        )
        # A reused track already owns its vocal, so a second speaker ID
        # for the same voice is exactly what guide_ref 6 warns against.
        if singer is not None and not music_is_reference:
            lines.append(MUSIC_VIDEO_SINGER.format(tag=singer.tag))
    else:
        lines.append(MUSIC_VIDEO_NO_LYRICS)
    if audio_available:
        lines.append(MUSIC_VIDEO_HEARD)
    return lines


def _video_role_line(role: str) -> str:
    """
    What the attached video is for, in the words the guide uses.

    guide_ref 2.3 reserves <Video N> for whole-video relationships, so a
    video that merely shows a subject must NOT get a line of its own -
    saying that outright is what stops the model inventing one.
    """
    if role == "edit_source":
        return (
            "- <Video 1> IS THE SOURCE BEING EDITED. The target video is "
            "that video with the requested changes applied. Give <Video 1> "
            "its own line in subject_definitions and its own line in "
            "retention_analysis. Describe the source's real content as it "
            "already is, apply only the requested changes, and say "
            "everything else is preserved. Do not re-plan the scene and do "
            "not invent cuts the source does not have. Open the body with "
            "\"[Shot 1] The shot begins from the source <Video 1>, ...\"."
        )
    if role == "continuation_source":
        return (
            "- <Video 1> IS THE SOURCE BEING CONTINUED. The target video "
            "begins from the state that video ends in and carries on from "
            "there; it does not replay it. Give <Video 1> its own line in "
            "subject_definitions and in retention_analysis."
        )
    if role == "structure_reference":
        return (
            "- <Video 1> supplies structure only: its camera movement, "
            "cuts, rhythm and temporal shape. It is not being edited or "
            "continued. Give it its own line in subject_definitions and a "
            "\"<Video 1> (cut and pacing structure): ...\" retention line. "
            "Anything visible reused from it is still a <Subject N>."
        )
    return (
        "- <Video 1> is attached only as the source of a subject's "
        "appearance or motion. Cite it INSIDE that subject's line and give "
        "it no line of its own and no retention line."
    )


def _cut_lines(
    cut_timestamps, cut_source, cut_kinds, duration, full_clip, frame_times,
) -> List[str]:
    """The shot-structure block, measured or guessed."""
    lines = []
    if full_clip:
        lines.append(
            "- THE COMPLETE SOURCE CLIP IS ATTACHED. Read the motion, cut "
            "timing and camera movement directly from it."
        )
    elif frame_times:
        lines.append(
            "- sampled frame timestamps (seconds): "
            + ", ".join(f"{t:.3f}" for t in frame_times)
        )

    if cut_source == "measured" and cut_timestamps:
        count = len(cut_timestamps)
        kinds = list(cut_kinds or [])
        rows = []
        for i, start in enumerate(cut_timestamps):
            end = (
                cut_timestamps[i + 1] if i + 1 < len(cut_timestamps)
                else duration
            )
            kind = kinds[i] if i < len(kinds) else (
                "start" if i == 0 else "hard cut"
            )
            entry = "opening shot" if i == 0 else f"entered by a {kind}"
            rows.append(
                f"    Shot {i + 1}: starts {start:.3f}s, "
                f"runs {max(0.0, end - start):.3f}s, {entry}"
            )
        lines.append(
            f"- MEASURED SHOT LIST - the clip has exactly {count} "
            f"{'shot' if count == 1 else 'shots'}. A shot-boundary detector "
            "read every frame to produce this. Write exactly these shots, "
            "in this order, with these start times:\n" + "\n".join(rows)
        )
    elif cut_timestamps:
        lines.append(
            "- detected hard-cut boundaries (seconds, a local guess - trust "
            "the frames over it): "
            + ", ".join(f"{t:.3f}" for t in cut_timestamps)
        )
    else:
        lines.append(
            "- no cuts detected; treat the clip as one continuous shot "
            "unless the frames clearly show otherwise"
        )
    return lines
