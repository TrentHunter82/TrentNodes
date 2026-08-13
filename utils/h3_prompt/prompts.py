"""
System prompt and user-context builder for the H3 Auto Prompt Generator.

The emitted prompt follows the OFFICIAL MiniMax H3 REF2VA format
(huggingface.co/MiniMaxAI/MiniMax-H3, docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
and docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md): six lowercase snake_case
sections in fixed order, one line per reference label in
subject_definitions and retention_analysis, a mandatory square-bracketed
task-type prefix on summary, fixed retention markers, camera/dialogue/
visible text inside detailed_description, and [Shot N] At MM:SS.mmm
labels. The prompt ends with the non_diegetic_music section. No
"Create a X-second..." preamble - duration, fps, and aspect ratio come
from the H3 node/UI parameters, not the prompt.

The trailing headerless block of "No ..." exclusion sentences is NOT
part of that format. The word "exclusion" appears nowhere in MiniMax's
guides, and no official example writes anything after non_diegetic_music.
It is a practitioner habit, because H3 has no negative-prompt field, and
it stays available behind AssemblyContext.append_exclusions - off by
default so the emitted prompt matches the spec.

The system prompt is a plain constant. Never .format() it (it contains
literal braces-free text but the rule keeps edits safe); all dynamic
content goes into the user message via build_user_context().
"""

from typing import Optional

SECTION_ORDER = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
]

# Pool used to pad the trailing exclusions block to MIN_EXCLUSIONS when
# append_exclusions is on. Off-spec, opt-in; see the module docstring.
# Ordered by usefulness for character-replacement work.
STOCK_EXCLUSIONS = [
    "No face morphing or identity drift of the subject in any frame.",
    "No changes to the core body movement and timing defined by <Video 1>.",
    "No alteration of the wardrobe shown in <Picture 1>.",
    "No hair style or hair color changes in any frame.",
    "No glasses or added accessories that are not in <Picture 1>.",
    "No extra characters, duplicated subjects, or added props.",
    "No extra spoken lines, subtitles, captions, or on-screen text.",
    "No morphing, compositing seams, dissolves, or crossfades.",
    "No copying of the background, pose, or lighting from <Picture 1>.",
    "No copying of any identity or facial features from the performer in <Video 1>.",
    "No slow motion or speed changes that are not present in <Video 1>.",
    "No camera moves beyond those present in <Video 1>.",
]

# Official summary task-type prefixes (VIDEO_PROMPT_WRITING_GUIDE_ref_en
# section 3). The prefix is mandatory: "It begins with a square-bracketed
# task-type prefix". Several are joined with " + " and never repeated.
TASK_TYPES = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)

# Retention markers are fixed English values, and the two sets do not mix
# (guide section 4.1 for visible content, 4.2 for audio).
RETENTION_MARKERS_VISUAL = (
    "fully_preserved", "partially_preserved", "attribute_transfer",
    "weak_reference",
)
RETENTION_MARKERS_AUDIO = (
    "fully_copy", "partially_copy", "reference", "weak_reference",
)

# First-frame alignment instruction. H3 reads a picture-to-timeline
# relationship from a first line above the sections, and the guide fixes
# two forms (VIDEO_PROMPT_WRITING_GUIDE_base_en section 2.1). I2VA is
# used verbatim, at 0.00 seconds and [Shot 1]. Any other moment is the
# L2VA form, which is the one the guide parameterizes. Writing the I2VA
# wording with a substituted time produces a sentence in neither form.
# The assembler prepends it rather than the VLM writing it - strip_wrapper
# deletes everything above subject_definitions, and a fixed string cannot
# drift.
ALIGNMENT_HOOK_I2VA = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)
ALIGNMENT_HOOK_L2VA = (
    "How the reference pictures align with the target video - "
    "<Picture 1> (from [Shot {shot}]) aligns with the {seconds:.2f}-second "
    "mark of the target video."
)

MIN_EXCLUSIONS = 8
# A TrentNodes budget, not a model limit: MiniMax states no character
# cap anywhere. It exists so a runaway description gets trimmed instead
# of shipped. The word counts below ARE official (guide section 5.2:
# "normally 350-500 English words").
MAX_PROMPT_CHARS = 7000
TARGET_PROMPT_CHARS = 5500
DETAILED_DESCRIPTION_WORDS = (350, 500)
# Official sentence budgets for the two audio sections.
SOUNDSCAPE_SENTENCES = (1, 4)
NON_DIEGETIC_SENTENCES = (1, 3)

# "upgraded" profile: practices merged 2026-08 from a battle-tested
# API-side prompt (positive assertions, tight budget, speed-worded
# camera, cut re-anchoring, resolved ending). Section format and
# <Picture 1>/<Video 1> tags stay official.
PROFILES = ("official", "upgraded")
UPGRADED_DESCRIPTION_WORDS = (120, 300)
UPGRADED_TARGET_CHARS = 3000

# Music-video mode. Sourced from the official guide
# (huggingface.co/MiniMaxAI/MiniMax-H3, VIDEO_PROMPT_WRITING_GUIDE_ref_en
# sections 5.4 and 6): lyrics live in <d> inside detailed_description and
# are never repeated in the audio sections; a reused track is <Audio N>
# with a fully_copy retention line; and verbal content that exists only
# inside a reused track is attributed to <Audio N> rather than given a
# new (Sx) speaker ID.
MUSIC_VIDEO_GUIDANCE = (
    "- MUSIC VIDEO MODE IS ON. The target is a music video, so the "
    "audio balance inverts from the usual documentary one:\n"
    "  * non_diegetic_music is the primary audio section and must "
    "never be \"N/A\". Give genre, instrumentation, tempo (a BPM when "
    "you can hear one), and how the track develops across the clip - "
    "intro, verse, chorus, drop, outro - and where those land against "
    "the shot list.\n"
    "  * overall_soundscape stays thin. A music video is scored wall "
    "to wall, so write only what is genuinely audible under the track: "
    "an instrument played on camera, a crowd, a room tone. Do not "
    "write footsteps, cloth rustle and impact sounds shot by shot; the "
    "music covers them.\n"
    "  * Cuts are musical. Where the shot list supports it, say the "
    "cut rhythm follows the beat rather than describing each cut as a "
    "narrative event.\n"
    "  * Performance is the action. Describe what the subject does to "
    "camera - the performance energy, the movement on the beat, where "
    "they look - with the same physical, observable language as any "
    "other shot."
)

MUSIC_VIDEO_LYRICS = (
    "- LYRICS, exact words: {lyrics}\n"
    "  Write them inside the shot where they are sung, as "
    "<d>[Language] ...</d>, exactly as given and in their original "
    "language. Put the actual language of these words in the brackets - "
    "it is not always English. State that the subject's mouth movement "
    "matches the words. Split them across shots if the clip cuts "
    "mid-line. NEVER repeat lyrics in overall_soundscape or "
    "non_diegetic_music - they belong only inside <d> in "
    "detailed_description."
)

MUSIC_VIDEO_SINGER = (
    "  The subject is the vocal source, so give them a speaker ID and "
    "keep the visual label with it: write \"{name} <Subject 1> (S1) "
    "sings, <d>[Language] ...</d>\"."
)

MUSIC_VIDEO_AUDIO_REUSE = (
    "- THE SONG IS SUPPLIED TO THE GENERATION AS <Audio 1>. For this "
    "run <Audio 1> is a fourth valid reference tag alongside "
    "<Picture 1>, <Video 1> and <Subject 1>. Define it on its own line "
    "in subject_definitions, give it the line \"<Audio 1>: fully_copy "
    "- <Audio 1> is reused as the target video's complete "
    "audience-only score.\" in retention_analysis, and write "
    "non_diegetic_music as a statement that <Audio 1> is reused "
    "directly rather than as a description of music to invent. A voice "
    "that exists only inside <Audio 1> is NOT a separate speaker: "
    "attribute it to the track, e.g. \"When <Audio 1> reaches the "
    "phrase <d>[Language] ...</d>, {name} <Subject 1> mouths the words "
    "in sync\". Do not assign an (Sx) ID to it."
)

SOUNDSCAPE_GUIDANCE = {
    "fight": (
        "Physical combat audio tied to visible action: body impacts, cloth "
        "movement, footwork scuffs, breath and grunts of effort, objects "
        "struck or dropped exactly when contact is visible. Room or "
        "environment tone underneath."
    ),
    "ambient": (
        "Environment-first audio: room tone or outdoor ambience, weather, "
        "distant traffic or machinery, footsteps and cloth movement "
        "matching the subject's visible motion. No speech."
    ),
    "dialogue": (
        "Speech-forward audio: clear voice presence for visible speakers, "
        "modest room tone underneath, movement sounds kept below the "
        "voice. Only include spoken lines that are explicitly provided."
    ),
}


SYSTEM_PROMPT = """You are an expert Minimax H3 prompt director. You analyze video frames plus one identity reference image, and you write one production-ready H3 REF2VA prompt that replaces the main performer in the video with the person shown in the reference image, while copying the video's motion, framing, and cut structure exactly.

## UNBREAKABLE RULES

1. Output exactly six sections, in this order, each header lowercase with a trailing colon on its own line: subject_definitions: summary: retention_analysis: detailed_description: overall_soundscape: non_diegetic_music:
2. In subject_definitions and retention_analysis, give every reference label its own line. One line per label, never a run-on paragraph.
3. No other text: no markdown, no code fences, no commentary, no title, no duration/fps/aspect-ratio claims. The output starts with "subject_definitions:" and ends with the non_diegetic_music section.
4. Reference tags are exactly <Picture 1> (identity reference image) and <Video 1> (motion source video). The person is <Subject 1>. Never renumber, never invent tags.
5. In detailed_description, label shots as [Shot 1] for the opening shot (no timestamp), then [Shot 2] At MM:SS.mmm, [Shot 3] At MM:SS.mmm, and so on. Times are when the shot starts, strictly increasing, inside the clip duration given in the task context. One [Shot N] per hard cut; do not invent cuts and do not merge two shots into one. When the task context supplies a MEASURED SHOT LIST, it came from a shot-boundary detector that watched every frame: write exactly that many shots, with exactly those start times, in that order. Otherwise derive the times from the frame timestamps you are given.
6. Write the subject's name followed by <Subject 1> in EVERY shot, e.g. "Aria Voss <Subject 1> pivots...". Also state the subject's position in frame in EVERY shot (e.g. "center frame", "left third, midground", "upper right quadrant, far background").
7. Name the exact wardrobe from the task context in subject_definitions, again in retention_analysis, and at least once in detailed_description.
8. Camera direction is prose inside each shot, using the official move vocabulary only: static, push in, pull out, pan left/right, tilt up/down, truck left/right, pedestal up/down, arc, roll clockwise/counterclockwise, tracking shot, POV, shake slightly/strongly. Qualify a move with the fixed phrases "with small amplitude" or "with large amplitude" and "at slow speed" or "at fast speed" - e.g. "the camera pushes in with small amplitude at slow speed". ONE dominant camera move per shot. Never stack conflicting moves.
8a. For an ordinary cut, use one of these exact phrasings: "the camera cuts to", "the shot cuts to", "the shot transitions to", "the shot changes to", "the shot switches to".
9. Describe observable, camera-visible behavior only. No emotions, no intentions. Weak: "she is furious". Strong: "her arm snaps forward and she shoves the crate off the table".
10. When the subject is distant, small in frame, back-turned, partially blocked, or seen from overhead, say so and state that identity and wardrobe remain locked to <Picture 1> in those moments.
11. Do not give the subject glasses or accessories unless they are visible in <Picture 1>.
12. detailed_description is 350-500 words. Prioritize action and camera accuracy over decoration.
13. Dialogue: only write a spoken line if the task context supplies one, formatted inside the shot as: <Subject 1> (S1) says, <d>[Language] Exact words.</d> Keep the visual label with the speaker ID. [Language] is the language of the words themselves, not always English. Never write an (Sx) ID in retention_analysis. If the context supplies none, write no <d> lines; if lips visibly move in the source, state that the mouth moves without intelligible speech and that no dialogue audio is generated.
14. overall_soundscape is 1-4 sentences of diegetic sound tied to visible events. non_diegetic_music is 1-3 sentences describing score construction (instrumentation, tempo, entries/exits), or exactly "N/A" when there is genuinely no non-diegetic music.

## SECTION CONTENT GUIDE

subject_definitions: One line per label. Line 1 defines <Subject 1> as the named person exactly as shown in <Picture 1>: face, hair style and color, skin tone, facial structure, apparent age, body proportions, overall likeness, and the exact wardrobe items, citing <Picture 1> as the source inside that line. Line 2 defines <Video 1> as supplying only the body movement, camera angles, framing, cut rhythm, and action of the original sequence - never the performer's identity or facial features. Do NOT give <Picture 1> a line of its own: an image that only defines a character is cited inside the <Subject 1> line, never as a standalone entry.

summary: One short paragraph, beginning with the square-bracketed task-type prefix given in the task context, copied exactly. After the prefix: generate a new video in which the named subject is continuously tracked through every frame of the scene from <Video 1>, fully preserving identity and wardrobe from <Picture 1>.

retention_analysis: One line per label, each "<Tag N> (scope): marker - explanation". The scope parenthetical says where the label applies. The marker is one of these fixed values and nothing else - for visible content: fully_preserved, partially_preserved, attribute_transfer, weak_reference; for audio: fully_copy, partially_copy, reference, weak_reference. Write "<Subject 1> (appears in [Shot 1], [Shot 2], ...): fully_preserved - ..." naming every shot the subject appears in, and "<Video 1> (cut and pacing structure): attribute_transfer - ...". Add the obscured/distance identity-lock language specific to what the frames actually show. Never write an (Sx) speaker ID here.

detailed_description: Open with one sentence of grounded cinematic style from the task context. Then the [Shot N] timeline per rules 5-13.

## WORKED EXAMPLE (fictional subject; your output follows this shape exactly)

subject_definitions:
<Subject 1> is Aria Voss, whose appearance comes from <Picture 1>: exact facial identity, dark shoulder-length hair, skin tone, facial structure, apparent age, body proportions, overall likeness, and the exact wardrobe of a charcoal utility jacket over a slate-gray tee, black cargo pants, and scuffed black combat boots.
<Video 1> is the source video supplying only the exact body movement, camera angles, framing, cut rhythm, and overall action of the original sequence, never the performer's identity or facial features.

summary:
[reference generation] Generate a new video in which Aria Voss is continuously tracked and followed through every frame of the warehouse walk-and-turn from <Video 1>, while fully preserving Aria Voss's identity and wardrobe from <Picture 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - preserve Aria Voss's exact facial identity, hair, skin tone, body proportions, and likeness with zero drift in every single frame, and fully preserve the charcoal utility jacket, slate-gray tee, black cargo pants, and combat boots.
<Video 1> (cut and pacing structure): attribute_transfer - generate the body movement, positioning, and camera work according to <Video 1> while always keeping Aria Voss clearly tracked, including the moment she turns away and her face leaves view; identity and wardrobe stay locked while she is back-turned.

detailed_description:
The target video has the grounded cinematic appearance of a gritty handheld thriller with overcast toplight and a desaturated color grade. [Shot 1] A tracking shot moves backward with small amplitude at slow speed, holding Aria Voss <Subject 1> center frame in a medium shot as she strides toward the camera down a warehouse aisle, arms swinging naturally, boots striking the concrete. [Shot 2] At 00:03.250, the camera cuts to a static wide shot from the side; Aria Voss <Subject 1> occupies the left third of frame in the midground, stops beside a stack of pallets, plants her left hand on the top pallet, and turns away from camera toward the loading door; while she is back-turned her identity and wardrobe remain locked to <Picture 1>. Her mouth does not move and no dialogue is spoken.

overall_soundscape:
Concrete-floored warehouse ambience with a low ventilation hum. Boot footfalls land exactly on Aria Voss's visible steps, cloth movement from the jacket on each arm swing, a soft palm slap when her hand meets the pallet.

non_diegetic_music:
N/A

## ANALYSIS METHOD

1. Establish the shot structure. If the task context carries a MEASURED SHOT LIST, that is the structure: one [Shot N] per listed shot, at the listed start time, and no others. If it does not, read the frame timestamps and place cuts at the large visual discontinuities between adjacent sampled frames.
1a. A listed shot entered by a dissolve, fade, wipe or other gradual transition is still its own [Shot N] at its listed start time. Write the incoming shot plainly with one of the ordinary cut phrasings from rule 8a - never describe the transition effect itself, because H3 renders what you describe. A listed sudden jump is a jump cut inside continuous action: keep the location and the wardrobe identical across it and let only the framing or the pose jump.
2. For each shot: identify the one dominant camera move, the subject's frame position, direction of travel, and every visible physical action in order.
3. If motion between frames is very small, treat it as slow or deliberate movement and say so; do not describe the subject as frozen.
4. Build the six sections. Target under 5500 characters total.
5. Self-check against every unbreakable rule, then output the prompt text only."""


UPGRADE_OVERRIDES = """

## UPGRADED PROFILE OVERRIDES

These overrides replace the numbered rules above wherever they conflict. Every other rule stands, including the six-section format and the <Picture 1> / <Video 1> / <Subject 1> tags.

A. Constraints: express every constraint as a positive assertion of the state you want, woven into retention_analysis or detailed_description: "the aisle stays completely empty of other people", "her charcoal utility jacket remains exactly as shown in <Picture 1> through every frame", "her face holds the exact likeness from <Picture 1> even while she is back-turned".
B. Length: target 1,800-3,000 characters total. detailed_description is 120-300 words, below the official 350-500 on purpose. Compress secondary environmental detail first; never compress the central action, the camera work, or the ending.
C. Camera: keep the official move names and the fixed amplitude and speed phrases from rule 8, then add the physical result: "the camera pushes in with small amplitude at slow speed, keeping her face the same size in frame". Name the shot size plainly (wide, medium, medium close-up, close-up).
D. Carry across cuts: when a shot reuses the subject, a prop, a garment, or the setting from an earlier shot, say so explicitly: "the same pallet stack from Shot 1", "still wearing the charcoal jacket from <Picture 1>". Re-anchoring is what holds identity steady across a cut.
E. Ending: the final sentences of detailed_description resolve the scene completely: final pose, the subject's position in frame, camera framing, environmental state, and whether the last image holds."""


def get_system_prompt(profile: str = "official") -> str:
    """System prompt for a profile: 'official' or 'upgraded'."""
    if profile == "upgraded":
        return SYSTEM_PROMPT + UPGRADE_OVERRIDES
    return SYSTEM_PROMPT


def shot_index_for_time(seconds: float, shot_starts: list) -> int:
    """
    1-based index of the shot containing `seconds`.

    The hook names the target-video shot the reference aligns with, so
    an alignment time inside shot 3 must say [Shot 3], not [Shot 1].
    """
    index = 1
    for i, start in enumerate(shot_starts or [0.0], start=1):
        if seconds + 1e-6 >= start:
            index = i
        else:
            break
    return index


def build_alignment_hook(seconds: float, shot_index: int = 1) -> str:
    """
    Render the official alignment instruction for this moment.

    0.00s on [Shot 1] is exactly the I2VA case, so that sentence is used
    verbatim. Anything else is the L2VA form, the one the guide writes
    with a variable shot and mark.
    """
    seconds = max(0.0, float(seconds))
    shot = max(1, int(shot_index))
    if seconds < 0.005 and shot == 1:
        return ALIGNMENT_HOOK_I2VA
    return ALIGNMENT_HOOK_L2VA.format(seconds=seconds, shot=shot)


def build_task_type(
    music_is_reference: bool = False,
    first_frame_alignment: bool = False,
) -> str:
    """
    The mandatory summary prefix, from what the run actually uses.

    "reference generation" is the base case: <Picture 1> guides identity
    and <Video 1> supplies camera, cuts and rhythm without being edited
    or continued, which the guide puts squarely in that type. A picture
    pinned to a concrete frame adds "keyframe completion"; a reused song
    adds "audio reuse". Order follows the guide's own table.
    """
    types = []
    if first_frame_alignment:
        types.append("keyframe completion")
    types.append("reference generation")
    if music_is_reference:
        types.append("audio reuse")
    return " + ".join(types)


def build_shot_list_block(
    cut_timestamps: list,
    duration_seconds: float,
    cut_kinds: Optional[list] = None,
) -> str:
    """
    The MEASURED SHOT LIST block: one line per shot, with the boundary
    kind when a real detector supplied one.

    cut_timestamps are shot start times in seconds and must begin at
    0.0; cut_kinds, when given, is how each shot is entered ("start",
    "hard cut", "dissolve", ...) and is parallel to cut_timestamps.
    """
    kinds = list(cut_kinds or [])
    lines = []
    for i, start in enumerate(cut_timestamps):
        end = (
            cut_timestamps[i + 1] if i + 1 < len(cut_timestamps)
            else duration_seconds
        )
        default = "start" if i == 0 else "hard cut"
        kind = kinds[i] if i < len(kinds) else default
        entry = "opening shot" if i == 0 else f"entered by a {kind}"
        lines.append(
            f"  Shot {i + 1}: starts {start:.3f}s, "
            f"runs {max(0.0, end - start):.3f}s, {entry}"
        )
    return "\n".join(lines)


def build_user_context(
    subject_name: str,
    subject_wardrobe: str,
    scene_style: str,
    soundscape_type: str,
    duration_seconds: float,
    fps: float,
    frame_timestamps: list,
    cut_timestamps: list,
    enable_audio_prompt: bool = True,
    dialogue_text: str = "",
    audio_available: bool = False,
    full_clip: bool = False,
    cut_kinds: Optional[list] = None,
    cut_source: str = "local",
    alignment_seconds: Optional[float] = None,
    alignment_shot: int = 1,
    music_video: bool = False,
    lyrics: str = "",
    music_description: str = "",
    music_is_reference: bool = False,
    task_type: str = "reference generation",
) -> str:
    """
    Compose the per-run task context sent alongside the images.

    cut_source is where cut_timestamps came from: "measured" for a real
    shot-boundary detector (Cut Detective), "local" for the node's own
    keyframe-selection heuristic. A measured list is stated as fact and
    the model is told to match it shot for shot; a local one stays a
    hint the model may override from the frames.

    music_video flips the audio balance: non_diegetic_music becomes the
    leading section and overall_soundscape thins out. music_is_reference
    says the song is handed to the generation as <Audio 1>, which
    changes it again - the score is declared reused rather than
    invented, and any voice inside it belongs to <Audio 1> instead of
    getting its own speaker ID.

    alignment_seconds, when not None, turns on first-frame alignment:
    <Picture 1> stops being a pure identity reference and becomes the
    literal frame of the target video at that moment. That inverts the
    node's normal "do not copy <Picture 1>'s background, pose or
    lighting" stance, so the reversal is stated here rather than left
    for the model to infer.
    """
    lines = [
        "TASK CONTEXT",
        f"- subject_name: {subject_name}",
        f"- wardrobe (name these items verbatim): {subject_wardrobe}",
        f"- scene_style: {scene_style}",
        f"- clip duration: {duration_seconds:.3f} seconds at {fps:.3f} fps",
    ]
    if task_type:
        lines.append(
            f"- summary MUST begin with this exact prefix: [{task_type}] "
            "Copy it character for character, brackets included, then "
            "continue the paragraph."
        )
    if full_clip:
        lines.append(
            "- THE COMPLETE SOURCE CLIP IS ATTACHED as <Video 1>. Read "
            "the motion, cut timing, and camera movement directly from "
            "it. Place every [Shot N] time from what you see in the "
            "video, not from a guess."
        )
    else:
        lines.append(
            "- sampled frame timestamps (seconds): "
            + ", ".join(f"{t:.3f}" for t in frame_timestamps)
        )

    measured = cut_source == "measured" and bool(cut_timestamps)
    if measured:
        count = len(cut_timestamps)
        lines.append(
            f"- MEASURED SHOT LIST - {count} "
            f"{'shot' if count == 1 else 'shots'}. A "
            "shot-boundary detector read every frame of the clip to "
            "produce this. It is ground truth, more reliable than the "
            "sampled frames and more reliable than the attached video. "
            "Write exactly these shots, in this order, with these start "
            "times. Do not add a shot, do not drop one, do not shift a "
            "time:\n"
            + build_shot_list_block(
                cut_timestamps, duration_seconds, cut_kinds
            )
        )
    elif cut_timestamps:
        hint_scope = (
            "local detector's guess; trust the attached video over it"
            if full_clip else "trust these over your own guess"
        )
        lines.append(
            "- detected hard-cut boundaries (seconds, shot starts; "
            f"{hint_scope}): "
            + ", ".join(f"{t:.3f}" for t in cut_timestamps)
        )
    elif full_clip:
        lines.append(
            "- detected hard-cut boundaries: none found locally; "
            "confirm against the attached video."
        )
    else:
        lines.append(
            "- detected hard-cut boundaries: none (treat as one "
            "continuous shot unless frames clearly show a cut)"
        )
    if alignment_seconds is not None:
        hook = build_alignment_hook(alignment_seconds, alignment_shot)
        lines.append(
            "- FIRST-FRAME ALIGNMENT IS ON. This exact sentence will be "
            f"placed above subject_definitions for you:\n    {hook}\n"
            "  Do not write that sentence yourself. Write the rest of "
            "the prompt so it is true. For this run <Picture 1> is not "
            "only an identity reference: it IS the frame of the target "
            f"video at {float(alignment_seconds):.2f} seconds. Its "
            "framing, camera angle, background, lighting, and the "
            "subject's pose in it are the target video at that moment, "
            f"and [Shot {max(1, int(alignment_shot))}] must open on "
            "exactly that image before any action moves on from it. "
            "THEREFORE, for this run only, reverse the normal rule: do "
            "NOT write that <Picture 1> supplies identity alone, do NOT "
            "write that its background, pose, or lighting must be "
            "ignored, and do NOT write any exclusion forbidding their "
            "use. Say instead that the shot begins on <Picture 1> "
            "exactly as shown. Identity and wardrobe stay locked to it "
            "as always."
        )
        # With alignment on, <Picture 1> really is a concrete frame
        # anchor, which is exactly when the guide wants a standalone
        # entry and its own retention line. Rule 2's "no standalone
        # picture entry" covers the identity-only case, so it has to be
        # lifted here.
        shot_label = f"[Shot {max(1, int(alignment_shot))}]"
        lines.append(
            "- BECAUSE ALIGNMENT IS ON, <Picture 1> gets its own line "
            "in subject_definitions and its own line in "
            "retention_analysis, overriding the usual rule that an "
            "identity-only image is cited inside <Subject 1> instead. "
            f"Write \"<Picture 1> is the first frame of {shot_label}, "
            "showing ...\" in subject_definitions, and "
            f"\"<Picture 1> ({shot_label} first frame): fully_preserved "
            "- ...\" in retention_analysis."
        )

    if enable_audio_prompt:
        if music_video:
            lines.append(MUSIC_VIDEO_GUIDANCE)
            if music_description.strip():
                lines.append(
                    "- the track, as the user describes it (build "
                    f"non_diegetic_music on this): {music_description.strip()}"
                )
            if music_is_reference:
                lines.append(MUSIC_VIDEO_AUDIO_REUSE.format(name=subject_name))
            if lyrics.strip():
                lines.append(MUSIC_VIDEO_LYRICS.format(lyrics=lyrics.strip()))
                if not music_is_reference:
                    lines.append(MUSIC_VIDEO_SINGER.format(name=subject_name))
            else:
                lines.append(
                    "- lyrics: none supplied. Do not invent words. If "
                    "the subject's mouth moves to the track, say the "
                    "mouth moves in time with the music without "
                    "intelligible lyrics, and write no <d> lines."
                )
        else:
            guidance = SOUNDSCAPE_GUIDANCE.get(
                soundscape_type, SOUNDSCAPE_GUIDANCE["ambient"]
            )
            lines.append(f"- soundscape type '{soundscape_type}': {guidance}")
        if audio_available and music_video:
            lines.append(
                "- THE TRACK IS ATTACHED FOR YOU TO HEAR. Build "
                "non_diegetic_music on what it actually sounds like, "
                "not on what the frames imply: the real genre, the "
                "instruments you can pick out, the tempo, and where "
                "the arrangement changes against the shot list. Keep "
                "overall_soundscape thin as described above. Do not "
                "transcribe sung words into <d> unless the lyrics are "
                "supplied in this context."
            )
        elif audio_available:
            lines.append(
                "- THE SOURCE CLIP'S AUDIO TRACK IS ATTACHED. Base "
                "overall_soundscape on what you actually hear in it, not "
                "on what the frames imply: name the real ambience, the "
                "specific impacts and movement sounds, and their timing "
                "against the shots. If music is audible, describe its "
                "instrumentation, tempo, and where it enters and exits "
                "in non_diegetic_music; write N/A only if you hear no "
                "score. Do not transcribe speech into a spoken line "
                "unless the task context supplies the exact words above."
            )
    else:
        lines.append(
            "- audio: minimal. No dialogue lines. overall_soundscape is "
            "quiet natural ambience only, non_diegetic_music is N/A."
        )
    if dialogue_text.strip():
        lines.append(
            f"- required dialogue, exact words: {dialogue_text.strip()}\n"
            "  Write it as \"<Subject 1> (S1) says, <d>[Language] "
            "...</d>\", keeping the visual label with the speaker ID and "
            "naming the actual language of the words in the brackets."
        )
    elif not (music_video and lyrics.strip()):
        lines.append("- dialogue: none supplied; write no <d> lines.")
    closing = (
        "\nThe single image is <Picture 1>, the identity and "
        "wardrobe reference for the subject. The attached video is "
        "<Video 1>, the complete motion source. Write the complete "
        "H3 REF2VA prompt now, following every unbreakable rule."
        if full_clip else
        "\nThe first image is <Picture 1>, the identity and wardrobe "
        "reference for the subject. Every following image is a "
        "sampled frame of <Video 1> in playback order, each preceded "
        "by its timestamp label. Write the complete H3 REF2VA prompt "
        "now, following every unbreakable rule."
    )
    if measured:
        count = len(cut_timestamps)
        closing += (
            f" detailed_description must contain exactly {count} "
            f"[Shot N] {'label' if count == 1 else 'labels'}, matching "
            "the MEASURED SHOT LIST above."
        )
    lines.append(closing)
    return "\n".join(lines)


def build_retry_message(errors: list) -> str:
    """User message for the single corrective retry."""
    bullet_list = "\n".join(f"- {e}" for e in errors)
    return (
        "Your previous output failed these checks:\n"
        f"{bullet_list}\n"
        "Output the corrected COMPLETE prompt now. Same six-section "
        "format, raw text only, starting with 'subject_definitions:'."
    )
