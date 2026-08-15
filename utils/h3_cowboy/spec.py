"""
The MiniMax H3 prompt format, as data.

Every constant here traces to a line in one of MiniMax's own documents, and
each carries the citation. The rule is: if the guide does not say it, this
file does not assert it - a judgement call is labelled as one. That is what
lets the next person argue with MiniMax rather than with a preference.

    guide_ref  = huggingface.co/MiniMaxAI/MiniMax-H3
                 docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
    guide_base = same repo, docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
    readme     = same repo, README.md

Fetch them with plain curl; the HF MCP tools return Cloudflare 502s.

Nothing here imports torch or a provider SDK, so the whole module is
readable offline and from a test.
"""

from typing import List

# Ref mode's six sections are already correct in the older package, and
# there must be exactly one copy of that list in the repo.
from ..h3_prompt.prompts import (  # noqa: F401
    RETENTION_MARKERS_AUDIO,
    RETENTION_MARKERS_VISUAL,
    SECTION_ORDER as REF_SECTION_ORDER,
    TASK_TYPES,
)

# Base mode is a genuinely different skeleton - three fields, not six.
# guide_base 2.2 "Part Two Contains the Three Core Fields". There is no
# subject_definitions, no summary, no retention_analysis, and the main
# field is renamed. Base mode is also a different checkpoint
# (H3-Base-FL2VA and friends vs H3-Base-Ref2VA), not a flag.
BASE_SECTION_ORDER = [
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
]

MODES = ("ref", "base_T2VA", "base_I2VA", "base_FL2VA", "base_L2VA")

# Which H3 checkpoint a mode actually needs. Emitted as an output so the
# fact is visible in the graph instead of being tribal knowledge.
CHECKPOINT_FOR_MODE = {
    "ref": "MiniMax-H3-Base-Ref2VA",
    "base_T2VA": "MiniMax-H3-Base-FL2VA",
    "base_I2VA": "MiniMax-H3-Base-FL2VA",
    "base_FL2VA": "MiniMax-H3-Base-FL2VA",
    "base_L2VA": "MiniMax-H3-Base-FL2VA",
}


# ---------------------------------------------------------------------------
# Subject kinds
# ---------------------------------------------------------------------------

# guide_ref 2.1: "<Subject N> is used for reusable visible content,
# including: People, animals, or objects / Scenes, backgrounds, or
# environments / Clothing, props, interfaces, or visual effects / Styles,
# actions, expressions, or poses". The tag is general; only this repo's
# older node narrowed it to a person.
SUBJECT_KINDS = (
    "person", "animal", "object", "scene", "wardrobe", "interface",
    "effect", "style", "action", "expression", "pose",
)

# Kinds that occupy a position in the frame, so "where is it" is a
# meaningful question. A scene IS the frame; a style and an action have no
# position at all. guide_ref 5.3 asks for a subject's "position in the
# frame" at its first clear appearance - which only parses for these.
SPATIAL_KINDS = frozenset(
    {"person", "animal", "object", "wardrobe", "interface", "effect"}
)

# What the node face calls each kind. "person" and "scene" are the
# guide's words and stay the canonical ones, but nobody wiring a
# reference picture thinks "I am plugging in a person kind" - they think
# character, or environment. Both spellings are accepted everywhere a
# kind is read, and the canonical one is what reaches the prompt.
KIND_ALIASES = {
    "character": "person",
    "char": "person",
    "environment": "scene",
    "env": "scene",
    "background": "scene",
    "location": "scene",
    "prop": "object",
    "clothing": "wardrobe",
    "outfit": "wardrobe",
    "ui": "interface",
    "vfx": "effect",
}

# The kind picker on a subject row, plain words first. Order is the order
# a row's dropdown shows, so the two everybody needs sit at the top.
KIND_CHOICES = (
    "character", "environment", "animal", "object", "wardrobe",
    "interface", "effect", "style", "action", "expression", "pose",
)


def canonical_kind(word: str) -> str:
    """
    The guide's kind for whatever the user typed or picked. "" if none.

    Case and surrounding punctuation are forgiven because both the row
    dropdown and the typed DSL come through here, and a kind that is
    nearly right should not silently become a subject named "Character".
    """
    kind = (word or "").strip().lower().rstrip(":")
    kind = KIND_ALIASES.get(kind, kind)
    return kind if kind in SUBJECT_KINDS else ""

# The one governing rule, guide_ref 2: "Give each item its own line and
# explain what its label denotes, its reference role, and the main
# features to follow". The SHAPE below is the guide's. The per-kind
# feature lists are a judgement call - the guide names features for no
# kind - anchored to its own examples where one exists.
KIND_CARDS = {
    "person": (
        "person: face and facial structure, hair style, length and colour, "
        "skin tone, apparent age, build and proportions, and the exact "
        "garments and accessories worn - name each garment, its colour and "
        "its cut. Never write \"the same outfit\" or \"her usual look\", "
        "and never write personality or backstory."
    ),
    "animal": (
        "animal: species or breed, coat colour, length and markings, body "
        "size and build, ear, tail and muzzle shape, and any collar or "
        "harness. The guide's own wording for one: \"thick white fur, "
        "pointed ears, a dark nose, and a curved tail\"."
    ),
    "object": (
        "object: what it is, material and finish, colour, size relative to "
        "a hand or a body, distinguishing marks such as wear, labels or "
        "damage, and its working state - open or closed, lit or unlit, "
        "full or empty, intact or broken. No brand narrative, no symbolism."
    ),
    "scene": (
        "scene: the architecture or terrain, the fixed large elements AND "
        "where they sit relative to each other, surface materials, and the "
        "light - source, direction, colour, hardness. The guide's own "
        "wording for one: \"an exposed brick wall, an orange tufted sofa "
        "with patterned pillows, a neon sign, and a wooden coffee table\". "
        "Never weather-as-mood, never \"cosy\" or \"atmospheric\"."
    ),
    "wardrobe": (
        "wardrobe: garment type and cut, colour, fabric and texture, "
        "fastenings and hardware, and how it sits and moves on the body. "
        "Do not describe who wore it in the source."
    ),
    "interface": (
        "interface: what it displays, its layout and typography, its "
        "colours and opacity, how it animates, and where it sits in frame. "
        "No implementation words."
    ),
    "effect": (
        "effect: what is visibly happening, its colour and density, how it "
        "moves and dissipates, and what it occludes. Describe the look, "
        "not the technique used to make it."
    ),
    "style": (
        "style: the medium (live-action, cinematic, 2D-animated, 3D CG, "
        "claymation, watercolour, vintage film), the colour treatment "
        "(grade, saturation, contrast, dominant hue), grain and texture, "
        "lens character (focal-length feel, depth of field, flare, "
        "distortion, halation), lighting quality, and framing habits. "
        "Mood words alone are not a style - never \"dreamy\", \"epic\" or "
        "\"cinematic\" on its own."
    ),
    "action": (
        "action: the movement itself as a sequence of body or object "
        "states - what initiates it, the path limbs or objects travel, the "
        "timing feel, and the end state. Never the emotion behind it, and "
        "never the identity of whoever performed it in the source: that "
        "identity is a different <Subject N>, or none at all."
    ),
    "expression": (
        "expression: the facial-muscle arrangement itself - brow, eyes, "
        "mouth, head angle. Not the feeling it signifies."
    ),
    "pose": (
        "pose: the body attitude itself - weight distribution, limb "
        "placement, spine line. Not the feeling it signifies."
    ),
}

# guide_ref 2.1: "When the same subject comes from multiple assets,
# combine the sources and state what each asset provides". Taught
# explicitly because it decides a question users hit immediately: a
# person and their motion are ONE subject when they are the same
# character, and TWO when the motion is being transplanted.
MULTI_SOURCE_RULE = (
    "When one subject is defined by several assets, combine them in the "
    "one line and say what each provides, e.g. \"<Subject 1> is the woman "
    "whose appearance comes from <Picture 1> and whose walking motion "
    "comes from <Video 1>\"."
)


# ---------------------------------------------------------------------------
# Base-mode instruction lines
# ---------------------------------------------------------------------------

# guide_base 2.1. Three things about these are load-bearing and will look
# like typos to anyone tidying up:
#
#   1. The separator is an EM DASH (U+2014), not a hyphen.
#   2. FL2VA writes BARE `Picture 1` and `Shot 1` - no angle brackets, no
#      square brackets - while I2VA and L2VA bracket both. That is
#      systematic: it holds in the instruction line AND in the body of
#      guide_base's own Case 3.
#   3. `N` and `S.SS` are only knowable AFTER the body exists ("N is the
#      index of the actual final shot, and S.SS is the effective video
#      duration"). So the line is rendered after the model answers, never
#      before - see render_instruction_line() below.
#
# tests/test_h3_cowboy_base.py locks all three.
INSTRUCTION_T2VA = ""   # "T2VA has no image-alignment instruction"

INSTRUCTION_I2VA = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)

INSTRUCTION_FL2VA = (
    "How the reference pictures align with the target video — "
    "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
    "target video; Picture 2 (from Shot {shot}) aligns with the "
    "{seconds:.2f}-second mark of the target video."
)

INSTRUCTION_L2VA = (
    "How the reference pictures align with the target video — "
    "<Picture 1> (from [Shot {shot}]) aligns with the {seconds:.2f}-second "
    "mark of the target video."
)


# ---------------------------------------------------------------------------
# Worked examples - always MiniMax's own, never ours
# ---------------------------------------------------------------------------

# A worked example teaches shape more strongly than any numbered rule, so
# a wrong one is worse than a missing one. We therefore never write one:
# we ship whichever official example matches the run, and exactly one -
# two examples of different shapes get blended.
#
# This is guide_ref section 7, verbatim. It is worth reading before
# touching any validator, because it settles several arguments at once:
#   - four subjects of three different kinds (environment, animal, two
#     people) in one prompt
#   - <Subject 2> appears in Shot 1 and Shot 2 and NOT Shot 3, so
#     per-shot presence is not a rule
#   - <Picture 1>-<Picture 4>, <Video 1> and <Video 2> are all cited
#     inside subject lines and NONE gets a standalone entry
#   - nothing sits above `subject_definitions:` - the alignment
#     instruction is a base-mode construct
#   - no subject has a proper name
EXAMPLE_REF_GENERATION = """subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall, an orange tufted sofa with patterned pillows, a neon sign, and a wooden coffee table.
<Subject 2> is the fluffy white Samoyed in <Picture 2>, <Picture 3>, and <Picture 4>, with thick white fur, pointed ears, a dark nose, and a curved tail.
<Subject 3> is the young blonde woman in <Video 1>, with long blonde hair and a light-pink button-down shirt with rolled-up sleeves.
<Subject 4> is the young man in <Video 2>, with short wavy brown hair and a dark-grey hoodie with drawstrings.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1), containing a spoken English vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 3> eating a cookie in <Subject 1>. <Subject 4> enters with <Subject 2>, which lunges toward the cookie. The three-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 3> and ends with a canned audience laugh.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the Samoyed's thick white fur, pointed ears, dark nose, and curved tail are retained.
<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the blonde woman's identity, long hair, and light-pink shirt are retained.
<Subject 4> (appears in [Shot 1], [Shot 2]): fully_preserved - the young man's short wavy brown hair and dark-grey hoodie are retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 3> without copying the original signal.

detailed_description:
The target video uses a realistic multi-camera sitcom style with warm indoor lighting.
[Shot 1] A medium shot establishes <Subject 1>, the coffee shop with its exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table. <Subject 3> (S1), the young woman with long blonde hair and a light-pink button-down shirt with rolled-up sleeves, sits on the sofa holding a chocolate-chip cookie. From the left, <Subject 4>, the young man with short wavy brown hair and a dark-grey hoodie with drawstrings, enters holding the leash of <Subject 2>, the thick-furred white Samoyed with pointed ears, a dark nose, and a curved tail. The dog lunges toward the cookie and pulls the leash taut. <Subject 3> (S1) jerks her hand back and, using the clear youthful voice timbre referenced from <Audio 1>, exclaims with light annoyance, <d>[English] Hey! Watch your dog!</d> She closes her lips and guards the cookie while <Subject 4> pulls the dog back.
[Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1, sitting beside <Subject 3> on the sofa and holding <Subject 2> securely in his arms. <Subject 4> (S2) says in a casual young male voice with a playful tone and an easy conversational pace, <d>[English] He just likes cookies more than me.</d> He closes his mouth into an apologetic smile and strokes the dog's thick white fur.
[Shot 3] At 00:05.000, the shot cuts to a close-up of <Subject 3> (S1), the blonde woman in the light-pink shirt from Shot 1. Her annoyance softens as she looks toward the Samoyed. <Subject 3> (S1) replies in the same clear youthful voice referenced from <Audio 1> with an amused cadence, <d>[English] Well, he has good taste at least.</d> She smiles and raises the cookie in a small toast-like gesture. A classic canned audience laugh begins immediately after the line and continues through the final frame.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene.

non_diegetic_music:
N/A"""

# guide_ref 3 mandates one fixed sentence for an editing job, and this
# is MiniMax's own H3-Context-IR output showing the whole shape
# (README, case-Ref2VA). It settles three things a generation example
# cannot:
#   - summary opens "[<types>] The target video is an edited version of
#     <Video 1>." after the prefix
#   - <Video 1> gets a standalone definition line, which a subject-source
#     video never does
#   - retention carries "<Video 1> (source video editing): fully_preserved"
#     even though the mouth was newly animated - guide_ref 4.2: new actions
#     are not losses of fidelity
# It also writes [video editing + audio reference + audio reuse], which is
# OUT of the guide's own table order. That is why the validator compares
# task-type SETS and never strings.
EXAMPLE_REF_EDITING = """subject_definitions:
<Subject 1> is the young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, an unbuttoned white shirt, and silver rings, holding a small black lamb in his arms in <Video 1>.
<Video 1> is the source video for the editing task.
<Audio 1> is the synchronized audio track of <Video 1>, providing the background music.
<Audio 2> is the voice timbre reference for <Subject 1>'s voice, containing a spoken male voiceover.

summary:
[video editing + audio reference + audio reuse] The target video is an edited version of <Video 1>. <Subject 1>, wearing a bright pink suit and holding a black lamb, stands in a grassy field with other white lambs in the background. The edit animates <Subject 1>'s face to speak the user-provided dialogue. <Audio 1> is partially reused as the continuous background music, while the target references the calm male voice timbre of <Audio 2> for <Subject 1>'s spoken lines.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the man retains his identity, wavy blonde hair, pink suit, white shirt, accessories, and the black lamb he holds, with his mouth newly animated to speak.
<Video 1> (source video editing): fully_preserved - the original camera framing, warm golden hour lighting, grassy hill setting, and background white lambs are maintained while the central character is edited.
<Audio 1>: partially_copy - the atmospheric background music from <Audio 1> is reused in the target video, mixed beneath the newly added spoken dialogue.
<Audio 2>: reference - the target audio references the male voice timbre from <Audio 2> to generate <Subject 1>'s spoken dialogue.

detailed_description:
The target video is in realistic photographic style.
[Shot 1] The shot begins from the source <Video 1>, showing <Subject 1>, a young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, and a casually unbuttoned white shirt. He stands confidently in a sunlit green pasture, gently holding a small black lamb securely in his arms. The warm, golden hour lighting casts soft shadows across his face and the bright pink fabric of his suit. Behind him, several white lambs stand and graze on the rolling grassy hill against a clear, pale blue sky. The atmospheric background music from <Audio 1> plays continuously throughout the scene. <Subject 1> physically speaks, his mouth movements naturally syncing to the new dialogue, with his voice timbre referencing the calm male delivery from <Audio 2>. Looking thoughtfully forward, <Subject 1> (S1) speaks softly, <d>[English] Follow the wind, live free.</d> As he delivers the line, he subtly shifts his weight, cradling the resting black lamb while the camera slowly pushes in. <Subject 1> (S1) continues his thought, <d>[English] Leave worries behind, enjoy the moment.</d> Exactly as his voice stops, his lips meet in a relaxed, peaceful smile, and his jaw ceases speaking motion. He then turns his gaze slightly away toward the horizon, gently stroking the black lamb's fleece with his fingers as the camera holds on this tranquil, sunlit state through the end of the video.

overall_soundscape:
The soundscape consists of the continuous, atmospheric background music from <Audio 1>, overlaid with the clear, calm male dialogue spoken by the main character, referencing the voice timbre of <Audio 2>.

non_diegetic_music:
The atmospheric, sustained background music from <Audio 1> is reused as the continuous score, playing quietly beneath the spoken dialogue."""


# The four base cases, guide_base section 5, verbatim. Extracted
# mechanically from the guide rather than transcribed - see
# docs/H3_COWBOY_BASE_MODE_HANDOFF.md section 5 for the six-line script
# that produces them, and tests/test_h3_cowboy_base.py for the shape lock.
#
# Each one is shipped whole, instruction line included, because the line
# teaches where it sits and what separates it from the fields. The system
# prompt then tells the model not to write it: N and S.SS are only
# knowable after the body exists, so the node renders it afterwards.
#
# What each case proves, and why all four are here rather than one:
#   1 T2VA  - no instruction line at all, and the same-line headers
#   2 I2VA  - the I2VA line verbatim; <Picture 1> cited inside [Shot 1]
#   3 FL2VA - BARE `Picture 1` and `Shot 1` in the line AND in the body;
#             a single shot; non_diegetic_music: N/A
#   4 L2VA  - bracketed <Picture 1> / [Shot 1]; the picture lands at the
#             END of the video, not the start

EXAMPLE_BASE_T2VA = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from the sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. The doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end."""

EXAMPLE_BASE_I2VA = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> remains beside the rain-covered train window, preserving her appearance, clothing, seat position, and the carriage layout. The camera trucks right with small amplitude at slow speed as she lifts her gaze from the folded letter toward the passing city lights. Her reflection moves across the glass while the quiet, breathy young woman (S1) says: <d>[English] I get off at the next station.</d> She folds the letter along its existing crease.

overall_soundscape: The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain ticks against the window while paper rustles softly in her hands.

non_diegetic_music: Sustained cello notes at a slow tempo with widely spaced piano tones, gradually decreasing in volume."""

EXAMPLE_BASE_FL2VA = """How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a rain-soaked cyclist begins in the position and framing established by Picture 1, holding a closed black umbrella beside a silver bicycle. The camera pulls out with small amplitude at slow speed as she releases the bicycle handle, raises the umbrella above her shoulder, and presses the runner upward until the canopy opens. Water rolls from the expanding fabric while she steps beneath it, rotates the handle into the final angle, and settles into the pose, spacing, and composition established by Picture 2 at the end of the shot.

overall_soundscape: Rain falls steadily on the pavement, followed by the metallic click of the umbrella runner and the soft snap of the canopy opening. Water drips from the bicycle frame as distant traffic passes.

non_diegetic_music: N/A"""

EXAMPLE_BASE_L2VA = """How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot begins with an intact drinking glass near the edge of a dark wooden table, while the same hand and sleeve visible in <Picture 1> approach from the right. The camera pushes in with small amplitude at slow speed as the fingertips strike the rim. The glass tips, falls, and hits the floor with a sharp impact; cracks spread through it as fragments slide outward. Toward the end, the moving pieces lose momentum and settle into the exact broken arrangement, hand position, camera angle, lighting, and final composition established by <Picture 1>.

overall_soundscape: Fingertips tap the glass before it scrapes across the tabletop, falls, and breaks with a sharp crash. Small fragments scatter and gradually stop sliding across the floor.

non_diegetic_music: A low electronic pulse at a slow tempo, ending immediately after the glass breaks."""

EXAMPLE_FOR_BASE_MODE = {
    "base_T2VA": EXAMPLE_BASE_T2VA,
    "base_I2VA": EXAMPLE_BASE_I2VA,
    "base_FL2VA": EXAMPLE_BASE_FL2VA,
    "base_L2VA": EXAMPLE_BASE_L2VA,
}

# The effective duration each case's own text states, so a test can run
# the case through the assembler and get its instruction line back
# unchanged. Case 3 says "eight-second single shot" and writes
# 8.00-second; Case 4 says six and writes 6.00-second. Cases 1 and 2
# state no duration, so these are only long enough to hold their shot
# times - they assert nothing about the guide.
EXAMPLE_BASE_DURATION = {
    "base_T2VA": 8.0,
    "base_I2VA": 5.0,
    "base_FL2VA": 8.0,
    "base_L2VA": 6.0,
}


# ---------------------------------------------------------------------------
# Asset roles -> task types
# ---------------------------------------------------------------------------

# guide_ref 3 is explicit that wiring alone cannot imply a task type:
# "The mere presence of video or audio does not automatically create a
# corresponding task type. If a reference video provides only camera
# movement, cuts, or rhythm, it normally belongs to reference
# generation. Use video editing or video continuation only when that
# video is directly edited or continued."
#
# So the user declares what the asset is FOR, and the task type follows.
# Declaring a role rather than typing a type keeps the fixed vocabulary
# typo-free and keeps the prefix consistent with the retention markers.
VIDEO_ROLES = {
    "subject_source": "reference generation",
    "structure_reference": "reference generation",
    "edit_source": "video editing",
    "continuation_source": "video continuation",
}
AUDIO_ROLES = {
    "none": None,
    "reuse": "audio reuse",
    "reference": "audio reference",
}

# Only these roles make <Video N> a label in its own right. guide_ref 2.3:
# "<Video N> is reserved for whole-video relationships, such as: Editing
# an original video / Continuing from the end of an original video /
# Referencing the original video's camera movement, cuts, rhythm, or
# temporal structure." A video that merely shows a subject is cited
# inside that subject's line and gets no entry.
VIDEO_ROLES_WITH_OWN_LINE = frozenset(
    {"structure_reference", "edit_source", "continuation_source"}
)

# guide_ref 3, the mandatory opener for an editing job.
EDIT_SUMMARY_OPENER = "The target video is an edited version of <Video 1>."
EDIT_VIDEO_DEFINITION = "<Video 1> is the source video for the target video edit."
EDIT_VIDEO_RETENTION = (
    "<Video 1> (source video editing): fully_preserved - the original "
    "framing, lighting and setting are maintained except where the edit "
    "changes them."
)


def derive_task_types(
    video_role: str = "subject_source",
    audio_role: str = "none",
    has_frame_anchor: bool = False,
) -> List[str]:
    """
    The task-type set for this run, from what its assets are FOR.

    Two auto-additions the guide mandates outright (section 3):
      - editing a source whose own track stays audible is ALSO audio reuse
      - continuing a source with new audio that only continues its
        character is ALSO audio reference

    Order is the guide's own table order, chosen purely so the same graph
    always emits the same prefix. The guide states no ordering rule, and
    its own example writes "[video editing + audio reference + audio
    reuse]" - out of table order - which is exactly why the validator
    compares sets and never strings.
    """
    types = []
    if has_frame_anchor:
        types.append("keyframe completion")
    video_type = VIDEO_ROLES.get(video_role, "reference generation")
    types.append(video_type)
    audio_type = AUDIO_ROLES.get(audio_role)
    if audio_type:
        types.append(audio_type)

    order = list(TASK_TYPES)
    seen = []
    for t in sorted(set(types), key=order.index):
        if t not in seen:
            seen.append(t)
    return seen


def format_task_type(types: List[str]) -> str:
    """'video editing + audio reuse' - joined per guide_ref 3."""
    return " + ".join(types)


def example_for(mode: str, task_types=()) -> str:
    """
    The single worked example to ship with this run.

    Selected by shape, not by subject kind: shape is what an example
    teaches, and it varies by mode and by whether a whole video is being
    consumed - not by whether the subject is a dog or a lens grade.
    """
    if mode == "ref":
        whole_video = {"video editing", "video continuation"}
        if whole_video & set(task_types or []):
            return EXAMPLE_REF_EDITING
        return EXAMPLE_REF_GENERATION
    if mode in EXAMPLE_FOR_BASE_MODE:
        # One per sub-mode, and never the neighbouring one: the four
        # differ in exactly the details that are easiest to get wrong
        # (bare vs bracketed tags, where the picture lands, whether an
        # instruction line exists at all).
        return EXAMPLE_FOR_BASE_MODE[mode]
    raise ValueError(
        f"no worked example for mode '{mode}'; legal modes are "
        + ", ".join(MODES)
    )


def render_instruction_line(
    sub_mode: str, final_shot: int = 1, duration_seconds: float = 0.0
) -> str:
    """
    The base-mode instruction line, rendered AFTER the body exists.

    guide_base 2.1: "N is the index of the actual final shot, and S.SS is
    the effective video duration formatted to exactly two decimal
    places." Neither is knowable before the model answers, so this is
    called with the shot count parsed out of its reply - never with an
    assumed one. The older node renders its hook before the call, which
    is wrong the moment the model writes a different number of shots.

    Returns "" for T2VA, which has no instruction line at all.
    """
    if sub_mode == "base_T2VA":
        return INSTRUCTION_T2VA
    if sub_mode == "base_I2VA":
        # Fixed string, no parameters: <Picture 1> is the first frame, so
        # it is always Shot 1 at 0.00 seconds.
        return INSTRUCTION_I2VA
    shot = max(1, int(final_shot))
    seconds = max(0.0, float(duration_seconds))
    if sub_mode == "base_FL2VA":
        return INSTRUCTION_FL2VA.format(shot=shot, seconds=seconds)
    if sub_mode == "base_L2VA":
        return INSTRUCTION_L2VA.format(shot=shot, seconds=seconds)
    raise ValueError(
        f"'{sub_mode}' is not a base mode; legal values are "
        + ", ".join(m for m in MODES if m != "ref")
    )
