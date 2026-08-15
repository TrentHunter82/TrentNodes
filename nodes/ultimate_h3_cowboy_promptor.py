"""
Ultimate H3 Cowboy Promptor - MiniMax H3 prompts for any kind of shot.

Where H3 Auto Prompt Generator writes one job (replace the performer in a
video with the person in a reference image), this writes any job the H3
spec describes: subjects of any kind, several at once, all six task
types, and both skeletons - the six-section full-reference format and the
three-field base format (T2VA / I2VA / FL2VA / L2VA).

h3_mode picks the skeleton, and the two are genuinely different formats
rather than two settings of one: different sections, different system
prompt, different H3 checkpoint. h3_checkpoint_hint says which.

Both nodes are installed on purpose. The older one is frozen and keeps
working; use it for character replacement you already have dialled in.
Use this one for everything else: objects, environments, styles, several
subjects in one shot.

Subjects are rows on the node face, one per picture slot: a kind, an
optional name, and a description. The rows appear as you use them - fill
the last one and the next arrives - and one rule runs the whole way
through:

    row N  IS  <Subject N>  IS  subject_N_image  IS  <Picture N>

so a skipped row leaves a hole rather than closing it. A row needs no
image, which is how you ask for a style or an action.

The typed `subjects` field is the escape hatch behind Show advanced, for
what a row cannot say - a seventh subject, or one citing two pictures at
once. Its syntax is a small DSL, kind first:

    person  Aria Voss     @Picture 1  -- dark hair, charcoal utility jacket
    scene   the warehouse @Picture 2  -- corrugated steel, sodium toplight
    style                             -- 16mm grain, halation, teal shadows

See docs/H3_COWBOY_HANDOFF.md.
"""

import json
from fractions import Fraction
from typing import List, Optional, Tuple

import torch

from ..utils.cut_detect.formats import parse_cut_times
from ..utils.h3_cowboy import prompts_base, prompts_ref, spec
from ..utils.h3_cowboy.assembler import (
    PICTURES_FOR_MODE,
    CowboyContext,
    process,
)
from ..utils.h3_cowboy.subjects import (
    SubjectRow,
    bind_images,
    merge_text_subjects,
    subjects_from_rows,
)
from ..utils.h3_cowboy.wiring import (
    H3_FPS,
    build_label_map,
    canvas_for,
    snap_length,
    trim_reference_frames,
)
from ..utils.h3_prompt import audio_io, video_io
from ..utils.h3_prompt.backends import (
    DEFAULT_MODELS,
    SEED_MAX,
    VLMAudio,
    VLMImage,
    get_backend,
    normalize_seed,
)
from ..utils.h3_prompt.imaging import (
    FRAME_MAX_SIDE,
    REFERENCE_MAX_SIDE,
    tensor_to_jpeg_b64,
)
from ..utils.h3_prompt.keyframes import frame_label, select_keyframes

MAX_RETRIES = 1
NUM_SUBJECT_SLOTS = 6      # matches RefFolderCowboy, so one wire-up drives both
LOG_PREFIX = "[H3Cowboy]"

# Base mode: nothing in an image says whether it is the first frame or
# the last one, so it is declared. The first value doubles as "unset",
# which is what lets base_L2VA run without warning about a widget the
# user never touched.
PICTURE_ROLES = ("first_frame", "last_frame")
DEFAULT_PICTURE_ROLE = PICTURE_ROLES[0]
PICTURE_ROLE_FOR_MODE = {
    "base_I2VA": "first_frame",
    "base_L2VA": "last_frame",
}
# T2VA has no clip to measure, so with no video, no frames and no
# duration_override there is nothing left to derive a length from. Five
# seconds is this node's guess, and it says so out loud - the number
# reaches the FL2VA and L2VA instruction lines as S.SS.
DEFAULT_BASE_DURATION = 5.0

# What each row's kind starts as. A character and the place they are in
# is the commonest pair by a distance, so row 1 and row 2 open on it.
DEFAULT_ROW_KINDS = ("character", "environment") + ("character",) * 4

# Where a music video's track comes from. Only reuse declares <Audio 1>
# to H3, which is what puts "audio reuse" in the task type.
MUSIC_SOURCES = ("auto", "generate_score", "reuse_audio_1")

# H3 reads reference frames as 24 fps and nothing resamples them, so a
# clip at another rate plays at the wrong speed. 23.976 material drifts
# by one frame in a thousand, which is not worth saying; 25 or 30 fps is.
FPS_WARN_TOLERANCE = 0.5

# Pass-through outputs, appended to the original five. The sampler's
# reference sockets are ordinary IMAGE / AUDIO inputs one wire each, so
# there is one output per slot - see docs/H3_REFERENCE_WIRING_HANDOFF.md.
PASSTHROUGH_TYPES = (
    tuple(["IMAGE"] * NUM_SUBJECT_SLOTS)
    + ("IMAGE", "AUDIO", "AUDIO", "INT", "INT", "INT", "STRING")
)
PASSTHROUGH_NAMES = (
    tuple(f"ref_image_{slot}" for slot in range(1, NUM_SUBJECT_SLOTS + 1))
    + ("ref_video", "ref_video_audio", "ref_audio",
       "width", "height", "length", "label_map")
)


class UltimateH3CowboyPromptor:
    """Write a MiniMax H3 prompt for any subject kind or task type."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "video": ("VIDEO", {
                "tooltip": "Source clip. Preferred input; carries its fps."
            }),
            "frames": ("IMAGE", {
                "tooltip": (
                    "Alternative to video: an IMAGE batch. Set fps to match."
                ),
            }),
            "fps": ("FLOAT", {
                "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                "tooltip": "Frame rate of the frames input.",
            }),
            "audio": ("AUDIO", {
                "tooltip": (
                    "The clip's track, for a provider that can hear it "
                    "(gemini). Becomes <Audio 1>."
                ),
            }),
            "api_key": ("STRING", {
                "default": "",
                "advanced": True,
                "tooltip": "Blank uses the provider's environment variable.",
            }),
            "video_role": (list(spec.VIDEO_ROLES), {
                "default": "subject_source",
                "tooltip": (
                    "What the wired video is FOR - this is what decides "
                    "the task type, and wiring alone cannot imply it. "
                    "subject_source: it just shows a subject, so it is "
                    "cited inside that subject's line and gets no entry. "
                    "structure_reference: its camera, cuts and rhythm are "
                    "followed. edit_source: the target IS this video, "
                    "edited. continuation_source: the target continues "
                    "from where it ends."
                ),
            }),
            "audio_role": (list(spec.AUDIO_ROLES), {
                "default": "none",
                "tooltip": (
                    "What the wired audio is FOR. reuse: the signal is "
                    "copied into the target (adds 'audio reuse'). "
                    "reference: only its timbre, style or beat is "
                    "followed, not the signal (adds 'audio reference')."
                ),
            }),
            "cut_times": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": (
                    "Wire any Cut Detective output here. The shot list "
                    "becomes ground truth for the [Shot N] times. A count "
                    "mismatch is only an error when a video is being "
                    "edited - otherwise the target's structure is not "
                    "bound to the reference's. In base mode there is no "
                    "source clip, so this reads as the shot structure you "
                    "are asking for: more than one entry is what lets "
                    "FL2VA write more than one shot."
                ),
            }),
            "dialogue": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": (
                    "Exact spoken words. They must reach a <d> block "
                    "verbatim, or the run retries once."
                ),
            }),
            "constraint_notes": ("STRING", {
                "multiline": True, "default": "",
                "tooltip": (
                    "Things that must not change. Folded into the prompt "
                    "as positive assertions, because H3 has no negative "
                    "field and the official format ends at "
                    "non_diegetic_music - nothing may follow it."
                ),
            }),
            "duration_override": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.01,
                "tooltip": (
                    "0 = use the clip's own duration. In a base mode with "
                    "no clip this IS the target length: it reaches the "
                    "instruction line as S.SS, and leaving it at 0 makes "
                    "the node guess 5 seconds and warn."
                ),
            }),
            "max_frames_to_analyze": ("INT", {
                "default": 8, "min": 2, "max": 16,
                "advanced": True,
                "tooltip": "Keyframes sampled from the clip for the VLM.",
            }),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": SEED_MAX,
                "advanced": True,
                "tooltip": "Passed to providers that support seeding.",
            }),
        }
        for slot in range(1, NUM_SUBJECT_SLOTS + 1):
            optional[f"subject_{slot}_image"] = ("IMAGE", {
                "tooltip": (
                    f"Reference image for line {slot} of the subjects "
                    f"field. This slot IS <Picture {slot}> - the numbers "
                    "always match, so wire them in order from slot 1. In a "
                    "base_* mode there are no subjects: slot 1 is the "
                    "anchor frame, and slot 2 is FL2VA's last frame."
                ),
            })

        # Appended last so an existing graph's widget order never shifts.
        optional["base_picture_role"] = (list(PICTURE_ROLES), {
            "default": DEFAULT_PICTURE_ROLE,
            "advanced": True,
            "tooltip": (
                "Base mode only: whether a single wired picture is the "
                "FIRST frame or the LAST one. Nothing in the pixels says "
                "which, so it has to be declared. h3_mode already declares "
                "it too (base_I2VA is first, base_L2VA is last); this is "
                "the cross-check, and a disagreement warns and follows "
                "h3_mode."
            ),
        })
        optional["fl2va_normalize_picture_tags"] = ("BOOLEAN", {
            "default": False,
            "advanced": True,
            "tooltip": (
                "base_FL2VA only. MiniMax's guide writes bare 'Picture 1' "
                "and 'Shot 1' for FL2VA, with no brackets, while I2VA and "
                "L2VA bracket both - in the instruction line AND in the "
                "body of its own worked example. Off reproduces that. On "
                "rewrites them to <Picture 1> and [Shot 1]. No validator "
                "can tell which generates a better video, so this exists "
                "to be A/B'd; the setting is recorded in analysis_json."
            ),
        })
        optional["snap_duration_to_h3_grid"] = ("BOOLEAN", {
            "default": True,
            "advanced": True,
            "tooltip": (
                "H3 renders whole frames on a 17k+5 grid at 24 fps, so it "
                "rounds a length UP: ask for 2.00 seconds and you get 2.33. "
                "On, the prompt states the length H3 really produces, and "
                "the length output matches it. Off keeps the requested "
                "number, so the prompt claims a shorter video than the one "
                "H3 makes. Both numbers are recorded in analysis_json."
            ),
        })

        # -- the subject rows -------------------------------------------
        # One row per <Picture N>. Appended last, and only the first few
        # are shown until you need more (js/h3_cowboy.js), so the node
        # face stays the size of the job in front of you.
        optional["subject_rows"] = ("INT", {
            "default": 2, "min": 0, "max": NUM_SUBJECT_SLOTS,
            "tooltip": (
                "How many subject rows to show. It only controls the node "
                "face: a row with anything typed in it is always used, and "
                "the count grows on its own when you wire an image or fill "
                f"the last row. Up to {NUM_SUBJECT_SLOTS}."
            ),
        })
        for slot in range(1, NUM_SUBJECT_SLOTS + 1):
            optional[f"subject_{slot}_kind"] = (list(spec.KIND_CHOICES), {
                "default": DEFAULT_ROW_KINDS[slot - 1],
                "tooltip": (
                    f"What subject_{slot}_image / row {slot} IS. character "
                    "and environment are the two everyone needs; the rest "
                    "are the guide's other reusable kinds. The kind picks "
                    "the features the prompt asks for - a character gets "
                    "face, hair and garments, an environment gets surfaces "
                    "and light, a style gets grain and grade."
                ),
            })
            optional[f"subject_{slot}_name"] = ("STRING", {
                "default": "",
                "tooltip": (
                    f"Optional name for row {slot}, e.g. 'Aria Voss' or "
                    "'the loading bay'. Blank is fine - MiniMax's own "
                    "example names nobody."
                ),
            })
            optional[f"subject_{slot}_description"] = ("STRING", {
                "multiline": True, "default": "",
                "tooltip": (
                    f"What row {slot} looks like: the features the target "
                    "video must keep. Type here and this row becomes "
                    f"<Subject {slot}>, and subject_{slot}_image becomes "
                    f"<Picture {slot}>. An empty row is not a subject."
                ),
            })

        # -- music video -------------------------------------------------
        optional["music_video"] = ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "Write the prompt as a music video. non_diegetic_music "
                "becomes the lead audio section instead of 'N/A', "
                "overall_soundscape thins to what is audible under the "
                "track, cuts are described as landing on the beat, and "
                "performance to camera becomes the action. Put the sung "
                "words in lyrics and the track in music_description."
            ),
        })
        optional["music_source"] = (list(MUSIC_SOURCES), {
            "default": "auto",
            "tooltip": (
                "Where the music comes from. auto: the song is declared as "
                "<Audio 1> when audio is connected, on the assumption that "
                "the same file reaches H3. generate_score: H3 invents the "
                "track. reuse_audio_1: the track reaches H3 as <Audio 1> "
                "even with nothing wired here. Declaring the reuse is what "
                "adds 'audio reuse' to the task type."
            ),
        })
        optional["lyrics"] = ("STRING", {
            "multiline": True, "default": "",
            "tooltip": (
                "Exact sung words, in their original language. They must "
                "reach a <d>[Language] ...</d> block in the shot where "
                "they are heard, or the run retries once. Blank means the "
                "mouth moves to the music with no intelligible words - H3 "
                "invents nonsense syllables if you ask for singing without "
                "giving it any."
            ),
        })
        optional["music_description"] = ("STRING", {
            "multiline": True, "default": "",
            "tooltip": (
                "The track, for non_diegetic_music: genre, "
                "instrumentation, tempo or BPM, and how it develops. "
                "Example: 'downtempo synthwave, ~92 BPM, analog pad and "
                "gated drums, the filter opens into the chorus at the "
                "second cut'. Blank lets the model infer it from the "
                "attached audio, or from the visuals if there is none."
            ),
        })

        return {
            "required": {
                "h3_mode": (list(spec.MODES), {
                    "default": "ref",
                    "tooltip": (
                        "ref writes the six-section Ref2VA format from "
                        "reference assets. The base_* modes write the "
                        "three-field base format and need a different H3 "
                        "checkpoint (see h3_checkpoint_hint): T2VA is text "
                        "only, I2VA anchors the first frame, FL2VA the "
                        "first and last, L2VA the last. In base mode the "
                        "subjects field is ignored - there are no "
                        "reference labels at all."
                    ),
                }),
                "subjects": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "advanced": True,
                    "tooltip": (
                        "EXTRA subjects, typed. The rows below are the "
                        "normal way in; this is the escape hatch for what "
                        "a row cannot say - a seventh subject, or one that "
                        "cites two pictures at once. One line each:\n"
                        "  <kind> [name] [@Picture N ...] -- <features>\n"
                        "Kinds: character, environment, "
                        + ", ".join(
                            k for k in spec.KIND_CHOICES
                            if k not in ("character", "environment")
                        )
                        + ".\nTyped lines are numbered AFTER the filled "
                        "rows. Ignored in the base_* modes, which have no "
                        "reference labels at all."
                    ),
                }),
                "target_description": ("STRING", {
                    "multiline": True,
                    "default": "the courier ducks under a roller shutter",
                    "tooltip": "What the target video should show.",
                }),
                "vlm_provider": (list(DEFAULT_MODELS), {
                    "default": "anthropic",
                    "tooltip": "Which VLM writes the prompt.",
                }),
                "model": ("STRING", {
                    "default": "auto",
                    "advanced": True,
                    "tooltip": "'auto' uses the provider's default model.",
                }),
            },
            "optional": optional,
        }

    # The first five outputs are frozen. A saved workflow stores links by
    # output INDEX, so anything new is appended and never inserted -
    # inserting silently re-points every wire after it in every graph.
    RETURN_TYPES = (
        "STRING", "FLOAT", "INT", "STRING", "STRING"
    ) + PASSTHROUGH_TYPES
    RETURN_NAMES = (
        "h3_prompt", "duration_seconds", "fps", "analysis_json",
        "h3_checkpoint_hint",
    ) + PASSTHROUGH_NAMES
    OUTPUT_TOOLTIPS = (
        "The H3 prompt.",
        "Target duration. With snap_duration_to_h3_grid on this is the "
        "length H3 really produces, not the length asked for.",
        "Frame rate of the source clip, rounded.",
        "Everything the run decided, including both durations.",
        "Which H3 checkpoint this prompt is written for.",
    ) + tuple(
        f"Whatever you plugged into subject_{slot}_image, untouched. Wire "
        f"it to the sampler's ref_image_{slot} socket. A gap stays a gap: "
        f"the prompt says <Picture {slot}> for this slot, so compacting it "
        "here is exactly the mismatch this node exists to stop."
        for slot in range(1, NUM_SUBJECT_SLOTS + 1)
    ) + (
        "The clip as IMAGE frames, which is what the sampler's ref_video_ "
        "socket takes. Empty in base mode, which has no <Video 1>.",
        "The wired audio, for when the clip's own track is reused. Connect "
        "this OR ref_audio, not both. Empty in base mode.",
        "The same audio again, for when only its timbre or beat is "
        "referenced. Connect this OR ref_video_audio, not both.",
        "Sampler width: the video's framing if there is one, else the "
        "first wired picture's, on H3's canvas grid.",
        "Sampler height, from the same source as width.",
        "Frame count on H3's 17k+5 grid. Wire it to the sampler's length.",
        "What each <Picture i> / <Video k> / <Audio j> tag will refer to "
        "once the sampler numbers them. Read it against the prompt.",
    )
    FUNCTION = "generate"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "Writes a MiniMax H3 prompt for any kind of shot, not just "
        "character replacement. In ref mode, subjects can be people, "
        "animals, objects, scenes, styles or actions, and several at once "
        "- each gets a <Subject N> label, a definition line and a "
        "retention line. The base_* modes write the three-field base "
        "format instead: T2VA from text alone, I2VA from a first frame, "
        "FL2VA between a first and a last frame, L2VA onto a last frame. "
        "Validated against MiniMax's published format: all six of the "
        "guides' own worked examples pass this node with zero errors. "
        "Wire Cut Detective into cut_times to pin the shot timeline. Every "
        "asset comes back out on its own socket, with the width, height "
        "and length MiniMaxH3ReferenceToVideo wants, so nothing is wired "
        "twice and the prompt's tags cannot drift from the sampler's."
    )

    def generate(
        self,
        h3_mode: str,
        subjects: str,
        target_description: str,
        vlm_provider: str,
        model: str,
        video=None,
        frames: Optional[torch.Tensor] = None,
        fps: float = 24.0,
        audio: Optional[dict] = None,
        api_key: str = "",
        video_role: str = "subject_source",
        audio_role: str = "none",
        cut_times: str = "",
        dialogue: str = "",
        constraint_notes: str = "",
        duration_override: float = 0.0,
        max_frames_to_analyze: int = 8,
        seed: int = 0,
        base_picture_role: str = DEFAULT_PICTURE_ROLE,
        fl2va_normalize_picture_tags: bool = False,
        snap_duration_to_h3_grid: bool = True,
        # Omitted means "no opinion", so nothing warns about a row the
        # caller never said was hidden. The UI always sends its widget.
        subject_rows: int = NUM_SUBJECT_SLOTS,
        music_video: bool = False,
        music_source: str = "auto",
        lyrics: str = "",
        music_description: str = "",
        # The six subject image sockets AND the three widgets on each
        # subject row all arrive here, keyed by name. Every reader picks
        # its own keys out, so one dict carries the lot.
        **subject_fields,
    ) -> Tuple:
        warnings: List[str] = []
        subject_images = subject_fields

        if h3_mode != "ref":
            # Base mode is a different skeleton, a different system
            # prompt and a different checkpoint, so it branches here
            # rather than threading a flag through the ref path.
            return self._generate_base(
                h3_mode, subjects, target_description, vlm_provider, model,
                video=video, frames=frames, fps=fps, audio=audio,
                api_key=api_key, video_role=video_role, audio_role=audio_role,
                cut_times=cut_times, dialogue=dialogue,
                constraint_notes=constraint_notes,
                duration_override=duration_override,
                max_frames_to_analyze=max_frames_to_analyze, seed=seed,
                base_picture_role=base_picture_role,
                fl2va_normalize_picture_tags=fl2va_normalize_picture_tags,
                snap_duration_to_h3_grid=snap_duration_to_h3_grid,
                music_video=music_video, lyrics=lyrics,
                music_description=music_description,
                subject_images=subject_images, warnings=warnings,
            )

        rows = self._read_rows(subject_fields)
        out_of_sight = [
            row.slot for row in rows
            if row.is_filled and row.slot > subject_rows
        ]
        if out_of_sight:
            # Only reachable from the API, or if the JS did not load. The
            # rows still count - saying so beats a subject nobody can see.
            warnings.append(
                "subject_rows is "
                + f"{subject_rows}, but row "
                + ", ".join(str(s) for s in out_of_sight)
                + " holds text and was used anyway. Raise subject_rows to "
                "see it."
            )
        parsed = merge_text_subjects(
            subjects_from_rows(rows, warnings), subjects, warnings,
        )
        if not parsed:
            raise RuntimeError(self._no_subjects_message(subjects))

        wired = [
            slot for slot in range(1, NUM_SUBJECT_SLOTS + 1)
            if subject_images.get(f"subject_{slot}_image") is not None
        ]
        image_tags = bind_images(parsed, wired, warnings)
        self._warn_gap_consequence(wired, warnings)

        images, real_fps, duration_source = self._resolve_frames(
            video, frames, fps, warnings
        )
        duration = images.shape[0] / real_fps
        if duration_override > 0:
            duration = float(duration_override)
            duration_source = "override"

        # Before the prompt is written, never after: the prompt states the
        # duration as fact, and every shot time sits inside it.
        requested_duration = duration
        length, duration = self._snap(
            duration, snap_duration_to_h3_grid, warnings
        )
        self._warn_reference_video(images, real_fps, warnings)

        measured, measured_kinds = self._resolve_cut_list(
            cut_times, duration, warnings
        )
        keyframes = select_keyframes(
            images, real_fps, max_frames=max_frames_to_analyze,
            known_boundaries=(
                [int(round(t * real_fps)) for t in measured]
                if measured else None
            ),
        )
        print(
            f"{LOG_PREFIX} {images.shape[0]} frames @ {real_fps:.3f} fps "
            f"({duration:.3f}s), {len(parsed)} subject(s), "
            f"{len(wired)} image(s) wired"
        )

        if video is None and video_role != "subject_source":
            warnings.append(
                f"video_role is '{video_role}' but no video is connected; "
                "it was ignored"
            )
            video_role = "subject_source"
        if audio is None and audio_role != "none":
            warnings.append(
                f"audio_role is '{audio_role}' but no audio is connected; "
                "it was ignored"
            )
            audio_role = "none"

        music_is_reference, audio_role = self._resolve_music(
            music_video, music_source, audio, audio_role, lyrics,
            music_description, warnings,
        )

        task_types = spec.derive_task_types(video_role, audio_role)
        task_type = spec.format_task_type(task_types)
        is_editing = "video editing" in task_types
        print(f"{LOG_PREFIX} task type: [{task_type}]")

        backend = get_backend(vlm_provider, model, api_key)
        vlm_images = self._build_images(
            parsed, subject_images, keyframes, images, warnings
        )
        vlm_audio = self._resolve_audio(backend, audio, warnings)

        user_context = prompts_ref.build_user_context(
            parsed, target_description, duration, real_fps, task_type,
            frame_timestamps=keyframes.timestamps,
            cut_timestamps=measured or [
                round(b / real_fps, 3)
                for b in keyframes.scene_boundaries if b > 0
            ],
            cut_source="measured" if measured else "local",
            cut_kinds=measured_kinds,
            audio_available=vlm_audio is not None,
            dialogue_text=dialogue,
            constraint_notes=constraint_notes,
            image_tags=image_tags,
            has_video=video is not None,
            video_role=video_role,
            music_video=music_video,
            lyrics=lyrics,
            music_description=music_description,
            music_is_reference=music_is_reference,
        )
        system = prompts_ref.build_system_prompt(parsed, task_types)

        ctx = CowboyContext(
            subjects=parsed, duration_seconds=duration, task_type=task_type,
            mode=h3_mode, known_shot_times=measured, is_editing=is_editing,
            dialogue_text=dialogue, wired_pictures=len(wired),
            has_video=video is not None, has_audio=vlm_audio is not None,
            lyrics=lyrics if music_video else "", music_video=music_video,
        )
        folded = normalize_seed(seed)
        if folded != seed:
            warnings.append(
                f"seed {seed} was folded to {folded} for the provider's "
                "signed-int32 range"
            )
        result, attempts = self._run(
            backend, system, vlm_images, user_context, ctx, folded, vlm_audio,
        )

        warnings.extend(result.warnings)
        warnings.extend(f"unresolved after retry: {e}" for e in
                        result.retry_errors)
        for warning in warnings:
            print(f"{LOG_PREFIX} warning: {warning}")

        analysis = {
            "mode": h3_mode,
            "task_type": task_type,
            "task_types": task_types,
            "video_role": video_role,
            "audio_role": audio_role,
            "music_video": music_video,
            "music_source": music_source,
            "music_is_reference": music_is_reference,
            "subjects": [s.describe() for s in parsed],
            "subject_kinds": [s.kind for s in parsed],
            "image_tags": image_tags,
            "cut_source": "measured" if measured else "local",
            "cut_timestamps": measured,
            "selected_frame_indices": keyframes.indices,
            "provider": vlm_provider,
            "attempts": attempts,
            "applied_fixes": result.applied_fixes,
            "warnings": warnings,
            "unresolved_errors": result.retry_errors,
            "char_count": result.char_count,
            "description_word_count": result.description_word_count,
            "duration_source": duration_source,
            "requested_duration_seconds": round(requested_duration, 3),
            "snapped_duration_seconds": round(duration, 3),
            "snap_duration_to_h3_grid": snap_duration_to_h3_grid,
            "h3_length_frames": length,
        }
        width, height = self._canvas(images, subject_images)
        return self._outputs(
            result.prompt, duration, real_fps, analysis, h3_mode,
            images=subject_images, frames=images, audio=audio,
            width=width, height=height, length=length,
            label_map=build_label_map(
                [f"subject_{slot}_image" for slot in wired],
                has_video=True, has_audio=audio is not None,
            ),
        )

    # -- base mode ----------------------------------------------------------

    def _generate_base(
        self, h3_mode: str, subjects: str, target_description: str,
        vlm_provider: str, model: str, video=None, frames=None,
        fps: float = 24.0, audio=None, api_key: str = "",
        video_role: str = "subject_source", audio_role: str = "none",
        cut_times: str = "", dialogue: str = "", constraint_notes: str = "",
        duration_override: float = 0.0, max_frames_to_analyze: int = 8,
        seed: int = 0, base_picture_role: str = DEFAULT_PICTURE_ROLE,
        fl2va_normalize_picture_tags: bool = False,
        snap_duration_to_h3_grid: bool = True,
        music_video: bool = False, lyrics: str = "",
        music_description: str = "",
        subject_images: Optional[dict] = None,
        warnings: Optional[List[str]] = None,
    ) -> Tuple:
        """
        Write a base-mode prompt: T2VA, I2VA, FL2VA or L2VA.

        Everything here is optional, including the clip - T2VA builds a
        timeline from text alone. The ref-mode widgets stay on the node
        face and are ignored with a warning rather than an error, because
        the same graph is meant to switch modes without being rewired.
        """
        warnings = warnings if warnings is not None else []
        subject_images = subject_images or {}

        anchors = self._resolve_anchors(h3_mode, subject_images, warnings)
        self._warn_ref_only_widgets(
            h3_mode, subjects, video, frames, audio, video_role, audio_role,
            base_picture_role, fl2va_normalize_picture_tags, warnings,
            rows=self._read_rows(subject_images),
            music_video=music_video,
            music_text=(lyrics + music_description),
        )

        images, real_fps, duration_source = self._resolve_frames(
            video, frames, fps, warnings, required=False
        )
        if duration_override > 0:
            duration = float(duration_override)
            duration_source = "override"
        elif images is not None:
            duration = images.shape[0] / real_fps
        else:
            duration = DEFAULT_BASE_DURATION
            duration_source = "default"
            warnings.append(
                f"no video, no frames and no duration_override, so the "
                f"target duration defaulted to {DEFAULT_BASE_DURATION:.2f}s. "
                "That number is written into the instruction line and the "
                "shot times; set duration_override to the length you "
                "actually want."
            )

        # Same snap as ref mode, and it matters more here: base mode
        # writes the duration into the instruction line as S.SS.
        requested_duration = duration
        length, duration = self._snap(
            duration, snap_duration_to_h3_grid, warnings
        )

        # A cut list is not a measurement here - base mode has no source
        # clip - so it is read as the shot structure the user is asking
        # for. More than one entry IS the "explicitly specified" that
        # guide_base 3.2 requires before FL2VA writes multiple shots.
        measured, _kinds = self._resolve_cut_list(
            cut_times, duration, warnings
        )
        keyframes = None
        if images is not None:
            keyframes = select_keyframes(
                images, real_fps, max_frames=max_frames_to_analyze
            )
        print(
            f"{LOG_PREFIX} {h3_mode} at {duration:.3f}s, "
            f"{len(anchors)} anchor picture(s), "
            f"{0 if keyframes is None else len(keyframes.indices)} "
            "context frame(s)"
        )

        backend = get_backend(vlm_provider, model, api_key)
        vlm_images = self._build_base_images(
            h3_mode, anchors, subject_images, keyframes, images
        )
        user_context = prompts_base.build_user_context(
            h3_mode, target_description, duration, real_fps,
            frame_timestamps=keyframes.timestamps if keyframes else None,
            dialogue_text=dialogue, constraint_notes=constraint_notes,
        )
        system = prompts_base.build_system_prompt(h3_mode)

        ctx = CowboyContext(
            subjects=[], duration_seconds=duration, task_type="",
            mode=h3_mode, known_shot_times=measured, is_editing=False,
            dialogue_text=dialogue, wired_pictures=len(anchors),
            has_video=False, has_audio=False,
            multi_shot_requested=len(measured) > 1,
            fl2va_normalize_picture_tags=fl2va_normalize_picture_tags,
        )
        folded = normalize_seed(seed)
        if folded != seed:
            warnings.append(
                f"seed {seed} was folded to {folded} for the provider's "
                "signed-int32 range"
            )
        result, attempts = self._run(
            backend, system, vlm_images, user_context, ctx, folded, None,
        )

        warnings.extend(result.warnings)
        warnings.extend(
            f"unresolved after retry: {e}" for e in result.retry_errors
        )
        for warning in warnings:
            print(f"{LOG_PREFIX} warning: {warning}")

        analysis = {
            "mode": h3_mode,
            "task_type": "",
            "task_types": [],
            "base_picture_role": base_picture_role,
            "fl2va_normalize_picture_tags": fl2va_normalize_picture_tags,
            "anchor_pictures": [f"Picture {slot}" for slot in anchors],
            "requested_shot_times": measured,
            "selected_frame_indices": (
                keyframes.indices if keyframes else []
            ),
            "provider": vlm_provider,
            "attempts": attempts,
            "applied_fixes": result.applied_fixes,
            "warnings": warnings,
            "unresolved_errors": result.retry_errors,
            "char_count": result.char_count,
            "description_word_count": result.description_word_count,
            "duration_source": duration_source,
            "requested_duration_seconds": round(requested_duration, 3),
            "snapped_duration_seconds": round(duration, 3),
            "snap_duration_to_h3_grid": snap_duration_to_h3_grid,
            "h3_length_frames": length,
        }
        # No frames and no audio out of base mode, on purpose: the base
        # format has no <Video 1> and no <Audio 1> to cite them with.
        width, height = self._canvas(None, subject_images)
        return self._outputs(
            result.prompt, duration, real_fps, analysis, h3_mode,
            images=subject_images, frames=None, audio=None,
            width=width, height=height, length=length,
            label_map=build_label_map(
                [f"subject_{slot}_image" for slot in anchors]
            ),
        )

    def _resolve_anchors(
        self, h3_mode: str, subject_images: dict, warnings: List[str],
    ) -> List[int]:
        """
        Which wired slots are this mode's frame anchors.

        subject_N_image IS <Picture N> in base mode too, so slot 1 is the
        anchor and slot 2 is FL2VA's last frame. Too few is the one
        genuinely impossible state on this path and raises; anything else
        warns.
        """
        wired = [
            slot for slot in range(1, NUM_SUBJECT_SLOTS + 1)
            if subject_images.get(f"subject_{slot}_image") is not None
        ]
        needed = PICTURES_FOR_MODE[h3_mode]
        anchors = [slot for slot in wired if slot <= needed]

        if len(anchors) < needed:
            missing = [
                f"subject_{slot}_image" for slot in range(1, needed + 1)
                if slot not in wired
            ]
            raise RuntimeError(
                f"h3_mode '{h3_mode}' needs {needed} anchor picture(s) and "
                f"{len(anchors)} are wired. Connect "
                + " and ".join(missing)
                + " (slot 1 is <Picture 1>, slot 2 is <Picture 2>), or "
                "switch to base_T2VA, which needs none."
            )
        extra = [slot for slot in wired if slot > needed]
        if extra:
            warnings.append(
                f"{h3_mode} uses {needed} picture(s), so "
                + ", ".join(f"subject_{slot}_image" for slot in extra)
                + " was ignored"
            )
        return anchors

    def _warn_ref_only_widgets(
        self, h3_mode: str, subjects: str, video, frames, audio,
        video_role: str, audio_role: str, base_picture_role: str,
        fl2va_normalize_picture_tags: bool, warnings: List[str],
        rows: Optional[List[SubjectRow]] = None,
        music_video: bool = False, music_text: str = "",
    ) -> None:
        """
        Say what base mode ignores, once each, and never raise.

        The node keeps one face for both modes so a graph can switch
        without being rewired, which means several widgets have nothing
        to do here. guide_base has no reference labels at all: no
        <Subject N>, no <Video N>, no <Audio N>, no task-type prefix.
        """
        filled = [row.slot for row in (rows or []) if row.is_filled]
        if subjects.strip() or filled:
            where = (
                "subject row " + ", ".join(str(s) for s in filled)
                if filled else "the subjects field"
            )
            warnings.append(
                f"base mode has no reference labels, so {where} was "
                "ignored. Describe what you want in target_description "
                "instead, or switch to ref mode."
            )
        if music_video or music_text.strip():
            warnings.append(
                "music_video is a ref-mode setting - the base format has no "
                "audio reference to reuse - so it was ignored. Describe the "
                "track in target_description, or switch to ref mode."
            )
        if video is not None or frames is not None:
            warnings.append(
                "base mode has no <Video N>: the wired clip supplies the "
                "duration and some context frames, and nothing in it can "
                "be cited by the prompt."
            )
        if audio is not None:
            warnings.append(
                "base mode has no <Audio N>; the wired audio was ignored. "
                "Describe the sound you want in target_description."
            )
        if video_role != "subject_source":
            warnings.append(
                f"video_role is '{video_role}', which is a ref-mode "
                "setting; it was ignored"
            )
        if audio_role != "none":
            warnings.append(
                f"audio_role is '{audio_role}', which is a ref-mode "
                "setting; it was ignored"
            )
        if fl2va_normalize_picture_tags and h3_mode != "base_FL2VA":
            warnings.append(
                "fl2va_normalize_picture_tags only applies to base_FL2VA; "
                f"it was ignored for {h3_mode}"
            )

        implied = PICTURE_ROLE_FOR_MODE.get(h3_mode)
        if (
            implied
            and base_picture_role != implied
            and base_picture_role != DEFAULT_PICTURE_ROLE
        ):
            warnings.append(
                f"base_picture_role says '{base_picture_role}' but h3_mode "
                f"'{h3_mode}' means '{implied}'. h3_mode wins. Switch to "
                + ("base_L2VA" if implied == "first_frame" else "base_I2VA")
                + " if the picture really is the "
                + base_picture_role.replace("_", " ")
            )

    def _build_base_images(
        self, h3_mode: str, anchors: List[int], subject_images: dict,
        keyframes, frames,
    ) -> List[VLMImage]:
        """Anchor frames first, in slot order, then any context frames."""
        roles = {
            "base_I2VA": {1: "the FIRST frame of the target video, at 0.00 "
                             "seconds"},
            "base_L2VA": {1: "the LAST frame of the target video"},
            "base_FL2VA": {
                1: "the FIRST frame of the target video, at 0.00 seconds",
                2: "the LAST frame of the target video",
            },
        }.get(h3_mode, {})

        out = []
        for slot in anchors:
            tensor = subject_images.get(f"subject_{slot}_image")
            if tensor is None:
                continue
            out.append(VLMImage(
                label=(
                    f"<Picture {slot}> - {roles.get(slot, 'a frame anchor')}"
                ),
                jpeg_b64=tensor_to_jpeg_b64(
                    tensor, max_side=REFERENCE_MAX_SIDE
                ),
            ))

        if keyframes is None or frames is None:
            return out
        total = len(keyframes.indices)
        for pos, (idx, ts) in enumerate(
            zip(keyframes.indices, keyframes.timestamps), start=1
        ):
            out.append(VLMImage(
                label=(
                    "Context frame " + frame_label(pos, total, ts, idx)
                    + " - NOT a frame of the target video and not citable"
                ),
                jpeg_b64=tensor_to_jpeg_b64(
                    frames[idx], max_side=FRAME_MAX_SIDE
                ),
            ))
        return out

    # -- the rows and the music ---------------------------------------------

    def _read_rows(self, fields: dict) -> List[SubjectRow]:
        """
        All six rows, whatever the row count says.

        subject_rows is a display setting - it decides how many rows the
        node face shows - and this deliberately ignores it. A row holding
        text is a subject even if it is scrolled out of sight, because a
        value that silently stops counting when a widget hides is the
        worst kind of surprise. The JS keeps a filled row visible for the
        same reason.
        """
        return [
            SubjectRow(
                slot=slot,
                kind=str(fields.get(
                    f"subject_{slot}_kind", DEFAULT_ROW_KINDS[slot - 1]
                )),
                name=str(fields.get(f"subject_{slot}_name", "") or ""),
                description=str(
                    fields.get(f"subject_{slot}_description", "") or ""
                ),
            )
            for slot in range(1, NUM_SUBJECT_SLOTS + 1)
        ]

    def _no_subjects_message(self, subjects: str) -> str:
        """
        Why the run has nothing to define, and what to do about it.

        Two different mistakes reach this point. Typing into the advanced
        field and getting the syntax wrong needs the syntax; an empty
        node needs to be told the rows exist at all.
        """
        rows_line = (
            "Fill row 1: pick what subject_1_image is - character, "
            "environment or another kind - and describe it in "
            "subject_1_description. A row needs no image, which is how "
            "you ask for a style or an action."
        )
        if subjects.strip():
            return (
                "No subjects could be read. The typed subjects field holds "
                "text, but no line parsed. Each line is "
                "'<kind> [name] [@Picture N] -- <features>', kind first, "
                "one of: " + ", ".join(spec.KIND_CHOICES) + ". " + rows_line
            )
        return "No subjects yet. " + rows_line

    def _resolve_music(
        self, music_video: bool, music_source: str, audio,
        audio_role: str, lyrics: str, music_description: str,
        warnings: List[str],
    ) -> Tuple[bool, str]:
        """
        (is the song reused as <Audio 1>, the audio_role that follows).

        Only a REUSED song changes the task type, and it changes it the
        same way a reused clip track does, so it goes through audio_role
        rather than beside it. 'auto' assumes the file wired here is the
        file the H3 graph is fed, which is the common case and not a
        guarantee - hence the two explicit values.
        """
        if not music_video:
            if lyrics.strip() or music_description.strip():
                warnings.append(
                    "lyrics or music_description are set but music_video is "
                    "off, so they were ignored. Turn music_video on."
                )
            elif music_source != "auto":
                warnings.append(
                    f"music_source is '{music_source}' but music_video is "
                    "off; it was ignored"
                )
            return False, audio_role

        is_reference = (
            music_source == "reuse_audio_1"
            or (music_source == "auto" and audio is not None)
        )
        if is_reference and audio_role != "reuse":
            if audio_role != "none":
                warnings.append(
                    f"audio_role '{audio_role}' was overridden by "
                    f"music_source '{music_source}': the song is reused, so "
                    "the task type says audio reuse"
                )
            audio_role = "reuse"
        if music_source == "reuse_audio_1" and audio is None:
            warnings.append(
                "music_source is 'reuse_audio_1' with no audio wired here. "
                "The prompt declares <Audio 1> anyway, so the H3 sampler "
                "must be given that track or the tag points at nothing."
            )
        if not is_reference and audio_role == "reuse":
            warnings.append(
                "audio_role is 'reuse' but music_source says the score is "
                "generated; the task type follows audio_role"
            )
        return is_reference, audio_role

    # -- pass-through and timing --------------------------------------------

    def _outputs(
        self, prompt: str, duration: float, fps: float, analysis: dict,
        mode: str, *, images: Optional[dict] = None, frames=None,
        audio=None, width: int = 0, height: int = 0, length: int = 0,
        label_map: str = "",
    ) -> Tuple:
        """
        The node's whole return tuple, built once for both modes.

        ComfyUI reads outputs by position, so ref mode and base mode must
        return the same count in the same order or the node breaks the
        moment h3_mode changes. Two hand-maintained return statements
        drift the first time someone adds an output to one of them.
        """
        images = images or {}
        return (
            prompt, round(duration, 3), int(round(fps)),
            json.dumps(analysis, indent=2),
            spec.CHECKPOINT_FOR_MODE.get(mode, ""),
            *(images.get(f"subject_{slot}_image")
              for slot in range(1, NUM_SUBJECT_SLOTS + 1)),
            frames, audio, audio,
            int(width), int(height), int(length), label_map,
        )

    def _snap(
        self, duration: float, enabled: bool, warnings: List[str],
    ) -> Tuple[int, float]:
        """
        (frame count on H3's grid, the duration the prompt should state).

        H3 rounds a length UP to the 17k+5 frame grid at 24 fps, so 2.00
        seconds really renders 2.33. The frame count is returned either
        way - the sampler only accepts grid values - but the duration is
        only corrected when the toggle is on. Off, the prompt claims a
        length H3 does not produce, which is worth a warning.
        """
        length, snapped = snap_length(duration)
        if enabled:
            print(
                f"{LOG_PREFIX} length {duration:.3f}s -> {snapped:.3f}s "
                f"({length} frames on H3's 17k+5 grid at {H3_FPS} fps)"
            )
            return length, snapped
        if abs(snapped - duration) > 1e-6:
            warnings.append(
                f"H3 renders {length} frames, which is {snapped:.3f}s at "
                f"{H3_FPS} fps, not the {duration:.3f}s this prompt states - "
                "frame counts sit on a 17k+5 grid and round up. Turn "
                "snap_duration_to_h3_grid on to make the prompt state the "
                "length H3 really produces."
            )
        return length, duration

    def _canvas(self, frames, subject_images: dict) -> Tuple[int, int]:
        """
        Sampler width and height, on H3's canvas grid.

        The video's shape wins over the first picture's, because it
        carries the shot's real framing. (0, 0) when nothing is wired.
        """
        source = frames
        if source is None:
            for slot in range(1, NUM_SUBJECT_SLOTS + 1):
                candidate = subject_images.get(f"subject_{slot}_image")
                if candidate is not None:
                    source = candidate
                    break
        if source is None or source.dim() != 4:
            return 0, 0
        return canvas_for(int(source.shape[2]), int(source.shape[1]))

    def _warn_gap_consequence(
        self, wired: List[int], warnings: List[str],
    ) -> None:
        """
        What a gap in the picture slots costs at the sampler.

        bind_images() already warns that the picture numbers skip. This
        adds what that means downstream, which is the failure this whole
        pass-through exists for: the sampler numbers by arrival, so it
        never sees the gap the prompt was written around.
        """
        if wired and wired != list(range(1, len(wired) + 1)):
            warnings.append(
                "The sampler numbers the images it receives, so every tag "
                "after the gap shifts down and will not match the prompt. "
                "Fill the slots in order."
            )

    def _warn_reference_video(
        self, images, real_fps: float, warnings: List[str],
    ) -> None:
        """The two ways a clip is wrong for the ref_video_ socket."""
        if abs(real_fps - H3_FPS) > FPS_WARN_TOLERANCE:
            warnings.append(
                f"the video is {real_fps:.3f} fps. H3 reads reference "
                "frames as 24 fps and nothing resamples them, so the "
                "reference plays at the wrong speed. Re-time the clip to "
                "24 fps first."
            )
        count = int(images.shape[0])
        if trim_reference_frames(count) == 0:
            warnings.append(
                f"the clip is {count} frame(s). The sampler needs at least "
                "5 on a ref_video_ socket and raises below that, so leave "
                "ref_video unconnected for this one."
            )

    # -- helpers ------------------------------------------------------------

    def _run(self, backend, system, images, user_context, ctx, seed, audio):
        """
        One call, then at most one retry carrying the validator's errors.

        The better attempt wins. Overwriting unconditionally lets a worse
        retry replace a good first pass, which is a real way to spend an
        API call and come out behind.
        """
        best = None
        attempts = []
        prompt_text = user_context
        for attempt in range(MAX_RETRIES + 1):
            reply = backend.generate(
                system, images, prompt_text, seed=seed, audio=audio,
            )
            result = process(reply.text, ctx)
            attempts.append({
                "errors": list(result.retry_errors),
                "fixes": list(result.applied_fixes),
            })
            if best is None or len(result.retry_errors) <= len(
                best.retry_errors
            ):
                best = result
            if not result.retry_errors or attempt == MAX_RETRIES:
                break
            print(
                f"{LOG_PREFIX} retrying with "
                f"{len(result.retry_errors)} validation error(s)"
            )
            prompt_text = (
                user_context
                + "\n\nYOUR PREVIOUS ATTEMPT:\n" + reply.text
                + "\n\nIt did not satisfy these. Fix every one and write "
                "the whole prompt again:\n"
                + "\n".join(f"- {e}" for e in result.retry_errors)
            )
        return best, attempts

    def _build_images(self, subjects, subject_images, keyframes, frames,
                      warnings) -> List[VLMImage]:
        """Subject pictures first, in slot order, then sampled frames."""
        out = []
        for subject in subjects:
            if subject.slot is None:
                continue
            tensor = subject_images.get(f"subject_{subject.slot}_image")
            if tensor is None:
                continue
            label = (
                f"Reference image <Picture {subject.slot}> - "
                f"{subject.kind}"
                + (f" '{subject.name}'" if subject.name else "")
                + f", which is {subject.tag}"
            )
            out.append(VLMImage(
                label=label,
                jpeg_b64=tensor_to_jpeg_b64(
                    tensor, max_side=REFERENCE_MAX_SIDE
                ),
            ))

        total = len(keyframes.indices)
        for pos, (idx, ts) in enumerate(
            zip(keyframes.indices, keyframes.timestamps), start=1
        ):
            out.append(VLMImage(
                label=frame_label(pos, total, ts, idx),
                jpeg_b64=tensor_to_jpeg_b64(
                    frames[idx], max_side=FRAME_MAX_SIDE
                ),
            ))
        return out

    def _resolve_audio(self, backend, audio, warnings) -> Optional[VLMAudio]:
        if audio is None:
            return None
        if not getattr(backend, "supports_audio", False):
            warnings.append(
                f"provider '{backend.name}' cannot accept audio; the "
                "soundscape will be inferred from the frames"
            )
            return None
        try:
            wav_b64, seconds = audio_io.audio_to_wav_b64(audio)
        except RuntimeError as exc:
            warnings.append(f"audio not sent: {exc}")
            return None
        return VLMAudio(wav_b64=wav_b64, duration_seconds=seconds)

    def _resolve_cut_list(self, cut_times: str, duration: float, warnings):
        """Read any Cut Detective output into (times, kinds)."""
        if not cut_times or not cut_times.strip():
            return [], []
        cuts = parse_cut_times(cut_times)
        if not cuts:
            warnings.append(
                "cut_times had no readable timestamps; falling back to "
                "local cut detection"
            )
            return [], []
        kept = [c for c in cuts if c.time < duration]
        if len(kept) != len(cuts):
            warnings.append(
                f"{len(cuts) - len(kept)} cut time(s) past the "
                f"{duration:.3f}s clip were dropped"
            )
        if not kept:
            return [], []
        if kept[0].time >= 0.001:
            kept.insert(0, type(kept[0])(0.0, "start"))
        else:
            kept[0] = type(kept[0])(0.0, "start")
        return [round(c.time, 3) for c in kept], [c.kind for c in kept]

    def _resolve_frames(
        self, video, frames, fps: float, warnings, required: bool = True,
    ):
        """
        Resolve (frames_tensor, fps, source) from the two input paths.

        required=False returns (None, fps, "none") instead of raising,
        which is the T2VA path: base mode can write a whole video from
        text, so there may be no clip at all. It is a parameter rather
        than a loosened error because ref mode genuinely cannot run
        without one, and that error is worth keeping sharp.
        """
        if video is not None:
            if frames is not None:
                warnings.append("both video and frames connected; using video")
            try:
                components = video.get_components()
            except AttributeError as exc:
                raise RuntimeError(
                    "The video input is not a ComfyUI VIDEO object. "
                    "Connect a Load Video node, or use the frames input."
                ) from exc
            rate = components.frame_rate
            real_fps = (
                float(rate) if isinstance(rate, Fraction)
                else float(rate or 24.0)
            )
            return components.images, real_fps, "video"

        if frames is not None:
            if frames.dim() != 4 or frames.shape[0] < 1:
                raise RuntimeError(
                    "frames must be a non-empty IMAGE batch (B, H, W, C)."
                )
            return frames, float(fps), "frames+fps"

        if not required:
            return None, float(fps), "none"

        raise RuntimeError(
            "Connect either a VIDEO (video input) or an IMAGE batch "
            "(frames input, with fps set)."
        )


NODE_CLASS_MAPPINGS = {
    "TrentUltimateH3CowboyPromptor": UltimateH3CowboyPromptor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrentUltimateH3CowboyPromptor": "Ultimate H3 Cowboy Promptor",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
