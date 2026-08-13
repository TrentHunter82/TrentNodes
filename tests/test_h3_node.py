"""
End-to-end node test with a fake VLM backend. CPU-only, no ComfyUI
server. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_h3_node.py
"""

import json
import os
import sys
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("nodes", "utils", "utils.h3_prompt"):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

import torch  # noqa: E402

from TrentNodes.nodes import h3_auto_prompt  # noqa: E402
from TrentNodes.utils.h3_prompt.backends import VLMResult  # noqa: E402

CANNED = """subject_definitions:
<Subject 1> is Aria Voss as shown in <Picture 1>. Preserve Aria Voss's exact facial identity, hair, skin tone, body proportions, and likeness. <Picture 1> also provides the exact wardrobe: a charcoal utility jacket, black cargo pants, and combat boots. <Video 1> supplies only the exact body movement, camera angles, and framing. Do not copy any identity from the performer in <Video 1>.

summary:
Generate a new video in which Aria Voss is continuously tracked through every frame of the scene from <Video 1>, preserving identity and wardrobe from <Picture 1>.

retention_analysis:
<Subject 1> Aria Voss: fully_preserved - zero drift in every frame, including the charcoal utility jacket, black cargo pants, and combat boots. <Video 1>: attribute_transfer - motion and camera only.

detailed_description:
The target video has the grounded cinematic appearance of a gritty thriller. [Shot 1] A static medium shot holds Aria Voss <Subject 1> center frame as she walks forward. [Shot 2] At 00:01.000, a handheld camera tracks left; Aria Voss <Subject 1> moves to the right third of frame, midground, and reaches for a crate.

overall_soundscape:
Room tone with footfalls matching Aria Voss's visible steps.

non_diegetic_music:
N/A

No face morphing of Aria Voss. No wardrobe changes. No extra characters. No text overlays. No morphing seams. No camera moves beyond <Video 1>. No hair changes. No added accessories.
"""


class FakeBackend:
    name = "fake"
    supports_audio = False
    supports_video = False

    def __init__(self):
        self.calls = 0
        self.last_user_text = ""
        # The retry message quotes the validator's errors back, so a
        # test about the task context must read the first call, not the
        # last, or it will match on error text instead.
        self.first_user_text = ""
        self.image_labels = []
        self.reply = CANNED

    def generate(self, system, images, user_text, max_tokens=4096, seed=0,
                 audio=None, video=None):
        self.calls += 1
        self.last_user_text = user_text
        if self.calls == 1:
            self.first_user_text = user_text
            self.image_labels = [im.label for im in images]
        assert "UNBREAKABLE RULES" in system
        assert images[0].label.startswith("Reference image <Picture 1>")
        assert "TASK CONTEXT" in user_text
        return VLMResult(text=self.reply, usage={"model": "fake-model"})


def _run_node(fake, **overrides):
    original = h3_auto_prompt.get_backend
    h3_auto_prompt.get_backend = lambda *a, **k: fake
    try:
        node = h3_auto_prompt.H3AutoPromptGenerator()
        kwargs = dict(
            reference_image=torch.rand((1, 96, 96, 3)),
            subject_name="Aria Voss",
            subject_wardrobe="charcoal utility jacket, black cargo pants, combat boots",
            scene_style="gritty thriller",
            soundscape_type="ambient",
            vlm_provider="anthropic",
            model="auto",
            max_frames_to_analyze=4,
            enable_audio_prompt=True,
            frames=torch.rand((24, 64, 64, 3)),
            fps=12.0,
        )
        kwargs.update(overrides)
        return node.generate(**kwargs)
    finally:
        h3_auto_prompt.get_backend = original


def test_node_end_to_end():
    fake = FakeBackend()
    prompt, prompt_b, duration, fps, analysis_json = _run_node(fake)

    assert fake.calls == 1
    assert prompt.startswith("subject_definitions:")
    assert "Aria Voss <Subject 1>" in prompt
    assert prompt_b == ""
    assert abs(duration - 2.0) < 1e-6
    assert fps == 12

    analysis = json.loads(analysis_json)
    assert analysis["provider"] == "anthropic"
    assert analysis["model"] == "fake-model"
    assert analysis["duration_source"] == "frames+fps"
    assert analysis["profile_mode"] == "official"
    assert len(analysis["selected_frame_indices"]) <= 4
    official = analysis["variants"]["official"]
    assert official["attempts"][0]["errors"] == []


def test_node_both_ab():
    fake = FakeBackend()
    prompt, prompt_b, _duration, _fps, analysis_json = _run_node(
        fake, prompt_profile="both_ab"
    )

    assert fake.calls == 2
    assert prompt.startswith("subject_definitions:")
    assert prompt_b.startswith("subject_definitions:")

    analysis = json.loads(analysis_json)
    assert analysis["profile_mode"] == "both_ab"
    assert set(analysis["variants"]) == {"official", "upgraded"}
    assert analysis["variants"]["upgraded"]["attempts"][0]["errors"] == []
    # The canned text carries an 8-sentence No-block; the upgraded
    # profile keeps it but flags it as off-profile
    assert any(
        "positive assertions" in w
        for w in analysis["variants"]["upgraded"]["warnings"]
    )


def test_upgraded_profile_no_padding():
    fake = FakeBackend()
    prompt, prompt_b, _d, _f, analysis_json = _run_node(
        fake, prompt_profile="upgraded"
    )
    assert prompt_b == ""
    analysis = json.loads(analysis_json)
    upgraded = analysis["variants"]["upgraded"]
    assert not any(
        "padded exclusions" in f for f in upgraded["applied_fixes"]
    )


def test_blank_cut_times_keeps_local_detection():
    _p, _b, _d, _f, analysis_json = _run_node(FakeBackend())
    analysis = json.loads(analysis_json)
    assert analysis["cut_source"] == "local"
    assert analysis["cut_kinds"] == []
    assert analysis["detection_method"] in ("hybrid", "intensity")


def test_cut_times_pin_the_shot_timeline():
    # The canned text writes [Shot 2] At 00:01.000; Cut Detective
    # measured 1.500, so the measured time must win.
    fake = FakeBackend()
    prompt, _b, _d, _f, analysis_json = _run_node(fake, cut_times="0.0, 1.5")

    assert "[Shot 2] At 00:01.500" in prompt
    assert "00:01.000" not in prompt
    assert fake.calls == 1  # count matched, so no retry

    analysis = json.loads(analysis_json)
    assert analysis["cut_source"] == "measured"
    assert analysis["cut_timestamps"] == [0.0, 1.5]
    assert analysis["detection_method"] == "supplied"
    # Keyframes land on the measured cut, not on a difference spike.
    assert 18 in analysis["selected_frame_indices"], analysis[
        "selected_frame_indices"
    ]


def test_cut_times_reads_a_cut_detective_shot_table():
    fake = FakeBackend()
    table = (
        "Shot 1 | 00:00.000 | 1.250s | start\n"
        "Shot 2 | 00:01.250 | 0.750s | dissolve over 0.125s"
    )
    prompt, _b, _d, _f, analysis_json = _run_node(fake, cut_times=table)

    assert "[Shot 2] At 00:01.250" in prompt
    analysis = json.loads(analysis_json)
    assert analysis["cut_timestamps"] == [0.0, 1.25]
    assert analysis["cut_kinds"] == ["start", "dissolve"]


def test_cut_times_shot_count_mismatch_forces_a_retry():
    # Three measured shots, two written: the model must be sent back.
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(
        fake, cut_times="0.0, 0.5, 1.2"
    )
    assert fake.calls == 2  # one retry, and the fake never corrects itself

    analysis = json.loads(analysis_json)
    unresolved = analysis["variants"]["official"]["unresolved_errors"]
    assert any("measured shot list has 3 shots" in e for e in unresolved), (
        unresolved
    )


def test_cut_times_past_the_clip_end_are_dropped():
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(
        fake, cut_times="0.0, 1.5, 9.0"  # clip is 2.000s
    )
    analysis = json.loads(analysis_json)
    assert analysis["cut_timestamps"] == [0.0, 1.5]
    assert any("past the" in w for w in analysis["warnings"]), analysis[
        "warnings"
    ]


def test_unreadable_cut_times_fall_back_to_local_detection():
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(fake, cut_times="no cuts here")
    analysis = json.loads(analysis_json)
    assert analysis["cut_source"] == "local"
    assert any("no readable timestamps" in w for w in analysis["warnings"])


def test_boundaries_only_cut_list_regains_shot_one():
    # Cut Detective with include_first_shot off emits no leading 0.000.
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(fake, cut_times="1.5")
    analysis = json.loads(analysis_json)
    assert analysis["cut_timestamps"] == [0.0, 1.5]
    assert analysis["cut_kinds"][0] == "start"


def test_alignment_toggle_is_off_by_default():
    prompt, _b, _d, _f, analysis_json = _run_node(FakeBackend())
    assert prompt.startswith("subject_definitions:")
    assert "fully referenced" not in prompt
    assert json.loads(analysis_json)["alignment_hook"] == ""


def test_first_frame_alignment_prepends_the_hook():
    fake = FakeBackend()
    prompt, _b, _d, _f, analysis_json = _run_node(
        fake, first_frame_alignment=True
    )
    expected = (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )
    assert prompt.startswith(expected)
    assert json.loads(analysis_json)["alignment_hook"] == expected
    # The task context tells the model the sentence is coming.
    assert "FIRST-FRAME ALIGNMENT IS ON" in fake.first_user_text


def test_alignment_time_names_the_shot_it_lands_in():
    # Cuts at 0.0 / 0.5 / 1.2; 1.3s falls inside shot 3. A non-zero
    # moment is the official L2VA sentence, not the I2VA one with a
    # substituted time - that hybrid appears in no MiniMax guide.
    fake = FakeBackend()
    prompt, _b, _d, _f, _a = _run_node(
        fake, first_frame_alignment=True, alignment_time_seconds=1.3,
        cut_times="0.0, 0.5, 1.2",
    )
    assert prompt.startswith(
        "How the reference pictures align with the target video - "
        "<Picture 1> (from [Shot 3]) aligns with the 1.30-second mark "
        "of the target video."
    ), prompt[:160]


# ---------------------------------------------------------------------------
# Hybrid graph: a character reference AND an injected opening frame
# ---------------------------------------------------------------------------

def test_a_hybrid_graph_aligns_picture_2_not_the_character_sheet():
    # The whole point: <Picture 1> may be a multi-angle character sheet,
    # so declaring it to BE the opening frame tells H3 to open the video
    # on a contact sheet.
    fake = FakeBackend()
    prompt, _b, _d, _f, analysis_json = _run_node(
        fake, first_frame_alignment=True,
        first_frame_image=torch.rand((1, 96, 96, 3)),
    )
    assert prompt.startswith(
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 2> (from [Shot 1]) is fully referenced."
    ), prompt[:140]
    analysis = json.loads(analysis_json)
    assert analysis["alignment_picture"] == "Picture 2"
    assert analysis["has_first_frame_image"] is True


def test_without_a_first_frame_image_picture_1_is_still_aligned():
    fake = FakeBackend()
    prompt, _b, _d, _f, analysis_json = _run_node(
        fake, first_frame_alignment=True
    )
    assert "<Picture 1> (from [Shot 1]) is fully referenced." in prompt
    assert json.loads(analysis_json)["alignment_picture"] == "Picture 1"


def test_the_hybrid_first_frame_reaches_the_vlm_as_picture_2():
    fake = FakeBackend()
    _run_node(
        fake, first_frame_alignment=True,
        first_frame_image=torch.rand((1, 96, 96, 3)),
    )
    assert fake.image_labels[0].startswith("Reference image <Picture 1>")
    assert fake.image_labels[1].startswith("Reference image <Picture 2>")
    assert "opening frame" in fake.image_labels[1]
    # <Picture 1> is flagged as not being a frame of the video.
    assert "NOT a frame of the target video" in fake.image_labels[0]


def test_the_hybrid_context_separates_the_two_pictures():
    fake = FakeBackend()
    _run_node(
        fake, first_frame_alignment=True,
        first_frame_image=torch.rand((1, 96, 96, 3)),
    )
    context = fake.first_user_text
    assert "THE TWO PICTURES ARE NOT INTERCHANGEABLE" in context
    assert "Never write that a shot begins from <Picture 1>" in context
    # <Picture 2> gets the standalone entry and the retention line.
    assert "<Picture 2> is the first frame of [Shot 1]" in context
    assert "<Picture 2> ([Shot 1] first frame): fully_preserved" in context
    # The image manifest names it too.
    assert "The second image is <Picture 2>" in context


def test_a_hybrid_keeps_picture_1_identity_only_language():
    # Under a single-picture hook the assembler strips "supplies identity
    # only" as a contradiction. In a hybrid that sentence is TRUE and has
    # to survive, because <Picture 1> really does supply identity alone.
    fake = FakeBackend()
    fake.reply = CANNED.replace(
        "<Video 1> supplies only the exact body movement",
        "<Picture 1> supplies only identity and wardrobe. "
        "<Video 1> supplies only the exact body movement",
    )
    prompt, _b, _d, _f, _a = _run_node(
        fake, first_frame_alignment=True,
        first_frame_image=torch.rand((1, 96, 96, 3)),
    )
    assert "<Picture 1> supplies only identity and wardrobe." in prompt


def test_a_first_frame_image_without_the_toggle_warns():
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(
        fake, first_frame_image=torch.rand((1, 96, 96, 3))
    )
    assert any(
        "first_frame_image is connected" in w
        for w in json.loads(analysis_json)["warnings"]
    )


def test_alignment_time_past_the_clip_is_clamped():
    fake = FakeBackend()
    prompt, _b, _d, _f, analysis_json = _run_node(
        fake, first_frame_alignment=True, alignment_time_seconds=99.0
    )
    assert prompt.startswith(
        "How the reference pictures align with the target video - "
        "<Picture 1> (from [Shot 1]) aligns with the 2.00-second mark"
    ), prompt[:160]
    assert any(
        "past the" in w and "clamped" in w
        for w in json.loads(analysis_json)["warnings"]
    )


def test_music_video_off_keeps_the_soundscape_type():
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(fake)
    assert "soundscape type 'ambient'" in fake.first_user_text
    assert "MUSIC VIDEO MODE" not in fake.first_user_text
    assert json.loads(analysis_json)["music_video"] is False


def test_music_video_replaces_the_soundscape_guidance():
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(fake, music_video=True)
    context = fake.first_user_text
    assert "MUSIC VIDEO MODE IS ON" in context
    assert "soundscape type 'ambient'" not in context
    assert "never be \"N/A\"" in context
    assert json.loads(analysis_json)["music_video"] is True


def test_lyrics_and_music_description_reach_the_context():
    fake = FakeBackend()
    _run_node(
        fake, music_video=True,
        lyrics="I'm lonely lonely lonely",
        music_description="downtempo synthwave, ~92 BPM",
    )
    assert "I'm lonely lonely lonely" in fake.first_user_text
    assert "downtempo synthwave, ~92 BPM" in fake.first_user_text
    # No audio connected, so the subject is the vocal source.
    assert "(S1) sings" in fake.first_user_text
    assert "<Audio 1>" not in fake.first_user_text


def test_no_lyrics_forbids_inventing_them():
    fake = FakeBackend()
    _run_node(fake, music_video=True)
    assert "Do not invent words" in fake.first_user_text


def test_music_video_ignores_lyrics_when_off():
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(fake, lyrics="la la la")
    assert "la la la" not in fake.first_user_text
    assert any(
        "music_video is off" in w
        for w in json.loads(analysis_json)["warnings"]
    )


def test_music_video_forces_audio_on():
    fake = FakeBackend()
    _p, _b, _d, _f, analysis_json = _run_node(
        fake, music_video=True, enable_audio_prompt=False
    )
    context = fake.first_user_text
    assert "MUSIC VIDEO MODE IS ON" in context
    assert "audio: minimal" not in context
    assert any(
        "enable_audio_prompt was treated as on" in w
        for w in json.loads(analysis_json)["warnings"]
    )


def test_connected_audio_declares_the_song_as_audio_1():
    # supports_audio so the track actually reaches the VLM; the
    # <Audio 1> declaration is about the H3 graph and does not depend
    # on it, but the "listen to the track" block does.
    fake = FakeBackend()
    fake.supports_audio = True
    audio = {
        "waveform": torch.zeros((1, 1, 16000)),
        "sample_rate": 16000,
    }
    _p, _b, _d, _f, analysis_json = _run_node(
        fake, music_video=True, audio=audio,
        lyrics="I'm lonely lonely lonely",
    )
    context = fake.first_user_text
    assert "SUPPLIED TO THE GENERATION AS <Audio 1>" in context
    assert "fully_copy" in context
    # A voice inside the reused track is not a new speaker.
    assert "Do not assign an (Sx) ID to it." in context
    assert "(S1) sings" not in context
    # The generic "name the impacts / N/A is fine" block would fight
    # music-video mode, so a music-specific one replaces it.
    assert "THE TRACK IS ATTACHED FOR YOU TO HEAR" in context
    assert "specific impacts and movement sounds" not in context
    assert json.loads(analysis_json)["music_is_reference"] is True


def test_node_requires_an_input():
    node = h3_auto_prompt.H3AutoPromptGenerator()
    try:
        node.generate(
            reference_image=torch.rand((1, 8, 8, 3)),
            subject_name="Aria Voss",
            subject_wardrobe="jacket",
            scene_style="style",
            soundscape_type="ambient",
            vlm_provider="anthropic",
            model="auto",
            max_frames_to_analyze=4,
            enable_audio_prompt=True,
        )
    except RuntimeError as exc:
        assert "Connect either" in str(exc)
    else:
        raise AssertionError("expected RuntimeError with no video/frames")


def test_node_class_contract():
    cls = h3_auto_prompt.H3AutoPromptGenerator
    assert cls.CATEGORY == "Trent/VLM"
    assert cls.RETURN_TYPES == ("STRING", "STRING", "FLOAT", "INT", "STRING")
    assert cls.RETURN_NAMES[1] == "h3_prompt_b"
    assert cls.FUNCTION == "generate"
    inputs = cls.INPUT_TYPES()
    assert "video" in inputs["optional"]
    assert inputs["optional"]["video"][0] == "VIDEO"
    assert inputs["required"]["prompt_profile"][0] == [
        "official", "upgraded", "both_ab"
    ]
    providers = inputs["required"]["vlm_provider"][0]
    for expected in ("anthropic", "openai", "kimi", "glm", "qwen_api",
                     "qwen_local", "minicpm_local", "magevl_local",
                     "ollama"):
        assert expected in providers, expected

    from TrentNodes.utils.h3_prompt import backends
    assert set(providers) == set(backends.DEFAULT_MODELS), (
        "node COMBO and backend registry drifted apart"
    )
    assert "TrentH3AutoPromptGenerator" in h3_auto_prompt.NODE_CLASS_MAPPINGS


def test_backend_registry():
    import os

    from TrentNodes.utils.h3_prompt import backends

    # Every node COMBO provider resolves through the registry
    for provider in ("kimi", "glm", "qwen_api"):
        assert provider in backends.DEFAULT_MODELS
        assert provider in backends.PROVIDER_ENV_VARS
        assert provider in backends.OPENAI_COMPAT_PROVIDERS
    assert backends.DEFAULT_MODELS["kimi"] == "kimi-k3"

    # Missing key errors name the provider's env var
    saved = os.environ.pop("MOONSHOT_API_KEY", None)
    try:
        backends.resolve_api_key("", "kimi")
    except RuntimeError as exc:
        assert "MOONSHOT_API_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing key")
    finally:
        if saved is not None:
            os.environ["MOONSHOT_API_KEY"] = saved

    # Compatible backend wires the provider base_url + default model
    backend = backends.get_backend("kimi", "auto", api_key="test-key")
    assert backend.model == "kimi-k3"
    assert str(backend._client.base_url).startswith(
        "https://api.moonshot.ai"
    )
    backend = backends.get_backend("glm", "auto", api_key="test-key")
    assert str(backend._client.base_url).startswith(
        "https://api.z.ai"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All node tests passed.")
