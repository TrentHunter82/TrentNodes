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

    def __init__(self):
        self.calls = 0

    def generate(self, system, images, user_text, max_tokens=4096, seed=0):
        self.calls += 1
        assert "UNBREAKABLE RULES" in system
        assert images[0].label.startswith("Reference image <Picture 1>")
        assert "TASK CONTEXT" in user_text
        return VLMResult(text=CANNED, usage={"model": "fake-model"})


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
                     "qwen_local", "minicpm_local", "ollama"):
        assert expected in providers, expected
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
