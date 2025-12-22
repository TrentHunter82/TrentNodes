"""
VidScribe MiniCPM Beta - Vision-language node for ComfyUI.

GPU-accelerated video/image description using MiniCPM-V 4.5 with:
- int4 quantization (~6-8GB VRAM)
- Smart frame sampling
- Auto-unload after idle
"""

import torch

from ..utils.minicpm_wrapper import (
    is_minicpm_available,
    load_minicpm_model,
    run_inference,
    sample_frames,
    tensors_to_pil,
    clear_minicpm_cache,
    SYSTEM_PROMPTS,
    SYSTEM_PROMPT_CHOICES,
)


class VidScribeMiniCPMBeta:
    """
    Vision-language model for describing images and video frames.

    Uses MiniCPM-V 4.5 with int4 quantization for efficient inference.
    Supports single images, multi-image comparison, and video frame
    sequences with temporal understanding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Describe what you see.",
                        "tooltip": "Question or instruction for the model"
                    }
                ),
                "mode": (
                    ["single_image", "multi_image", "video_frames"],
                    {
                        "default": "video_frames",
                        "tooltip": (
                            "single_image: analyze first image only. "
                            "multi_image: compare multiple images. "
                            "video_frames: understand as video sequence"
                        )
                    }
                ),
            },
            "optional": {
                "system_prompt": (
                    SYSTEM_PROMPT_CHOICES,
                    {
                        "default": "default",
                        "tooltip": (
                            "Preset personalities: default (balanced), "
                            "detailed (thorough), concise (brief), "
                            "narrator (cinematic), technical (analytical), "
                            "accessible (audio description), creative (artistic), "
                            "none (no system prompt), custom (your own)"
                        )
                    }
                ),
                "thinking_mode": (
                    ["fast", "deep_thinking"],
                    {
                        "default": "fast",
                        "tooltip": (
                            "fast: quick responses. "
                            "deep_thinking: slower but more thorough analysis"
                        )
                    }
                ),
                "use_all_frames": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Process every frame instead of smart sampling. "
                            "WARNING: Uses significantly more VRAM and time"
                        )
                    }
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 4096,
                        "step": 64,
                        "tooltip": "Maximum length of generated response"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Creativity level. "
                            "0.0 = deterministic, 1.0 = creative"
                        )
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffff,
                        "tooltip": "Random seed (0 = random)"
                    }
                ),
                "custom_system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Your custom system prompt. "
                            "Only used when system_prompt is set to 'custom'"
                        )
                    }
                ),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("response", "images")
    FUNCTION = "describe"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "Vision-language model for describing images and video frames. "
        "Uses MiniCPM-V 4.5 with int4 quantization (~6-8GB VRAM). "
        "Auto-downloads model on first use."
    )

    def describe(
        self,
        images: torch.Tensor,
        prompt: str,
        mode: str,
        system_prompt: str = "default",
        thinking_mode: str = "fast",
        use_all_frames: bool = False,
        max_tokens: int = 512,
        temperature: float = 0.7,
        seed: int = 0,
        custom_system_prompt: str = ""
    ):
        """
        Run vision-language inference on images.

        Args:
            images: (B, H, W, C) tensor of images
            prompt: Text prompt/question
            mode: Processing mode
            system_prompt: Preset system prompt key
            thinking_mode: Reasoning depth
            use_all_frames: Skip smart sampling
            max_tokens: Max response length
            temperature: Sampling temperature
            seed: Random seed
            custom_system_prompt: Custom system prompt (when system_prompt="custom")

        Returns:
            Tuple of (response_text, passthrough_images)
        """
        # Check dependencies
        if not is_minicpm_available():
            return (
                "[Error] MiniCPM dependencies not installed. "
                "Run: pip install transformers accelerate bitsandbytes",
                images
            )

        # Resolve system prompt
        if system_prompt == "custom":
            resolved_system_prompt = custom_system_prompt
        else:
            resolved_system_prompt = SYSTEM_PROMPTS.get(system_prompt, "")

        # Get frame count for logging
        n_frames = images.shape[0]
        print(f"[TrentNodes] VidScribe: {n_frames} frames, mode={mode}")

        # Smart frame sampling (unless disabled)
        sampled_images, indices = sample_frames(images, use_all_frames)
        n_sampled = sampled_images.shape[0]

        if n_sampled != n_frames:
            print(
                f"[TrentNodes] Sampled {n_sampled}/{n_frames} frames "
                f"(stride={n_frames // n_sampled})"
            )

        # Convert tensors to PIL images
        pil_images = tensors_to_pil(sampled_images)

        # Run inference
        response = run_inference(
            images=pil_images,
            prompt=prompt,
            mode=mode,
            system_prompt=resolved_system_prompt,
            thinking_mode=thinking_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )

        print(f"[TrentNodes] VidScribe response: {len(response)} chars")

        # Return response and passthrough original images
        return (response, images)


class UnloadMiniCPM:
    """
    Manually unload MiniCPM model to free VRAM.

    Use this node if you need to immediately free GPU memory
    without waiting for the auto-unload timeout.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger": ("*", {"tooltip": "Connect any output to trigger unload"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Trent/VLM"
    DESCRIPTION = (
        "Manually unload MiniCPM model to free VRAM. "
        "Model auto-unloads after 60s idle, but use this for immediate cleanup."
    )

    def unload(self, trigger=None):
        """Unload the MiniCPM model."""
        clear_minicpm_cache()
        return ("MiniCPM model unloaded",)


# Node registration
NODE_CLASS_MAPPINGS = {
    "VidScribeMiniCPMBeta": VidScribeMiniCPMBeta,
    "UnloadMiniCPM": UnloadMiniCPM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VidScribeMiniCPMBeta": "VidScribe MiniCPM Beta",
    "UnloadMiniCPM": "Unload MiniCPM",
}
