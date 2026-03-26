"""
Video Degradation node for ComfyUI.

Applies configurable, temporally coherent degradation to video
frame batches for generating synthetic training pairs.
"""

from typing import Any, Dict, Tuple

import torch

from ..utils.degradation import apply_degradation_pipeline


class VideoDegradationNode:
    """Apply realistic video degradation for training data."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Batch of video frames (B, H, W, C) "
                        "float32 in [0, 1]"
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2 ** 32 - 1,
                    "tooltip": (
                        "Random seed for reproducible "
                        "degradation"
                    ),
                }),
                "degradation_preset": (
                    [
                        "custom",
                        "mild", "moderate", "severe",
                        "phone_indoor",
                        "social_media_reupload",
                        "zoom_call",
                        "dashcam",
                        "night_handheld",
                        "old_youtube",
                        "old_vhs",
                        "shaky_handheld",
                        "security_cam",
                        "livestream",
                        "old_film",
                        "underwater",
                    ],
                    {
                        "default": "custom",
                        "tooltip": (
                            "Quick presets that override "
                            "individual parameters. "
                            "'custom' uses your settings."
                        ),
                    },
                ),
            },
            "optional": {
                # -- Motion Blur --
                "motion_blur_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle motion blur",
                }),
                "motion_blur_intensity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Motion blur strength "
                        "(maps to kernel size 3-45)"
                    ),
                }),
                "motion_blur_angle_mode": (
                    [
                        "random_consistent",
                        "random_per_frame",
                        "horizontal",
                        "vertical",
                        "diagonal",
                    ],
                    {
                        "default": "random_consistent",
                        "tooltip": (
                            "How blur direction is "
                            "determined across frames"
                        ),
                    },
                ),
                # -- Defocus Blur --
                "defocus_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Toggle defocus/out-of-focus blur"
                    ),
                }),
                "defocus_intensity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Defocus strength "
                        "(maps to sigma 0.5-15.0)"
                    ),
                }),
                "defocus_mode": (
                    [
                        "uniform", "breathing",
                        "rack_focus", "edge_softness",
                    ],
                    {
                        "default": "uniform",
                        "tooltip": (
                            "Temporal behavior of defocus "
                            "blur across frames"
                        ),
                    },
                ),
                # -- Noise --
                "noise_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle noise injection",
                }),
                "noise_intensity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Noise strength",
                }),
                "noise_type": (
                    [
                        "gaussian", "poisson",
                        "film_grain", "sensor", "mixed",
                    ],
                    {
                        "default": "gaussian",
                        "tooltip": "Type of noise to add",
                    },
                ),
                # -- Compression --
                "compression_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Toggle compression artifacts"
                    ),
                }),
                "compression_quality": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": (
                        "Quality level "
                        "(lower = more artifacts)"
                    ),
                }),
                "compression_mode": (
                    ["jpeg", "h264_sim", "blockiness"],
                    {
                        "default": "jpeg",
                        "tooltip": (
                            "Type of compression artifact "
                            "to simulate"
                        ),
                    },
                ),
                # -- Color / Temporal --
                "chromatic_aberration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Color fringing at edges, "
                        "especially toward borders"
                    ),
                }),
                "temporal_flicker": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Per-frame brightness/contrast "
                        "variation"
                    ),
                }),
                "resolution_degradation": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Detail loss via downscale+upscale "
                        "(0=none, 1=4x downscale)"
                    ),
                }),
                "color_degradation": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Desaturation, color shift, "
                        "and banding"
                    ),
                }),
                "interlacing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Interlacing/combing artifacts "
                        "(needs 2+ frames)"
                    ),
                }),
                "rolling_shutter": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Rolling shutter skew simulation"
                    ),
                }),
                "vignette": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Dark corner vignetting",
                }),
                "lens_distortion": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Barrel/pincushion lens distortion"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("degraded_images", "degradation_map",)
    OUTPUT_TOOLTIPS = (
        "Degraded video frames",
        "JSON describing all applied degradations",
    )

    FUNCTION = "degrade"
    CATEGORY = "Trent/Video"
    DESCRIPTION = (
        "Apply configurable, temporally coherent degradation "
        "to video frames for generating synthetic training "
        "pairs. Supports motion blur, defocus, noise, "
        "compression artifacts, chromatic aberration, "
        "and more."
    )

    def degrade(
        self,
        images: torch.Tensor,
        seed: int = 0,
        degradation_preset: str = "custom",
        motion_blur_enabled: bool = False,
        motion_blur_intensity: float = 0.0,
        motion_blur_angle_mode: str = "random_consistent",
        defocus_enabled: bool = False,
        defocus_intensity: float = 0.0,
        defocus_mode: str = "uniform",
        noise_enabled: bool = False,
        noise_intensity: float = 0.0,
        noise_type: str = "gaussian",
        compression_enabled: bool = False,
        compression_quality: int = 100,
        compression_mode: str = "jpeg",
        chromatic_aberration: float = 0.0,
        temporal_flicker: float = 0.0,
        resolution_degradation: float = 0.0,
        color_degradation: float = 0.0,
        interlacing: float = 0.0,
        rolling_shutter: float = 0.0,
        vignette: float = 0.0,
        lens_distortion: float = 0.0,
    ) -> Tuple[torch.Tensor, str]:
        """Apply degradation pipeline to video frames."""
        params = {
            "degradation_preset": degradation_preset,
            "motion_blur_enabled": motion_blur_enabled,
            "motion_blur_intensity": motion_blur_intensity,
            "motion_blur_angle_mode": (
                motion_blur_angle_mode
            ),
            "defocus_enabled": defocus_enabled,
            "defocus_intensity": defocus_intensity,
            "defocus_mode": defocus_mode,
            "noise_enabled": noise_enabled,
            "noise_intensity": noise_intensity,
            "noise_type": noise_type,
            "compression_enabled": compression_enabled,
            "compression_quality": compression_quality,
            "compression_mode": compression_mode,
            "chromatic_aberration": chromatic_aberration,
            "temporal_flicker": temporal_flicker,
            "resolution_degradation": resolution_degradation,
            "color_degradation": color_degradation,
            "interlacing": interlacing,
            "rolling_shutter": rolling_shutter,
            "vignette": vignette,
            "lens_distortion": lens_distortion,
        }

        degraded, deg_map = apply_degradation_pipeline(
            images, params, seed,
        )

        return (degraded, deg_map)


NODE_CLASS_MAPPINGS = {
    "VideoDegradation": VideoDegradationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoDegradation": "Video Degradation (TrentNodes)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
