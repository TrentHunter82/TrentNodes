"""
Batch Slowdown - GPU-accelerated frame duplication for video/batch slowdown.

Supports multiple input modes:
- Direct multiplier (2x, 3x, 1.5x, etc.)
- Target frame count
- FPS conversion (24fps -> 60fps)

Works with IMAGE, MASK, and LATENT batches.
"""

import torch
from typing import Dict, Any, Tuple, Optional


class BatchSlowdown:
    """
    Slow down video/image batches by intelligent frame duplication.

    Supports both integer and decimal multipliers with configurable
    distribution strategies for non-integer slowdowns.

    Modes:
    - multiplier: Direct slowdown factor (2.0 = 2x slower)
    - target_frames: Specify exact output frame count
    - fps_convert: Convert between frame rates (24fps -> 60fps)

    Distribution (for non-integer multipliers):
    - spread: Evenly distribute extra frames across the sequence
    - front_weighted: Extra frames concentrated at start
    - back_weighted: Extra frames concentrated at end
    """

    MODES = ["multiplier", "target_frames", "fps_convert"]
    DISTRIBUTIONS = ["spread", "front_weighted", "back_weighted"]
    DATA_TYPES = ["IMAGE", "MASK", "LATENT"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "data_type": (cls.DATA_TYPES, {
                    "default": "IMAGE",
                    "tooltip": "Type of data to process"
                }),
                "mode": (cls.MODES, {
                    "default": "multiplier",
                    "tooltip": "How to specify the slowdown amount"
                }),
                "distribution": (cls.DISTRIBUTIONS, {
                    "default": "spread",
                    "tooltip": "How to distribute extra frames for "
                               "decimal multipliers"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Image batch to slow down"
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask batch to slow down"
                }),
                "latent": ("LATENT", {
                    "tooltip": "Latent batch to slow down"
                }),
                "multiplier": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Slowdown multiplier (2.0 = 2x slower, "
                               "0.5 = 2x faster)"
                }),
                "target_frames": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Target frame count for output"
                }),
                "source_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 0.1,
                    "tooltip": "Source video FPS (for fps_convert mode)"
                }),
                "target_fps": ("FLOAT", {
                    "default": 60.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 0.1,
                    "tooltip": "Target video FPS (for fps_convert mode)"
                }),
                "enable_speedup": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow speedup (sample every Nth frame) "
                               "when multiplier < 1"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "images",
        "mask",
        "latent",
        "original_count",
        "new_count",
        "effective_mult",
        "info"
    )

    FUNCTION = "slowdown"
    CATEGORY = "Trent/Video"
    DESCRIPTION = """Slow down or speed up batches by frame duplication/sampling.

Modes:
- multiplier: Direct slowdown factor (2.0 = 2x slower)
- target_frames: Specify exact output frame count
- fps_convert: Convert between frame rates (24fps -> 60fps)

Distribution (for non-integer multipliers):
- spread: Evenly distribute extra frames
- front_weighted: Extra frames at start
- back_weighted: Extra frames at end"""

    def compute_multiplier(
        self,
        mode: str,
        original_count: int,
        multiplier: float,
        target_frames: int,
        source_fps: float,
        target_fps: float
    ) -> float:
        """Calculate effective multiplier based on mode."""
        if mode == "multiplier":
            return multiplier
        elif mode == "target_frames":
            if original_count == 0:
                return 1.0
            return target_frames / original_count
        elif mode == "fps_convert":
            if source_fps == 0:
                return 1.0
            return target_fps / source_fps
        return multiplier

    def compute_repeat_counts(
        self,
        original_count: int,
        multiplier: float,
        distribution: str,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute per-frame repeat counts for slowdown.

        For 4 frames at 1.5x = 6 total frames:
        - spread:         [2, 1, 2, 1] (balanced across sequence)
        - front_weighted: [2, 2, 1, 1] (extras at start)
        - back_weighted:  [1, 1, 2, 2] (extras at end)

        Args:
            original_count: Number of input frames
            multiplier: Slowdown factor (1.5 = 50% more frames)
            distribution: How to distribute the extra frames
            device: Torch device for tensor creation

        Returns:
            Tensor of shape (original_count,) with repeat count per frame
        """
        target_count = max(1, int(round(original_count * multiplier)))
        base_repeats = target_count // original_count
        extra_frames = target_count % original_count

        # Start with base repeats for all frames
        repeats = torch.full(
            (original_count,),
            base_repeats,
            dtype=torch.long,
            device=device
        )

        if extra_frames == 0:
            return repeats

        # Distribute extra frames based on strategy
        if distribution == "spread":
            # Evenly spread extra duplications across frames
            indices = torch.linspace(
                0, original_count - 1, extra_frames,
                device=device
            ).long()
            repeats[indices] += 1

        elif distribution == "front_weighted":
            # Put extra frames at the beginning
            repeats[:extra_frames] += 1

        elif distribution == "back_weighted":
            # Put extra frames at the end
            repeats[-extra_frames:] += 1

        return repeats

    def compute_skip_indices(
        self,
        original_count: int,
        multiplier: float,
        distribution: str,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute which frames to keep for speedup (multiplier < 1).

        Args:
            original_count: Number of input frames
            multiplier: Speedup factor (0.5 = keep half the frames)
            distribution: How to select frames
            device: Torch device for tensor creation

        Returns:
            Tensor of indices to keep
        """
        target_count = max(1, int(round(original_count * multiplier)))

        if distribution == "spread":
            # Evenly sample frames across the sequence
            indices = torch.linspace(
                0, original_count - 1, target_count,
                device=device
            ).long()
        elif distribution == "front_weighted":
            # Keep frames from the start
            indices = torch.arange(target_count, device=device)
        elif distribution == "back_weighted":
            # Keep frames from the end
            start = original_count - target_count
            indices = torch.arange(start, original_count, device=device)
        else:
            indices = torch.linspace(
                0, original_count - 1, target_count,
                device=device
            ).long()

        return indices

    def process_latent(
        self,
        latent: dict,
        repeats: torch.Tensor
    ) -> dict:
        """
        Process latent dict, expanding samples and optional fields.

        Args:
            latent: Latent dictionary with 'samples' key
            repeats: Per-frame repeat counts

        Returns:
            New latent dict with expanded tensors
        """
        result = {}

        # Expand the samples tensor
        samples = latent["samples"]
        result["samples"] = torch.repeat_interleave(samples, repeats, dim=0)

        # Expand noise_mask if present
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
            result["noise_mask"] = torch.repeat_interleave(mask, repeats, dim=0)

        # Expand batch_index list if present
        if "batch_index" in latent:
            old_indices = latent["batch_index"]
            new_indices = []
            for i, count in enumerate(repeats.tolist()):
                if i < len(old_indices):
                    new_indices.extend([old_indices[i]] * count)
            result["batch_index"] = new_indices

        # Copy any other keys as-is
        for key in latent:
            if key not in result:
                result[key] = latent[key]

        return result

    def process_latent_speedup(
        self,
        latent: dict,
        indices: torch.Tensor
    ) -> dict:
        """
        Process latent dict for speedup, selecting specific frames.

        Args:
            latent: Latent dictionary with 'samples' key
            indices: Frame indices to keep

        Returns:
            New latent dict with selected frames
        """
        result = {}

        # Select frames from samples
        samples = latent["samples"]
        result["samples"] = samples[indices]

        # Select from noise_mask if present
        if "noise_mask" in latent:
            mask = latent["noise_mask"]
            result["noise_mask"] = mask[indices]

        # Select from batch_index list if present
        if "batch_index" in latent:
            old_indices = latent["batch_index"]
            idx_list = indices.tolist()
            result["batch_index"] = [
                old_indices[i] for i in idx_list if i < len(old_indices)
            ]

        # Copy any other keys as-is
        for key in latent:
            if key not in result:
                result[key] = latent[key]

        return result

    def generate_info(
        self,
        data_type: str,
        mode: str,
        distribution: str,
        original_count: int,
        new_count: int,
        effective_mult: float,
        is_speedup: bool,
        repeats_or_indices: torch.Tensor
    ) -> str:
        """Generate human-readable info string."""
        lines = [
            "Batch Slowdown Report",
            "=" * 21,
            f"Data type: {data_type}",
            f"Mode: {mode}",
            f"Original frames: {original_count}",
            f"New frames: {new_count}",
            f"Effective multiplier: {effective_mult:.2f}x",
            f"Distribution: {distribution}",
        ]

        if is_speedup:
            lines.append(f"Operation: SPEEDUP (sampling)")
            preview = repeats_or_indices[:10].tolist()
            lines.append(f"Kept indices: {preview}{'...' if len(repeats_or_indices) > 10 else ''}")
        else:
            lines.append(f"Operation: SLOWDOWN (duplication)")
            preview = repeats_or_indices[:10].tolist()
            lines.append(f"Repeat pattern: {preview}{'...' if len(repeats_or_indices) > 10 else ''}")

        return "\n".join(lines)

    def slowdown(
        self,
        data_type: str,
        mode: str,
        distribution: str,
        images: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        latent: Optional[dict] = None,
        multiplier: float = 2.0,
        target_frames: int = 60,
        source_fps: float = 24.0,
        target_fps: float = 60.0,
        enable_speedup: bool = False,
        **kwargs
    ) -> Tuple:
        """
        Main processing function.

        Args:
            data_type: Which input type to process
            mode: How to calculate the multiplier
            distribution: How to distribute frames for decimal multipliers
            images: Optional image batch
            mask: Optional mask batch
            latent: Optional latent dict
            multiplier: Direct multiplier value
            target_frames: Target frame count for target_frames mode
            source_fps: Source FPS for fps_convert mode
            target_fps: Target FPS for fps_convert mode
            enable_speedup: Allow speedup when multiplier < 1

        Returns:
            Tuple of (images, mask, latent, orig_count, new_count, eff_mult, info)
        """
        # Get input tensor and determine original count
        if data_type == "IMAGE":
            if images is None:
                raise ValueError("IMAGE type selected but no images provided")
            tensor = images
            original_count = images.shape[0]
            device = images.device
        elif data_type == "MASK":
            if mask is None:
                raise ValueError("MASK type selected but no mask provided")
            tensor = mask
            original_count = mask.shape[0]
            device = mask.device
        elif data_type == "LATENT":
            if latent is None:
                raise ValueError("LATENT type selected but no latent provided")
            tensor = latent["samples"]
            original_count = tensor.shape[0]
            device = tensor.device
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Handle edge case of empty batch
        if original_count == 0:
            info = "Empty batch - nothing to process"
            return (images, mask, latent, 0, 0, 1.0, info)

        # Compute effective multiplier
        eff_mult = self.compute_multiplier(
            mode, original_count, multiplier,
            target_frames, source_fps, target_fps
        )

        # Initialize outputs as passthroughs
        out_images = images
        out_mask = mask
        out_latent = latent

        # Handle speedup vs slowdown
        if eff_mult < 1.0:
            if not enable_speedup:
                # Return unchanged with warning
                info = (
                    f"Multiplier {eff_mult:.2f}x would speed up video, "
                    f"but enable_speedup is False.\n"
                    f"Returning {original_count} frames unchanged."
                )
                return (
                    images, mask, latent,
                    original_count, original_count, 1.0, info
                )

            # Speedup: sample frames
            indices = self.compute_skip_indices(
                original_count, eff_mult, distribution, device
            )
            new_count = len(indices)

            if data_type == "IMAGE":
                out_images = images[indices]
            elif data_type == "MASK":
                out_mask = mask[indices]
            elif data_type == "LATENT":
                out_latent = self.process_latent_speedup(latent, indices)

            info = self.generate_info(
                data_type, mode, distribution,
                original_count, new_count, eff_mult,
                is_speedup=True, repeats_or_indices=indices
            )

        else:
            # Slowdown: duplicate frames
            repeats = self.compute_repeat_counts(
                original_count, eff_mult, distribution, device
            )
            new_count = repeats.sum().item()

            if data_type == "IMAGE":
                out_images = torch.repeat_interleave(images, repeats, dim=0)
            elif data_type == "MASK":
                out_mask = torch.repeat_interleave(mask, repeats, dim=0)
            elif data_type == "LATENT":
                out_latent = self.process_latent(latent, repeats)

            info = self.generate_info(
                data_type, mode, distribution,
                original_count, new_count, eff_mult,
                is_speedup=False, repeats_or_indices=repeats
            )

        # Calculate actual effective multiplier achieved
        actual_mult = new_count / original_count if original_count > 0 else 1.0

        return (
            out_images,
            out_mask,
            out_latent,
            original_count,
            new_count,
            actual_mult,
            info
        )


NODE_CLASS_MAPPINGS = {
    "BatchSlowdown": BatchSlowdown,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchSlowdown": "Batch Slowdown",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
