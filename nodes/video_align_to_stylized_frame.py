"""
VideoAlignToStylizedFrame - TrentNodes

Automatically aligns a video batch on X and Y axes by computing the
translation offset between the video's first frame and a stylized reference
image, then applying that offset uniformly to all frames.

Uses edge-based phase correlation (torch.fft) for robustness against
color/texture differences introduced by style transfer.
"""

import torch
import torch.nn.functional as F

import comfy.model_management as mm

from ..utils.image_ops import extract_edges, to_grayscale


def _phase_correlation(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
) -> tuple[int, int]:
    """
    Compute translation offset from img_a to img_b via phase correlation.

    Args:
        img_a: Tensor (H, W) float32, reference (stylized frame)
        img_b: Tensor (H, W) float32, source (first video frame)

    Returns:
        (offset_x, offset_y) in pixels. Applying this offset to img_b
        aligns it to img_a.
    """
    H, W = img_a.shape

    fa = torch.fft.rfft2(img_a)
    fb = torch.fft.rfft2(img_b)

    cross = fa * fb.conj()
    eps = 1e-8
    cross = cross / (cross.abs() + eps)

    correlation = torch.fft.irfft2(cross, s=(H, W))

    peak_flat = correlation.argmax()
    peak_y = (peak_flat // W).item()
    peak_x = (peak_flat % W).item()

    # Wrap negative offsets (phase correlation folds at half-size)
    if peak_y > H // 2:
        peak_y -= H
    if peak_x > W // 2:
        peak_x -= W

    return int(peak_x), int(peak_y)


def _apply_translation(
    frames: torch.Tensor,
    tx: float,
    ty: float,
    border_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Apply X/Y translation to a batch of frames.

    Args:
        frames: Tensor (B, H, W, C)
        tx: Pixel shift in X
        ty: Pixel shift in Y
        border_mode: "zeros", "replicate", or "reflect"
        device: torch device

    Returns:
        Translated frames (B, H, W, C)
    """
    B, H, W, C = frames.shape

    theta = torch.zeros(B, 2, 3, device=device, dtype=frames.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    # Normalised translation: pixel_shift / half_dim
    theta[:, 0, 2] = -tx / (W / 2.0)
    theta[:, 1, 2] = -ty / (H / 2.0)

    frames_bchw = frames.permute(0, 3, 1, 2)
    grid = F.affine_grid(theta, frames_bchw.shape, align_corners=False)

    padding_mode = {
        "zeros": "zeros",
        "replicate": "border",
        "reflect": "reflection",
    }.get(border_mode, "zeros")

    aligned = F.grid_sample(
        frames_bchw, grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    return aligned.permute(0, 2, 3, 1)


class VideoAlignToStylizedFrame:
    """
    Aligns a video batch to a stylized reference frame using X/Y translation.

    Computes the pixel offset between the video's first frame and the
    reference, then shifts all frames by that amount. Edge-based alignment
    is recommended when the reference has been style-transferred.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": (
                        "Batch of video frames (B, H, W, C). "
                        "The first frame is used for alignment detection."
                    ),
                }),
                "reference_frame": ("IMAGE", {
                    "tooltip": (
                        "Single stylized reference frame to align to. "
                        "Must be the same resolution as video_frames."
                    ),
                }),
                "alignment_method": (
                    ["edges", "pixels"],
                    {
                        "default": "edges",
                        "tooltip": (
                            "edges: align on Sobel edges (robust for "
                            "style-transferred images). "
                            "pixels: align on raw grayscale intensity."
                        ),
                    },
                ),
                "border_mode": (
                    ["zeros", "replicate", "reflect"],
                    {
                        "default": "zeros",
                        "tooltip": (
                            "How to fill pixels revealed after shifting. "
                            "zeros=black, replicate=edge pixels, "
                            "reflect=mirror."
                        ),
                    },
                ),
                "max_offset": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": (
                            "Maximum allowed offset in pixels. "
                            "Detected offsets larger than this are clamped "
                            "to zero (guards against false correlations)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("aligned_frames", "offset_x", "offset_y")
    FUNCTION = "align"
    CATEGORY = "Trent/Video"

    def align(
        self,
        video_frames: torch.Tensor,
        reference_frame: torch.Tensor,
        alignment_method: str,
        border_mode: str,
        max_offset: int,
    ) -> tuple:
        device = mm.get_torch_device()

        video_frames = video_frames.to(device)
        reference_frame = reference_frame.to(device)

        # Use first video frame for alignment
        first_frame = video_frames[0:1]   # (1, H, W, C)

        # Ensure reference is a single frame
        ref = reference_frame[0:1]        # (1, H, W, C)

        # Resize ref to match video resolution if needed
        _, H, W, _ = first_frame.shape
        _, rH, rW, _ = ref.shape
        if rH != H or rW != W:
            ref_bchw = ref.permute(0, 3, 1, 2)
            ref_bchw = F.interpolate(
                ref_bchw, size=(H, W), mode="bilinear",
                align_corners=False,
            )
            ref = ref_bchw.permute(0, 2, 3, 1)

        if alignment_method == "edges":
            # (1, H, W) edge maps
            src_map = extract_edges(first_frame, device)[0]
            ref_map = extract_edges(ref, device)[0]
        else:
            src_map = to_grayscale(first_frame)[0]
            ref_map = to_grayscale(ref)[0]

        tx, ty = _phase_correlation(ref_map, src_map)

        # Clamp implausibly large offsets
        if abs(tx) > max_offset or abs(ty) > max_offset:
            tx, ty = 0, 0

        if tx == 0 and ty == 0:
            return (video_frames.cpu(), 0, 0)

        aligned = _apply_translation(
            video_frames, float(tx), float(ty), border_mode, device,
        )

        return (aligned.cpu(), tx, ty)


NODE_CLASS_MAPPINGS = {
    "VideoAlignToStylizedFrame": VideoAlignToStylizedFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAlignToStylizedFrame": "Video Align To Stylized Frame",
}
