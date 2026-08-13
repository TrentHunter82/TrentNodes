"""
Multi-Load Cowboy -- six image loaders in one node, with a shared resize.

Loads up to six images from the ComfyUI input folder and resizes them all
to one target size using the same fit modes as KJ's Resize Image V2.
Empty slots are skipped rather than raising, so a half-filled grid is a
valid graph.

The node hands back both a batch (only the filled slots, all the same
size) and the six images one by one, so a slot can feed its own branch.

The canvas UI lives in ../js/multi_load_cowboy.js. Without it the six
slots fall back to plain file combos, which still load and still resize.
"""

import hashlib
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers

# Sentinel for an unfilled slot. It is a real combo option, so an empty
# slot round-trips through the workflow JSON like any other value.
EMPTY = "(empty)"

SLOT_COUNT = 6
UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]
RESIZE_MODES = ["stretch", "resize", "pad", "pad_edge", "crop", "total_pixels"]
CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]
MAX_RESOLUTION = 16384


def slot_names() -> List[str]:
    """Widget names for the six image slots, in grid order."""
    return [f"image_{i}" for i in range(1, SLOT_COUNT + 1)]


def input_image_files() -> List[str]:
    """Sorted list of loadable image files in the input directory."""
    input_dir = folder_paths.get_input_directory()
    try:
        names = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
    except OSError:
        return []
    return sorted(folder_paths.filter_files_content_types(names, ["image"]))


def parse_color(text: str) -> Tuple[float, float, float]:
    """
    Parse a pad colour into three 0-1 floats.

    Accepts "0, 0, 0", "255,128,0" and "#rrggbb". Anything else is black.
    """
    if not isinstance(text, str):
        return (0.0, 0.0, 0.0)

    value = text.strip()
    if value.startswith("#"):
        value = value[1:]
        if len(value) == 3:
            value = "".join(c * 2 for c in value)
        if len(value) == 6:
            try:
                return tuple(
                    int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)
                )
            except ValueError:
                return (0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.0)

    parts = [p for p in value.replace(";", ",").split(",") if p.strip()]
    if len(parts) < 3:
        return (0.0, 0.0, 0.0)

    channels = []
    for part in parts[:3]:
        try:
            number = float(part.strip())
        except ValueError:
            return (0.0, 0.0, 0.0)
        # Treat 0-1 input as normalised, 0-255 as bytes.
        if number > 1.0:
            number = number / 255.0
        channels.append(min(1.0, max(0.0, number)))
    return (channels[0], channels[1], channels[2])


def as_int(value, default: int = 0) -> int:
    """
    Read a size widget that might not hold a number.

    A workflow saved by an older build of the grid can carry a combo
    string where an INT belongs. Falling back to the default keeps such a
    workflow running instead of failing on int("pad").
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def round_to_divisor(value: int, divisor: int) -> int:
    """Round a dimension down to a multiple of divisor, never below it."""
    value = int(value)
    if divisor is None or divisor <= 1:
        return max(1, value)
    value = value - (value % divisor)
    return max(divisor, value)


def plan_size(
    mode: str,
    src_w: int,
    src_h: int,
    width: int,
    height: int,
    divisible_by: int = 1,
) -> Tuple[int, int, int, int]:
    """
    Work out the geometry for one image.

    Returns (scaled_w, scaled_h, canvas_w, canvas_h). The canvas equals
    the scaled size for every mode except the pad modes, where the image
    is scaled to fit inside a larger canvas.

    A width or height of 0 means "take it from the other side", and 0 for
    both means "keep the source size".
    """
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    width = max(0, int(width))
    height = max(0, int(height))

    # crop and total_pixels need a full box to work from.
    if mode in ("crop", "total_pixels") and (width == 0 or height == 0):
        mode = "resize"

    if mode == "stretch":
        out_w = round_to_divisor(width or src_w, divisible_by)
        out_h = round_to_divisor(height or src_h, divisible_by)
        return out_w, out_h, out_w, out_h

    if mode == "crop":
        out_w = round_to_divisor(width, divisible_by)
        out_h = round_to_divisor(height, divisible_by)
        return out_w, out_h, out_w, out_h

    if mode == "total_pixels":
        budget = float(width) * float(height)
        aspect = src_w / src_h
        out_w = round_to_divisor(
            int(round(math.sqrt(budget * aspect))), divisible_by
        )
        out_h = round_to_divisor(
            int(round(math.sqrt(budget / aspect))), divisible_by
        )
        return out_w, out_h, out_w, out_h

    # resize, pad and pad_edge all fit the image inside the box first.
    if width == 0 and height == 0:
        scaled_w, scaled_h = src_w, src_h
    elif width == 0:
        ratio = height / src_h
        scaled_w, scaled_h = int(round(src_w * ratio)), height
    elif height == 0:
        ratio = width / src_w
        scaled_w, scaled_h = width, int(round(src_h * ratio))
    else:
        ratio = min(width / src_w, height / src_h)
        scaled_w = int(round(src_w * ratio))
        scaled_h = int(round(src_h * ratio))

    if mode == "resize":
        out_w = round_to_divisor(scaled_w, divisible_by)
        out_h = round_to_divisor(scaled_h, divisible_by)
        return out_w, out_h, out_w, out_h

    # Pad modes: round the canvas first, then re-fit inside it so the
    # image never overhangs the rounded box.
    canvas_w = round_to_divisor(width or scaled_w, divisible_by)
    canvas_h = round_to_divisor(height or scaled_h, divisible_by)
    ratio = min(canvas_w / src_w, canvas_h / src_h)
    scaled_w = max(1, min(canvas_w, int(round(src_w * ratio))))
    scaled_h = max(1, min(canvas_h, int(round(src_h * ratio))))
    return scaled_w, scaled_h, canvas_w, canvas_h


def crop_box(
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    position: str,
) -> Tuple[int, int, int, int]:
    """
    Largest box of the target aspect that fits in the source.

    Returns (x, y, width, height).
    """
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    target = max(1, int(out_w)) / max(1, int(out_h))

    if src_w / src_h > target:
        box_w = max(1, min(src_w, int(round(src_h * target))))
        box_h = src_h
    else:
        box_w = src_w
        box_h = max(1, min(src_h, int(round(src_w / target))))

    x = (src_w - box_w) // 2
    y = (src_h - box_h) // 2
    if position == "top":
        y = 0
    elif position == "bottom":
        y = src_h - box_h
    elif position == "left":
        x = 0
    elif position == "right":
        x = src_w - box_w
    return x, y, box_w, box_h


def pad_offsets(
    canvas_w: int,
    canvas_h: int,
    inner_w: int,
    inner_h: int,
    position: str,
) -> Tuple[int, int]:
    """Top-left corner for an image placed inside a larger canvas."""
    left = (canvas_w - inner_w) // 2
    top = (canvas_h - inner_h) // 2
    if position == "top":
        top = 0
    elif position == "bottom":
        top = canvas_h - inner_h
    elif position == "left":
        left = 0
    elif position == "right":
        left = canvas_w - inner_w
    return max(0, left), max(0, top)


def place_on_canvas(
    image: torch.Tensor,
    mask: torch.Tensor,
    canvas_w: int,
    canvas_h: int,
    position: str,
    color: Tuple[float, float, float],
    edge: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Put an image on a bigger canvas.

    The mask marks the new area with 1.0, the way KJ's pad node does, so
    the padding can be used straight away as an outpaint mask.
    """
    batch, height, width, channels = image.shape
    if canvas_w <= width and canvas_h <= height:
        return image, mask

    canvas_w = max(canvas_w, width)
    canvas_h = max(canvas_h, height)
    left, top = pad_offsets(canvas_w, canvas_h, width, height, position)

    if edge:
        # Replicate the border pixels outward.
        right = canvas_w - width - left
        bottom = canvas_h - height - top
        padded = F.pad(
            image.movedim(-1, 1),
            (left, right, top, bottom),
            mode="replicate",
        ).movedim(1, -1)
    else:
        padded = image.new_empty((batch, canvas_h, canvas_w, channels))
        for channel in range(channels):
            padded[..., channel] = color[channel] if channel < 3 else 1.0
        padded[:, top:top + height, left:left + width, :] = image

    out_mask = mask.new_ones((mask.shape[0], canvas_h, canvas_w))
    out_mask[:, top:top + height, left:left + width] = mask
    return padded, out_mask


def load_image_file(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read one image file into an (1, H, W, 3) tensor and its alpha mask.

    Only the first frame of an animated file is used: one slot is one
    picture.
    """
    img = node_helpers.pillow(Image.open, path)
    img = node_helpers.pillow(ImageOps.exif_transpose, img)

    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))

    rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    image = torch.from_numpy(rgb)[None, ]

    if "A" in img.getbands():
        alpha = np.array(img.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(alpha)
    else:
        mask = torch.zeros(rgb.shape[:2], dtype=torch.float32)
    return image, mask[None, ]


class MultiLoadCowboy:
    """
    Load up to six images in one node and resize them all the same way.

    Empty slots are skipped, never an error. The batch output carries
    only the filled slots; the numbered outputs carry one slot each.
    """

    CATEGORY = "Trent/Image"
    DISPLAY_NAME = "Multi-Load Cowboy"
    FUNCTION = "load"

    RETURN_TYPES = (
        "IMAGE", "MASK", "INT", "INT", "INT",
        "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE",
    )
    RETURN_NAMES = (
        "images", "masks", "count", "width", "height",
        "image_1", "image_2", "image_3", "image_4", "image_5", "image_6",
    )
    OUTPUT_TOOLTIPS = (
        "Every filled slot as one batch, all at the same size",
        "Alpha of each filled slot; padding reads as 1.0",
        "How many slots are filled",
        "Width of the batch",
        "Height of the batch",
        "Slot 1 on its own (empty slot outputs nothing)",
        "Slot 2 on its own (empty slot outputs nothing)",
        "Slot 3 on its own (empty slot outputs nothing)",
        "Slot 4 on its own (empty slot outputs nothing)",
        "Slot 5 on its own (empty slot outputs nothing)",
        "Slot 6 on its own (empty slot outputs nothing)",
    )

    DESCRIPTION = (
        "Load up to six images in one node and resize them all to the "
        "same target. Drop files on a cell, or click a cell to browse. "
        "Empty slots are skipped instead of raising an error. The batch "
        "output holds only the filled slots."
    )

    @classmethod
    def INPUT_TYPES(cls):
        files = [EMPTY] + input_image_files()
        slots = {
            name: (files, {
                "default": EMPTY,
                "tooltip": f"Image for grid slot {index}. "
                           f"{EMPTY} leaves the slot out.",
            })
            for index, name in enumerate(slot_names(), start=1)
        }
        return {
            "required": {
                **slots,
                "width": ("INT", {
                    "default": 1024, "min": 0, "max": MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Target width. 0 takes it from the height.",
                }),
                "height": ("INT", {
                    "default": 1024, "min": 0, "max": MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Target height. 0 takes it from the width.",
                }),
                "resize_mode": (RESIZE_MODES, {
                    "default": "pad",
                    "tooltip": "stretch ignores the aspect ratio. resize "
                               "fits inside the box. pad and pad_edge fit "
                               "then fill the rest. crop fills the box and "
                               "cuts the overflow. total_pixels keeps the "
                               "aspect and matches width x height in area.",
                }),
                "upscale_method": (UPSCALE_METHODS, {
                    "default": "lanczos",
                    "tooltip": "Sampler used for the resize.",
                }),
                "crop_position": (CROP_POSITIONS, {
                    "default": "center",
                    "tooltip": "Which part to keep when cropping, and "
                               "where to sit the image when padding.",
                }),
                "pad_color": ("STRING", {
                    "default": "0, 0, 0",
                    "tooltip": "Pad colour as R, G, B or #rrggbb.",
                }),
                "divisible_by": ("INT", {
                    "default": 8, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Round the output down to a multiple of "
                               "this. Use 8 or 16 for most models.",
                }),
            },
            "optional": {
                "device": (["cpu", "gpu"], {
                    "default": "cpu",
                    "tooltip": "Where to run the resize. lanczos is CPU "
                               "only and falls back on its own.",
                }),
            },
        }

    # --- validation -----------------------------------------------------

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """
        Accept empty slots, and accept files uploaded after page load.

        Naming the slots here turns off the stock combo check, which
        would otherwise reject a file that arrived after the node
        definition was sent to the browser.
        """
        missing = []
        for name in slot_names():
            value = kwargs.get(name, EMPTY)
            if value in (EMPTY, None, ""):
                continue
            if not folder_paths.exists_annotated_filepath(value):
                missing.append(f"{name}: {value}")
        if missing:
            return "Image file not found -> " + "; ".join(missing)
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Re-run when any slot points at a new or edited file."""
        digest = hashlib.sha256()
        for name in slot_names():
            value = kwargs.get(name, EMPTY)
            digest.update(str(value).encode("utf-8"))
            if value in (EMPTY, None, ""):
                continue
            try:
                path = folder_paths.get_annotated_filepath(value)
                stat = os.stat(path)
                digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
            except OSError:
                digest.update(b"missing")
        return digest.hexdigest()

    # --- resize ---------------------------------------------------------

    def resize_one(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        width: int,
        height: int,
        resize_mode: str,
        upscale_method: str,
        crop_position: str,
        color: Tuple[float, float, float],
        divisible_by: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize one slot's image and mask to the planned geometry."""
        src_h, src_w = image.shape[1], image.shape[2]
        scaled_w, scaled_h, canvas_w, canvas_h = plan_size(
            resize_mode, src_w, src_h, width, height, divisible_by
        )

        image = image.to(device)
        mask = mask.to(device)

        if resize_mode == "crop":
            x, y, box_w, box_h = crop_box(
                src_w, src_h, scaled_w, scaled_h, crop_position
            )
            image = image[:, y:y + box_h, x:x + box_w, :]
            mask = mask[:, y:y + box_h, x:x + box_w]

        if image.shape[1] != scaled_h or image.shape[2] != scaled_w:
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1), scaled_w, scaled_h,
                upscale_method, crop="disabled",
            ).movedim(1, -1)
            # lanczos on a single-channel tensor is unsupported, so masks
            # ride along on bilinear.
            mask_method = (
                "bilinear" if upscale_method == "lanczos" else upscale_method
            )
            mask = comfy.utils.common_upscale(
                mask.unsqueeze(1), scaled_w, scaled_h,
                mask_method, crop="disabled",
            ).squeeze(1)

        if canvas_w > scaled_w or canvas_h > scaled_h:
            image, mask = place_on_canvas(
                image, mask, canvas_w, canvas_h, crop_position, color,
                edge=(resize_mode == "pad_edge"),
            )

        return image.clamp(0.0, 1.0).cpu(), mask.clamp(0.0, 1.0).cpu()

    def build_batch(
        self,
        images: List[torch.Tensor],
        masks: List[torch.Tensor],
        color: Tuple[float, float, float],
        crop_position: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack the filled slots into one batch.

        The fit modes that keep the aspect ratio can leave slots at
        different sizes. Rather than fail, the odd ones out are padded up
        to the largest size in the set.
        """
        heights = {img.shape[1] for img in images}
        widths = {img.shape[2] for img in images}

        if len(heights) > 1 or len(widths) > 1:
            target_h, target_w = max(heights), max(widths)
            print(
                f"[MultiLoadCowboy] Slots differ in size; padding "
                f"{len(images)} images up to {target_w}x{target_h} for the "
                f"batch output. The numbered outputs keep their own size."
            )
            padded_images, padded_masks = [], []
            for img, msk in zip(images, masks):
                img, msk = place_on_canvas(
                    img, msk, target_w, target_h, crop_position, color
                )
                padded_images.append(img)
                padded_masks.append(msk)
            images, masks = padded_images, padded_masks

        return torch.cat(images, dim=0), torch.cat(masks, dim=0)

    # --- main -----------------------------------------------------------

    def load(
        self,
        width: int,
        height: int,
        resize_mode: str,
        upscale_method: str,
        crop_position: str,
        pad_color: str,
        divisible_by: int,
        device: str = "cpu",
        **kwargs,
    ):
        width = as_int(width, 1024)
        height = as_int(height, 1024)
        divisible_by = as_int(divisible_by, 8)
        if resize_mode not in RESIZE_MODES:
            resize_mode = "pad"
        if upscale_method not in UPSCALE_METHODS:
            upscale_method = "lanczos"
        if crop_position not in CROP_POSITIONS:
            crop_position = "center"
        color = parse_color(pad_color)

        if device == "gpu" and upscale_method != "lanczos":
            work_device = comfy.model_management.get_torch_device()
        else:
            work_device = torch.device("cpu")

        slots: List[Optional[torch.Tensor]] = []
        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for name in slot_names():
            value = kwargs.get(name, EMPTY)
            if value in (EMPTY, None, ""):
                slots.append(None)
                continue

            path = folder_paths.get_annotated_filepath(value)
            if not os.path.isfile(path):
                # Already reported by VALIDATE_INPUTS; skip rather than
                # take the whole graph down over one stale slot.
                print(f"[MultiLoadCowboy] {name}: file is gone, skipping "
                      f"'{value}'")
                slots.append(None)
                continue

            image, mask = load_image_file(path)
            image, mask = self.resize_one(
                image, mask, width, height, resize_mode, upscale_method,
                crop_position, color, divisible_by, work_device,
            )
            slots.append(image)
            images.append(image)
            masks.append(mask)

        count = len(images)
        if count == 0:
            # Nothing loaded is a state, not a crash. Hand back one black
            # frame at the requested size so the graph keeps running.
            blank_w = round_to_divisor(width or 64, divisible_by)
            blank_h = round_to_divisor(height or 64, divisible_by)
            print("[MultiLoadCowboy] Every slot is empty; returning one "
                  f"black {blank_w}x{blank_h} frame and count 0.")
            batch = torch.zeros((1, blank_h, blank_w, 3), dtype=torch.float32)
            batch_mask = torch.ones(
                (1, blank_h, blank_w), dtype=torch.float32
            )
        else:
            batch, batch_mask = self.build_batch(
                images, masks, color, crop_position
            )

        return (
            batch, batch_mask, count, batch.shape[2], batch.shape[1],
            *slots,
        )


NODE_CLASS_MAPPINGS = {
    "MultiLoadCowboy": MultiLoadCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiLoadCowboy": "Multi-Load Cowboy",
}
