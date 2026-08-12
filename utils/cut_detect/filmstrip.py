"""
Film-strip contact sheet for a detected shot list.

Renders one labelled thumbnail per shot on sprocketed film rows, colour
coded by how each shot is entered, over a proportional timeline ribbon
that shows shot lengths and where the cuts land. Returns a ComfyUI IMAGE
tensor so it can go straight into Preview Image or Save Image.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .detectors import Shot, ShotList
from .formats import format_timecode

# Page furniture, in pixels at the default thumbnail width.
PAGE_PAD = 24
CARD_GAP = 10
LABEL_H = 40
SPROCKET_H = 16
HEADER_H = 74
RIBBON_H = 66
MAX_SHEET_WIDTH = 2400

BG = (18, 18, 20)
FILM_BASE = (32, 32, 36)
SPROCKET = (12, 12, 14)
TEXT = (238, 238, 240)
TEXT_DIM = (150, 150, 158)

# One colour per boundary kind, used on the card's top edge, its label
# text, and its tick on the ribbon.
KIND_COLORS = {
    "start": (120, 200, 130),
    "hard cut": (255, 92, 92),
    "sudden jump": (255, 168, 64),
    "dissolve": (96, 180, 255),
    "fade": (168, 132, 255),
    "wipe": (72, 220, 210),
    "push": (72, 220, 210),
    "slide": (72, 220, 210),
    "zoom": (72, 220, 210),
    "doorway": (72, 220, 210),
}
DEFAULT_COLOR = (200, 200, 208)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))),
        "fonts", "FreeMono.ttf",
    ),
]
_FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
] + _FONT_CANDIDATES


def _load_font(size: int, bold: bool = False):
    for path in (_FONT_BOLD_CANDIDATES if bold else _FONT_CANDIDATES):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _kind_color(kind: str) -> Tuple[int, int, int]:
    return KIND_COLORS.get(kind.split(" over ")[0], DEFAULT_COLOR)


def _frame_to_pil(images: torch.Tensor, index: int, size) -> Image.Image:
    """Pull one frame out of the batch as a resized RGB PIL image."""
    index = int(min(max(index, 0), images.shape[0] - 1))
    frame = images[index, :, :, :3].detach().float().clamp(0.0, 1.0)
    array = (frame * 255.0).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="RGB").resize(size, Image.LANCZOS)


def _draw_sprockets(draw: ImageDraw.ImageDraw, x0, y0, x1, y1):
    """The perforated band that makes a row read as film."""
    draw.rectangle([x0, y0, x1, y1], fill=SPROCKET)
    hole_w, hole_h = 12, max(5, (y1 - y0) - 8)
    pitch = hole_w + 12
    y_mid = (y0 + y1) // 2
    for x in range(x0 + 8, x1 - hole_w, pitch):
        draw.rounded_rectangle(
            [x, y_mid - hole_h // 2, x + hole_w, y_mid + hole_h // 2],
            radius=2, fill=(52, 52, 58),
        )


def _shot_caption(shot: Shot) -> Tuple[str, str]:
    """(headline, detail) for one card."""
    headline = f"Shot {shot.index}   {format_timecode(shot.start_time)}"
    entry = shot.entry
    if shot.transition_frames:
        span = shot.transition_frames
        length = (span[1] - span[0]) / shot.fps
        entry = f"{entry} {length:.2f}s"
    return headline, f"{entry}   ({shot.duration:.2f}s)"


def _draw_ribbon(
    draw: ImageDraw.ImageDraw, shots: ShotList, x0: int, y0: int, width: int
):
    """
    Proportional timeline: every shot a segment as wide as it is long,
    with a coloured tick and a time label at each cut.
    """
    bar_h = 22
    duration = max(shots.duration, 1e-6)
    font = _load_font(13)
    small = _load_font(11)

    draw.rectangle([x0, y0, x0 + width, y0 + bar_h], fill=(40, 40, 46))

    label_slots: List[Tuple[int, int]] = []
    for shot in shots.shots:
        seg_x0 = x0 + int(width * shot.start_time / duration)
        seg_x1 = x0 + int(width * shot.end_time / duration)
        color = _kind_color(shot.entry)
        shade = tuple(int(c * 0.42) for c in color)
        draw.rectangle([seg_x0, y0, max(seg_x1, seg_x0 + 1), y0 + bar_h],
                       fill=shade)

        if shot.index == 1:
            continue
        draw.rectangle([seg_x0 - 1, y0 - 4, seg_x0 + 1, y0 + bar_h + 4],
                       fill=color)

        # Labels are dropped rather than overlapped when cuts crowd.
        text = format_timecode(shot.start_time)
        text_w = int(draw.textlength(text, font=small))
        left = seg_x0 - text_w // 2
        if all(left > prev_right + 6 or left + text_w < prev_left - 6
               for prev_left, prev_right in label_slots):
            draw.text((left, y0 + bar_h + 7), text, font=small, fill=TEXT_DIM)
            label_slots.append((left, left + text_w))

    draw.text((x0, y0 - 20), "0:00", font=font, fill=TEXT_DIM)
    end = f"{format_timecode(shots.duration)}"
    draw.text((x0 + width - int(draw.textlength(end, font=font)), y0 - 20),
              end, font=font, fill=TEXT_DIM)


def render_film_strip(
    images: torch.Tensor,
    shots: ShotList,
    thumb_width: int = 240,
    columns: int = 0,
    show_timeline: bool = True,
    title: Optional[str] = None,
) -> torch.Tensor:
    """
    Render the contact sheet.

    Args:
        images: the (B, H, W, C) [0,1] batch the shots were detected in.
        shots: the detected ShotList.
        thumb_width: width of each shot thumbnail in pixels.
        columns: cards per row; 0 fits as many as MAX_SHEET_WIDTH allows.
        show_timeline: draw the proportional ribbon under the strip.
        title: header line; defaults to a detector + shot-count summary.

    Returns:
        A (1, H, W, 3) float tensor in [0,1] - one ComfyUI IMAGE.
    """
    if images.dim() != 4 or images.shape[0] == 0 or not shots.shots:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    src_h, src_w = int(images.shape[1]), int(images.shape[2])
    thumb_width = max(64, int(thumb_width))
    thumb_h = max(36, round(thumb_width * src_h / max(src_w, 1)))

    card_w = thumb_width
    card_h = thumb_h + LABEL_H

    if columns and columns > 0:
        cols = int(columns)
    else:
        usable = MAX_SHEET_WIDTH - 2 * PAGE_PAD
        cols = max(1, (usable + CARD_GAP) // (card_w + CARD_GAP))
    cols = max(1, min(cols, len(shots.shots)))
    rows = (len(shots.shots) + cols - 1) // cols

    strip_w = cols * card_w + (cols - 1) * CARD_GAP
    row_h = card_h + 2 * SPROCKET_H
    sheet_w = strip_w + 2 * PAGE_PAD
    sheet_h = (
        HEADER_H
        + rows * row_h + max(0, rows - 1) * CARD_GAP
        + (RIBBON_H if show_timeline else 0)
        + PAGE_PAD
    )

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)

    # Header
    heading = title or "Cut Detective"
    draw.text((PAGE_PAD, PAGE_PAD - 6), heading,
              font=_load_font(24, bold=True), fill=TEXT)
    subtitle = (
        f"{len(shots.shots)} shots, {shots.num_cuts} cuts  |  "
        f"{shots.duration:.2f}s at {shots.fps:.2f} fps  |  "
        f"detector: {shots.detector}"
    )
    draw.text((PAGE_PAD, PAGE_PAD + 26), subtitle,
              font=_load_font(14), fill=TEXT_DIM)

    head_font = _load_font(14, bold=True)
    detail_font = _load_font(13)

    y = HEADER_H
    for row in range(rows):
        row_shots = shots.shots[row * cols:(row + 1) * cols]
        _draw_sprockets(draw, PAGE_PAD, y, PAGE_PAD + strip_w, y + SPROCKET_H)
        body_y = y + SPROCKET_H
        draw.rectangle(
            [PAGE_PAD, body_y, PAGE_PAD + strip_w, body_y + card_h],
            fill=FILM_BASE,
        )

        for col, shot in enumerate(row_shots):
            x = PAGE_PAD + col * (card_w + CARD_GAP)
            sheet.paste(
                _frame_to_pil(images, shot.start_frame,
                              (thumb_width, thumb_h)),
                (x, body_y),
            )

            color = _kind_color(shot.entry)
            # Coloured top edge: the cut this shot is entered through.
            draw.rectangle([x, body_y, x + card_w - 1, body_y + 3], fill=color)

            headline, detail = _shot_caption(shot)
            draw.text((x + 6, body_y + thumb_h + 5), headline,
                      font=head_font, fill=TEXT)
            draw.text((x + 6, body_y + thumb_h + 22), detail,
                      font=detail_font, fill=color)

        _draw_sprockets(draw, PAGE_PAD, body_y + card_h,
                        PAGE_PAD + strip_w, body_y + card_h + SPROCKET_H)
        y += row_h + CARD_GAP

    if show_timeline:
        _draw_ribbon(draw, shots, PAGE_PAD, y + 26, strip_w)

    array = np.asarray(sheet, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
