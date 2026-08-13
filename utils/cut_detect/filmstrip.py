"""
Film-strip contact sheet for a detected shot list.

Renders labelled thumbnails on sprocketed film rows, colour coded by how
each shot is entered, over a proportional timeline ribbon that shows shot
lengths and where the cuts land. One thumbnail per shot by default;
thumbs_per_shot samples further into long shots. Returns a ComfyUI IMAGE
tensor so it can go straight into Preview Image or Save Image.
"""

import os
from dataclasses import dataclass
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
# Ceiling on thumbnails per sheet. thumbs_per_shot degrades against it
# rather than any shot being dropped.
MAX_CARDS = 240

BG = (18, 18, 20)
FILM_BASE = (32, 32, 36)
SPROCKET = (12, 12, 14)
TEXT = (238, 238, 240)
TEXT_DIM = (150, 150, 158)
# Warning amber, for the subtitle note when the detector the user asked
# for never ran.
FALLBACK_COLOR = (255, 168, 64)

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


@dataclass
class _Card:
    """One thumbnail on the sheet."""
    shot: Shot
    frame: int
    lead: bool   # first card of its shot: coloured edge, full caption


def _plan_cards(
    shots: ShotList, thumbs_per_shot: int, max_cards: int = MAX_CARDS
) -> Tuple[List[_Card], int]:
    """
    Choose which frames to show. Returns (cards, thumbs_actually_used).

    The first card of a shot is always its start frame - the frame right
    after the cut is what tells you the cut is real. Later cards are the
    centres of equal slices. A shot with fewer frames than thumbs asked
    for emits one card per frame rather than repeating one.

    When the budget is tight the thumbs per shot degrade; a shot is
    never dropped, because a missing shot misrepresents the detection.
    """
    count = max(1, len(shots.shots))
    per = max(1, min(int(thumbs_per_shot), max_cards // count))

    cards: List[_Card] = []
    for shot in shots.shots:
        span = max(1, shot.frame_count)
        n = max(1, min(per, span))
        cards.append(_Card(shot, shot.start_frame, True))
        for i in range(1, n):
            # Centre of slice i, so the samples spread across the shot.
            offset = int(span * (i + 0.5) / n)
            cards.append(
                _Card(shot, shot.start_frame + min(offset, span - 1), False)
            )
    return cards, per


def _ribbon_segments(
    shots: ShotList,
) -> List[Tuple[float, float, Tuple[int, int, int], bool]]:
    """
    Tile [0, duration] with no gaps: (start, end, colour, is_transition).

    A transition's frames belong to no Shot - the shot starts where the
    effect ends - so drawing shots alone left a hole at every dissolve.
    Each transition is emitted as its own brighter slice, which reads as
    a ramp rather than as missing footage.
    """
    segments = []
    cursor = 0.0
    for shot in shots.shots:
        color = _kind_color(shot.entry)
        if shot.transition_frames:
            span = shot.transition_frames
            t0 = max(cursor, span[0] / shot.fps)
            t1 = min(shot.start_time, span[1] / shot.fps)
            if t1 > cursor:
                if t0 > cursor:
                    segments.append((cursor, t0, _kind_color("start"), False))
                segments.append((max(t0, cursor), t1, color, True))
                cursor = t1
        if shot.start_time > cursor:
            # No labelled transition, but a gap all the same: hand it to
            # the incoming shot so the bar stays continuous.
            segments.append((cursor, shot.start_time, color, True))
        segments.append((shot.start_time, shot.end_time, color, False))
        cursor = max(cursor, shot.end_time)
    if shots.duration > cursor and segments:
        segments.append((cursor, shots.duration, segments[-1][2], False))
    return segments


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

    for start, end, color, is_transition in _ribbon_segments(shots):
        seg_x0 = x0 + int(width * start / duration)
        seg_x1 = x0 + int(width * end / duration)
        # A transition slice reads brighter than the shot it opens, and
        # gets a 2px floor - a 4-frame dissolve is sub-pixel otherwise.
        scale = 0.72 if is_transition else 0.42
        shade = tuple(int(c * scale) for c in color)
        floor = 2 if is_transition else 1
        draw.rectangle([seg_x0, y0, max(seg_x1, seg_x0 + floor), y0 + bar_h],
                       fill=shade)

    label_slots: List[Tuple[int, int]] = []
    for shot in shots.shots:
        seg_x0 = x0 + int(width * shot.start_time / duration)
        color = _kind_color(shot.entry)
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
    thumbs_per_shot: int = 1,
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

    cards, thumbs_used = _plan_cards(shots, thumbs_per_shot)

    # The width cap used to apply only when fitting automatically, so an
    # explicit columns=32 at thumb_width=640 rendered a 20,838px sheet
    # that no preview could show.
    fit_cols = max(
        1, (MAX_SHEET_WIDTH - 2 * PAGE_PAD + CARD_GAP) // (card_w + CARD_GAP)
    )
    cols = min(int(columns), fit_cols) if columns and columns > 0 else fit_cols
    cols = max(1, min(cols, len(cards)))
    rows = (len(cards) + cols - 1) // cols

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
    if thumbs_used < int(thumbs_per_shot):
        # Never silently: the sheet shows fewer frames than was asked.
        subtitle += f"  |  {thumbs_used}/{int(thumbs_per_shot)} thumbs per shot"
    subtitle_font = _load_font(14)
    draw.text((PAGE_PAD, PAGE_PAD + 26), subtitle,
              font=subtitle_font, fill=TEXT_DIM)
    # Someone reading the sheet must not have to open the report to
    # learn that the detector they asked for never ran.
    if shots.fallback:
        draw.text(
            (PAGE_PAD + draw.textlength(subtitle, font=subtitle_font),
             PAGE_PAD + 26),
            f"  (fell back from {shots.requested})",
            font=subtitle_font, fill=FALLBACK_COLOR,
        )

    head_font = _load_font(14, bold=True)
    detail_font = _load_font(13)

    y = HEADER_H
    for row in range(rows):
        row_cards = cards[row * cols:(row + 1) * cols]
        _draw_sprockets(draw, PAGE_PAD, y, PAGE_PAD + strip_w, y + SPROCKET_H)
        body_y = y + SPROCKET_H
        draw.rectangle(
            [PAGE_PAD, body_y, PAGE_PAD + strip_w, body_y + card_h],
            fill=FILM_BASE,
        )

        for col, card in enumerate(row_cards):
            shot = card.shot
            x = PAGE_PAD + col * (card_w + CARD_GAP)
            sheet.paste(
                _frame_to_pil(images, card.frame, (thumb_width, thumb_h)),
                (x, body_y),
            )

            color = _kind_color(shot.entry)
            if card.lead:
                # Coloured top edge: the cut this shot is entered
                # through. Only the lead card gets one, so a boundary
                # stays the only coloured edge even when a shot wraps
                # across rows.
                draw.rectangle([x, body_y, x + card_w - 1, body_y + 3],
                               fill=color)
                headline, detail = _shot_caption(shot)
                draw.text((x + 6, body_y + thumb_h + 5), headline,
                          font=head_font, fill=TEXT)
                draw.text((x + 6, body_y + thumb_h + 22), detail,
                          font=detail_font, fill=color)
            else:
                within = card.frame / max(shot.fps, 1e-6)
                draw.text((x + 6, body_y + thumb_h + 5),
                          format_timecode(within),
                          font=detail_font, fill=TEXT_DIM)

        _draw_sprockets(draw, PAGE_PAD, body_y + card_h,
                        PAGE_PAD + strip_w, body_y + card_h + SPROCKET_H)
        y += row_h + CARD_GAP

    if show_timeline:
        _draw_ribbon(draw, shots, PAGE_PAD, y + 26, strip_w)

    array = np.asarray(sheet, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
