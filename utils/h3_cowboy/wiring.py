"""
The arithmetic MiniMaxH3ReferenceToVideo does to your assets, as plain
functions.

No torch and no ComfyUI, so every rule here is testable offline. Each one
is a deliberate copy of code in comfy_extras/nodes_minimax_h3.py rather
than an import: comfy_extras is not a stable API surface for a custom
node, and a silent upstream change should fail a test here instead of
mis-sizing a real generation. Every function names the lines it copies.

    sampler = comfy_extras/nodes_minimax_h3.py
"""

from typing import Dict, List, Tuple

# sampler: FPS = 24, hardcoded. Reference video frames are READ as 24 fps
# and nothing resamples them.
H3_FPS = 24

# sampler: align_frame_count() / video_latent_t(). Frame counts live on a
# 17k+5 grid.
FRAME_GRID_MODULUS = 17
FRAME_GRID_REMAINDER = 5

# sampler: CANVAS_MULTIPLE, BASE_SHORT_EDGE, MAX_PIXELS.
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


def snap_length(duration_seconds: float, fps: int = H3_FPS) -> Tuple[int, float]:
    """
    (frame count on H3's grid, the duration that really produces).

    The sampler's align_frame_count() rounds UP to the next n where
    n % 17 == 5, so the length you ask for is rarely the length you get:

        2.00s -> 56 frames  -> 2.333s   (+17%)
        5.00s -> 124 frames -> 5.167s
        8.00s -> 192 frames -> 8.000s   (exact)

    That second number is the one a prompt should state. Base mode writes
    it into the instruction line as S.SS, and every mode places its shot
    times inside it, so prompting 2.00 and rendering 2.33 leaves a third
    of a second the model was never told about.
    """
    frames = max(FRAME_GRID_REMAINDER, int(round(float(duration_seconds) * fps)))
    while frames % FRAME_GRID_MODULUS != FRAME_GRID_REMAINDER:
        frames += 1
    return frames, frames / float(fps)


def trim_reference_frames(count: int) -> int:
    """
    The frame count the sampler will actually keep from a reference video.

    It trims DOWNWARDS to the grid (`while n % 17 != 5: n -= 1`), unlike
    the generation length which rounds up. Below 5 frames it raises
    instead, so 0 here means "the sampler will reject this".
    """
    if count < FRAME_GRID_REMAINDER:
        return 0
    frames = int(count)
    while frames % FRAME_GRID_MODULUS != FRAME_GRID_REMAINDER:
        frames -= 1
    return frames


def canvas_for(width: int, height: int) -> Tuple[int, int]:
    """
    The sampler's adapt_canvas(): 768 short edge, 768*1344 area cap,
    each axis rounded to a multiple of 32.
    """
    width, height = max(1, int(width)), max(1, int(height))
    ratio = width / height
    if ratio >= 1.0:
        nominal_w, nominal_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nominal_w, nominal_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nominal_w * nominal_h > MAX_PIXELS:
        scale = (MAX_PIXELS / (nominal_w * nominal_h)) ** 0.5
        nominal_w, nominal_h = nominal_w * scale, nominal_h * scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def build_label_map(
    pictures: List[str], has_video: bool = False, has_audio: bool = False,
) -> str:
    """
    What each tag will refer to, in the sampler's own presentation order.

    The sampler's docstring states that order outright: "images, then
    videos (each soundtrack's <Audio j> label right before its
    <Video k>), then standalone audio. Ordinals are 1-based per type."

    So a soundtrack numbers BEFORE any standalone audio, and both are
    <Audio 1> when only one of them exists. Ordinals count the assets
    that ARRIVE, never the socket they came from - which is why a gap in
    the picture slots shifts every tag after it.
    """
    lines = []
    for index, source in enumerate(pictures, start=1):
        lines.append(f"<Picture {index}> = {source}")
    if has_audio and has_video:
        lines.append("<Audio 1> = audio (as the video's soundtrack)")
    elif has_audio:
        lines.append("<Audio 1> = audio (standalone)")
    if has_video:
        lines.append("<Video 1> = video")
    if not lines:
        return "nothing connected"
    return "\n".join(lines)
