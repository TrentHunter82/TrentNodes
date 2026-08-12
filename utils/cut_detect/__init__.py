"""
Shot-boundary detection, cut serialization, and film-strip rendering.

Backs the Cut Detective node (nodes/cut_detective.py) and feeds the H3
Auto Prompt Generator an authoritative cut list.
"""

from .detectors import (
    DETECTOR_CHOICES,
    Shot,
    ShotList,
    detect_shots,
)
from .filmstrip import render_film_strip
from .formats import (
    format_cut_times,
    format_report,
    format_shot_table,
    parse_cut_times,
    shots_to_json,
)

__all__ = [
    "DETECTOR_CHOICES",
    "Shot",
    "ShotList",
    "detect_shots",
    "render_film_strip",
    "format_cut_times",
    "format_report",
    "format_shot_table",
    "parse_cut_times",
    "shots_to_json",
]
