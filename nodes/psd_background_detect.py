"""
PSD Background Detect

Scores layers from a PSDLayerSplitter manifest and
recommends a contiguous index range that's likely the
background. Outputs feed directly into PSDLayerCompositor's
replacement_index / replacement_end_index inputs.
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple


def _score_layer(
    lyr: Dict[str, Any],
    name_re: re.Pattern,
    canvas_w: int,
    canvas_h: int,
    bottom_bonus: float,
) -> Tuple[float, List[str]]:
    """Return (score, reasons) for one layer."""
    score = 0.0
    reasons: List[str] = []
    name = lyr.get("original_name", "") or ""
    kind = lyr.get("kind", "")

    if name_re.search(name):
        score += 3.0
        reasons.append(f"name~/{name_re.pattern}/")

    # Prefer pre-enriched flags; fall back to recompute
    # from raw geometry if manifest is older (v1).
    covers = lyr.get("covers_canvas")
    if covers is None:
        w = int(lyr.get("size", {}).get("width", 0))
        h = int(lyr.get("size", {}).get("height", 0))
        covers = (
            w >= int(canvas_w * 0.9)
            and h >= int(canvas_h * 0.9)
        )
    if covers:
        score += 2.0
        reasons.append("covers_canvas")

    opaque = lyr.get("is_fully_opaque")
    if opaque is None:
        op = int(lyr.get("opacity", 0))
        bm = str(lyr.get("blend_mode", "")).lower()
        opaque = (op == 255 and "normal" in bm)
    if opaque:
        score += 1.0
        reasons.append("opaque")

    bbox_ratio = lyr.get("bbox_area_ratio")
    if bbox_ratio is None:
        w = int(lyr.get("size", {}).get("width", 0))
        h = int(lyr.get("size", {}).get("height", 0))
        canvas_area = max(1, canvas_w * canvas_h)
        bbox_ratio = (
            (w * h) / canvas_area
            if w > 0 and h > 0 else 0.0
        )
    if bbox_ratio > 0.5:
        score += 1.5
        reasons.append(f"bbox~{bbox_ratio:.2f}")

    score += bottom_bonus
    if bottom_bonus > 0:
        reasons.append(f"bottom+{bottom_bonus:.2f}")

    if kind in ("type", "adjustment"):
        score -= 2.0
        reasons.append(f"penalty:{kind}")

    # Name-based text detection: catches rasterized text
    # layers (kind='pixel') that wouldn't trigger the
    # kind-based penalty above. Heavier hit because these
    # often cover the full canvas and would otherwise
    # outscore real backgrounds.
    if re.search(r"(?i)\btext\b", name):
        score -= 5.0
        reasons.append("penalty:text-name")

    return score, reasons


class PSDBackgroundDetect:
    """
    Score layers from a PSDLayerSplitter manifest and
    recommend a contiguous bottom range as the background.
    Wire bg_start / bg_end into PSDLayerCompositor's
    replacement_index / replacement_end_index.
    """

    CATEGORY = "Trent/PSD"
    FUNCTION = "detect"

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "bg_start", "bg_end", "confidence", "rationale",
    )
    OUTPUT_TOOLTIPS = (
        "First (bottom) index of detected background range",
        "Last (inclusive) index of detected background range",
        "0..1 confidence in the detection",
        "Per-layer scoring breakdown (human-readable)",
    )
    DESCRIPTION = (
        "Detect the background layer range in a PSD by "
        "scoring layers from a PSDLayerSplitter manifest. "
        "Walks bottom-up, greedily includes contiguous "
        "layers whose score >= threshold. Returns -1, -1 "
        "if no candidate beats min_confidence."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/layer/folder",
                    "tooltip": (
                        "Folder from PSDLayerSplitter "
                        "(must contain _manifest.json)"
                    ),
                }),
                "name_pattern": ("STRING", {
                    "default": (
                        r"(?i)(\b(bg|background|backdrop|"
                        r"sky|wall|floor|fondo|hintergrund|"
                        r"arrière-plan|fond)\b|背景|배경)"
                    ),
                    "tooltip": (
                        "Regex matched against layer "
                        "names. Big score boost on hit. "
                        "Default covers English plus a few "
                        "common localizations: Chinese/"
                        "Japanese 背景, Korean 배경, "
                        "Spanish fondo, German hintergrund, "
                        "French arrière-plan / fond."
                    ),
                }),
                "min_confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Below this, output -1, -1. "
                        "Detector won't guess wildly."
                    ),
                }),
                "max_range_size": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 999,
                    "step": 1,
                    "tooltip": (
                        "Cap on contiguous layers in "
                        "the detected range."
                    ),
                }),
                "score_threshold": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Minimum per-layer score to be "
                        "included in the range."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def detect(
        self,
        folder_path,
        name_pattern,
        min_confidence,
        max_range_size,
        score_threshold,
        **kwargs,
    ) -> Tuple[int, int, float, str]:
        folder_path = str(folder_path or "").strip()
        name_pattern = str(
            name_pattern or r"(?i)\b(bg|background)\b"
        )
        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            min_confidence = 0.5
        try:
            max_range_size = int(max_range_size)
        except (TypeError, ValueError):
            max_range_size = 6
        try:
            score_threshold = float(score_threshold)
        except (TypeError, ValueError):
            score_threshold = 2.5

        if not folder_path or not os.path.isdir(folder_path):
            return (
                -1, -1, 0.0,
                f"folder_path invalid: {folder_path!r}",
            )

        manifest_path = os.path.join(
            folder_path, "_manifest.json"
        )
        if not os.path.isfile(manifest_path):
            return (
                -1, -1, 0.0,
                f"_manifest.json not found in {folder_path}",
            )

        with open(
            manifest_path, "r", encoding="utf-8"
        ) as f:
            manifest = json.load(f)

        canvas_w = int(manifest.get("canvas_width", 1))
        canvas_h = int(manifest.get("canvas_height", 1))
        layers = manifest.get("layers", [])
        if not layers:
            return (-1, -1, 0.0, "no layers in manifest")

        try:
            name_re = re.compile(name_pattern)
        except re.error as e:
            return (
                -1, -1, 0.0,
                f"invalid name_pattern regex: {e}",
            )

        layers_sorted = sorted(
            layers, key=lambda lyr: int(lyr["index"])
        )
        n = len(layers_sorted)

        # Score every layer. Bottom-bias decays linearly.
        scored: List[Tuple[int, float, List[str], str]] = []
        for pos, lyr in enumerate(layers_sorted):
            bottom_bonus = max(
                0.0, 0.5 * (1.0 - pos / max(1, n - 1))
            ) * 5.0
            score, reasons = _score_layer(
                lyr, name_re, canvas_w, canvas_h,
                bottom_bonus,
            )
            scored.append((
                int(lyr["index"]),
                score,
                reasons,
                lyr.get("original_name", ""),
            ))

        # Walk bottom-up, greedily extend the range while
        # score stays at/above threshold.
        included: List[Tuple[int, float, List[str], str]] = []
        for idx, score, reasons, name in scored:
            if score < score_threshold:
                break
            included.append((idx, score, reasons, name))
            if len(included) >= max_range_size:
                break

        if not included:
            rationale = self._format_rationale(
                scored, included, score_threshold,
            )
            return (-1, -1, 0.0, rationale)

        # Confidence: mean score normalized against a
        # plausible ceiling (~10) and clipped to [0, 1].
        mean_score = sum(s for _, s, _, _ in included) / len(included)
        confidence = max(0.0, min(1.0, mean_score / 10.0))

        if confidence < min_confidence:
            rationale = self._format_rationale(
                scored, included, score_threshold,
                note=(
                    f"confidence {confidence:.2f} < "
                    f"min_confidence {min_confidence:.2f}"
                ),
            )
            return (-1, -1, confidence, rationale)

        bg_start = included[0][0]
        bg_end = included[-1][0]
        rationale = self._format_rationale(
            scored, included, score_threshold,
        )
        return (bg_start, bg_end, confidence, rationale)

    @staticmethod
    def _format_rationale(
        scored, included, score_threshold, note=None,
    ) -> str:
        lines = []
        if note:
            lines.append(f"note: {note}")
        lines.append(
            f"threshold: {score_threshold:.2f}  "
            f"included: {len(included)} of {len(scored)}"
        )
        included_idxs = {idx for idx, _, _, _ in included}
        for idx, score, reasons, name in scored:
            mark = "X" if idx in included_idxs else " "
            r = ", ".join(reasons) if reasons else "-"
            lines.append(
                f"  [{mark}] idx={idx:>3} "
                f"score={score:>5.2f}  {name!r}  ({r})"
            )
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "PSDBackgroundDetect": PSDBackgroundDetect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSDBackgroundDetect": "PSD Background Detect",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
