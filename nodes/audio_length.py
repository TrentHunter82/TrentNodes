"""
Audio Length - Calculate the duration of audio in seconds, rounded up.
"""

import math
from typing import Dict, Any, Tuple

from ..utils.audio_utils import extract_audio_from_dict, get_audio_duration


class AudioLength:
    """
    Returns the length of an audio input in whole seconds,
    always rounded up to the nearest second.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio to measure"
                }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT",)
    RETURN_NAMES = ("seconds", "seconds_exact",)

    FUNCTION = "measure"
    CATEGORY = "Trent/Audio"
    DESCRIPTION = (
        "Calculates the length of audio in seconds. "
        "Returns both the rounded-up integer and exact float."
    )

    def measure(self, audio: Any) -> Tuple[int, float]:
        waveform, sample_rate = extract_audio_from_dict(audio)
        exact = get_audio_duration(waveform, sample_rate)
        rounded = math.ceil(exact)
        return (rounded, exact)


NODE_CLASS_MAPPINGS = {
    "AudioLength": AudioLength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLength": "Audio Length in Seconds",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
