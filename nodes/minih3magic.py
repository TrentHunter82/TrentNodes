import torch

# The node assumes H3's native frame rate when converting frames to audio time.
H3_FPS = 24.0


class MiniH3FrameAdjusterNode:
    """
    A ComfyUI node that adjusts video frame counts to meet MiniMax H3 requirements.
    It always rounds up to the next valid frame count by adding blank gray frames.
    MiniMax H3 needs frame counts of the form (n*17)+5 (e.g., 5, 22, 39, 56, 73, etc.)

    If audio is connected, the node pads it with silence (or trims it) so the
    audio duration matches the adjusted clip length at 24 fps.
    """

    DESCRIPTION = (
        "Rounds a clip up to the next MiniMax H3-valid frame count ((n*17)+5: "
        "5, 22, 39, 56, 73, ...) by appending blank gray frames. Optional audio "
        "is padded with silence or trimmed to match the new length at 24 fps."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames to pad. Gray frames are appended "
                               "until the count hits the next (n*17)+5 value.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio track. Padded with trailing "
                               "silence (or trimmed) to match the adjusted "
                               "clip length at 24 fps.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "INT")
    RETURN_NAMES = ("adjusted_images", "adjusted_audio", "final_frame_count", "frames_added", "duration_seconds")
    OUTPUT_TOOLTIPS = (
        "Original frames plus appended gray padding frames.",
        "Audio padded/trimmed to the adjusted clip length (None if no audio was connected).",
        "Total frame count after padding — always of the form (n*17)+5.",
        "Number of gray frames that were appended (0 if the input count was already valid).",
        "Adjusted clip duration at 24 fps, rounded to the nearest whole second.",
    )
    FUNCTION = "adjust_frame_count"
    CATEGORY = "Trent/Utilities"

    def adjust_frame_count(self, images, audio=None):
        """
        Main function to adjust frame count for MiniMax H3 compatibility.
        It always rounds up to the next valid count by adding gray frames.
        """
        current_count = images.shape[0]

        # H3 valid counts are (n*17)+5, starting from n=0 (5 frames).
        # Smallest valid count is 5.
        if current_count < 5:
            target_count = 5
        else:
            remainder = (current_count - 5) % 17
            if remainder == 0:
                target_count = current_count  # Already a valid count
            else:
                target_count = current_count + (17 - remainder)

        frames_to_add = target_count - current_count

        if frames_to_add > 0:
            # Get image dimensions (height, width, channels)
            _, h, w, c = images.shape

            # Create a single gray frame (0.5 for float tensors, which is common in ComfyUI)
            gray_frame = torch.full((1, h, w, c), 0.5, dtype=images.dtype, device=images.device)

            # Duplicate the gray frame for the number of frames to add
            added_frames = gray_frame.repeat(frames_to_add, 1, 1, 1)

            # Concatenate original frames with the new gray frames
            adjusted_images = torch.cat([images, added_frames], dim=0)
        else:
            adjusted_images = images

        final_count = adjusted_images.shape[0]

        adjusted_audio = self._match_audio_length(audio, final_count)

        # Duration of the adjusted clip at H3's native 24 fps,
        # rounded to the nearest whole second for INT consumers
        duration_seconds = round(final_count / H3_FPS)

        return (adjusted_images, adjusted_audio, final_count, frames_to_add, duration_seconds)

    @staticmethod
    def _match_audio_length(audio, frame_count):
        """
        Pad the waveform with trailing silence (or trim it) so its duration
        equals frame_count / 24 fps. Returns None when no audio is connected.
        """
        if audio is None:
            return None

        waveform = audio["waveform"]  # [batch, channels, samples]
        sample_rate = audio["sample_rate"]

        target_samples = round(frame_count / H3_FPS * sample_rate)
        current_samples = waveform.shape[-1]

        if current_samples < target_samples:
            silence = torch.zeros(
                (*waveform.shape[:-1], target_samples - current_samples),
                dtype=waveform.dtype, device=waveform.device,
            )
            waveform = torch.cat([waveform, silence], dim=-1)
        elif current_samples > target_samples:
            waveform = waveform[..., :target_samples]

        return {"waveform": waveform, "sample_rate": sample_rate}


# Node mappings
NODE_CLASS_MAPPINGS = {
    "MiniH3FrameAdjusterNode": MiniH3FrameAdjusterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniH3FrameAdjusterNode": "MiniH3 Magic (Frame Adjuster)"
}
