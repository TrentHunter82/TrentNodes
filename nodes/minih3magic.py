import torch

class MiniH3FrameAdjusterNode:
    """
    A ComfyUI node that adjusts video frame counts to meet MiniMax H3 requirements.
    It always rounds up to the next valid frame count by adding blank gray frames.
    MiniMax H3 needs frame counts of the form (n*17)+5 (e.g., 5, 22, 39, 56, 73, etc.)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("adjusted_images", "final_frame_count", "frames_added", "duration_seconds")
    FUNCTION = "adjust_frame_count"
    CATEGORY = "Trent/Utilities"

    def adjust_frame_count(self, images):
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

        # Duration of the adjusted clip at H3's native 24 fps
        duration_seconds = final_count / 24.0

        return (adjusted_images, final_count, frames_to_add, duration_seconds)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "MiniH3FrameAdjusterNode": MiniH3FrameAdjusterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniH3FrameAdjusterNode": "MiniH3 Magic (Frame Adjuster)"
}
