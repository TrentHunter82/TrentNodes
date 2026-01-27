"""
Image+Text Grid
A ComfyUI custom node for creating grid layouts of images
with captions.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import Dict, Any, Tuple, List


class ImageTextGrid:
    """
    Creates a grid layout of images with text captions below each.

    Pass a batch of images (up to 9 used) with optional captions.
    Images preserve their original aspect ratio.
    """

    BACKGROUND_COLORS = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "images_per_row": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": (
                        "Number of images per row in the grid"
                    )
                }),
                "image_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 1024,
                    "step": 8,
                    "tooltip": (
                        "Max width/height for each image cell"
                    )
                }),
                "caption_height": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Height reserved for caption text"
                }),
                "font_size": ("INT", {
                    "default": 12,
                    "min": 8,
                    "max": 48,
                    "step": 1,
                    "tooltip": "Font size for caption text"
                }),
                "padding": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 2,
                    "tooltip": "Padding between images"
                }),
                "background_color": (
                    list(cls.BACKGROUND_COLORS.keys()), {
                        "default": "white",
                        "tooltip": "Background color of the layout"
                    }
                ),
                "caption_prefix": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Bold text prepended before each caption"
                    )
                }),
            },
            "optional": {
                "caption_1": ("STRING", {"forceInput": True}),
                "caption_2": ("STRING", {"forceInput": True}),
                "caption_3": ("STRING", {"forceInput": True}),
                "caption_4": ("STRING", {"forceInput": True}),
                "caption_5": ("STRING", {"forceInput": True}),
                "caption_6": ("STRING", {"forceInput": True}),
                "caption_7": ("STRING", {"forceInput": True}),
                "caption_8": ("STRING", {"forceInput": True}),
                "caption_9": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("layout",)

    FUNCTION = "create_layout"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Creates a grid layout from a batch of images "
        "(up to 9) with optional captions."
    )

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a single image tensor to PIL Image."""
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def resize_keep_aspect(
        self, pil_img: Image.Image, max_size: int
    ) -> Image.Image:
        """Resize image to fit within max_size box,
        preserving aspect ratio."""
        w, h = pil_img.size
        scale = min(max_size / w, max_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return pil_img.resize((new_w, new_h), Image.LANCZOS)

    def create_layout(
        self,
        images: torch.Tensor,
        images_per_row: int,
        image_size: int,
        caption_height: int,
        font_size: int,
        padding: int,
        background_color: str,
        caption_prefix: str = "",
        caption_1=None, caption_2=None, caption_3=None,
        caption_4=None, caption_5=None, caption_6=None,
        caption_7=None, caption_8=None, caption_9=None,
    ) -> Tuple[torch.Tensor]:
        """Create the grid layout with images and captions."""

        captions = [
            caption_1, caption_2, caption_3,
            caption_4, caption_5, caption_6,
            caption_7, caption_8, caption_9
        ]

        # Take up to 9 images from the batch
        if images.dim() == 3:
            images = images.unsqueeze(0)
        num_images = min(images.shape[0], 9)

        if num_images == 0:
            placeholder = torch.zeros(
                (1, 256, 256, 3), dtype=torch.float32
            )
            return (placeholder,)

        # Get colors
        bg_color = self.BACKGROUND_COLORS.get(
            background_color, (255, 255, 255)
        )
        text_color = (
            (0, 0, 0) if background_color == "white"
            else (255, 255, 255)
        )

        # Calculate layout dimensions
        num_rows = (
            (num_images + images_per_row - 1) // images_per_row
        )
        total_width = (
            images_per_row * (image_size + padding) + padding
        )
        total_height = num_rows * (
            image_size + caption_height + padding
        ) + padding

        # Create layout canvas
        layout = Image.new(
            'RGB', (total_width, total_height), color=bg_color
        )
        draw = ImageDraw.Draw(layout)

        # Try to load regular and bold fonts
        font = None
        bold_font = None
        font_paths = [
            ("arial.ttf", "arialbd.ttf"),
            (
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans-Bold.ttf",
            ),
        ]
        for reg_path, bold_path in font_paths:
            try:
                font = ImageFont.truetype(
                    reg_path, font_size
                )
                bold_font = ImageFont.truetype(
                    bold_path, font_size
                )
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()
            bold_font = font

        # Place each image and caption
        for i in range(num_images):
            row = i // images_per_row
            col = i % images_per_row

            x = padding + col * (image_size + padding)
            y = padding + row * (
                image_size + caption_height + padding
            )

            # Convert tensor to PIL and resize preserving
            # aspect ratio
            pil_img = self.tensor_to_pil(images[i])
            pil_img = self.resize_keep_aspect(
                pil_img, image_size
            )

            # Center image within its cell
            offset_x = (image_size - pil_img.width) // 2
            offset_y = (image_size - pil_img.height) // 2
            layout.paste(pil_img, (x + offset_x, y + offset_y))

            # Draw caption if caption_height > 0
            caption = captions[i] if i < len(captions) else None
            caption = str(caption) if caption else ""
            if caption_height > 0 and (caption or caption_prefix):
                text_x = x + 5
                text_y = y + image_size + 5

                # Draw bold prefix first
                if caption_prefix:
                    draw.text(
                        (text_x, text_y),
                        caption_prefix,
                        font=bold_font,
                        fill=text_color
                    )
                    bbox = draw.textbbox(
                        (text_x, text_y),
                        caption_prefix,
                        font=bold_font
                    )
                    text_y = bbox[3] + 2

                # Draw wrapped caption below prefix
                if caption:
                    chars_per_line = max(
                        1,
                        (image_size - 10)
                        // (font_size // 2 + 1)
                    )
                    wrapped = textwrap.fill(
                        caption, width=chars_per_line
                    )

                    lines = wrapped.split('\n')
                    remaining_h = (
                        caption_height
                        - (text_y - (y + image_size)) - 5
                    )
                    max_lines = max(
                        1, remaining_h // (font_size + 2)
                    )
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        if lines[-1]:
                            lines[-1] = (
                                lines[-1][:-3] + "..."
                            )
                    wrapped = '\n'.join(lines)

                    draw.text(
                        (text_x, text_y),
                        wrapped,
                        font=font,
                        fill=text_color
                    )

        # Convert back to tensor
        layout_np = (
            np.array(layout).astype(np.float32) / 255.0
        )
        layout_tensor = torch.from_numpy(layout_np).unsqueeze(0)

        return (layout_tensor,)


NODE_CLASS_MAPPINGS = {
    "ImageTextGrid": ImageTextGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTextGrid": "Image+Text Grid",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
