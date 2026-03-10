"""Node that triggers automatic browser download of images."""

import json
import os
import random

import folder_paths
import numpy as np
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class BrowserDownload:
    """Saves images and triggers automatic download in the
    user's browser. Useful for remote/headless setups where
    you want files delivered to your local Downloads folder.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Images to download."},
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "download",
                        "tooltip": (
                            "Filename prefix. Supports ComfyUI "
                            "format strings like "
                            "%date:yyyy-MM-dd%."
                        ),
                    },
                ),
            },
            "optional": {
                "format": (
                    ["png", "jpg", "webp"],
                    {
                        "default": "png",
                        "tooltip": (
                            "Image format. JPG/WebP are smaller "
                            "for faster download."
                        ),
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 95,
                        "min": 1,
                        "max": 100,
                        "tooltip": (
                            "Quality for JPG/WebP (1-100). "
                            "Ignored for PNG."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "download_images"
    OUTPUT_NODE = True
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Saves images and automatically triggers a browser "
        "download to your local machine. Perfect for remote "
        "or headless ComfyUI setups."
    )

    def download_images(
        self,
        images,
        filename_prefix="download",
        format="png",
        quality=95,
        prompt=None,
        extra_pnginfo=None,
    ):
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(
                np.clip(i, 0, 255).astype(np.uint8)
            )

            batch_filename = filename.replace(
                "%batch_num%", str(batch_number)
            )
            ext = format
            file = f"{batch_filename}_{counter:05}_.{ext}"
            filepath = os.path.join(full_output_folder, file)

            if format == "png":
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text(
                            "prompt", json.dumps(prompt)
                        )
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(
                                x,
                                json.dumps(extra_pnginfo[x]),
                            )
                img.save(
                    filepath,
                    pnginfo=metadata,
                    compress_level=self.compress_level,
                )
            elif format == "jpg":
                img.save(filepath, quality=quality)
            elif format == "webp":
                img.save(filepath, quality=quality)

            results.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                }
            )
            counter += 1

        return {
            "ui": {
                "images": results,
                "browser_download": True,
            }
        }
