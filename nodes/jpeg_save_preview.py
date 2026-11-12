"""
Save JPEG / Preview JPEG - write image batches straight to .jpg.

From AIWarper's ComfyUI-WarperNodes (Save JPEG / Preview JPEG), added
here with his permission.
"""

import os
import uuid

import folder_paths
import numpy as np
from PIL import Image


class SaveJPEG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to save. Alpha is dropped."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI",
                    "tooltip": "File name start. Subfolders and %batch_num% work."}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100,
                    "tooltip": "JPEG quality. Colour is kept at full resolution (4:4:4)."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Trent/Image"
    DISPLAY_NAME = "Save JPEG"
    DESCRIPTION = ("Saves every image of a batch as a .jpg in the output folder and shows "
                   "them in the node. No PNG is written and no workflow is embedded.")

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    def save_images(self, images, filename_prefix="ComfyUI", quality=95):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix + self.prefix_append, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for batch_number, image in enumerate(images):
            pixels = np.clip(255.0 * image.detach().cpu().numpy(), 0, 255).astype(np.uint8)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            file = "%s_%05d_.jpg" % (filename.replace("%batch_num%", str(batch_number)), counter)
            Image.fromarray(pixels).convert("RGB").save(
                os.path.join(full_output_folder, file), format="JPEG", quality=quality, subsampling=0, optimize=True)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1
        return {"ui": {"images": results}}


class PreviewJPEG(SaveJPEG):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to show. Alpha is dropped."}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "JPEG quality of the preview files."}),
            }
        }

    DISPLAY_NAME = "Preview JPEG"
    DESCRIPTION = "Shows every image of a batch as a .jpg from the temp folder. Nothing is saved to the output folder."

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + uuid.uuid4().hex


NODE_CLASS_MAPPINGS = {
    "SaveJPEG": SaveJPEG,
    "PreviewJPEG": PreviewJPEG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveJPEG": "Save JPEG",
    "PreviewJPEG": "Preview JPEG",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
