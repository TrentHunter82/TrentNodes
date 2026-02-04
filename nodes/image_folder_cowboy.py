"""
Image Folder Cowboy - An improved image directory iterator with proper sorting.

Fixes issues with cspnodes ImageDirIterator:
1. Natural sorting for filenames (img1 < img2 < img10)
2. Sorted subdirectory processing
3. Configurable index overflow handling (wrap, clamp, error)
"""

import glob
import os
import random
import re
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def natural_sort_key(path: str) -> List:
    """
    Generate a sort key for natural sorting of filenames.

    Splits filename into text and numeric chunks so that:
    - 'img1.png' < 'img2.png' < 'img10.png'
    - 'frame_001_v2.png' < 'frame_001_v10.png'

    Args:
        path: File path to generate sort key for

    Returns:
        List of alternating strings and integers for comparison
    """
    name = os.path.basename(path)
    # Split on digit sequences, keeping the delimiters
    parts = re.split(r'(\d+)', name)
    # Convert numeric parts to int, lowercase text parts for case-insensitive
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def get_sort_key(sort_by: str):
    """
    Get the appropriate sort key function for the given sort method.

    Args:
        sort_by: Sorting method name

    Returns:
        Sort key function or None for random/none
    """
    sort_functions = {
        "name": natural_sort_key,
        "date_modified": lambda x: os.path.getmtime(x),
        "size": lambda x: os.path.getsize(x),
        "random": None,
        "none": None,
    }
    return sort_functions.get(sort_by)


# Valid image extensions
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff'}


class ImageFolderCowboy:
    """
    Load images from a directory with proper natural sorting and flexible
    index handling.

    Unlike other directory iterators, this node:
    - Uses natural sorting (img1 < img2 < img10)
    - Sorts subdirectories alphabetically
    - Offers wrap/clamp/error modes for index overflow
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/images",
                    "tooltip": "Root directory to search for images"
                }),
                "patterns": ("STRING", {
                    "default": "**/*.png, **/*.jpg, **/*.jpeg, **/*.webp",
                    "tooltip": "Comma-separated glob patterns for matching files"
                }),
                "image_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Starting index (or batch index if increment_by_batch)"
                }),
                "sort_by": (["name", "date_modified", "size", "random", "none"], {
                    "default": "name",
                    "tooltip": "Sorting method: name uses natural sort"
                }),
                "sort_order": (["ascending", "descending"], {
                    "default": "ascending",
                    "tooltip": "Sort direction"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of images to load per execution"
                }),
            },
            "optional": {
                "increment_by_batch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, index is multiplied by batch_size"
                }),
                "index_mode": (["wrap", "clamp", "error"], {
                    "default": "wrap",
                    "tooltip": "How to handle index overflow"
                }),
                "sort_subdirs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Sort subdirectories alphabetically"
                }),
                "randomize_final": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Shuffle entire list after sorting"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "masks", "filenames", "total_count", "file_paths")
    OUTPUT_IS_LIST = (True, True, True, False, True)
    OUTPUT_TOOLTIPS = (
        "Batch of loaded images",
        "Alpha channel masks (inverted, white = transparent)",
        "Filenames without extension",
        "Total number of images found",
        "Full file paths"
    )

    FUNCTION = "load_images"
    CATEGORY = "Trent/Image"
    DESCRIPTION = (
        "Load images from a directory with natural sorting. "
        "Properly handles numbered files (img1 < img2 < img10) and "
        "offers configurable index overflow behavior."
    )

    def discover_files(
        self,
        directory: str,
        patterns: str
    ) -> List[str]:
        """
        Discover image files matching the given patterns.

        Args:
            directory: Root directory to search
            patterns: Comma-separated glob patterns

        Returns:
            List of unique file paths
        """
        if not directory or not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Parse patterns
        pattern_list = [p.strip() for p in patterns.split(',') if p.strip()]
        if not pattern_list:
            pattern_list = ["**/*.png", "**/*.jpg"]

        # Collect files from all patterns
        found_files = set()
        for pattern in pattern_list:
            full_pattern = os.path.join(directory, pattern)
            matches = glob.glob(full_pattern, recursive=True)
            for match in matches:
                ext = os.path.splitext(match)[1].lower()
                if ext in VALID_EXTENSIONS and os.path.isfile(match):
                    found_files.add(match)

        if not found_files:
            raise FileNotFoundError(
                f"No images found in '{directory}' matching '{patterns}'"
            )

        return list(found_files)

    def sort_files(
        self,
        files: List[str],
        sort_by: str,
        sort_order: str,
        sort_subdirs: bool,
        randomize_final: bool
    ) -> List[str]:
        """
        Sort files with subdirectory grouping.

        Args:
            files: List of file paths
            sort_by: Sorting method
            sort_order: ascending or descending
            sort_subdirs: Whether to sort subdirectories
            randomize_final: Shuffle entire list after sorting

        Returns:
            Sorted list of file paths
        """
        # Group by subdirectory
        subdirs: Dict[str, List[str]] = {}
        for f in files:
            subdir = os.path.dirname(f)
            if subdir not in subdirs:
                subdirs[subdir] = []
            subdirs[subdir].append(f)

        # Get subdirectory order
        if sort_subdirs:
            subdir_order = sorted(subdirs.keys())
        else:
            subdir_order = list(subdirs.keys())

        # Sort files within each subdirectory
        reverse = (sort_order == "descending")
        sort_key = get_sort_key(sort_by)

        sorted_files = []
        for subdir in subdir_order:
            subdir_files = subdirs[subdir]

            if sort_by == "random":
                random.shuffle(subdir_files)
            elif sort_by == "none":
                pass  # Keep discovery order
            else:
                subdir_files.sort(key=sort_key, reverse=reverse)

            sorted_files.extend(subdir_files)

        # Optional final shuffle
        if randomize_final:
            random.shuffle(sorted_files)

        return sorted_files

    def calculate_indices(
        self,
        total_count: int,
        image_index: int,
        batch_size: int,
        increment_by_batch: bool,
        index_mode: str
    ) -> Tuple[int, int]:
        """
        Calculate start and end indices with overflow handling.

        Args:
            total_count: Total number of files
            image_index: User-specified index
            batch_size: Number of images to load
            increment_by_batch: Multiply index by batch_size
            index_mode: wrap, clamp, or error

        Returns:
            Tuple of (start_index, end_index)

        Raises:
            IndexError: If index_mode is 'error' and index is out of bounds
        """
        # Calculate raw start index
        if increment_by_batch:
            start = image_index * batch_size
        else:
            start = image_index

        # Handle overflow based on mode
        if index_mode == "wrap":
            start = start % total_count
        elif index_mode == "clamp":
            max_start = max(0, total_count - batch_size)
            start = max(0, min(start, max_start))
        elif index_mode == "error":
            if start < 0 or start >= total_count:
                raise IndexError(
                    f"Index {start} out of range [0, {total_count})"
                )

        # Calculate end index (with wrapping for batch)
        end = start + batch_size

        return start, end

    def load_image_batch(
        self,
        files: List[str],
        start: int,
        end: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[str]]:
        """
        Load a batch of images from the file list.

        Args:
            files: Sorted list of all file paths
            start: Start index
            end: End index (may exceed len(files) for wrapping)

        Returns:
            Tuple of (images, masks, filenames, file_paths)
        """
        total = len(files)
        images = []
        masks = []
        filenames = []
        file_paths = []

        for i in range(start, end):
            # Wrap index if needed
            idx = i % total
            file_path = files[idx]

            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)

                # Convert to RGB for image tensor
                rgb = img.convert("RGB")
                image_np = np.array(rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                images.append(image_tensor)

                # Extract alpha channel for mask (inverted)
                if 'A' in img.getbands():
                    alpha = np.array(img.getchannel('A')).astype(np.float32)
                    alpha = alpha / 255.0
                    mask_tensor = 1.0 - torch.from_numpy(alpha)
                else:
                    # No alpha = fully opaque = mask of zeros
                    h, w = image_np.shape[:2]
                    mask_tensor = torch.zeros((h, w), dtype=torch.float32)
                masks.append(mask_tensor)

                # Extract filename without extension
                basename = os.path.basename(file_path)
                name_no_ext = os.path.splitext(basename)[0]
                filenames.append(name_no_ext)
                file_paths.append(file_path)

            except Exception as e:
                print(f"[ImageFolderCowboy] Error loading {file_path}:")
                print(f"  {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

        return images, masks, filenames, file_paths

    def load_images(
        self,
        directory: str,
        patterns: str,
        image_index: int,
        sort_by: str,
        sort_order: str,
        batch_size: int,
        increment_by_batch: bool = False,
        index_mode: str = "wrap",
        sort_subdirs: bool = True,
        randomize_final: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], int, List[str]]:
        """
        Main execution function for the node.

        Args:
            directory: Root directory path
            patterns: Comma-separated glob patterns
            image_index: Starting index
            sort_by: Sorting method
            sort_order: Sort direction
            batch_size: Images per batch
            increment_by_batch: Multiply index by batch_size
            index_mode: Overflow handling mode
            sort_subdirs: Sort subdirectories alphabetically
            randomize_final: Shuffle after sorting

        Returns:
            Tuple of (images, masks, filenames, total_count, file_paths)
        """
        # Discover files
        print(f"[ImageFolderCowboy] Searching directory: {directory}")
        print(f"[ImageFolderCowboy] Using patterns: {patterns}")
        files = self.discover_files(directory, patterns)
        print(f"[ImageFolderCowboy] Found {len(files)} files")

        # Sort files
        sorted_files = self.sort_files(
            files, sort_by, sort_order, sort_subdirs, randomize_final
        )

        total_count = len(sorted_files)

        # Calculate indices
        start, end = self.calculate_indices(
            total_count, image_index, batch_size,
            increment_by_batch, index_mode
        )
        print(f"[ImageFolderCowboy] Loading indices {start} to {end}")

        # Load images
        images, masks, filenames, file_paths = self.load_image_batch(
            sorted_files, start, end
        )

        # Handle case where no images loaded successfully
        if not images:
            # Provide more detail about what went wrong
            attempted = sorted_files[start:end] if end <= len(sorted_files) else (
                sorted_files[start:] + sorted_files[:end - len(sorted_files)]
            )
            msg = (
                f"Failed to load any images from batch.\n"
                f"  Directory: {directory}\n"
                f"  Files found: {total_count}\n"
                f"  Attempted to load: {attempted[:5]}"
                f"{'...' if len(attempted) > 5 else ''}"
            )
            raise RuntimeError(msg)

        # Resize masks to match if needed (for batching consistency)
        if len(masks) > 1:
            # Get max dimensions
            max_h = max(m.shape[0] for m in masks)
            max_w = max(m.shape[1] for m in masks)

            resized_masks = []
            for mask in masks:
                if mask.shape[0] != max_h or mask.shape[1] != max_w:
                    mask_4d = mask.unsqueeze(0).unsqueeze(0)
                    resized = F.interpolate(
                        mask_4d, size=(max_h, max_w), mode='bilinear',
                        align_corners=False
                    )
                    resized_masks.append(resized.squeeze(0).squeeze(0))
                else:
                    resized_masks.append(mask)
            masks = resized_masks

        return (images, masks, filenames, total_count, file_paths)


NODE_CLASS_MAPPINGS = {
    "ImageFolderCowboy": ImageFolderCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFolderCowboy": "Image Folder Cowboy",
}
