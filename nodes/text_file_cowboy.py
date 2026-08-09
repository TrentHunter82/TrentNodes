"""
Text File Cowboy - A directory iterator for text files.

Companion to VideoFolderCowboy / ImageFolderCowboy for per-item
prompt files:
1. Natural sorting for filenames (prompt1 < prompt2 < prompt10)
2. Filename-key lookup (wire VideoFolderCowboy.filename in, and
   clip 000.mp4 loads prompts/000.txt)
3. Configurable index overflow handling (wrap, clamp, error)
4. Cache-busting on file edits via mtime IS_CHANGED
"""

import glob
import os
from typing import Any, Dict, List, Tuple

from .image_folder_cowboy import natural_sort_key


class TextFileCowboy:
    """
    Load one text file from a directory, selected by filename key
    or by index.

    Pairs a folder of prompt .txt files with a folder of media
    files iterated by the other cowboy nodes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/prompts",
                    "tooltip": (
                        "Directory containing the text files"
                    ),
                }),
            },
            "optional": {
                "filename_key": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "If connected, load "
                        "directory/<filename_key><extension>. "
                        "Wire VideoFolderCowboy.filename here so "
                        "clip 000.mp4 loads 000.txt"
                    ),
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": (
                        "Index into the natural-sorted file list "
                        "(used only when filename_key is not "
                        "connected)"
                    ),
                }),
                "extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Text file extension",
                }),
                "index_mode": (
                    ["wrap", "clamp", "error"],
                    {
                        "default": "wrap",
                        "tooltip": (
                            "How to handle index overflow"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "filename", "total_files")
    OUTPUT_TOOLTIPS = (
        "Contents of the loaded text file",
        "Filename without extension",
        "Total number of text files found",
    )

    FUNCTION = "load_text"
    CATEGORY = "Trent/Text"
    DESCRIPTION = (
        "Load one text file from a directory, keyed by filename "
        "or index. Wire VideoFolderCowboy.filename into "
        "filename_key to pair each clip with its own prompt "
        "file (000.mp4 -> 000.txt). Edits to the file bust the "
        "cache automatically."
    )

    @classmethod
    def discover_files(
        cls,
        directory: str,
        extension: str,
    ) -> List[str]:
        """
        Discover text files with the given extension.

        Args:
            directory: Directory to search
            extension: File extension (with or without dot)

        Returns:
            Natural-sorted list of file paths
        """
        if not directory or not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        ext = extension.strip()
        if ext and not ext.startswith('.'):
            ext = '.' + ext

        pattern = os.path.join(directory, f"*{ext}")
        files = [
            f for f in glob.glob(pattern)
            if os.path.isfile(f)
        ]
        files.sort(key=natural_sort_key)
        return files

    @classmethod
    def resolve_path(
        cls,
        directory: str,
        filename_key: str,
        index: int,
        extension: str,
        index_mode: str,
    ) -> Tuple[str, int]:
        """
        Resolve the target file path.

        Args:
            directory: Directory to search
            filename_key: Explicit filename (no extension)
            index: Index into the sorted list
            extension: File extension
            index_mode: wrap, clamp, or error

        Returns:
            Tuple of (file_path, total_files)

        Raises:
            FileNotFoundError: Keyed file missing
            IndexError: Mode 'error' and index out of range
        """
        ext = extension.strip()
        if ext and not ext.startswith('.'):
            ext = '.' + ext

        if filename_key:
            path = os.path.join(
                directory, f"{filename_key}{ext}"
            )
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Text file not found: {path}"
                )
            total = len(
                cls.discover_files(directory, extension)
            )
            return path, total

        files = cls.discover_files(directory, extension)
        total = len(files)
        if total == 0:
            raise FileNotFoundError(
                f"No '*{ext}' files in '{directory}'"
            )

        if index_mode == "wrap":
            idx = index % total
        elif index_mode == "clamp":
            idx = max(0, min(index, total - 1))
        else:
            if index < 0 or index >= total:
                raise IndexError(
                    f"Text index {index} out of range "
                    f"[0, {total})"
                )
            idx = index

        return files[idx], total

    @classmethod
    def IS_CHANGED(
        cls,
        directory: str,
        filename_key: str = "",
        index: int = 0,
        extension: str = ".txt",
        index_mode: str = "wrap",
    ) -> str:
        try:
            path, _ = cls.resolve_path(
                directory, filename_key, index,
                extension, index_mode,
            )
            return f"{path}:{os.path.getmtime(path)}"
        except (FileNotFoundError, IndexError, OSError):
            return "missing"

    def load_text(
        self,
        directory: str,
        filename_key: str = "",
        index: int = 0,
        extension: str = ".txt",
        index_mode: str = "wrap",
    ) -> Tuple[str, str, int]:
        """
        Main execution function for the node.

        Args:
            directory: Directory containing text files
            filename_key: Explicit filename (no extension)
            index: Index into the sorted list
            extension: File extension
            index_mode: Overflow handling mode

        Returns:
            Tuple of (text, filename, total_files)
        """
        path, total = self.resolve_path(
            directory, filename_key, index,
            extension, index_mode,
        )

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        filename = os.path.splitext(
            os.path.basename(path)
        )[0]

        print(
            f"[TextFileCowboy] Loaded '{filename}' "
            f"({len(text)} chars) from {total} files"
        )

        return (text, filename, total)


NODE_CLASS_MAPPINGS = {
    "TextFileCowboy": TextFileCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextFileCowboy": "Text File Cowboy",
}
