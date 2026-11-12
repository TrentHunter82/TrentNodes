"""
Tests for Save JPEG / Preview JPEG (from AIWarper's ComfyUI-WarperNodes).

Run:
    cd /home/trent/ComfyUI && venv/bin/python -m pytest custom_nodes/TrentNodes/tests/test_jpeg_save_preview.py

Only ComfyUI's folder service is stubbed, so real .jpg files are written to a temp folder.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "jpeg_save_preview.py"


class JPEGNodesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.output, self.temp = root / "output", root / "temp"
        self.output.mkdir()
        self.temp.mkdir()
        paths = types.ModuleType("folder_paths")
        paths.get_output_directory = lambda: str(self.output)
        paths.get_temp_directory = lambda: str(self.temp)
        paths.get_save_image_path = self.save_path
        spec = importlib.util.spec_from_file_location("trent_jpeg_test", MODULE_PATH)
        self.nodes = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"folder_paths": paths}):
            spec.loader.exec_module(self.nodes)

    def save_path(self, prefix, directory, width, height):
        target = Path(directory) / prefix
        target.parent.mkdir(parents=True, exist_ok=True)
        subfolder = str(target.parent.relative_to(directory))
        return str(target.parent), target.name, len(list(target.parent.glob("*.jpg"))) + 1, subfolder, prefix

    def image_path(self, item):
        return (self.temp if item["type"] == "temp" else self.output) / item["subfolder"] / item["filename"]

    def test_save_writes_one_jpeg_per_image(self):
        result = self.nodes.SaveJPEG().save_images(torch.rand((2, 32, 48, 3)), "nested/frame_%batch_num%")
        items = result["ui"]["images"]
        self.assertEqual(len(items), 2)
        for index, item in enumerate(items):
            self.assertEqual((item["type"], item["subfolder"]), ("output", "nested"))
            self.assertIn("frame_%d_" % index, item["filename"])
            with Image.open(self.image_path(item)) as image:
                self.assertEqual((image.format, image.size, image.mode), ("JPEG", (48, 32), "RGB"))
        self.assertEqual(list(self.temp.iterdir()), [])

    def test_preview_writes_to_temp_only(self):
        result = self.nodes.PreviewJPEG().save_images(torch.rand((1, 16, 16, 3)), quality=60)
        item = result["ui"]["images"][0]
        self.assertEqual(item["type"], "temp")
        self.assertTrue(self.image_path(item).is_file())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_alpha_and_single_channel_become_rgb(self):
        for channels in (4, 1):
            item = self.nodes.SaveJPEG().save_images(torch.rand((1, 8, 8, channels)), "c%d" % channels)["ui"]["images"][0]
            with Image.open(self.image_path(item)) as image:
                self.assertEqual(image.mode, "RGB")

    def test_lower_quality_makes_a_smaller_file(self):
        frames = torch.rand((1, 128, 128, 3))
        sizes = [self.image_path(self.nodes.SaveJPEG().save_images(frames, "q%d" % q, q)["ui"]["images"][0]).stat().st_size
                 for q in (95, 30)]
        self.assertGreater(sizes[0], sizes[1])


if __name__ == "__main__":
    unittest.main()
