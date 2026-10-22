"""Eneas segmentation through a dependency-isolated worker."""

import json
import os
import signal
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar

PACK = Path(__file__).resolve().parents[1]


class TrentEneasSegment:
    DISPLAY_NAME = "Eneas Segment & Track (Trent)"
    CATEGORY = "Trent/Segmentation"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    DESCRIPTION = "Track one object using text or points, or mask all objects in a category. White is foreground. Models download on first use."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "RGB video frames or a single image to segment."}),
            "mode": (["track_text", "track_points", "category"], {"tooltip": "track_text tracks one object found by the text; track_points tracks one object picked by points; category masks every object matching the text on each frame."}),
        }, "optional": {
            "text": ("STRING", {"default": "the person", "multiline": True, "tooltip": "Object description for track_text, or category name for category mode. Ignored in track_points mode."}),
            "points": ("STRING", {"default": "[]", "multiline": True, "tooltip": "JSON pixel coordinates: [[x,y,1],[x,y,0]]. 1 includes, 0 excludes. Used only in track_points mode."}),
            "annotation_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "Zero-based frame to annotate. Tracking propagates both forwards and backwards."}),
            "offload_frames_to_cpu": ("BOOLEAN", {"default": True, "tooltip": "Keep video frames in system RAM instead of VRAM during tracking. Lowers VRAM use on long clips; not used in category mode."}),
            "accept_threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": "Category mode only. Detections with image-text similarity at or above this are accepted without VLM review."}),
            "reject_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "tooltip": "Category mode only. Detections with similarity below this are dropped; ones between the thresholds go to the VLM. Must not exceed accept_threshold."}),
        }}

    def segment(self, images, mode, text="the person", points="[]", annotation_frame=0,
                offload_frames_to_cpu=True, accept_threshold=0.9, reject_threshold=0.1):
        if mode not in {"track_text", "track_points", "category"}:
            raise ValueError("Unknown Eneas mode")
        count, height, width, channels = images.shape
        if count == 0 or channels < 3:
            raise ValueError("Eneas needs a nonempty RGB image batch")
        if mode != "category" and not 0 <= annotation_frame < count:
            raise ValueError(f"annotation_frame must be between 0 and {count - 1}")
        parsed = None
        if mode == "track_points":
            parsed = json.loads(points)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("Provide points as [[x,y,1],[x,y,0]]")
            for point in parsed:
                if (not isinstance(point, list) or len(point) != 3
                        or not all(isinstance(v, (int, float)) for v in point)
                        or not 0 <= point[0] < width or not 0 <= point[1] < height
                        or point[2] not in (0, 1)):
                    raise ValueError(f"Each point needs x in [0,{width}), y in [0,{height}), and label 0 or 1")
        elif not text.strip():
            raise ValueError("Enter an object description or category")
        if mode == "category" and reject_threshold > accept_threshold:
            raise ValueError("reject_threshold must not exceed accept_threshold")

        python = PACK / ".eneas/source/.venv/bin/python"
        if not python.is_file():
            raise RuntimeError("Install Eneas first: bash custom_nodes/TrentNodes/tools/install_eneas.sh")
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("Eneas tracking requires an NVIDIA CUDA GPU")
        config_path = PACK / ".eneas/config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        roots = folder_paths.get_folder_paths("eneas") if "eneas" in folder_paths.folder_names_and_paths else []
        model_root = config.get("model_root", roots[0] if roots else str(Path(folder_paths.models_dir) / "eneas"))
        env = os.environ.copy()
        env.update(HF_HOME=str(Path(model_root) / "huggingface"), HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        progress = ProgressBar(count + 1)
        with tempfile.TemporaryDirectory(prefix="trent-eneas-", dir=folder_paths.get_temp_directory()) as directory:
            work = Path(directory)
            frames = work / "frames"
            frames.mkdir()
            for index, frame in enumerate(images):
                mm.throw_exception_if_processing_interrupted()
                array = (frame[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(array).save(frames / f"{index:08d}.jpg", quality=100, subsampling=0)
                progress.update(1)
            request = dict(mode=mode, text=text, points=parsed, annotation_frame=annotation_frame,
                           offload_frames_to_cpu=offload_frames_to_cpu, accept_threshold=accept_threshold,
                           reject_threshold=reject_threshold, count=count, height=height, width=width,
                           device=str(device), model_root=model_root, **{"florence_path": config.get("florence_path")})
            (work / "request.json").write_text(json.dumps(request))
            mm.unload_all_models()
            mm.soft_empty_cache()
            with (work / "worker.log").open("w+") as log:
                process = subprocess.Popen([str(python), "-u", str(PACK / "tools/eneas_worker.py"), str(work)],
                                           env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(work), start_new_session=True)
                try:
                    while True:
                        mm.throw_exception_if_processing_interrupted()
                        try:
                            process.wait(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                    if process.returncode:
                        log.seek(0)
                        raise RuntimeError("Eneas failed:\n" + log.read()[-10000:])
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            masks = np.load(work / "masks.npy", allow_pickle=False)
            if masks.shape != (count, height, width):
                raise RuntimeError(f"Eneas returned unexpected mask shape: {masks.shape}")
            progress.update(1)
            return (torch.from_numpy(masks).float().div_(255),)


NODE_CLASS_MAPPINGS = {"TrentEneasSegment": TrentEneasSegment}
NODE_DISPLAY_NAME_MAPPINGS = {"TrentEneasSegment": TrentEneasSegment.DISPLAY_NAME}
