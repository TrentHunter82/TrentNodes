"""
Remotion bridge utilities for TrentNodes.

Shared helpers for Node.js detection, subprocess execution,
tensor I/O, and video frame extraction used by the Remotion
Get Down node suite.
"""

import glob
import json
import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image


def build_env(project: dict) -> dict:
    """
    Build environment dict for subprocess calls to Node.js.

    Ensures Node.js and npm are on PATH even when ComfyUI runs
    from a context where they might not be (conda, WSL2, etc.).

    Args:
        project: REMOTION_PROJECT config dict.

    Returns:
        Environment dict suitable for subprocess calls.
    """
    env = os.environ.copy()

    node_exe = project.get("node_executable", "node")
    if os.path.isabs(node_exe):
        node_dir = os.path.dirname(node_exe)
        env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")

    extra_patterns = [
        os.path.expanduser("~/.nvm/versions/node/*/bin"),
        os.path.expanduser("~/.fnm/node-versions/*/installation/bin"),
        os.path.expanduser("~/.volta/bin"),
        "/usr/local/bin",
    ]
    for pattern in extra_patterns:
        for path in glob.glob(pattern):
            if path not in env.get("PATH", ""):
                env["PATH"] = path + os.pathsep + env["PATH"]

    return env


def validate_node_js(node_executable: str = "node") -> str:
    """
    Check if Node.js is available and return its version string.

    Args:
        node_executable: Path or name of the node binary.

    Returns:
        Version string (e.g. "v20.11.0").

    Raises:
        RuntimeError: If Node.js is not found or fails to run.
    """
    try:
        result = subprocess.run(
            [node_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        raise RuntimeError(
            f"Node.js check failed: {result.stderr.strip()}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Node.js executable '{node_executable}' not found. "
            f"Install Node.js (v16+) and ensure it is on PATH, "
            f"or provide the full path in the node_executable input."
        )


def validate_npx(npx_executable: str = "npx") -> str:
    """
    Check if npx is available and return its version string.

    Args:
        npx_executable: Path or name of the npx binary.

    Returns:
        Version string.

    Raises:
        RuntimeError: If npx is not found or fails to run.
    """
    try:
        result = subprocess.run(
            [npx_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        raise RuntimeError(
            f"npx check failed: {result.stderr.strip()}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"npx executable '{npx_executable}' not found. "
            f"It should be installed alongside Node.js."
        )


def load_image_as_tensor(filepath: str) -> torch.Tensor:
    """
    Load an image file as a ComfyUI IMAGE tensor.

    Args:
        filepath: Absolute path to the image file.

    Returns:
        Tensor of shape (1, H, W, 3) float32 in [0, 1].
    """
    img = Image.open(filepath).convert("RGB")
    np_img = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)


def load_images_from_directory(
    directory: str,
    pattern: str = "*.png",
    max_frames: int = 0,
    start_index: int = 0,
) -> torch.Tensor:
    """
    Load a directory of numbered images as a stacked IMAGE tensor.

    Args:
        directory: Path to the image directory.
        pattern: Glob pattern for matching frame files.
        max_frames: Maximum frames to load (0 = all).
        start_index: Skip this many files from the start.

    Returns:
        Tensor of shape (B, H, W, 3) float32 in [0, 1].

    Raises:
        ValueError: If no matching files are found.
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    files = files[start_index:]
    if max_frames > 0:
        files = files[:max_frames]

    if not files:
        raise ValueError(
            f"No files matching '{pattern}' found in {directory}"
        )

    tensors = []
    for f in files:
        img = Image.open(f).convert("RGB")
        np_img = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(np_img))

    return torch.stack(tensors, dim=0)


def probe_video(video_path: str) -> dict:
    """
    Probe a video file with ffprobe and return metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: fps (float), width (int), height (int),
        duration (float), frame_count (int).

    Raises:
        RuntimeError: If ffprobe fails.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {video_path}: {result.stderr[:500]}"
        )

    data = json.loads(result.stdout)
    info = {
        "fps": 24.0,
        "width": 0,
        "height": 0,
        "duration": 0.0,
        "frame_count": 0,
    }

    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        r_fps = stream.get("r_frame_rate", "24/1")
        num, den = map(int, r_fps.split("/"))
        info["fps"] = num / den if den > 0 else 24.0
        info["width"] = int(stream.get("width", 0))
        info["height"] = int(stream.get("height", 0))
        nb = stream.get("nb_frames")
        if nb and nb != "N/A":
            info["frame_count"] = int(nb)
        break

    fmt = data.get("format", {})
    dur = fmt.get("duration")
    if dur and dur != "N/A":
        info["duration"] = float(dur)

    if info["frame_count"] == 0 and info["duration"] > 0:
        info["frame_count"] = int(
            round(info["duration"] * info["fps"])
        )

    return info


def extract_video_frames(
    video_path: str,
    output_dir: str = None,
    max_frames: int = 0,
    start_frame: int = 0,
    frame_step: int = 1,
    resize_width: int = 0,
    resize_height: int = 0,
) -> tuple:
    """
    Extract frames from a video using ffmpeg.

    Args:
        video_path: Path to the video file.
        output_dir: Directory for extracted PNGs.  Auto-created
            temp dir if None.
        max_frames: Max frames to extract (0 = all).
        start_frame: First frame number to extract.
        frame_step: Extract every Nth frame.
        resize_width: Resize width (0 = original).
        resize_height: Resize height (0 = original).

    Returns:
        Tuple of (output_dir, frame_count, fps).
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="remotion_frames_")
    os.makedirs(output_dir, exist_ok=True)

    info = probe_video(video_path)
    fps = info["fps"]

    vf_filters = []

    if start_frame > 0:
        vf_filters.append(f"select='gte(n\\,{start_frame})'")

    if frame_step > 1:
        vf_filters.append(f"select='not(mod(n\\,{frame_step}))'")

    if resize_width > 0 and resize_height > 0:
        vf_filters.append(f"scale={resize_width}:{resize_height}")

    cmd = ["ffmpeg", "-y", "-i", video_path]

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    cmd.extend(["-vsync", "vfr"])

    if max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])

    cmd.append(os.path.join(output_dir, "frame_%06d.png"))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed: {proc.stderr[:1000]}"
        )

    extracted = sorted(
        glob.glob(os.path.join(output_dir, "frame_*.png"))
    )

    return output_dir, len(extracted), fps


def auto_type(value_str: str):
    """
    Parse a string value into its most specific Python type.

    Tries bool, int, float, JSON (for arrays/objects), then
    falls back to plain string.

    Args:
        value_str: The string to parse.

    Returns:
        The parsed value in its detected type.
    """
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        pass
    return value_str
