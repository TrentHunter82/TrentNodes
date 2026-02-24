"""
FAL Kling O3 Pro Video-to-Video Reference node.

Calls the FAL AI Kling V2V API to generate a new video from
a reference video and text prompt. Supports optional style
reference images and character/object element injection.

API endpoint: fal-ai/kling-video/o3/pro/video-to-video/reference
"""

import os
import io
import re
import subprocess
import tempfile

import cv2
import fal_client
import numpy as np
import requests
import torch
from PIL import Image

from ..utils.audio_utils import (
    extract_audio_from_dict,
    save_audio_to_wav,
)

LOG = "[Kling V2V]"

ENDPOINT = (
    "fal-ai/kling-video/o3/pro/video-to-video/reference"
)


class FalKlingV2V:
    """
    Generate video using FAL Kling O3 Pro video-to-video.

    Accepts video frames, encodes them as mp4, uploads to FAL,
    and calls the Kling V2V Reference endpoint. The result
    video is downloaded and returned as an IMAGE batch.

    Prompt syntax:
    - @Video1 references the input video
    - @Image1, @Image2 reference style images
    - @Element1, @Element2 reference injected characters
    """

    ASPECT_RATIOS = ["auto", "16:9", "9:16", "1:1"]
    DURATIONS = ["5", "10"]

    # Maps optional input names to FAL @tag references.
    # Only inputs that produce a prompt tag are listed.
    _TAG_MAP = {
        "ref_image_1": "@Image1",
        "ref_image_2": "@Image2",
        "element_1_face": "@Element1",
        "element_2_face": "@Element2",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "tooltip": "FAL AI API key",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "Text prompt for generation. "
                        "@tags are auto-appended for "
                        "connected inputs, or place "
                        "them manually: @Video1, "
                        "@Image1, @Element1, etc."
                    ),
                }),
                "video": ("IMAGE", {
                    "tooltip": (
                        "Input video as an IMAGE batch "
                        "[B, H, W, C]. Will be encoded "
                        "to mp4 at the specified fps."
                    ),
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 60,
                    "step": 1,
                    "tooltip": (
                        "Frame rate for encoding the "
                        "input video"
                    ),
                }),
            },
            "optional": {
                "ref_image_1": ("IMAGE", {
                    "tooltip": (
                        "Style/appearance reference image "
                        "(@Image1 in prompt)"
                    ),
                }),
                "ref_image_2": ("IMAGE", {
                    "tooltip": (
                        "Style/appearance reference image "
                        "(@Image2 in prompt)"
                    ),
                }),
                "element_1_face": ("IMAGE", {
                    "tooltip": (
                        "Frontal face image for Element 1 "
                        "(@Element1 in prompt). "
                        "Requires element_1_ref too."
                    ),
                }),
                "element_1_ref": ("IMAGE", {
                    "tooltip": (
                        "Reference image for Element 1 "
                        "appearance. Required with "
                        "element_1_face."
                    ),
                }),
                "element_2_face": ("IMAGE", {
                    "tooltip": (
                        "Frontal face image for Element 2 "
                        "(@Element2 in prompt). "
                        "Requires element_2_ref too."
                    ),
                }),
                "element_2_ref": ("IMAGE", {
                    "tooltip": (
                        "Reference image for Element 2 "
                        "appearance. Required with "
                        "element_2_face."
                    ),
                }),
                "audio": ("AUDIO", {
                    "tooltip": (
                        "Optional audio to embed in "
                        "the uploaded video. Works "
                        "with keep_audio to preserve "
                        "it in the generated output."
                    ),
                }),
                "aspect_ratio": (cls.ASPECT_RATIOS, {
                    "default": "auto",
                    "tooltip": "Output video aspect ratio",
                }),
                "duration": (cls.DURATIONS, {
                    "default": "5",
                    "tooltip": (
                        "Output video duration in seconds"
                    ),
                }),
                "keep_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Preserve original audio from "
                        "the reference video"
                    ),
                }),
                "nth_frame": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": (
                        "Extract every Nth frame from the "
                        "result (1 = all frames)"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("frames", "video_url", "info")
    OUTPUT_TOOLTIPS = (
        "Generated video frames as IMAGE batch",
        "URL of the generated video on FAL CDN",
        "Status and diagnostic info",
    )
    FUNCTION = "generate"
    CATEGORY = "Trent/API"
    DESCRIPTION = (
        "Calls FAL Kling O3 Pro video-to-video reference "
        "API. Encodes input frames to mp4, uploads to FAL, "
        "generates a new video, and returns the frames. "
        "Costs $0.336 per second of generated video."
    )

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    @staticmethod
    def _build_prompt(prompt, connected):
        """Auto-append @tags for connected inputs.

        If the user already placed the tag in their prompt
        text, it will not be duplicated.

        Args:
            prompt: user's raw prompt text
            connected: dict of input_name -> value
                       (None means not connected)

        Returns:
            (final_prompt, list_of_tags_added)
        """
        added = []
        for input_name, tag in FalKlingV2V._TAG_MAP.items():
            if connected.get(input_name) is None:
                continue
            if tag.lower() in prompt.lower():
                continue
            added.append(tag)

        if added:
            final = prompt.rstrip() + " " + " ".join(added)
        else:
            final = prompt
        return final, added

    @staticmethod
    def _error_return(msg):
        """Return a consistent error tuple."""
        clean = re.sub(
            r"data:[^;]+;base64,[A-Za-z0-9+/=]{100,}",
            "<base64_removed>",
            str(msg),
        )
        print(f"{LOG} ERROR: {clean}")
        empty = torch.zeros((1, 1, 1, 3))
        return (empty, "", clean)

    @staticmethod
    def _tensor_to_frame(tensor):
        """Convert a single image tensor to uint8 numpy RGB."""
        arr = tensor.cpu().numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    # FAL enforces a 10 MB limit on uploaded images.
    _MAX_IMAGE_BYTES = 10 * 1024 * 1024

    @staticmethod
    def _upload_image(image_tensor):
        """Upload an IMAGE tensor to FAL and return the URL.

        Images are saved as JPEG to keep file size well
        under the FAL 10 MB limit.  If the JPEG still
        exceeds the limit the image is progressively
        downscaled until it fits.
        """
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        arr = image_tensor.cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(
            np.uint8
        )
        pil_img = Image.fromarray(arr)

        quality = 92
        max_bytes = FalKlingV2V._MAX_IMAGE_BYTES

        for _ in range(5):
            buf = io.BytesIO()
            pil_img.save(
                buf, format="JPEG", quality=quality
            )
            size = buf.tell()
            if size <= max_bytes:
                break
            # Downscale by 75 % and lower quality
            w, h = pil_img.size
            pil_img = pil_img.resize(
                (int(w * 0.75), int(h * 0.75)),
                Image.LANCZOS,
            )
            quality = max(quality - 10, 60)
            print(
                f"{LOG} Image too large "
                f"({size / 1024 / 1024:.1f} MB), "
                f"resizing to {pil_img.size} "
                f"q={quality}"
            )

        url = fal_client.upload(
            buf.getvalue(),
            content_type="image/jpeg",
        )
        return url

    @staticmethod
    def _upload_video(frames_tensor, fps, audio_dict=None):
        """Encode frames to mp4 and upload to FAL CDN.

        If *audio_dict* is provided the audio is muxed into
        the mp4 via ffmpeg before uploading.

        Args:
            frames_tensor: [B, H, W, C] float32 0-1
            fps: frame rate for the encoded video
            audio_dict: optional ComfyUI AUDIO dict

        Returns:
            FAL CDN URL string
        """
        b, h, w, _c = frames_tensor.shape
        print(
            f"{LOG} Encoding {b} frames at {fps} fps "
            f"({w}x{h})"
        )

        suffix = ".mp4"
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        )
        tmp_path = tmp.name
        tmp.close()

        wav_path = None
        muxed_path = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            tmp_path, fourcc, fps, (w, h)
        )

        try:
            for i in range(b):
                rgb = FalKlingV2V._tensor_to_frame(
                    frames_tensor[i]
                )
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()

            upload_path = tmp_path

            # --- mux audio if provided ---
            if audio_dict is not None:
                waveform, sr = extract_audio_from_dict(
                    audio_dict
                )
                wav_path = save_audio_to_wav(
                    waveform, sr
                )
                muxed = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False
                )
                muxed_path = muxed.name
                muxed.close()

                print(
                    f"{LOG} Muxing audio "
                    f"({sr} Hz) into video..."
                )
                cmd = [
                    "ffmpeg", "-y",
                    "-i", tmp_path,
                    "-i", wav_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    muxed_path,
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(
                        f"{LOG} ffmpeg warning: "
                        f"{result.stderr[:200]}"
                    )
                if os.path.exists(muxed_path) and \
                        os.path.getsize(muxed_path) > 0:
                    upload_path = muxed_path
                    print(f"{LOG} Audio muxed OK")
                else:
                    print(
                        f"{LOG} Audio mux failed, "
                        f"uploading without audio"
                    )

            print(f"{LOG} Uploading video to FAL CDN...")
            url = fal_client.upload_file(upload_path)
            print(f"{LOG} Uploaded: {url[:80]}...")
            return url
        finally:
            writer.release()
            for p in (tmp_path, wav_path, muxed_path):
                if p and os.path.exists(p):
                    os.unlink(p)

    @staticmethod
    def _download_frames(video_url, nth_frame=1):
        """Download video and extract frames as tensor.

        Args:
            video_url: URL of the video to download
            nth_frame: extract every Nth frame

        Returns:
            torch.Tensor [B, H, W, C] float32 0-1
        """
        print(f"{LOG} Downloading result video...")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        )
        tmp_path = tmp.name
        tmp.close()

        try:
            resp = requests.get(video_url, stream=True)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(
                    chunk_size=8192
                ):
                    f.write(chunk)

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise RuntimeError(
                    "Could not open downloaded video"
                )

            total = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(
                f"{LOG} Result: {total} frames at "
                f"{fps:.1f} fps"
            )

            frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % nth_frame == 0:
                    rgb = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )
                    normed = (
                        rgb.astype(np.float32) / 255.0
                    )
                    frames.append(normed)
                idx += 1

            cap.release()

            if not frames:
                raise RuntimeError(
                    "No frames extracted from result"
                )

            tensor = torch.from_numpy(np.stack(frames))
            print(
                f"{LOG} Extracted {len(frames)} frames "
                f"-> {list(tensor.shape)}"
            )
            return tensor
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def generate(
        self,
        api_key,
        prompt,
        video,
        fps,
        ref_image_1=None,
        ref_image_2=None,
        element_1_face=None,
        element_1_ref=None,
        element_2_face=None,
        element_2_ref=None,
        audio=None,
        aspect_ratio="auto",
        duration="5",
        keep_audio=True,
        nth_frame=1,
    ):
        # --- validate ---
        if not api_key or not api_key.strip():
            return self._error_return(
                "API key is required"
            )
        if video is None:
            return self._error_return(
                "Video frames input is required"
            )

        try:
            # --- set API key ---
            clean_key = api_key.strip()
            os.environ["FAL_KEY"] = clean_key
            preview = clean_key[:8] + "..."
            print(f"{LOG} API key: {preview}")

            # --- upload video ---
            video_url = self._upload_video(
                video, fps, audio
            )

            # --- upload reference images ---
            image_urls = []
            for label, img in [
                ("ref_image_1", ref_image_1),
                ("ref_image_2", ref_image_2),
            ]:
                if img is not None:
                    print(f"{LOG} Uploading {label}...")
                    url = self._upload_image(img)
                    image_urls.append(url)
                    print(
                        f"{LOG} {label}: {url[:60]}..."
                    )

            # --- upload element images ---
            elements = []
            for idx, (face, ref) in enumerate([
                (element_1_face, element_1_ref),
                (element_2_face, element_2_ref),
            ], start=1):
                if face is None:
                    continue
                if ref is None:
                    print(
                        f"{LOG} Skipping element_{idx}: "
                        f"both face and ref are "
                        f"required by the API"
                    )
                    continue
                print(
                    f"{LOG} Uploading element_{idx} "
                    f"face..."
                )
                face_url = self._upload_image(face)
                print(
                    f"{LOG} Uploading "
                    f"element_{idx} ref..."
                )
                ref_url = self._upload_image(ref)
                elements.append({
                    "frontal_image_url": face_url,
                    "reference_image_urls": [ref_url],
                })

            # --- auto-insert @tags for connected inputs ---
            connected = {
                "ref_image_1": ref_image_1,
                "ref_image_2": ref_image_2,
                "element_1_face": element_1_face,
                "element_2_face": element_2_face,
            }
            final_prompt, tags_added = self._build_prompt(
                prompt, connected
            )
            if tags_added:
                print(
                    f"{LOG} Auto-appended tags: "
                    f"{', '.join(tags_added)}"
                )
            print(f"{LOG} Final prompt: {final_prompt}")

            # --- build arguments ---
            arguments = {
                "prompt": final_prompt,
                "video_url": video_url,
                "keep_audio": keep_audio,
                "aspect_ratio": aspect_ratio,
                "duration": str(duration),
            }
            if image_urls:
                arguments["image_urls"] = image_urls
            if elements:
                arguments["elements"] = elements

            safe_args = {
                k: (
                    v if not isinstance(v, str)
                    or len(v) < 100
                    else v[:60] + "..."
                )
                for k, v in arguments.items()
            }
            print(f"{LOG} Arguments: {safe_args}")

            # --- call API ---
            def on_queue_update(update):
                if isinstance(
                    update, fal_client.InProgress
                ):
                    for log_entry in update.logs:
                        msg = log_entry.get(
                            "message",
                            str(log_entry),
                        )
                        print(f"{LOG} FAL: {msg}")

            print(f"{LOG} Calling {ENDPOINT}...")
            result = fal_client.subscribe(
                ENDPOINT,
                arguments=arguments,
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            # --- extract result ---
            if (
                "video" not in result
                or "url" not in result["video"]
            ):
                return self._error_return(
                    "No video URL in API response: "
                    + str(result)[:200]
                )

            result_url = result["video"]["url"]
            print(f"{LOG} Video ready: {result_url}")

            # --- download and extract frames ---
            frames = self._download_frames(
                result_url, nth_frame
            )

            parts = [
                f"Generated {frames.shape[0]} frames "
                f"({frames.shape[2]}x{frames.shape[1]})",
                f"Prompt sent: {final_prompt}",
            ]
            if tags_added:
                parts.append(
                    "Auto-added: "
                    + ", ".join(tags_added)
                )
            info = "\n".join(parts)
            print(f"{LOG} {info}")
            return (frames, result_url, info)

        except Exception as exc:
            return self._error_return(str(exc))


NODE_CLASS_MAPPINGS = {
    "FalKlingV2V": FalKlingV2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalKlingV2V": "FAL Kling V2V (O3 Pro)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
