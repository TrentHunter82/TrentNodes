"""
Midjourney API client for TrentNodes.

Wraps the Yunwu MJ API (https://yunwu.ai). Sync, requests-based.
The reference implementation uses aiohttp + asyncio, but ComfyUI
nodes already run in worker threads, so blocking HTTP keeps things
simple and avoids event-loop juggling.

Endpoints:
    POST /mj/submit/imagine
    POST /mj/submit/action
    POST /mj/submit/blend
    GET  /mj/task/{task_id}/fetch

All requests use:
    Authorization: Bearer <api_key>
    Content-Type: application/json; charset=utf-8
"""

import base64
import io
import os
import time

import numpy as np
import requests
import torch
from PIL import Image


DEFAULT_API_URL = "https://yunwu.ai"
DEFAULT_TIMEOUT_S = 600
DEFAULT_POLL_INTERVAL_S = 3


def resolve_credentials(widget_key, widget_url):
    """
    Resolve API key and URL from a widget value, falling back to
    environment variables.

    Order: widget value -> env var -> (raise for key, default URL).

    Args:
        widget_key: api_key string from the node widget (may be empty)
        widget_url: api_url string from the node widget (may be empty)

    Returns:
        Tuple (api_key, api_url)

    Raises:
        RuntimeError: if no API key can be resolved
    """
    api_key = (widget_key or "").strip()
    if not api_key:
        api_key = (os.environ.get("MJ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Midjourney API key missing. Paste it into the node's "
            "api_key widget or export MJ_API_KEY in the shell that "
            "starts ComfyUI."
        )

    api_url = (widget_url or "").strip()
    if not api_url:
        api_url = (os.environ.get("MJ_API_URL") or "").strip()
    if not api_url:
        api_url = DEFAULT_API_URL

    return api_key, api_url


def tensor_to_base64_png(image):
    """
    Convert a ComfyUI IMAGE tensor to a base64-encoded PNG string
    with the data-URI prefix Yunwu expects.

    Args:
        image: torch.Tensor of shape (B, H, W, C) or (H, W, C),
            values in [0, 1].

    Returns:
        String like "data:image/png;base64,iVBORw0KG..."
    """
    if image.dim() == 4:
        image = image[0]
    arr = image.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def url_to_image_tensor(url, timeout_s=120):
    """
    Download an image from a URL and return it as a ComfyUI IMAGE
    tensor of shape (1, H, W, 3), float32 in [0, 1].

    Args:
        url: Image URL
        timeout_s: HTTP timeout in seconds

    Returns:
        torch.Tensor of shape (1, H, W, 3)
    """
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()

    pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.array(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


def buttons_list_to_dict(buttons):
    """
    Normalize the Yunwu `buttons` list into a label -> customId map.

    Yunwu returns:
        [{"label": "U1", "customId": "MJ::JOB::upsample::1::<hash>"},
         ...]

    Args:
        buttons: list of button dicts from a SUCCESS fetch response

    Returns:
        Dict mapping label string to customId string
    """
    out = {}
    if not buttons:
        return out
    for btn in buttons:
        label = (btn.get("label") or "").strip()
        custom_id = btn.get("customId") or btn.get("custom_id")
        if label and custom_id:
            out[label] = custom_id
    return out


class MJClient:
    """
    Sync client for the Yunwu Midjourney API.

    Usage:
        client = MJClient(api_key, api_url)
        result = client.imagine("a cat --ar 16:9")
        # result -> {"task_id", "image_url", "buttons", "raw"}
    """

    def __init__(
        self,
        api_key,
        api_url=DEFAULT_API_URL,
        timeout_s=DEFAULT_TIMEOUT_S,
        poll_interval_s=DEFAULT_POLL_INTERVAL_S,
    ):
        if not api_key:
            raise RuntimeError("MJClient requires a non-empty api_key")
        self.api_key = api_key
        self.base = (api_url or DEFAULT_API_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json; charset=utf-8",
        }

    def _post_submit(self, path, payload):
        """
        POST to a /mj/submit/* endpoint and return the new task_id.

        Yunwu submit responses look like:
            {"code": 1, "description": "Success", "result": "<task_id>"}
        """
        url = self.base + path
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Midjourney submit failed (network error to "
                + url + "): " + str(exc)
            ) from exc

        if resp.status_code >= 400:
            raise RuntimeError(
                "Midjourney submit failed (HTTP "
                + str(resp.status_code) + "): " + resp.text[:500]
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                "Midjourney submit returned non-JSON response: "
                + resp.text[:500]
            ) from exc

        code = data.get("code")
        if code != 1:
            description = data.get("description") or "(no description)"
            raise RuntimeError(
                "Midjourney submit rejected (code="
                + str(code) + "): " + str(description)
            )

        task_id = data.get("result")
        if not task_id:
            raise RuntimeError(
                "Midjourney submit returned no task id: "
                + str(data)[:500]
            )
        return str(task_id)

    def _fetch(self, task_id):
        """GET /mj/task/{task_id}/fetch and return parsed JSON."""
        url = self.base + "/mj/task/" + task_id + "/fetch"
        try:
            resp = requests.get(
                url, headers=self._headers(), timeout=60
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Midjourney fetch failed (network error): " + str(exc)
            ) from exc

        if resp.status_code >= 400:
            raise RuntimeError(
                "Midjourney fetch failed (HTTP "
                + str(resp.status_code) + "): " + resp.text[:500]
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise RuntimeError(
                "Midjourney fetch returned non-JSON response: "
                + resp.text[:500]
            ) from exc

    def _wait_for_success(self, task_id):
        """
        Poll the task until SUCCESS or until timeout/failure.

        Returns the full SUCCESS payload.
        """
        deadline = time.monotonic() + self.timeout_s
        last_status = "UNKNOWN"

        while time.monotonic() < deadline:
            payload = self._fetch(task_id)
            status = (payload.get("status") or "").upper()
            last_status = status or last_status

            if status == "SUCCESS":
                return payload
            if status in ("FAILURE", "FAILED"):
                reason = (
                    payload.get("failReason")
                    or payload.get("description")
                    or "(no failReason)"
                )
                raise RuntimeError(
                    "Midjourney task " + task_id + " failed: "
                    + str(reason)
                )
            if status not in (
                "NOT_START",
                "SUBMITTED",
                "IN_PROGRESS",
                "MODAL",
            ):
                # Unknown status -- log via exception so the user
                # sees the raw payload instead of hanging forever.
                if status == "":
                    pass  # transient empty status, just keep polling
                else:
                    # Treat as in-progress but cap at deadline.
                    pass

            time.sleep(self.poll_interval_s)

        raise RuntimeError(
            "Midjourney task " + task_id
            + " timed out after " + str(self.timeout_s)
            + "s (last status: " + last_status + ")"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def imagine(self, prompt):
        """
        Submit a text prompt and wait for the 4-image grid result.

        Returns:
            Dict with keys: task_id, image_url, buttons (label->id), raw
        """
        payload = {
            "prompt": prompt,
            "base64Array": [],
            "notifyHook": "",
            "state": "",
        }
        task_id = self._post_submit("/mj/submit/imagine", payload)
        result = self._wait_for_success(task_id)
        return {
            "task_id": task_id,
            "image_url": result.get("imageUrl"),
            "buttons": buttons_list_to_dict(result.get("buttons")),
            "raw": result,
        }

    def action(self, task_id, custom_id):
        """
        Submit an upscale or variation action against a previous task
        using its `customId` from the buttons list.

        Returns:
            Dict with keys: task_id, image_url, buttons, raw
        """
        payload = {
            "chooseSameChannel": True,
            "customId": custom_id,
            "taskId": task_id,
            "notifyHook": "",
            "state": "",
        }
        new_task_id = self._post_submit(
            "/mj/submit/action", payload
        )
        result = self._wait_for_success(new_task_id)
        return {
            "task_id": new_task_id,
            "image_url": result.get("imageUrl"),
            "buttons": buttons_list_to_dict(result.get("buttons")),
            "raw": result,
        }

    def blend(
        self,
        b64_images,
        dimensions="SQUARE",
        bot_type="MID_JOURNEY",
    ):
        """
        Submit a Blend job mixing 2-5 base64 PNG images.

        Args:
            b64_images: list of "data:image/png;base64,..." strings
            dimensions: SQUARE | PORTRAIT | LANDSCAPE
            bot_type: MID_JOURNEY | NIJI_JOURNEY

        Returns:
            Dict with keys: task_id, image_url, buttons, raw
        """
        payload = {
            "botType": bot_type,
            "base64Array": b64_images,
            "dimensions": dimensions,
            "notifyHook": "",
            "state": "",
        }
        task_id = self._post_submit("/mj/submit/blend", payload)
        result = self._wait_for_success(task_id)
        return {
            "task_id": task_id,
            "image_url": result.get("imageUrl"),
            "buttons": buttons_list_to_dict(result.get("buttons")),
            "raw": result,
        }
