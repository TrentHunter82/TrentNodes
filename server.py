"""
Server routes for TrentNodes.

Provides API endpoints for folder browsing and file listing.
"""

import asyncio
import os
import re
import shutil

import server

web = server.web


def is_safe_path(path: str) -> bool:
    """
    Check if a path is safe to access.

    Prevents directory traversal attacks and access to sensitive directories.

    Args:
        path: Absolute path to check

    Returns:
        True if path is safe to access
    """
    # Normalize the path to resolve .. and symlinks
    try:
        real_path = os.path.realpath(path)
    except (OSError, ValueError):
        return False

    # Block access to sensitive system directories
    sensitive_dirs = [
        '/etc',
        '/root',
        '/var',
        '/usr',
        '/bin',
        '/sbin',
        '/boot',
        '/proc',
        '/sys',
        '/dev',
    ]

    # Windows sensitive paths
    if os.name == 'nt':
        sensitive_dirs = [
            'C:\\Windows',
            'C:\\Program Files',
            'C:\\Program Files (x86)',
        ]

    for sensitive in sensitive_dirs:
        if real_path.startswith(sensitive):
            return False

    return True


def strip_path(path: str) -> str:
    """
    Strip leading/trailing quotes and whitespace from path.

    Args:
        path: Path string to clean

    Returns:
        Cleaned path string
    """
    path = path.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path.startswith("'") and path.endswith("'"):
        path = path[1:-1]
    return path


@server.PromptServer.instance.routes.get("/trent/browse")
async def browse_folder(request):
    """
    List contents of a directory for folder browsing.

    Query parameters:
        path: Directory path to list (required)
        extensions: Comma-separated list of valid extensions (optional)
        dirs_only: If "true", only return directories (optional)

    Returns:
        JSON array of items, directories end with "/"
    """
    query = request.rel_url.query

    if "path" not in query:
        return web.json_response([])

    # Clean and resolve path
    path = strip_path(query["path"])

    # Handle empty path - return common starting points
    if not path:
        items = []
        # Add home directory
        home = os.path.expanduser("~")
        if os.path.isdir(home):
            items.append(home + os.sep)
        # On Windows, add drives
        if os.name == 'nt':
            import string
            for drive in string.ascii_uppercase:
                drive_path = f"{drive}:\\"
                if os.path.exists(drive_path):
                    items.append(drive_path)
        else:
            # On Unix, add root and common paths
            items.append("/")
            for common in ["/home/", "/mnt/", "/media/"]:
                if os.path.isdir(common.rstrip("/")):
                    items.append(common)
        return web.json_response(items)

    # Normalize path
    path = os.path.abspath(os.path.expanduser(path))

    # Security check
    if not is_safe_path(path):
        return web.json_response([])

    if not os.path.exists(path):
        return web.json_response([])

    if not os.path.isdir(path):
        # If it's a file, list its parent directory
        path = os.path.dirname(path)

    # Parse extensions filter
    extensions = None
    if "extensions" in query:
        extensions = set(
            ext.lower().strip().lstrip('.')
            for ext in query["extensions"].split(',')
            if ext.strip()
        )

    dirs_only = query.get("dirs_only", "").lower() == "true"

    # List directory contents
    items = []
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_dir():
                    items.append(entry.name + os.sep)
                elif not dirs_only:
                    # Check extension filter
                    if extensions:
                        ext = os.path.splitext(entry.name)[1].lower().lstrip('.')
                        if ext not in extensions:
                            continue
                    items.append(entry.name)
            except OSError:
                # Skip entries that can't be accessed
                continue
    except PermissionError:
        return web.json_response([])
    except OSError:
        return web.json_response([])

    # Sort: directories first (alphabetically), then files (alphabetically)
    dirs = sorted([i for i in items if i.endswith(os.sep)], key=str.lower)
    files = sorted([i for i in items if not i.endswith(os.sep)], key=str.lower)

    return web.json_response(dirs + files)


@server.PromptServer.instance.routes.get("/trent/validate_path")
async def validate_path(request):
    """
    Validate that a path exists and is accessible.

    Query parameters:
        path: Path to validate (required)
        type: "dir" or "file" (optional, default checks either)

    Returns:
        JSON with {valid: bool, type: "dir"|"file"|null, count: int}
    """
    query = request.rel_url.query

    if "path" not in query:
        return web.json_response({"valid": False, "type": None, "count": 0})

    path = strip_path(query["path"])
    path = os.path.abspath(os.path.expanduser(path))

    if not is_safe_path(path):
        return web.json_response({"valid": False, "type": None, "count": 0})

    if not os.path.exists(path):
        return web.json_response({"valid": False, "type": None, "count": 0})

    result = {"valid": True, "type": None, "count": 0}

    if os.path.isdir(path):
        result["type"] = "dir"
        # Choose extension set based on media type
        media_type = query.get("media", "image")
        if media_type == "video":
            valid_exts = {
                '.mp4', '.avi', '.mov', '.mkv', '.webm',
                '.flv', '.wmv', '.m4v', '.mpg', '.mpeg',
            }
        else:
            valid_exts = {
                '.png', '.jpg', '.jpeg', '.bmp',
                '.gif', '.webp', '.tiff',
            }
        try:
            count = 0
            for entry in os.scandir(path):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in valid_exts:
                        count += 1
            result["count"] = count
        except OSError:
            pass
    elif os.path.isfile(path):
        result["type"] = "file"
        result["count"] = 1

    expected_type = query.get("type")
    if expected_type and result["type"] != expected_type:
        result["valid"] = False

    return web.json_response(result)


@server.PromptServer.instance.routes.post("/trent/for_next_time/clear")
async def for_next_time_clear(request):
    """
    Clear a Save for Next Time slot from the node's Clear button.

    Body: {"slot_name": str, "include_pinned": bool (optional)}

    Returns {ok, removed, kept_pinned} where removed/kept_pinned
    are entry-name lists.
    """
    from .utils import for_next_time as store

    try:
        data = await request.json()
    except ValueError:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)

    slot_name = data.get("slot_name")
    if not isinstance(slot_name, str) or not slot_name.strip():
        return web.json_response(
            {"ok": False, "error": "slot_name required"}, status=400
        )

    try:
        removed, kept = store.clear_slot(
            slot_name, include_pinned=bool(data.get("include_pinned"))
        )
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)

    return web.json_response(
        {"ok": True, "removed": removed, "kept_pinned": kept}
    )


@server.PromptServer.instance.routes.post("/trent/first_frame")
async def extract_first_frame(request):
    """
    Extract frame 1 of a video into the input folder as a PNG.

    Body: {"video": str} -- a VHS `video` widget value. Either an
    annotated input filename ("clip.mp4", "clip.mp4 [input]") or a
    filesystem path (the *Path node variants).

    Returns {ok, filename} where filename lives in the input folder,
    ready for a LoadImage node.
    """
    import folder_paths

    try:
        data = await request.json()
    except ValueError:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)

    video = data.get("video")
    if not isinstance(video, str) or not video.strip():
        return web.json_response(
            {"ok": False, "error": "video required"}, status=400
        )
    video = strip_path(video)

    # Resolve like VHS does: direct path first (Path nodes), then the
    # annotated-filename form used by the upload nodes.
    path = None
    expanded = os.path.expanduser(video)
    if os.path.isfile(expanded):
        path = os.path.abspath(expanded)
    else:
        try:
            candidate = folder_paths.get_annotated_filepath(video)
        except Exception:  # noqa: BLE001 - any resolve failure means "not found"
            candidate = None
        if candidate and os.path.isfile(candidate):
            path = candidate
    if path is None or not is_safe_path(path):
        return web.json_response(
            {"ok": False, "error": f"video not found: {video}"}, status=404
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            return web.json_response(
                {"ok": False, "error": "ffmpeg not found"}, status=500
            )

    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^\w.-]+", "_", stem) or "video"
    out_name = f"{stem}_firstframe.png"
    out_path = os.path.join(folder_paths.get_input_directory(), out_name)

    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-y", "-i", path, "-frames:v", "1", out_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.isfile(out_path):
        tail = (stderr or b"").decode(errors="replace")[-400:]
        return web.json_response(
            {"ok": False, "error": f"ffmpeg failed: {tail}"}, status=500
        )

    return web.json_response({"ok": True, "filename": out_name})


@server.PromptServer.instance.routes.post("/trent/text_holdup")
async def text_holdup_submit(request):
    """
    Release a TextHoldupCowboy node that is waiting mid-run.

    Body: {"gate_id": str, "text": str}
    """
    from .nodes.text_holdup_cowboy import resolve_gate

    try:
        data = await request.json()
    except ValueError:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)

    gate_id = data.get("gate_id")
    text = data.get("text")
    if not gate_id or not isinstance(text, str):
        return web.json_response(
            {"ok": False, "error": "gate_id and text required"}, status=400
        )

    ok = resolve_gate(gate_id, text)
    if not ok:
        return web.json_response(
            {"ok": False, "error": "gate not found (already released?)"},
            status=404,
        )
    return web.json_response({"ok": True})
