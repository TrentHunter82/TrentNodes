"""
Server routes for TrentNodes.

Provides API endpoints for folder browsing and file listing.
"""

import os

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
        # Count image files
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff'}
        try:
            count = 0
            for entry in os.scandir(path):
                if entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in image_exts:
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
