"""
WSL path converter.

ComfyUI runs inside WSL, so nodes that take a path string (VHS
"Load Video (Path)", Load Image From Path, save nodes) cannot read a
Windows path. Explorer hands out
``\\\\wsl.localhost\\Ubuntu-24.04\\home\\trent\\...`` and Windows apps hand
out ``C:\\Users\\Trent\\...``. This node rewrites either form into the
Linux path the running process can actually open, and back again.
"""
import os
import re
from urllib.parse import unquote, urlparse

# UNC hosts Windows uses to reach the WSL filesystem. The component after
# the host is the distro name, which is meaningless inside the distro.
_WSL_UNC_HOSTS = ("wsl.localhost", "wsl$", "wsl")

_DEFAULT_AUTOMOUNT_ROOT = "/mnt"


def automount_root() -> str:
    """
    Return the directory where WSL mounts Windows drives.

    Defaults to /mnt but /etc/wsl.conf can move it. Read without a config
    parser: the file is tiny and a parser failure must not break a path
    conversion.
    """
    try:
        with open("/etc/wsl.conf", "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return _DEFAULT_AUTOMOUNT_ROOT

    in_automount = False
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_automount = line.lower().startswith("[automount")
            continue
        if not in_automount or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().lower() == "root":
            root = value.strip().strip('"').strip("'").replace("\\", "/")
            root = root.rstrip("/")
            return root or "/"
    return _DEFAULT_AUTOMOUNT_ROOT


def distro_name() -> str:
    """Return the WSL distro name used in \\\\wsl.localhost UNC paths."""
    return os.environ.get("WSL_DISTRO_NAME") or "Ubuntu"


def _clean(raw: str) -> str:
    """Strip quotes, whitespace and any file:// wrapper from a raw path."""
    text = (raw or "").strip()
    # Explorer's "Copy as path" wraps the result in double quotes.
    for quote in ('"', "'"):
        if len(text) >= 2 and text.startswith(quote) and text.endswith(quote):
            text = text[1:-1].strip()
            break
    if text[:5].lower() == "file:":
        parsed = urlparse(text)
        host = parsed.netloc
        path = unquote(parsed.path)
        # file://wsl.localhost/... keeps the host outside the path.
        text = f"//{host}{path}" if host else path
        # file:///C:/x gives /C:/x — drop the leading slash before the drive.
        if re.match(r"^/[A-Za-z]:", text):
            text = text[1:]
    return text


def _squash(path: str) -> str:
    """Collapse repeated slashes and drop a trailing slash."""
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def to_linux_path(raw: str, root: str = None) -> str:
    """
    Convert a Windows, UNC or WSL path to the Linux path WSL can open.

    Already-Linux paths pass through with slashes normalized. A UNC path
    to a real network share has no Linux equivalent, so it is returned
    unchanged apart from slash direction.
    """
    text = _clean(raw)
    if not text:
        return ""

    root = automount_root() if root is None else root
    text = text.replace("\\", "/")

    # \\wsl.localhost\<distro>\home\trent -> /home/trent
    if text.startswith("//"):
        parts = [p for p in text[2:].split("/") if p]
        if parts and parts[0].lower() in _WSL_UNC_HOSTS:
            return _squash("/" + "/".join(parts[2:]))
        # A real network share: keep the leading // that _squash would eat.
        return "//" + _squash("/".join(parts))

    # C:\Users\Trent -> /mnt/c/Users/Trent
    drive = re.match(r"^([A-Za-z]):(/.*|)$", text)
    if drive:
        letter = drive.group(1).lower()
        return _squash(f"{root}/{letter}{drive.group(2)}")

    return _squash(text)


def to_windows_path(raw: str, root: str = None, distro: str = None) -> str:
    """
    Convert a Linux path back to the Windows form Explorer understands.

    Paths under the automount root map to their real drive letter.
    Everything else becomes a \\\\wsl.localhost\\<distro>\\... UNC path.
    """
    text = _clean(raw)
    if not text:
        return ""

    root = automount_root() if root is None else root
    distro = distro_name() if distro is None else distro

    # Already a Windows path: only fix the slashes.
    if re.match(r"^([A-Za-z]):([\\/].*|)$", text) or text.startswith("\\\\"):
        return text.replace("/", "\\")
    if text.startswith("//"):
        return text.replace("/", "\\")

    text = _squash(text.replace("\\", "/"))

    mounted = re.match(rf"^{re.escape(root)}/([A-Za-z])(/.*|)$", text)
    if mounted:
        tail = mounted.group(2).replace("/", "\\") or "\\"
        return f"{mounted.group(1).upper()}:{tail}"

    if text.startswith("/"):
        return f"\\\\wsl.localhost\\{distro}" + text.replace("/", "\\")

    return text.replace("/", "\\")


class WSLPathConverter:
    """Rewrite a pasted Windows/WSL path into a usable path string."""

    DISPLAY_NAME = "WSL Path Converter"
    CATEGORY = "Trent/Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": (
                        "\\\\wsl.localhost\\Ubuntu-24.04\\home\\... "
                        "or C:\\Users\\..."
                    ),
                }),
                "direction": (
                    ["auto", "to_linux", "to_windows"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "append": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "optional file name to join on",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("path", "exists", "kind")
    FUNCTION = "convert"

    @classmethod
    def IS_CHANGED(cls, path, direction, append=""):
        # The target can appear or disappear between runs, so never cache.
        return float("nan")

    def convert(self, path, direction="auto", append=""):
        if direction == "auto":
            # Give each host the form it can actually open.
            direction = "to_windows" if os.name == "nt" else "to_linux"

        if direction == "to_windows":
            result = to_windows_path(path)
            separator = "\\"
        else:
            result = to_linux_path(path)
            separator = "/"

        extra = _clean(append).strip("\\/")
        if extra:
            extra = extra.replace("\\", separator).replace("/", separator)
            result = f"{result.rstrip(separator)}{separator}{extra}"

        # Only a Linux path is meaningful to os.path inside WSL.
        probe = result if separator == "/" else to_linux_path(result)
        try:
            if os.path.isdir(probe):
                kind = "dir"
            elif os.path.isfile(probe):
                kind = "file"
            else:
                kind = "missing"
        except OSError:
            kind = "missing"

        return (result, kind != "missing", kind)


NODE_CLASS_MAPPINGS = {
    "WSLPathConverter": WSLPathConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WSLPathConverter": "WSL Path Converter",
}
