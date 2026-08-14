"""
Unit tests for the WSL path converter. No torch and no ComfyUI needed.
Run from the ComfyUI root:

    cd /home/trent/ComfyUI && \
        venv/bin/python custom_nodes/TrentNodes/tests/test_wsl_path.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "..", "nodes", "wsl_path.py")

spec = importlib.util.spec_from_file_location("wsl_path", MODULE)
wsl_path = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wsl_path)

to_linux = wsl_path.to_linux_path
to_windows = wsl_path.to_windows_path

FAILURES = []


def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}\n    got:  {got!r}\n    want: {want!r}")


# --- Windows / UNC -> Linux ------------------------------------------------
check(
    "wsl.localhost UNC",
    to_linux(r"\\wsl.localhost\Ubuntu-24.04\home\trent\ComfyUI\output"),
    "/home/trent/ComfyUI/output",
)
check(
    "wsl$ legacy UNC",
    to_linux(r"\\wsl$\Ubuntu\home\trent\a.mp4"),
    "/home/trent/a.mp4",
)
check(
    "UNC root only",
    to_linux(r"\\wsl.localhost\Ubuntu-24.04"),
    "/",
)
check(
    "drive letter",
    to_linux(r"C:\Users\Trent\Downloads\clip.mp4", root="/mnt"),
    "/mnt/c/Users/Trent/Downloads/clip.mp4",
)
check("bare drive", to_linux("D:", root="/mnt"), "/mnt/d")
check(
    "forward-slash drive",
    to_linux("D:/models/wan", root="/mnt"),
    "/mnt/d/models/wan",
)
check(
    "custom automount root",
    to_linux(r"C:\x", root="/windows"),
    "/windows/c/x",
)
check(
    "quoted copy-as-path",
    to_linux('"\\\\wsl.localhost\\Ubuntu-24.04\\home\\trent\\a b.mp4"'),
    "/home/trent/a b.mp4",
)
check(
    "file:// URI with percent escapes",
    to_linux("file://wsl.localhost/Ubuntu-24.04/home/trent/a%20b.mp4"),
    "/home/trent/a b.mp4",
)
check(
    "file:/// drive URI",
    to_linux("file:///C:/Users/Trent/a.mp4", root="/mnt"),
    "/mnt/c/Users/Trent/a.mp4",
)
check(
    "trailing slash and dupes",
    to_linux(r"\\wsl.localhost\Ubuntu-24.04\home\\trent\\"),
    "/home/trent",
)
check("empty stays empty", to_linux("   "), "")

# --- Linux passthrough -----------------------------------------------------
check(
    "linux path unchanged",
    to_linux("/home/trent/ComfyUI/output/H3-hybrid/033_00031-audio.mp4"),
    "/home/trent/ComfyUI/output/H3-hybrid/033_00031-audio.mp4",
)
check("relative path unchanged", to_linux("output/H3-hybrid"), "output/H3-hybrid")

# --- Real network share is not a WSL path ----------------------------------
check(
    "network share kept",
    to_linux(r"\\nas\media\clip.mp4"),
    "//nas/media/clip.mp4",
)

# --- Linux -> Windows ------------------------------------------------------
check(
    "mounted drive back",
    to_windows("/mnt/c/Users/Trent/a.mp4", root="/mnt"),
    r"C:\Users\Trent\a.mp4",
)
check("drive root back", to_windows("/mnt/d", root="/mnt"), "D:\\")
check(
    "distro UNC back",
    to_windows("/home/trent/ComfyUI/output", root="/mnt", distro="Ubuntu-24.04"),
    r"\\wsl.localhost\Ubuntu-24.04\home\trent\ComfyUI\output",
)
check(
    "windows input untouched",
    to_windows(r"C:\Users\Trent", root="/mnt"),
    r"C:\Users\Trent",
)

# --- Round trip ------------------------------------------------------------
for original in ("/home/trent/ComfyUI/output", "/mnt/c/Users/Trent/a.mp4"):
    check(
        f"round trip {original}",
        to_linux(to_windows(original, root="/mnt", distro="Ubuntu-24.04"),
                 root="/mnt"),
        original,
    )

# --- Node behaviour --------------------------------------------------------
node = wsl_path.WSLPathConverter()
target = os.path.abspath(os.path.join(HERE, "..", "nodes"))

path, exists, kind = node.convert(target, "to_linux")
check("node reports dir", (exists, kind), (True, "dir"))

path, exists, kind = node.convert(target, "to_linux", append="wsl_path.py")
check("node append joins", path, os.path.join(target, "wsl_path.py"))
check("node append is a file", (exists, kind), (True, "file"))

path, exists, kind = node.convert("/no/such/place", "to_linux")
check("missing target", (exists, kind), (False, "missing"))

path, _, _ = node.convert(
    r"\\wsl.localhost\Ubuntu-24.04\home\trent", "to_linux", append="\\a\\b.mp4"
)
check("append normalizes separators", path, "/home/trent/a/b.mp4")

if FAILURES:
    print(f"FAIL: {len(FAILURES)} case(s)")
    for item in FAILURES:
        print("  " + item)
    sys.exit(1)

print("OK: all wsl_path cases passed")
