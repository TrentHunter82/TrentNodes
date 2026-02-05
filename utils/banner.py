"""
Terminal banner and colored output utilities for TrentNodes.
"""
import os
import sys

# ANSI color codes
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"

# Rainbow palette for banner (ANSI 256 colors)
PALETTE = [196, 202, 208, 214, 220, 190, 154, 118, 82, 46, 51, 39, 27, 21]

# ASCII art banner
BANNER = r"""
 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


        T  R  E  N  T  N  O  D  E  S


 ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
""".rstrip("\n")


def setup_terminal():
    """
    Configure terminal for proper output.

    - Enables ANSI colors on Windows
    - Sets UTF-8 encoding for proper character display
    """
    # Windows ANSI support (safe no-op elsewhere)
    if os.name == "nt":
        try:
            import colorama
            colorama.just_fix_windows_console()
        except Exception:
            pass

    # Make sure block characters render on Windows terminals
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def ansi256(color_code: int) -> str:
    """
    Get ANSI 256-color escape sequence.

    Args:
        color_code: Color code (0-255)

    Returns:
        ANSI escape sequence string
    """
    return f"\033[38;5;{color_code}m"


def print_banner(use_color: bool = True):
    """
    Print the TrentNodes ASCII art banner.

    Args:
        use_color: Whether to use rainbow colors
    """
    for i, line in enumerate(BANNER.splitlines()):
        if not line:
            sys.stdout.write("\n")
            continue
        if use_color:
            color = ansi256(PALETTE[i % len(PALETTE)])
            sys.stdout.write(color + line + RESET + "\n")
        else:
            sys.stdout.write(line + "\n")
    sys.stdout.flush()


def print_status(ok: bool, msg: str, details: str = None):
    """
    Print a status message with color coding.

    Args:
        ok: True for success (green), False for error (red)
        msg: Main status message
        details: Optional additional details (printed in yellow)
    """
    if os.environ.get("TRENT_NODES_STATUS", "1") == "0":
        return

    color = GREEN if ok else RED
    prefix = "[OK] " if ok else "[ERROR] "
    sys.stdout.write(color + prefix + msg + RESET + "\n")

    if details:
        sys.stdout.write(YELLOW + details + RESET + "\n")

    sys.stdout.flush()


def should_show_banner() -> bool:
    """Check if banner should be displayed based on environment."""
    return os.environ.get("TRENT_NODES_BANNER", "1") != "0"


def should_use_color() -> bool:
    """Check if colors should be used based on environment."""
    return os.environ.get("TRENT_NODES_COLOR", "1") != "0"
