"""
@author: Trent
@title: Trent's ComfyUI Nodes
@nickname: Trent Nodes
@description: Custom nodes for video processing, keyframe management, scene detection, and video analysis
"""

# --- Trent Nodes discovery ----------------------------------------------------
import importlib, pkgutil, inspect, re

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _pretty_name(name: str) -> str:
    # "HelloImageNode" -> "Hello Image Node"
    return re.sub(r"(?<!^)([A-Z])", r" \1", name).strip()

def _discover_nodes():
    # Import the "nodes" subpackage (must have nodes/__init__.py)
    try:
        from . import nodes as _nodes_pkg
    except Exception as e:
        print(f"[TrentNodes] Could not import 'nodes' package: {e}")
        return

    prefix = f"{__name__}.nodes."
    for _finder, modname, _ispkg in pkgutil.walk_packages(_nodes_pkg.__path__, prefix=prefix):
        try:
            module = importlib.import_module(modname)
        except Exception as e:
            print(f"[TrentNodes] Import error in {modname}: {e}")
            continue

        # Preferred: module provides explicit mappings
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))
            continue

        # Fallback: auto-register classes that look like ComfyUI nodes
        for attr, obj in vars(module).items():
            if inspect.isclass(obj) and all(hasattr(obj, x) for x in ("INPUT_TYPES", "RETURN_TYPES", "FUNCTION")):
                NODE_CLASS_MAPPINGS[attr] = obj
                # Ensure a category so it groups under your folder
                if not hasattr(obj, "CATEGORY"):
                    setattr(obj, "CATEGORY", "Trent Tools/Auto")
                # Nice display name if none provided
                NODE_DISPLAY_NAME_MAPPINGS.setdefault(attr, getattr(obj, "DISPLAY_NAME", _pretty_name(attr)))

# call discovery immediately so mappings are populated for the self-test
_discover_nodes()
# ----------------------------------------------------------------------------- 


# --- Trent Nodes: banner + self-test (append after discovery) ----------------
import os, sys, time, traceback

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

RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
PALETTE = [196,202,208,214,220,190,154,118,82,46,51,39,27,21]
def _ansi256(c): return f"\033[38;5;{c}m"

# Your left-aligned banner art
BANNER = r"""
                                    )                       
  *   )                     )   ( /(        (              
` )  /( (      (         ( /(   )\())       )\ )   (       
 ( )(_)))(    ))\  (     )\()) ((_)\   (   (()/(  ))\ (    
(_(_())(()\  /((_) )\ ) (_))/   _((_)  )\   ((_))/((_))\   
|_   _| ((_)(_))  _(_/( | |_   | \| | ((_)  _| |(_)) ((_)  
  | |  | '_|/ -_)| ' \))|  _|  | .` |/ _ \/ _` |/ -_)(_-<  
  |_|  |_|  \___||_||_|  \__|  |_|\_|\___/\__,_|\___|/__/  
                                                                                                                                                                                                                                                                                
""".rstrip("\n")

def _print_banner(use_color=True):
    for i, line in enumerate(BANNER.splitlines()):
        if not line:
            sys.stdout.write("\n"); continue
        if use_color:
            sys.stdout.write(_ansi256(PALETTE[i % len(PALETTE)]) + line + RESET + "\n")
        else:
            sys.stdout.write(line + "\n")
    sys.stdout.flush()

def _validate_node_class(cls):
    problems = []
    if not hasattr(cls, "FUNCTION"): problems.append("missing FUNCTION")
    if not hasattr(cls, "RETURN_TYPES"): problems.append("missing RETURN_TYPES")
    if not hasattr(cls, "INPUT_TYPES"): problems.append("missing INPUT_TYPES()")
    else:
        try:
            it = cls.INPUT_TYPES()
            if not isinstance(it, dict):
                problems.append("INPUT_TYPES() did not return a dict")
        except Exception as e:
            problems.append(f"INPUT_TYPES() error: {e}")
    if not hasattr(cls, "CATEGORY"):
        problems.append("missing CATEGORY (node won't show in a folder)")
    return problems

def _print_status(ok, msg, details=None):
    if os.environ.get("TRENT_NODES_STATUS", "1") == "0":
        return
    sys.stdout.write((GREEN if ok else RED) + ("✅ " if ok else "❌ ") + msg + RESET + "\n")
    if details:
        sys.stdout.write(YELLOW + details + RESET + "\n")
    sys.stdout.flush()

def _self_test():
    start = time.perf_counter()
    try:
        mappings = globals().get("NODE_CLASS_MAPPINGS", {})
        display = globals().get("NODE_DISPLAY_NAME_MAPPINGS", {})
        if not mappings:
            raise RuntimeError("No nodes registered in NODE_CLASS_MAPPINGS")
        issues = []
        for name, cls in mappings.items():
            probs = _validate_node_class(cls)
            if probs:
                issues.append(f"{name}: " + ", ".join(probs))
        unknown_display = [k for k in display.keys() if k not in mappings]
        if unknown_display:
            issues.append("DISPLAY_NAME keys without classes: " + ", ".join(unknown_display))
        elapsed = int((time.perf_counter() - start) * 1000)
        if not issues:
            _print_status(True, f"Trent Nodes loaded OK — {len(mappings)} node(s) ready ({elapsed} ms).")
        else:
            msg = f"Trent Nodes loaded with {len(issues)} issue(s) ({elapsed} ms)."
            _print_status(False, msg, details="\n - " + "\n - ".join(issues))
            if os.environ.get("TRENT_STRICT", "0") == "1":
                raise RuntimeError(msg + "\n" + "\n".join(issues))
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        _print_status(False, f"Trent Nodes failed to initialize: {e}", details=tb)
        if os.environ.get("TRENT_STRICT", "0") == "1":
            raise

# Print once per interpreter session
if not globals().get("_TRENT_INIT_RAN"):
    globals()["_TRENT_INIT_RAN"] = True
    if os.environ.get("TRENT_NODES_BANNER", "1") != "0":
        _print_banner(use_color=os.environ.get("TRENT_NODES_COLOR", "1") != "0")
    _self_test()

# WEB_DIRECTORY for serving custom JavaScript files
WEB_DIRECTORY = "./js"

# Export everything needed
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
# -----------------------------------------------------------------------------
