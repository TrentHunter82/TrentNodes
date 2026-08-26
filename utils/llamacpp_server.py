"""
Managed llama-server (llama.cpp) lifecycle for local GGUF LLM nodes.

One slot PER PORT: the H3 Skill Promptor's text VLM (default 8735) and
the H3 Audio Soundscaper's omni model (default 8736) coexist. Within a
port, reuse happens on ServerSpec equality; a different spec stops that
port's process and spawns a new one. Process death releases its VRAM,
which is the crash-isolation point of running the LLM out of process -
comfy.model_management cannot see llama.cpp allocations either way.

No comfy/torch/folder_paths imports at module level: the module must
import from the offline dev CLI and from pytest. Anything ComfyUI- or
CUDA-specific is imported lazily inside the function that needs it.

Locking: _lock guards the managed-slot STATE. The startup health poll
runs OUTSIDE the lock so stop_server() (the ComfyUI Cancel path and the
atexit handler) is never blocked behind a cold model load; the poll
also honors ComfyUI's processing interrupt.

Test seams: `_popen` and `_http_get` are module attributes so tests can
monkeypatch process creation and health probing without sockets.
"""

import atexit
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Tuple

DEFAULT_CTX = 32768
DEFAULT_NGL = 99
DEFAULT_PORT = 8735  # free on this box; avoids 8188 (ComfyUI), 8642 (Hermes), 8080, 11434
DEFAULT_REASONING_EFFORT = "low"

# Unsloth instruct-mode sampling for Qwen3.8. The GGUF-embedded defaults
# are thinking-mode (temp 1.0 / top-p 0.95) - launch flags override them
# so a bare curl behaves; the node still sends sampling per request.
DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.80
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_PRESENCE_PENALTY = 1.5

HEALTH_POLL_S = 1.0
HEALTH_TIMEOUT_S = 300.0
HEALTH_PROGRESS_EVERY_S = 15.0
STOP_GRACE_S = 10.0

# Spawn headroom beyond the weight files: KV cache, compute buffers,
# vision-encode transients, CUDA context.
SPAWN_VRAM_OVERHEAD_BYTES = 4 * (1024**3)

_INSTALL_HINT = (
    "llama-server was not found. Build a current CUDA llama.cpp "
    "(>= b10450 for the qwen35 arch) and either put llama-server on "
    "PATH, set the LLAMA_SERVER_BIN environment variable, or keep the "
    "default location ~/llama.cpp/build/bin/llama-server."
)


@dataclass(frozen=True)
class ServerSpec:
    model_path: str
    mmproj_path: Optional[str] = None
    ctx_size: int = DEFAULT_CTX
    n_gpu_layers: int = DEFAULT_NGL
    port: int = DEFAULT_PORT
    # Launch-time default only; every request overrides it via
    # chat_template_kwargs, so it must NOT force a respawn on change.
    reasoning_effort: str = field(default=DEFAULT_REASONING_EFFORT, compare=False)
    extra_args: str = ""


@dataclass
class ServerHandle:
    base_url: str
    spec: Optional[ServerSpec]  # None = attached to a server we did not spawn
    log_path: Optional[str] = None
    alias: Optional[str] = None    # model id the server reports, when known
    vision: Optional[bool] = None  # True/False when known, None = unknown


class _Slot:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.spec: Optional[ServerSpec] = None
        self.log_path: Optional[str] = None
        self.log_file = None


_slots: dict = {}  # port -> _Slot
_lock = threading.RLock()


# --------------------------------------------------------------------
# test seams

def _popen(cmd, **kwargs):
    return subprocess.Popen(cmd, **kwargs)


def _http_get(url: str, timeout: float = 3.0):
    """Return (status_code, body_str). Raises OSError family on no-listener."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:  # 4xx/5xx still means a listener
        return exc.code, exc.read().decode("utf-8", "replace")


# --------------------------------------------------------------------
# discovery helpers

def find_llama_server(explicit: str = "") -> str:
    """Resolve the llama-server binary; RuntimeError with install hint."""
    candidates = []
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    env = os.environ.get("LLAMA_SERVER_BIN", "")
    if env:
        candidates.append(os.path.expanduser(env))
    which = shutil.which("llama-server")
    if which:
        candidates.append(which)
    candidates.append(
        os.path.join(os.path.expanduser("~"), "llama.cpp", "build", "bin", "llama-server")
    )
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise RuntimeError(_INSTALL_HINT)


def _stem(path: str) -> str:
    """Basename minus a literal .gguf suffix. NEVER generic splitext:
    model names carry interior dots ('Qwen3.8-27B-...'), and splitext
    would truncate an extensionless alias at the first version dot -
    which once made a server reject its own identical model."""
    name = os.path.basename(path)
    return name[:-5] if name.lower().endswith(".gguf") else name


def _same_model_alias(alias: Optional[str], model_path: str) -> bool:
    """Loose compare: hand-started servers default the alias to the model
    path or filename with extension; normalize both sides to the stem."""
    if not alias:
        return False
    return _stem(alias) == _stem(model_path)


def find_mmproj_for(model_path: str, candidates: Optional[list] = None) -> Optional[str]:
    """
    Pair a model gguf with its mmproj by filename. Handles both naming
    conventions: '<Model>-mmproj-F16.gguf' (suffix) and
    'mmproj-<Model>-F16.gguf' (the llama.cpp/Unsloth prefix form). When
    scoring fails and exactly ONE mmproj sits next to the model, that
    one wins.
    """
    directory = os.path.dirname(os.path.abspath(model_path))
    stem = _stem(model_path)
    if candidates is None:
        try:
            candidates = [
                os.path.join(directory, name)
                for name in os.listdir(directory)
                if name.lower().endswith(".gguf") and "mmproj" in name.lower()
            ]
        except OSError:
            return None
    best, best_len = None, 0
    for candidate in candidates:
        other = _stem(candidate)
        stripped = re.sub(r"^mmproj[-_.]*", "", other, flags=re.I)
        score = max(
            len(os.path.commonprefix([stem, other])),
            len(os.path.commonprefix([stem, stripped])),
        )
        # require a meaningful shared prefix, not just "Q" or ""
        if score > max(best_len, 4):
            best, best_len = candidate, score
    if best is None and len(candidates) == 1:
        return candidates[0]
    return best


def default_log_path(port: int) -> str:
    return os.path.join(tempfile.gettempdir(), f"trentnodes-llamacpp-{port}.log")


def build_command(binary: str, spec: ServerSpec) -> list:
    """Pure command assembly - unit-testable without spawning anything."""
    cmd = [
        binary,
        "-m", spec.model_path,
        "--host", "127.0.0.1",
        "--port", str(spec.port),
        "-ngl", str(spec.n_gpu_layers),
        "-c", str(spec.ctx_size),
        "-np", "1",
        "-fa", "on",
        "--jinja",
    ]
    if spec.reasoning_effort:
        # Qwen3.8-style templates take this; an empty string omits the
        # flag for families whose template has no such variable (omni).
        cmd += [
            "--chat-template-kwargs",
            json.dumps({"reasoning_effort": spec.reasoning_effort}),
        ]
    cmd += [
        "--temp", str(DEFAULT_TEMP),
        "--top-p", str(DEFAULT_TOP_P),
        "--top-k", str(DEFAULT_TOP_K),
        "--min-p", str(DEFAULT_MIN_P),
        "--presence-penalty", str(DEFAULT_PRESENCE_PENALTY),
        "--no-webui",
        "-a", _stem(spec.model_path),
    ]
    if spec.mmproj_path:
        cmd += ["--mmproj", spec.mmproj_path]
    if spec.extra_args:
        cmd += shlex.split(spec.extra_args)
    return cmd


# --------------------------------------------------------------------
# health / attach

def _health(base_root: str, timeout: float = 3.0):
    """Return "ok", "loading", or "down" for http://host:port (no /v1)."""
    try:
        status, _ = _http_get(base_root + "/health", timeout=timeout)
    except (OSError, ValueError):
        return "down"
    if status == 200:
        return "ok"
    if status == 503:
        return "loading"
    return "down"


def served_model_info(base_root: str, timeout: float = 3.0) -> Tuple[Optional[str], Optional[bool]]:
    """(alias, vision) from /v1/models. vision is None when the server
    does not report capabilities (non-llama-server endpoints)."""
    try:
        status, body = _http_get(base_root + "/v1/models", timeout=timeout)
        if status != 200:
            return None, None
        data = json.loads(body)
    except (OSError, ValueError):
        return None, None
    alias = None
    entries = data.get("data") or []
    if entries:
        alias = entries[0].get("id")
    vision = None
    # llama-server also returns a "models" list carrying capabilities
    for entry in data.get("models") or []:
        capabilities = entry.get("capabilities")
        if isinstance(capabilities, list):
            vision = "multimodal" in capabilities
            break
    return alias, vision


def served_model_alias(base_root: str, timeout: float = 3.0) -> Optional[str]:
    return served_model_info(base_root, timeout=timeout)[0]


def attach(base_url: str, expect_alias: str = "", timeout: float = 5.0) -> ServerHandle:
    """
    Attach to an already-running OpenAI-compatible server. base_url may
    be given with or without the /v1 suffix. Servers without a root
    /health endpoint (LM Studio, vLLM) are probed via /v1/models.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    state = _health(root, timeout=timeout)
    if state == "loading":
        raise RuntimeError(
            f"The server at {root} is still loading its model. Retry shortly."
        )
    alias, vision = served_model_info(root, timeout=timeout)
    if state != "ok" and alias is None:
        raise RuntimeError(
            f"No healthy server answered at {root} (tried /health and "
            f"/v1/models). Start one, or leave base_url empty to let the "
            f"node spawn a managed server."
        )
    if expect_alias and alias and not _same_model_alias(alias, expect_alias):
        raise RuntimeError(
            f"The server at {root} serves '{alias}', not the requested "
            f"'{expect_alias}'. Stop it, use another port, or clear the "
            f"model selection to use what it serves."
        )
    return ServerHandle(
        base_url=root + "/v1", spec=None, log_path=None,
        alias=alias, vision=vision,
    )


# --------------------------------------------------------------------
# VRAM gate

def _free_vram_bytes() -> Optional[int]:
    try:
        import torch  # lazy: keep module importable without torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def _required_vram_bytes(spec: ServerSpec) -> int:
    """Weights (as stored, Q-quant mmap ~= VRAM use) + fixed headroom."""
    total = SPAWN_VRAM_OVERHEAD_BYTES
    for path in (spec.model_path, spec.mmproj_path):
        if path:
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total


def _unload_comfy_models() -> None:
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass  # dev CLI / tests: nothing to unload


def _check_interrupted() -> None:
    """Let ComfyUI's Cancel button abort a cold load; no-op elsewhere."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


# --------------------------------------------------------------------
# lifecycle

def _slot_alive(slot: Optional[_Slot]) -> bool:
    return (slot is not None and slot.proc is not None
            and slot.proc.poll() is None)


def _log_tail(path: Optional[str], lines: int = 25) -> str:
    if not path or not os.path.exists(path):
        return "(no log file)"
    try:
        with open(path, "r", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return "(log unreadable)"


def _stop_slot(slot: _Slot) -> bool:
    proc = slot.proc
    if proc is None:
        return False
    stopped = False
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=STOP_GRACE_S)
            stopped = True
        except OSError:
            pass
    if slot.log_file is not None:
        try:
            slot.log_file.close()
        except OSError:
            pass
    slot.proc = None
    slot.spec = None
    slot.log_file = None
    return stopped


def stop_server(port: Optional[int] = None) -> bool:
    """SIGTERM managed server(s) (llama-server handles it gracefully),
    escalate to SIGKILL after STOP_GRACE_S. port=None stops EVERY slot.
    True if something was stopped."""
    with _lock:
        if port is not None:
            slot = _slots.get(port)
            return _stop_slot(slot) if slot else False
        stopped = False
        for slot in _slots.values():
            stopped = _stop_slot(slot) or stopped
        return stopped


def stop_orphan(port: int = DEFAULT_PORT) -> bool:
    """
    Stop a llama-server on `port` that this process did NOT spawn - the
    leftover of a hard ComfyUI crash (atexit never ran). Matches only
    the llama-server binary with this exact port on its command line.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", rf"llama-server\b.*--port {port}\b"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    stopped = False
    for token in result.stdout.split():
        try:
            os.kill(int(token), signal.SIGTERM)
            stopped = True
        except (OSError, ValueError):
            pass
    return stopped


atexit.register(stop_server)


def touch() -> None:
    """Idle-clock hook. The resident policy makes this a no-op today;
    kept so callers do not change if an idle timeout returns later."""


def status() -> dict:
    with _lock:
        report = {}
        for port, slot in _slots.items():
            alive = _slot_alive(slot)
            report[port] = {
                "running": alive,
                "pid": slot.proc.pid if alive else None,
                "spec": slot.spec.__dict__ if (alive and slot.spec) else None,
                "log_path": slot.log_path,
            }
        return report


def _wait_healthy(proc, root: str, log_path: str, timeout_s: float) -> None:
    """Poll /health until ready. Runs WITHOUT the lock. Raises on exit,
    timeout, or a ComfyUI cancel."""
    started = time.monotonic()
    last_note = started
    while True:
        _check_interrupted()
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup "
                f"(code {proc.returncode}). Log tail ({log_path}):\n"
                f"{_log_tail(log_path)}"
            )
        if _health(root) == "ok":
            return
        now = time.monotonic()
        if now - started > timeout_s:
            raise RuntimeError(
                f"llama-server did not become healthy within "
                f"{int(timeout_s)} s. Log tail ({log_path}):\n"
                f"{_log_tail(log_path)}"
            )
        if now - last_note >= HEALTH_PROGRESS_EVERY_S:
            print(
                f"[TrentNodes llamacpp] loading model... "
                f"{int(now - started)} s"
            )
            last_note = now
        time.sleep(HEALTH_POLL_S)


def ensure_server(
    spec: ServerSpec,
    free_vram_first: bool = False,
    binary: str = "",
    health_timeout_s: float = HEALTH_TIMEOUT_S,
) -> ServerHandle:
    """
    Return a handle to a healthy server matching `spec`, reusing the
    managed one when the spec is unchanged, else stop-and-respawn.
    """
    root = f"http://127.0.0.1:{spec.port}"
    handle = ServerHandle(
        root + "/v1", spec, default_log_path(spec.port),
        alias=_stem(spec.model_path), vision=bool(spec.mmproj_path),
    )

    with _lock:
        slot = _slots.setdefault(spec.port, _Slot())
        if _slot_alive(slot) and slot.spec == spec:
            if _health(root) == "ok":
                return handle
            # process alive but unhealthy: fall through to restart
        if _slot_alive(slot):
            _stop_slot(slot)

        # Someone else's server on the requested port?
        state = _health(root)
        if state in ("ok", "loading"):
            alias, vision = served_model_info(root) if state == "ok" else (None, None)
            if _same_model_alias(alias, spec.model_path):
                return ServerHandle(root + "/v1", None, None,
                                    alias=alias, vision=vision)
            raise RuntimeError(
                f"Port {spec.port} is already serving "
                f"'{alias or 'a still-loading model'}', not "
                f"'{_stem(spec.model_path)}'. Stop that server or pick "
                f"another port."
            )

        resolved_binary = find_llama_server(binary)

        if not os.path.isfile(spec.model_path):
            raise RuntimeError(f"Model file not found: {spec.model_path}")
        if spec.mmproj_path and not os.path.isfile(spec.mmproj_path):
            raise RuntimeError(f"mmproj file not found: {spec.mmproj_path}")
        if spec.model_path.startswith("/mnt/"):
            print(
                "[TrentNodes llamacpp] WARNING: model is on a Windows drvfs "
                "mount; cold loads are slow. Copy it to ~/ComfyUI/models/LLM."
            )

        if free_vram_first:
            _unload_comfy_models()
        free = _free_vram_bytes()
        required = _required_vram_bytes(spec)
        if free is not None and free < required:
            raise RuntimeError(
                f"Only {free / 1024**3:.1f} GiB of VRAM is free; this model "
                f"needs ~{required / 1024**3:.1f} GiB. Enable "
                f"free_vram_first, or free VRAM and retry."
            )

        log_path = default_log_path(spec.port)
        if os.path.exists(log_path):  # keep the previous run's evidence
            try:
                os.replace(log_path, log_path + ".prev")
            except OSError:
                pass
        log_file = open(log_path, "w")  # noqa: SIM115 - handed to Popen
        cmd = build_command(resolved_binary, spec)
        try:
            proc = _popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            log_file.close()
            raise RuntimeError(f"Failed to launch llama-server: {exc}") from exc

        slot.proc = proc
        slot.spec = spec
        slot.log_path = log_path
        slot.log_file = log_file

    # Poll OUTSIDE the lock: stop_server()/atexit stay responsive during
    # the (possibly minutes-long) cold load, and Cancel can abort it.
    try:
        _wait_healthy(proc, root, log_path, health_timeout_s)
    except BaseException:
        stop_server(port=spec.port)
        raise
    return handle
