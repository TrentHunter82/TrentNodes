"""
Unit tests for utils/llamacpp_server.py - pure logic only.

Process creation and HTTP probing are monkeypatched through the module's
_popen/_http_get seams, so nothing here opens a socket or spawns a
binary. Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_llamacpp_server.py
or  venv/bin/pytest custom_nodes/TrentNodes/tests/test_llamacpp_server.py
"""

import os
import sys
import tempfile
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

if "TrentNodes" not in sys.modules:
    pkg = types.ModuleType("TrentNodes")
    pkg.__path__ = [PKG]
    sys.modules["TrentNodes"] = pkg
    for sub in ("utils",):
        m = types.ModuleType(f"TrentNodes.{sub}")
        m.__path__ = [os.path.join(PKG, *sub.split("."))]
        sys.modules[f"TrentNodes.{sub}"] = m

from TrentNodes.utils import llamacpp_server as srv  # noqa: E402


class FakeProc:
    def __init__(self):
        self.pid = 4242
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _reset(monkey_state):
    srv._slots.clear()


def _tmp_gguf(name):
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "wb") as handle:
        handle.write(b"GGUF")
    return path


# ---------------------------------------------------------------- build_command

def test_build_command_flags():
    spec = srv.ServerSpec(
        model_path="/models/LLM/Foo-27B.gguf",
        mmproj_path="/models/LLM/Foo-27B-mmproj.gguf",
        ctx_size=32768,
        port=8735,
        extra_args="--no-mmproj-offload",
    )
    cmd = srv.build_command("/bin/llama-server", spec)
    joined = " ".join(cmd)
    assert cmd[0] == "/bin/llama-server"
    assert "-m /models/LLM/Foo-27B.gguf" in joined
    assert "--mmproj /models/LLM/Foo-27B-mmproj.gguf" in joined
    assert "--host 127.0.0.1" in joined
    assert "--port 8735" in joined
    assert "-c 32768" in joined
    assert "--jinja" in cmd
    assert "--no-webui" in cmd
    assert "-a Foo-27B" in joined            # alias = model stem
    assert cmd[-1] == "--no-mmproj-offload"  # extra_args pass through
    assert '"reasoning_effort": "low"' in joined


def test_build_command_text_only_has_no_mmproj():
    spec = srv.ServerSpec(model_path="/m/x.gguf", mmproj_path=None)
    assert "--mmproj" not in srv.build_command("/bin/s", spec)


# ------------------------------------------------------------- find_mmproj_for

def test_find_mmproj_pairs_by_longest_prefix():
    picked = srv.find_mmproj_for(
        "/models/LLM/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        candidates=[
            "/models/LLM/Other-9B-mmproj-F16.gguf",
            "/models/LLM/Qwen3.8-27B-mmproj-F16.gguf",
        ],
    )
    assert picked == "/models/LLM/Qwen3.8-27B-mmproj-F16.gguf"


def test_find_mmproj_handles_prefix_naming():
    # llama.cpp/Unsloth convention: mmproj FIRST in the filename
    picked = srv.find_mmproj_for(
        "/models/LLM/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        candidates=[
            "/models/LLM/mmproj-Other-9B-F16.gguf",
            "/models/LLM/mmproj-Qwen3-VL-8B-Instruct-F16.gguf",
        ],
    )
    assert picked == "/models/LLM/mmproj-Qwen3-VL-8B-Instruct-F16.gguf"


def test_find_mmproj_single_candidate_fallback():
    picked = srv.find_mmproj_for(
        "/models/LLM/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        candidates=["/models/LLM/Zeta-mmproj.gguf"],
    )
    assert picked == "/models/LLM/Zeta-mmproj.gguf"
    assert srv.find_mmproj_for(
        "/models/LLM/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        candidates=["/models/LLM/Zeta-mmproj.gguf",
                    "/models/LLM/Yeta-mmproj.gguf"],
    ) is None  # two unrelated candidates: refuse to guess


def test_spec_equality_ignores_reasoning_effort():
    a = srv.ServerSpec(model_path="/m/x.gguf", reasoning_effort="low")
    b = srv.ServerSpec(model_path="/m/x.gguf", reasoning_effort="high")
    assert a == b  # per-request setting must never force a respawn


def test_same_model_alias_normalizes():
    assert srv._same_model_alias("Foo-27B", "/models/LLM/Foo-27B.gguf")
    assert srv._same_model_alias("/x/Foo-27B.gguf", "/y/Foo-27B.gguf")
    assert not srv._same_model_alias("Bar-9B", "/models/LLM/Foo-27B.gguf")
    assert not srv._same_model_alias(None, "/models/LLM/Foo-27B.gguf")


def test_same_model_alias_survives_dotted_names():
    # Regression: splitext-based stemming truncated 'Qwen3.8-...' to
    # 'Qwen3' and a server REJECTED its own identical model.
    assert srv._same_model_alias(
        "Qwen3.8-27B-UD-Q4_K_XL",
        "/models/LLM/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    )
    assert srv._stem("Qwen3.8-27B-UD-Q4_K_XL") == "Qwen3.8-27B-UD-Q4_K_XL"
    assert srv._stem("Qwen3.8-27B-UD-Q4_K_XL.gguf") == "Qwen3.8-27B-UD-Q4_K_XL"


# ----------------------------------------------------------------- attach

def test_attach_ok(monkeypatch=None):
    def fake_get(url, timeout=3.0):
        if url.endswith("/health"):
            return 200, '{"status":"ok"}'
        return 200, '{"data":[{"id":"Foo-27B"}]}'
    old = srv._http_get
    srv._http_get = fake_get
    try:
        handle = srv.attach("http://127.0.0.1:9999/v1")
        assert handle.base_url == "http://127.0.0.1:9999/v1"
        assert handle.spec is None
        try:
            srv.attach("http://127.0.0.1:9999", expect_alias="Bar-9B")
            assert False, "alias mismatch must raise"
        except RuntimeError as exc:
            assert "Foo-27B" in str(exc)
    finally:
        srv._http_get = old


def test_attach_down_is_actionable():
    def fake_get(url, timeout=3.0):
        raise OSError("refused")
    old = srv._http_get
    srv._http_get = fake_get
    try:
        try:
            srv.attach("http://127.0.0.1:9999")
            assert False, "down server must raise"
        except RuntimeError as exc:
            assert "9999" in str(exc)
    finally:
        srv._http_get = old


# ------------------------------------------------------------- ensure_server

def _fake_env(health_after_spawn="ok"):
    """Patch seams: server is down until _popen runs, then healthy."""
    state = {"spawned": [], "proc": None}

    def fake_get(url, timeout=3.0):
        proc = state["proc"]
        if proc is None or proc.poll() is not None:  # dead = port free
            raise OSError("refused")
        if url.endswith("/health"):
            return (200, '{"status":"ok"}') if health_after_spawn == "ok" \
                else (503, '{"error":{"code":503}}')
        return 200, '{"data":[{"id":"m"}]}'

    def fake_popen(cmd, **kwargs):
        proc = FakeProc()
        state["spawned"].append(cmd)
        state["proc"] = proc
        return proc

    return state, fake_get, fake_popen


def test_ensure_server_spawns_then_reuses():
    model = _tmp_gguf("fake-model.gguf")
    spec = srv.ServerSpec(model_path=model, mmproj_path=None, port=18999)
    state, fake_get, fake_popen = _fake_env()
    old = (srv._http_get, srv._popen, srv._free_vram_bytes)
    srv._http_get, srv._popen = fake_get, fake_popen
    srv._free_vram_bytes = lambda: 90 * 1024**3
    _reset(None)
    try:
        handle = srv.ensure_server(spec)
        assert handle.base_url == "http://127.0.0.1:18999/v1"
        assert len(state["spawned"]) == 1
        handle2 = srv.ensure_server(spec)          # same spec: no respawn
        assert len(state["spawned"]) == 1
        assert handle2.base_url == handle.base_url
        changed = srv.ServerSpec(model_path=model, mmproj_path=None,
                                 port=18999, ctx_size=8192)
        srv.ensure_server(changed)                  # new spec: respawn
        assert len(state["spawned"]) == 2
    finally:
        srv._http_get, srv._popen, srv._free_vram_bytes = old
        _reset(None)


def test_ensure_server_vram_gate():
    model = _tmp_gguf("fake-model.gguf")
    spec = srv.ServerSpec(model_path=model, mmproj_path=None, port=18999)
    state, fake_get, fake_popen = _fake_env()
    old = (srv._http_get, srv._popen, srv._free_vram_bytes)
    srv._http_get, srv._popen = fake_get, fake_popen
    srv._free_vram_bytes = lambda: 1 * 1024**3
    _reset(None)
    try:
        try:
            srv.ensure_server(spec)
            assert False, "low VRAM must refuse to spawn"
        except RuntimeError as exc:
            assert "VRAM" in str(exc)
        assert not state["spawned"]
    finally:
        srv._http_get, srv._popen, srv._free_vram_bytes = old
        _reset(None)


def test_ensure_server_foreign_port_conflict():
    model = _tmp_gguf("fake-model.gguf")
    spec = srv.ServerSpec(model_path=model, mmproj_path=None, port=18999)

    def fake_get(url, timeout=3.0):  # someone else's healthy server
        if url.endswith("/health"):
            return 200, '{"status":"ok"}'
        return 200, '{"data":[{"id":"someone-elses-model"}]}'

    old = srv._http_get
    srv._http_get = fake_get
    _reset(None)
    try:
        try:
            srv.ensure_server(spec)
            assert False, "foreign model on the port must raise"
        except RuntimeError as exc:
            assert "someone-elses-model" in str(exc)
    finally:
        srv._http_get = old
        _reset(None)


def test_stop_server_terminates():
    _reset(None)
    slot = srv._slots.setdefault(18999, srv._Slot())
    proc = FakeProc()
    slot.proc = proc
    assert srv.stop_server() is True     # port=None stops every slot
    assert proc.terminated
    assert slot.proc is None
    assert srv.stop_server() is False    # nothing left to stop
    _reset(None)


def test_two_ports_coexist():
    model = _tmp_gguf("fake-model.gguf")
    spec_a = srv.ServerSpec(model_path=model, mmproj_path=None, port=18999)
    spec_b = srv.ServerSpec(model_path=model, mmproj_path=None, port=19001)

    procs = {}

    def fake_popen(cmd, **kwargs):
        port = int(cmd[cmd.index("--port") + 1])
        procs[port] = FakeProc()
        return procs[port]

    def fake_get(url, timeout=3.0):
        port = int(url.split(":")[2].split("/")[0])
        proc = procs.get(port)
        if proc is None or proc.poll() is not None:
            raise OSError("refused")
        if url.endswith("/health"):
            return 200, '{"status":"ok"}'
        return 200, '{"data":[{"id":"m"}]}'

    old = (srv._http_get, srv._popen, srv._free_vram_bytes)
    srv._http_get, srv._popen = fake_get, fake_popen
    srv._free_vram_bytes = lambda: 90 * 1024**3
    _reset(None)
    try:
        handle_a = srv.ensure_server(spec_a)
        handle_b = srv.ensure_server(spec_b)   # must NOT stop port 18999
        assert not procs[18999].terminated
        assert handle_a.base_url != handle_b.base_url
        srv.ensure_server(spec_a)              # still reuses, no respawn
        assert len(procs) == 2
        assert srv.stop_server(port=19001) is True
        assert not procs[18999].terminated     # other slot untouched
        assert procs[19001].terminated
    finally:
        srv._http_get, srv._popen, srv._free_vram_bytes = old
        _reset(None)


def test_build_command_empty_reasoning_effort_omits_flag():
    spec = srv.ServerSpec(model_path="/m/x.gguf", reasoning_effort="")
    assert "--chat-template-kwargs" not in srv.build_command("/bin/s", spec)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all llamacpp_server tests passed")
