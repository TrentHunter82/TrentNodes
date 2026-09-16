"""One private Codex app-server process per prompt, using managed ChatGPT auth."""

import json
from collections import deque
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time


class CodexError(RuntimeError):
    pass


def executable():
    configured = os.environ.get("TRENT_CODEX_BIN")
    found = configured or shutil.which("codex")
    if not found:
        candidate = Path.home() / ".local" / "bin" / "codex"
        if candidate.is_file():
            found = str(candidate)
    if not found:
        raise CodexError("Install Codex CLI in the environment running ComfyUI, then run codex login.")
    return found


class CodexSession:
    def __init__(self, cwd, instructions, model="", effort="", timeout=600,
                 interrupt=lambda: None):
        self.cwd = str(cwd)
        self.instructions = instructions
        self.model = model.strip()
        self.effort = effort
        self.interrupt = interrupt
        self.deadline = time.monotonic() + timeout
        self.messages = queue.Queue()
        self.pending = deque()
        self.sequence = 0
        self.thread_id = None
        self.turn_id = None
        self.process = None
        self.model_name = ""

    def __enter__(self):
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            env.pop(key, None)
        command = [executable(), "app-server", "--listen", "stdio://"]
        for setting in (
            'forced_login_method="chatgpt"', 'model_provider="openai"',
            'web_search="disabled"', 'features.shell_tool=false',
            'features.apps=false', 'features.multi_agent=false',
            'features.code_mode=false', 'project_doc_max_bytes=0',
        ):
            command.extend(["-c", setting])
        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NO_WINDOW,
        }
        try:
            self.process = subprocess.Popen(
                command, cwd=self.cwd, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1, **options,
            )
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            self.request("initialize", {
                "clientInfo": {"name": "trent_h3_promptor", "version": "1.0.0"},
                "capabilities": {"experimentalApi": True},
            })
            self.send({"method": "initialized", "params": {}})
            account = self.request("account/read", {"refreshToken": False}).get("account")
            if not account or account.get("type") != "chatgpt":
                raise CodexError("H3 Codex Promptor requires ChatGPT subscription sign-in. Run codex login in the same environment/user as ComfyUI. API-key billing is not used.")
            config = self.request("config/read", {"includeLayers": False}).get("config", {})
            overrides = {"developer_instructions": "", "project_doc_max_bytes": 0}
            # Disable installed external integrations for this prompt session only.
            for name in config.get("mcp_servers", {}):
                overrides[f"mcp_servers.{name}.enabled"] = False
            for name in config.get("plugins", {}):
                overrides[f"plugins.{name}.enabled"] = False
            params = {
                "cwd": self.cwd, "ephemeral": True, "approvalPolicy": "never",
                "sandbox": "read-only", "modelProvider": "openai",
                "baseInstructions": self.instructions,
                "config": overrides, "environments": [],
            }
            if self.model:
                params["model"] = self.model
            result = self.request("thread/start", params)
            self.thread_id = result["thread"]["id"]
            self.model_name = result.get("model", self.model)
            return self
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.messages.put(json.loads(line))
                except json.JSONDecodeError:
                    self.messages.put({"transport_error": True})
        finally:
            self.messages.put(None)

    def send(self, message):
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CodexError("Codex app-server disconnected. Check codex login status and CLI version.") from exc

    def _receive(self):
        while True:
            self.interrupt()
            if time.monotonic() >= self.deadline:
                raise CodexError("Codex prompt timed out. The request was cancelled; no automatic retry was submitted.")
            try:
                message = self.messages.get(timeout=0.15)
            except queue.Empty:
                continue
            if message is None or message.get("transport_error"):
                raise CodexError("Codex app-server stopped or returned invalid data. Update Codex CLI and check codex login status.")
            if "method" in message and "id" in message:
                self.send({"id": message["id"], "error": {
                    "code": -32601, "message": "This prompt-only client does not authorize tools or approvals.",
                }})
                raise CodexError("Codex requested an interactive tool or approval. This node only writes prompts; no action was approved.")
            return message

    def next_message(self):
        self.interrupt()
        return self.pending.popleft() if self.pending else self._receive()

    def request(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        self.send({"id": request_id, "method": method, "params": params})
        while True:
            message = self._receive()
            if message.get("id") == request_id:
                if "error" in message:
                    # Server errors can echo config values; keep credentials out of the UI.
                    raise CodexError(f"Codex rejected {method}. Check your CLI version, sign-in, model availability and account limits.")
                return message["result"]
            if message.get("method") in ("item/completed", "turn/completed"):
                self.pending.append(message)

    def turn(self, inputs, schema):
        params = {"threadId": self.thread_id, "input": inputs, "outputSchema": schema}
        if self.effort:
            params["effort"] = self.effort
        result = self.request("turn/start", params)
        self.turn_id = result["turn"]["id"]
        text = ""
        while True:
            message = self.next_message()
            event = message.get("method")
            data = message.get("params", {})
            if data.get("threadId") != self.thread_id:
                continue
            if event == "item/completed":
                item = data.get("item", {})
                if item.get("type") == "agentMessage":
                    text = item.get("text", "")
            if event == "turn/completed":
                turn = data["turn"]
                self.turn_id = None
                if turn.get("status") != "completed":
                    code = (turn.get("error") or {}).get("codexErrorInfo")
                    hint = " Check your subscription limits or model access."
                    if code == "contextWindowExceeded":
                        hint = " Reduce the number of references or the context length."
                    elif code == "unauthorized":
                        hint = " Your Codex login has expired. Run codex login in the same environment/user as ComfyUI."
                    raise CodexError("Codex did not complete this prompt." + hint)
                if not text.strip():
                    raise CodexError("Codex completed without a prompt. No automatic retry was submitted.")
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise CodexError("Codex returned invalid structured output. No automatic retry was submitted.") from exc

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            if self.thread_id and self.turn_id:
                try:
                    self.send({"id": 999999, "method": "turn/interrupt", "params": {
                        "threadId": self.thread_id, "turnId": self.turn_id,
                    }})
                except CodexError:
                    pass
            if os.name == "nt":
                self.process.terminate()
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3)
        for pipe in (self.process.stdin, self.process.stdout):
            pipe.close()

    def __exit__(self, *args):
        self.close()
