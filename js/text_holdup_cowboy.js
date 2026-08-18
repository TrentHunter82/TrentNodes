// TextHoldupCowboy - editable pause dialog for mid-run text review.
// Listens for "trent-text-holdup" from the backend, shows the text in a
// textarea with a countdown, and POSTs the (possibly edited) text back.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let activeDialog = null; // { gateId, cleanup }
const queue = []; // payloads waiting while a dialog is open

function postResult(gateId, text) {
    return api.fetchApi("/trent/text_holdup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gate_id: gateId, text }),
    });
}

function showDialog(payload) {
    const { gate_id, node_id, text, timeout } = payload;

    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", zIndex: "10000",
        background: "rgba(0,0,0,0.55)",
        display: "flex", alignItems: "center", justifyContent: "center",
    });

    const panel = document.createElement("div");
    Object.assign(panel.style, {
        background: "var(--comfy-menu-bg, #202020)",
        color: "var(--fg-color, #ddd)",
        border: "1px solid #555", borderRadius: "8px",
        padding: "16px", width: "min(720px, 90vw)",
        maxHeight: "85vh", display: "flex", flexDirection: "column",
        gap: "10px", boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
        fontFamily: "sans-serif",
    });

    const title = document.createElement("div");
    title.textContent = `🤠 Text Holdup — node #${node_id}`;
    Object.assign(title.style, { fontSize: "15px", fontWeight: "bold" });

    const countdown = document.createElement("div");
    Object.assign(countdown.style, {
        fontSize: "12px", opacity: "0.8", fontVariantNumeric: "tabular-nums",
    });

    const textarea = document.createElement("textarea");
    textarea.value = text;
    Object.assign(textarea.style, {
        width: "100%", minHeight: "260px", resize: "vertical",
        background: "var(--comfy-input-bg, #151515)",
        color: "var(--input-text, #ddd)",
        border: "1px solid #444", borderRadius: "4px",
        padding: "8px", fontFamily: "monospace", fontSize: "13px",
        boxSizing: "border-box",
    });

    const buttons = document.createElement("div");
    Object.assign(buttons.style, {
        display: "flex", gap: "8px", justifyContent: "flex-end",
    });

    function makeButton(label, primary) {
        const b = document.createElement("button");
        b.textContent = label;
        Object.assign(b.style, {
            padding: "6px 14px", borderRadius: "4px", cursor: "pointer",
            border: "1px solid #555",
            background: primary ? "#3a6ea5" : "#333",
            color: "#eee", fontSize: "13px",
        });
        return b;
    }

    const passBtn = makeButton("Pass Unchanged", false);
    const sendBtn = makeButton("Send (Ctrl+Enter)", true);
    buttons.append(passBtn, sendBtn);

    panel.append(title, countdown, textarea, buttons);
    overlay.append(panel);
    document.body.append(overlay);
    textarea.focus();
    textarea.setSelectionRange(0, 0);

    let remaining = Math.round(timeout);
    countdown.textContent = `Auto-continues in ${remaining}s`;
    const ticker = setInterval(() => {
        remaining -= 1;
        countdown.textContent = `Auto-continues in ${Math.max(remaining, 0)}s`;
        if (remaining <= 0) cleanup(); // server releases itself on timeout
    }, 1000);

    function cleanup() {
        clearInterval(ticker);
        overlay.remove();
        document.removeEventListener("keydown", onKey, true);
        activeDialog = null;
        if (queue.length) showDialog(queue.shift());
    }

    function finish(value) {
        postResult(gate_id, value).catch((e) =>
            console.warn("[TextHoldupCowboy] submit failed:", e)
        );
        cleanup();
    }

    function onKey(e) {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            e.stopPropagation();
            finish(textarea.value);
        }
    }
    document.addEventListener("keydown", onKey, true);

    sendBtn.onclick = () => finish(textarea.value);
    passBtn.onclick = () => finish(text);

    activeDialog = { gateId: gate_id, cleanup };
}

api.addEventListener("trent-text-holdup", ({ detail }) => {
    if (activeDialog) queue.push(detail);
    else showDialog(detail);
});

api.addEventListener("trent-text-holdup-close", ({ detail }) => {
    const gateId = detail?.gate_id;
    if (activeDialog && activeDialog.gateId === gateId) {
        activeDialog.cleanup();
    } else {
        const i = queue.findIndex((p) => p.gate_id === gateId);
        if (i >= 0) queue.splice(i, 1);
    }
});

app.registerExtension({ name: "TrentNodes.TextHoldupCowboy" });
