/**
 * Control Overlay Rodeo - in-node control-map player.
 *
 * Python writes every connected layer to temp/ as JPEG frames and sends
 * a manifest; this file loads them and composites the enabled layers on
 * a canvas each frame. Layer toggles, opacity and blend live in the
 * hidden layer_settings widget: changing them previews instantly, and
 * the next queue re-runs the node so its output matches. Loop is
 * playback-only and lives in node.properties.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LABELS = {
    original: "Original",
    depth: "Depth",
    hed: "HED",
    canny: "Canny",
    pose: "Pose",
};

// The Python composite() mirrors each of these canvas modes.
const BLENDS = {
    normal: "source-over",
    screen: "screen",
    lighten: "lighten",
    add: "lighter",
    multiply: "multiply",
    difference: "difference",
};

/* Every preset keeps the original underneath; a preset only shows up
   when all of its layers are connected. */
const PRESETS = [
    ["Pose", ["pose"]],
    ["HED", ["hed"]],
    ["Canny", ["canny"]],
    ["Depth", ["depth"]],
    ["Pose + HED", ["pose", "hed"]],
    ["Pose + Canny", ["pose", "canny"]],
    ["Depth + Pose", ["depth", "pose"]],
];

const STYLE = `
.cor-root {
    --cor-accent: #6fc3b2;
    display: flex; flex-direction: column; gap: 6px;
    width: 100%; height: 100%; box-sizing: border-box; padding: 2px;
    font: 12px sans-serif; color: var(--input-text, #ddd);
}
.cor-stage {
    position: relative; flex: 1 1 auto; min-height: 80px;
    background: #000; border-radius: 6px; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
}
.cor-stage canvas {
    display: block; width: 100%; height: 100%; object-fit: contain;
}
.cor-badge {
    position: absolute; top: 6px; left: 6px; padding: 2px 6px;
    border-radius: 4px; background: rgba(0, 0, 0, 0.65);
    font-size: 11px; pointer-events: none;
}
.cor-empty { position: absolute; color: #777; pointer-events: none; }
.cor-row {
    display: flex; align-items: center; gap: 5px; flex-wrap: wrap;
    flex: none;
}
.cor-btn {
    background: var(--comfy-input-bg, #222); color: inherit;
    border: 1px solid var(--border-color, #444); border-radius: 5px;
    padding: 3px 8px; cursor: pointer; font: inherit;
}
.cor-btn:hover { border-color: var(--cor-accent); }
.cor-btn.on {
    background: color-mix(in srgb, var(--cor-accent) 30%, transparent);
    border-color: var(--cor-accent);
}
.cor-loop { margin-left: auto; display: flex; align-items: center; gap: 4px; }
.cor-root input[type=range] {
    accent-color: var(--cor-accent); width: 100%; min-width: 0; margin: 0;
}
.cor-root input[type=checkbox] { accent-color: var(--cor-accent); margin: 0; }
.cor-scrub { flex: none; }
.cor-layers {
    display: flex; flex-direction: column; gap: 4px; flex: none;
    border-top: 1px solid var(--border-color, #444); padding-top: 6px;
}
.cor-layer {
    display: grid; grid-template-columns: 92px 1fr 36px 84px;
    align-items: center; gap: 6px;
}
.cor-layer .cor-btn { text-align: left; }
.cor-layer select {
    background: var(--comfy-input-bg, #222); color: inherit;
    border: 1px solid var(--border-color, #444); border-radius: 5px;
    font: inherit; padding: 2px;
}
.cor-pct { text-align: right; font-variant-numeric: tabular-nums; }
.cor-info {
    flex: none; color: var(--descrip-text, #999); font-size: 11px;
    font-variant-numeric: tabular-nums;
}
.cor-overlay {
    position: fixed; inset: 0; z-index: 10000;
    background: rgba(0, 0, 0, 0.92);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 8px;
}
.cor-overlay canvas {
    width: 96vw; height: 90vh; object-fit: contain;
}
.cor-overlay .cor-info { color: #bbb; font-size: 12px; }
`;

function injectStyle() {
    if (document.getElementById("cor-style")) return;
    const style = document.createElement("style");
    style.id = "cor-style";
    style.textContent = STYLE;
    document.head.appendChild(style);
}

function el(tag, className, text) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined) e.textContent = text;
    return e;
}

function setupRodeo(node) {
    injectStyle();

    const S = {
        data: null,
        images: {},   // layer name -> Image[]
        loaded: 0,
        failed: 0,
        frame: 0,
        playing: false,
        raf: 0,
        last: 0,
        acc: 0,
        expanded: null,
    };

    const settingsWidget = node.widgets.find((w) => w.name === "layer_settings");
    let layers = JSON.parse(settingsWidget.value);
    const saveLayers = () => {
        settingsWidget.value = JSON.stringify(layers);
    };
    const loop = () => node.properties.loop ?? true;

    // ── DOM ───────────────────────────────────────────

    const root = el("div", "cor-root");
    const stage = el("div", "cor-stage");
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const badge = el("div", "cor-badge");
    const empty = el("div", "cor-empty", "Run the workflow to load frames");
    badge.style.display = "none";
    stage.append(canvas, badge, empty);

    const transport = el("div", "cor-row");
    const playBtn = el("button", "cor-btn", "Play");
    const prevBtn = el("button", "cor-btn", "−1");
    const nextBtn = el("button", "cor-btn", "+1");
    const expandBtn = el("button", "cor-btn", "Expand");
    const loopLabel = el("label", "cor-loop");
    const loopBox = document.createElement("input");
    loopBox.type = "checkbox";
    loopLabel.append(loopBox, document.createTextNode("Loop"));
    playBtn.title = "Play or pause (Space in the expanded view)";
    prevBtn.title = "Step back one frame (Left Arrow in the expanded view)";
    nextBtn.title = "Step forward one frame (Right Arrow in the expanded view)";
    expandBtn.title = "Show the player full screen (Esc or click to close)";
    loopLabel.title = "Restart from the first frame at the end";
    transport.append(playBtn, prevBtn, nextBtn, expandBtn, loopLabel);

    const scrub = document.createElement("input");
    scrub.type = "range";
    scrub.className = "cor-scrub";
    scrub.min = 0;
    scrub.max = 0;
    scrub.value = 0;
    scrub.title = "Drag to scrub through the frames";

    const presets = el("div", "cor-row");
    const layerBox = el("div", "cor-layers");
    const info = el("div", "cor-info");

    root.append(stage, transport, scrub, presets, layerBox, info);

    const widget = node.addDOMWidget("rodeo_player", "rodeo", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => controlsHeight() + 120,
    });
    widget.serialize = false;

    function controlsHeight() {
        let h = 0;
        for (const child of [transport, scrub, presets, layerBox, info]) {
            if (child.style.display === "none") continue;
            h += child.offsetHeight + 6;
        }
        if (h > 0) return h;
        // Not laid out yet: rough row estimate
        const rows = S.data ? S.data.layers.length : 0;
        return 110 + rows * 28;
    }

    // ── Drawing ───────────────────────────────────────

    const frameCount = () => (S.data ? S.data.frames : 0);
    const layerImage = (name, f) => {
        const list = S.images[name];
        return list ? list[Math.min(f, list.length - 1)] : null;
    };
    const ready = (img) => img && img.complete && img.naturalWidth > 0;

    function draw() {
        if (!S.data) return;
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (const { name } of S.data.layers) {
            const cfg = layers[name];
            const img = layerImage(name, S.frame);
            if (!cfg.on || cfg.opacity <= 0 || !ready(img)) continue;
            ctx.globalAlpha = cfg.opacity;
            ctx.globalCompositeOperation = BLENDS[cfg.blend] || "source-over";
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1;
    }

    function updateInfo() {
        if (!S.data) {
            info.textContent = "";
            return;
        }
        const n = frameCount();
        const total = S.data.layers.reduce((a, l) => a + l.count, 0);
        let text = `Frame ${S.frame + 1}/${n} · ` +
            `${(S.frame / S.data.fps).toFixed(3)}s · ` +
            `${S.data.layers.length} layers`;
        if (S.loaded + S.failed < total) {
            text += ` · loading ${S.loaded}/${total}`;
        }
        if (S.failed) {
            text += ` · ${S.failed} frames missing, run again`;
        }
        info.textContent = text;
        if (S.expanded) S.expanded.info.textContent = info.textContent;
    }

    function setFrame(f) {
        const n = frameCount();
        if (!n) return;
        S.frame = Math.max(0, Math.min(n - 1, f));
        scrub.value = S.frame;
        draw();
        updateInfo();
    }

    // ── Playback ──────────────────────────────────────

    function frameReady(f) {
        return S.data.layers.every(({ name }) =>
            !layers[name].on || ready(layerImage(name, f)));
    }

    function tick(t) {
        S.raf = 0;
        if (!S.playing || !S.data) return;
        if (S.last) S.acc += (t - S.last) / 1000;
        S.last = t;
        const step = 1 / S.data.fps;
        if (S.acc >= step) {
            const n = frameCount();
            let f = S.frame + Math.floor(S.acc / step);
            S.acc %= step;
            if (f >= n) {
                if (loop()) {
                    f %= n;
                } else {
                    f = n - 1;
                    S.playing = false;
                    updateTransport();
                }
            }
            // Hold on frames that are still downloading
            if (frameReady(f)) setFrame(f);
        }
        if (S.playing) S.raf = requestAnimationFrame(tick);
    }

    function play() {
        if (frameCount() < 2) return;
        if (S.frame >= frameCount() - 1 && !loop()) S.frame = 0;
        S.playing = true;
        S.last = 0;
        S.acc = 0;
        if (!S.raf) S.raf = requestAnimationFrame(tick);
        updateTransport();
    }

    function pause() {
        S.playing = false;
        if (S.raf) cancelAnimationFrame(S.raf);
        S.raf = 0;
        updateTransport();
    }

    function updateTransport() {
        playBtn.textContent = S.playing ? "Pause" : "Play";
        loopBox.checked = loop();
    }

    playBtn.onclick = () => (S.playing ? pause() : play());
    prevBtn.onclick = () => {
        pause();
        setFrame(S.frame - 1);
    };
    nextBtn.onclick = () => {
        pause();
        setFrame(S.frame + 1);
    };
    loopBox.onchange = () => {
        node.properties.loop = loopBox.checked;
    };
    scrub.oninput = () => {
        pause();
        setFrame(Number(scrub.value));
    };

    // ── Layer rows and presets ────────────────────────

    function renderLayers() {
        layerBox.replaceChildren();
        presets.replaceChildren();
        const present = new Set(S.data ? S.data.layers.map((l) => l.name) : []);
        layerBox.style.display = present.size ? "" : "none";
        presets.style.display = present.size > 1 ? "" : "none";

        for (const name of Object.keys(LABELS)) {
            if (!present.has(name)) continue;
            const cfg = layers[name];
            const row = el("div", "cor-layer");

            const toggle = el("button", "cor-btn");
            const syncToggle = () => {
                toggle.textContent = `${LABELS[name]} ${cfg.on ? "on" : "off"}`;
                toggle.classList.toggle("on", cfg.on);
            };
            toggle.title = `Show or hide the ${LABELS[name]} layer`;
            toggle.onclick = () => {
                cfg.on = !cfg.on;
                syncToggle();
                saveLayers();
                draw();
            };
            syncToggle();

            const opacity = document.createElement("input");
            opacity.type = "range";
            opacity.min = 0;
            opacity.max = 100;
            opacity.value = Math.round(cfg.opacity * 100);
            opacity.title = `${LABELS[name]} layer opacity`;
            const pct = el("span", "cor-pct", `${opacity.value}%`);
            opacity.oninput = () => {
                cfg.opacity = opacity.value / 100;
                pct.textContent = `${opacity.value}%`;
                saveLayers();
                draw();
            };

            const blend = document.createElement("select");
            for (const mode of Object.keys(BLENDS)) {
                blend.append(new Option(mode, mode, false, mode === cfg.blend));
            }
            blend.title = `${LABELS[name]} layer blend mode`;
            blend.onchange = () => {
                cfg.blend = blend.value;
                saveLayers();
                draw();
            };

            row.append(toggle, opacity, pct, blend);
            layerBox.append(row);
        }

        for (const [label, names] of PRESETS) {
            if (!names.every((n) => present.has(n))) continue;
            const btn = el("button", "cor-btn", label);
            btn.title = "Show only the original and the layers in this preset";
            btn.onclick = () => {
                for (const name of present) {
                    layers[name].on = name === "original" || names.includes(name);
                }
                saveLayers();
                renderLayers();
                draw();
            };
            presets.append(btn);
        }
    }

    // ── Expand overlay ────────────────────────────────

    function onOverlayKey(e) {
        if (e.key === "Escape") collapse();
        else if (e.key === " ") S.playing ? pause() : play();
        else if (e.key === "ArrowLeft") prevBtn.onclick();
        else if (e.key === "ArrowRight") nextBtn.onclick();
        else return;
        e.preventDefault();
        e.stopPropagation();
    }

    function expand() {
        if (!S.data || S.expanded) return;
        const overlay = el("div", "cor-overlay");
        const hint = el("div", "cor-info",
            "Space play/pause · ← → step · Esc or click to close");
        const overlayInfo = el("div", "cor-info");
        overlay.append(canvas, overlayInfo, hint);
        overlay.onclick = collapse;
        document.body.append(overlay);
        window.addEventListener("keydown", onOverlayKey, true);
        S.expanded = { overlay, info: overlayInfo };
        updateInfo();
    }

    function collapse() {
        if (!S.expanded) return;
        window.removeEventListener("keydown", onOverlayKey, true);
        stage.prepend(canvas);
        S.expanded.overlay.remove();
        S.expanded = null;
    }

    expandBtn.onclick = expand;

    // ── Data from Python ──────────────────────────────

    function fitNodeToFrames() {
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if (!S.data || !stage.clientWidth) return;
            const want = Math.min(900, Math.round(
                stage.clientWidth * S.data.height / S.data.width));
            // computeSize() counts the stage at its 120px minimum
            const extra = Math.max(0, want - 120);
            node.setSize([node.size[0], node.computeSize()[1] + extra]);
            node.setDirtyCanvas(true, true);
        }));
    }

    function load(data) {
        pause();
        for (const list of Object.values(S.images)) {
            for (const img of list) img.src = "";
        }
        S.data = data;
        S.images = {};
        S.loaded = 0;
        S.failed = 0;
        S.frame = Math.min(S.frame, data.frames - 1);

        const first = data.layers[0];
        for (const { name, count } of data.layers) {
            const list = [];
            for (let i = 0; i < count; i++) {
                const img = new Image();
                img.onload = () => {
                    S.loaded++;
                    if (name === first.name && i === 0) {
                        canvas.width = img.naturalWidth;
                        canvas.height = img.naturalHeight;
                    }
                    if (layerImage(name, S.frame) === img) draw();
                    updateInfo();
                };
                img.onerror = () => {
                    S.failed++;
                    updateInfo();
                };
                const file = `${name}_${String(i).padStart(5, "0")}.jpg`;
                img.src = api.apiURL(
                    `/view?filename=${encodeURIComponent(file)}` +
                    `&subfolder=${encodeURIComponent(data.subfolder)}&type=temp`);
                list.push(img);
            }
            S.images[name] = list;
        }

        badge.textContent = `${data.width} × ${data.height}`;
        badge.style.display = "";
        empty.style.display = "none";
        scrub.max = Math.max(0, data.frames - 1);
        scrub.value = S.frame;
        renderLayers();
        updateTransport();
        updateInfo();
        fitNodeToFrames();
    }

    const onExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message?.rodeo?.[0]) load(message.rodeo[0]);
    };

    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        layers = JSON.parse(settingsWidget.value);
        renderLayers();
        updateTransport();
        return r;
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        pause();
        collapse();
        return onRemoved?.apply(this, arguments);
    };

    renderLayers();
    updateTransport();
}

app.registerExtension({
    name: "Trent.ControlOverlayRodeo",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ControlOverlayRodeo") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setupRodeo(this);
            const [w, h] = this.computeSize();
            this.setSize([Math.max(w, 420), h]);
            return r;
        };
    },
});
