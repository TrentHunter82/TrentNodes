/**
 * H3 Dial Operator - several dials on one node face.
 *
 * All state lives as JSON in the node's hidden "dials" widget, so save,
 * load, undo and the API see plain widget values. This file only draws
 * that state and writes edits back:
 *
 *   { slots: [{ id, on, file, name, a, b, strength, save }],
 *     input: { on, strength }, mix_name, save_mix }
 *
 * A slot with `file` uses that saved dial; otherwise its A and B terms
 * are made into a dial on the next run (clip needed). Save writes at once
 * when the dial is already made, else it is flagged for the next run.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "H3DialOperator";
const RANGE = 20;
const NEW = "";

const CSS = `
.hdo-root {
    --hdo-panel: #0e2322;
    --hdo-card: #13302e;
    --hdo-pill: #232526;
    --hdo-line: rgba(255, 255, 255, .08);
    --hdo-text: #e4ecea;
    --hdo-muted: #8ea7a3;
    --hdo-pos: #4fd1b5;
    --hdo-neg: #ef7a8a;
    --hdo-wait: #e8b75a;
    --hdo-font: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
    --hdo-mono: "Cascadia Code", ui-monospace, Consolas, monospace;
    width: 100%; height: 100%;
    font-family: var(--hdo-font); color: var(--hdo-text);
    font-size: 12px; user-select: none;
    display: flex; flex-direction: column; gap: 8px;
    padding: 2px 4px; box-sizing: border-box;
}
.hdo-root * { box-sizing: border-box; }
.hdo-card {
    background: var(--hdo-card); border-radius: 12px; padding: 8px 10px;
    display: flex; flex-direction: column; gap: 7px;
    box-shadow: inset 0 0 0 1px var(--hdo-line);
}
.hdo-card.off { opacity: .45; }
.hdo-row { display: flex; align-items: center; gap: 8px; min-width: 0; }
.hdo-led {
    width: 14px; height: 14px; border-radius: 50%; flex: none; cursor: pointer;
    background: #555a5c; box-shadow: inset 0 0 0 2px #2c3031;
}
.hdo-led.on { background: var(--hdo-pos); box-shadow: 0 0 8px var(--hdo-pos); }
.hdo-pill {
    flex: 1; min-width: 0; height: 28px; border-radius: 14px;
    background: var(--hdo-pill); display: flex; align-items: center;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);
}
.hdo-arrow {
    width: 26px; height: 100%; border: 0; background: none; flex: none;
    color: var(--hdo-text); cursor: pointer; font-size: 11px;
}
.hdo-arrow:hover { color: var(--hdo-pos); }
.hdo-pill select, .hdo-pill .hdo-val {
    flex: 1; min-width: 0; height: 100%; border: 0; outline: 0;
    background: none; color: var(--hdo-text); font: inherit;
    font-size: 13px; text-align: center; text-align-last: center;
    appearance: none; cursor: pointer;
}
.hdo-pill select option { background: #1b1d1e; }
.hdo-pill .hdo-val { font-family: var(--hdo-mono); cursor: ns-resize; line-height: 28px; }
.hdo-x {
    border: 0; background: none; color: var(--hdo-muted); cursor: pointer;
    font-size: 15px; width: 20px; flex: none;
}
.hdo-x:hover { color: var(--hdo-neg); }
.hdo-terms { display: grid; grid-template-columns: 1fr auto 1fr; gap: 6px; align-items: center; }
.hdo-term {
    width: 100%; height: 26px; border-radius: 8px; border: 0; outline: 0;
    padding: 0 8px; font: inherit; color: var(--hdo-text);
    background: rgba(0,0,0,.35); box-shadow: inset 0 0 0 1px var(--hdo-line);
}
.hdo-term:focus { box-shadow: inset 0 0 0 1px var(--hdo-pos); }
.hdo-term[readonly] { color: var(--hdo-muted); }
.hdo-term.a::placeholder { color: rgba(239,122,138,.6); }
.hdo-term.b::placeholder { color: rgba(79,209,181,.6); }
.hdo-vs { color: var(--hdo-muted); font-size: 11px; }
.hdo-knob { width: 64px; height: 64px; flex: none; cursor: ns-resize; touch-action: none; }
.hdo-side { display: flex; flex-direction: column; gap: 6px; flex: 1; min-width: 0; }
.hdo-ends { display: flex; justify-content: space-between; color: var(--hdo-muted); font-size: 10px; padding: 0 10px; }
.hdo-save { display: flex; flex-direction: column; align-items: center; gap: 3px; flex: none; width: 46px; }
.hdo-save span { font-size: 11px; color: var(--hdo-muted); }
.hdo-save button {
    width: 30px; height: 30px; border-radius: 50%; border: 0; cursor: pointer;
    background: radial-gradient(circle at 40% 35%, #8b8f90, #4a4e4f);
    box-shadow: 0 0 0 3px #2a2d2e, inset 0 0 0 5px #6d7172;
}
.hdo-save.wait button { box-shadow: 0 0 0 3px var(--hdo-wait), inset 0 0 0 5px #6d7172; }
.hdo-save.done button { background: var(--hdo-pos); box-shadow: 0 0 0 3px #1f5a50, 0 0 10px var(--hdo-pos); }
.hdo-foot { display: flex; align-items: center; gap: 8px; }
.hdo-add {
    height: 28px; padding: 0 12px; border-radius: 14px; border: 0; cursor: pointer;
    background: var(--hdo-pill); color: var(--hdo-text); font: inherit;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);
}
.hdo-add:hover { color: var(--hdo-pos); }
.hdo-note { color: var(--hdo-muted); font-size: 11px; min-height: 14px; }
`;

let dialList = [];

async function refreshDials() {
    try {
        const resp = await api.fetchApi("/trent/h3_dial_operator/dials");
        const data = await resp.json();
        if (data.ok) dialList = data.dials;
    } catch (err) {
        console.error("[H3DialOperator] could not list dials", err);
    }
    for (const node of app.graph?._nodes || []) {
        if (node.type === NODE_NAME) node.hdoRender?.();
    }
}

function round(v, step) {
    return Math.round(v / step) * step;
}

function fmt(v) {
    return (v > 0 ? "+" : "") + (Number.isInteger(v) ? v : v.toFixed(2).replace(/0$/, ""));
}

function knobSvg(value) {
    const start = -135, sweep = 270, r = 24, c = 32;
    const ang = start + sweep * (value + RANGE) / (2 * RANGE);
    const pt = (deg, rad) => {
        const t = (deg - 90) * Math.PI / 180;
        return [c + rad * Math.cos(t), c + rad * Math.sin(t)];
    };
    const arc = (a0, a1) => {
        const [x0, y0] = pt(a0, r), [x1, y1] = pt(a1, r);
        const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
        return `M${x0} ${y0}A${r} ${r} 0 ${large} ${a1 > a0 ? 1 : 0} ${x1} ${y1}`;
    };
    const color = value < 0 ? "var(--hdo-neg)" : "var(--hdo-pos)";
    const [px, py] = pt(ang, r - 9);
    return `<svg viewBox="0 0 64 64" width="64" height="64">
        <path d="${arc(start, start + sweep)}" fill="none" stroke="#2b2f30" stroke-width="5" stroke-linecap="round"/>
        ${value !== 0 ? `<path d="${arc(0, ang)}" fill="none" stroke="${color}" stroke-width="5" stroke-linecap="round"/>` : ""}
        <circle cx="32" cy="32" r="17" fill="#232526" stroke="rgba(255,255,255,.18)"/>
        <line x1="32" y1="32" x2="${px}" y2="${py}" stroke="${value ? color : "#8ea7a3"}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="32" y1="5" x2="32" y2="9" stroke="#8ea7a3" stroke-width="1.5"/>
    </svg>`;
}

/* Drag up or down to turn; Shift for fine steps, double-click for 0. */
function bindTurn(el, get, set) {
    el.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        el.setPointerCapture(e.pointerId);
        const y0 = e.clientY, v0 = get();
        const move = (ev) => {
            const fine = ev.shiftKey;
            const step = fine ? 0.05 : 0.25;
            const v = v0 + (y0 - ev.clientY) * (fine ? 0.02 : 0.15);
            set(Math.max(-RANGE, Math.min(RANGE, round(v, step))));
        };
        const up = () => {
            el.removeEventListener("pointermove", move);
            el.removeEventListener("pointerup", up);
        };
        el.addEventListener("pointermove", move);
        el.addEventListener("pointerup", up);
    });
    el.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        set(0);
    });
}

function h(tag, cls, html) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (html !== undefined) el.innerHTML = html;
    return el;
}

function attach(node) {
    const data = node.widgets.find((w) => w.name === "dials");
    data.hidden = true;
    data.options = { ...data.options, hidden: true };

    const root = h("div", "hdo-root");
    let note = "";

    const state = () => {
        try {
            const s = JSON.parse(data.value || "{}");
            s.slots = s.slots || [];
            return s;
        } catch {
            return { slots: [] };
        }
    };
    const commit = (s, redraw = true) => {
        data.value = JSON.stringify(s);
        node.graph?.setDirtyCanvas(true, true);
        if (redraw) render();
    };
    const label = (slot) => slot.name ||
        (slot.a || slot.b ? `${slot.a || "?"} vs ${slot.b || "?"}` : "new dial");

    function strengthRow(get, set, onSave) {
        const row = h("div", "hdo-row");
        const knob = h("div", "hdo-knob", knobSvg(get()));
        const side = h("div", "hdo-side");
        const pill = h("div", "hdo-pill");
        const dec = h("button", "hdo-arrow", "&#9664;");
        const val = h("div", "hdo-val", `Strength ${fmt(get())}`);
        const inc = h("button", "hdo-arrow", "&#9654;");
        const ends = h("div", "hdo-ends", `<span>A &#8592; &minus;${RANGE}</span><span>+${RANGE} &#8594; B</span>`);
        const update = (v) => {
            set(v);
            knob.innerHTML = knobSvg(v);
            val.textContent = `Strength ${fmt(v)}`;
        };
        dec.onclick = () => update(Math.max(-RANGE, round(get() - 0.5, 0.25)));
        inc.onclick = () => update(Math.min(RANGE, round(get() + 0.5, 0.25)));
        bindTurn(knob, get, update);
        bindTurn(val, get, update);
        pill.append(dec, val, inc);
        side.append(pill, ends);
        row.append(knob, side);
        if (onSave) row.append(onSave);
        return row;
    }

    function slotCard(s, slot) {
        const card = h("div", "hdo-card" + (slot.on === false ? " off" : ""));
        const top = h("div", "hdo-row");
        const led = h("div", "hdo-led" + (slot.on === false ? "" : " on"));
        led.title = "On / off";
        led.onclick = () => { slot.on = slot.on === false; commit(s); };

        const pill = h("div", "hdo-pill");
        const options = [{ file: NEW, name: "new from text" }, ...dialList];
        const sel = h("select");
        for (const d of options) {
            const o = h("option");
            o.value = d.file;
            o.textContent = d.file ? d.name : `✎ ${d.name}`;
            sel.append(o);
        }
        if (slot.file && !options.some((d) => d.file === slot.file)) {
            const o = h("option");
            o.value = slot.file;
            o.textContent = `${slot.name || slot.file} (missing)`;
            sel.append(o);
        }
        sel.value = slot.file || NEW;
        const pick = (file) => {
            const d = dialList.find((x) => x.file === file);
            slot.file = d ? d.file : null;
            slot.name = d ? d.name : "";
            if (d) { slot.a = d.a; slot.b = d.b; }
            slot.save = false;
            commit(s);
        };
        sel.onchange = () => pick(sel.value);
        const step = (dir) => {
            const i = options.findIndex((d) => d.file === (slot.file || NEW));
            pick(options[(i + dir + options.length) % options.length].file);
        };
        const prev = h("button", "hdo-arrow", "&#9664;");
        const next = h("button", "hdo-arrow", "&#9654;");
        prev.onclick = () => step(-1);
        next.onclick = () => step(1);
        pill.append(prev, sel, next);

        const del = h("button", "hdo-x", "&times;");
        del.title = "Remove this dial";
        del.onclick = () => { s.slots = s.slots.filter((x) => x !== slot); commit(s); };
        top.append(led, pill, del);
        card.append(top);

        const fileOnly = slot.file && !slot.a && !slot.b;
        const terms = h("div", "hdo-terms");
        const a = h("input", "hdo-term a");
        const b = h("input", "hdo-term b");
        a.placeholder = "A: e.g. slow motion";
        b.placeholder = "B: e.g. fast motion";
        a.value = fileOnly ? "(image or mixed dial)" : slot.a || "";
        b.value = fileOnly ? "" : slot.b || "";
        a.readOnly = b.readOnly = !!fileOnly;
        const edit = () => {
            slot.a = a.value;
            slot.b = b.value;
            if (slot.file) { slot.file = null; slot.name = ""; sel.value = NEW; }
            slot.save = false;
            saveBox.className = "hdo-save";
            commit(s, false);
        };
        a.oninput = b.oninput = edit;
        a.onchange = b.onchange = () => render();
        terms.append(a, h("span", "hdo-vs", "&#8596;"), b);
        card.append(terms);

        const saveBox = h("div", "hdo-save" + (slot.file ? " done" : slot.save ? " wait" : ""));
        const saveBtn = h("button");
        saveBox.append(h("span", "", "Save"), saveBtn);
        saveBox.title = slot.file ? `Saved: ${slot.file}` :
            slot.save ? "Saves on the next run" : "Save this dial";
        saveBtn.onclick = async () => {
            if (slot.file || !slot.a?.trim() || !slot.b?.trim()) return;
            const subjects = node.widgets.find((w) => w.name === "subjects")?.value ?? "";
            try {
                const resp = await api.fetchApi("/trent/h3_dial_operator/save", {
                    method: "POST",
                    body: JSON.stringify({ slot, subjects }),
                });
                const out = await resp.json();
                if (!out.ok) throw new Error(out.error);
                if (out.file) {
                    slot.file = out.file;
                    slot.name = out.file.split("/").pop().replace(".safetensors", "");
                    slot.save = false;
                    note = `Saved ${slot.name}`;
                    await refreshDials();
                } else {
                    slot.save = true;
                    note = "Saves on the next run (the dial is not made yet)";
                }
            } catch (err) {
                note = `Save failed: ${err.message}`;
            }
            commit(s);
        };

        card.append(strengthRow(
            () => slot.strength ?? 0,
            (v) => { slot.strength = v; commit(s, false); },
            saveBox));
        return card;
    }

    function render() {
        const s = state();
        root.replaceChildren();
        for (const slot of s.slots) root.append(slotCard(s, slot));

        const input = node.inputs?.find((i) => i.name === "dial");
        if (input?.link != null) {
            s.input = s.input || { on: true, strength: 1 };
            const card = h("div", "hdo-card" + (s.input.on === false ? " off" : ""));
            const top = h("div", "hdo-row");
            const led = h("div", "hdo-led" + (s.input.on === false ? "" : " on"));
            led.onclick = () => { s.input.on = s.input.on === false; commit(s); };
            top.append(led, h("div", "hdo-vs", "dial input"));
            card.append(top, strengthRow(
                () => s.input.strength ?? 1,
                (v) => { s.input.strength = v; commit(s, false); }));
            root.append(card);
        }

        const foot = h("div", "hdo-foot");
        const add = h("button", "hdo-add", "+ Add dial");
        add.onclick = () => {
            s.slots.push({ id: Date.now().toString(36), on: true, file: null, name: "", a: "", b: "", strength: 1 });
            commit(s);
        };
        const mix = h("input", "hdo-term");
        mix.placeholder = "mix name (empty = auto)";
        mix.value = s.mix_name || "";
        mix.oninput = () => { s.mix_name = mix.value; commit(s, false); };
        const mixBox = h("div", "hdo-save" + (s.save_mix ? " wait" : ""));
        const mixBtn = h("button");
        mixBox.append(h("span", "", "Mix"), mixBtn);
        mixBox.title = s.save_mix ? "Saves the mix on the next run" :
            "Save all active dials at their strengths as one dial";
        mixBtn.onclick = () => {
            s.save_mix = !s.save_mix;
            note = s.save_mix ? "The mix saves on the next run" : "";
            commit(s);
        };
        foot.append(add, mix, mixBox);
        root.append(foot, h("div", "hdo-note", note));

        requestAnimationFrame(() => {
            const want = node.computeSize()[1];
            if (node.size[1] < want) node.setSize([node.size[0], want]);
            node.graph?.setDirtyCanvas(true, true);
        });
    }

    const height = () => 14 + 40 + state().slots.length * 152 +
        (node.inputs?.find((i) => i.name === "dial")?.link != null ? 112 : 0);
    const widget = node.addDOMWidget("dial_operator", "hdo_panel", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: height,
        getMaxHeight: height,
    });
    widget.serialize = false;
    widget.computeSize = (width) => [width, height()];

    node.hdoRender = render;
    node.hdoNote = (text) => { note = text; render(); };
    return { render, state, commit };
}

app.registerExtension({
    name: "TrentNodes.H3DialOperator",

    setup() {
        const style = document.createElement("style");
        style.textContent = CSS;
        document.head.append(style);
        refreshDials();
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const ui = attach(this);
            this.hdo = ui;
            const s = ui.state();
            if (!s.slots.length) {
                s.slots.push({ id: Date.now().toString(36), on: true, file: null, name: "", a: "", b: "", strength: 1 });
                ui.commit(s);
            }
            this.color = "#1a4d4a";
            this.bgcolor = "#0e2322";
            this.size[0] = Math.max(this.size[0], 440);
            ui.render();
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this.hdo?.render();
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            this.hdo?.render();
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const out = message?.h3_dial_operator?.[0];
            if (!out || !this.hdo) return result;
            const s = this.hdo.state();
            const saved = [];
            for (const entry of out.slots || []) {
                const slot = s.slots.find((x) => x.id === entry.id);
                if (slot && entry.saved) {
                    slot.file = entry.saved;
                    slot.name = entry.saved.split("/").pop().replace(".safetensors", "");
                    slot.save = false;
                    saved.push(slot.name);
                }
            }
            if (out.mix) {
                s.save_mix = false;
                saved.push(out.mix.split("/").pop().replace(".safetensors", ""));
            }
            this.hdo.commit(s, false);
            if (saved.length) {
                refreshDials();
                this.hdoNote(`Saved ${saved.join(", ")}`);
            }
            return result;
        };
    },
});
