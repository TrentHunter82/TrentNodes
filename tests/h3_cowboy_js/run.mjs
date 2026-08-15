// Checks js/h3_cowboy.js against a mocked ComfyUI frontend.
//
//   node tests/h3_cowboy_js/run.mjs
//
// Same trick as tests/multi_load_cowboy/run.mjs: the extension imports
// "../../scripts/app.js", which only resolves when ComfyUI serves it, so
// the source is staged next to a mock and that one import is rewritten.
// No DOM is needed - this extension only reads and writes widget flags.
//
// What matters here is the rule the visibility logic exists to protect:
// a hidden widget must still be IN node.widgets, because a workflow
// stores widget values positionally.

import { mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "..", "js", "h3_cowboy.js");

const stage = mkdtempSync(join(tmpdir(), "trentnodes-h3c-"));
process.on("exit", () => rmSync(stage, { recursive: true, force: true }));

mkdirSync(join(stage, "scripts"));
writeFileSync(
    join(stage, "scripts", "app.js"),
    `export const app = {
        extension: null,
        graph: { setDirtyCanvas() {} },
        registerExtension(ext) { app.extension = ext; },
    };`,
);
writeFileSync(
    join(stage, "extension.js"),
    readFileSync(source, "utf8").replaceAll("../../scripts/", "./scripts/"),
);

// The extension defers its resize to requestAnimationFrame.
globalThis.requestAnimationFrame = (fn) => fn();

const { app } = await import(pathToFileURL(join(stage, "scripts", "app.js")));
await import(pathToFileURL(join(stage, "extension.js")));

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

let failures = 0;

function check(name, cond, detail = "") {
    if (cond) {
        console.log(`  PASS  ${name}`);
    } else {
        failures++;
        console.log(`  FAIL  ${name}${detail ? `\n        ${detail}` : ""}`);
    }
}

const SLOTS = 6;

/** A node carrying the same widgets and sockets as the Python side. */
function makeNode(values = {}, wired = []) {
    const widgets = [];
    const add = (name, value) => widgets.push({ name, value, options: {} });

    add("h3_mode", "ref");
    add("subjects", "");
    add("target_description", "the courier ducks under a shutter");
    add("vlm_provider", "anthropic");
    add("model", "auto");
    add("fps", 24.0);
    add("api_key", "");
    add("video_role", "subject_source");
    add("audio_role", "none");
    add("cut_times", "");
    add("dialogue", "");
    add("constraint_notes", "");
    add("duration_override", 0.0);
    add("max_frames_to_analyze", 8);
    add("seed", 0);
    add("base_picture_role", "first_frame");
    add("fl2va_normalize_picture_tags", false);
    add("snap_duration_to_h3_grid", true);
    add("subject_rows", 2);
    for (let slot = 1; slot <= SLOTS; slot++) {
        add(`subject_${slot}_kind`, slot === 2 ? "environment" : "character");
        add(`subject_${slot}_name`, "");
        add(`subject_${slot}_description`, "");
    }
    add("music_video", false);
    add("music_source", "auto");
    add("lyrics", "");
    add("music_description", "");

    for (const widget of widgets) {
        if (widget.name in values) widget.value = values[widget.name];
    }

    const inputs = ["video", "frames", "audio"].concat(
        Array.from({ length: SLOTS }, (_, i) => `subject_${i + 1}_image`),
    ).map((name) => ({ name, link: wired.includes(name) ? 1 : null }));

    const node = {
        constructor: { comfyClass: "TrentUltimateH3CowboyPromptor" },
        widgets,
        inputs,
        size: [400, 600],
        computeSize: () => [400, 600],
        setSize() {},
        setDirtyCanvas() {},
    };
    app.extension.nodeCreated(node);
    return node;
}

const widget = (node, name) => node.widgets.find((w) => w.name === name);
const isHidden = (node, name) => !!widget(node, name)?.hidden;
const shown = (node, name) => !isHidden(node, name);

// ---------------------------------------------------------------------------
// The rule that protects saved workflows
// ---------------------------------------------------------------------------

console.log("\nWidget array");

{
    const node = makeNode();
    const before = node.widgets.length;
    widget(node, "h3_mode").value = "base_T2VA";
    widget(node, "h3_mode").callback?.("base_T2VA");
    check(
        "hiding never removes a widget from node.widgets",
        node.widgets.length === before,
        `${before} -> ${node.widgets.length}`,
    );
    check(
        "both spellings of hidden are written",
        widget(node, "subject_1_kind").hidden === true &&
            widget(node, "subject_1_kind").options.hidden === true,
    );
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

console.log("\nSubject rows");

{
    const node = makeNode();
    check("row 1 shows on a fresh node", shown(node, "subject_1_description"));
    check("row 2 shows on a fresh node", shown(node, "subject_2_description"));
    check("row 3 stays out of the way", isHidden(node, "subject_3_description"));
    check("row 6 stays out of the way", isHidden(node, "subject_6_kind"));
}

{
    // Fill the last visible row and the next one appears - that is the
    // whole "expands as you use it" behaviour.
    const node = makeNode();
    const desc = widget(node, "subject_2_description");
    desc.value = "wet concrete, sodium toplight";
    desc.callback?.(desc.value);
    check("filling the last row reveals the next", shown(node, "subject_3_kind"));
    check(
        "the row count follows",
        widget(node, "subject_rows").value === 3,
        String(widget(node, "subject_rows").value),
    );
}

{
    const node = makeNode({}, ["subject_4_image"]);
    check("wiring an image opens its row", shown(node, "subject_4_description"));
    check("and the rows before it", shown(node, "subject_3_description"));
}

{
    // A row holding text must never be hidden: Python reads all six, so
    // a hidden filled row is a subject nobody can see or edit.
    const node = makeNode({
        subject_5_description: "16mm grain, halation",
        subject_rows: 1,
    });
    check("a filled row is never hidden", shown(node, "subject_5_description"));
}

// ---------------------------------------------------------------------------
// Modes and music
// ---------------------------------------------------------------------------

console.log("\nModes and music");

{
    const node = makeNode();
    check("music fields are folded away", isHidden(node, "lyrics"));
    const toggle = widget(node, "music_video");
    toggle.value = true;
    toggle.callback?.(true);
    check("music_video on reveals lyrics", shown(node, "lyrics"));
    check("and the track description", shown(node, "music_description"));
    check("and where the song comes from", shown(node, "music_source"));
}

{
    const node = makeNode({ h3_mode: "base_FL2VA" });
    check("base mode hides the rows", isHidden(node, "subject_1_description"));
    check("base mode hides the typed subjects", isHidden(node, "subjects"));
    check("base mode hides music_video", isHidden(node, "music_video"));
    check("base mode shows its own settings", shown(node, "base_picture_role"));
}

{
    const node = makeNode();
    check("ref mode hides the base settings", isHidden(node, "base_picture_role"));
}

// ---------------------------------------------------------------------------
// Widgets that describe a socket
// ---------------------------------------------------------------------------

console.log("\nSocket-driven widgets");

{
    const node = makeNode();
    check("fps hides with no frames wired", isHidden(node, "fps"));
    check("video_role hides with no clip", isHidden(node, "video_role"));
    check("audio_role hides with no audio", isHidden(node, "audio_role"));
}

{
    const node = makeNode({}, ["frames", "audio"]);
    check("fps shows for a frames batch", shown(node, "fps"));
    check("video_role shows once a clip is wired", shown(node, "video_role"));
    check("audio_role shows once audio is wired", shown(node, "audio_role"));
}

console.log(
    failures ? `\n${failures} failure(s)\n` : "\nAll h3 cowboy JS tests passed.\n",
);
process.exit(failures ? 1 : 0);
