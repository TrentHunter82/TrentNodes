/**
 * H3 Hermes Prompt Director - show only the widgets this run uses.
 *
 * The node carries every widget both modes can need: six subject rows of
 * three widgets each, a music-video block, and the base-mode settings.
 * Drawn all at once that is a wall. This hides the ones that cannot do
 * anything right now, so the face is the size of the job in front of you:
 *
 *   - subject rows appear one at a time. Fill the last one and the next
 *     appears; wire subject_4_image and rows 1-4 are there waiting.
 *   - the music fields only exist once music_video is on.
 *   - ref-mode widgets vanish in the base_* modes, and the base-mode
 *     settings vanish in ref mode. Python still warns if one is set - the
 *     hiding is a courtesy, never the rule.
 *
 * Nothing is ever removed from node.widgets. A workflow stores widget
 * values POSITIONALLY, so a widget missing from the array at save time
 * hands its neighbours the wrong values on reload - the bug Multi-Load
 * Cowboy was built twice to fix. Setting `hidden` keeps the array whole:
 * the frontend's own isWidgetVisible()/getLayoutWidgets() skip it, and
 * DOM widgets (the multiline boxes) check the same flag.
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "TrentH3HermesPromptDirector";
const SLOTS = 6;

/* Widgets that only ref mode reads. */
const REF_ONLY = ["subjects", "video_role", "audio_role", "music_video"];
/* Shown once music_video is on. */
const MUSIC_EXTRA = ["music_source", "lyrics", "music_description"];
/* Widgets that only the base_* modes read. */
const BASE_ONLY = ["base_picture_role", "fl2va_normalize_picture_tags"];
/*
 * Widgets that describe an input, and so mean nothing until it is wired.
 * fps describes the frames batch; a VIDEO carries its own rate. The two
 * role pickers say what a wired asset is FOR, which is not a question
 * until something is on the socket.
 */
const INPUT_DRIVEN = {
    fps: (node) => isWired(node, "frames"),
    video_role: (node) => isWired(node, "video") || isWired(node, "frames"),
    audio_role: (node) => isWired(node, "audio"),
};

const rowWidgets = (slot) => [
    `subject_${slot}_kind`,
    `subject_${slot}_name`,
    `subject_${slot}_description`,
];

/* Every widget this extension owns the visibility of. */
const MANAGED = [
    ...REF_ONLY,
    ...MUSIC_EXTRA,
    ...BASE_ONLY,
    ...Object.keys(INPUT_DRIVEN),
    "subject_rows",
    ...Array.from({ length: SLOTS }, (_, i) => rowWidgets(i + 1)).flat(),
];

const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));

const findWidget = (node, name) =>
    node.widgets?.find((w) => w.name === name);

/**
 * Hide or show one widget.
 *
 * Both spellings are written: the canvas renderer reads widget.hidden,
 * and the Vue node renderer reads options.hidden. Setting one and not
 * the other leaves a widget that vanishes in one and not the other.
 */
function setHidden(widget, hidden) {
    if (!widget) return;
    widget.hidden = hidden;
    widget.options = widget.options || {};
    widget.options.hidden = hidden;
}

const hasText = (node, name) => {
    const value = findWidget(node, name)?.value;
    return typeof value === "string" && value.trim().length > 0;
};

const isWired = (node, name) =>
    (node.inputs || []).some(
        (input) => input.name === name && input.link != null
    );

const isImageWired = (node, slot) => isWired(node, `subject_${slot}_image`);

/** The highest row that holds anything - typed, or with an image wired. */
function highestUsedRow(node) {
    let highest = 0;
    for (let slot = 1; slot <= SLOTS; slot++) {
        if (
            hasText(node, `subject_${slot}_name`) ||
            hasText(node, `subject_${slot}_description`) ||
            isImageWired(node, slot)
        ) {
            highest = slot;
        }
    }
    return highest;
}

/**
 * How many rows to draw.
 *
 * Always one spare after the last used row, so there is somewhere to
 * type the next subject without hunting for a count. A used row is never
 * hidden, whatever the count says - Python reads all six rows, so a row
 * that holds text and cannot be seen is a subject nobody can edit.
 */
function visibleRows(node) {
    const asked = Number(findWidget(node, "subject_rows")?.value ?? 2);
    const used = highestUsedRow(node);
    return clamp(Math.max(asked, used + 1, 1), 1, SLOTS);
}

function applyVisibility(node) {
    if (!node.widgets) return;

    const mode = String(findWidget(node, "h3_mode")?.value ?? "ref");
    const isRef = mode === "ref";
    const music = isRef && !!findWidget(node, "music_video")?.value;
    const rows = visibleRows(node);

    const countWidget = findWidget(node, "subject_rows");
    if (countWidget && countWidget.value !== rows) {
        countWidget.value = rows;
    }

    const hidden = new Set(MANAGED);
    const show = (name) => hidden.delete(name);

    if (isRef) {
        REF_ONLY.forEach(show);
        show("subject_rows");
        for (let slot = 1; slot <= rows; slot++) rowWidgets(slot).forEach(show);
        if (music) MUSIC_EXTRA.forEach(show);
    } else {
        BASE_ONLY.forEach(show);
    }
    // These describe a socket, so they follow the socket, not the mode.
    for (const [name, isWanted] of Object.entries(INPUT_DRIVEN)) {
        if (isWanted(node) && (isRef || name === "fps")) show(name);
        else hidden.add(name);
    }

    let changed = false;
    for (const name of MANAGED) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        const wanted = hidden.has(name);
        if (!!widget.hidden !== wanted) changed = true;
        setHidden(widget, wanted);
    }
    if (!changed) return;

    // Keep the width the user chose; the height follows what is left.
    requestAnimationFrame(() => {
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), size[1]]);
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas(true, true);
    });
}

/** Re-check after any change that could add or remove a row. */
function watch(node) {
    const schedule = () => applyVisibility(node);

    for (const widget of node.widgets || []) {
        const isTrigger =
            widget.name === "h3_mode" ||
            widget.name === "music_video" ||
            widget.name === "subject_rows" ||
            /^subject_\d+_(name|description)$/.test(widget.name);
        if (!isTrigger) continue;
        const original = widget.callback;
        widget.callback = function (...args) {
            const out = original?.apply(this, args);
            schedule();
            return out;
        };
    }

    const onConnections = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const out = onConnections?.apply(this, args);
        schedule();
        return out;
    };

    // A textarea typed into does not always fire its callback, so catch
    // the graph's own change notification too.
    const onWidgetChanged = node.onWidgetChanged;
    node.onWidgetChanged = function (...args) {
        const out = onWidgetChanged?.apply(this, args);
        schedule();
        return out;
    };
}

app.registerExtension({
    name: "Trent.H3HermesPromptDirector",

    async nodeCreated(node) {
        if (node.constructor?.comfyClass !== NODE_NAME) return;
        watch(node);
        applyVisibility(node);

        // A loaded workflow restores its values after creation, so the
        // first pass above ran against the defaults. Re-run once the
        // real values are in.
        const onConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const out = onConfigure?.apply(this, args);
            requestAnimationFrame(() => applyVisibility(this));
            return out;
        };
    },
});
