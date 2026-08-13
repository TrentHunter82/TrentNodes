// Checks js/multi_load_cowboy.js against a mocked ComfyUI frontend.
//
//   node tests/multi_load_cowboy/run.mjs
//
// The extension imports "../../scripts/app.js", which only resolves when
// ComfyUI serves it. So we copy the source next to the mocks in ./scripts
// and rewrite that one import, rather than keeping a second copy of the
// logic. ./scripts/dom.js stands in for the browser.

import { mkdtempSync, readFileSync, rmSync, writeFileSync, cpSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "..", "js", "multi_load_cowboy.js");

const stage = mkdtempSync(join(tmpdir(), "trentnodes-mlc-"));
process.on("exit", () => rmSync(stage, { recursive: true, force: true }));

cpSync(join(here, "scripts"), join(stage, "scripts"), { recursive: true });
writeFileSync(
    join(stage, "extension.js"),
    readFileSync(source, "utf8").replaceAll("../../scripts/", "./scripts/"),
);

const { installDOM } = await import(
    pathToFileURL(join(stage, "scripts", "dom.js"))
);
installDOM();

const { app } = await import(pathToFileURL(join(stage, "scripts", "app.js")));
const { api } = await import(pathToFileURL(join(stage, "scripts", "api.js")));
await import(pathToFileURL(join(stage, "extension.js")));

const EMPTY = "(empty)";
const SLOT_NAMES = ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6"];

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

function equal(name, actual, expected) {
    const a = JSON.stringify(actual);
    const e = JSON.stringify(expected);
    check(name, a === e, `expected ${e}\n        actual   ${a}`);
}

// ---------------------------------------------------------------------------
// A stand-in for a registered ComfyUI node
// ---------------------------------------------------------------------------

function makeNodeType() {
    const nodeType = { prototype: {} };
    app.extension.beforeRegisterNodeDef(nodeType, { name: "MultiLoadCowboy" });
    return nodeType;
}

function makeNode(values = []) {
    const widgets = SLOT_NAMES.map((name, i) => ({
        name,
        type: "combo",
        value: values[i] || EMPTY,
        options: { values: [EMPTY, "a.png", "b.png", "c.png"] },
        callback: null,
    }));
    for (const name of ["width", "height", "resize_mode"]) {
        widgets.push({
            name,
            type: name === "resize_mode" ? "combo" : "number",
            value: name === "resize_mode" ? "pad" : 1024,
            options: {},
            computeSize: () => [0, 20],
        });
    }

    return {
        widgets,
        /* Eleven outputs: images, masks, count, width, height, image_1..6. */
        outputs: Array.from({ length: 11 }, (_, i) => ({ name: `out_${i}` })),
        size: [380, 300],
        graph: { setDirtyCanvas: () => {} },
        addDOMWidget(name, type, element, options) {
            const widget = { name, type, element, options, value: undefined };
            this.widgets.push(widget);
            return widget;
        },
        computeSize() {
            let height = 60;
            for (const w of this.widgets) {
                height += w.computeSize ? w.computeSize(this.size[0])[1] + 4 : 24;
            }
            return [this.size[0], height];
        },
        setSize(size) {
            this.size = [size[0], size[1]];
        },
    };
}

function create(values) {
    const nodeType = makeNodeType();
    const node = makeNode(values);
    nodeType.prototype.onNodeCreated.call(node);
    const grid = node.widgets.find((w) => w.type === "mlc_grid");
    const lift = node.widgets.find((w) => w.type === "mlc_lift");
    const root = grid.element;
    const panel = root.findAll("mlc-panel")[0];
    return {
        nodeType, node, grid, lift, root, panel,
        cells: root.findAll("mlc-cell"),
    };
}

function slotValues(node) {
    return SLOT_NAMES.map((n) => node.widgets.find((w) => w.name === n).value);
}

// ---------------------------------------------------------------------------

console.log("Node creation");

{
    const { node, grid, lift, root, panel, cells } = create(["a.png", EMPTY, "b.png"]);

    check("grid widget was added", !!grid);
    check("lift widget was added", !!lift);
    check("lift then grid sit above the settings",
        node.widgets[0] === lift && node.widgets[1] === grid);
    check("grid is not serialised", grid.serialize === false &&
        grid.options.serialize === false);
    check("lift is not serialised", lift.serialize === false &&
        lift.options.serialize === false);
    equal("six cells drawn", cells.length, 6);

    const combos = SLOT_NAMES.map((n) => node.widgets.find((w) => w.name === n));
    check("combos are hidden",
        combos.every((w) => w.hidden === true &&
            String(w.type).includes("converted-widget")));
    equal("hidden combos take no room", combos[0].computeSize(380), [0, -4]);
    check("original combo type is kept for a rollback",
        combos.every((w) => w.origType === "combo"));

    check("filled cells are marked",
        cells[0].classList.contains("is-filled") &&
        !cells[1].classList.contains("is-filled") &&
        cells[2].classList.contains("is-filled"));
    check("empty cells are draggable only when filled",
        cells[0].draggable === true && cells[1].draggable === false);

    const count = root.findAll("mlc-count")[0];
    equal("counter reads the filled slots", count.textContent, "2 / 6");

    const size = grid.computeSize(node.size[0]);
    check("widget height comes from the layout", size[1] > 100 && size[1] < 400,
        `got ${size[1]}`);
    equal("three columns on a wide node",
        panel.style.getPropertyValue("--mlc-cols"), "3");
    check("node was widened for the grid", node.size[0] >= 380);

    /* The lift pulls the grid up beside the output labels. */
    const pull = -lift.computeSize(node.size[0])[1];
    check("lift pulls the grid up", pull > 100,
        `pull ${pull}`);
    check("lift never rides into the title bar",
        pull <= node.outputs.length * 20 - 12, `pull ${pull}`);
    check("lift matches its layout hook",
        lift.computeLayoutSize().minHeight === -pull);
    check("panel leaves room for the output labels",
        parseInt(panel.style.width) <= node.size[0] - 96,
        `panel ${panel.style.width} on node ${node.size[0]}`);
}

console.log("\nEditing slots");

{
    const { node, root, cells } = create(["a.png", "b.png"]);

    // The X on a filled cell empties it.
    const clear = cells[0].findAll("mlc-clear")[0];
    await clear.dispatch("click");
    equal("clear empties one slot", slotValues(node)[0], EMPTY);
    check("cleared cell drops its thumbnail",
        !cells[0].classList.contains("is-filled"));

    // The header broom empties everything.
    const sweep = root.findAll("mlc-sweep")[0];
    await sweep.dispatch("click");
    equal("sweep clears the grid", slotValues(node),
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]);
}

console.log("\nDrag and drop");

const file = (name) =>
    new File([new Uint8Array([1])], name, { type: "image/png" });

{
    const { node, cells } = create();
    api.uploads = [];

    await cells[2].dispatch("drop", {
        dataTransfer: { files: [file("one.png"), file("two.png")] },
    });
    equal("a drop fills from the cell it landed on", slotValues(node),
        [EMPTY, EMPTY, "one.png", "two.png", EMPTY, EMPTY]);
    equal("both files were uploaded", api.uploads, ["one.png", "two.png"]);
}

{
    const { node, root } = create(["a.png"]);
    await root.dispatch("drop", {
        dataTransfer: { files: [file("three.png")] },
    });
    equal("a drop on the panel takes the first empty slot", slotValues(node),
        ["a.png", "three.png", EMPTY, EMPTY, EMPTY, EMPTY]);
}

{
    const { node, cells } = create(["a.png", EMPTY, "b.png"]);
    await cells[0].dispatch("dragstart", {
        dataTransfer: { setData: () => {}, effectAllowed: "" },
    });
    await cells[1].dispatch("drop", { dataTransfer: { files: [] } });
    equal("dragging a cell onto an empty one moves it", slotValues(node),
        [EMPTY, "a.png", "b.png", EMPTY, EMPTY, EMPTY]);
}

{
    const { node, cells } = create(["a.png", "b.png"]);
    await cells[0].dispatch("dragstart", {
        dataTransfer: { setData: () => {}, effectAllowed: "" },
    });
    await cells[1].dispatch("drop", { dataTransfer: { files: [] } });
    equal("dragging onto a filled cell swaps the two", slotValues(node),
        ["b.png", "a.png", EMPTY, EMPTY, EMPTY, EMPTY]);
}

{
    const { node, cells } = create();
    api.uploadStatus = 500;
    app.toasts = [];
    await cells[0].dispatch("drop", {
        dataTransfer: { files: [file("bad.png")] },
    });
    api.uploadStatus = 200;
    equal("a failed upload leaves the slot alone", slotValues(node)[0], EMPTY);
    check("a failed upload tells the user", app.toasts.length === 1,
        JSON.stringify(app.toasts));
}

console.log("\nThe picker");

{
    const { node, cells } = create([EMPTY, "b.png"]);

    await cells[0].dispatch("click");
    const pop = document.body.children.find(
        (el) => el.classList.contains("mlc-pop"));
    check("clicking a cell opens the picker", !!pop);

    const options = pop.findAll("mlc-opt");
    check("the picker lists the input folder", options.length === 3,
        `got ${options.length}`);

    await options[1].dispatch("click");
    equal("picking a file fills the slot", slotValues(node)[0], "b.png");
    check("picking closes the picker",
        !document.body.children.some((el) => el.classList.contains("mlc-pop")));

    // Upload straight from the picker.
    await cells[3].dispatch("click");
    const pop2 = document.body.children.find(
        (el) => el.classList.contains("mlc-pop"));
    globalThis.__nextFiles = [file("four.png")];
    api.uploads = [];
    await pop2.findAll("mlc-btn")[0].dispatch("click");
    await new Promise((r) => setTimeout(r, 0));
    equal("the picker can upload into its own slot",
        slotValues(node)[3], "four.png");

    // Escape closes it.
    await cells[4].dispatch("click");
    for (const fn of document.listeners.get("keydown") || []) {
        fn({ key: "Escape", stopPropagation() {} });
    }
    check("escape closes the picker",
        !document.body.children.some((el) => el.classList.contains("mlc-pop")));
}

console.log("\nRedraw and resize");

{
    const { nodeType, node, grid, lift, panel, cells } = create();

    // A workflow load writes the combo values, then configure redraws.
    node.widgets.find((w) => w.name === "image_1").value = "c.png";
    nodeType.prototype.onConfigure.call(node);
    await new Promise((r) => setTimeout(r, 5));
    check("configure redraws the grid from the loaded values",
        cells[0].classList.contains("is-filled"));

    // A narrow node falls back to two columns beside the labels.
    node.size[0] = 330;
    grid.computeSize(node.size[0]);
    equal("two columns when the band gets tight",
        panel.style.getPropertyValue("--mlc-cols"), "2");

    // Narrower still: give up the band and sit under the slots full width.
    node.size[0] = 240;
    grid.computeSize(node.size[0]);
    equal("no lift when there is no room to share",
        lift.computeSize(node.size[0])[1], 0);
    check("panel goes full width when it drops below the slots",
        parseInt(panel.style.width) >= node.size[0] - 24,
        `panel ${panel.style.width} on node ${node.size[0]}`);

    // A resize can never crush the grid.
    const size = [240, 10];
    node.onResize(size);
    check("resize keeps the grid visible", size[1] > 100, `got ${size[1]}`);
}

console.log("\nOther node types are untouched");

{
    const other = { prototype: {} };
    app.extension.beforeRegisterNodeDef(other, { name: "SomethingElse" });
    check("no hooks added to a foreign node",
        Object.keys(other.prototype).length === 0);
}

console.log(failures ? `\n${failures} failed` : "\nall passed");
process.exit(failures ? 1 : 0);
