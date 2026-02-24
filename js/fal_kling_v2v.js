import { app } from "/scripts/app.js";

/**
 * FAL Kling V2V - @ Tag Autocomplete
 *
 * Adds autocomplete to the prompt widget. When the user
 * types "@", a dropdown appears with available FAL tags
 * based on which image inputs are currently connected.
 */

const TAG_MAP = {
    video:          "@Video1",
    ref_image_1:    "@Image1",
    ref_image_2:    "@Image2",
    element_1_face: "@Element1",
    element_2_face: "@Element2",
};

// Inputs where connection is required to show the tag.
// "video" is always shown since it is a required input.
const ALWAYS_SHOW = new Set(["video"]);

app.registerExtension({
    name: "TrentNodes.FalKlingV2VAutocomplete",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "FalKlingV2V") {
            return;
        }

        const promptWidget = node.widgets?.find(
            (w) => w.name === "prompt"
        );
        if (!promptWidget || !promptWidget.inputEl) {
            return;
        }

        const el = promptWidget.inputEl;
        let dropdown = null;
        let selectedIdx = 0;

        // ------------------------------------------------
        // helpers
        // ------------------------------------------------

        /**
         * Check if a named input slot has a link.
         */
        const isConnected = (name) => {
            const slot = node.inputs?.find(
                (i) => i.name === name
            );
            return slot && slot.link !== null;
        };

        /**
         * Build the list of available tags right now.
         */
        const getAvailableTags = () => {
            const tags = [];
            for (const [input, tag] of Object.entries(TAG_MAP)) {
                if (ALWAYS_SHOW.has(input) || isConnected(input)) {
                    tags.push({ input, tag });
                }
            }
            return tags;
        };

        /**
         * Get the partial word being typed at the cursor,
         * starting from the last "@".
         * Returns { start, text } or null.
         */
        const getAtWord = () => {
            const pos = el.selectionStart;
            const before = el.value.substring(0, pos);
            const atIdx = before.lastIndexOf("@");
            if (atIdx === -1) return null;
            // Only trigger if @ is at start or preceded by
            // whitespace (don't trigger inside a URL etc.)
            if (atIdx > 0 && !/\s/.test(before[atIdx - 1])) {
                return null;
            }
            return {
                start: atIdx,
                text: before.substring(atIdx),
            };
        };

        /**
         * Filter tags by the partial text after @.
         */
        const filterTags = (partial) => {
            const q = partial.toLowerCase().replace(/^@/, "");
            const tags = getAvailableTags();
            if (!q) return tags;
            return tags.filter((t) =>
                t.tag.toLowerCase().includes(q) ||
                t.input.toLowerCase().includes(q)
            );
        };

        // ------------------------------------------------
        // dropdown UI
        // ------------------------------------------------

        const createDropdown = () => {
            removeDropdown();
            dropdown = document.createElement("ul");
            Object.assign(dropdown.style, {
                position: "fixed",
                zIndex: "99999",
                background: "#1e1e2e",
                border: "1px solid #444",
                borderRadius: "6px",
                padding: "4px 0",
                margin: "0",
                listStyle: "none",
                minWidth: "160px",
                maxHeight: "200px",
                overflowY: "auto",
                boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
                fontFamily: "monospace",
                fontSize: "13px",
            });
            document.body.appendChild(dropdown);
            positionDropdown();
        };

        const positionDropdown = () => {
            if (!dropdown) return;
            const rect = el.getBoundingClientRect();
            dropdown.style.left = rect.left + "px";
            dropdown.style.top = (rect.bottom + 2) + "px";
        };

        const removeDropdown = () => {
            if (dropdown) {
                dropdown.remove();
                dropdown = null;
            }
            selectedIdx = 0;
        };

        const renderItems = (items) => {
            if (!dropdown) return;
            dropdown.innerHTML = "";
            items.forEach((item, idx) => {
                const li = document.createElement("li");
                const connected = isConnected(item.input)
                    || ALWAYS_SHOW.has(item.input);
                const badge = connected ? " *" : "";
                li.textContent = item.tag + badge;
                Object.assign(li.style, {
                    padding: "4px 12px",
                    cursor: "pointer",
                    color: connected ? "#a6e3a1" : "#888",
                    background:
                        idx === selectedIdx
                            ? "#313244"
                            : "transparent",
                });
                li.addEventListener("mouseenter", () => {
                    selectedIdx = idx;
                    highlightSelected(items);
                });
                li.addEventListener("mousedown", (e) => {
                    e.preventDefault();
                    insertTag(item.tag);
                });
                dropdown.appendChild(li);
            });
        };

        const highlightSelected = (items) => {
            if (!dropdown) return;
            const children = dropdown.children;
            for (let i = 0; i < children.length; i++) {
                children[i].style.background =
                    i === selectedIdx
                        ? "#313244"
                        : "transparent";
            }
        };

        // ------------------------------------------------
        // insertion
        // ------------------------------------------------

        const insertTag = (tag) => {
            const atWord = getAtWord();
            if (!atWord) {
                removeDropdown();
                return;
            }
            const before = el.value.substring(0, atWord.start);
            const after = el.value.substring(el.selectionStart);
            el.value = before + tag + " " + after;
            const newPos = before.length + tag.length + 1;
            el.selectionStart = newPos;
            el.selectionEnd = newPos;
            promptWidget.value = el.value;
            if (promptWidget.callback) {
                promptWidget.callback(el.value);
            }
            removeDropdown();
            el.focus();
        };

        // ------------------------------------------------
        // event handlers
        // ------------------------------------------------

        let currentItems = [];

        const onInput = () => {
            const atWord = getAtWord();
            if (!atWord) {
                removeDropdown();
                return;
            }
            currentItems = filterTags(atWord.text);
            if (currentItems.length === 0) {
                removeDropdown();
                return;
            }
            selectedIdx = 0;
            if (!dropdown) createDropdown();
            renderItems(currentItems);
        };

        const onKeyDown = (e) => {
            if (!dropdown || currentItems.length === 0) {
                return;
            }
            if (e.key === "ArrowDown") {
                e.preventDefault();
                selectedIdx =
                    (selectedIdx + 1) % currentItems.length;
                highlightSelected(currentItems);
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                selectedIdx =
                    (selectedIdx - 1 + currentItems.length) %
                    currentItems.length;
                highlightSelected(currentItems);
            } else if (e.key === "Enter" || e.key === "Tab") {
                e.preventDefault();
                insertTag(currentItems[selectedIdx].tag);
            } else if (e.key === "Escape") {
                e.preventDefault();
                removeDropdown();
            }
        };

        const onBlur = () => {
            // Small delay so mousedown on dropdown fires first
            setTimeout(removeDropdown, 150);
        };

        el.addEventListener("input", onInput);
        el.addEventListener("keydown", onKeyDown);
        el.addEventListener("blur", onBlur);
    },
});
