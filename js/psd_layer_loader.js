/**
 * PSD Layer Loader - Browse button for folder_path.
 *
 * Reuses the same PathBrowser pattern from psd_layer_splitter.js.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* ---- Shared CSS (idempotent) ---- */

const BROWSER_CSS = `
.trent-folder-browser {
    position: fixed;
    background: #1a1a1a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 12px;
    z-index: 10000;
    min-width: 400px;
    max-width: 600px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, sans-serif;
}
.trent-folder-browser .browser-header {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 8px;
}
.trent-folder-browser .browser-header .title {
    font-size: 12px; color: #888; flex-shrink: 0;
}
.trent-folder-browser .browser-input {
    flex: 1; background: #2a2a2a;
    border: 1px solid #444; border-radius: 4px;
    padding: 8px 10px; color: #fff;
    font-size: 13px; font-family: monospace; outline: none;
}
.trent-folder-browser .browser-input:focus {
    border-color: #5c9cff;
}
.trent-folder-browser .browser-list {
    max-height: 300px; overflow-y: auto;
    margin-top: 8px; border: 1px solid #333;
    border-radius: 4px; background: #222;
}
.trent-folder-browser .browser-item {
    padding: 6px 10px; cursor: pointer;
    font-size: 13px; color: #ccc;
    border-bottom: 1px solid #333;
    display: flex; align-items: center; gap: 8px;
}
.trent-folder-browser .browser-item:last-child {
    border-bottom: none;
}
.trent-folder-browser .browser-item:hover {
    background: #333;
}
.trent-folder-browser .browser-item.is-dir {
    color: #5c9cff; font-weight: 500;
}
.trent-folder-browser .browser-item .icon {
    font-size: 14px; width: 18px; text-align: center;
}
.trent-folder-browser .browser-buttons {
    display: flex; gap: 8px; margin-top: 10px;
    justify-content: flex-end;
}
.trent-folder-browser .browser-btn {
    padding: 6px 16px; border: none;
    border-radius: 4px; cursor: pointer;
    font-size: 12px; font-weight: 500;
}
.trent-folder-browser .browser-btn.primary {
    background: #5c9cff; color: #fff;
}
.trent-folder-browser .browser-btn.primary:hover {
    background: #4a8ae6;
}
.trent-folder-browser .browser-btn.secondary {
    background: #444; color: #ccc;
}
.trent-folder-browser .browser-btn.secondary:hover {
    background: #555;
}
.trent-folder-browser .browser-status {
    font-size: 11px; color: #666;
    margin-top: 8px; text-align: center;
}
.trent-folder-browser .browser-status.valid {
    color: #4caf50;
}
.trent-folder-browser .browser-status.invalid {
    color: #f44336;
}
`;

let cssInjected = false;
function injectCSS() {
    if (cssInjected) return;
    if (document.getElementById("trent-folder-browser-css")) {
        cssInjected = true;
        return;
    }
    cssInjected = true;
    const style = document.createElement("style");
    style.id = "trent-folder-browser-css";
    style.textContent = BROWSER_CSS;
    document.head.appendChild(style);
}

/* ---- Path helpers ---- */

function getParentPath(path) {
    if (!path) return "";
    path = path.replace(/\\/g, "/");
    if (path.endsWith("/") && path.length > 1) {
        path = path.slice(0, -1);
    }
    const last = path.lastIndexOf("/");
    if (last <= 0) return "/";
    return path.slice(0, last + 1);
}

function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}

/* ---- Folder browser ---- */

class FolderBrowser {
    constructor(initialPath, onSelect) {
        this.currentPath = initialPath || "";
        this.onSelect = onSelect;
        this.dialog = null;
        this.items = [];
    }

    async show(event) {
        injectCSS();
        this.dialog = document.createElement("div");
        this.dialog.className = "trent-folder-browser";
        this.dialog.innerHTML = `
            <div class="browser-header">
                <span class="title">Layer Folder:</span>
                <input type="text" class="browser-input"
                       value="${escapeHtml(this.currentPath)}" />
            </div>
            <div class="browser-list"></div>
            <div class="browser-status"></div>
            <div class="browser-buttons">
                <button class="browser-btn secondary"
                        data-action="cancel">Cancel</button>
                <button class="browser-btn primary"
                        data-action="select">Select Folder</button>
            </div>
        `;
        document.body.appendChild(this.dialog);

        const x = event?.clientX ?? window.innerWidth / 2 - 200;
        const y = event?.clientY ?? window.innerHeight / 2 - 200;
        this.dialog.style.left =
            Math.min(Math.max(10, x), window.innerWidth - 430) + "px";
        this.dialog.style.top =
            Math.min(Math.max(10, y), window.innerHeight - 410) + "px";

        this.input = this.dialog.querySelector(".browser-input");
        this.list = this.dialog.querySelector(".browser-list");
        this.status = this.dialog.querySelector(".browser-status");

        this.input.addEventListener("input", () => this.onInputChange());
        this.input.addEventListener("keydown", (e) => this.onKeyDown(e));
        this.dialog.querySelector('[data-action="cancel"]')
            .addEventListener("click", () => this.close());
        this.dialog.querySelector('[data-action="select"]')
            .addEventListener("click", () => this.selectCurrent());

        this._onClickOutside = (e) => {
            if (this.dialog && !this.dialog.contains(e.target)) {
                this.close();
            }
        };
        setTimeout(() => {
            document.addEventListener("mousedown", this._onClickOutside);
        }, 100);

        this.input.focus();
        this.input.select();
        await this.loadPath(this.currentPath);
    }

    close() {
        document.removeEventListener("mousedown", this._onClickOutside);
        if (this.dialog) { this.dialog.remove(); this.dialog = null; }
    }

    async onInputChange() {
        this.currentPath = this.input.value;
        await this.loadPath(this.currentPath);
    }

    onKeyDown(e) {
        if (e.key === "Escape") this.close();
        else if (e.key === "Enter") this.selectCurrent();
        else if (e.key === "Tab" && this.items.length > 0) {
            e.preventDefault();
            const firstDir = this.items.find(
                i => i.endsWith("/") || i.endsWith("\\")
            );
            if (firstDir) this.navigateTo(firstDir);
        } else if (e.key === "Backspace" && e.ctrlKey) {
            e.preventDefault();
            this.currentPath = getParentPath(this.currentPath);
            this.input.value = this.currentPath;
            this.loadPath(this.currentPath);
        }
    }

    async loadPath(path) {
        try {
            const params = new URLSearchParams({ path: path || "" });
            const resp = await api.fetchApi("/trent/browse?" + params);
            this.items = await resp.json();
            this.renderList();
            await this.validatePath(path);
        } catch (err) {
            this.items = [];
            this.renderList();
            this.status.textContent = "Error loading path";
            this.status.className = "browser-status invalid";
        }
    }

    async validatePath(path) {
        if (!path) {
            this.status.textContent = "Select a layer folder";
            this.status.className = "browser-status";
            return;
        }
        try {
            const params = new URLSearchParams({
                path, type: "dir"
            });
            const resp = await api.fetchApi(
                "/trent/validate_path?" + params
            );
            const result = await resp.json();
            if (result.valid && result.type === "dir") {
                this.status.textContent =
                    `Valid folder - ${result.count} file(s)`;
                this.status.className = "browser-status valid";
            } else {
                this.status.textContent = "Not a valid folder";
                this.status.className = "browser-status invalid";
            }
        } catch {
            this.status.textContent = "Could not validate";
            this.status.className = "browser-status";
        }
    }

    renderList() {
        this.list.innerHTML = "";
        if (this.currentPath && this.currentPath !== "/") {
            const up = document.createElement("div");
            up.className = "browser-item is-dir";
            up.innerHTML =
                '<span class="icon">..</span>' +
                '<span>Parent Directory</span>';
            up.addEventListener("click", () => {
                this.currentPath = getParentPath(this.currentPath);
                this.input.value = this.currentPath;
                this.loadPath(this.currentPath);
            });
            this.list.appendChild(up);
        }
        for (const item of this.items) {
            const isDir = item.endsWith("/") || item.endsWith("\\");
            const div = document.createElement("div");
            div.className = "browser-item" + (isDir ? " is-dir" : "");
            const icon = isDir ? "[D]" : "[F]";
            const display = item.replace(/[/\\]$/, "");
            div.innerHTML =
                `<span class="icon">${icon}</span>` +
                `<span>${escapeHtml(display)}</span>`;
            if (isDir) {
                div.addEventListener("click", () => this.navigateTo(item));
            }
            this.list.appendChild(div);
        }
        if (this.items.length === 0 && this.currentPath) {
            const empty = document.createElement("div");
            empty.className = "browser-item";
            empty.style.color = "#666";
            empty.style.fontStyle = "italic";
            empty.textContent = "Empty or inaccessible";
            this.list.appendChild(empty);
        }
    }

    navigateTo(item) {
        let base = this.currentPath || "";
        if (base && !base.endsWith("/") && !base.endsWith("\\")) {
            base += "/";
        }
        this.currentPath = base + item;
        this.input.value = this.currentPath;
        this.loadPath(this.currentPath);
    }

    selectCurrent() {
        let path = this.currentPath;
        if (path.endsWith("/") || path.endsWith("\\")) {
            path = path.slice(0, -1);
        }
        if (this.onSelect) this.onSelect(path);
        this.close();
    }
}

/* ---- Register ---- */

app.registerExtension({
    name: "Trent.PSDLayerLoader",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "PSDLayerLoader") return;

        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = orig?.apply(this, arguments);

            const folderWidget = this.widgets?.find(
                w => w.name === "folder_path"
            );
            if (!folderWidget) return result;

            const node = this;
            const btn = this.addWidget(
                "button",
                "browse_folder",
                "Browse...",
                function () {
                    const browser = new FolderBrowser(
                        folderWidget.value,
                        (selected) => {
                            folderWidget.value = selected;
                            if (folderWidget.callback) {
                                folderWidget.callback(selected);
                            }
                            node.setDirtyCanvas(true, true);
                        }
                    );
                    browser.show();
                }
            );

            const wi = this.widgets.indexOf(folderWidget);
            const bi = this.widgets.indexOf(btn);
            if (wi >= 0 && bi > wi + 1) {
                this.widgets.splice(bi, 1);
                this.widgets.splice(wi + 1, 0, btn);
            }

            return result;
        };
    },
});

console.log("[TrentNodes] PSDLayerLoader extension loaded");
