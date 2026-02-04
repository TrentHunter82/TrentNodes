/**
 * Image Folder Cowboy - Folder browser widget
 *
 * Adds a browse button next to the directory input.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// CSS for the folder browser dialog
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
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.trent-folder-browser .browser-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.trent-folder-browser .browser-header .title {
    font-size: 12px;
    color: #888;
    flex-shrink: 0;
}

.trent-folder-browser .browser-input {
    flex: 1;
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 8px 10px;
    color: #fff;
    font-size: 13px;
    font-family: monospace;
    outline: none;
}

.trent-folder-browser .browser-input:focus {
    border-color: #5c9cff;
}

.trent-folder-browser .browser-list {
    max-height: 300px;
    overflow-y: auto;
    margin-top: 8px;
    border: 1px solid #333;
    border-radius: 4px;
    background: #222;
}

.trent-folder-browser .browser-item {
    padding: 6px 10px;
    cursor: pointer;
    font-size: 13px;
    color: #ccc;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: center;
    gap: 8px;
}

.trent-folder-browser .browser-item:last-child {
    border-bottom: none;
}

.trent-folder-browser .browser-item:hover {
    background: #333;
}

.trent-folder-browser .browser-item.is-dir {
    color: #5c9cff;
    font-weight: 500;
}

.trent-folder-browser .browser-item .icon {
    font-size: 14px;
    width: 18px;
    text-align: center;
}

.trent-folder-browser .browser-buttons {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    justify-content: flex-end;
}

.trent-folder-browser .browser-btn {
    padding: 6px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
}

.trent-folder-browser .browser-btn.primary {
    background: #5c9cff;
    color: #fff;
}

.trent-folder-browser .browser-btn.primary:hover {
    background: #4a8ae6;
}

.trent-folder-browser .browser-btn.secondary {
    background: #444;
    color: #ccc;
}

.trent-folder-browser .browser-btn.secondary:hover {
    background: #555;
}

.trent-folder-browser .browser-status {
    font-size: 11px;
    color: #666;
    margin-top: 8px;
    text-align: center;
}

.trent-folder-browser .browser-status.valid {
    color: #4caf50;
}

.trent-folder-browser .browser-status.invalid {
    color: #f44336;
}
`;

// Inject CSS once
let cssInjected = false;
function injectCSS() {
    if (cssInjected) return;
    cssInjected = true;
    const style = document.createElement("style");
    style.id = "trent-folder-browser-css";
    style.textContent = BROWSER_CSS;
    document.head.appendChild(style);
}

// Path utilities
function getParentPath(path) {
    if (!path) return "";
    // Normalize separators
    path = path.replace(/\\/g, "/");
    // Remove trailing slash
    if (path.endsWith("/") && path.length > 1) {
        path = path.slice(0, -1);
    }
    const lastSlash = path.lastIndexOf("/");
    if (lastSlash <= 0) return "/";
    return path.slice(0, lastSlash + 1);
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// Folder browser dialog
class FolderBrowser {
    constructor(initialPath, onSelect) {
        this.currentPath = initialPath || "";
        this.onSelect = onSelect;
        this.dialog = null;
        this.items = [];
    }

    async show(event) {
        injectCSS();

        // Create dialog
        this.dialog = document.createElement("div");
        this.dialog.className = "trent-folder-browser";

        this.dialog.innerHTML = `
            <div class="browser-header">
                <span class="title">Path:</span>
                <input type="text" class="browser-input" value="${escapeHtml(this.currentPath)}" />
            </div>
            <div class="browser-list"></div>
            <div class="browser-status"></div>
            <div class="browser-buttons">
                <button class="browser-btn secondary" data-action="cancel">Cancel</button>
                <button class="browser-btn primary" data-action="select">Select Folder</button>
            </div>
        `;

        document.body.appendChild(this.dialog);

        // Position near cursor or center
        const x = event?.clientX ?? window.innerWidth / 2 - 200;
        const y = event?.clientY ?? window.innerHeight / 2 - 200;

        // Keep on screen
        const dialogWidth = 420;
        const dialogHeight = 400;
        const finalX = Math.min(Math.max(10, x), window.innerWidth - dialogWidth - 10);
        const finalY = Math.min(Math.max(10, y), window.innerHeight - dialogHeight - 10);

        this.dialog.style.left = finalX + "px";
        this.dialog.style.top = finalY + "px";

        // Get elements
        this.input = this.dialog.querySelector(".browser-input");
        this.list = this.dialog.querySelector(".browser-list");
        this.status = this.dialog.querySelector(".browser-status");

        // Event handlers
        this.input.addEventListener("input", () => this.onInputChange());
        this.input.addEventListener("keydown", (e) => this.onKeyDown(e));

        this.dialog.querySelector('[data-action="cancel"]').addEventListener("click", () => this.close());
        this.dialog.querySelector('[data-action="select"]').addEventListener("click", () => this.selectCurrent());

        // Close on click outside
        this._onClickOutside = (e) => {
            if (this.dialog && !this.dialog.contains(e.target)) {
                this.close();
            }
        };
        setTimeout(() => {
            document.addEventListener("mousedown", this._onClickOutside);
        }, 100);

        // Focus input
        this.input.focus();
        this.input.select();

        // Load initial listing
        await this.loadPath(this.currentPath);
    }

    close() {
        document.removeEventListener("mousedown", this._onClickOutside);
        if (this.dialog) {
            this.dialog.remove();
            this.dialog = null;
        }
    }

    async onInputChange() {
        this.currentPath = this.input.value;
        await this.loadPath(this.currentPath);
    }

    onKeyDown(e) {
        if (e.key === "Escape") {
            this.close();
        } else if (e.key === "Enter") {
            this.selectCurrent();
        } else if (e.key === "Tab" && this.items.length > 0) {
            e.preventDefault();
            const firstDir = this.items.find(i => i.endsWith("/") || i.endsWith("\\"));
            if (firstDir) {
                this.navigateTo(firstDir);
            }
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
            const response = await api.fetchApi("/trent/browse?" + params);
            this.items = await response.json();
            this.renderList();
            await this.validatePath(path);
        } catch (err) {
            console.error("[ImageFolderCowboy] Failed to load path:", err);
            this.items = [];
            this.renderList();
            this.status.textContent = "Error loading directory";
            this.status.className = "browser-status invalid";
        }
    }

    async validatePath(path) {
        if (!path) {
            this.status.textContent = "Enter a path or select a folder";
            this.status.className = "browser-status";
            return;
        }

        try {
            const params = new URLSearchParams({ path, type: "dir" });
            const response = await api.fetchApi("/trent/validate_path?" + params);
            const result = await response.json();

            if (result.valid && result.type === "dir") {
                this.status.textContent = `Valid folder - ${result.count} image(s) found`;
                this.status.className = "browser-status valid";
            } else {
                this.status.textContent = "Not a valid folder";
                this.status.className = "browser-status invalid";
            }
        } catch (err) {
            this.status.textContent = "Could not validate path";
            this.status.className = "browser-status";
        }
    }

    renderList() {
        this.list.innerHTML = "";

        // Add parent directory option
        if (this.currentPath && this.currentPath !== "/" && this.currentPath !== "") {
            const parentItem = document.createElement("div");
            parentItem.className = "browser-item is-dir";
            parentItem.innerHTML = '<span class="icon">..</span><span>Parent Directory</span>';
            parentItem.addEventListener("click", () => {
                this.currentPath = getParentPath(this.currentPath);
                this.input.value = this.currentPath;
                this.loadPath(this.currentPath);
            });
            this.list.appendChild(parentItem);
        }

        for (const item of this.items) {
            const isDir = item.endsWith("/") || item.endsWith("\\");
            const div = document.createElement("div");
            div.className = "browser-item" + (isDir ? " is-dir" : "");

            const icon = isDir ? "[D]" : "[F]";
            const displayName = item.replace(/[/\\]$/, "");

            div.innerHTML = `<span class="icon">${icon}</span><span>${escapeHtml(displayName)}</span>`;

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
        let basePath = this.currentPath || "";
        if (basePath && !basePath.endsWith("/") && !basePath.endsWith("\\")) {
            basePath += "/";
        }
        this.currentPath = basePath + item;
        this.input.value = this.currentPath;
        this.loadPath(this.currentPath);
    }

    selectCurrent() {
        let path = this.currentPath;
        // Remove trailing slash for consistency
        if (path.endsWith("/") || path.endsWith("\\")) {
            path = path.slice(0, -1);
        }
        if (this.onSelect) {
            this.onSelect(path);
        }
        this.close();
    }
}

// Register extension
app.registerExtension({
    name: "Trent.ImageFolderCowboy",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ImageFolderCowboy") {
            return;
        }

        console.log("[TrentNodes] Registering ImageFolderCowboy browse button");

        // Chain onto onNodeCreated to add browse button
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = origOnNodeCreated?.apply(this, arguments);

            // Find directory widget
            const dirWidget = this.widgets?.find(w => w.name === "directory");
            if (!dirWidget) {
                console.warn("[ImageFolderCowboy] Could not find directory widget");
                return result;
            }

            // Add a browse button widget after the directory widget
            const node = this;
            const browseWidget = this.addWidget("button", "browse_folder", "Browse...", function() {
                const browser = new FolderBrowser(dirWidget.value, (selectedPath) => {
                    dirWidget.value = selectedPath;
                    if (dirWidget.callback) {
                        dirWidget.callback(selectedPath);
                    }
                    node.setDirtyCanvas(true, true);
                });
                browser.show();
            });

            // Move browse button right after directory widget
            const dirIndex = this.widgets.indexOf(dirWidget);
            const browseIndex = this.widgets.indexOf(browseWidget);
            if (dirIndex >= 0 && browseIndex > dirIndex + 1) {
                // Remove and reinsert at correct position
                this.widgets.splice(browseIndex, 1);
                this.widgets.splice(dirIndex + 1, 0, browseWidget);
            }

            return result;
        };
    }
});

console.log("[TrentNodes] ImageFolderCowboy extension loaded");
