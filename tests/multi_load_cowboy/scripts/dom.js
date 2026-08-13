/**
 * The smallest DOM that js/multi_load_cowboy.js needs to run under node.
 *
 * Only the pieces the extension actually touches are here: element
 * creation, class lists, listeners with dispatch, and enough of document
 * and window for the popover to place itself.
 */

class ClassList {
    constructor(el) {
        this.el = el;
        this.set = new Set();
    }
    add(...names) {
        for (const n of names) if (n) this.set.add(n);
        this.sync();
    }
    remove(...names) {
        for (const n of names) this.set.delete(n);
        this.sync();
    }
    toggle(name, force) {
        const on = force === undefined ? !this.set.has(name) : !!force;
        if (on) this.set.add(name);
        else this.set.delete(name);
        this.sync();
        return on;
    }
    contains(name) {
        return this.set.has(name);
    }
    sync() {
        this.el._className = [...this.set].join(" ");
    }
}

export class El {
    constructor(tag) {
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.parent = null;
        this.dataset = {};
        this.style = {
            props: {},
            setProperty(name, value) {
                this.props[name] = value;
            },
            getPropertyValue(name) {
                return this.props[name];
            },
        };
        this.classList = new ClassList(this);
        this._className = "";
        this.listeners = new Map();
        this.textContent = "";
        this.innerHTML = "";
        if (this.tagName === "INPUT" || this.tagName === "TEXTAREA") {
            this.value = "";
        }
    }

    get className() {
        return this._className;
    }
    set className(value) {
        this._className = value;
        this.classList.set = new Set(String(value).split(/\s+/).filter(Boolean));
    }

    appendChild(child) {
        child.parent = this;
        this.children.push(child);
        return child;
    }
    append(...kids) {
        for (const kid of kids) this.appendChild(kid);
    }
    replaceChildren(...kids) {
        this.children = [];
        for (const kid of kids) this.appendChild(kid);
    }
    remove() {
        if (!this.parent) return;
        const at = this.parent.children.indexOf(this);
        if (at >= 0) this.parent.children.splice(at, 1);
        this.parent = null;
    }
    contains(other) {
        if (other === this) return true;
        return this.children.some((c) => c.contains(other));
    }

    setAttribute(name, value) {
        this[name] = value;
    }
    getAttribute(name) {
        return this[name];
    }
    removeAttribute(name) {
        delete this[name];
    }

    addEventListener(type, fn) {
        if (!this.listeners.has(type)) this.listeners.set(type, []);
        this.listeners.get(type).push(fn);
    }
    removeEventListener(type, fn) {
        const list = this.listeners.get(type) || [];
        const at = list.indexOf(fn);
        if (at >= 0) list.splice(at, 1);
    }
    /** Returns the promise of every handler, so async ones can be awaited. */
    dispatch(type, event = {}) {
        const base = {
            type,
            target: this,
            preventDefault() {},
            stopPropagation() {},
        };
        const merged = { ...base, ...event };
        const out = (this.listeners.get(type) || []).map((fn) =>
            fn.call(this, merged)
        );
        return Promise.all(out);
    }

    getBoundingClientRect() {
        return { left: 40, top: 60, right: 140, bottom: 140,
                 width: 100, height: 80 };
    }
    focus() {}
    click() {
        if (this.tagName === "INPUT" && this.type === "file") {
            const files = globalThis.__nextFiles || [];
            globalThis.__nextFiles = null;
            this.files = files;
            this.dispatch(files.length ? "change" : "cancel");
            return;
        }
        this.dispatch("click");
    }

    /** Test-only: every element in the tree matching a class name. */
    findAll(cls, out = []) {
        if (this.classList.contains(cls)) out.push(this);
        for (const kid of this.children) kid.findAll(cls, out);
        return out;
    }
}

export function installDOM() {
    const head = new El("head");
    const body = new El("body");
    const doc = {
        head,
        body,
        listeners: new Map(),
        createElement: (tag) => new El(tag),
        getElementById: () => null,
        addEventListener(type, fn) {
            if (!doc.listeners.has(type)) doc.listeners.set(type, []);
            doc.listeners.get(type).push(fn);
        },
        removeEventListener(type, fn) {
            const list = doc.listeners.get(type) || [];
            const at = list.indexOf(fn);
            if (at >= 0) list.splice(at, 1);
        },
    };
    globalThis.document = doc;
    globalThis.window = { innerWidth: 1600, innerHeight: 900 };
    globalThis.getComputedStyle = () => ({ backgroundColor: "rgb(26, 27, 30)" });
    globalThis.alert = (msg) => {
        globalThis.__alerts = globalThis.__alerts || [];
        globalThis.__alerts.push(msg);
    };
    return { doc, body };
}
