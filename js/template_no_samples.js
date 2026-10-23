import { api } from "../../scripts/api.js";

/*
 * Template sample skip
 * --------------------
 * Opening a workflow template downloads its sample input media from GitHub
 * and uploads it into input/. Frontend 1.57 does this in loadTemplateData()
 * only when the template index entry has io.inputs, so hiding that list makes
 * the template open as its bare workflow JSON.
 *
 * Toggled from the Trent Tools sidebar (see color_palette.js). State persists
 * in localStorage and is read live, so no reload is needed after toggling.
 */

const LS_SKIP = "TrentTemplates.skipSamples";

let skipSamples = localStorage.getItem(LS_SKIP) === "1"; // default OFF

export function getSkipSamples() { return skipSamples; }

export function setSkipSamples(on) {
    skipSamples = !!on;
    localStorage.setItem(LS_SKIP, on ? "1" : "0");
}

const getCoreWorkflowTemplates = api.getCoreWorkflowTemplates.bind(api);
api.getCoreWorkflowTemplates = async (locale) => {
    const modules = await getCoreWorkflowTemplates(locale);
    for (const template of modules.flatMap((m) => m.templates ?? [])) {
        const inputs = template.io?.inputs;
        if (!inputs) continue;
        // The template store copies entries shallowly, so the copies share
        // this io object and the getter stays live.
        Object.defineProperty(template.io, "inputs", {
            get: () => (skipSamples ? undefined : inputs),
            enumerable: true,
        });
    }
    return modules;
};
