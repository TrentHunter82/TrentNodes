import { app } from "../../scripts/app.js";

/**
 * Browser Download node frontend extension.
 * Listens for execution output and triggers automatic
 * file downloads through the browser.
 */
app.registerExtension({
    name: "TrentNodes.BrowserDownload",

    async nodeCreated(node) {
        if (node.comfyClass !== "BrowserDownload") return;

        node.onExecuted = function (message) {
            if (!message || !message.images) return;
            // Only trigger if the Python node flagged it
            if (!message.browser_download) return;

            for (const img of message.images) {
                triggerDownload(img);
            }
        };
    },
});

/**
 * Fetches the image from the ComfyUI server and triggers
 * a browser download via a temporary anchor element.
 */
function triggerDownload(imageInfo) {
    const params = new URLSearchParams({
        filename: imageInfo.filename,
        subfolder: imageInfo.subfolder || "",
        type: imageInfo.type || "output",
    });

    const url = `/view?${params.toString()}`;

    fetch(url)
        .then((response) => {
            if (!response.ok) {
                throw new Error(
                    `Failed to fetch image: ${response.status}`
                );
            }
            return response.blob();
        })
        .then((blob) => {
            const blobUrl = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = blobUrl;
            a.download = imageInfo.filename;
            a.style.display = "none";
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                a.remove();
                URL.revokeObjectURL(blobUrl);
            }, 100);
        })
        .catch((err) => {
            console.error(
                "[TrentNodes] BrowserDownload failed:",
                err
            );
        });
}
