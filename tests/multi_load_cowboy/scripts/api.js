/** The listing the mocked /object_info hands back. */
const FILES = ["(empty)", "a.png", "b.png", "c.png"];

export const api = {
    calls: [],
    uploads: [],
    /** Set to a status other than 200 to make the next upload fail. */
    uploadStatus: 200,

    apiURL: (path) => `http://comfy${path}`,

    async fetchApi(path, options) {
        api.calls.push(path);

        if (path === "/upload/image") {
            const file = options.body.get("image");
            if (api.uploadStatus !== 200) {
                return {
                    status: api.uploadStatus,
                    ok: false,
                    statusText: "boom",
                    json: async () => ({}),
                };
            }
            api.uploads.push(file.name);
            return {
                status: 200,
                ok: true,
                json: async () => ({
                    name: file.name,
                    subfolder: "",
                    type: "input",
                }),
            };
        }

        if (path.startsWith("/object_info/")) {
            return {
                status: 200,
                ok: true,
                json: async () => ({
                    MultiLoadCowboy: {
                        input: { required: { image_1: [FILES, {}] } },
                    },
                }),
            };
        }

        return { status: 404, ok: false, statusText: "not found" };
    },
};
