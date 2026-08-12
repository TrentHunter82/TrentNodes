export const api = {
    calls: [],
    async queuePrompt(number, prompt, options) {
        // Snapshot what would actually be sent to the server.
        api.calls.push({ number, output: prompt.output, options });
        return { prompt_id: "test" };
    },
};
