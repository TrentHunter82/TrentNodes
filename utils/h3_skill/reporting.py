"""
Verbose reporting for the H3 skill nodes.

The nodes accumulate a human-readable report that normally comes back
only as a node output AFTER the run finishes. With the verbose widget
on, every report line also prints to the ComfyUI console the moment it
happens, plus delimited dumps of the payloads too big for the report:
the system prompt, the user context, the model's thinking, and the raw
replies.
"""


class VerboseReport(list):
    """A report list that mirrors appends to stdout when verbose."""

    def __init__(self, tag: str, verbose: bool):
        super().__init__()
        self.tag = tag
        self.verbose = bool(verbose)

    def append(self, line):
        super().append(line)
        if self.verbose:
            print(f"[{self.tag}] {line}", flush=True)

    def extend(self, lines):
        for line in lines:
            self.append(line)

    def dump(self, title: str, text):
        """Print a delimited payload block (console only, never in the
        report output). None skips silently - e.g. no thinking came
        back from an attached non-llama server."""
        if not self.verbose or text is None:
            return
        text = str(text)
        print(f"[{self.tag}] ---- {title} ({len(text)} chars) ----",
              flush=True)
        print(text, flush=True)
        print(f"[{self.tag}] ---- end {title} ----", flush=True)
