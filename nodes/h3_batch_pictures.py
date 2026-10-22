"""
H3 Batch Pictures - Batch Images with sockets named after the MiniMax H3
prompt tags, so the socket an image plugs into is the tag you write.

Batch index 0 is <Picture 1>, index 1 is <Picture 2>, and so on.
"""

from comfy_api.latest import io
from comfy_extras.nodes_post_processing import batch_images

NAMES = [f"<Picture {i}>" for i in range(1, 10)]


class H3BatchPictures(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3BatchPictures",
            display_name="H3 Batch Pictures",
            category="Trent/MiniMax",
            description="Batch Images with sockets named <Picture 1>, <Picture 2>... to match the MiniMax H3 prompt tags.",
            search_aliases=["h3 pictures", "picture tags", "batch images h3", "minimax references"],
            inputs=[
                io.Autogrow.Input(
                    "pictures",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("picture"),
                        names=NAMES,
                        min=1,
                    ),
                ),
                io.String.Input("roles", multiline=True, default="",
                    tooltip="One line per picture, in socket order: what each picture supplies, e.g. identity of the woman, her red jacket."),
                io.Boolean.Input("first_frame", default=False,
                    tooltip="<Picture 1> is the first frame of the target video. Also anchor it with Add Guide for MiniMax H3 at frame_idx 0."),
                io.Boolean.Input("last_frame", default=False,
                    tooltip="The last connected picture is the last frame of the target video. Also anchor it with Add Guide for MiniMax H3 at frame_idx -1."),
            ],
            outputs=[
                io.Image.Output(),
                io.String.Output(display_name="reference_roles",
                    tooltip="Connect to H3 Codex Promptor reference_roles, with this batch on reference_images. Labels assume no first_frame/last_frame is connected there."),
            ],
        )

    @classmethod
    def execute(cls, pictures: io.Autogrow.Type, roles: str = "", first_frame=False, last_frame=False) -> io.NodeOutput:
        connected = [name for name in NAMES if pictures.get(name) is not None]
        if connected != NAMES[:len(connected)]:
            missing = [name for name in NAMES[:NAMES.index(connected[-1])] if name not in connected]
            raise ValueError(f"H3 Batch Pictures: {', '.join(missing)} empty. Fill the gap so batch order matches the picture tags.")
        lines = [line.strip() for line in roles.splitlines()]
        entries = []
        for i, name in enumerate(connected):
            parts = []
            if first_frame and i == 0:
                parts.append("first frame of the target video")
            if last_frame and i == len(connected) - 1:
                parts.append("last frame of the target video")
            if i < len(lines) and lines[i]:
                parts.append(lines[i])
            entries.append(f"{name} (batch index {i})" + (": " + "; ".join(parts) if parts else ""))
        reference_roles = "\n".join(entries)
        return io.NodeOutput(batch_images([pictures[name] for name in connected]), reference_roles)


NODE_CLASS_MAPPINGS = {"H3BatchPictures": H3BatchPictures}
NODE_DISPLAY_NAME_MAPPINGS = {"H3BatchPictures": "H3 Batch Pictures"}
