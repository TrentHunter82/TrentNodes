class GetLastIndex:
    CATEGORY = "Trent/Utils"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("last_index", "count")
    OUTPUT_TOOLTIPS = (
        "Zero-based index of the last frame (count - 1, or 0 for empty)",
        "Number of frames in the batch",
    )
    FUNCTION = "get"
    DESCRIPTION = "Reports the last frame index (count - 1) and the frame count for an IMAGE batch."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    def get(self, images):
        count = int(images.shape[0])
        last_index = max(count - 1, 0)
        return (last_index, count)


NODE_CLASS_MAPPINGS = {"TrentGetEffingLastIndex": GetLastIndex}
NODE_DISPLAY_NAME_MAPPINGS = {"TrentGetEffingLastIndex": "Get Effing Last Index Number"}
