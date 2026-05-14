import torch


class DrawGrayBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "bboxes": ("BOUNDING_BOX", {"forceInput": True}),
                "expand": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    FUNCTION = "draw"
    CATEGORY = "Trent/Masks"
    DESCRIPTION = "Fills each bbox region with neutral gray (0.5) and outputs a matching binary mask. Consumes the BOUNDING_BOX type from comfy-core's RT-DETR Detect / Draw BBoxes nodes."

    def draw(self, images, bboxes, expand=0):
        b, h, w, _ = images.shape

        if isinstance(bboxes, dict):
            bboxes = [[bboxes]]
        elif not isinstance(bboxes, list) or not bboxes:
            bboxes = [[]]
        elif isinstance(bboxes[0], dict):
            bboxes = [bboxes]
        if len(bboxes) == 1:
            bboxes = bboxes * b
        bboxes = (bboxes + [[]] * b)[:b]

        out_img = images.clone()
        out_mask = torch.zeros((b, h, w), dtype=images.dtype, device=images.device)

        for i in range(b):
            for det in bboxes[i]:
                x1 = max(0, int(det["x"]) - expand)
                y1 = max(0, int(det["y"]) - expand)
                x2 = min(w, int(det["x"] + det["width"]) + expand)
                y2 = min(h, int(det["y"] + det["height"]) + expand)
                if x2 <= x1 or y2 <= y1:
                    continue
                out_img[i, y1:y2, x1:x2, :] = 0.5
                out_mask[i, y1:y2, x1:x2] = 1.0

        return (out_img, out_mask)


NODE_CLASS_MAPPINGS = {"DrawGrayBBox": DrawGrayBBox}
NODE_DISPLAY_NAME_MAPPINGS = {"DrawGrayBBox": "Draw Gray BBox"}
