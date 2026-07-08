import torch


def _frame_bbox(present):
    """Return (x1, y1, x2, y2) tight bbox of a boolean [H, W] tensor, or None if empty."""
    rows = present.any(dim=1)
    cols = present.any(dim=0)
    if not bool(rows.any()):
        return None
    ys = torch.nonzero(rows, as_tuple=False)
    xs = torch.nonzero(cols, as_tuple=False)
    y1 = int(ys[0])
    y2 = int(ys[-1]) + 1
    x1 = int(xs[0])
    x2 = int(xs[-1]) + 1
    return x1, y1, x2, y2


class MaskToBBoxMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "fill": ("BOOLEAN", {"default": True}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 256, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("bbox_mask", "images")
    FUNCTION = "convert"
    CATEGORY = "Trent/Masks"
    DESCRIPTION = (
        "Per frame, finds the bounding box of the input mask (pixels above threshold) and "
        "outputs a rectangular bbox mask that tracks each frame's mask area. Frames with no "
        "mask present yield an empty bbox for that frame. Optionally draws the bbox over an "
        "input image or batch of images (outline unless 'fill' is on)."
    )

    def convert(self, mask, threshold=0.5, padding=0, fill=True, outline_thickness=3, image=None):
        # Normalize mask to [B, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        b_mask, mh, mw = mask.shape[0], mask.shape[1], mask.shape[2]

        # Decide output spatial size: match the image if one is provided, else the mask.
        if image is not None:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            b_img, oh, ow = image.shape[0], image.shape[1], image.shape[2]
        else:
            b_img, oh, ow = 0, mh, mw

        b = max(b_mask, b_img)

        # Scale mask -> output size once (bbox is computed in output space so coords line up).
        if (mh, mw) != (oh, ow):
            mask_rs = torch.nn.functional.interpolate(
                mask.unsqueeze(1), size=(oh, ow), mode="bilinear", align_corners=False
            ).squeeze(1)
        else:
            mask_rs = mask

        device = mask_rs.device
        out_mask = torch.zeros((b, oh, ow), dtype=torch.float32, device=device)

        if image is not None:
            out_img = image.clone()
            if b_img < b:  # broadcast a single image across the mask batch
                out_img = out_img[:1].repeat(b, 1, 1, 1)
            out_img = out_img.to(device)
        else:
            out_img = torch.zeros((b, oh, ow, 3), dtype=torch.float32, device=device)

        t = outline_thickness

        for i in range(b):
            m = mask_rs[i if b_mask > 1 else 0]
            bbox = _frame_bbox(m > threshold)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(ow, x2 + padding)
            y2 = min(oh, y2 + padding)
            if x2 <= x1 or y2 <= y1:
                continue

            if fill:
                out_mask[i, y1:y2, x1:x2] = 1.0
            else:
                out_mask[i, y1:min(y1 + t, y2), x1:x2] = 1.0  # top
                out_mask[i, max(y2 - t, y1):y2, x1:x2] = 1.0  # bottom
                out_mask[i, y1:y2, x1:min(x1 + t, x2)] = 1.0  # left
                out_mask[i, y1:y2, max(x2 - t, x1):x2] = 1.0  # right

            if image is not None:
                # Draw a white outline over the image so the picture stays visible.
                out_img[i, y1:min(y1 + t, y2), x1:x2, :] = 1.0
                out_img[i, max(y2 - t, y1):y2, x1:x2, :] = 1.0
                out_img[i, y1:y2, x1:min(x1 + t, x2), :] = 1.0
                out_img[i, y1:y2, max(x2 - t, x1):x2, :] = 1.0
            else:
                # No image: visualize the bbox mask itself as a grayscale image.
                out_img[i, :, :, :] = out_mask[i].unsqueeze(-1)

        return (out_mask, out_img)


NODE_CLASS_MAPPINGS = {"MaskToBBoxMask": MaskToBBoxMask}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskToBBoxMask": "Mask to BBox Mask"}
