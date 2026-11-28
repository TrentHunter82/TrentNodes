"""
Kandinsky Video SR Upscale (Trent) -- the ComfyUI-KandinskySR video upscaler with
the NABLA sparse-attention keep threshold exposed.

NABLA keeps, for each 64-token query block, the key blocks that hold the top P of
its attention mass plus a local window. The checkpoint stores P = 0.8. At 4K
(3840x2176, 547 frames) P = 0.6 cut the DiT time by about 11% with PSNR 42.3 dB
against 0.8. P is set on the loaded DiT only for this run and restored after, so
other workflows keep the checkpoint value.

Needs ComfyUI-KandinskySR installed (it runs that pack's own node); no patch.
"""

import nodes


class KandinskyVideoSRUpscaleTrent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": ("KANDINSKY_VIDEO_SR",),
                "scale": (["2x", "2.25x", "4x"], {"default": "2x"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "nabla_P": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05,
                                      "tooltip": "NABLA sparse attention keep threshold. The checkpoint's "
                                                 "value is 0.8; 0.6 is about 11% faster at 4K with nearly "
                                                 "the same detail. Lower is faster and may soften detail."}),
                "window_frames": ("INT", {"default": 361, "min": 17, "max": 505, "step": 8,
                                          "tooltip": "Frames per temporal window (1 + 8k). Two windows always "
                                                     "share at least 9 frames; for a 547-frame clip 281 avoids "
                                                     "running 169 frames twice (about 18% faster)."}),
                "stitch": (["pixels", "latent"], {"default": "latent"}),
                "kernels": (["fast", "exact"], {"default": "fast"}),
                "overlap": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05, "advanced": True}),
                "encoder": (["auto", "hevc nvenc", "h264 nvenc", "h264"], {"default": "auto", "advanced": True}),
            },
            "optional": {
                "video": ("VIDEO",),
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate of the images input; a video keeps its own."}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "upscale"
    CATEGORY = "Trent/Video"
    DESCRIPTION = ("Kandinsky Video SR Upscale with the NABLA attention keep threshold (nabla_P) "
                   "exposed for speed. Needs the ComfyUI-KandinskySR pack.")

    def upscale(self, models, scale, seed, nabla_P, window_frames, stitch, kernels, overlap, encoder,
                video=None, images=None, fps=24.0):
        node = nodes.NODE_CLASS_MAPPINGS.get("KandinskyVideoSRUpscale")
        if node is None:
            raise RuntimeError("Kandinsky Video SR Upscale (Trent) needs the ComfyUI-KandinskySR pack installed.")
        dits = [card.dit.patcher.model for card in models.cards]
        saved = [dit.attention for dit in dits]
        for dit in dits:
            dit.attention = dict(dit.attention, P=nabla_P)
        try:
            out = node.execute(models, scale, seed, overlap, encoder, fps, kernels, stitch, window_frames,
                               video=video, images=images)
        finally:
            for dit, attention in zip(dits, saved):
                dit.attention = attention
        return (out.result[0],)


NODE_CLASS_MAPPINGS = {"KandinskyVideoSRUpscaleTrent": KandinskyVideoSRUpscaleTrent}
NODE_DISPLAY_NAME_MAPPINGS = {"KandinskyVideoSRUpscaleTrent": "Kandinsky Video SR Upscale (Trent)"}
