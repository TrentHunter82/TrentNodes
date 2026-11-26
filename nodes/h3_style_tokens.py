"""Experimental: append shuffled style-image vision tokens to H3 conditioning.

The images are encoded on their own, so the prompt and its <Picture 1> stay
untouched. Only the vision tokens are kept, and they stay tagged as vision.
"""

import torch

import comfy.utils

from .h3_image_dial import shuffle_tiles, WIDTH, HEIGHT


class H3StyleTokens:
    DISPLAY_NAME = "Style Tokens (H3, experimental)"
    DESCRIPTION = "Adds style images to the conditioning as extra tokens, with no <Picture> label and no first-frame change."
    CATEGORY = "Trent/MiniMax"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "style_images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "tile_size": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            },
        }

    def add(self, conditioning, clip, style_images, strength, tile_size):
        if strength == 0:
            return (conditioning,)
        images = comfy.utils.common_upscale(style_images[..., :3].movedim(-1, 1), WIDTH, HEIGHT, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
        tokens = []
        for image in images:
            if tile_size:
                image = shuffle_tiles(image, tile_size, 1)
            # half size per image, so five images add about as many tokens as one full frame
            small = comfy.utils.common_upscale(image.unsqueeze(0).movedim(-1, 1), WIDTH // 2, HEIGHT // 2, "area", "disabled").movedim(1, -1)
            encoded, extra = clip.encode_from_tokens_scheduled(clip.tokenize("", images=[small]))[0]
            tokens.append(encoded[0][extra["minimax_token_tags"].view(-1) == 0])
        style = torch.cat(tokens).unsqueeze(0) * strength

        out = []
        for c, extra in conditioning:
            extra = extra.copy()
            s = style.to(c)
            extra["minimax_token_tags"] = torch.cat([extra["minimax_token_tags"], torch.zeros(s.shape[1], dtype=torch.long)])
            out.append([torch.cat([c, s], dim=1), extra])
        return (out,)


class H3StyleContext:
    DISPLAY_NAME = "Style Context (H3, experimental)"
    DESCRIPTION = "Adds style images as <Picture 2> onward with a sentence that gives them a job. The conditioning's own <Picture 1> is not touched."
    CATEGORY = "Trent/MiniMax"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "style_images": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default": "Render the whole video in the art style of the pictures above. Copy only how they are drawn, not their subjects."}),
            },
        }

    def add(self, conditioning, clip, style_images, text):
        images = comfy.utils.common_upscale(style_images[..., :3].movedim(-1, 1), WIDTH // 2, HEIGHT // 2, "area", "disabled").movedim(1, -1).clamp(0, 1)
        # A small grey placeholder takes the <Picture 1> label and is cut off again.
        placeholder = torch.full((1, HEIGHT // 4, WIDTH // 4, 3), 0.5)
        encoded, extra = clip.encode_from_tokens_scheduled(clip.tokenize(text, images=[placeholder] + [i.unsqueeze(0) for i in images]))[0]
        tags = extra["minimax_token_tags"].view(-1)
        first_block_end = (tags == 0).nonzero()[0].item()
        while tags[first_block_end] == 0:
            first_block_end += 1
        style, style_tags = encoded[:, first_block_end:], tags[first_block_end:]

        out = []
        for c, c_extra in conditioning:
            c_extra = c_extra.copy()
            c_extra["minimax_token_tags"] = torch.cat([c_extra["minimax_token_tags"], style_tags.to(c_extra["minimax_token_tags"])])
            out.append([torch.cat([c, style.to(c)], dim=1), c_extra])
        return (out,)


class H3SwapPictureTokens:
    DISPLAY_NAME = "Swap Picture Tokens (H3, experimental)"
    DESCRIPTION = "Moves only the vision tokens of the conditioning toward another image of the same layout. Prompt tokens and the first-frame latent stay as they are."
    CATEGORY = "Trent/MiniMax"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "swap"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            },
        }

    def swap(self, conditioning, clip, image, prompt, width, height, strength):
        image = comfy.utils.common_upscale(image[:1, ..., :3].movedim(-1, 1), width, height, "area", "disabled").movedim(1, -1)
        other, _ = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=[image]))[0]
        out = []
        for c, extra in conditioning:
            if other.shape != c.shape:
                raise ValueError(f"Swap Picture Tokens: token layouts differ, {other.shape} vs {c.shape}.")
            mask = extra["minimax_token_tags"].view(-1) == 0
            c = c.clone()
            o = other.to(c)
            c[:, mask] = c[:, mask] + strength * (o[:, mask] - c[:, mask])
            out.append([c, extra])
        return (out,)


NODE_CLASS_MAPPINGS = {"H3StyleTokens": H3StyleTokens, "H3StyleContext": H3StyleContext, "H3SwapPictureTokens": H3SwapPictureTokens}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.DISPLAY_NAME for k, v in NODE_CLASS_MAPPINGS.items()}
