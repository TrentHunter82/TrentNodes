"""Make Image Dial: turn reference images into a ComfyUI-Dials steering dial for MiniMax H3.

The H3 text encoder also reads images, so a dial can come from pictures instead
of prompt pairs: encode the images, run them through condition_proj and
token_refiner, and take the difference of the token means. Cutting the images
into shuffled tiles first breaks up objects but keeps colour, line and texture,
which pulls the dial toward style and away from subject.
"""

import torch

import comfy.model_management
import comfy.utils

# Short fixed texts, so no single wording ends up in the dial.
TEXTS = ("A video.", "A person in a scene.", "An animated film scene with music.", "A landscape in motion.")
SHUFFLES = (1, 2, 3)
WIDTH, HEIGHT = 1024, 576


def shuffle_tiles(image, tile, seed):
    # [H, W, 3], cropped to whole tiles
    rows, cols = image.shape[0] // tile, image.shape[1] // tile
    tiles = image[:rows * tile, :cols * tile].reshape(rows, tile, cols, tile, 3).permute(0, 2, 1, 3, 4).flatten(0, 1)
    tiles = tiles[torch.randperm(tiles.shape[0], generator=torch.Generator().manual_seed(seed))]
    return tiles.unflatten(0, (rows, cols)).permute(0, 2, 1, 3, 4).flatten(2, 3).flatten(0, 1)


class H3MakeImageDial:
    DISPLAY_NAME = "Make Image Dial (H3)"
    DESCRIPTION = ("Makes a dial from images. Use several images with one look but different subjects. "
                   "Positive strength moves toward style_images, negative moves away. "
                   "Needs ComfyUI-Dials for Apply Dial and Save Dial.")
    CATEGORY = "Trent/MiniMax"
    RETURN_TYPES = ("DIAL", "STRING")
    RETURN_NAMES = ("dial", "report")
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "style_images": ("IMAGE", {"tooltip": "The look you want. Use different subjects, so only the look is shared."}),
                "name": ("STRING", {"default": "my_image_dial"}),
                "tile_size": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Shuffles the images into tiles of this size to reduce subject. 0 keeps whole images."}),
            },
            "optional": {
                "neutral_images": ("IMAGE", {"tooltip": "Plain images to compare against, such as photos. Without them, flat grey is used."}),
            },
        }

    def make(self, clip, model, style_images, name, tile_size, neutral_images=None):
        dm = model.model.diffusion_model
        comfy.model_management.load_models_gpu([model])
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype_inference()

        def encode(image, text):
            tokens = clip.encode_from_tokens_scheduled(clip.tokenize(text, images=[image.unsqueeze(0)]))[0][0]
            comfy.model_management.load_models_gpu([model])
            return dm.token_refiner(dm.condition_proj(tokens[0].to(device, dtype))).float().mean(0).cpu()

        def encode_set(images):
            # One vector per image: its mean over every text and shuffle.
            images = comfy.utils.common_upscale(images[..., :3].movedim(-1, 1), WIDTH, HEIGHT, "lanczos", "disabled").movedim(1, -1).clamp(0, 1)
            out = []
            for image in images:
                views = [shuffle_tiles(image, tile_size, seed) for seed in SHUFFLES] if tile_size else [image]
                out.append(torch.stack([encode(view, text) for view in views for text in TEXTS]).mean(0))
            return torch.stack(out)

        style = encode_set(style_images)
        neutral = encode_set(torch.full((1, HEIGHT, WIDTH, 3), 0.5) if neutral_images is None else neutral_images)
        vec = style.mean(0) - neutral.mean(0)
        agree = torch.nn.functional.cosine_similarity(style - neutral.mean(0), vec.unsqueeze(0)).mean().item()

        dial = {"vec": vec, "target": "token_refiner", "mode": "post",
                "meta": {"name": name, "source": "image dial", "n_pairs": str(style.shape[0]),
                         "pair_agreement_cos": f"{agree:.3f}", "inject": "post_refiner (hook)",
                         "debias_removed": "off", "calibrated": "False", "format": "ComfyUI-Dials v1"}}
        report = (f"image dial '{name}': {style.shape[0]} style, {neutral.shape[0]} neutral, tile {tile_size}, "
                  f"agreement {agree:.3f}, |dial| {vec.norm().item():.1f}")
        return (dial, report)


NODE_CLASS_MAPPINGS = {"H3MakeImageDial": H3MakeImageDial}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MakeImageDial": H3MakeImageDial.DISPLAY_NAME}
