"""Experimental: apply an H3 dial to the text rows of a block range only.

A normal dial shifts the text once, before block 0. This adds the shift at the
input of first_block and subtracts it at the input of last_block + 1. Blocks
also change the text rows, so the subtraction does not restore them exactly.
"""

import logging

import comfy.patcher_extension


class H3DialBlockRange:
    DISPLAY_NAME = "Dial Block Range (H3, test)"
    CATEGORY = "Trent/MiniMax"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "dial": ("DIAL",),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05}),
                "first_block": ("INT", {"default": 0, "min": 0, "max": 49}),
                "last_block": ("INT", {"default": 49, "min": 0, "max": 49}),
            },
        }

    def apply(self, model, dial, strength, first_block, last_block):
        if strength == 0:
            return (model,)
        shift = dial["vec"] * strength

        def shifted(index, sign, prior):
            def patch(args, extra):
                h = args["img"]
                a, b, _ = next(s for s in args["layout"].segments if s[2] == "text")
                h[a:b] += sign * shift.to(h)
                logging.info(f"dial block range: block {index} sign {sign:+d} rows {b - a} |shift| {shift.norm().item():.1f} other patch {prior is not None}")
                return prior(args, extra) if prior is not None else extra["original_block"](args)
            return patch

        def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
            # Runs inside the forward, so patches from nodes after this one, such as sparse attention, are wrapped and not lost.
            options = transformer_options.copy()
            replace = options.get("patches_replace", {}).copy()
            dit = replace.get("dit", {}).copy()
            dit[("double_block", first_block)] = shifted(first_block, 1, dit.get(("double_block", first_block)))
            if last_block < 49:
                dit[("double_block", last_block + 1)] = shifted(last_block + 1, -1, dit.get(("double_block", last_block + 1)))
            replace["dit"] = dit
            options["patches_replace"] = replace
            return executor(x, timestep, context, options, **kwargs)

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3_dial_block_range", wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"H3DialBlockRange": H3DialBlockRange}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DialBlockRange": H3DialBlockRange.DISPLAY_NAME}
