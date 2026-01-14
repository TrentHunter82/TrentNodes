"""
LoRA Test Prompt Generator Node for ComfyUI

Generates a set of 10 test prompts designed to validate different types of
LoRA models. Supports subject/person, style, product, and vehicle LoRAs
with carefully crafted prompts to test various scenarios.
"""


# Prompt templates organized by LoRA type
# Each list contains 10 prompts with {trigger} placeholder

PROMPT_TEMPLATES = {
    "subject_person": [
        "A medium shot photo of {trigger}, three-quarter view, "
        "neutral expression, studio lighting, gray background",

        "A medium close-up of {trigger}, front-facing, "
        "sitting in a sunlit cafe, candid moment",

        "A close-up portrait of {trigger}, three-quarter view, "
        "Rembrandt lighting, dark moody background, cinematic",

        "A full body photo of {trigger}, low angle shot, "
        "standing on a city rooftop, golden hour",

        "A medium shot of {trigger}, slight high angle, "
        "genuine laughter, eyes crinkled, joyful expression",

        "A side profile photo of {trigger}, medium close-up, "
        "clean background, soft rim lighting",

        "A full body photo of {trigger}, three-quarter rear view, "
        "walking through autumn leaves, motion and movement",

        "An extreme close-up portrait of {trigger}, front-facing, "
        "sharp focus on eyes, f/1.4 bokeh",

        "A medium shot of {trigger}, over-the-shoulder angle, "
        "1970s film photography style, warm tones, grain",

        "A cowboy shot of {trigger}, low angle, "
        "lit only by neon signs, blue and pink color cast, night",
    ],

    "style": [
        "A portrait of a woman, {trigger}",
        "A mountain lake at sunrise, {trigger}",
        "An old European cathedral, {trigger}",
        "A vase of flowers on a wooden table, {trigger}",
        "A knight fighting a dragon, {trigger}",
        "The feeling of loneliness, {trigger}",
        "A wolf howling at the moon, {trigger}",
        "A rainy Tokyo street at night, {trigger}",
        "An enchanted forest with glowing mushrooms, {trigger}",
        "A single apple on white background, {trigger}",
    ],

    "product": [
        "A photo of {trigger}, centered on white background, "
        "studio lighting, product photography",

        "A photo of {trigger}, on a modern desk in a home office, "
        "natural window light",

        "A photo of {trigger}, being held in a person's hand, "
        "shallow depth of field",

        "A photo of {trigger}, on a coffee shop table, morning light, "
        "bokeh background",

        "A macro photo of {trigger}, extreme close-up showing "
        "material texture and details",

        "A photo of {trigger}, three-quarter view from above, soft shadows",

        "A photo of {trigger}, next to everyday objects for size comparison, "
        "clean composition",

        "A photo of {trigger}, single spotlight on black background, "
        "high contrast",

        "A photo of {trigger}, on a rock by the ocean at sunset, "
        "adventure lifestyle",

        "A photo of {trigger}, top-down view with complementary props, "
        "styled flat lay composition",
    ],

    "vehicle": [
        "A photo of {trigger}, three-quarter front view, on empty asphalt, "
        "overcast soft light",

        "A photo of {trigger}, perfect side profile, clean background, "
        "studio lighting",

        "A photo of {trigger}, rear three-quarter angle, sunset lighting, "
        "low angle",

        "A close-up photo of {trigger}, front grille and headlights, "
        "shallow depth of field",

        "A photo of {trigger}, parked on a wet city street at night, "
        "reflections, neon lights",

        "A photo of {trigger}, on a winding mountain road, "
        "dramatic landscape background",

        "A photo of {trigger}, driving at speed, motion blur on wheels "
        "and background, panning shot",

        "Interior photo of {trigger}, dashboard and steering wheel, "
        "natural daylight through windows",

        "A photo of {trigger}, low angle hero shot, dramatic clouds, "
        "epic composition",

        "An aerial photo of {trigger}, directly overhead, "
        "on geometric pavement pattern",
    ],
}


class LoRATestPromptGenerator:
    """
    Generates 10 test prompts for validating LoRA models.

    Different prompt sets are provided for various LoRA types:
    - subject_person: Portrait/character LoRAs
    - style: Artistic style LoRAs
    - product: Object/product LoRAs
    - vehicle: Car/vehicle LoRAs

    Each prompt set tests different scenarios relevant to that LoRA type.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_word": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "The LoRA trigger token to insert into prompts"
                }),
                "lora_type": ([
                    "subject_person",
                    "style",
                    "product",
                    "vehicle"
                ], {
                    "default": "subject_person",
                    "tooltip": "Type of LoRA being tested"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Seed for any randomization (reserved)"
                }),
            },
            "optional": {
                "quality_suffix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Additional quality tags to append to each prompt "
                        "(e.g., '8k, detailed')"
                    )
                }),
            },
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5",
        "prompt_6", "prompt_7", "prompt_8", "prompt_9", "prompt_10",
        "all_prompts",
    )
    FUNCTION = "generate_prompts"
    CATEGORY = "TrentNodes/Testing"
    DESCRIPTION = (
        "Generates 10 test prompts for validating LoRA models across "
        "different scenarios"
    )

    def generate_prompts(
        self,
        trigger_word: str,
        lora_type: str,
        seed: int,
        quality_suffix: str = ""
    ) -> tuple:
        """
        Generate 10 test prompts for the specified LoRA type.

        Args:
            trigger_word: The LoRA trigger token to insert
            lora_type: Type of LoRA (subject_person, style, product, vehicle)
            seed: Random seed (reserved for future use)
            quality_suffix: Optional quality tags to append

        Returns:
            Tuple of 11 strings: prompt_1 through prompt_10, plus all_prompts
        """
        # Get templates for the selected LoRA type
        templates = PROMPT_TEMPLATES.get(lora_type, PROMPT_TEMPLATES["style"])

        # Generate prompts by replacing {trigger} placeholder
        prompts = []
        for template in templates:
            prompt = template.replace("{trigger}", trigger_word)

            # Append quality suffix if provided
            if quality_suffix and quality_suffix.strip():
                prompt = f"{prompt}, {quality_suffix.strip()}"

            prompts.append(prompt)

        # Ensure we have exactly 10 prompts
        while len(prompts) < 10:
            prompts.append("")

        # Combine all prompts with newlines for the combined output
        all_prompts = "\n".join(prompts)

        # Return individual prompts + combined
        return (
            prompts[0], prompts[1], prompts[2], prompts[3], prompts[4],
            prompts[5], prompts[6], prompts[7], prompts[8], prompts[9],
            all_prompts,
        )


NODE_CLASS_MAPPINGS = {
    "LoRATestPromptGenerator": LoRATestPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRATestPromptGenerator": "LoRA Test Prompt Generator"
}
