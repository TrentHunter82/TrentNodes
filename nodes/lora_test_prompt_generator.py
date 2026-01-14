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
        "A headshot of {trigger}, front-facing, direct eye contact, "
        "neutral expression, studio lighting, gray background",

        "A medium shot of {trigger}, three-quarter view, "
        "sitting in a sunlit cafe, natural candid moment",

        "A close-up portrait of {trigger}, three-quarter view, "
        "Rembrandt lighting, dark moody background, cinematic",

        "A full body shot of {trigger}, low angle, "
        "standing on a city rooftop at golden hour, confident pose",

        "A medium close-up of {trigger}, front-facing, slight high angle, "
        "genuine laughter, eyes crinkled, joyful expression",

        "A side profile portrait of {trigger}, clean background, "
        "dramatic rim lighting, sharp jawline detail",

        "A full body shot of {trigger}, walking towards camera, "
        "autumn park setting, natural movement, shallow depth of field",

        "An extreme close-up of {trigger}, front-facing, "
        "sharp focus on eyes and skin texture, f/1.4 bokeh",

        "A medium shot of {trigger}, looking back over shoulder, "
        "1970s film photography aesthetic, warm tones, film grain",

        "A cowboy shot of {trigger}, low angle, three-quarter view, "
        "lit by neon signs, blue and pink color cast, urban night",
    ],

    "style": [
        "A portrait of an elderly man with weathered face, {trigger}",
        "A mountain lake reflecting snow-capped peaks at sunrise, {trigger}",
        "An ancient European cathedral interior with light streaming "
        "through stained glass, {trigger}",
        "A still life of flowers in a ceramic vase on rustic wood, {trigger}",
        "An armored knight battling a fire-breathing dragon, {trigger}",
        "A solitary figure sitting on a bench in an empty park, {trigger}",
        "A wolf howling on a cliff under a full moon, {trigger}",
        "A rainy Tokyo alley at night with glowing signs and umbrellas, "
        "{trigger}",
        "A mystical forest path with bioluminescent mushrooms, {trigger}",
        "A red apple with water droplets on white background, {trigger}",
    ],

    "product": [
        "{trigger} centered on pure white background, "
        "professional studio lighting, e-commerce product photo",

        "{trigger} on a minimalist desk setup, "
        "soft natural window light, modern workspace context",

        "{trigger} held in a person's hand, "
        "shallow depth of field, lifestyle product shot",

        "{trigger} on marble surface in bright cafe, "
        "morning light, artful bokeh background",

        "Macro detail shot of {trigger}, "
        "extreme close-up revealing texture and craftsmanship",

        "{trigger} hero shot, three-quarter view from above, "
        "soft gradient shadows, floating appearance",

        "{trigger} with scale reference objects nearby, "
        "clean editorial composition, size context",

        "{trigger} dramatically lit with single spotlight, "
        "black background, high contrast product art",

        "{trigger} in outdoor adventure context, "
        "golden hour on rocks by ocean, lifestyle branding",

        "Flat lay of {trigger} with styled props, "
        "top-down view, curated color palette, social media ready",
    ],

    "vehicle": [
        "{trigger}, three-quarter front view, "
        "clean asphalt surface, soft overcast lighting",

        "{trigger}, perfect side profile, "
        "infinite white studio background, showroom lighting",

        "{trigger}, rear three-quarter angle, "
        "sunset backlighting, low camera angle, dramatic",

        "Front fascia detail of {trigger}, "
        "grille and headlights in focus, shallow depth of field",

        "{trigger} parked on wet city street at night, "
        "reflections in puddles, neon sign ambiance",

        "{trigger} on winding mountain road, "
        "dramatic cliff backdrop, adventure automotive",

        "{trigger} in motion, panning shot, "
        "wheels and background showing motion blur, dynamic",

        "Interior cabin of {trigger}, "
        "dashboard and steering wheel, natural daylight, luxurious",

        "{trigger}, low angle hero shot, "
        "dramatic storm clouds, epic wide composition",

        "{trigger} in rain, water droplets on body, "
        "moody overcast lighting, cinematic atmosphere",
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
