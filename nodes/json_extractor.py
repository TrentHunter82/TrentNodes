# Place this file in your ComfyUI/custom_nodes directory as json_extractor.py
# (Overwrite the previous version if it exists)
# Restart ComfyUI to load the updated node.
# The node will appear in the "utils" category as "JSON Params Extractor".

import json
import os

class JSONParamsExtractorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "json_input": ("STRING", {"multiline": True, "default": "{}"}),
                "json_file_path": ("STRING", {"default": "", "multiline": False}),
                "prefix": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multi_line_string",)

    FUNCTION = "extract_and_format"

    CATEGORY = "utils"

    def extract_and_format(self, json_input="{}", json_file_path="", prefix=""):
        json_data = None

        # Prioritize loading from file if path is provided and valid
        if json_file_path and os.path.isfile(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                return (f"Error loading JSON file: {str(e)}",)
        elif json_input:
            # Fallback to direct JSON string input
            try:
                json_data = json.loads(json_input)
            except json.JSONDecodeError:
                return ("Invalid JSON input string",)
        else:
            return ("No JSON input or file provided",)

        if not isinstance(json_data, dict):
            return ("JSON data must be a dictionary/object",)

        # Define the specific keys in the desired order, with display names
        keys_to_extract = [
            ("height", "Height"),
            ("width", "Width"),
            ("num_frames", "Num Frames"),
            ("fps", "FPS"),
            ("guidance_scale", "Guidance Scale"),
            ("seed", "Seed"),
            ("steps", "Steps"),
            ("use_timestep_transform", "Use Timestep Transform"),
            ("shift_value", "Shift Value"),
            ("use_guidance_schedule", "Use Guidance Schedule"),
            ("add_quality_guidance", "Add Quality Guidance"),
            ("clip_value", "Clip Value"),
            ("use_negative_prompts", "Use Negative Prompts"),
            ("skip_control", "Skip Control"),
            ("caching_coefficient", "Caching Coefficient"),
            ("caching_warmup", "Caching Warmup"),
            ("caching_cooldown", "Caching Cooldown"),
        ]

        lines = []
        for key, display_name in keys_to_extract:
            if key in json_data:
                value = json_data[key]
                # Convert value to string; handle booleans nicely
                if isinstance(value, bool):
                    value_str = str(value).lower()  # "true" or "false"
                else:
                    value_str = str(value)
                line = f"{prefix}{display_name} {value_str}".strip()
                lines.append(line)

        if not lines:
            return ("No matching parameters found",)
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "JSONParamsExtractorNode": JSONParamsExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JSONParamsExtractorNode": "JSON Params Extractor"
}