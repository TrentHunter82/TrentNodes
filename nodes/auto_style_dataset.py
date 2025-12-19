"""
Auto Style Dataset Node for ComfyUI.

Outputs 35 prompt strings from an external config file with optional
prepend/append text for synthetic dataset generation.
"""

import os


# Number of prompt outputs
NUM_PROMPTS = 35

# Config file path (same directory as this node)
PROMPTS_FILE = os.path.join(
    os.path.dirname(__file__),
    "auto_style_dataset_prompts.txt"
)


class AutoStyleDataset:
    """
    Generates 35 prompt strings for synthetic dataset creation.

    Reads prompts from an external text file (one per line) and applies
    optional prepend/append text to each output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prepend": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text to prepend to all prompts"
                }),
                "append": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Text to append to all prompts"
                }),
            }
        }

    RETURN_TYPES = tuple(["STRING"] * NUM_PROMPTS)
    RETURN_NAMES = tuple([f"prompt_{i:02d}" for i in range(NUM_PROMPTS)])
    FUNCTION = "generate_prompts"
    CATEGORY = "Trent/Text"
    DESCRIPTION = (
        "Outputs 35 prompt strings from auto_style_dataset_prompts.txt "
        "with optional prepend/append text"
    )

    def generate_prompts(self, prepend="", append=""):
        """
        Read prompts from file and apply prepend/append to each.

        Args:
            prepend: Text to add before each prompt
            append: Text to add after each prompt

        Returns:
            Tuple of 35 strings
        """
        prompts = [""] * NUM_PROMPTS

        try:
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines[:NUM_PROMPTS]):
                # Keep the line as-is (preserving leading space)
                # Just strip trailing whitespace/newlines
                base_prompt = line.rstrip('\n\r')
                prompts[i] = f"{prepend}{base_prompt}{append}"

        except FileNotFoundError:
            error_msg = f"[AutoStyleDataset] File not found: {PROMPTS_FILE}"
            print(error_msg)
            prompts = [error_msg] * NUM_PROMPTS

        except IOError as e:
            error_msg = f"[AutoStyleDataset] Error reading file: {e}"
            print(error_msg)
            prompts = [error_msg] * NUM_PROMPTS

        return tuple(prompts)


NODE_CLASS_MAPPINGS = {
    "AutoStyleDataset": AutoStyleDataset
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoStyleDataset": "Auto Style Dataset"
}
