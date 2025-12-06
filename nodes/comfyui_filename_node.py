class CustomFilenameGenerator:
    """
    A ComfyUI node that generates custom filenames based on the format:
    PROJ_SEQ_SHOT_DEPARTMENT_WORKFLOW_PASS_INITIALS_VERSION
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("STRING", {"default": "PROJ", "multiline": False}),
                "sequence": ("STRING", {"default": "SEQ", "multiline": False}),
                "shot": ("STRING", {"default": "SHOT", "multiline": False}),
                "department": ("STRING", {"default": "DEPT", "multiline": False}),
                "workflow": ("STRING", {"default": "WORKFLOW", "multiline": False}),
                "pass_name": ("STRING", {"default": "PASS", "multiline": False}),
                "initials": ("STRING", {"default": "AB", "multiline": False}),
                "version": ("STRING", {"default": "v001", "multiline": False}),
            },
            "optional": {
                "separator": ("STRING", {"default": "_", "multiline": False}),
                "force_uppercase": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "generate_filename"
    CATEGORY = "Trent/Utilities"
    
    def generate_filename(self, project, sequence, shot, department, workflow, 
                         pass_name, initials, version, separator="_", force_uppercase=True):
        """
        Generate a custom filename based on the provided components.
        
        Args:
            project: Project name/code
            sequence: Sequence identifier
            shot: Shot identifier
            department: Department code (e.g., VFX, COMP, ANIM)
            workflow: Workflow identifier
            pass_name: Pass name (e.g., beauty, shadow, reflection)
            initials: Artist initials
            version: Version string (e.g., v001, v002)
            separator: Character to separate components (default: "_")
            force_uppercase: Whether to convert all components to uppercase
        
        Returns:
            tuple: (formatted_filename,)
        """
        
        # Collect all components in order
        components = [
            project,
            sequence, 
            shot,
            department,
            workflow,
            pass_name,
            initials,
            version
        ]
        
        # Clean up components (remove extra whitespace, handle empty values)
        cleaned_components = []
        for component in components:
            if component and isinstance(component, str):
                cleaned = component.strip()
                if cleaned:  # Only add non-empty components
                    if force_uppercase:
                        cleaned = cleaned.upper()
                    cleaned_components.append(cleaned)
        
        # Join components with separator
        filename = separator.join(cleaned_components)
        
        return (filename,)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CustomFilenameGenerator": CustomFilenameGenerator
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomFilenameGenerator": "Custom Filename Generator"
}