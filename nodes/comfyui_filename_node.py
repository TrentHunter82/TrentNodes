class CustomFilenameGenerator:
    """
    A ComfyUI node that generates custom filenames based on the format:
    PROJ_SEQ_SHOT_DEPARTMENT_WORKFLOW_PASS_INITIALS_VERSION
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("STRING", {"default": "PROJ", "multiline": False, "tooltip": "Project name or code. First part of the filename; leave empty to skip it."}),
                "sequence": ("STRING", {"default": "SEQ", "multiline": False, "tooltip": "Sequence identifier. Leave empty to skip it."}),
                "shot": ("STRING", {"default": "SHOT", "multiline": False, "tooltip": "Shot identifier. Leave empty to skip it."}),
                "department": ("STRING", {"default": "DEPT", "multiline": False, "tooltip": "Department code, e.g. VFX, COMP, or ANIM. Leave empty to skip it."}),
                "workflow": ("STRING", {"default": "WORKFLOW", "multiline": False, "tooltip": "Workflow identifier. Leave empty to skip it."}),
                "pass_name": ("STRING", {"default": "PASS", "multiline": False, "tooltip": "Render pass name, e.g. beauty or shadow. Leave empty to skip it."}),
                "initials": ("STRING", {"default": "AB", "multiline": False, "tooltip": "Artist initials. Leave empty to skip them."}),
                "version": ("STRING", {"default": "v001", "multiline": False, "tooltip": "Version string, e.g. v001. Last part of the filename."}),
            },
            "optional": {
                "separator": ("STRING", {"default": "_", "multiline": False, "tooltip": "Text placed between the parts."}),
                "force_uppercase": ("BOOLEAN", {"default": True, "tooltip": "Convert every part to uppercase."}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "generate_filename"
    CATEGORY = "Trent/Utilities"
    DESCRIPTION = "Builds a pipeline-style filename such as PROJ_SEQ_SHOT_DEPT_WORKFLOW_PASS_AB_V001 from its parts. Empty parts are skipped and whitespace is trimmed."
    
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
