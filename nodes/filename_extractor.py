import re
import os

class FilenameExtractor:
    """
    A ComfyUI custom node that extracts a portion of a filename.
    Searches for a pattern of 3 letters + 3 numbers (e.g., CRC040)
    and returns everything up to and including that pattern.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # This accepts the full filename or path as a string
                "filename": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }
    
    # This defines what the node returns
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("extracted_name", "pattern_found", "full_filename")
    
    FUNCTION = "extract_filename"
    CATEGORY = "Trent/Utilities"
    
    def extract_filename(self, filename):
        """
        Main function that processes the filename.
        
        Args:
            filename: The input filename (can be full path or just filename)
            
        Returns:
            tuple: (extracted_name, pattern_found, full_filename)
        """
        
        # If a full path is provided, get just the filename
        base_filename = os.path.basename(filename)
        
        # Remove the file extension if present
        name_without_ext = os.path.splitext(base_filename)[0]
        
        # Regex pattern: looks for 3 uppercase letters followed by 3 digits
        # The pattern captures everything from the start up to and including this code
        pattern = r'^(.*?[A-Z]{3}\d{3})'
        
        # Search for the pattern in the filename
        match = re.search(pattern, name_without_ext)
        
        if match:
            # Extract everything up to and including the pattern
            extracted = match.group(1)
            
            # Extract just the code part (e.g., "CRC040")
            code_match = re.search(r'[A-Z]{3}\d{3}', extracted)
            pattern_found = code_match.group(0) if code_match else "Not found"
            
            return (extracted, pattern_found, base_filename)
        else:
            # If no pattern is found, return the original filename and an error message
            return (name_without_ext, "Pattern not found", base_filename)


# This is required for ComfyUI to recognize the node
NODE_CLASS_MAPPINGS = {
    "FilenameExtractor": FilenameExtractor
}

# This sets the display name in the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "FilenameExtractor": "Filename Extractor"
}
