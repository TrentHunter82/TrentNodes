import os
import shutil
from pathlib import Path

class SmartFileTransferNode:
    """
    A ComfyUI node that copies or moves files with intelligent name collision handling.
    If a file already exists, it automatically creates a unique name like file(1).ext, file(2).ext, etc.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter source file path"
                }),
                
                "destination_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter destination path or folder"
                }),
                
                "operation": (["copy", "move"], {
                    "default": "copy"
                }),
                
                "collision_handling": (["auto_rename", "overwrite", "skip"], {
                    "default": "auto_rename"
                }),
                
                "create_directories": ("BOOLEAN", {
                    "default": True
                }),
                
                # Custom naming options
                "rename_pattern": (["parentheses", "underscore", "dash"], {
                    "default": "parentheses"
                })
            },
            "optional": {
                "custom_filename": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: custom filename (leave empty to keep original)"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("result_message", "final_path", "success")
    FUNCTION = "smart_transfer_file"
    CATEGORY = "Trent/Utilities"
    OUTPUT_NODE = True
    
    def find_unique_filename(self, base_path, rename_pattern):
        """
        Find a unique filename by adding numbers if the file already exists.
        
        Args:
            base_path: Path object representing the desired file path
            rename_pattern: Style of numbering ("parentheses", "underscore", "dash")
        
        Returns:
            Path object with a unique filename
        """
        if not base_path.exists():
            return base_path
        
        # Split the filename into name and extension
        file_stem = base_path.stem  # filename without extension
        file_suffix = base_path.suffix  # extension including the dot
        parent_dir = base_path.parent
        
        # Try different numbering until we find one that doesn't exist
        counter = 1
        while True:
            if rename_pattern == "parentheses":
                # Creates: filename(1).ext, filename(2).ext, etc.
                new_name = f"{file_stem}({counter}){file_suffix}"
            elif rename_pattern == "underscore":
                # Creates: filename_1.ext, filename_2.ext, etc.
                new_name = f"{file_stem}_{counter}{file_suffix}"
            elif rename_pattern == "dash":
                # Creates: filename-1.ext, filename-2.ext, etc.
                new_name = f"{file_stem}-{counter}{file_suffix}"
            else:
                # Fallback to parentheses
                new_name = f"{file_stem}({counter}){file_suffix}"
            
            new_path = parent_dir / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1
            
            # Safety check to prevent infinite loops (though practically unnecessary)
            if counter > 9999:
                raise Exception(f"Could not find unique filename after 9999 attempts for {base_path}")
    
    def smart_transfer_file(self, source_path, destination_path, operation, collision_handling, 
                           create_directories, rename_pattern, custom_filename=""):
        """
        Transfer a file with intelligent collision handling
        """
        try:
            # Validate source file
            source = Path(source_path.strip())
            
            if not source.exists():
                return (f"Error: Source file '{source}' does not exist.", "", False)
            
            if not source.is_file():
                return (f"Error: '{source}' is not a file.", "", False)
            
            # Determine destination
            dest = Path(destination_path.strip())
            
            # Handle custom filename
            if custom_filename.strip():
                filename_to_use = custom_filename.strip()
                # If custom filename doesn't have an extension, preserve original
                if not Path(filename_to_use).suffix and source.suffix:
                    filename_to_use += source.suffix
            else:
                filename_to_use = source.name
            
            # Determine final destination path
            if dest.is_dir() or dest.suffix == "":
                # Destination is a directory
                final_dest = dest / filename_to_use
            else:
                # Destination includes filename
                final_dest = dest
            
            # Create destination directory if needed
            if create_directories:
                final_dest.parent.mkdir(parents=True, exist_ok=True)
            elif not final_dest.parent.exists():
                return (f"Error: Destination directory '{final_dest.parent}' does not exist.", "", False)
            
            # Handle file collision based on the selected method
            if collision_handling == "auto_rename":
                # Find a unique filename if there's a collision
                final_dest = self.find_unique_filename(final_dest, rename_pattern)
                collision_msg = ""
                if final_dest.name != filename_to_use:
                    collision_msg = f" (renamed to avoid collision)"
                    
            elif collision_handling == "skip":
                # Skip if file exists
                if final_dest.exists():
                    return (f"Skipped: File '{final_dest.name}' already exists at destination.", str(final_dest), True)
                    
            elif collision_handling == "overwrite":
                # Overwrite existing files (original behavior)
                collision_msg = ""
                if final_dest.exists():
                    collision_msg = f" (overwrote existing file)"
            
            # Perform the file operation
            if operation == "copy":
                shutil.copy2(source, final_dest)
                result_message = f"Successfully copied '{source.name}' to '{final_dest.name}'{collision_msg}"
            else:  # move
                shutil.move(str(source), str(final_dest))
                result_message = f"Successfully moved '{source.name}' to '{final_dest.name}'{collision_msg}"
            
            return (result_message, str(final_dest), True)
            
        except PermissionError as e:
            return (f"Permission error: {str(e)}", "", False)
        except FileNotFoundError as e:
            return (f"File not found error: {str(e)}", "", False)
        except Exception as e:
            return (f"Unexpected error: {str(e)}", "", False)


class FileCollisionTestNode:
    """
    A utility node to help test collision handling by creating dummy files
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter path for test file creation"
                }),
                
                "file_content": ("STRING", {
                    "default": "This is a test file",
                    "multiline": True,
                    "placeholder": "Content to write to the test file"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("result_message", "success")
    FUNCTION = "create_test_file"
    CATEGORY = "Trent/Utilities"
    OUTPUT_NODE = True
    
    def create_test_file(self, file_path, file_content):
        """
        Create a test file for collision testing
        """
        try:
            file_path = Path(file_path.strip())
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the test file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            return (f"Test file created successfully at '{file_path}'", True)
            
        except Exception as e:
            return (f"Error creating test file: {str(e)}", False)


# Example workflow helper node
class FileListNode:
    """
    A node that lists files in a directory to help visualize collision handling results
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter directory path to list"
                }),
                
                "show_full_paths": ("BOOLEAN", {
                    "default": False
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_list",)
    FUNCTION = "list_directory_files"
    CATEGORY = "Trent/Utilities"
    OUTPUT_NODE = True
    
    def list_directory_files(self, directory_path, show_full_paths):
        """
        List all files in a directory
        """
        try:
            dir_path = Path(directory_path.strip())
            
            if not dir_path.exists():
                return (f"Directory '{dir_path}' does not exist.",)
            
            if not dir_path.is_dir():
                return (f"'{dir_path}' is not a directory.",)
            
            # Get all files (not directories)
            files = [f for f in dir_path.iterdir() if f.is_file()]
            
            if not files:
                return (f"No files found in '{dir_path}'",)
            
            # Sort files by name
            files.sort(key=lambda x: x.name)
            
            # Format the output
            if show_full_paths:
                file_list = "\n".join([str(f) for f in files])
            else:
                file_list = "\n".join([f.name for f in files])
            
            return (f"Files in '{dir_path}' ({len(files)} files):\n{file_list}",)
            
        except Exception as e:
            return (f"Error listing files: {str(e)}",)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "SmartFileTransferNode": SmartFileTransferNode,
    "FileCollisionTestNode": FileCollisionTestNode,
    "FileListNode": FileListNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartFileTransferNode": "Smart File Transfer (Auto-Rename)",
    "FileCollisionTestNode": "Create Test File",
    "FileListNode": "List Directory Files"
}