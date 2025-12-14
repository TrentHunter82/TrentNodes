import os
import glob
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image

class LatestVideoFinalFrameNode:
    """
    A ComfyUI node that finds the latest video file in a directory 
    and extracts its final frame.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input parameters for this node.
        ComfyUI uses this method to understand what inputs the node expects.
        """
        return {
            "required": {
                # Directory path where we'll look for video files
                "directory_path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Path to directory containing video files"
                }),
                # Whether to search subdirectories recursively
                "recursive_search": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Search subdirectories for video files"
                }),
                # File extensions to consider as video files
                "video_extensions": ("STRING", {
                    "default": "mp4,avi,mov,mkv,webm,flv,wmv",
                    "multiline": False,
                    "tooltip": "Comma-separated list of video file extensions"
                })
            }
        }
    
    # Define what type of data this node returns
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("final_frame", "video_filename")
    
    # This helps ComfyUI categorize the node in the interface
    CATEGORY = "Trent/Video"
    
    # A unique identifier for this node type
    FUNCTION = "extract_final_frame"

    def extract_final_frame(self, directory_path, recursive_search, video_extensions):
        """
        Main function that performs the video processing.
        This is where the actual work happens.
        """
        try:
            # First, let's find all video files in the specified directory
            video_files = self._find_video_files(directory_path, recursive_search, video_extensions)
            
            if not video_files:
                # If no videos found, create a black placeholder image
                # This prevents the node from crashing the workflow
                placeholder = self._create_placeholder_image("No video files found")
                return (placeholder, "No video found")
            
            # Find the most recently modified video file
            # We use os.path.getmtime() to get modification time
            latest_video = max(video_files, key=os.path.getmtime)
            
            # Extract the final frame from this video
            final_frame = self._extract_last_frame(latest_video)
            
            # Get just the filename for return value
            video_filename = os.path.basename(latest_video)
            
            return (final_frame, video_filename)
            
        except Exception as e:
            # Error handling - create placeholder with error message
            error_msg = f"Error: {str(e)}"
            placeholder = self._create_placeholder_image(error_msg)
            return (placeholder, error_msg)
    
    def _find_video_files(self, directory_path, recursive_search, video_extensions):
        """
        Helper method to find all video files in the directory.
        This separates the file-finding logic for better organization.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory does not exist: {directory_path}")
        
        # Parse the extensions string into a list
        # We strip whitespace and convert to lowercase for consistency
        extensions = [ext.strip().lower().lstrip('.') for ext in video_extensions.split(',')]
        
        video_files = []
        
        # Create search patterns for each extension
        for ext in extensions:
            if recursive_search:
                # Use ** for recursive search (Python 3.5+)
                pattern = os.path.join(directory_path, f"**/*.{ext}")
                video_files.extend(glob.glob(pattern, recursive=True))
            else:
                # Search only in the specified directory
                pattern = os.path.join(directory_path, f"*.{ext}")
                video_files.extend(glob.glob(pattern))
        
        return video_files
    
    def _extract_last_frame(self, video_path):
        """
        Extract the final frame from a video file.
        This is the core video processing logic.
        """
        # OpenCV is the most reliable library for video processing
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            # Get total number of frames in the video
            # This helps us jump directly to the end
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                raise ValueError("Video appears to have no frames")
            
            # Jump to the last frame (frame indexing starts at 0)
            # We use total_frames - 1 to get the actual last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            
            # Read the frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                raise ValueError("Could not read the final frame")
            
            # Convert the frame to the format ComfyUI expects
            return self._opencv_to_comfyui_image(frame)
            
        finally:
            # Always release the video capture object
            # This prevents memory leaks and file handle issues
            cap.release()
    
    def _opencv_to_comfyui_image(self, opencv_frame):
        """
        Convert OpenCV image format to ComfyUI's expected format.
        This handles the color space and tensor conversions.
        """
        # OpenCV uses BGR color order, but most systems expect RGB
        rgb_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for easier manipulation
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert PIL image to numpy array
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # ComfyUI expects images in the format [batch, height, width, channels]
        # Add batch dimension at the beginning
        image_tensor = torch.from_numpy(image_array)[None,]
        
        return image_tensor
    
    def _create_placeholder_image(self, message, width=512, height=512):
        """
        Create a placeholder image with text when something goes wrong.
        This provides visual feedback in the ComfyUI interface.
        """
        # Create a black image
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add text to the image using OpenCV
        # This helps users understand what happened
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # White text
        thickness = 2
        
        # Calculate text size to center it
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(placeholder, message, (text_x, text_y), 
                   font, font_scale, color, thickness)
        
        # Convert to ComfyUI format
        return self._opencv_to_comfyui_image(placeholder)


# This is how ComfyUI discovers and registers custom nodes
# The key should match your class name
NODE_CLASS_MAPPINGS = {
    "LatestVideoFinalFrameNode": LatestVideoFinalFrameNode
}

# Display names that appear in the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatestVideoFinalFrameNode": "Latest Video Final Frame"
}
