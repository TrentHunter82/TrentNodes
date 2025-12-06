import os
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
from pathlib import Path
import json
import shutil
from typing import List, Tuple, Dict, Any, Optional
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re

from comfy.utils import ProgressBar, common_upscale

class EnhancedVideoCutter:
    """
    An advanced node that accurately detects scene cuts in video frames and exports
    them as organized, cleanly-named MP4 files with comprehensive metadata.
    """
    
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("video_paths", "output_folder", "metadata_json",)
    FUNCTION = "process_video_cuts"
    CATEGORY = "Trent/Video"
    
    DESCRIPTION = """
    Enhanced Video Cutter with improved accuracy and file management:
    - Adaptive scene detection with motion analysis
    - Clean, customizable file naming
    - Metadata tracking and export
    - Frame-accurate cutting
    - Better threshold calibration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_folder": ("STRING", {"default": "./output/video_cuts"}),
                "base_filename": ("STRING", {"default": "scene", 
                                "description": "Base name for output files (e.g., 'scene' produces 'scene_001.mp4')"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
                "threshold": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 100.0, "step": 0.1,
                             "description": "Adaptive threshold for scene detection"}),
                "min_scene_length": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1,
                                   "description": "Minimum frames per scene"}),
                "output_quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1}),
                "detection_method": (["adaptive", "motion_aware", "histogram", "edge_based", "hybrid"], 
                                   {"default": "adaptive"}),
                "naming_pattern": (["sequential", "timestamp", "frame_range", "custom"], 
                                 {"default": "sequential",
                                  "description": "How to name output files"}),
                "create_metadata": ("BOOLEAN", {"default": True,
                                "description": "Create JSON metadata file with scene information"}),
                "organize_subfolders": ("BOOLEAN", {"default": False,
                                     "description": "Organize scenes into subfolders by length/type"}),
                "motion_threshold": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                  "description": "Motion detection sensitivity for motion_aware method"}),
                "edge_sensitivity": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                                  "description": "Edge detection sensitivity (lower = more sensitive)"}),
                "max_workers": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "debug_mode": ("BOOLEAN", {"default": False,
                            "description": "Save debug information and intermediate frames"}),
            },
            "optional": {
                "custom_prefix": ("STRING", {"default": "",
                               "description": "Custom prefix for filenames"}),
                "zero_padding": ("INT", {"default": 3, "min": 1, "max": 6, "step": 1,
                               "description": "Number of digits for sequential numbering"}),
            }
        }
    
    def process_video_cuts(self, images: torch.Tensor, output_folder: str, base_filename: str,
                          fps: int, threshold: float, min_scene_length: int, output_quality: int,
                          detection_method: str, naming_pattern: str, create_metadata: bool,
                          organize_subfolders: bool, motion_threshold: float, edge_sensitivity: float,
                          max_workers: int, use_gpu: bool, debug_mode: bool,
                          custom_prefix: str = "", zero_padding: int = 3) -> Tuple[str, str, str]:
        """
        Enhanced video cutting with improved accuracy and organization.
        """
        print(f"[Enhanced Video Cutter] Starting processing with {detection_method} detection")
        
        # Setup output directory structure
        output_folder = os.path.abspath(output_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if organize_subfolders:
            session_folder = os.path.join(output_folder, f"session_{timestamp}")
            os.makedirs(session_folder, exist_ok=True)
            output_folder = session_folder
        else:
            os.makedirs(output_folder, exist_ok=True)
        
        # Initialize debug folder if needed
        if debug_mode:
            debug_folder = os.path.join(output_folder, "debug")
            os.makedirs(debug_folder, exist_ok=True)
        
        # Get batch dimensions
        batch_size, height, width, channels = images.shape
        print(f"[Enhanced Video Cutter] Processing {batch_size} frames ({width}x{height})")
        
        # Prepare images for processing
        device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        
        # Convert to numpy for processing
        np_images = (images * 255).cpu().numpy().astype(np.uint8)
        
        # Perform adaptive scene detection
        scene_data = self._detect_scenes_advanced(
            np_images, threshold, min_scene_length, detection_method, 
            motion_threshold, edge_sensitivity, device, debug_mode, 
            debug_folder if debug_mode else None
        )
        
        # Create organized video files
        video_info = self._create_organized_videos(
            np_images, scene_data, output_folder, base_filename, fps, 
            output_quality, naming_pattern, custom_prefix, zero_padding,
            organize_subfolders, max_workers
        )
        
        # Generate metadata if requested
        metadata_path = ""
        if create_metadata:
            metadata_path = self._create_metadata_file(
                video_info, scene_data, output_folder, 
                width, height, fps, detection_method, threshold
            )
        
        # Generate summary
        print(f"\n[Enhanced Video Cutter] Summary:")
        print(f"  - Total frames processed: {batch_size}")
        print(f"  - Scenes detected: {len(scene_data['scenes'])}")
        print(f"  - Videos created: {len(video_info)}")
        print(f"  - Output folder: {output_folder}")
        
        if debug_mode:
            self._save_debug_info(scene_data, debug_folder)
        
        # Return paths
        video_paths = [info['path'] for info in video_info]
        return ",".join(video_paths), output_folder, metadata_path
    
    def _detect_scenes_advanced(self, np_images: np.ndarray, threshold: float, 
                               min_scene_length: int, method: str, motion_threshold: float,
                               edge_sensitivity: float, device: str, debug_mode: bool,
                               debug_folder: Optional[str]) -> Dict:
        """
        Advanced scene detection with multiple algorithms and adaptive thresholds.
        """
        batch_size = np_images.shape[0]
        
        # Initialize detection data
        detection_data = {
            'scenes': [],
            'scores': [],
            'method': method,
            'frame_metrics': []
        }
        
        print(f"[Enhanced Video Cutter] Running {method} detection on {batch_size} frames...")
        
        # Select detection algorithm
        if method == "adaptive":
            scenes = self._adaptive_detection(np_images, threshold, min_scene_length, detection_data)
        elif method == "motion_aware":
            scenes = self._motion_aware_detection(np_images, motion_threshold, min_scene_length, detection_data)
        elif method == "histogram":
            scenes = self._histogram_detection(np_images, threshold, min_scene_length, detection_data)
        elif method == "edge_based":
            scenes = self._edge_based_detection(np_images, edge_sensitivity, min_scene_length, detection_data)
        else:  # hybrid
            scenes = self._hybrid_detection(np_images, threshold, min_scene_length, detection_data)
        
        detection_data['scenes'] = scenes
        
        # Validate and refine cuts
        detection_data['scenes'] = self._validate_scene_cuts(
            np_images, scenes, min_scene_length, debug_mode
        )
        
        return detection_data
    
    def _adaptive_detection(self, frames: np.ndarray, base_threshold: float, 
                           min_length: int, detection_data: Dict) -> List[Dict]:
        """
        Adaptive detection that adjusts threshold based on content dynamics.
        """
        batch_size = frames.shape[0]
        scenes = []
        
        # Calculate frame differences
        differences = []
        gray_frames = []
        
        print("[Enhanced Video Cutter] Computing frame metrics...")
        pbar = ProgressBar(batch_size)
        
        # Pre-compute grayscale frames
        for i in range(batch_size):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray_frames.append(gray)
            pbar.update_absolute(i)
        
        # Compute differences between consecutive frames
        for i in range(1, batch_size):
            # Multi-metric difference calculation
            diff_intensity = np.mean(cv2.absdiff(gray_frames[i], gray_frames[i-1]))
            
            # Structural similarity for more accurate detection
            diff_struct = self._compute_structural_diff(gray_frames[i], gray_frames[i-1])
            
            # Color histogram difference
            diff_color = self._compute_color_histogram_diff(frames[i], frames[i-1])
            
            # Weighted combination
            combined_diff = (diff_intensity * 0.4) + (diff_struct * 0.3) + (diff_color * 0.3)
            differences.append(combined_diff)
            
            detection_data['frame_metrics'].append({
                'frame': i,
                'intensity_diff': float(diff_intensity),
                'structural_diff': float(diff_struct),
                'color_diff': float(diff_color),
                'combined': float(combined_diff)
            })
        
        # Calculate adaptive threshold based on content statistics
        if differences:
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            adaptive_threshold = mean_diff + (base_threshold / 10) * std_diff
            
            print(f"[Enhanced Video Cutter] Adaptive threshold: {adaptive_threshold:.2f} "
                  f"(mean: {mean_diff:.2f}, std: {std_diff:.2f})")
        else:
            adaptive_threshold = base_threshold
        
        # Detect scene boundaries
        current_scene_start = 0
        
        for i in range(len(differences)):
            if differences[i] > adaptive_threshold:
                # Check if minimum scene length is met
                if (i + 1) - current_scene_start >= min_length:
                    scenes.append({
                        'start': current_scene_start,
                        'end': i + 1,
                        'length': (i + 1) - current_scene_start,
                        'confidence': float(differences[i] / adaptive_threshold)
                    })
                    current_scene_start = i + 1
        
        # Add final scene
        if batch_size - current_scene_start >= min_length:
            scenes.append({
                'start': current_scene_start,
                'end': batch_size,
                'length': batch_size - current_scene_start,
                'confidence': 1.0
            })
        
        return scenes
    
    def _motion_aware_detection(self, frames: np.ndarray, motion_threshold: float,
                               min_length: int, detection_data: Dict) -> List[Dict]:
        """
        Detection based on optical flow and motion analysis.
        """
        batch_size = frames.shape[0]
        scenes = []
        
        print("[Enhanced Video Cutter] Analyzing motion patterns...")
        
        # Initialize optical flow parameters
        flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Convert frames to grayscale for optical flow
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        
        # Calculate optical flow between consecutive frames
        motion_scores = []
        for i in range(1, batch_size):
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i], None, **flow_params
            )
            
            # Calculate motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_score = np.mean(magnitude)
            motion_scores.append(motion_score)
            
            detection_data['frame_metrics'].append({
                'frame': i,
                'motion_score': float(motion_score),
                'max_motion': float(np.max(magnitude)),
                'motion_std': float(np.std(magnitude))
            })
        
        # Detect sudden motion changes
        current_scene_start = 0
        
        for i in range(1, len(motion_scores)):
            motion_change = abs(motion_scores[i] - motion_scores[i-1])
            
            if motion_change > motion_threshold:
                if (i + 1) - current_scene_start >= min_length:
                    scenes.append({
                        'start': current_scene_start,
                        'end': i + 1,
                        'length': (i + 1) - current_scene_start,
                        'confidence': float(motion_change / motion_threshold)
                    })
                    current_scene_start = i + 1
        
        # Add final scene
        if batch_size - current_scene_start >= min_length:
            scenes.append({
                'start': current_scene_start,
                'end': batch_size,
                'length': batch_size - current_scene_start,
                'confidence': 1.0
            })
        
        return scenes
    
    def _histogram_detection(self, frames: np.ndarray, threshold: float,
                            min_length: int, detection_data: Dict) -> List[Dict]:
        """
        Enhanced histogram-based detection with multiple color spaces.
        """
        batch_size = frames.shape[0]
        scenes = []
        
        print("[Enhanced Video Cutter] Computing color histograms...")
        
        # Pre-compute histograms for all frames
        histograms = []
        for i in range(batch_size):
            # Convert to multiple color spaces for robustness
            hsv = cv2.cvtColor(frames[i], cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(frames[i], cv2.COLOR_RGB2LAB)
            
            # Compute histograms in different color spaces
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256])
            
            # Normalize and combine
            cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_l, hist_l, 0, 1, cv2.NORM_MINMAX)
            
            histograms.append({
                'hue': hist_h,
                'saturation': hist_s,
                'lightness': hist_l
            })
        
        # Compare consecutive histograms
        current_scene_start = 0
        
        for i in range(1, batch_size):
            # Compare using multiple metrics
            diff_h = cv2.compareHist(histograms[i]['hue'], histograms[i-1]['hue'], 
                                    cv2.HISTCMP_CHISQR)
            diff_s = cv2.compareHist(histograms[i]['saturation'], histograms[i-1]['saturation'],
                                    cv2.HISTCMP_CHISQR)
            diff_l = cv2.compareHist(histograms[i]['lightness'], histograms[i-1]['lightness'],
                                    cv2.HISTCMP_CHISQR)
            
            # Weighted combination
            combined_diff = (diff_h * 0.4) + (diff_s * 0.3) + (diff_l * 0.3)
            
            if combined_diff > threshold:
                if i - current_scene_start >= min_length:
                    scenes.append({
                        'start': current_scene_start,
                        'end': i,
                        'length': i - current_scene_start,
                        'confidence': float(combined_diff / threshold)
                    })
                    current_scene_start = i
        
        # Add final scene
        if batch_size - current_scene_start >= min_length:
            scenes.append({
                'start': current_scene_start,
                'end': batch_size,
                'length': batch_size - current_scene_start,
                'confidence': 1.0
            })
        
        return scenes
    
    def _edge_based_detection(self, frames: np.ndarray, sensitivity: float,
                             min_length: int, detection_data: Dict) -> List[Dict]:
        """
        Edge-based scene detection using Canny edge detector.
        """
        batch_size = frames.shape[0]
        scenes = []
        
        print("[Enhanced Video Cutter] Analyzing edge patterns...")
        
        # Convert to grayscale and detect edges
        edge_maps = []
        for i in range(batch_size):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection with adaptive thresholds
            median_val = np.median(blurred)
            lower = int(max(0, (1.0 - sensitivity) * median_val))
            upper = int(min(255, (1.0 + sensitivity) * median_val))
            edges = cv2.Canny(blurred, lower, upper)
            
            edge_maps.append(edges)
        
        # Compare edge maps between consecutive frames
        current_scene_start = 0
        edge_threshold = 0.3  # Percentage of edge change
        
        for i in range(1, batch_size):
            # Calculate edge difference
            edge_diff = cv2.absdiff(edge_maps[i], edge_maps[i-1])
            diff_ratio = np.sum(edge_diff > 0) / edge_diff.size
            
            if diff_ratio > edge_threshold * (2 - sensitivity):
                if i - current_scene_start >= min_length:
                    scenes.append({
                        'start': current_scene_start,
                        'end': i,
                        'length': i - current_scene_start,
                        'confidence': float(diff_ratio / edge_threshold)
                    })
                    current_scene_start = i
        
        # Add final scene
        if batch_size - current_scene_start >= min_length:
            scenes.append({
                'start': current_scene_start,
                'end': batch_size,
                'length': batch_size - current_scene_start,
                'confidence': 1.0
            })
        
        return scenes
    
    def _hybrid_detection(self, frames: np.ndarray, threshold: float,
                         min_length: int, detection_data: Dict) -> List[Dict]:
        """
        Combines multiple detection methods for maximum accuracy.
        """
        # Run multiple detection methods
        adaptive_scenes = self._adaptive_detection(frames, threshold, min_length, {'frame_metrics': []})
        motion_scenes = self._motion_aware_detection(frames, 15.0, min_length, {'frame_metrics': []})
        histogram_scenes = self._histogram_detection(frames, threshold * 3, min_length, {'frame_metrics': []})
        
        # Merge and vote on scene boundaries
        all_boundaries = set()
        
        for scenes_list in [adaptive_scenes, motion_scenes, histogram_scenes]:
            for scene in scenes_list:
                all_boundaries.add(scene['start'])
                all_boundaries.add(scene['end'])
        
        # Sort boundaries and create consensus scenes
        boundaries = sorted(list(all_boundaries))
        scenes = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            if end - start >= min_length:
                # Calculate confidence based on how many methods detected this boundary
                confidence = 0
                for method_scenes in [adaptive_scenes, motion_scenes, histogram_scenes]:
                    for scene in method_scenes:
                        if scene['start'] <= start and scene['end'] >= end:
                            confidence += 0.33
                
                scenes.append({
                    'start': start,
                    'end': end,
                    'length': end - start,
                    'confidence': min(1.0, confidence)
                })
        
        return scenes
    
    def _compute_structural_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute structural similarity difference between two frames.
        """
        # Calculate SSIM (Structural Similarity Index)
        # Using a simplified version for speed
        
        # Calculate means
        mu1 = cv2.GaussianBlur(frame1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(frame2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances
        sigma1_sq = cv2.GaussianBlur(frame1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(frame2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(frame1 * frame2, (11, 11), 1.5) - mu1_mu2
        
        # SSIM formula constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        # Return dissimilarity (1 - SSIM)
        return float(1 - np.mean(ssim_map))
    
    def _compute_color_histogram_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute color histogram difference between two frames.
        """
        # Convert to HSV for better color comparison
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)
        
        # Calculate histograms with fewer bins for speed
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [16, 16], [0, 180, 0, 256])
        
        # Normalize
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare using Bhattacharyya distance
        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
    
    def _validate_scene_cuts(self, frames: np.ndarray, scenes: List[Dict],
                            min_length: int, debug_mode: bool) -> List[Dict]:
        """
        Validate and refine scene cuts for accuracy.
        """
        if not scenes:
            return scenes
        
        print("[Enhanced Video Cutter] Validating scene cuts...")
        
        validated_scenes = []
        
        for scene in scenes:
            # Skip scenes that are too short
            if scene['length'] < min_length:
                continue
            
            # Verify the cut is accurate by checking frame similarity at boundaries
            if scene['start'] > 0:
                # Check if the boundary frames are actually different
                boundary_diff = self._compute_structural_diff(
                    cv2.cvtColor(frames[scene['start']], cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(frames[scene['start'] - 1], cv2.COLOR_RGB2GRAY)
                )
                
                # If frames are too similar, adjust the boundary
                if boundary_diff < 0.1:  # Very similar frames
                    # Try to find a better cut point nearby
                    best_cut = scene['start']
                    best_diff = boundary_diff
                    
                    for offset in range(-2, 3):
                        test_idx = scene['start'] + offset
                        if 0 < test_idx < len(frames):
                            test_diff = self._compute_structural_diff(
                                cv2.cvtColor(frames[test_idx], cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor(frames[test_idx - 1], cv2.COLOR_RGB2GRAY)
                            )
                            if test_diff > best_diff:
                                best_cut = test_idx
                                best_diff = test_diff
                    
                    scene['start'] = best_cut
                    scene['length'] = scene['end'] - scene['start']
            
            # Add validated scene
            if scene['length'] >= min_length:
                validated_scenes.append(scene)
        
        # Merge very short gaps between scenes
        merged_scenes = []
        i = 0
        while i < len(validated_scenes):
            current_scene = validated_scenes[i].copy()
            
            # Check if there's a next scene
            if i + 1 < len(validated_scenes):
                next_scene = validated_scenes[i + 1]
                gap = next_scene['start'] - current_scene['end']
                
                # If gap is very small, merge the scenes
                if gap < 3:
                    current_scene['end'] = next_scene['end']
                    current_scene['length'] = current_scene['end'] - current_scene['start']
                    current_scene['confidence'] = max(current_scene['confidence'], 
                                                     next_scene['confidence'])
                    i += 2  # Skip the next scene since we merged it
                else:
                    merged_scenes.append(current_scene)
                    i += 1
            else:
                merged_scenes.append(current_scene)
                i += 1
        
        print(f"[Enhanced Video Cutter] Validated {len(merged_scenes)} scenes "
              f"(from {len(scenes)} detected)")
        
        return merged_scenes
    
    def _create_organized_videos(self, frames: np.ndarray, scene_data: Dict, 
                                output_folder: str, base_filename: str, fps: int,
                                quality: int, naming_pattern: str, custom_prefix: str,
                                zero_padding: int, organize_subfolders: bool,
                                max_workers: int) -> List[Dict]:
        """
        Create organized video files with clean naming.
        """
        scenes = scene_data['scenes']
        video_info = []
        
        print(f"[Enhanced Video Cutter] Creating {len(scenes)} video files...")
        
        # Prepare file naming
        def generate_filename(scene_idx: int, scene: Dict) -> str:
            # Clean the base filename
            clean_base = re.sub(r'[^\w\-_]', '', base_filename)
            
            if custom_prefix:
                clean_prefix = re.sub(r'[^\w\-_]', '', custom_prefix)
                name_base = f"{clean_prefix}_{clean_base}"
            else:
                name_base = clean_base
            
            if naming_pattern == "sequential":
                # Simple sequential naming: scene_001.mp4
                return f"{name_base}_{str(scene_idx + 1).zfill(zero_padding)}.mp4"
            
            elif naming_pattern == "timestamp":
                # Include timestamp: scene_001_1234567890.mp4
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                return f"{name_base}_{str(scene_idx + 1).zfill(zero_padding)}_{timestamp}.mp4"
            
            elif naming_pattern == "frame_range":
                # Include frame range: scene_001_f0000-0100.mp4
                return f"{name_base}_{str(scene_idx + 1).zfill(zero_padding)}_f{scene['start']:04d}-{scene['end']:04d}.mp4"
            
            else:  # custom
                # Custom pattern with all available info
                return f"{name_base}_{str(scene_idx + 1).zfill(zero_padding)}_{scene['length']}f.mp4"
        
        # Organize into subfolders if requested
        def get_output_path(filename: str, scene: Dict) -> str:
            if organize_subfolders:
                # Categorize by scene length
                if scene['length'] < 30:
                    subfolder = "short_clips"
                elif scene['length'] < 150:
                    subfolder = "medium_clips"
                else:
                    subfolder = "long_clips"
                
                folder_path = os.path.join(output_folder, subfolder)
                os.makedirs(folder_path, exist_ok=True)
                return os.path.join(folder_path, filename)
            else:
                return os.path.join(output_folder, filename)
        
        # Function to create a single video
        def create_video(scene_idx: int, scene: Dict) -> Dict:
            try:
                filename = generate_filename(scene_idx, scene)
                filepath = get_output_path(filename, scene)
                
                # Calculate quality settings
                crf = str(int((100 - quality) / 4) + 1)
                
                # Get frame dimensions
                height, width = frames[0].shape[:2]
                
                # Setup ffmpeg command
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", "rgb24",
                    "-r", str(fps),
                    "-i", "pipe:",
                    "-c:v", "libx264",
                    "-crf", crf,
                    "-preset", "fast",
                    "-tune", "film",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-loglevel", "error",
                    filepath
                ]
                
                # Create video
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Write frames
                for i in range(scene['start'], scene['end']):
                    process.stdin.write(frames[i].tobytes())
                
                process.stdin.close()
                process.wait()
                
                if process.returncode != 0:
                    stderr = process.stderr.read().decode('utf-8')
                    raise RuntimeError(f"FFmpeg error: {stderr}")
                
                # Get file size
                file_size = os.path.getsize(filepath)
                
                return {
                    'success': True,
                    'idx': scene_idx,
                    'path': filepath,
                    'filename': filename,
                    'start_frame': scene['start'],
                    'end_frame': scene['end'],
                    'frame_count': scene['length'],
                    'duration_sec': scene['length'] / fps,
                    'file_size': file_size,
                    'confidence': scene.get('confidence', 1.0)
                }
                
            except Exception as e:
                print(f"[Enhanced Video Cutter] Error creating video {scene_idx}: {str(e)}")
                return {
                    'success': False,
                    'idx': scene_idx,
                    'error': str(e)
                }
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, scene in enumerate(scenes):
                futures.append(executor.submit(create_video, idx, scene))
            
            pbar = ProgressBar(len(futures))
            completed = 0
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    video_info.append(result)
                    print(f"[Enhanced Video Cutter] Created: {result['filename']} "
                          f"({result['frame_count']} frames, {result['duration_sec']:.2f}s)")
                
                completed += 1
                pbar.update_absolute(completed)
        
        # Sort by index to maintain order
        video_info.sort(key=lambda x: x['idx'])
        
        return video_info
    
    def _create_metadata_file(self, video_info: List[Dict], scene_data: Dict,
                            output_folder: str, width: int, height: int, 
                            fps: int, method: str, threshold: float) -> str:
        """
        Create a JSON metadata file with all scene information.
        """
        metadata = {
            'creation_time': datetime.now().isoformat(),
            'processing_info': {
                'detection_method': method,
                'threshold': threshold,
                'frame_dimensions': f"{width}x{height}",
                'fps': fps,
                'total_scenes': len(video_info)
            },
            'scenes': []
        }
        
        for info in video_info:
            scene_meta = {
                'index': info['idx'],
                'filename': info['filename'],
                'path': info['path'],
                'start_frame': info['start_frame'],
                'end_frame': info['end_frame'],
                'frame_count': info['frame_count'],
                'duration_seconds': info['duration_sec'],
                'file_size_bytes': info['file_size'],
                'confidence': info['confidence']
            }
            metadata['scenes'].append(scene_meta)
        
        # Add frame metrics if available
        if 'frame_metrics' in scene_data and scene_data['frame_metrics']:
            metadata['frame_analysis'] = scene_data['frame_metrics'][:100]  # Limit to first 100
        
        # Save metadata file
        metadata_path = os.path.join(output_folder, "scene_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Enhanced Video Cutter] Metadata saved to: {metadata_path}")
        
        return metadata_path
    
    def _save_debug_info(self, scene_data: Dict, debug_folder: str):
        """
        Save debug information for analysis.
        """
        # Save detailed metrics
        debug_file = os.path.join(debug_folder, "detection_debug.json")
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'method': scene_data['method'],
            'scenes_detected': len(scene_data['scenes']),
            'scenes': scene_data['scenes'],
            'frame_metrics': scene_data.get('frame_metrics', [])
        }
        
        with open(debug_file, 'w') as f:
            json.dump(debug_info, f, indent=2)
        
        print(f"[Enhanced Video Cutter] Debug info saved to: {debug_file}")

# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "EnhancedVideoCutter": EnhancedVideoCutter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedVideoCutter": "Enhanced Video Cutter"
}