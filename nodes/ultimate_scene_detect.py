import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import time
import os
from PIL import Image
import subprocess

class DetectionMethod(Enum):
    FAST_HISTOGRAM = "fast_histogram"
    FAST_OPTICAL_FLOW = "fast_optical_flow"
    HYBRID_FAST = "hybrid_fast"

class ExportFormat(Enum):
    JSON = "json"
    EDL = "edl"
    PREMIERE_MARKERS = "premiere_markers"
    CSV = "csv"

@dataclass
class SceneCut:
    """Represents a detected scene cut with essential information"""
    frame_number: int
    timestamp: float
    confidence: float
    method: str

class FastSceneDetector:
    """
    Optimized scene detection focused on speed and accuracy
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.logger = self._setup_logging()
        
        # Optimized parameters for speed and accuracy
        self.histogram_threshold = 0.4  # More sensitive for better detection
        self.flow_threshold = 8.0       # Adjusted for better motion detection
        self.min_scene_frames = 3       # Minimum frames between cuts
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("FastSceneDetector")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def fast_histogram_detection(self, frames: torch.Tensor) -> List[SceneCut]:
        """
        Optimized histogram-based detection focused on speed
        """
        cuts = []
        frames_np = frames.cpu().numpy()
        
        # Convert to numpy format and ensure we have RGB
        if len(frames_np.shape) == 4:
            frames_np = frames_np.transpose(0, 2, 3, 1)  # BCHW to BHWC
        
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # Process frames in batches for speed
        prev_hist = None
        
        for i, frame in enumerate(frames_np):
            # Calculate histogram - simplified for speed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB histogram - combine all channels
                hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            else:
                # Grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize
            
            if prev_hist is not None:
                # Chi-square distance (faster than Bhattacharyya for our use case)
                diff = np.sum((hist - prev_hist) ** 2 / (hist + prev_hist + 1e-8))
                
                if diff > self.histogram_threshold:
                    confidence = min(diff / self.histogram_threshold, 1.0)
                    cuts.append(SceneCut(
                        frame_number=i,
                        timestamp=i / 30.0,  # Will be updated with actual framerate
                        confidence=confidence,
                        method="fast_histogram"
                    ))
                    self.logger.debug(f"Histogram cut detected at frame {i}, diff: {diff:.3f}")
            
            prev_hist = hist
        
        return cuts
    
    def fast_optical_flow_detection(self, frames: torch.Tensor) -> List[SceneCut]:
        """
        Simplified optical flow detection for speed
        """
        cuts = []
        frames_np = frames.cpu().numpy()
        
        if len(frames_np.shape) == 4:
            # Convert to grayscale for optical flow
            frames_gray = np.mean(frames_np, axis=1)
        else:
            frames_gray = frames_np
            
        frames_gray = (frames_gray * 255).astype(np.uint8)
        
        for i in range(1, len(frames_gray)):
            prev_frame = frames_gray[i-1]
            curr_frame = frames_gray[i]
            
            try:
                # Use simpler optical flow calculation
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, curr_frame, None,
                    pyr_scale=0.5, levels=2, winsize=10,  # Reduced for speed
                    iterations=2, poly_n=3, poly_sigma=1.1, 
                    flags=0
                )
                
                # Calculate flow magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                
                # Simple threshold on mean magnitude
                mean_magnitude = np.mean(magnitude)
                
                if mean_magnitude > self.flow_threshold:
                    confidence = min(mean_magnitude / self.flow_threshold, 1.0)
                    cuts.append(SceneCut(
                        frame_number=i,
                        timestamp=i / 30.0,
                        confidence=confidence,
                        method="fast_optical_flow"
                    ))
                    self.logger.debug(f"Flow cut detected at frame {i}, magnitude: {mean_magnitude:.3f}")
                    
            except cv2.error:
                continue
        
        return cuts
    
    def detect_cuts(self, frames: torch.Tensor, method: DetectionMethod, frame_rate: float = 30.0) -> List[SceneCut]:
        """
        Main detection function optimized for speed and accuracy
        """
        self.logger.info(f"Detecting cuts using {method.value} on {len(frames)} frames...")
        start_time = time.time()
        
        if method == DetectionMethod.FAST_HISTOGRAM:
            cuts = self.fast_histogram_detection(frames)
        elif method == DetectionMethod.FAST_OPTICAL_FLOW:
            cuts = self.fast_optical_flow_detection(frames)
        elif method == DetectionMethod.HYBRID_FAST:
            # Combine both methods but weight histogram higher (it's more reliable)
            hist_cuts = self.fast_histogram_detection(frames)
            flow_cuts = self.fast_optical_flow_detection(frames)
            
            # Merge cuts from both methods
            all_cuts = hist_cuts + flow_cuts
            cuts = self._merge_nearby_cuts(all_cuts)
        else:
            cuts = self.fast_histogram_detection(frames)  # Default fallback
        
        # Update timestamps with correct framerate
        for cut in cuts:
            cut.timestamp = cut.frame_number / frame_rate
        
        # Filter cuts that are too close together
        cuts = self._filter_minimum_distance(cuts)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Detected {len(cuts)} cuts in {processing_time:.2f} seconds")
        
        return cuts
    
    def _merge_nearby_cuts(self, cuts: List[SceneCut]) -> List[SceneCut]:
        """Merge cuts that are very close to each other"""
        if not cuts:
            return cuts
            
        # Sort by frame number
        cuts.sort(key=lambda x: x.frame_number)
        
        merged = [cuts[0]]
        
        for cut in cuts[1:]:
            last_cut = merged[-1]
            
            # If cuts are within 5 frames, merge them
            if cut.frame_number - last_cut.frame_number <= 5:
                # Keep the one with higher confidence
                if cut.confidence > last_cut.confidence:
                    merged[-1] = cut
            else:
                merged.append(cut)
        
        return merged
    
    def _filter_minimum_distance(self, cuts: List[SceneCut]) -> List[SceneCut]:
        """Filter out cuts that are too close together"""
        if not cuts:
            return cuts
            
        cuts.sort(key=lambda x: x.frame_number)
        filtered = [cuts[0]]
        
        for cut in cuts[1:]:
            if cut.frame_number - filtered[-1].frame_number >= self.min_scene_frames:
                filtered.append(cut)
        
        return filtered

class VideoSplitter:
    """
    Handles splitting video into separate files based on detected cuts
    """
    
    def __init__(self):
        self.logger = logging.getLogger("VideoSplitter")
    
    def split_video_frames(self, frames: torch.Tensor, cuts: List[SceneCut], 
                          output_dir: str, base_name: str = "scene") -> List[str]:
        """
        Split frames into separate video segments and save as image sequences
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_segments = []
        frames_np = frames.cpu().numpy()
        
        # Convert tensor format to numpy images
        if len(frames_np.shape) == 4:
            frames_np = frames_np.transpose(0, 2, 3, 1)  # BCHW to BHWC
        
        frames_np = (frames_np * 255).astype(np.uint8)
        
        # Add start and end boundaries
        cut_frames = [0] + [cut.frame_number for cut in cuts] + [len(frames_np)]
        
        for i in range(len(cut_frames) - 1):
            start_frame = cut_frames[i]
            end_frame = cut_frames[i + 1]
            
            segment_name = f"{base_name}_{i+1:03d}"
            segment_dir = output_path / segment_name
            segment_dir.mkdir(exist_ok=True)
            
            # Save frames for this segment
            segment_frames = frames_np[start_frame:end_frame]
            
            for j, frame in enumerate(segment_frames):
                frame_path = segment_dir / f"frame_{j+1:04d}.png"
                
                # Convert to PIL Image and save
                if len(frame.shape) == 3:
                    img = Image.fromarray(frame, 'RGB')
                else:
                    img = Image.fromarray(frame, 'L')
                
                img.save(str(frame_path))
            
            saved_segments.append(str(segment_dir))
            self.logger.info(f"Saved segment {i+1}: {len(segment_frames)} frames to {segment_name}")
        
        return saved_segments
    
    def create_video_segments(self, frames: torch.Tensor, cuts: List[SceneCut],
                             output_dir: str, base_name: str = "scene",
                             frame_rate: float = 30.0, format: str = "mp4") -> List[str]:
        """
        Create actual video files from frame segments using ffmpeg
        """
        # First create image sequences
        segment_dirs = self.split_video_frames(frames, cuts, output_dir, base_name)
        
        video_files = []
        
        for i, segment_dir in enumerate(segment_dirs):
            segment_path = Path(segment_dir)
            video_name = f"{base_name}_{i+1:03d}.{format}"
            video_path = Path(output_dir) / video_name
            
            # Check if ffmpeg is available
            try:
                # Create video from image sequence using ffmpeg
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite existing files
                    '-framerate', str(frame_rate),
                    '-i', str(segment_path / 'frame_%04d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '18',  # High quality
                    str(video_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    video_files.append(str(video_path))
                    self.logger.info(f"Created video: {video_name}")
                    
                    # Clean up image sequence directory
                    import shutil
                    shutil.rmtree(segment_path)
                else:
                    self.logger.warning(f"ffmpeg failed for {video_name}: {result.stderr}")
                    video_files.append(str(segment_path))  # Return image sequence path instead
                    
            except FileNotFoundError:
                self.logger.warning("ffmpeg not found. Keeping image sequences instead of creating videos.")
                video_files.append(str(segment_path))
        
        return video_files

class FastExporter:
    """Simplified, fast export system"""
    
    @staticmethod
    def export_cuts(cuts: List[SceneCut], output_path: str, format: ExportFormat) -> bool:
        """Export cuts in specified format"""
        try:
            if format == ExportFormat.JSON:
                data = {
                    'cuts': [
                        {
                            'frame': cut.frame_number,
                            'timestamp': cut.timestamp,
                            'confidence': cut.confidence,
                            'method': cut.method
                        }
                        for cut in cuts
                    ],
                    'total_cuts': len(cuts),
                    'export_time': time.time()
                }
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif format == ExportFormat.CSV:
                with open(output_path, 'w') as f:
                    f.write("Frame,Timestamp,Confidence,Method\n")
                    for cut in cuts:
                        f.write(f"{cut.frame_number},{cut.timestamp:.3f},{cut.confidence:.3f},{cut.method}\n")
                        
            elif format == ExportFormat.PREMIERE_MARKERS:
                with open(output_path, 'w') as f:
                    f.write("Marker Name,Description,In,Out,Duration,Marker Type\n")
                    for i, cut in enumerate(cuts):
                        # Convert to timecode
                        hours = int(cut.timestamp // 3600)
                        minutes = int((cut.timestamp % 3600) // 60)
                        seconds = int(cut.timestamp % 60)
                        frames = int((cut.timestamp % 1) * 30)
                        timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
                        
                        f.write(f'"Cut {i+1}","Confidence: {cut.confidence:.3f}",{timecode},{timecode},00:00:00:01,Comment\n')
            
            return True
        except Exception as e:
            logging.error(f"Export failed: {e}")
            return False

# ComfyUI Node Implementation
class UltimateSceneCutterNode:
    """
    Fast, accurate scene detection and video splitting node
    """
    
    def __init__(self):
        self.detector = FastSceneDetector()
        self.splitter = VideoSplitter()
        self.exporter = FastExporter()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "detection_method": (["fast_histogram", "fast_optical_flow", "hybrid_fast"],),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "output_folder": ("STRING", {"default": "./scene_cuts"}),
                "base_filename": ("STRING", {"default": "scene"}),
                "create_videos": ("BOOLEAN", {"default": True}),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            },
            "optional": {
                "min_scene_length": ("INT", {"default": 30, "min": 1, "max": 300}),
                "export_format": (["json", "csv", "premiere_markers"],),
                "video_format": (["mp4", "avi", "mov"],),
                "keep_originals": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("original_images", "cuts_report", "output_paths", "num_segments")
    FUNCTION = "detect_and_split"
    CATEGORY = "Trent/Video"
    
    def detect_and_split(self, images, detection_method, sensitivity, output_folder, 
                        base_filename, create_videos, frame_rate,
                        min_scene_length=30, export_format="json", video_format="mp4",
                        keep_originals=False):
        
        # Convert images to proper tensor format
        if isinstance(images, list):
            frames = torch.stack(images)
        else:
            frames = images
            
        # Ensure correct tensor format [batch, channels, height, width]
        if len(frames.shape) == 4 and frames.shape[-1] in [1, 3, 4]:
            frames = frames.permute(0, 3, 1, 2)
        
        # Update detector parameters based on user settings
        self.detector.histogram_threshold = sensitivity
        self.detector.flow_threshold = sensitivity * 20.0  # Scale for flow detection
        self.detector.min_scene_frames = min_scene_length
        
        # Detect cuts
        method_enum = DetectionMethod(detection_method)
        cuts = self.detector.detect_cuts(frames, method_enum, frame_rate)
        
        # Split video if requested
        output_paths = []
        if create_videos or not create_videos:  # Always create some kind of output
            if create_videos:
                video_files = self.splitter.create_video_segments(
                    frames, cuts, output_folder, base_filename, frame_rate, video_format
                )
                output_paths = video_files
            else:
                # Just create image sequences
                image_dirs = self.splitter.split_video_frames(
                    frames, cuts, output_folder, base_filename
                )
                output_paths = image_dirs
        
        # Export cut data
        export_path = Path(output_folder) / f"{base_filename}_cuts.{export_format}"
        self.exporter.export_cuts(cuts, str(export_path), ExportFormat(export_format))
        
        # Generate report
        report = self._generate_report(cuts, len(frames), frame_rate, output_paths)
        
        return (images, report, str(export_path), len(output_paths))
    
    def _generate_report(self, cuts: List[SceneCut], total_frames: int, frame_rate: float, output_paths: List[str]):
        """Generate a comprehensive report"""
        if not cuts:
            return "No scene cuts detected. Consider lowering sensitivity or checking your video content."
        
        total_duration = total_frames / frame_rate
        avg_confidence = np.mean([cut.confidence for cut in cuts])
        
        # Calculate segment lengths
        segment_lengths = []
        cut_frames = [0] + [cut.frame_number for cut in cuts] + [total_frames]
        
        for i in range(len(cut_frames) - 1):
            length = (cut_frames[i + 1] - cut_frames[i]) / frame_rate
            segment_lengths.append(length)
        
        avg_segment_length = np.mean(segment_lengths)
        
        report = f"""FAST SCENE CUTTER REPORT
================================
Total Frames: {total_frames}
Video Duration: {total_duration:.2f} seconds
Cuts Detected: {len(cuts)}
Average Confidence: {avg_confidence:.3f}
Average Segment Length: {avg_segment_length:.2f} seconds

Segments Created: {len(output_paths)}

Cut Timeline (first 20):
"""
        
        for i, cut in enumerate(cuts[:20]):
            mins, secs = divmod(cut.timestamp, 60)
            report += f"  {i+1:2d}. {int(mins):02d}:{secs:05.2f} (Frame {cut.frame_number}) - Confidence: {cut.confidence:.3f}\n"
        
        if len(cuts) > 20:
            report += f"  ... and {len(cuts) - 20} more cuts\n"
        
        report += f"\nOutput files saved to: {Path(output_paths[0]).parent if output_paths else 'None'}"
        
        return report

# Node registration
NODE_CLASS_MAPPINGS = {
    "UltimateSceneCutter": UltimateSceneCutterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSceneCutter": "Ultimate Scene Cutter & Splitter"
}