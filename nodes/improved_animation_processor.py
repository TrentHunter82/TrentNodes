import torch
import torch.nn.functional as F
import numpy as np

class AnimationDuplicateFrameProcessor:
    """
    A ComfyUI node that processes animation frames to replace duplicate sequences 
    with gray frames, making animation timing structure visible.
    
    Enhanced with multiple similarity metrics for more robust duplicate detection.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Batch of frames from video
                "similarity_method": (["hybrid", "ssim", "histogram", "perceptual"], {
                    "default": "hybrid",
                    "tooltip": "Method for calculating frame similarity"
                }),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.85,  # More reasonable default
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "How similar frames need to be to count as duplicates"
                }),
                "motion_tolerance": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Tolerance for small movements/changes"
                }),
                "gray_style": (["solid_gray", "desaturated", "dimmed"], {
                    "default": "desaturated",
                    "tooltip": "How to render the gray replacement frames"
                }),
                "gray_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Intensity of gray effect"
                }),
                "preserve_first": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the first frame of each duplicate sequence unchanged"
                }),
                "preserve_last": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the last frame of each duplicate sequence unchanged"
                }),
                "min_sequence_length": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Minimum frames in a sequence to consider as duplicates"
                }),
            },
            "optional": {
                "debug_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print detailed analysis of duplicate sequences found"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("processed_frames", "duplicate_mask", "timing_report")
    FUNCTION = "process_animation_timing"
    CATEGORY = "Trent/Video"
    
    def calculate_ssim(self, frame1, frame2):
        """
        Calculate Structural Similarity Index (SSIM) between two frames.
        More perceptually meaningful than cosine similarity.
        """
        # Convert to grayscale
        gray1 = torch.sum(frame1 * torch.tensor([0.299, 0.587, 0.114]), dim=-1)
        gray2 = torch.sum(frame2 * torch.tensor([0.299, 0.587, 0.114]), dim=-1)
        
        # Add batch and channel dimensions for SSIM calculation
        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)
        
        # SSIM calculation constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Calculate means
        mu1 = F.avg_pool2d(gray1, 3, 1, 1)
        mu2 = F.avg_pool2d(gray2, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.avg_pool2d(gray1 * gray1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(gray2 * gray2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(gray1 * gray2, 3, 1, 1) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def calculate_histogram_similarity(self, frame1, frame2):
        """
        Calculate histogram similarity between two frames.
        Good for detecting color/brightness changes.
        """
        # Convert to numpy for histogram calculation
        f1_np = frame1.cpu().numpy()
        f2_np = frame2.cpu().numpy()
        
        # Calculate histograms for each channel
        similarities = []
        for c in range(3):  # RGB channels
            hist1 = np.histogram(f1_np[:, :, c], bins=32, range=(0, 1))[0]
            hist2 = np.histogram(f2_np[:, :, c], bins=32, range=(0, 1))[0]
            
            # Normalize histograms
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Calculate correlation
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            similarities.append(correlation if not np.isnan(correlation) else 0.0)
        
        return np.mean(similarities)
    
    def calculate_perceptual_similarity(self, frame1, frame2):
        """
        Calculate perceptual similarity using edge detection and texture analysis.
        Good for detecting structural changes while ignoring lighting.
        """
        # Convert to grayscale
        gray1 = torch.sum(frame1 * torch.tensor([0.299, 0.587, 0.114]), dim=-1)
        gray2 = torch.sum(frame2 * torch.tensor([0.299, 0.587, 0.114]), dim=-1)
        
        # Simple edge detection using Sobel-like filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Add batch and channel dimensions
        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)
        
        # Calculate edges
        edges1_x = F.conv2d(gray1, sobel_x, padding=1)
        edges1_y = F.conv2d(gray1, sobel_y, padding=1)
        edges1 = torch.sqrt(edges1_x**2 + edges1_y**2)
        
        edges2_x = F.conv2d(gray2, sobel_x, padding=1)
        edges2_y = F.conv2d(gray2, sobel_y, padding=1)
        edges2 = torch.sqrt(edges2_x**2 + edges2_y**2)
        
        # Calculate correlation between edge maps
        flat1 = edges1.flatten()
        flat2 = edges2.flatten()
        
        correlation = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
        return correlation.item()
    
    def calculate_frame_similarity(self, frame1, frame2, method, motion_tolerance):
        """
        Calculate similarity between two frames using the specified method.
        Returns a confidence score between 0.0 and 1.0.
        """
        if method == "ssim":
            similarity = self.calculate_ssim(frame1, frame2)
        elif method == "histogram":
            similarity = self.calculate_histogram_similarity(frame1, frame2)
        elif method == "perceptual":
            similarity = self.calculate_perceptual_similarity(frame1, frame2)
        elif method == "hybrid":
            # Combine multiple methods for more robust detection
            ssim_score = self.calculate_ssim(frame1, frame2)
            hist_score = self.calculate_histogram_similarity(frame1, frame2)
            perc_score = self.calculate_perceptual_similarity(frame1, frame2)
            
            # Weighted combination (SSIM gets highest weight)
            similarity = 0.5 * ssim_score + 0.3 * hist_score + 0.2 * perc_score
        
        # Apply motion tolerance - if similarity is close to threshold, be more lenient
        # This helps with compression artifacts and minor changes
        if similarity > (1.0 - motion_tolerance):
            similarity = min(1.0, similarity + motion_tolerance * 0.5)
        
        return similarity
    
    def create_gray_frame(self, original_frame, gray_style, gray_intensity):
        """Create a gray version of the original frame based on the selected style."""
        if gray_style == "solid_gray":
            gray_frame = torch.full_like(original_frame, gray_intensity)
        elif gray_style == "desaturated":
            grayscale = torch.sum(original_frame * torch.tensor([0.299, 0.587, 0.114]), 
                                 dim=-1, keepdim=True)
            grayscale = grayscale.expand(-1, -1, 3)
            gray_frame = grayscale * gray_intensity + original_frame * (1 - gray_intensity)
        elif gray_style == "dimmed":
            gray_frame = original_frame * gray_intensity
            
        return gray_frame
    
    def analyze_duplicate_sequences(self, images, similarity_threshold, motion_tolerance, 
                                  similarity_method, min_sequence_length, debug_info):
        """
        Analyze the entire batch to find duplicate frame sequences with improved logic.
        """
        batch_size = images.shape[0]
        
        # Calculate similarities between consecutive frames
        similarities = []
        for i in range(1, batch_size):
            sim = self.calculate_frame_similarity(
                images[i-1], images[i], similarity_method, motion_tolerance
            )
            similarities.append(sim)
        
        if debug_info:
            print(f"\n=== Enhanced Animation Timing Analysis ===")
            print(f"Method: {similarity_method}, Threshold: {similarity_threshold}")
            print(f"Motion tolerance: {motion_tolerance}")
            print(f"Min sequence length: {min_sequence_length}")
        
        # Find sequences of consecutive similar frames
        sequences = []
        current_sequence_start = 0
        current_sequence_length = 1
        
        for i, similarity in enumerate(similarities):
            frame_idx = i + 1  # similarities[i] compares frame i with frame i+1
            
            if similarity >= similarity_threshold:
                current_sequence_length += 1
                if debug_info:
                    print(f"Frame {frame_idx}: SIMILAR (score: {similarity:.3f})")
            else:
                # End of sequence - check if it meets minimum length
                if current_sequence_length >= min_sequence_length:
                    sequences.append((current_sequence_start, current_sequence_length))
                    
                    if debug_info:
                        print(f"Sequence {current_sequence_start}-{current_sequence_start + current_sequence_length - 1}: "
                              f"{current_sequence_length} frames")
                
                current_sequence_start = frame_idx
                current_sequence_length = 1
                
                if debug_info:
                    print(f"Frame {frame_idx}: NEW SCENE (score: {similarity:.3f})")
        
        # Handle final sequence
        if current_sequence_length >= min_sequence_length:
            sequences.append((current_sequence_start, current_sequence_length))
        
        # Generate detailed report
        total_frames_to_replace = 0
        report = f"Enhanced Animation Timing Analysis:\n"
        report += f"Method: {similarity_method}\n"
        report += f"Total frames: {batch_size}\n"
        report += f"Duplicate sequences found: {len(sequences)}\n"
        report += f"Min sequence length: {min_sequence_length}\n"
        
        if sequences:
            report += f"\nSequence breakdown:\n"
            for i, (start, length) in enumerate(sequences):
                report += f"  Sequence {i+1}: frames {start}-{start+length-1} ({length} frames)\n"
                # Count frames that will actually be replaced 
                frames_to_replace = length
                if self.preserve_first_frame and length > 1:
                    frames_to_replace -= 1
                if self.preserve_last_frame and length > 1:
                    frames_to_replace -= 1
                # Make sure we don't go negative for very short sequences
                frames_to_replace = max(0, frames_to_replace)
                total_frames_to_replace += frames_to_replace
        
        report += f"Frames that will be replaced: {total_frames_to_replace}\n"
        
        if debug_info and similarities:
            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            report += f"\nSimilarity statistics:\n"
            report += f"  Average: {avg_sim:.3f}\n"
            report += f"  Min: {min_sim:.3f}\n"
            report += f"  Max: {max_sim:.3f}\n"
        
        return sequences, report
    
    def process_animation_timing(self, images, similarity_method, similarity_threshold, 
                               motion_tolerance, gray_style, gray_intensity, preserve_first, 
                               preserve_last, min_sequence_length, debug_info=False):
        """
        Main processing function with enhanced duplicate detection.
        """
        
        # Store preserve settings for use in analyze_duplicate_sequences
        self.preserve_first_frame = preserve_first
        self.preserve_last_frame = preserve_last
        
        if debug_info:
            print(f"Starting enhanced animation timing processing...")
            print(f"Input batch shape: {images.shape}")
        
        # Analyze the batch to identify duplicate sequences
        sequences, report = self.analyze_duplicate_sequences(
            images, similarity_threshold, motion_tolerance, similarity_method, 
            min_sequence_length, debug_info
        )
        
        # Create a copy of the original images to modify
        processed_images = images.clone()
        
        # Create mask tensor
        batch_size, height, width, channels = images.shape
        mask_tensor = torch.zeros(batch_size, height, width, dtype=images.dtype, device=images.device)
        
        # Process each sequence and replace appropriate frames
        frames_processed = 0
        
        if debug_info:
            print(f"\nProcessing {len(sequences)} sequences...")
            print(f"Preserve first frame: {preserve_first}")
            print(f"Preserve last frame: {preserve_last}")
        
        for seq_idx, (start_frame, sequence_length) in enumerate(sequences):
            end_frame = start_frame + sequence_length
            last_frame = end_frame - 1
            
            if debug_info:
                print(f"Processing sequence {seq_idx + 1}: frames {start_frame}-{last_frame}")
            
            for frame_idx in range(start_frame, end_frame):
                # Determine if this frame should be replaced
                should_replace = True
                
                if preserve_first and frame_idx == start_frame:
                    # Don't replace the first frame of the sequence
                    should_replace = False
                    if debug_info:
                        print(f"  Frame {frame_idx}: PRESERVED (first in sequence)")
                elif preserve_last and frame_idx == last_frame:
                    # Don't replace the last frame of the sequence
                    should_replace = False
                    if debug_info:
                        print(f"  Frame {frame_idx}: PRESERVED (last in sequence)")
                
                if should_replace:
                    # Replace this frame with a gray version
                    gray_frame = self.create_gray_frame(
                        images[frame_idx], gray_style, gray_intensity
                    )
                    processed_images[frame_idx] = gray_frame
                    
                    # Set mask to white (1.0) for this frame
                    mask_tensor[frame_idx] = 1.0
                    frames_processed += 1
                    
                    if debug_info:
                        print(f"  Frame {frame_idx}: REPLACED with gray")
        
        if debug_info:
            print(f"Total frames replaced: {frames_processed}")
            mask_white_pixels = torch.sum(mask_tensor > 0).item()
            print(f"Mask white pixels: {mask_white_pixels}")
        
        # Add processing summary to report
        report += f"\nProcessing results:\n"
        report += f"Frames replaced with gray: {frames_processed}\n"
        report += f"Method used: {similarity_method}\n"
        report += f"Motion tolerance: {motion_tolerance}\n"
        report += f"Gray style: {gray_style} (intensity: {gray_intensity})\n"
        report += f"Preserve first frame: {preserve_first}\n"
        report += f"Preserve last frame: {preserve_last}\n"
        
        return (processed_images, mask_tensor, report)

# Required mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AnimationDuplicateFrameProcessor": AnimationDuplicateFrameProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationDuplicateFrameProcessor": "Enhanced Animation Timing Processor"
}