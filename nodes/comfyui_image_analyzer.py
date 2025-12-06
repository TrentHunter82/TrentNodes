import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for ComfyUI
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from PIL import Image

class ImageBatchAnalyzer:
    """
    A ComfyUI node that analyzes a batch of images and generates
    comprehensive visual reports about their properties.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the inputs this node accepts.
        ComfyUI uses this method to build the UI for your node.
        """
        return {
            "required": {
                # Images come in as a batch tensor [B, H, W, C]
                "images": ("IMAGE",),
            },
            "optional": {
                # Let users choose what kind of analysis to perform
                "analysis_type": (["comprehensive", "bit_depth", "color_distribution", "statistics"],),
                "graph_width": ("INT", {
                    "default": 1920,
                    "min": 640,
                    "max": 3840,
                    "step": 1
                }),
                "graph_height": ("INT", {
                    "default": 1080,
                    "min": 480,
                    "max": 2160,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("analysis_graph", "statistics_text")
    FUNCTION = "analyze_batch"
    CATEGORY = "Trent Nodes"
    
    def analyze_batch(self, images, analysis_type="comprehensive", graph_width=1920, graph_height=1080):
        """
        Main function that orchestrates the analysis.
        
        Args:
            images: Tensor of shape [batch, height, width, channels]
            analysis_type: Type of analysis to perform
            graph_width/height: Dimensions of output graph
        
        Returns:
            Tuple of (graph_image_tensor, statistics_string)
        """
        # Convert from ComfyUI's tensor format to numpy for easier analysis
        # ComfyUI images are typically float32 in range [0, 1]
        images_np = images.cpu().numpy()
        batch_size = images_np.shape[0]
        
        # Gather statistics for each image in the batch
        stats = self._gather_statistics(images_np)
        
        # Create the appropriate visualization based on analysis type
        if analysis_type == "comprehensive":
            fig = self._create_comprehensive_graph(stats, batch_size)
        elif analysis_type == "bit_depth":
            fig = self._create_bit_depth_graph(stats, batch_size)
        elif analysis_type == "color_distribution":
            fig = self._create_color_distribution_graph(stats, batch_size)
        else:  # statistics
            fig = self._create_statistics_graph(stats, batch_size)
        
        # Set figure size based on user preferences
        fig.set_size_inches(graph_width / 100, graph_height / 100)
        
        # Convert matplotlib figure to image tensor that ComfyUI can use
        graph_tensor = self._fig_to_tensor(fig)
        plt.close(fig)
        
        # Create a text summary of the statistics
        text_summary = self._create_text_summary(stats, batch_size)
        
        return (graph_tensor, text_summary)
    
    def _gather_statistics(self, images_np):
        """
        Analyze each image and extract relevant statistics.
        This is where we examine bit depth, color depth, and distribution.
        """
        batch_size = images_np.shape[0]
        stats = {
            'effective_bit_depth': [],
            'unique_values_per_channel': [],
            'mean_values': [],
            'std_values': [],
            'channel_ranges': [],
            'color_entropy': [],
            'histogram_data': []
        }
        
        for i in range(batch_size):
            img = images_np[i]
            
            # Calculate effective bit depth
            # This tells us how many unique intensity levels are actually used
            effective_bits = self._calculate_effective_bit_depth(img)
            stats['effective_bit_depth'].append(effective_bits)
            
            # Count unique values per channel (R, G, B)
            unique_counts = []
            for channel in range(img.shape[2]):
                channel_data = img[:, :, channel]
                unique_vals = len(np.unique(channel_data))
                unique_counts.append(unique_vals)
            stats['unique_values_per_channel'].append(unique_counts)
            
            # Calculate mean and standard deviation per channel
            mean_vals = [img[:, :, c].mean() for c in range(img.shape[2])]
            std_vals = [img[:, :, c].std() for c in range(img.shape[2])]
            stats['mean_values'].append(mean_vals)
            stats['std_values'].append(std_vals)
            
            # Calculate dynamic range per channel
            ranges = [img[:, :, c].max() - img[:, :, c].min() for c in range(img.shape[2])]
            stats['channel_ranges'].append(ranges)
            
            # Calculate color entropy (measure of color diversity)
            entropy = self._calculate_entropy(img)
            stats['color_entropy'].append(entropy)
            
            # Store histogram data for visualization
            hist_data = self._calculate_histogram(img)
            stats['histogram_data'].append(hist_data)
        
        return stats
    
    def _calculate_effective_bit_depth(self, img):
        """
        Calculate the effective bit depth by finding how many unique values
        are actually present in the image. An 8-bit image can have 256 values
        per channel, but might only use 128 unique values (7 effective bits).
        """
        # Flatten all channels together
        flattened = img.flatten()
        unique_values = len(np.unique(flattened))
        
        # Calculate bits needed to represent this many unique values
        if unique_values <= 1:
            return 1
        effective_bits = np.log2(unique_values)
        
        return effective_bits
    
    def _calculate_entropy(self, img):
        """
        Calculate Shannon entropy as a measure of color diversity.
        Higher entropy means more diverse colors.
        """
        # Convert to histogram
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 1))
        # Normalize to get probability distribution
        hist = hist / hist.sum()
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        # Calculate Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def _calculate_histogram(self, img):
        """
        Calculate color histograms for each channel.
        """
        histograms = []
        for channel in range(img.shape[2]):
            hist, bins = np.histogram(img[:, :, channel], bins=64, range=(0, 1))
            histograms.append((hist, bins))
        return histograms
    
    def _create_comprehensive_graph(self, stats, batch_size):
        """
        Create a comprehensive multi-panel graph showing all analysis metrics.
        """
        fig = Figure(facecolor='white')
        
        # Create a 3x2 grid of subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Effective Bit Depth over frames
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(range(batch_size), stats['effective_bit_depth'], 
                marker='o', linewidth=2, markersize=4, color='#2E86AB')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Effective Bit Depth')
        ax1.set_title('Bit Depth Analysis', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Color Entropy over frames
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(range(batch_size), stats['color_entropy'], 
                marker='s', linewidth=2, markersize=4, color='#A23B72')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Shannon Entropy')
        ax2.set_title('Color Diversity (Entropy)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Unique values per channel (averaged across batch)
        ax3 = fig.add_subplot(gs[1, 0])
        avg_unique = np.mean(stats['unique_values_per_channel'], axis=0)
        channels = ['R', 'G', 'B'] if len(avg_unique) == 3 else [f'Ch{i}' for i in range(len(avg_unique))]
        colors = ['#F18F01', '#C73E1D', '#6A994E'] if len(avg_unique) == 3 else None
        ax3.bar(channels, avg_unique, color=colors, alpha=0.7)
        ax3.set_ylabel('Average Unique Values')
        ax3.set_title('Color Depth by Channel', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Mean intensity per channel over frames
        ax4 = fig.add_subplot(gs[1, 1])
        mean_vals_array = np.array(stats['mean_values'])
        for ch in range(mean_vals_array.shape[1]):
            ax4.plot(range(batch_size), mean_vals_array[:, ch], 
                    label=channels[ch] if ch < len(channels) else f'Ch{ch}',
                    linewidth=2, marker='o', markersize=3)
        ax4.set_xlabel('Frame Number')
        ax4.set_ylabel('Mean Intensity (0-1)')
        ax4.set_title('Average Brightness per Channel', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Dynamic range per channel over frames
        ax5 = fig.add_subplot(gs[2, 0])
        ranges_array = np.array(stats['channel_ranges'])
        for ch in range(ranges_array.shape[1]):
            ax5.plot(range(batch_size), ranges_array[:, ch],
                    label=channels[ch] if ch < len(channels) else f'Ch{ch}',
                    linewidth=2, marker='s', markersize=3)
        ax5.set_xlabel('Frame Number')
        ax5.set_ylabel('Dynamic Range')
        ax5.set_title('Contrast Range per Channel', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Standard deviation trends
        ax6 = fig.add_subplot(gs[2, 1])
        std_vals_array = np.array(stats['std_values'])
        for ch in range(std_vals_array.shape[1]):
            ax6.plot(range(batch_size), std_vals_array[:, ch],
                    label=channels[ch] if ch < len(channels) else f'Ch{ch}',
                    linewidth=2, marker='^', markersize=3)
        ax6.set_xlabel('Frame Number')
        ax6.set_ylabel('Standard Deviation')
        ax6.set_title('Variation per Channel', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle(f'Image Batch Analysis - {batch_size} Frames', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def _create_bit_depth_graph(self, stats, batch_size):
        """
        Focused graph showing bit depth analysis in detail.
        """
        fig = Figure(facecolor='white')
        ax = fig.add_subplot(111)
        
        bit_depths = stats['effective_bit_depth']
        ax.plot(range(batch_size), bit_depths, linewidth=3, 
               marker='o', markersize=8, color='#2E86AB', 
               markerfacecolor='white', markeredgewidth=2)
        
        # Add reference lines for common bit depths
        ax.axhline(y=8, color='red', linestyle='--', alpha=0.5, label='8-bit reference')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10-bit reference')
        ax.axhline(y=16, color='green', linestyle='--', alpha=0.5, label='16-bit reference')
        
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Effective Bit Depth', fontsize=12)
        ax.set_title(f'Bit Depth Analysis - {batch_size} Frames', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        avg_depth = np.mean(bit_depths)
        min_depth = np.min(bit_depths)
        max_depth = np.max(bit_depths)
        textstr = f'Average: {avg_depth:.2f} bits\nMin: {min_depth:.2f} bits\nMax: {max_depth:.2f} bits'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig
    
    def _create_color_distribution_graph(self, stats, batch_size):
        """
        Show color channel distributions across the batch.
        """
        fig = Figure(facecolor='white')
        
        # Create subplots for each channel
        n_channels = len(stats['unique_values_per_channel'][0])
        
        for ch in range(n_channels):
            ax = fig.add_subplot(n_channels, 1, ch + 1)
            
            # Extract data for this channel
            channel_unique = [frame[ch] for frame in stats['unique_values_per_channel']]
            channel_mean = [frame[ch] for frame in stats['mean_values']]
            
            # Plot unique values as bars
            ax.bar(range(batch_size), channel_unique, alpha=0.6, 
                  color=['#F18F01', '#C73E1D', '#6A994E'][ch] if ch < 3 else '#888888',
                  label='Unique Values')
            
            # Overlay mean values as a line
            ax2 = ax.twinx()
            ax2.plot(range(batch_size), channel_mean, 
                    color='black', linewidth=2, marker='o', markersize=4,
                    label='Mean Intensity')
            
            channel_name = ['Red', 'Green', 'Blue'][ch] if ch < 3 else f'Channel {ch}'
            ax.set_ylabel('Unique Values', fontsize=10)
            ax2.set_ylabel('Mean Intensity', fontsize=10)
            ax.set_title(f'{channel_name} Channel Analysis', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if ch == n_channels - 1:
                ax.set_xlabel('Frame Number')
        
        fig.suptitle(f'Color Distribution Analysis - {batch_size} Frames', 
                    fontsize=14, fontweight='bold')
        
        return fig
    
    def _create_statistics_graph(self, stats, batch_size):
        """
        Statistical overview with key metrics.
        """
        fig = Figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Calculate summary statistics
        avg_bit_depth = np.mean(stats['effective_bit_depth'])
        std_bit_depth = np.std(stats['effective_bit_depth'])
        avg_entropy = np.mean(stats['color_entropy'])
        
        mean_vals_array = np.array(stats['mean_values'])
        avg_brightness = mean_vals_array.mean()
        
        unique_vals_array = np.array(stats['unique_values_per_channel'])
        
        # Create formatted text
        stats_text = f"""
        IMAGE BATCH STATISTICS REPORT
        ═══════════════════════════════════════
        
        Batch Size: {batch_size} frames
        
        BIT DEPTH ANALYSIS
        ─────────────────────────────────
        Average Effective Bit Depth: {avg_bit_depth:.2f} bits
        Bit Depth Variation (σ): {std_bit_depth:.2f} bits
        Min/Max Bit Depth: {min(stats['effective_bit_depth']):.2f} / {max(stats['effective_bit_depth']):.2f} bits
        
        COLOR ANALYSIS
        ─────────────────────────────────
        Average Color Entropy: {avg_entropy:.2f}
        Average Brightness: {avg_brightness:.3f} (0-1 scale)
        
        CHANNEL STATISTICS
        ─────────────────────────────────
        Average Unique Values per Channel:
          • Channel 0: {unique_vals_array[:, 0].mean():.0f} unique values
          • Channel 1: {unique_vals_array[:, 1].mean():.0f} unique values
          • Channel 2: {unique_vals_array[:, 2].mean():.0f} unique values
        
        QUALITY INDICATORS
        ─────────────────────────────────
        Bit Depth Consistency: {'Excellent' if std_bit_depth < 0.5 else 'Good' if std_bit_depth < 1.0 else 'Variable'}
        Color Diversity: {'High' if avg_entropy > 6 else 'Medium' if avg_entropy > 4 else 'Low'}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        return fig
    
    def _fig_to_tensor(self, fig):
        """
        Convert matplotlib figure to a tensor that ComfyUI can use.
        ComfyUI expects images as [B, H, W, C] tensors with float32 values in [0, 1].
        """
        # Render figure to RGB buffer
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get RGB buffer
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        
        # Convert to numpy array
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        
        # Drop alpha channel and convert to float [0, 1]
        img_array = img_array[:, :, :3].astype(np.float32) / 255.0
        
        # Add batch dimension [1, H, W, C]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return img_tensor
    
    def _create_text_summary(self, stats, batch_size):
        """
        Create a text summary of the analysis that can be used for logging
        or exported to other nodes.
        """
        summary_lines = [
            f"Analyzed {batch_size} frames",
            f"Average bit depth: {np.mean(stats['effective_bit_depth']):.2f} bits",
            f"Average entropy: {np.mean(stats['color_entropy']):.2f}",
            f"Bit depth range: {min(stats['effective_bit_depth']):.2f} - {max(stats['effective_bit_depth']):.2f}",
        ]
        
        return "\n".join(summary_lines)


# This is the required mapping for ComfyUI to recognize your node
NODE_CLASS_MAPPINGS = {
    "ImageBatchAnalyzer": ImageBatchAnalyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchAnalyzer": "Image Batch Analyzer"
}