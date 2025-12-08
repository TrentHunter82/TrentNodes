# Trent Nodes

Professional video processing, scene detection, and utility nodes for ComfyUI.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-orange)](https://github.com/comfyanonymous/ComfyUI)

## Installation

### Via ComfyUI Manager (Recommended)
Search for "Trent Nodes" in ComfyUI Manager and click Install.

### Via Comfy CLI
```bash
comfy node registry-install trentnodes
```

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TrentHunter82/TrentNodes.git
cd TrentNodes
pip install -r requirements.txt
```

## Nodes

All nodes are organized under the `Trent/` category for easy navigation.

### 📹 Trent/Video (7 nodes)

**Enhanced Video Cutter**  
Advanced scene detection with adaptive thresholding and motion analysis. Exports scenes as individual MP4 files with clean naming and comprehensive metadata tracking.

**Ultimate Scene Detect**  
High-precision scene boundary detection using configurable threshold algorithms. Identifies cuts, fades, and transitions in video sequences.

**Video Folder Analyzer**  
Scans directories for video files and generates detailed reports including resolution, frame rate, codec, duration, and file size. Outputs as text, JSON, or markdown.

**Latest Video Last N Frames**  
Extracts the final N frames from the most recently modified video in a specified directory. Useful for monitoring render outputs.

**Latest Video Final Frame**  
Retrieves the last frame from the newest video file in a folder. Streamlines iterative video generation workflows.

**Cross Dissolve with Overlap**  
Creates smooth frame transitions with configurable overlap duration. Blends adjacent frames for professional video effects.

**Enhanced Animation Timing Processor**  
Analyzes animation sequences to detect duplicate frames, timing patterns, and frame holds. Optimizes animation frame sequences.

### 🖼️ Trent/Image (3 nodes)

**Cherry Pick Frames**  
Flexible frame selector with multiple modes for extracting specific frames from image batches. Supports first N frames, last N frames, specific indices (comma-separated like "0,5,10,75"), or every Nth frame. Dynamic outputs adjust based on your selection. Perfect for grabbing keyframes, endpoints, or evenly-spaced samples from video batches.

**Bevel/Emboss Effect**  
Applies depth and dimensionality to images through configurable bevel and emboss filters. Includes adjustable angle, depth, and smoothing parameters.

**Image Batch Analyzer**  
Comprehensive statistical analysis of image batches. Generates histograms, color distribution charts, and detailed reports on brightness, contrast, and color composition.

### 🔧 Trent/Utilities (7 nodes)

**Smart File Transfer (Auto-Rename)**  
Intelligent file management with automatic conflict resolution, checksums, and organized directory structures. Safely transfers files with duplicate detection.

**Custom Filename Generator**  
Creates structured filenames using templates with support for timestamps, counters, and metadata variables. Ensures consistent file naming across workflows.

**Filename Extractor**  
Parses filenames to extract embedded metadata, timestamps, and structured information. Converts filenames into usable workflow data.

**JSON → Multi-Line Summary**  
Converts complex JSON data into human-readable multi-line summaries. Formats nested structures for display and logging.

**JSON Extractor**  
Extracts specific values from JSON objects using path notation. Simplifies working with structured data in workflows.

**Number Counter**  
Generates sequential numbers with configurable start, step, and padding. Essential for batch processing and frame numbering.

**Wan2.1 Frame Adjuster**  
Specialized utility for adjusting frame timing and synchronization in Wan 2.1 video generation workflows.

### 🎭 Trent/Masks (1 node)

**Latent Aligned Mask**  
Creates precision masks aligned to latent space dimensions. Ensures proper mask scaling for latent-based video and image processing.

### 🎬 Trent/Keyframes (1 node)

**Wan Vace Keyframe Builder**  
Dynamic keyframe sequencing for Wan Vace video generation. Features interactive UI with drag-and-drop image inputs, frame-accurate positioning, automatic resizing, and synchronized mask generation. Supports up to 256 frames with customizable filler frames.

## Requirements

- ComfyUI (latest version recommended)
- Python 3.10+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0
- colorama >= 0.4.6

## Features

✅ **24 professional nodes** for video and image workflows  
✅ **Organized categories** - all nodes under `Trent/` namespace  
✅ **Auto-discovery** - drop nodes in `nodes/` folder and restart  
✅ **Colorful startup banner** with load validation  
✅ **Comprehensive error checking** on initialization  
✅ **Registry published** - semantic versioning support  

## Development

```bash
# Clone the repository
git clone https://github.com/TrentHunter82/TrentNodes.git
cd TrentNodes

# Install dependencies
pip install -r requirements.txt

# Add new nodes
# Just drop .py files in nodes/ folder - they auto-register!
```

## Contributing

Pull requests welcome! Please:
- Follow existing code style
- Add docstrings to new nodes
- Test thoroughly before submitting
- Update this README with new nodes

## Support

- **Issues**: [GitHub Issues](https://github.com/TrentHunter82/TrentNodes/issues)
- **Registry**: [Comfy Registry](https://registry.comfy.org/publishers/flippingsigmas)
- **ComfyUI Discord**: [Join Server](https://discord.com/invite/comfyorg)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Trent** - [Trent Films](https://github.com/TrentHunter82)

---

*Made with ❤️ for the ComfyUI community*
