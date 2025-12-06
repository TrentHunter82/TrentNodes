# ComfyUI-TrentNodes

A comprehensive collection of custom ComfyUI nodes for advanced video processing, keyframe management, scene detection, and video analysis workflows.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-orange)](https://github.com/comfyanonymous/ComfyUI)

## 🎯 Features

### Video Processing & Analysis
- **Enhanced Video Cutter**: Advanced scene detection with adaptive thresholding and motion analysis
- **Video Folder Analyzer**: Comprehensive analysis of video files with detailed metadata reporting
- **Latest Video Frame Extractor**: Extract final frames from directories of videos
- **Smart File Transfer**: Intelligent file management and organization tools

### Keyframe Management
- **Wan Vace Keyframe Builder**: Dynamic keyframe sequencing for Wan Vace video generation
  - Dynamically add/remove image inputs
  - Frame-accurate positioning
  - Automatic mask generation
  - Interactive UI with custom JavaScript controls

### Scene Detection & Analysis
- **Ultimate Scene Detect**: Advanced scene detection with configurable parameters
- **Animation Frame Analyzer**: Analyze animation sequences for duplicates and patterns
- **Cross Dissolve Overlap**: Sophisticated frame blending and transitions

### Utility Nodes
- **Custom Utility Nodes**: Collection of helpful workflow utilities
- **JSON Tools**: JSON extraction and summary nodes
- **Filename Extractors**: Parse and extract information from filenames
- **Image Analyzer**: Comprehensive image analysis with statistical reports
- **Latent Aligned Mask**: Precision masking tools for latent space operations

### Effects
- **Bevel & Emboss**: Advanced image effects for creating depth and dimension

## 📦 Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "Trent Nodes"
3. Click Install
4. Restart ComfyUI

### Via Comfy CLI
```bash
comfy node registry-install comfyui-trentnodes
```

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-TrentNodes.git
cd ComfyUI-TrentNodes
pip install -r requirements.txt
```

## 🚀 Quick Start

After installation, you'll find Trent Nodes organized in the following categories:

- **Trent Tools/** - Main utility and processing nodes
- **Trent/Video/** - Video processing and analysis
- **Wan/Vace/** - Wan Vace keyframe tools
- **Trent/Scene/** - Scene detection and analysis

### Example: Scene Detection Workflow
1. Load your video frames as an IMAGE batch
2. Add **Enhanced Video Cutter** node
3. Configure threshold and minimum scene length
4. Get individual scene videos with clean filenames

### Example: Keyframe Sequencing for Wan Vace
1. Add **Wan Vace Keyframe Builder** node
2. Set frame count (e.g., 16 frames)
3. Connect images to dynamic inputs
4. Position each image using frame sliders
5. Output synchronized image batch and masks

## 🛠️ Requirements

- ComfyUI (latest version recommended)
- Python 3.10+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0

## 📖 Node Documentation

### Enhanced Video Cutter
Advanced scene detection with adaptive thresholding:
- **Inputs**: IMAGE batch, output folder, FPS, threshold
- **Outputs**: Video paths, metadata JSON
- **Features**: Motion analysis, frame-accurate cutting, clean naming

### Wan Vace Keyframe Builder
Dynamic keyframe positioning for video generation:
- **Inputs**: Frame count, images (dynamic), frame positions
- **Outputs**: Image batch, mask batch
- **Features**: Auto-resizing, filler frame generation, JavaScript UI

### Video Folder Analyzer
Comprehensive video file analysis:
- **Inputs**: Folder path, include subfolders, output format
- **Outputs**: Detailed report (text/JSON/markdown), statistics
- **Features**: Multiple format support, recursive scanning

## 🎨 Categories

All nodes are organized with the `Trent` prefix for easy discovery:
- `Trent Tools/` - Core utilities
- `Trent/Video/` - Video processing
- `Trent/Scene/` - Scene analysis
- `Wan/Vace/` - Keyframe tools

## 🐛 Known Issues

- Some nodes require ffmpeg for video encoding
- Large video batches may require significant VRAM

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

**Trent** - [Flipping Sigmas](https://github.com/TrentHunter82)

## 🙏 Acknowledgments

- ComfyUI team for the amazing framework
- ComfyUI community for inspiration and feedback
- Wan Vace team for their excellent video generation models

## 📊 Version History

- **1.0.0** - Initial release
  - Core video processing nodes
  - Scene detection tools
  - Wan Vace keyframe builder
  - Comprehensive utility nodes

## 🔗 Links

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [Report Issues](https://github.com/yourusername/ComfyUI-TrentNodes/issues)

---

Made with ❤️ by Trent for the ComfyUI community
