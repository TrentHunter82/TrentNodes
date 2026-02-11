# Trent Nodes

Professional video processing, scene detection, and utility nodes for ComfyUI.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Nodes-orange)](https://github.com/comfyanonymous/ComfyUI)

---

### NEW: Grid Paste -- Bulk Node Duplication

> Select any nodes, groups, or subgraphs and instantly stamp out N copies
> arranged in a clean, non-overlapping grid. Two modes:
>
> | Shortcut | Mode | What it does |
> |---|---|---|
> | **Ctrl+Shift+G** | Grid Paste | Independent copies with internal links preserved |
> | **Ctrl+Shift+Alt+G** | Grid Paste Connected | Every copy's inputs wired back to the original upstream nodes (bulk Ctrl+Shift+V) |
>
> Also available from the **command palette**, **TrentNodes menu**, and the
> **selection toolbox** when items are selected. Works with nodes, groups,
> reroutes, and subgraphs. Fully undoable as a single Ctrl+Z.

---

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

### 📹 Trent/Video (6 nodes)

**Chop Cuts**
Accurate scene detection and video splitting. Automatically detects cuts, fades, and transitions using multi-metric analysis, then exports each scene as a separate MP4 file with a detailed report of cut locations and timestamps.

**Video Folder Analyzer**
Scans directories for video files and generates detailed reports including resolution, frame rate, codec, duration, and file size. Outputs as text, JSON, or markdown.

**Latest Video Last N Frames**
Extracts the final N frames from the most recently modified video in a specified directory. Useful for monitoring render outputs.

**Latest Video Final Frame**
Retrieves the last frame from the newest video file in a folder. Streamlines iterative video generation workflows.

**Cross Dissolve with Overlap**
Creates smooth frame transitions with configurable overlap duration. Blends adjacent frames for professional video effects.

**Batch Slowdown**
GPU-accelerated frame duplication to slow down image, mask, or latent batches. Supports multiple input modes: direct multiplier (2x, 3x, 1.5x), target frame count, or FPS conversion (24fps to 60fps). Features smart decimal distribution for non-integer slowdowns and optional speedup mode for sampling every Nth frame.

### 🎞️ Animation/Timing (2 nodes)

**Enhanced Animation Timing Processor**
Analyzes animation sequences to detect duplicate frames and replaces them with gray frames for video generation workflows. Features multiple similarity detection methods (hybrid, SSIM, histogram, perceptual), configurable preservation options for sequence first/last frames, and **keyframe alignment** that automatically inserts padding frames to ensure keyframes land on multiples of 4 (or any configurable multiple) for glitch-free video generation. Outputs include processed frames, duplicate mask, timing report, and removal indices for the companion Frame Remover node.

**Animation Frame Remover**
Removes padding frames inserted by the Enhanced Animation Timing Processor. Connect the `removal_indices` output from the processor to automatically strip the temporary padding frames after video generation, returning to the original frame count while preserving the generated content.

### 🖼️ Trent/Image (6 nodes)

**Image+Text Grid**
Creates a grid layout of images with text captions below each. Features auto-grid layout (set images_per_row to 0) that picks optimal columns via ceil(sqrt(n)), aspect-aware cell sizing based on median batch aspect ratio, and automatic centering of the last row when it has fewer images. Configure grid layout with images per row, image size, caption height, font size, padding, and background color. Note: when receiving images from a list-based node (e.g. StringListCowboy), use an ImageListToImageBatch node upstream to collect all images into a single batch before the grid. Perfect for contact sheets, comparison grids, or captioned image galleries.

**Align Stylized Frame**
Aligns AI-stylized images back to their original subject position with pixel-perfect precision. Uses BiRefNet (BEN2) for high-quality subject segmentation, SD 1.5 inpainting for clean plate background generation, and area-based scaling with centroid positioning for accurate subject placement. Eliminates ghosting artifacts when compositing stylized subjects onto original backgrounds.

**Cherry Pick Frames**
Flexible frame selector with multiple modes for extracting specific frames from image batches. Supports first N frames, last N frames, specific indices (comma-separated like "0,5,10,75"), or every Nth frame. Dynamic outputs adjust based on your selection. Perfect for grabbing keyframes, endpoints, or evenly-spaced samples from video batches.

**Bevel/Emboss Effect**
Applies depth and dimensionality to images through configurable bevel and emboss filters. Includes adjustable angle, depth, and smoothing parameters.

**Image Batch Analyzer**
Comprehensive statistical analysis of image batches. Generates histograms, color distribution charts, and detailed reports on brightness, contrast, and color composition.

**Multi-Batch Combine**
Concatenates multiple image batches into a single output batch. Accepts up to 8 optional inputs - unconnected inputs are simply skipped. Handles dimension mismatches automatically with configurable resize modes: largest (resize all to max dimensions), first (match first batch), or custom (specify target width/height). GPU-accelerated resizing via bilinear, nearest, bicubic, or area interpolation.

### 🔧 Trent/Utilities (10 nodes)

**Smart File Transfer (Auto-Rename)**
Intelligent file management with automatic conflict resolution, checksums, and organized directory structures. Safely transfers files with duplicate detection.

**Custom Filename Generator**
Creates structured filenames using templates with support for timestamps, counters, and metadata variables. Ensures consistent file naming across workflows.

**Filename Extractor**
Parses filenames to extract embedded metadata, timestamps, and structured information. Converts filenames into usable workflow data.

**JSON Multi-Line Summary**
Converts complex JSON data into human-readable multi-line summaries. Formats nested structures for display and logging.

**JSON Extractor**
Extracts specific values from JSON objects using path notation. Simplifies working with structured data in workflows.

**Number Counter**
Generates sequential numbers with configurable start, step, and padding. Essential for batch processing and frame numbering.

**Text File Line Loader**
Loads individual lines from text files by index. Useful for iterating through prompt lists or configuration files.

**File List**
Lists files in a directory with filtering options. Returns file paths for batch processing workflows.

**Create Text File**
Creates text files with custom content. Specify a file path and content to write. Automatically adds .txt extension if none provided. Creates parent directories as needed.

**Wan2.1 Frame Adjuster**
Adjusts frame amount to always satisfy Wan 4x+1 requirements by adding gray frames to the end of a batch; use a Get Frame Range from Batch node before combining video with the original amount of frames for less headaches when using Wan.

### 🎭 Trent/Masks (4 nodes)

**Latent Aligned Mask**
Creates precision masks aligned to latent space dimensions. Ensures proper mask scaling for latent-based video and image processing.

**Latent Aligned Mask (Advanced)**
Extended version with additional parameters for fine-tuned mask generation including feathering, inversion, and composite operations.

**Latent Aligned Mask (Simple)**
Streamlined mask creation with minimal inputs for quick latent-aligned masks in simple workflows.

**Latent Aligned Mask (Wan)**
Specialized variant optimized for Wan video model requirements with automatic 4x+1 frame alignment.

### 🎬 Trent/Keyframes (1 node)

**Wan Vace Keyframe Builder**
Dynamic keyframe sequencing for Wan Vace video generation. Features interactive UI with drag-and-drop image inputs, frame-accurate positioning, automatic resizing, and synchronized mask generation. Supports up to 256 frames with customizable filler frames.

### 📝 Trent/Text (2 nodes)

**Auto Style Dataset**
Generates 35 prompt strings for synthetic dataset creation. Reads prompts from an external config file and applies optional prepend/append text to each output. Perfect for batch generation of training data with consistent formatting.

**String List Cowboy**
Lassos strings together into a list with optional prefix/suffix branding. Works like Impact Pack's MakeAnyList but specialized for strings - connect any inputs and they get collected into a string list. Each string gets the prefix prepended and suffix appended. Dynamic inputs expand as you connect more values.

### 🔀 Trent/Flow (2 nodes)

**This, That, or The Other**
Parallel gating node with 3 independent input/output channels. Each input passes to its corresponding output ONLY if truthy (non-None, non-zero, non-empty). Falsy inputs block their downstream path via ExecutionBlocker. Uses lazy evaluation to avoid evaluating inputs until needed. Dynamic inputs expand as you connect (1-3 slots).

**First Valid**
Fallback chain node that outputs the FIRST truthy value. Checks inputs in priority order: first → second → third. Returns the first truthy value found, or blocks if all are falsy. Uses lazy evaluation to skip evaluating later inputs once a truthy value is found. Perfect for providing fallback values like default images, prompts, or parameters.

### 🧪 TrentNodes/Testing (1 node)

**LoRA Test Prompt Generator**
Generates 10 test prompts specifically designed to validate different types of LoRA models. Supports four LoRA categories:
- **subject_person**: Portrait/character LoRAs with varied lighting, poses, and environments
- **style**: Artistic style LoRAs across diverse subjects and scenes
- **product**: Object/product LoRAs with studio and lifestyle contexts
- **vehicle**: Car/vehicle LoRAs covering angles, lighting, and motion

Outputs 10 individual prompt strings plus a combined `all_prompts` output for easy batch processing. Includes optional quality suffix to append tags like "8k, detailed" to all prompts.

### 👁️ Trent/VLM (2 nodes)

**VidScribe MiniCPM Beta**
GPU-accelerated vision-language model for describing images and video frames using MiniCPM-V 4.5. Features:
- int4 quantization (~6-8GB VRAM)
- Smart frame sampling (auto-selects ~32 frames from longer videos)
- Auto-unload after 60s idle to free VRAM
- System prompt presets (default, detailed, concise, narrator, technical, accessible, creative)
- Three modes: single image, multi-image comparison, video frame sequence with temporal understanding
- Deep thinking mode for more thorough analysis

**Unload MiniCPM**
Manually unload MiniCPM model to immediately free VRAM. Connect any output to trigger. Useful when you need GPU memory for other operations without waiting for the 60-second auto-unload timeout.

### Canvas Tools (Frontend Extensions)

These are canvas-level tools that operate on the ComfyUI graph directly -- no Python backend nodes required. They register as commands with keybindings, menu entries, and selection toolbox buttons.

**Grid Paste**
Duplicate any selection of nodes, groups, reroutes, or subgraphs into an automatically-arranged grid. Select your nodes, hit **Ctrl+Shift+G**, type how many copies you want, and they appear in a clean grid at your cursor position. The grid auto-calculates a roughly-square layout (e.g. 9 copies = 3x3) with 50px padding, sizes each cell to the bounding box of your selection, and preserves all internal connections between copied nodes. Widget values, node colors, group rectangles -- everything comes along for the ride.

**Grid Paste Connected**
Same grid layout, but every copy's external inputs are wired back to the original source nodes -- the same behavior as Ctrl+Shift+V but applied in bulk. Hit **Ctrl+Shift+Alt+G** to use this mode. Perfect for scenarios like pasting 6 KSamplers that all need to connect to the same checkpoint loader, or duplicating a ControlNet processing chain where every copy should read from the same source image.

Both modes wrap the entire operation in a single undo transaction, so one Ctrl+Z reverts everything. Maximum 100 copies per operation.

### 🎤 Trent/LipSync (11 nodes)

Complete lip sync pipeline for non-human character animation. Converts audio to mouth shapes and composites them onto tracked positions in video frames.

**Audio To Phonemes**
Extracts phonemes from audio using Vosk speech recognition. Returns timestamped phoneme data for mouth shape mapping. Automatically downloads the required Vosk model on first use.

**Phoneme To Mouth Shapes**
Converts phoneme data to a sequence of mouth shape indices (A-H + X for silence). Maps speech sounds to the 9 standard mouth positions used in animation.

**Mouth Shape Loader**
Loads 9 mouth shape images from a folder. Expects files named A.png through H.png plus X.png (silence). Validates all shapes are present and correctly sized.

**Mouth Shape Preview**
Previews mouth shapes with their corresponding phoneme labels. Useful for verifying mouth shape assets before use.

**Mouth Shape Compositor**
Basic compositor that places mouth shapes on frames at a fixed position. Use for static characters or simple animations.

**Mouth Shape Compositor (Tracked)**
Advanced compositor with tracking support. Places mouth shapes at positions determined by either:
- **Point tracking**: Use tracked (x,y) coordinates from Point Tracker
- **Mask tracking**: Use per-frame masks from SAM3 to find mouth centroids

Features BiRefNet background removal, scaling, offset adjustment, and optional RGBA output for further compositing.

**Creature Lip Sync**
All-in-one lip sync node combining audio analysis, mouth shape selection, and compositing in a single streamlined node. Ideal for quick character animation setups.

**Point Tracker**
Robust point tracking using pyramidal Lucas-Kanade optical flow. Click a point on frame 1 and track it through the entire video. Features:
- Sub-pixel accuracy with Scharr gradients
- Multi-stage recovery (adaptive template, original template, full-frame search)
- Periodic drift validation against original template
- GPU-accelerated template matching for large search areas
- Configurable window size up to 1025px for full-frame tracking

**Point Preview**
Click-to-pick interface for selecting the initial tracking point. Click anywhere on the image to set coordinates, which pass directly to Point Tracker.

**Points To Masks**
Converts point sequences to gaussian masks for use with mask-based compositing.

**Remove Mouth Background**
Standalone background removal using BiRefNet or color keying. Returns mouth shapes with alpha channel for custom compositing workflows.

#### LipSync Workflow

1. **Audio To Phonemes** - Extract speech from audio
2. **Phoneme To Mouth Shapes** - Convert to mouth indices
3. **Mouth Shape Loader** - Load your 9 mouth images
4. **Point Preview** - Click to select tracking point
5. **Point Tracker** - Track the point through video
6. **Mouth Shape Compositor (Tracked)** - Composite mouths onto frames

## Requirements

- ComfyUI (latest version recommended)
- Python 3.10+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0
- colorama >= 0.4.6
- vosk >= 0.3.45 (for lip sync speech recognition)
- transformers >= 4.40.0 (for BiRefNet and MiniCPM-V)
- accelerate (for MiniCPM-V model loading)

## Features

✅ **47 professional nodes** for video, image, VLM, flow control, and lip sync workflows
✅ **Canvas tools** - Grid Paste for bulk node duplication with auto-layout
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
