#!/bin/bash
# Script to organize ALL nodes under Trent/ namespace

echo "Organizing all nodes under Trent/ namespace..."

# Video Processing & Analysis
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Video"/' nodes/enhanced_video_cutter.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Video"/' nodes/latest_video_last_frames_node.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Video"/' nodes/LatestVideoFinalFramefromDirNode..py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Video"/' nodes/improved_animation_processor.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Video"/' nodes/crossdissolveoverlap.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Video"/' nodes/ultimate_scene_detect.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Video"/' nodes/video_folderanalyzer_node.py

# Image Processing & Effects
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Image"/' nodes/comfy_bevel_emboss.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Image"/' nodes/comfyui_image_analyzer.py

# Utilities & Tools
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Utilities"/' nodes/comfyui_filename_node.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Utilities"/' nodes/JSONSummary.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Utilities"/' nodes/wanmagic.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Utilities"/' nodes/filename_extractor.py
sed -i 's/CATEGORY = .*/CATEGORY = "Trent\/Utilities"/' nodes/json_extractor.py

# Smart File Transfer has multiple CATEGORY lines, update all
sed -i 's/    CATEGORY = "Trent Nodes"/    CATEGORY = "Trent\/Utilities"/' nodes/smart_file_transfer.py
sed -i 's/    CATEGORY = "file_operations"/    CATEGORY = "Trent\/Utilities"/' nodes/smart_file_transfer.py

# Custom_Utility_Nodes.py has TWO classes, update both
sed -i 's/    CATEGORY = "Custom_Utility_Nodes"/    CATEGORY = "Trent\/Utilities"/' nodes/Custom_Utility_Nodes.py
sed -i 's/    CATEGORY = "Trent Nodes"/    CATEGORY = "Trent\/Utilities"/' nodes/Custom_Utility_Nodes.py

# Masks & Latents - update all instances
sed -i 's/    CATEGORY = "mask\/Wan"/    CATEGORY = "Trent\/Masks"/' nodes/latent_aligned_mask_node.py
sed -i 's/    CATEGORY = "mask\/processing"/    CATEGORY = "Trent\/Masks"/' nodes/latent_aligned_mask_node.py

# WanVace Keyframes
sed -i 's/CATEGORY = "Wan\/Vace"/CATEGORY = "Trent\/Keyframes"/' nodes/wan_vace/wan_vace_keyframes.py

echo ""
echo "✅ All nodes reorganized!"
