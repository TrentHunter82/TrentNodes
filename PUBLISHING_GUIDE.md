# 🚀 Publishing Guide for ComfyUI-TrentNodes

## ✅ What's Been Done

Your nodes have been successfully reorganized and merged into a single, registry-ready repository:

### Structure
```
ComfyUI-TrentNodes/
├── __init__.py                    # Main loader with auto-discovery + banner
├── pyproject.toml                 # Registry metadata (needs your Publisher ID)
├── requirements.txt               # All dependencies consolidated
├── README.md                      # Comprehensive documentation
├── LICENSE                        # MIT License
├── .gitignore                     # Python/ComfyUI ignores
├── .github/
│   └── workflows/
│       └── publish.yml           # Auto-publishing GitHub Action
├── nodes/                        # All your nodes in one place
│   ├── __init__.py
│   ├── Custom_Utility_Nodes.py
│   ├── enhanced_video_cutter.py
│   ├── video_folderanalyzer_node.py
│   ├── filename_extractor.py     # Previously loose file - now merged
│   ├── wan_vace/                 # WanVace nodes as subfolder
│   │   ├── __init__.py
│   │   └── wan_vace_keyframes.py
│   └── [16+ other node files]
├── js/
│   └── wan_vace_keyframes.js     # Custom JavaScript for WanVace UI
├── utils/
│   └── common.py
├── assets/
│   └── images/
└── example_workflows/
    └── hello_image_example.json
```

### Key Improvements
✅ **WanVace nodes integrated** as a subfolder under `nodes/wan_vace/`
✅ **JavaScript files** moved to `/js/` and served via `WEB_DIRECTORY` in main `__init__.py`
✅ **Loose files merged** - `video_folderanalyzer_node.py` and `filename_extractor.py` added to nodes/
✅ **Requirements consolidated** - opencv, numpy, pillow, matplotlib, colorama
✅ **Auto-discovery** - Your existing `__init__.py` logic preserved
✅ **Beautiful banner** - Kept your ASCII art and color system
✅ **Registry-ready** - pyproject.toml template created

## 📋 Next Steps to Publish

### 1. Create GitHub Repository
```bash
cd /path/to/ComfyUI-TrentNodes

# Initialize git
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Merged Trent Nodes for ComfyUI Registry"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/ComfyUI-TrentNodes.git
git branch -M main
git push -u origin main
```

### 2. Set Up ComfyUI Registry Account
1. Go to https://registry.comfy.org
2. Sign up with GitHub or Google
3. **Create a Publisher**:
   - Click "Create Publisher" 
   - Choose a unique Publisher ID (e.g., `trentfilms`, `trentvideo`, etc.)
   - ⚠️ **This cannot be changed later!**
   - Your ID appears after the `@` symbol on your profile

### 3. Update pyproject.toml
Edit `/pyproject.toml` and fill in:
```toml
[project.urls]
Repository = "https://github.com/YOUR_USERNAME/ComfyUI-TrentNodes"  # Your actual GitHub URL

[tool.comfy]
PublisherId = "your-publisher-id"  # From registry.comfy.org (step 2)
DisplayName = "Trent's Video & Scene Tools"
Icon = ""  # Optional: Add icon URL later
```

### 4. Create Registry API Key
1. On registry.comfy.org, go to your publisher page
2. Click "Create API Key"
3. Name it (e.g., "GitHub Actions Publishing")
4. **Copy and save it** - you can't see it again!

### 5. Add GitHub Secret
1. Go to your GitHub repo → Settings → Secrets and Variables → Actions
2. Click "New repository secret"
3. Name: `REGISTRY_ACCESS_TOKEN`
4. Value: Paste your API key from step 4
5. Click "Add secret"

### 6. First Publish

**Option A: Using Comfy CLI (Quick Test)**
```bash
# Make sure you're in ComfyUI environment
cd /path/to/ComfyUI-TrentNodes

# Publish
comfy node publish
# When prompted, paste your API key (right-click to paste on Windows)
```

**Option B: Using GitHub Actions (Recommended)**
```bash
# Just commit and push the pyproject.toml
git add pyproject.toml
git commit -m "Update publisher info for registry"
git push

# GitHub Actions will automatically publish!
```

### 7. Verify Publication
After publishing, check:
- https://registry.comfy.org/nodes
- Search for "Trent" or your publisher name
- Your nodes should appear!

## 🔄 Future Updates

To publish new versions:

1. **Make your changes** to any node files
2. **Update version** in `pyproject.toml`:
   ```toml
   version = "1.0.1"  # Increment according to semantic versioning
   ```
   - **PATCH** (1.0.0 → 1.0.1): Bug fixes
   - **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
   - **MAJOR** (1.0.0 → 2.0.0): Breaking changes (input/output changes)

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "v1.0.1: Fixed scene detection threshold bug"
   git push
   ```

4. **Auto-publish** - GitHub Actions handles the rest!

## 📊 Node Count
- **Total nodes**: 22 Python files
- **Categories**: Video Processing, Scene Detection, Keyframes, Utilities, Effects
- **Special features**: WanVace keyframe builder with custom JavaScript UI

## ⚠️ Important Notes

1. **Publisher ID is permanent** - Choose carefully!
2. **Name is taken from pyproject.toml** `name` field: `comfyui-trentnodes`
3. **Users install with**: `comfy node registry-install comfyui-trentnodes`
4. **JavaScript files work** because `WEB_DIRECTORY = "./js"` in main `__init__.py`
5. **All nodes auto-register** thanks to your discovery system

## 🐛 Testing Locally

Before publishing, test locally:

```bash
# Copy to ComfyUI custom_nodes
cp -r ComfyUI-TrentNodes /path/to/ComfyUI/custom_nodes/

# Restart ComfyUI and check:
# 1. Banner shows in console
# 2. All nodes load (check for errors)
# 3. WanVace node has dynamic inputs (JavaScript working)
# 4. No import errors
```

## 📞 Need Help?

- **Registry docs**: https://docs.comfy.org/registry/publishing
- **ComfyUI Discord**: https://discord.com/invite/comfyorg
- **Issues**: Report on your GitHub repo

---

## 🎉 You're Ready!

Everything is set up and ready to go. Just follow the steps above to get your nodes on the registry!

**Pro tip**: Do a test publish with version 0.1.0 first, verify everything works, then bump to 1.0.0 for the official release.
