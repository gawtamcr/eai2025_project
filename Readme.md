# 3D Semantic Object Mapping Lab

## Overview
This lab introduces 3D semantic mapping using state-of-the-art foundation vision models with RGB-D sensor data. You'll work with real-world indoor scene data from the ARKitScenes dataset, learning to bridge computer vision and 3D spatial understanding.

## What You Will Learn
Through this lab, you will get familiar with:

- **Foundation Vision Models**: Working with OwlV2 (open-vocabulary object detection), SAM (segmentation), and CLIP (vision-language understanding)
- **3D Computer Vision**: Converting 2D detections to 3D representations using camera geometry, depth data, and coordinate transformations
- **Sensor Fusion**: Combining RGB images, depth maps, and camera poses to build 3D scene understanding
- **Multi-View Processing**: Aggregating information across multiple viewpoints to create robust object representations
- **Open-Vocabulary Understanding**: Moving beyond fixed object categories to query scenes with arbitrary text prompts

The lab progresses from object detection to scene understanding, giving you experience with the full pipeline from raw sensor data to semantic 3D maps.

## Lab Structure
There are three levels of progressive difficulty: **Level E → Level C → Level A** (solve them in order).

- **Level E**: Open-vocabulary 3D object detection using OwlV2 
- **Level C**: Enhanced OwlV2 detection with SAM segmentation refinement
- **Level A**: Dense semantic mapping enabling text-based 3D queries

Each level builds on the previous one, demonstrating how multiple vision models can be combined for increasingly sophisticated 3D understanding.

**Important**: Each group member should understand the code and be able to explain the processing pipeline. The grade for Level C requires completing Level E, and Level A requires completing all three levels.

## Implementation Details

**Jupyter Notebooks**: Each level has its own notebook (`lab2_E.ipynb`, `lab2_C.ipynb`, `lab2_A.ipynb`).

**Code Tasks**: Look for TODO sections in the notebooks where you'll implement core algorithms. These are relatively short, self-contained tasks.

**Utilities**: The `lab_utils/` folder contains helper functions for data loading and visualization - you don't need to modify these, but feel free to explore them to understand the full system.

**Evaluation**: Levels E and C include automatic evaluation against ground truth with minimum IoU requirements. Level A focuses on functionality and understanding.

**References** (optional reading):
- OWLV2 (Google Deepmind): https://arxiv.org/pdf/2306.09683
- SAM (Meta AI): https://arxiv.org/pdf/2304.02643
- CLIP (OpenAI): https://arxiv.org/pdf/2103.00020

## Installation:

Each notebook contains installation commands in the first cells. Make sure to:

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv_lab2
   source venv_lab2/bin/activate  # or venv_lab2\Scripts\activate on Windows
   ```

2. Run the installation cells in each notebook

3. **Important**: After installing rerun visualization packages, close and reopen VSCode for the 3D viewer to work properly.

## Troubleshooting

**GPU Memory Issues**: If the kernel dies, reduce `max_frames` in the configuration or restart the kernel before running. (Sometimes closing and reopening vscode refreshes the cache and frees memory.)

**Visualization Issues**: If rerun 3D viewer doesn't appear:
- Close and reopen VSCode after package installation
- Allow third-party widget sources when prompted
- Manually add `jsdelivr.com` and `unpkg.com` in VSCode settings under "Jupyter: Widget Script Sources"

**Best Practice**: Restart the kernel before running the full pipeline to clear GPU memory.
