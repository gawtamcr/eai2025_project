#
# =====================================================================================
# Full, Integrated Script for Semantic Map Generation and Vision Model Assessment
# =====================================================================================
#

#
# Part 1: Setup and Dependencies
# -------------------------------------------------------------------------------------
#
# !pip install -q -U google-genai torch torchvision transformers pillow numpy opencv-python open3d pydantic matplotlib

import os
import google.generativeai as genai
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from typing import Dict
import torch

# (Assuming all your utility scripts like lab_utils are in the path)
# You would need to place all the necessary functions from your notebooks here.
# For brevity, I'm assuming they are defined above this script.
# These would include:
# - Config class
# - load_sam_model, load_clip_model
# - get_frame_list, load_camera_poses, validate_and_align_frame_data
# - run_full_pipeline (and all its dependencies like SemanticVoxelGrid, process_frame_level_a, etc.)
# - render_voxel_similarity_from_pose

#
# Part 2: Secure API Key Configuration
# -------------------------------------------------------------------------------------
#
### IMPROVEMENT ###
# Replaces the error-prone `open("gemini_api.txt")`.
# For local use, set an environment variable 'GEMINI_API_KEY'.
# For Google Colab, use the "Secrets" tab (key icon on the left) to store your key.
try:
    # Used in Google Colab
    from google.colab import userdata
    api_key = userdata.get('GEMINI_API_KEY')
except ImportError:
    # Used in local environments
    api_key = os.environ.get('GEMINI_API_KEY')

if not api_key:
    raise ValueError("Gemini API key not found. Please set it as an environment variable or a Colab secret.")

genai.configure(api_key=api_key)
client = genai.GenerativeModel('gemini-1.5-flash') # Using the client from the new SDK version

#
# Part 3: Gemini Assessment Function
# -------------------------------------------------------------------------------------
#

# Pydantic model for structured output from Gemini
class AssessmentResult(BaseModel):
    """Defines the structure for the model's assessment."""
    is_object_present: bool = Field(description="True if the object is in the RGB image, otherwise False.")
    is_map_correct: bool = Field(description="True if the similarity map correctly highlights the object's position.")
    reasoning: str = Field(description="A brief explanation for the decisions.")

# The assessment function, now modified to take two separate images
def assess_semantic_map(rgb_image_path: str, semantic_map_path: str, object_query: str) -> AssessmentResult:
    """
    Asks the Gemini model to assess if the semantic map corresponds to the RGB image.

    ### IMPROVEMENT ###
    This function now accepts two separate image paths and sends both to the model.
    This provides cleaner input and should yield better results.
    """
    print(f"🤖 Assessing query: '{object_query}' for {os.path.basename(rgb_image_path)}...")

    # Load images
    rgb_image = Image.open(rgb_image_path)
    semantic_map = Image.open(semantic_map_path)

    ### IMPROVEMENT ###
    # The prompt is now clearer, referring to "the first image (RGB)" and "the second image (similarity map)".
    prompt = f"""
    Analyze the two images provided.
    The first image is an RGB photo of a scene.
    The second image is a voxel similarity map for the object '{object_query}'. In this map, yellow indicates high similarity and purple indicates low similarity.

    Your tasks are:
    1.  Look at the first image (the RGB photo). Is a '{object_query}' present in the scene?
    2.  If the object is present, look at the second image (the similarity map). Does the map correctly show high similarity (yellow areas) where the '{object_query}' is located in the RGB photo?

    Provide your assessment in the requested JSON format.
    """

    # Generate content using the model with two images
    response = client.generate_content(
        [prompt, rgb_image, semantic_map],
        generation_config={"response_mime_type": "application/json",
                           "response_schema": AssessmentResult}
    )

    # Use the built-in Pydantic parsing
    return response.candidates[0].content.parts[0].json

#
# Part 4: Main Execution and Evaluation Loop
# -------------------------------------------------------------------------------------
#

def main_evaluation_loop():
    """
    Main loop to generate data and run assessments.
    """
    #
    # Step A: Run the Level A pipeline to get the semantic voxel grid
    # (This part is from your `lab2_A_generate.ipynb`)
    #
    print("="*60)
    print("STEP 1: Building Semantic Voxel Grid...")
    print("="*60)
    # This assumes 'config' and 'run_full_pipeline' are defined as in your notebook
    config = Config()
    results = run_full_pipeline(config)
    voxel_grid = results["voxel_grid"]
    clip_model = results["clip_model"]
    clip_processor = results["clip_processor"]
    device = results["device"]

    #
    # Step B: Set up paths and queries for the evaluation
    #
    print("\n" + "="*60)
    print("STEP 2: Preparing for Automated Assessment...")
    print("="*60)

    # Load frame data
    camera_poses = load_camera_poses(config.TRAJ_FILE_PATH)
    frames = get_frame_list(config.RGB_PATH, config.LEVEL_A_CONFIG['frame_skip'])
    aligned_frames = validate_and_align_frame_data(
        frames, camera_poses,
        config.RGB_PATH, config.DEPTH_PATH, config.INTRINSICS_PATH
    )

    queries = ['sofa', 'floor', 'wall', 'painting']
    output_dir = "assessment_data"
    os.makedirs(output_dir, exist_ok=True)

    #
    # Step C: Loop through frames and queries, generate images, and assess
    #
    print("\n" + "="*60)
    print("STEP 3: Running Generation and Assessment Loop...")
    print("="*60)
    total_frames_to_process = 2 # Let's just process 2 for this example

    for i in range(total_frames_to_process):
        frame_data = aligned_frames[i]
        rgb_path = frame_data['rgb_path']

        for q in queries:
            query_output_dir = os.path.join(output_dir, q)
            os.makedirs(query_output_dir, exist_ok=True)

            # Generate the semantic map for the current frame and query
            semantic_map_img = render_voxel_similarity_from_pose(
                voxel_grid,
                frame_data['camera_pose'],
                frame_data['camera_intrinsics'],
                clip_model,
                clip_processor,
                device,
                text_query=q
            )
            semantic_map_path = os.path.join(query_output_dir, f"map_{i}.png")
            Image.fromarray(semantic_map_img).save(semantic_map_path)

            # Call the Gemini assessment function
            try:
                assessment = assess_semantic_map(rgb_path, semantic_map_path, q)
                print(f"  - Object Present: {assessment.is_object_present}")
                print(f"  - Map Correct:    {assessment.is_map_correct}")
                print(f"  - Reasoning:      {assessment.reasoning}\n")
            except Exception as e:
                print(f"  - ❗️ Error during assessment for query '{q}': {e}\n")

#
# Run the main loop if the script is executed
#
if __name__ == '__main__':
    # You would need to copy all the function/class definitions from your notebooks
    # into this script for this to be a standalone file.
    main_evaluation_loop()
    print("To run the full loop, uncomment 'main_evaluation_loop()' and ensure all helper functions are defined.")