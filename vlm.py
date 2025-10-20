#!/usr/bin/env python3

import os
import pandas as pd
from json_utils import parse_json, encode_image_to_base64, save_states_to_json, extract_json_from_string
import vlm_api
# import qwen_utils as vlm_api  # Adjust based on your vlm_api module

# VLM Configuration
HOSTS = {
    "gemini": vlm_api.gemini_request,
    # "openrouter": vlm_api.openrouter_request,
    # "qwen_local": vlm_api.vlm_request_qwen,  # Adjust based on your vlm_api module
    # "dummy": vlm_api.dummy_request,
}

# System prompts for semantic map assessment
SYSTEM_PROMPT = """You are an expert at analyzing images and semantic similarity maps. 
Your task is to assess whether a given object is present in an RGB image and whether 
a corresponding similarity map correctly highlights the object's location.

In the similarity maps provided:
- Yellow/bright areas indicate HIGH similarity (value close to 1)
- Purple/dark areas indicate LOW similarity (value close to 0)

You must respond in the following JSON format:
{
  "assessment": {
    "is_object_present": true/false,
    "is_map_correct": true/false,
    "reasoning": "Brief explanation for your decisions"
  }
}
"""


class VLMSemanticMapProcessor:
    def __init__(self, host, model, model_file_name):
        """
        Initialize the semantic map processor.
        
        Args:
            host: VLM host type ("gemini", "openrouter", "qwen_local", "dummy")
            model: Specific model to use
            model_file_name: Name for output files
        """
        self.host = host
        self.model = model
        self.model_file_name = model_file_name
        self.vlm_function = HOSTS[host]
        
        # Configuration
        self.base_dir = "E_grade_pics"
        self.queries = ['ball', 'door', 'floor', 'painting', 'sofa', 'table', 'wall', 'window']
        self.num_frames = 78
        
        # Setup output file
        self.output_dir = "semantic_map_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_csv = os.path.join(self.output_dir, f"{self.model_file_name}_assessment_results.csv")
        
        # Initialize CSV with headers
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = ['frame_index', 'query', 'object_present', 'map_correct', 'reasoning']
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.output_csv, index=False)
        print(f"📝 Initialized CSV file: {self.output_csv}")
    
    def _append_to_csv(self, result):
        """Append a single result to the CSV file"""
        df = pd.DataFrame([result])
        df.to_csv(self.output_csv, mode='a', header=False, index=False)
        print(f"💾 Saved result to CSV: Frame {result['frame_index']}, Query '{result['query']}'")
    
    ##########################################################################################################################################
    ########## PATHS TO IMAGE FILES ##########
    ##########################################################################################################################################
    def load_image_pair(self, query, frame_idx):
        """Load and encode RGB and semantic map images to base64"""
        query_dir = os.path.join(self.base_dir, query)
        scene = '47333473' # 47333473 # 40753679 # 

        rgb_path = os.path.join(query_dir, f"{scene}_{query}_{frame_idx}_rgb.png")
        voxel_path = os.path.join(query_dir, f"{scene}_{query}_{frame_idx}_rendered.png")
        
        # Check if files exist
        if not os.path.exists(rgb_path):
            return None, None, f"RGB image not found: {rgb_path}"
        if not os.path.exists(voxel_path):
            return None, None, f"Voxel map not found: {voxel_path}"
            
        try:
            rgb_b64 = encode_image_to_base64(rgb_path)
            voxel_b64 = encode_image_to_base64(voxel_path)
            return rgb_b64, voxel_b64, None
        except Exception as e:
            return None, None, f"Error loading images: {str(e)}"
    
    def build_messages(self, rgb_image, semantic_map, object_query):
        """Build messages for VLM API call based on model type"""
        user_prompt = f"""Analyze the two images provided.
The first image is an RGB photo of a scene.
The second image is a voxel similarity map for the object '{object_query}'. 
In this map, yellow indicates high similarity and purple indicates low similarity.

Your tasks are:
1. Look at the first image (the RGB photo). Is a '{object_query}' present in the scene?
2. If the object is present, look at the second image (the similarity map). Does the map correctly show high similarity (yellow areas) where the '{object_query}' is located in the RGB photo?

Provide your assessment in the requested JSON format."""
        
        if self.host == "gemini":
            return [{
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": rgb_image}},
                    {"inline_data": {"mime_type": "image/png", "data": semantic_map}}
                ]
            }]
        elif self.host == "qwen_local":
            return [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": f"data:image/png;base64,{rgb_image}"}]},
                {"role": "user", "content": [{"type": "image", "image": f"data:image/png;base64,{semantic_map}"}]}
            ]
        elif self.host == "openrouter":
            return [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb_image}", "detail": "high"}}
                ]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{semantic_map}", "detail": "high"}}
                ]}
            ]
        else:
            # Default format for dummy or other hosts
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": f"[RGB Image: base64 data]"},
                {"role": "user", "content": f"[Semantic Map: base64 data]"}
            ]
    
    def process_vlm_response(self, content, messages, query, frame_idx):
        """Process VLM response and extract assessment"""
        assessment_data, success = parse_json(content)
        
        if success and "assessment" in assessment_data:
            assessment = assessment_data["assessment"]
            result = {
                "frame_index": frame_idx,
                "query": query,
                "object_present": assessment.get("is_object_present", "N/A"),
                "map_correct": assessment.get("is_map_correct", "N/A"),
                "reasoning": assessment.get("reasoning", "N/A")
            }
            print(f"✓ Frame {frame_idx}: Object Present: {result['object_present']}, Map Correct: {result['map_correct']}")
            return result, True
        else:
            print("✗ Failed to parse JSON from VLM response, checking for embedded JSON...")
            assessment_data, success = extract_json_from_string(content)
            
            if success and "assessment" in assessment_data:
                assessment = assessment_data["assessment"]
                result = {
                    "frame_index": frame_idx,
                    "query": query,
                    "object_present": assessment.get("is_object_present", "N/A"),
                    "map_correct": assessment.get("is_map_correct", "N/A"),
                    "reasoning": assessment.get("reasoning", "N/A")
                }
                print(f"✓ Frame {frame_idx}: Object Present: {result['object_present']}, Map Correct: {result['map_correct']}")
                return result, True
            else:
                print("✗ Failed to extract JSON from VLM response. Reprompting...")
                return self.reprompt_vlm(content, messages, query, frame_idx)
    
    def reprompt_vlm(self, content, messages, query, frame_idx):
        """Attempt to reprompt VLM for valid JSON output"""
        print("Attempting reprompt...")
        try:
            # Build model-specific reprompt messages
            if self.host == "gemini":
                reprompt_messages = messages + [
                    {"role": "model", "parts": [{"text": content}]},
                    {"role": "user", "parts": [{"text": "The output given by you has format error, please output your result according to the given format."}]}
                ]
            elif self.host in ["qwen_local", "openrouter"]:
                reprompt_messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": [{"type": "text", "text": "The output given by you has format error, please output your result according to the given format."}]}
                ]
            else:
                reprompt_messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "The output given by you has format error, please output your result according to the given format."}
                ]

            content = self.vlm_function(reprompt_messages, subcategory=self.model)
            print(f"VLM Reprompt Response:\n{content}")

            assessment_data, success = parse_json(content)
            
            if success and "assessment" in assessment_data:
                assessment = assessment_data["assessment"]
                result = {
                    "frame_index": frame_idx,
                    "query": query,
                    "object_present": assessment.get("is_object_present", "N/A"),
                    "map_correct": assessment.get("is_map_correct", "N/A"),
                    "reasoning": assessment.get("reasoning", "N/A")
                }
                print(f"✓ Frame {frame_idx} (after reprompt): Object Present: {result['object_present']}, Map Correct: {result['map_correct']}")
                return result, True
            else:
                result = {
                    "frame_index": frame_idx,
                    "query": query,
                    "object_present": "N/A",
                    "map_correct": "N/A",
                    "reasoning": f"Failed to get valid JSON after reprompt"
                }
                print(f"✗ Failed to get valid JSON after reprompt for frame {frame_idx}")
                return result, False
        except Exception as e:
            result = {
                "frame_index": frame_idx,
                "query": query,
                "object_present": "N/A",
                "map_correct": "N/A",
                "reasoning": f"Reprompt failed: {str(e)}"
            }
            print(f"⚠ Reprompt failed for frame {frame_idx}: {e}")
            return result, False
    
    def process_all_queries(self):
        """Process all queries and frames"""
        print("\n" + "="*60)
        print("SEMANTIC MAP EVALUATION")
        print("="*60)
        print(f"Host: {self.host}")
        print(f"Model: {self.model}")
        print("="*60)
        
        total_processed = 0
        
        for query in self.queries:
            print(f"\n📋 Processing query: '{query}'")
            print("-" * 60)
            
            query_dir = os.path.join(self.base_dir, query)
            
            # Check if query directory exists
            if not os.path.exists(query_dir):
                print(f"  ⚠️  Warning: Directory not found: {query_dir}")
                continue
            
            # Process each frame
            for frame_idx in range(self.num_frames):
                print(f"\n🤖 Assessing frame {frame_idx} for '{query}'...")
                
                # Load images
                rgb_b64, voxel_b64, error = self.load_image_pair(query, frame_idx)
                
                if error:
                    result = {
                        "frame_index": frame_idx,
                        "query": query,
                        "object_present": "N/A",
                        "map_correct": "N/A",
                        "reasoning": error
                    }
                    print(f"  ⚠️  {error}")
                    self._append_to_csv(result)
                    total_processed += 1
                    continue
                
                # Build messages and call VLM
                try:
                    messages = self.build_messages(rgb_b64, voxel_b64, query)
                    content = self.vlm_function(messages, subcategory=self.model)
                    print(f"VLM Response:\n{content}")
                    
                    result, success = self.process_vlm_response(content, messages, query, frame_idx)
                    self._append_to_csv(result)
                    total_processed += 1
                    
                except Exception as e:
                    result = {
                        "frame_index": frame_idx,
                        "query": query,
                        "object_present": "N/A",
                        "map_correct": "N/A",
                        "reasoning": f"Error: {str(e)}"
                    }
                    print(f"  ❌ Error during assessment: {str(e)}")
                    self._append_to_csv(result)
                    total_processed += 1
        
        return total_processed
    
    def print_summary(self):
        """Print summary statistics from the CSV file"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        try:
            df = pd.read_csv(self.output_csv)
            
            print(f"✅ Results saved to: {self.output_csv}")
            print(f"\n📊 Summary:")
            print(f"  Total assessments: {len(df)}")
            
            valid_assessments = df[df['object_present'] != 'N/A']
            print(f"  Valid assessments: {len(valid_assessments)}")
            print(f"  Failed assessments: {len(df) - len(valid_assessments)}")
            
            if len(valid_assessments) > 0:
                print(f"\n  Object present (True): {sum(valid_assessments['object_present'] == True)}")
                print(f"  Object present (False): {sum(valid_assessments['object_present'] == False)}")
                print(f"  Map correct (True): {sum(valid_assessments['map_correct'] == True)}")
                print(f"  Map correct (False): {sum(valid_assessments['map_correct'] == False)}")
        except Exception as e:
            print(f"⚠️  Error reading CSV for summary: {e}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)


def main():
    """Main execution function"""
    
    # Configuration
    host = "gemini"  # Options: "gemini", "openrouter", "qwen_local", "dummy"
    model = "gemini-2.5-flash"  # Adjust based on your model
    model_file_name = "gemini25flash-data47333473"  # Name for output files
    
    print(f"Starting semantic map evaluation:")
    print(f"  Host: {host}")
    print(f"  Model: {model}")
    print(f"  Output name: {model_file_name}")
    
    # Create processor and run
    processor = VLMSemanticMapProcessor(host, model, model_file_name)
    total_processed = processor.process_all_queries()
    processor.print_summary()
    
    print(f"\n✓ All processing completed! Total assessments: {total_processed}")


if __name__ == "__main__":
    main()