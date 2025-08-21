# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess waterbird dataset to parquet format for VERL training.
This script converts waterbird classification data to the format expected by VERL.
"""

import argparse
from code import interact
import os
import json
import random
from typing import List, Dict, Any

import datasets
from verl.utils.hdfs_io import copy, makedirs


def _read_image_bytes(image_path: str) -> bytes:
    """Read an image file and return raw bytes.

    Returns empty bytes (b"") on failure.
    """
    try:
        if not image_path or not os.path.isfile(image_path):
            return b""
        with open(image_path, "rb") as f:
            data = f.read()
        return data
    except Exception:
        return b""


def create_waterbird_dataset_from_json(
    json_file: str,
    image_dir: str,
    split_mapping: Dict[int, str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create waterbird dataset from JSON file.
    
    Args:
        json_file: Path to JSON file containing dataset information
        image_dir: Directory containing image files
        split_mapping: Mapping from split numbers to split names
    
    Returns:
        Dictionary with train, validation, and test datasets
    """
    if split_mapping is None:
        split_mapping = {0: "train", 1: "validation", 2: "test"}
    
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Define class names
    class_names = ["landbird", "waterbird"]
    
    # Define prompts based on prompt.xml
    system_prompt = """**Role and Goal:**
You are an expert AI assistant specializing in ornithological image analysis. Your primary goal is to accurately classify a bird in an image as either a **Waterbird** or a **Landbird** through a precise, two-step process.

**Process Overview:**
The task will be completed in two distinct steps. You must strictly follow the instructions for each step.
1.  **Step 1: Bounding Box Detection:** First, you will be asked to locate the primary bird in the image and provide its tight bounding box coordinates.
2.  **Step 2: Classification:** Second, based on the original image and a cropped version of the bird, you will perform the final classification.

**Category Definitions:**
* **Waterbirds** include, but are not limited to, the following species: albatross, auklet, cormorant, frigatebird, fulmar, gull, jaeger, kittiwake, pelican, puffin, tern, gadwall, grebe, mallard, merganser, guillemot, or Pacific loon.
* **Landbirds** are all other bird species that are not categorized as Waterbirds.

**Output Format Constraints:**
You must adhere to the specific output format requested in each step.
* Step 1 Format: <box>[x1,y1,x2,y2]</box>

Begin with Step 1 when prompted.
"""
    
    user_prompt = """
Step 1: Analyze the following image. Your task is to generate a tight bounding box that covers only the bird with minimal background. Provide the coordinates in the format <box>[x1,y1,x2,y2]</box>.

<image>"""
    
    # Initialize datasets
    datasets_dict = {split_name: [] for split_name in split_mapping.values()}
    
    for item in data:
        # Extract information from JSON item
        image_path = item.get("img_filename", "")
        y = item.get("y", 0)  # 0 for landbird, 1 for waterbird
        split = item.get("split", 0)
        idx=item.get("img_id",0)
        # Map split number to split name
        split_name = split_mapping.get(split, "train")
        
        # Get class name
        class_name = class_names[y]
        
        # Create full image path
        full_image_path = os.path.join(image_dir, image_path) if image_path else ""
        # Prepare images field with raw bytes and relative path
        if full_image_path:
            image_entry = {
                "bytes": _read_image_bytes(full_image_path),
                "path": image_path,
            }
            images_field = [image_entry]
        else:
            images_field = []
        # print(full_image_path)
        # Create sample
        sample = {
            "data_source": "waterbird",
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": user_prompt
                }
            ],
            "images": images_field,
            "ability": "image_classification",
            "reward_model": {
                "style": "rule", 
                "ground_truth": class_name
            },
            "extra_info": {
                # "split": split_name,
                "idx":idx,
                "split": split,
                # "class_idx": y,
                "class_name": class_name,
                # "image_path": image_path,
                "image_path": full_image_path,
                # "original_data": item
                "interaction_kwargs":
                    {
                        "name":"waterbird",
                        "img_path":full_image_path,
                        "ground_truth":class_name
                    }
            },
        }
        
        datasets_dict[split_name].append(sample)
    
    return datasets_dict


def shuffle_dataset(dataset: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """
    Randomly shuffle the dataset.
    
    Args:
        dataset: List of data samples
        seed: Random seed for reproducibility
    
    Returns:
        Shuffled dataset
    """
    random.seed(seed)
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    return shuffled_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess waterbird dataset for VERL")
    parser.add_argument("--json_file", required=True,
                       help="Path to JSON file containing dataset information (e.g., data_waterbird.json)")
    parser.add_argument("--image_dir", required=True,
                       help="Directory containing image files")
    parser.add_argument("--output_dir", default="/data/wangnn/repos/verl/debiasing/data",
                       help="Output directory for processed data")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for output")
    parser.add_argument("--shuffle", action="store_true", default=True,
                       help="Shuffle the dataset (default: True)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for shuffling")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process waterbird dataset from JSON file
    print(f"Processing waterbird dataset from {args.json_file}...")
    datasets_dict = create_waterbird_dataset_from_json(
        args.json_file, 
        args.image_dir
    )
    
    # Shuffle datasets if requested
    if args.shuffle:
        print("Shuffling datasets...")
        for split_name in datasets_dict:
            datasets_dict[split_name] = shuffle_dataset(
                datasets_dict[split_name], 
                args.seed
            )
    
    # Convert to datasets.Dataset
    train_ds = datasets.Dataset.from_list(datasets_dict["train"])
    val_ds = datasets.Dataset.from_list(datasets_dict["validation"])
    test_ds = datasets.Dataset.from_list(datasets_dict["test"])
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    # Save to parquet
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    test_ds.to_parquet(test_path)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved validation data to {val_path}")
    print(f"Saved test data to {test_path}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        # makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
        print(f"Copied data to HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    main()
