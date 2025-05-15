"""
Script Name: converting_yaml.py

Description:
This script converts image and annotation data from JSON format (with base64-encoded images and bounding boxes) 
to the YOLO format, which is commonly used for object detection training. It reads the dataset split files 
(train, val, test), decodes the images, saves them in JPEG format, and generates corresponding `.txt` files 
with bounding box coordinates in YOLO format.

Author: Davang Sikand
Date: May 2025
"""

import os
import json
import base64
from io import BytesIO
from PIL import Image

# Define input JSON files for each split
json_files = {
    "train": "cholect50_yolo_train.json",
    "val": "cholect50_yolo_val.json",
    "test": "cholect50_yolo_test.json"
}

# Define output directory where YOLO-formatted data will be saved
output_dir = "/home/davang/Documents/Surgical_DataScience/rendezvous/pytorch/CHOLECT50_RESEARCH/datasets"

# Create directory structure if it doesn't exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

def convert_json_to_yolo(json_path, split):
    """
    Converts a single JSON file (train/val/test) into YOLO format.
    Saves decoded images and corresponding label files with bounding boxes.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    for sample in data:
        video_no = sample["video_no"]
        image_no = sample["image_no"]

        try:
            # Decode base64 image
            image_data = base64.b64decode(sample["image"])
            img = Image.open(BytesIO(image_data))

            # Convert to RGB if image is in a different mode
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Save image to disk
            img_filename = f"{video_no}_{image_no}.jpg"
            img_path = os.path.join(output_dir, split, "images", img_filename)
            img.save(img_path)

        except Exception as e:
            print(f"⚠️ Error decoding/saving image {video_no}_{image_no}: {e}")
            continue

        # Generate YOLO label file
        label_path = os.path.join(output_dir, split, "labels", f"{video_no}_{image_no}.txt")
        with open(label_path, "w") as label_file:
            instrument_index = 1  # Begin with instrument_label_1

            while f"instrument_label_{instrument_index}" in sample:
                instrument_key = f"instrument_label_{instrument_index}"
                bbox_key = f"boundingbox_instrument_{instrument_index}"

                if bbox_key in sample and sample[bbox_key]:
                    class_id = sample[instrument_key]

                    try:
                        x, y, w, h = sample[bbox_key]  # Values must be in YOLO format [0,1]
                        label_file.write(f"{class_id} {x} {y} {w} {h}\n")
                    except Exception as e:
                        print(f"⚠️ Error writing label for {video_no}_{image_no}: {e}")

                instrument_index += 1  # Move to the next instrument

        print(f"✅ Processed: {img_filename}")

# Process all splits
for split, json_path in json_files.items():
    convert_json_to_yolo(json_path, split)

print("🎯 All JSON files successfully converted to YOLO format!")
