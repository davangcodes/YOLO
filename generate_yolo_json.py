"""
Script: generate_yolo_json.py

Description:
This script processes the CholecT50 dataset to generate YOLO-style JSON files
(train, validation, test) with:
- Base64-encoded images
- Bounding boxes in YOLO format [x_center, y_center, width, height]
- Instrument labels (instrument_label_i and boundingbox_instrument_i)

Author: Davang Sikand
Date: May 2025
"""

import os
import json
import base64
from tqdm import tqdm

class CholecT50YOLODataset:
    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (str): Path to the dataset root directory
        """
        self.dataset_dir = dataset_dir
        self.train_videos = ["VID68", "VID70", "VID73"]
        self.val_videos = ["VID74"]
        self.test_videos = ["VID75"]

    def encode_image(self, img_path):
        """Convert image to base64 string."""
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def convert_bbox_to_yolo(self, bx, by, bw, bh):
        """
        Convert bounding box to YOLO format.

        Args:
            bx, by: top-left x, y
            bw, bh: width and height

        Returns:
            [x_center, y_center, width, height] in normalized format
        """
        x_center = bx + (bw / 2)
        y_center = by + (bh / 2)
        return [round(x_center, 6), round(y_center, 6), round(bw, 6), round(bh, 6)]

    def process_video(self, video_id):
        """
        Process all frames and annotations in a single video.

        Args:
            video_id (str): Video identifier (e.g., 'VID68')

        Returns:
            list: List of dictionary entries for each frame in YOLO-style JSON format
        """
        video_path = os.path.join(self.dataset_dir, "videos", video_id)
        label_path = os.path.join(self.dataset_dir, "labels", f"{video_id}.json")

        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

        json_output = []
        for frame_no, objects in tqdm(annotations.items(), desc=f"Processing {video_id}"):
            image_path = os.path.join(video_path, f"{str(frame_no).zfill(6)}.png")
            if not os.path.exists(image_path):
                continue  # Skip missing frames

            # Convert image to base64
            image_base64 = self.encode_image(image_path)

            json_obj = {
                "video_no": video_id,
                "image_no": str(frame_no).zfill(6),
                "image": image_base64
            }

            # Loop through all instruments in the frame
            instrument_counter = 1
            for obj in objects:
                instrument_id = obj[1]       # Already an integer
                bx, by, bw, bh = obj[3:7]     # Bounding box

                if bw == -1 or bh == -1:
                    continue  # Skip invalid boxes

                bbox_yolo = self.convert_bbox_to_yolo(bx, by, bw, bh)

                # Add instrument label and bbox to JSON object
                json_obj[f"instrument_label_{instrument_counter}"] = instrument_id
                json_obj[f"boundingbox_instrument_{instrument_counter}"] = bbox_yolo

                instrument_counter += 1

            json_output.append(json_obj)
        return json_output

    def process_dataset(self, split="train"):
        """
        Generate YOLO-style JSON for the given split.

        Args:
            split (str): One of ['train', 'val', 'test']

        Returns:
            list: JSON entries for that split
        """
        if split == "train":
            video_list = self.train_videos
        elif split == "val":
            video_list = self.val_videos
        else:
            video_list = self.test_videos

        dataset_json = []
        for video_id in tqdm(video_list, desc=f"Processing {split} set"):
            dataset_json.extend(self.process_video(video_id))

        return dataset_json


if __name__ == "__main__":
    # Path to dataset directory
    dataset_dir = "../../../../../../../mount/Data1/Davang/cholect50-challenge-val"  # <-- Update if needed
    dataloader = CholecT50YOLODataset(dataset_dir)

    # Generate JSONs
    train_json = dataloader.process_dataset(split="train")
    val_json = dataloader.process_dataset(split="val")
    test_json = dataloader.process_dataset(split="test")

    # Save to disk
    with open("cholect50_yolo_train.json", "w") as f:
        json.dump(train_json, f, indent=4)
    with open("cholect50_yolo_val.json", "w") as f:
        json.dump(val_json, f, indent=4)
    with open("cholect50_yolo_test.json", "w") as f:
        json.dump(test_json, f, indent=4)

    print("✅ Train, Validation, and Test JSON datasets saved!")
