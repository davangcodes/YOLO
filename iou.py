"""
Script: IoU.py

Description:
This script evaluates a YOLOv8-trained model on the CholecT50 surgical dataset.
It computes:
- IoU@50 per class and overall
- mAP@50
- Precision, Recall, and F1-Score

It uses:
- Ground truth labels in YOLO format
- PyTorch + torchmetrics for mAP computation
- Ultralytics YOLOv8 inference
- Custom box IoU logic for class-wise accuracy

Author: Davang Sikand
Date: May 2025
"""

import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from ultralytics import YOLO

def xywh_to_xyxy(boxes):
    """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2] format."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

class CholecT50YOLODataset:
    def __init__(self, yaml_path, model_path="runs/detect/train16/weights/best.pt"):
        # Load dataset config from YAML
        with open(yaml_path, "r") as f:
            dataset_info = yaml.safe_load(f)

        self.test_dir = os.path.join(dataset_info["path"], dataset_info["test"])
        self.num_classes = dataset_info["nc"]
        self.class_names = dataset_info["names"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        
        print(f"✅ Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def compute_metrics(self):
        metric = MeanAveragePrecision(iou_type="bbox")
        preds, targets = [], []

        class_iou_hits = defaultdict(int)
        class_counts = defaultdict(int)

        image_files = [f for f in os.listdir(self.test_dir) if f.endswith(".jpg")]

        for image_file in tqdm(image_files, desc="Evaluating Test Set"):
            image_path = os.path.join(self.test_dir, image_file)
            label_path = os.path.join(self.test_dir.replace("images", "labels"), image_file.replace(".jpg", ".txt"))

            if not os.path.exists(label_path):
                continue

            # Read ground truth labels
            with open(label_path, "r") as f:
                lines = f.readlines()
                if not lines:
                    continue
                true_classes = [int(line.strip().split()[0]) for line in lines]
                true_bboxes = [list(map(float, line.strip().split()[1:])) for line in lines]

            gt_boxes = torch.tensor(true_bboxes)
            gt_labels = torch.tensor(true_classes)

            # Run YOLOv8 inference
            results = self.model(image_path)[0]
            if results.boxes is None or len(results.boxes) == 0:
                continue

            pred_boxes = results.boxes.xywh.cpu()
            pred_scores = results.boxes.conf.cpu()
            pred_labels = results.boxes.cls.cpu().int()

            # Convert to xyxy format for IoU
            pred_boxes_xyxy = xywh_to_xyxy(pred_boxes)
            gt_boxes_xyxy = xywh_to_xyxy(gt_boxes)

            # Collect for mAP
            preds.append({
                "boxes": pred_boxes_xyxy,
                "scores": pred_scores,
                "labels": pred_labels
            })
            targets.append({
                "boxes": gt_boxes_xyxy,
                "labels": gt_labels
            })

            # Class-wise IoU@50 accuracy
            for cls in torch.unique(gt_labels):
                cls = int(cls.item())
                idx_p = (pred_labels == cls)
                idx_t = (gt_labels == cls)

                if idx_p.sum() == 0 or idx_t.sum() == 0:
                    continue

                iou_matrix = box_iou(pred_boxes_xyxy[idx_p], gt_boxes_xyxy[idx_t])
                iou_max = iou_matrix.max(dim=1)[0]  # Best IoU for each predicted box

                class_iou_hits[cls] += (iou_max >= 0.50).sum().item()
                class_counts[cls] += len(iou_max)

        if not preds or not targets:
            print("❌ No valid predictions or targets found.")
            return {}

        # Compute mAP and metrics
        metric.update(preds, targets)
        results = metric.compute()

        # IoU@50 accuracy per class
        iou_scores = {
            self.class_names[cls]: round(class_iou_hits[cls] / (class_counts[cls] + 1e-6), 4)
            for cls in range(self.num_classes)
        }

        return {
            "IoU@50_overall": round(np.mean(list(iou_scores.values())), 4),
            "mAP@50": round(results["map_50"].item(), 4),
            "Precision": round(results["map"].item(), 4),
            "Recall": round(results["mar_100"].item(), 4),
            "F1-Score": round(2 * (results["map"].item() * results["mar_100"].item()) /
                              (results["map"].item() + results["mar_100"].item() + 1e-6), 4),
            "IoU@50_per_class": iou_scores
        }

if __name__ == "__main__":
    dataloader = CholecT50YOLODataset("cholect50_yolo.yaml")
    print("✅ Loaded dataset from YAML!")

    metrics = dataloader.compute_metrics()

    print("\n🎯 Final Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for cls, score in v.items():
                print(f"  {cls}: {score}")
        else:
            print(f"{k}: {v}")
