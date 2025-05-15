"""
Script: yolo_code.py

Description:
This script provides a full training and evaluation pipeline for YOLOv8 on the CholecT50 surgical dataset.

Key Features:
- Computes class imbalance weights (optional for analysis)
- Trains YOLOv8 (default: yolov8n.pt)
- Computes IoU@50, mAP@50, Precision, Recall, and F1-Score
- Evaluates metrics on train, val, and test splits

Author: Davang Sikand
"""

import os
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from collections import Counter
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO


class CholecT50YOLOPipeline:
    def __init__(self, yaml_path, model_name='yolov8n.pt'):
        """
        Initialize the pipeline with dataset YAML and YOLO model.

        Args:
            yaml_path (str): Path to YOLO dataset YAML file.
            model_name (str): YOLO model variant or checkpoint path.
        """
        with open(yaml_path, "r") as f:
            dataset_info = yaml.safe_load(f)

        self.dataset_path = dataset_info["path"]
        self.train_dir = os.path.join(self.dataset_path, dataset_info["train"])
        self.val_dir = os.path.join(self.dataset_path, dataset_info["val"])
        self.test_dir = os.path.join(self.dataset_path, dataset_info["test"])
        self.num_classes = dataset_info["nc"]
        self.class_names = dataset_info["names"]
        self.yaml_path = yaml_path

        self.model_name = model_name
        self.model = YOLO(model_name)

        print("📊 Computing class weights...")
        self.class_weights = self.compute_class_weights()
        print("✅ Class Weights:", self.class_weights)

    def compute_class_weights(self):
        """
        Compute inverse frequency class weights from training label files.
        Useful for analyzing class imbalance.

        Returns:
            dict: Normalized class weights.
        """
        class_counts = Counter()
        label_path = os.path.join(self.train_dir, "labels")
        label_files = [f for f in os.listdir(label_path) if f.endswith(".txt")]

        for file in label_files:
            with open(os.path.join(label_path, file)) as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1

        total = sum(class_counts.values())
        weights = {cls: total / (count + 1e-6) for cls, count in class_counts.items()}
        norm = sum(weights.values())
        return {cls: w / norm for cls, w in weights.items()}

    def train_yolo_model(self, epochs=100, imgsz=640):
        """
        Train YOLOv8 on the specified dataset.

        Args:
            epochs (int): Number of training epochs.
            imgsz (int): Image resolution.
        """
        print("🚀 Starting YOLOv8 training...")
        self.model.train(
            data=self.yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            device=0 if torch.cuda.is_available() else "cpu"
        )
        print("✅ Training completed.")
        # Reload best weights after training
        self.model = YOLO("runs/detect/train/weights/best.pt")

    def _xywhn_to_xyxy(self, box, image_size):
        """
        Convert normalized YOLO bbox to absolute xyxy format.

        Args:
            box (list): [x_center, y_center, width, height] in normalized format.
            image_size (tuple): (width, height) of image.

        Returns:
            list: [x1, y1, x2, y2] bounding box in pixel coords.
        """
        w, h = image_size
        x_c, y_c, bw, bh = box
        x1 = (x_c - bw / 2) * w
        y1 = (y_c - bh / 2) * h
        x2 = (x_c + bw / 2) * w
        y2 = (y_c + bh / 2) * h
        return [x1, y1, x2, y2]

    def compute_metrics(self, image_dir, label_dir):
        """
        Compute detection metrics (IoU@50, mAP, Precision, Recall, F1) for a directory.

        Args:
            image_dir (str): Path to image directory.
            label_dir (str): Path to corresponding YOLO label directory.

        Returns:
            dict: Evaluation metrics.
        """
        metric = MeanAveragePrecision(iou_type="bbox")
        preds, targets = [], []

        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

        for file in label_files:
            image_file = file.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, file)

            if not os.path.exists(image_path):
                continue

            # Load image
            image = Image.open(image_path).convert("RGB")
            results = self.model(image, verbose=False)[0]

            # Predictions
            pred_boxes = results.boxes.xyxy.cpu() if results.boxes else torch.zeros((0, 4))
            pred_scores = results.boxes.conf.cpu() if results.boxes else torch.zeros((0,))
            pred_labels = results.boxes.cls.int().cpu() if results.boxes else torch.zeros((0,), dtype=torch.int)

            # Ground truths
            with open(label_path, "r") as f:
                gt = [line.strip().split() for line in f]
                gt_labels = torch.tensor([int(x[0]) for x in gt])
                gt_boxes = torch.tensor([self._xywhn_to_xyxy(list(map(float, x[1:])), image.size) for x in gt])

            preds.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
            targets.append({"boxes": gt_boxes, "labels": gt_labels})

        if preds:
            metric.update(preds, targets)
            result = metric.compute()
            return {
                "IoU@50": round(result["map_50"].item(), 4),
                "Precision": round(result["map"].item(), 4),
                "Recall": round(result["mar_100"].item(), 4),
                "F1-Score": round(
                    2 * result["map"].item() * result["mar_100"].item() /
                    (result["map"].item() + result["mar_100"].item() + 1e-6), 4),
                "mAP@50": round(result["map_50"].item(), 4)
            }

        return {
            "IoU@50": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "mAP@50": 0.0
        }

    def evaluate_all(self):
        """
        Evaluate metrics on Train, Validation, and Test splits.
        """
        print("\n📈 Evaluating Train Set:")
        train_metrics = self.compute_metrics(
            os.path.join(self.train_dir, "images"),
            os.path.join(self.train_dir, "labels"))

        print("📈 Evaluating Validation Set:")
        val_metrics = self.compute_metrics(
            os.path.join(self.val_dir, "images"),
            os.path.join(self.val_dir, "labels"))

        print("📈 Evaluating Test Set:")
        test_metrics = self.compute_metrics(
            os.path.join(self.test_dir, "images"),
            os.path.join(self.test_dir, "labels"))

        print("\n🔍 Final Metrics:")
        for split, metrics in zip(["Train", "Validation", "Test"], [train_metrics, val_metrics, test_metrics]):
            print(f"\n--- {split} ---")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    pipeline = CholecT50YOLOPipeline("cholect50_yolo.yaml")
    pipeline.train_yolo_model(epochs=100)
    pipeline.evaluate_all()
