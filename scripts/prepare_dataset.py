#!/usr/bin/env python3
"""
Dataset Preparation Script for YOLOv8 Emotion Detection

This script helps prepare emotion datasets in YOLO format.
Supports various input formats and creates train/val/test splits.

Usage:
    python scripts/prepare_dataset.py --input data/raw --output data/processed
"""

import argparse
import os
import shutil
import random
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

class DatasetPreparer:
    """
    Dataset preparation utility for YOLOv8 emotion detection
    """

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.class_mapping = {cls: idx for idx, cls in enumerate(self.emotion_classes)}

    def prepare_from_folders(self, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """
        Prepare dataset from folder structure where each emotion is in separate folder

        Expected structure:
        input_dir/
        ├── angry/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── happy/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── ...

        Args:
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
        """
        print("Preparing dataset from folder structure...")

        # Create output directories
        self._create_yolo_structure()

        all_data = []

        # Process each emotion folder
        for emotion_folder in self.input_dir.iterdir():
            if not emotion_folder.is_dir():
                continue

            emotion_name = emotion_folder.name.lower()
            if emotion_name not in self.emotion_classes:
                print(f"Warning: Unknown emotion class '{emotion_name}', skipping...")
                continue

            class_id = self.class_mapping[emotion_name]

            # Process images in emotion folder
            for img_file in emotion_folder.glob("*.jpg"):
                if not img_file.exists():
                    continue

                # Load image to get dimensions
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"Warning: Could not load image {img_file}")
                    continue

                height, width = image.shape[:2]

                # For face emotion detection, assume entire image is the face
                # Create bounding box for the entire image (you may want to modify this)
                bbox = self._create_full_image_bbox(width, height)

                all_data.append({
                    'image_path': img_file,
                    'class_id': class_id,
                    'bbox': bbox,
                    'image_size': (width, height)
                })

        # Split data
        train_data, temp_data = train_test_split(all_data, train_size=train_ratio, random_state=42)
        val_size = val_ratio / (1 - train_ratio)
        val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)

        print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        # Copy files and create labels
        self._copy_and_label_data(train_data, 'train')
        self._copy_and_label_data(val_data, 'val')
        self._copy_and_label_data(test_data, 'test')

        # Create dataset YAML
        self._create_dataset_yaml()

        print("Dataset preparation completed!")

    def prepare_from_coco_annotations(self, annotations_file: str, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """
        Prepare dataset from COCO format annotations

        Args:
            annotations_file (str): Path to COCO annotations JSON file
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
        """
        print("Preparing dataset from COCO annotations...")

        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        # Create output directories
        self._create_yolo_structure()

        # Parse COCO data
        all_data = self._parse_coco_annotations(coco_data)

        # Split data
        train_data, temp_data = train_test_split(all_data, train_size=train_ratio, random_state=42)
        val_size = val_ratio / (1 - train_ratio)
        val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)

        print(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        # Copy files and create labels
        self._copy_and_label_data(train_data, 'train')
        self._copy_and_label_data(val_data, 'val')
        self._copy_and_label_data(test_data, 'test')

        # Create dataset YAML
        self._create_dataset_yaml()

        print("Dataset preparation completed!")

    def augment_dataset(self, augmentation_factor: int = 2):
        """
        Apply data augmentation to increase dataset size

        Args:
            augmentation_factor (int): Factor by which to multiply dataset size
        """
        print(f"Applying data augmentation (factor: {augmentation_factor})...")

        for split in ['train', 'val']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'

            if not images_dir.exists():
                continue

            original_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

            for img_file in original_images:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                # Load corresponding label
                label_file = labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    continue

                with open(label_file, 'r') as f:
                    label_data = f.read().strip()

                # Apply augmentations
                for i in range(augmentation_factor - 1):
                    aug_img = self._apply_augmentation(img)
                    aug_name = f"{img_file.stem}_aug{i+1}{img_file.suffix}"

                    # Save augmented image
                    cv2.imwrite(str(images_dir / aug_name), aug_img)

                    # Copy label (assuming augmentation doesn't change bbox significantly)
                    with open(labels_dir / f"{img_file.stem}_aug{i+1}.txt", 'w') as f:
                        f.write(label_data)

        print("Data augmentation completed!")

    def _create_yolo_structure(self):
        """Create YOLO dataset directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def _create_full_image_bbox(self, width: int, height: int) -> List[float]:
        """
        Create bounding box for entire image (for face-only images)

        Args:
            width (int): Image width
            height (int): Image height

        Returns:
            List[float]: YOLO format bbox [x_center, y_center, width, height] normalized
        """
        return [0.5, 0.5, 1.0, 1.0]  # Entire image

    def _copy_and_label_data(self, data: List[Dict], split: str):
        """Copy images and create YOLO format labels"""
        images_dir = self.output_dir / split / 'images'
        labels_dir = self.output_dir / split / 'labels'

        for idx, item in enumerate(data):
            # Copy image
            src_path = item['image_path']
            dst_path = images_dir / f"{split}_{idx:06d}{src_path.suffix}"
            shutil.copy2(src_path, dst_path)

            # Create label file
            label_file = labels_dir / f"{split}_{idx:06d}.txt"
            with open(label_file, 'w') as f:
                class_id = item['class_id']
                bbox = item['bbox']
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\\n")

    def _parse_coco_annotations(self, coco_data: Dict) -> List[Dict]:
        """Parse COCO annotations to internal format"""
        # This is a simplified implementation
        # You would need to adapt this based on your specific COCO annotation format
        all_data = []

        images = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        for annotation in coco_data['annotations']:
            image_info = images[annotation['image_id']]
            category_name = categories[annotation['category_id']]

            if category_name.lower() not in self.emotion_classes:
                continue

            # Convert COCO bbox to YOLO format
            x, y, w, h = annotation['bbox']
            img_w, img_h = image_info['width'], image_info['height']

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h

            all_data.append({
                'image_path': self.input_dir / image_info['file_name'],
                'class_id': self.class_mapping[category_name.lower()],
                'bbox': [x_center, y_center, width, height],
                'image_size': (img_w, img_h)
            })

        return all_data

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to image"""
        # Random brightness
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness)

        # Random rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

        # Random horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)

        return image

    def _create_dataset_yaml(self):
        """Create YOLO dataset configuration YAML"""
        yaml_content = f"""# YOLOv8 Emotion Detection Dataset Configuration
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(self.emotion_classes)}
names:
"""
        for idx, emotion in enumerate(self.emotion_classes):
            yaml_content += f"  {idx}: {emotion}\\n"

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"Dataset YAML created at: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLOv8 emotion detection')
    parser.add_argument('--input', type=str, required=True,
                      help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for processed dataset')
    parser.add_argument('--format', type=str, choices=['folders', 'coco'], default='folders',
                      help='Input dataset format')
    parser.add_argument('--annotations', type=str,
                      help='COCO annotations file (for COCO format)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                      help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                      help='Validation data ratio')
    parser.add_argument('--augment', type=int, default=1,
                      help='Data augmentation factor')

    args = parser.parse_args()

    # Validate arguments
    if args.format == 'coco' and not args.annotations:
        print("Error: COCO format requires --annotations file")
        return

    try:
        # Initialize preparer
        preparer = DatasetPreparer(args.input, args.output)

        # Prepare dataset based on format
        if args.format == 'folders':
            preparer.prepare_from_folders(args.train_ratio, args.val_ratio)
        elif args.format == 'coco':
            preparer.prepare_from_coco_annotations(args.annotations, args.train_ratio, args.val_ratio)

        # Apply augmentation if requested
        if args.augment > 1:
            preparer.augment_dataset(args.augment)

        print("Dataset preparation completed successfully!")

    except Exception as e:
        print(f"Dataset preparation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()