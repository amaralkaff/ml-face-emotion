#!/usr/bin/env python3
"""
Dataset Download Script for YOLOv8 Emotion Detection

This script downloads popular emotion detection datasets.

Usage:
    python scripts/download_datasets.py --dataset fer2013 --output data/raw
"""

import argparse
import os
import requests
import zipfile
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

class DatasetDownloader:
    """Download and prepare emotion detection datasets"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_fer2013(self):
        """
        Download and prepare FER2013 dataset

        Note: FER2013 requires Kaggle API credentials or manual download
        This method provides instructions for manual download
        """
        print("FER2013 Dataset Download Instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Download the dataset manually")
        print("3. Extract to:", self.output_dir / "fer2013")
        print("4. Run: python scripts/prepare_dataset.py --input data/raw/fer2013 --output data/fer2013_yolo")
        print()
        print("Alternative: Install Kaggle API and run:")
        print("kaggle datasets download -d msambare/fer2013 -p", self.output_dir)

        # Check if already downloaded
        fer2013_path = self.output_dir / "fer2013"
        if fer2013_path.exists():
            print(f"FER2013 dataset found at: {fer2013_path}")
            return True

        return False

    def download_sample_emotions(self):
        """Download a small sample dataset for testing"""
        print("Creating sample emotion dataset for testing...")

        sample_dir = self.output_dir / "sample_emotions"

        # Create emotion directories
        emotions = ['angry', 'happy', 'sad', 'surprise', 'neutral']

        for emotion in emotions:
            emotion_dir = sample_dir / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)

            # Create sample colored images for each emotion
            for i in range(10):
                # Generate a simple colored image as placeholder
                img = self._create_sample_face(emotion, i)
                img_path = emotion_dir / f"{emotion}_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)

        print(f"Sample dataset created at: {sample_dir}")
        return True

    def download_affectnet_info(self):
        """Provide information about AffectNet dataset"""
        print("AffectNet Dataset Information:")
        print("AffectNet is a large-scale dataset with 1M+ images")
        print("Visit: http://mohammadmahoor.com/affectnet/")
        print("Registration required for download")
        print("Contains 8 emotion categories + valence/arousal annotations")
        print()

    def download_raf_db_info(self):
        """Provide information about RAF-DB dataset"""
        print("RAF-DB (Real-world Affective Faces Database) Information:")
        print("Contains ~15,000 facial images with emotion labels")
        print("Visit: http://www.whdeng.cn/raf/model1.html")
        print("Registration required for download")
        print("7 basic emotions + compound emotions")
        print()

    def create_demo_dataset(self):
        """Create a demo dataset with sample images for testing"""
        print("Creating demo dataset...")

        demo_dir = self.output_dir / "demo"
        images_dir = demo_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create some demo images
        demo_images = []

        for i, emotion in enumerate(['happy', 'sad', 'angry', 'surprise', 'neutral']):
            # Create a simple demo image
            img = self._create_demo_face(emotion, i)
            img_path = images_dir / f"demo_{emotion}_{i}.jpg"
            cv2.imwrite(str(img_path), img)
            demo_images.append(img_path)

        # Create a simple annotation file
        annotations = []
        for i, (img_path, emotion) in enumerate(zip(demo_images, ['happy', 'sad', 'angry', 'surprise', 'neutral'])):
            annotations.append({
                'image': img_path.name,
                'emotion': emotion,
                'bbox': [50, 50, 200, 200]  # Simple bbox
            })

        # Save annotations
        import json
        with open(demo_dir / "annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"Demo dataset created at: {demo_dir}")
        return True

    def _create_sample_face(self, emotion: str, index: int) -> np.ndarray:
        """Create a sample face image for testing"""
        # Create a 224x224 colored image
        img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Define colors for different emotions
        emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 0, 255), # Magenta
            'neutral': (128, 128, 128), # Gray
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 0)     # Dark Green
        }

        color = emotion_colors.get(emotion, (255, 255, 255))

        # Fill with base color
        img[:] = color

        # Add some simple face features
        center = (112, 112)

        # Face outline
        cv2.circle(img, center, 80, (255, 255, 255), 2)

        # Eyes
        cv2.circle(img, (85, 90), 8, (0, 0, 0), -1)
        cv2.circle(img, (139, 90), 8, (0, 0, 0), -1)

        # Nose
        cv2.line(img, (112, 105), (112, 125), (0, 0, 0), 2)

        # Mouth (different for each emotion)
        if emotion == 'happy':
            cv2.ellipse(img, (112, 140), (20, 10), 0, 0, 180, (0, 0, 0), 2)
        elif emotion == 'sad':
            cv2.ellipse(img, (112, 150), (20, 10), 0, 180, 360, (0, 0, 0), 2)
        elif emotion == 'angry':
            cv2.line(img, (95, 140), (129, 140), (0, 0, 0), 3)
        elif emotion == 'surprise':
            cv2.circle(img, (112, 140), 10, (0, 0, 0), 2)
        else:  # neutral
            cv2.line(img, (95, 140), (129, 140), (0, 0, 0), 2)

        # Add text label
        cv2.putText(img, f"{emotion}_{index}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return img

    def _create_demo_face(self, emotion: str, index: int) -> np.ndarray:
        """Create a more realistic demo face for testing"""
        # Create a larger image with gradient background
        img = np.zeros((300, 300, 3), dtype=np.uint8)

        # Create gradient background
        for i in range(300):
            for j in range(300):
                img[i, j] = [50 + i//6, 50 + j//6, 100]

        # Draw face
        center = (150, 150)

        # Face oval
        cv2.ellipse(img, center, (100, 120), 0, 0, 360, (220, 180, 140), -1)
        cv2.ellipse(img, center, (100, 120), 0, 0, 360, (200, 160, 120), 3)

        # Eyes
        cv2.ellipse(img, (120, 120), (15, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (180, 120), (15, 8), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (120, 120), 6, (0, 0, 0), -1)
        cv2.circle(img, (180, 120), 6, (0, 0, 0), -1)

        # Eyebrows (different for emotions)
        if emotion == 'angry':
            cv2.line(img, (105, 105), (135, 110), (0, 0, 0), 4)
            cv2.line(img, (165, 110), (195, 105), (0, 0, 0), 4)
        else:
            cv2.ellipse(img, (120, 105), (20, 5), 0, 0, 180, (0, 0, 0), 3)
            cv2.ellipse(img, (180, 105), (20, 5), 0, 0, 180, (0, 0, 0), 3)

        # Nose
        cv2.line(img, (150, 135), (150, 155), (160, 120, 100), 3)
        cv2.circle(img, (145, 160), 3, (160, 120, 100), -1)
        cv2.circle(img, (155, 160), 3, (160, 120, 100), -1)

        # Mouth (emotion-specific)
        if emotion == 'happy':
            cv2.ellipse(img, (150, 180), (25, 15), 0, 0, 180, (200, 80, 80), -1)
            cv2.ellipse(img, (150, 180), (25, 15), 0, 0, 180, (0, 0, 0), 2)
        elif emotion == 'sad':
            cv2.ellipse(img, (150, 195), (25, 15), 0, 180, 360, (200, 80, 80), -1)
            cv2.ellipse(img, (150, 195), (25, 15), 0, 180, 360, (0, 0, 0), 2)
        elif emotion == 'surprise':
            cv2.circle(img, (150, 180), 15, (200, 80, 80), -1)
            cv2.circle(img, (150, 180), 15, (0, 0, 0), 2)
        else:  # neutral, angry
            cv2.line(img, (125, 180), (175, 180), (0, 0, 0), 3)

        # Add emotion label
        cv2.putText(img, emotion.upper(), (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

def main():
    parser = argparse.ArgumentParser(description='Download emotion detection datasets')
    parser.add_argument('--dataset', type=str,
                       choices=['fer2013', 'sample', 'demo', 'affectnet_info', 'raf_db_info'],
                       default='sample',
                       help='Dataset to download or create')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory for datasets')

    args = parser.parse_args()

    try:
        downloader = DatasetDownloader(args.output)

        if args.dataset == 'fer2013':
            downloader.download_fer2013()
        elif args.dataset == 'sample':
            downloader.download_sample_emotions()
        elif args.dataset == 'demo':
            downloader.create_demo_dataset()
        elif args.dataset == 'affectnet_info':
            downloader.download_affectnet_info()
        elif args.dataset == 'raf_db_info':
            downloader.download_raf_db_info()

        print("Dataset download/creation completed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()