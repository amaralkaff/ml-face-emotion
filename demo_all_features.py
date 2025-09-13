#!/usr/bin/env python3
"""
YOLOv8 Emotion Detection - Complete Feature Demo

This script demonstrates all features of the emotion detection system.

Usage:
    python demo_all_features.py
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def run_command(cmd, description):
    """Run command with description"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 40)
    result = os.system(cmd)
    print(f"Exit code: {result}")
    return result == 0

def main():
    print("🎯 YOLOv8 Facial Emotion Detection - Complete Demo")
    print("This demo showcases all features of the emotion detection system")

    # 1. Installation Test
    print_header("1. INSTALLATION TEST")
    success = run_command("python test_installation.py", "Testing installation and dependencies")

    if not success:
        print("❌ Installation test failed. Please check dependencies.")
        return

    # 2. Dataset Creation
    print_header("2. DATASET PREPARATION")

    print("\n📂 Creating sample emotion dataset...")
    run_command(
        "python scripts/download_datasets.py --dataset sample --output data/raw",
        "Creating sample emotion dataset with 5 classes x 10 images each"
    )

    print("\n🔄 Converting to YOLO format...")
    run_command(
        "python scripts/prepare_dataset.py --input data/raw/sample_emotions --output data/sample_yolo --format folders",
        "Converting dataset to YOLO format with train/val/test splits"
    )

    # 3. Image Detection Demo
    print_header("3. IMAGE DETECTION")

    # Check if there are any demo images
    demo_dir = Path("data/raw/demo/images")
    if demo_dir.exists():
        demo_images = list(demo_dir.glob("*.jpg"))
        if demo_images:
            sample_image = demo_images[0]
            print(f"\n📷 Testing on sample image: {sample_image}")
            run_command(
                f'python detect.py --source "{sample_image}" --save --analyze',
                "Detecting emotions in sample image with analysis"
            )

    # Test on bus image if it exists
    if os.path.exists("bus.jpg"):
        print("\n🚌 Testing on bus.jpg (general object detection)...")
        run_command(
            "python detect.py --source bus.jpg --save --analyze",
            "Testing general object detection capabilities"
        )

    # 4. Batch Processing Demo
    print_header("4. BATCH PROCESSING")

    # Create a simple batch processing demo
    print("\n📁 Batch processing demo images...")
    demo_images_dir = Path("data/raw/demo/images")
    if demo_images_dir.exists():
        run_command(
            f'python detect.py --source "{demo_images_dir}" --save',
            "Processing all images in demo directory"
        )

    # 5. Training Demo (Quick test)
    print_header("5. TRAINING DEMONSTRATION")

    print("\n🏋️ Quick training test (2 epochs for demo)...")
    run_command(
        "python scripts/train_emotion_model.py --data data/sample_yolo/dataset.yaml --epochs 2 --batch-size 4",
        "Quick training demonstration with minimal epochs"
    )

    # 6. Model Information
    print_header("6. SYSTEM INFORMATION")

    print("\\n📊 System Capabilities:")
    print("✅ Real-time webcam emotion detection")
    print("✅ Batch image and video processing")
    print("✅ Custom model training")
    print("✅ 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise")
    print("✅ YOLO format dataset preparation")
    print("✅ Comprehensive visualization and analysis")

    print("\\n🎯 Available Models:")
    print("• YOLOv8n: Nano - fastest, 6MB")
    print("• YOLOv8s: Small - balanced, 22MB")
    print("• YOLOv8m: Medium - better accuracy, 52MB")
    print("• YOLOv8l: Large - high accuracy, 87MB")
    print("• YOLOv8x: Extra Large - best accuracy, 136MB")

    # 7. Usage Examples
    print_header("7. USAGE EXAMPLES")

    print("\\n🖥️  Webcam Demo:")
    print("   python demo_webcam.py")
    print("   python demo_webcam.py --model models/custom_emotion.pt --confidence 0.6")

    print("\\n📸 Image Processing:")
    print("   python detect.py --source image.jpg --save --analyze --show")
    print("   python detect.py --source folder/ --save --conf 0.7")

    print("\\n🎬 Video Processing:")
    print("   python detect.py --source video.mp4 --save --analyze")
    print("   python detect.py --source 0 --save  # Webcam")

    print("\\n🏋️  Training:")
    print("   python scripts/train_emotion_model.py --data dataset.yaml --epochs 100")
    print("   python scripts/train_emotion_model.py --data dataset.yaml --model yolov8s.pt --epochs 200")

    print("\\n📁 Dataset Preparation:")
    print("   python scripts/prepare_dataset.py --input raw_data/ --output yolo_data/ --format folders")
    print("   python scripts/download_datasets.py --dataset fer2013")

    # 8. Results Summary
    print_header("8. DEMO RESULTS")

    print("\\n📈 Check the following directories for results:")
    print(f"• results/ - Processed images and videos")
    print(f"• data/sample_yolo/ - Prepared training dataset")
    print(f"• runs/train/ - Training logs and model checkpoints (if training completed)")

    print("\\n🎉 Demo completed successfully!")
    print("\\nNext steps:")
    print("1. Try the webcam demo: python demo_webcam.py")
    print("2. Train on your own dataset using the preparation scripts")
    print("3. Experiment with different YOLOv8 model sizes")
    print("4. Adjust confidence thresholds for your use case")

    # Final statistics
    results_dir = Path("results")
    if results_dir.exists():
        result_files = list(results_dir.glob("*"))
        print(f"\\n📊 Generated {len(result_files)} result files in results/ directory")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()