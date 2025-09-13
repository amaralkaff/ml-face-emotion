#!/usr/bin/env python3
"""
YOLOv8 Emotion Detection Training Script

This script trains a custom YOLOv8 model for facial emotion detection.

Usage:
    python scripts/train_emotion_model.py --data data/dataset.yaml --epochs 100
"""

import os
import sys
import platform
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

def create_dataset_yaml(data_dir: str, classes: list, output_path: str = "data/dataset.yaml"):
    """
    Create YOLO dataset YAML configuration file

    Args:
        data_dir (str): Path to dataset directory
        classes (list): List of emotion class names
        output_path (str): Output path for YAML file
    """
    dataset_config = {
        'path': str(Path(data_dir).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': {i: class_name for i, class_name in enumerate(classes)}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset YAML created at: {output_path}")
    return output_path

def validate_dataset_structure(data_path: str):
    """
    Validate that the dataset has the correct YOLO format structure

    Args:
        data_path (str): Path to dataset directory
    """
    required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']

    for req_dir in required_dirs:
        full_path = os.path.join(data_path, req_dir)
        if not os.path.exists(full_path):
            print(f"Warning: Missing directory {full_path}")
            return False

    print("Dataset structure validation passed")
    return True

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Emotion Detection Model')

    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                      help='Path to dataset YAML file or dataset directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='Base model to start training from')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Image size for training')
    parser.add_argument('--lr0', type=float, default=0.01,
                      help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                      help='Final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937,
                      help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                      help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                      help='Warmup epochs')

    # Output arguments
    parser.add_argument('--project', type=str, default='runs/train',
                      help='Project directory')
    parser.add_argument('--name', type=str, default='emotion_detection',
                      help='Experiment name')
    parser.add_argument('--save-period', type=int, default=10,
                      help='Save model every x epochs')

    # Advanced arguments
    default_workers = 0 if platform.system() == 'Windows' else 4
    parser.add_argument('--workers', type=int, default=default_workers,
                      help='Number of worker processes (set to 0 on Windows to avoid multiprocessing issues)')
    parser.add_argument('--device', type=str, default='',
                      help='Device to use for training (cpu, 0, 1, etc.)')
    parser.add_argument('--resume', type=str, default='',
                      help='Resume training from checkpoint')
    parser.add_argument('--amp', action='store_true',
                      help='Use Automatic Mixed Precision training')

    args = parser.parse_args()

    try:
        # Check if CUDA is available
        device = torch.cuda.is_available()
        print(f"CUDA available: {device}")
        if device:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.get_device_name()}")

        # Load base model
        print(f"Loading base model: {args.model}")
        model = YOLO(args.model)

        # Handle data path
        if args.data.endswith('.yaml'):
            data_yaml = args.data
        else:
            # Assume it's a directory, create YAML
            emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            data_yaml = create_dataset_yaml(args.data, emotion_classes)
            validate_dataset_structure(args.data)

        print(f"Using dataset configuration: {data_yaml}")

        # Training configuration
        train_config = {
            'data': data_yaml,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.img_size,
            'lr0': args.lr0,
            'lrf': args.lrf,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'project': args.project,
            'name': args.name,
            'save_period': args.save_period,
            'workers': args.workers,
            'resume': args.resume if args.resume else None,
            'device': 'cpu' if not torch.cuda.is_available() else (args.device if args.device else '0'),
            'amp': args.amp,
            'plots': True,
            'val': True,
            'save': True,
            'cache': 'disk' if platform.system() == 'Windows' else True,  # Use disk cache on Windows for stability
            'augment': True,  # Use data augmentation
        }

        # Remove None values
        train_config = {k: v for k, v in train_config.items() if v is not None and v != ''}

        print("Starting training with configuration:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")

        # Start training
        results = model.train(**train_config)

        print("\\nTraining completed successfully!")
        print(f"Best model saved at: {model.trainer.best}")
        print(f"Training results: {results}")

        # Save final model with custom name
        final_model_path = f"models/emotion_yolov8_{args.name}.pt"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Validation
        print("\\nRunning validation...")
        val_results = model.val()
        print(f"Validation results: {val_results}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Multiprocessing guard for Windows
    if platform.system() == 'Windows':
        import multiprocessing
        multiprocessing.freeze_support()
    main()
