#!/usr/bin/env python3
"""
YOLOv8 Facial Emotion Detection - Main Inference Script

Usage:
    python detect.py --source image.jpg --model yolov8n.pt
    python detect.py --source video.mp4 --model models/emotion_model.pt --save
    python detect.py --source 0 --model yolov8n.pt  # Webcam
"""

import argparse
import os
import sys
from pathlib import Path
import json

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from emotion_detector import EmotionDetector

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Facial Emotion Detection')
    parser.add_argument('--source', type=str, required=True,
                      help='Source: image file, video file, or camera (0 for webcam)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='Path to YOLOv8 model file')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--save', action='store_true',
                      help='Save detection results')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                      help='IoU threshold')
    parser.add_argument('--show', action='store_true',
                      help='Show results (for images)')
    parser.add_argument('--analyze', action='store_true',
                      help='Analyze and visualize emotion statistics')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    try:
        # Initialize emotion detector
        detector = EmotionDetector(model_path=args.model, config_path=args.config)

        # Update config with command line arguments
        detector.config['model']['confidence'] = args.conf
        detector.config['model']['iou_threshold'] = args.iou

        # Determine source type and process
        source = args.source

        if source.isdigit():
            # Webcam input
            print(f"Starting webcam emotion detection (camera {source})")
            detector.detect_emotions_webcam(camera_id=int(source))

        elif os.path.isfile(source):
            # File input (image or video)
            file_ext = Path(source).suffix.lower()

            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # Image processing
                print(f"Processing image: {source}")
                annotated_image, detections = detector.detect_emotions_image(source)

                # Save results
                if args.save:
                    output_image_path = os.path.join(args.output, f"annotated_{Path(source).name}")
                    import cv2
                    cv2.imwrite(output_image_path, annotated_image)
                    print(f"Annotated image saved to: {output_image_path}")

                    # Save detection data
                    detection_file = os.path.join(args.output, f"detections_{Path(source).stem}.json")
                    with open(detection_file, 'w') as f:
                        json.dump(detections, f, indent=2)
                    print(f"Detection data saved to: {detection_file}")

                # Show results
                if args.show:
                    import cv2
                    cv2.imshow('Emotion Detection Results', annotated_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                # Analyze results
                if args.analyze:
                    stats = detector.analyze_results(detections, save_plot=True)
                    print(f"\\nDetection Statistics:")
                    print(f"Total detections: {stats['total_detections']}")
                    print(f"Emotions found: {list(stats['emotion_counts'].keys())}")
                    print(f"Most common emotion: {stats['most_common_emotion']}")

            elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                # Video processing
                print(f"Processing video: {source}")
                output_video_path = None
                if args.save:
                    output_video_path = os.path.join(args.output, f"annotated_{Path(source).name}")

                all_detections = detector.detect_emotions_video(source, output_video_path)

                # Flatten detections for analysis
                flat_detections = []
                for frame_detections in all_detections:
                    flat_detections.extend(frame_detections)

                # Save detection data
                if args.save:
                    detection_file = os.path.join(args.output, f"detections_{Path(source).stem}.json")
                    with open(detection_file, 'w') as f:
                        json.dump(all_detections, f, indent=2)
                    print(f"Detection data saved to: {detection_file}")

                # Analyze results
                if args.analyze:
                    stats = detector.analyze_results(flat_detections, save_plot=True)
                    print(f"\\nVideo Analysis Statistics:")
                    print(f"Total frames processed: {len(all_detections)}")
                    print(f"Total detections: {stats['total_detections']}")
                    print(f"Emotions found: {list(stats['emotion_counts'].keys())}")
                    print(f"Most common emotion: {stats['most_common_emotion']}")

            else:
                print(f"Unsupported file format: {file_ext}")
                return

        else:
            print(f"Source not found: {source}")
            return

        print("\\nProcessing complete!")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()