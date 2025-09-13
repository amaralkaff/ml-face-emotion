"""
YOLOv8 Facial Emotion Detection Utility

This module provides classes and functions for detecting facial emotions using YOLOv8.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

class EmotionDetector:
    """
    YOLOv8-based Facial Emotion Detection class

    Supports both pre-trained and custom emotion detection models.
    """

    def __init__(self, model_path: str = "yolov8n.pt", config_path: str = "config.yaml"):
        """
        Initialize the emotion detector

        Args:
            model_path (str): Path to YOLOv8 model file
            config_path (str): Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.emotion_classes = []

        self._load_config()
        self._load_model()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
                self.emotion_classes = self.config.get('classes', [
                    'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
                ])
                print(f"Loaded config with {len(self.emotion_classes)} emotion classes")
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found. Using default settings.")
            self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            self.config = {
                'model': {'confidence': 0.5, 'iou_threshold': 0.45},
                'inference': {'save_results': True, 'show_labels': True, 'show_confidence': True}
            }

    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try to download default YOLOv8 model
            try:
                self.model = YOLO("yolov8n.pt")
                print("Using default YOLOv8n model")
            except Exception as e2:
                print(f"Failed to load any model: {e2}")
                raise

    def detect_emotions_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect emotions in a single image

        Args:
            image_path (str): Path to input image

        Returns:
            Tuple[np.ndarray, List[Dict]]: Processed image and detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Run inference
        results = self.model(image,
                           conf=self.config['model'].get('confidence', 0.5),
                           iou=self.config['model'].get('iou_threshold', 0.45))

        # Process results
        detections = []
        annotated_image = image.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get emotion label
                    if class_id < len(self.emotion_classes):
                        emotion = self.emotion_classes[class_id]
                    else:
                        emotion = f"class_{class_id}"

                    # Store detection
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'emotion': emotion,
                        'class_id': class_id
                    })

                    # Draw bounding box and label
                    if self.config['inference'].get('show_labels', True):
                        annotated_image = self._draw_detection(
                            annotated_image,
                            [int(x1), int(y1), int(x2), int(y2)],
                            emotion,
                            confidence
                        )

        return annotated_image, detections

    def detect_emotions_video(self, video_path: str, output_path: Optional[str] = None) -> List[List[Dict]]:
        """
        Detect emotions in video frames

        Args:
            video_path (str): Path to input video
            output_path (Optional[str]): Path to save output video

        Returns:
            List[List[Dict]]: Detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on frame
            results = self.model(frame,
                               conf=self.config['model'].get('confidence', 0.5),
                               iou=self.config['model'].get('iou_threshold', 0.45))

            # Process frame results
            frame_detections = []
            annotated_frame = frame.copy()

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        if class_id < len(self.emotion_classes):
                            emotion = self.emotion_classes[class_id]
                        else:
                            emotion = f"class_{class_id}"

                        frame_detections.append({
                            'frame': frame_count,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'emotion': emotion,
                            'class_id': class_id
                        })

                        if self.config['inference'].get('show_labels', True):
                            annotated_frame = self._draw_detection(
                                annotated_frame,
                                [int(x1), int(y1), int(x2), int(y2)],
                                emotion,
                                confidence
                            )

            all_detections.append(frame_detections)

            # Write frame to output video
            if out:
                out.write(annotated_frame)

            frame_count += 1

            # Print progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()
        if out:
            out.release()

        print(f"Video processing complete. Processed {frame_count} frames.")
        return all_detections

    def detect_emotions_webcam(self, camera_id: int = 0):
        """
        Real-time emotion detection from webcam

        Args:
            camera_id (int): Camera device ID (default: 0)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        print("Press 'q' to quit, 's' to save current frame")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = self.model(frame,
                               conf=self.config['model'].get('confidence', 0.5),
                               iou=self.config['model'].get('iou_threshold', 0.45))

            # Process and display results
            annotated_frame = frame.copy()

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        if class_id < len(self.emotion_classes):
                            emotion = self.emotion_classes[class_id]
                        else:
                            emotion = f"class_{class_id}"

                        annotated_frame = self._draw_detection(
                            annotated_frame,
                            [int(x1), int(y1), int(x2), int(y2)],
                            emotion,
                            confidence
                        )

            # Display frame
            cv2.imshow('Emotion Detection', annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"results/webcam_frame_{frame_count:04d}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"Frame saved to {save_path}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def _draw_detection(self, image: np.ndarray, bbox: List[int],
                       emotion: str, confidence: float) -> np.ndarray:
        """
        Draw bounding box and label on image

        Args:
            image (np.ndarray): Input image
            bbox (List[int]): Bounding box coordinates [x1, y1, x2, y2]
            emotion (str): Emotion label
            confidence (float): Confidence score

        Returns:
            np.ndarray: Annotated image
        """
        x1, y1, x2, y2 = bbox

        # Define colors for different emotions
        emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 255, 0),    # Green
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255)  # Orange
        }

        color = emotion_colors.get(emotion, (255, 255, 255))  # Default white

        # Draw bounding box
        thickness = self.config['inference'].get('line_thickness', 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if self.config['inference'].get('show_confidence', True):
            label = f"{emotion}: {confidence:.2f}"
        else:
            label = emotion

        # Get text size
        text_scale = self.config['inference'].get('text_scale', 0.6)
        text_thickness = max(1, thickness - 1)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)

        # Draw background rectangle for text
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

        # Draw text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   text_scale, (255, 255, 255), text_thickness)

        return image

    def analyze_results(self, detections: List[Dict], save_plot: bool = True) -> Dict:
        """
        Analyze detection results and create visualizations

        Args:
            detections (List[Dict]): List of detection results
            save_plot (bool): Whether to save analysis plots

        Returns:
            Dict: Analysis statistics
        """
        if not detections:
            return {"total_detections": 0, "emotion_counts": {}}

        # Count emotions
        emotion_counts = {}
        total_detections = len(detections)

        for detection in detections:
            emotion = detection['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Calculate percentages
        emotion_percentages = {emotion: (count / total_detections) * 100
                             for emotion, count in emotion_counts.items()}

        # Create visualization
        if save_plot:
            self._create_emotion_plot(emotion_counts, emotion_percentages)

        # Prepare statistics
        stats = {
            "total_detections": total_detections,
            "emotion_counts": emotion_counts,
            "emotion_percentages": emotion_percentages,
            "most_common_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
        }

        return stats

    def _create_emotion_plot(self, emotion_counts: Dict, emotion_percentages: Dict):
        """Create emotion distribution plot"""
        plt.figure(figsize=(12, 6))

        # Subplot 1: Count distribution
        plt.subplot(1, 2, 1)
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())

        colors = ['red', 'green', 'magenta', 'yellow', 'gray', 'blue', 'orange'][:len(emotions)]
        bars = plt.bar(emotions, counts, color=colors, alpha=0.7)
        plt.title('Emotion Detection Counts')
        plt.xlabel('Emotions')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        # Subplot 2: Percentage distribution
        plt.subplot(1, 2, 2)
        plt.pie(emotion_percentages.values(), labels=emotion_percentages.keys(),
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Emotion Distribution (%)')

        plt.tight_layout()
        plt.savefig('results/emotion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Emotion analysis plot saved to results/emotion_analysis.png")