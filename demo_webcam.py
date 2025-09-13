#!/usr/bin/env python3
"""
Real-time Webcam Emotion Detection Demo

This script demonstrates real-time facial emotion detection using YOLOv8.

Usage:
    python demo_webcam.py --model yolov8n.pt
    python demo_webcam.py --model models/emotion_yolov8.pt --confidence 0.6
"""

import argparse
import cv2
import sys
from pathlib import Path
import time
import numpy as np

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from emotion_detector import EmotionDetector

class WebcamEmotionDemo:
    """Real-time webcam emotion detection demo"""

    def __init__(self, model_path: str, config_path: str = "config.yaml", camera_id: int = 0):
        self.detector = EmotionDetector(model_path, config_path)
        self.camera_id = camera_id
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Statistics tracking
        self.emotion_history = []
        self.detection_history = []

    def run_demo(self):
        """Run the webcam emotion detection demo"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting webcam emotion detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset statistics")
        print("  'h' - Show/hide help")
        print("  'f' - Toggle FPS display")

        show_help = True
        show_fps = True
        last_fps_update = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Run emotion detection
            start_inference = time.time()
            results = self.detector.model(frame,
                                        conf=self.detector.config['model'].get('confidence', 0.5),
                                        iou=self.detector.config['model'].get('iou_threshold', 0.45),
                                        verbose=False)

            inference_time = time.time() - start_inference

            # Process results
            annotated_frame = self._process_and_annotate_frame(frame, results)

            # Update statistics
            self._update_statistics(results, inference_time)

            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_update >= 1.0:
                self.fps = self.frame_count / (current_time - self.start_time)
                last_fps_update = current_time

            # Draw overlays
            annotated_frame = self._draw_overlays(annotated_frame, show_help, show_fps)

            # Display frame
            cv2.imshow('YOLOv8 Emotion Detection Demo', annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_current_frame(annotated_frame)
            elif key == ord('r'):
                self._reset_statistics()
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('f'):
                show_fps = not show_fps

            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Print final statistics
        self._print_session_statistics()

    def _process_and_annotate_frame(self, frame, results):
        """Process detection results and annotate frame"""
        annotated_frame = frame.copy()
        frame_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get emotion label
                    if class_id < len(self.detector.emotion_classes):
                        emotion = self.detector.emotion_classes[class_id]
                    else:
                        emotion = f"class_{class_id}"

                    frame_detections.append({
                        'emotion': emotion,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })

                    # Draw detection
                    annotated_frame = self.detector._draw_detection(
                        annotated_frame,
                        [int(x1), int(y1), int(x2), int(y2)],
                        emotion,
                        confidence
                    )

                    # Add confidence bar
                    self._draw_confidence_bar(annotated_frame, [int(x1), int(y1), int(x2), int(y2)], confidence)

        return annotated_frame

    def _draw_confidence_bar(self, image, bbox, confidence):
        """Draw confidence bar next to detection"""
        x1, y1, x2, y2 = bbox
        bar_width = 20
        bar_height = y2 - y1
        bar_x = x2 + 5

        # Background bar
        cv2.rectangle(image, (bar_x, y1), (bar_x + bar_width, y2), (50, 50, 50), -1)

        # Confidence bar
        conf_height = int(bar_height * confidence)
        conf_y = y2 - conf_height

        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        cv2.rectangle(image, (bar_x, conf_y), (bar_x + bar_width, y2), color, -1)

        # Border
        cv2.rectangle(image, (bar_x, y1), (bar_x + bar_width, y2), (255, 255, 255), 1)

    def _draw_overlays(self, frame, show_help, show_fps):
        """Draw help text and statistics on frame"""
        overlay = frame.copy()
        alpha = 0.7

        # FPS display
        if show_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Help text
        if show_help:
            help_y = 60
            help_texts = [
                "Controls:",
                "Q - Quit",
                "S - Save frame",
                "R - Reset stats",
                "H - Toggle help",
                "F - Toggle FPS"
            ]

            max_width = max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for text in help_texts])
            cv2.rectangle(frame, (5, 45), (max_width + 20, help_y + len(help_texts) * 20), (0, 0, 0), -1)

            for i, text in enumerate(help_texts):
                y_pos = help_y + i * 20
                cv2.putText(frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Emotion statistics (bottom right)
        if self.emotion_history:
            self._draw_emotion_stats(frame)

        return frame

    def _draw_emotion_stats(self, frame):
        """Draw emotion statistics on frame"""
        if not self.emotion_history:
            return

        # Count recent emotions (last 30 detections)
        recent_emotions = self.emotion_history[-30:]
        emotion_counts = {}

        for emotion in recent_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Draw stats box
        stats_x = frame.shape[1] - 250
        stats_y = frame.shape[0] - 150
        stats_height = len(emotion_counts) * 25 + 40

        cv2.rectangle(frame, (stats_x, stats_y - stats_height),
                     (frame.shape[1] - 10, stats_y), (0, 0, 0), -1)

        # Title
        cv2.putText(frame, "Recent Emotions:", (stats_x + 5, stats_y - stats_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Emotion counts
        y_offset = 45
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(recent_emotions)) * 100
            text = f"{emotion}: {count} ({percentage:.1f}%)"
            cv2.putText(frame, text, (stats_x + 5, stats_y - stats_height + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 25

    def _update_statistics(self, results, inference_time):
        """Update detection statistics"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id < len(self.detector.emotion_classes):
                        emotion = self.detector.emotion_classes[class_id]
                        self.emotion_history.append(emotion)

        # Keep history manageable
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-100:]

        self.detection_history.append({
            'frame': self.frame_count,
            'inference_time': inference_time,
            'num_detections': len(boxes) if 'boxes' in locals() and boxes is not None else 0
        })

    def _save_current_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = int(time.time())
        filename = f"results/webcam_emotion_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved: {filename}")

    def _reset_statistics(self):
        """Reset statistics tracking"""
        self.emotion_history.clear()
        self.detection_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
        print("Statistics reset")

    def _print_session_statistics(self):
        """Print final session statistics"""
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0

        print(f"\\n=== Session Statistics ===")
        print(f"Total frames: {self.frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")

        if self.emotion_history:
            emotion_counts = {}
            for emotion in self.emotion_history:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            print(f"\\nEmotion Distribution:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.emotion_history)) * 100
                print(f"  {emotion}: {count} ({percentage:.1f}%)")

        if self.detection_history:
            avg_inference = np.mean([d['inference_time'] for d in self.detection_history])
            print(f"\\nAverage inference time: {avg_inference*1000:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description='Real-time Webcam Emotion Detection Demo')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLOv8 model file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')

    args = parser.parse_args()

    try:
        # Create results directory
        Path("results").mkdir(exist_ok=True)

        # Initialize and run demo
        demo = WebcamEmotionDemo(args.model, args.config, args.camera)

        # Update confidence if provided
        demo.detector.config['model']['confidence'] = args.confidence

        demo.run_demo()

    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()