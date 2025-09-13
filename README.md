# YOLOv8 Facial Emotion Detection

Real-time emotion detection using YOLOv8. Detects 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Webcam (real-time)
python demo_webcam.py

# Image/Video
python detect.py --source path/to/file --save

# Training
python scripts/train_emotion_model.py --data dataset.yaml --epochs 100
```

## Project Structure

```
ml-face-emotion/
├── detect.py                # Main detection script
├── demo_webcam.py           # Webcam demo
├── scripts/                 # Training & dataset tools
├── utils/                   # Core detection module
├── data/                    # Datasets
├── models/                  # Trained models
└── results/                 # Output files
```

## Dataset

Using [AffectNet YOLO Format](https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format) dataset from Kaggle.

```bash
# Download from Kaggle and extract to data/
# Dataset already in YOLO format, ready for training
```

## Training

```bash
# Basic training
python scripts/train_emotion_model.py --data dataset.yaml --epochs 100

# Resume training
python scripts/train_emotion_model.py --resume last.pt
```

## Detection Options

```bash
python detect.py --source INPUT [OPTIONS]
  --source    Image, video, or 0 for webcam
  --model     Model path (default: yolov8n.pt)
  --conf      Confidence threshold (default: 0.5)
  --save      Save results
  --show      Display results
```

## Emotion Classes

1. Angry (Red)
2. Disgust (Green)
3. Fear (Magenta)
4. Happy (Yellow)
5. Neutral (Gray)
6. Sad (Blue)
7. Surprise (Orange)

## Model Options

- **YOLOv8n**: Fast, 6MB (real-time)
- **YOLOv8s**: Balanced, 22MB
- **YOLOv8m**: Accurate, 52MB
- **YOLOv8l**: High accuracy, 87MB
- **YOLOv8x**: Best accuracy, 136MB

## Requirements

- Python 3.8+
- 8GB RAM minimum
- GPU recommended for training

## License

MIT License