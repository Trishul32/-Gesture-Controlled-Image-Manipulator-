# Gesture-Controlled Image Manipulator

Real-Time Hand Gesture Based Image Transformations using OpenCV and MediaPipe.

## Overview

This project demonstrates a touchless, gesture-based interface for controlling image transformations. Using a standard webcam, the system detects hand gestures and maps them to image manipulation operations in real-time.

## Features

- **Real-time Hand Detection**: 21-point hand landmark detection using MediaPipe
- **Gesture Recognition**: Intuitive gesture-to-action mapping
- **Image Transformations**: Scale, rotate, translate, and flip operations
- **Low Latency**: ~30ms frame processing time

## Gesture Controls

| Gesture | Action |
|---------|--------|
| Pinch (thumb + index) | Zoom in/out |
| Tilt hand | Rotate image |
| Move hand | Pan image |
| Make fist | Toggle horizontal flip |

## Technology Stack

- **Python** - Core programming language
- **OpenCV** - Image processing and transformations
- **MediaPipe** - 21-point hand landmark detection
- **NumPy** - Matrix and array computations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Trishul32/-Gesture-Controlled-Image-Manipulator-.git
cd -Gesture-Controlled-Image-Manipulator-
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the MediaPipe model:
The model is included in `assets/hand_landmarker.task`. If missing, download from:
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

Run the main application:
```bash
python src/main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset all transformations |
| `c` | Recalibrate gestures |
| `s` | Save screenshot |

## Project Structure

```
cv_project/
├── src/
│   ├── hand_detector.py       # Hand landmark detection (Person A)
│   ├── gesture_recognizer.py  # Gesture interpretation (Person B)
│   ├── image_transformer.py   # Image transformations (Person C)
│   └── main.py               # Main application (Person D)
├── assets/
│   └── hand_landmarker.task  # MediaPipe model
├── requirements.txt
└── README.md
```

## Module Descriptions

### Hand Detector (`hand_detector.py`)
- Captures webcam frames
- Uses MediaPipe HandLandmarker for 21-point detection
- Provides normalized landmark coordinates (0-1)

### Gesture Recognizer (`gesture_recognizer.py`)
- Interprets landmarks as gestures
- Calculates pinch distance for scaling
- Measures hand tilt angle for rotation
- Tracks centroid movement for translation
- Detects fist gesture for flip toggle

### Image Transformer (`image_transformer.py`)
- Applies transformations using OpenCV:
  - `cv2.resize()` - Scaling
  - `cv2.getRotationMatrix2D()` + `cv2.warpAffine()` - Rotation
  - Affine transformation - Translation
  - `cv2.flip()` - Reflection

### Main Application (`main.py`)
- Integrates all modules
- Provides UI with live visualization
- Handles user input

## Performance

| Metric | Value |
|--------|-------|
| Static Gesture Accuracy | 90-95% |
| Dynamic Gesture Accuracy | 80-90% |
| Frame Processing Time | ~30ms |

## Team

This is a Computer Vision AAT Mini Project demonstrating practical application of transformation matrices and human-computer interaction.

## License

MIT License
