# Tennis Video Analysis

An AI-powered tennis video analysis system that uses computer vision and deep learning to automatically analyze tennis match footage. The system tracks players and the ball, detects court geometry, identifies shot events and shot types, estimates movement and ball speed, and generates tactical statistics through a web-based interface.
## Demo
<img width="735" height="411" alt="Tennis Video Analysis Demo" src="https://github.com/user-attachments/assets/49171a7b-0b4e-469a-a237-5b29d3993697" />

## Features

- **Player Detection and Tracking**
  - Detects players using YOLO.
  - Tracks players across video frames.
  - Filters detections to identify the two active tennis players based on court position.

- **Tennis Ball Detection and Tracking**
  - Uses a custom-trained YOLO model for tennis ball detection.
  - Applies spatial and temporal filtering to reduce false detections.
  - Interpolates missing ball positions to improve trajectory continuity.

- **Court Keypoint Detection**
  - Uses a ResNet50-based regression model to predict 14 tennis court keypoints.
  - Maps detected court geometry to a standardized mini-court representation.

- **Mini-Court Visualization**
  - Uses homography transformation to project player and ball positions onto a 2D court.
  - Visualizes player movement and ball trajectory.

- **Shot Event Detection**
  - Detects likely racket-hit events from changes in the ball trajectory.
  - Associates each hit with the most likely player.

- **Shot Type Classification**
  - Uses MediaPipe Pose to extract player body keypoints around detected hit frames.
  - Classifies shots into:
    - Serve
    - Forehand
    - Backhand
    - Volley
    - Smash

- **Performance Analysis**
  - Estimates ball speed.
  - Estimates player movement speed.
  - Calculates player recovery rate.
  - Analyzes shot placement by court depth and horizontal zones.
  - Analyzes shot direction patterns.
  - Generates basic tactical feedback.

- **Web Interface**
  - Upload tennis videos through a Flask web application.
  - Automatically processes uploaded footage.
  - Displays the analyzed video and performance statistics.

## System Pipeline

```text
Uploaded Tennis Video
        |
        v
+---------------------+
| Court Detection     |
| ResNet50 Keypoints  |
+---------------------+
        |
        v
+---------------------+
| Player Tracking     |
| YOLO + Tracking     |
+---------------------+
        |
        v
+---------------------+
| Ball Detection      |
| Custom YOLO Model   |
+---------------------+
        |
        v
+---------------------+
| Trajectory          |
| Interpolation       |
+---------------------+
        |
        +----------------------+
        |                      |
        v                      v
+------------------+   +-------------------+
| Mini-Court       |   | Hit Detection     |
| Homography       |   | Ball Trajectory   |
+------------------+   +-------------------+
        |                      |
        |                      v
        |              +-------------------+
        |              | Pose Estimation   |
        |              | MediaPipe Pose    |
        |              +-------------------+
        |                      |
        |                      v
        |              +-------------------+
        |              | Shot Type         |
        |              | Classification    |
        |              +-------------------+
        |                      |
        +-----------+----------+
                    |
                    v
          +--------------------+
          | Tennis Analytics   |
          | & Visualization    |
          +--------------------+
                    |
                    v
          Processed Video +
          Performance Statistics
```
## Tech Stack

### Computer Vision & Machine Learning

- Python
- YOLO / Ultralytics
- PyTorch
- Torchvision
- OpenCV
- MediaPipe
- NumPy
- Pandas

### Web Application

- Flask
- HTML
- CSS
- JavaScript

### Video Processing

- OpenCV
- FFmpeg

## Project Structure

```text
tennis-video-analysis/
├── analysis/
│   ├── constants/
│   ├── core_analysis/
│   ├── court_line_detector/
│   ├── mini_court/
│   ├── models/
│   ├── pose/
│   ├── shot/
│   ├── trackers/
│   ├── training/
│   ├── utils/
│   └── analyze.py
│
├── website/
│   ├── static/
│   ├── templates/
│   ├── auth.py
│   ├── models.py
│   └── views.py
│
├── app.py
├── .gitignore
└── README.md
```

## Analysis Pipeline

The main analysis pipeline is implemented in `analysis/analyze.py`.

### 1. Player Tracking

Players are detected and tracked using YOLO. Detected people are filtered using the predicted court boundaries to retain the two active tennis players.

### 2. Court Detection

A ResNet50-based regression model predicts 14 court keypoints (28 coordinate values). These keypoints define the geometry of the tennis court and support subsequent coordinate transformation.

### 3. Ball Tracking

A custom YOLO tennis-ball detector identifies candidate ball positions. Additional filtering considers candidate size, court boundaries, temporal consistency, and nearby player regions.

Missing detections are interpolated to create a more continuous ball trajectory.

### 4. Mini-Court Mapping

A homography transformation maps player and ball coordinates from the original camera view to a standardized 2D mini-court.

This representation is used for movement, distance, shot placement, and tactical analysis.

### 5. Shot Detection

Potential hit frames are detected by analyzing changes in the vertical movement of the ball trajectory.

The system then identifies the player most likely to have hit the ball based on spatial proximity.

### 6. Shot Classification

MediaPipe Pose extracts body landmarks around each detected hit event.

The current implementation uses pose, ball position, player position, temporal wrist movement, and court location to classify the event into five shot categories:

- Serve
- Forehand
- Backhand
- Volley
- Smash

### 7. Performance Statistics

The system derives several match-level metrics, including:

- Number of detected shots
- Average, minimum, and maximum ball speed
- Player movement speed
- Shot placement distribution
- Shot direction distribution
- Player recovery rate
- Shot-type distribution
- Tactical feedback

## Web Application

The Flask web interface allows users to upload tennis videos in:

- MP4
- AVI
- MOV
- MKV

Uploaded videos are processed by the analysis pipeline and returned with an annotated video and calculated statistics.

The current maximum upload size is **200 MB**.

## Model Training

The repository includes Jupyter notebooks used for model development:

```text
analysis/training/
├── tennis_ball_detector_training.ipynb
└── tennis_court_keypoints_training.ipynb
```

Training curves for the court keypoint model are also included.

Large datasets, generated training outputs, videos, and large model weights are excluded from Git using `.gitignore`.

## Installation

Clone the repository:

```bash
git clone https://github.com/Natasha123085/tennis-video-analysis.git
cd tennis-video-analysis
```

Create and activate a Python virtual environment.

Then install the required dependencies.

> **Note:** A `requirements.txt` file will be added to provide reproducible dependency installation.

FFmpeg must also be installed and available from the system command line for final video encoding.

## Running the Web Application

Run:

```bash
python app.py
```

Then open the local Flask server in your browser.

Upload a supported tennis video and wait for the analysis pipeline to complete.

## Current Limitations

- Shot classification currently uses heuristic rules based on pose and ball position rather than a learned temporal action-recognition model.
- Performance depends on camera angle, video quality, court visibility, and ball detection accuracy.
- Some model weights are not included in the repository because of file-size constraints.
- Player tracking currently assumes a standard singles tennis match with two primary players.
- Some analysis components assume a fixed video frame rate.

## Future Work

Future improvements may include:

- Train a temporal deep-learning model for shot classification.
- Improve small-object ball tracking across challenging frames.
- Support different camera viewpoints more robustly.
- Improve rally segmentation and tactical pattern recognition.
- Add player-specific match summaries and visual analytics.
- Improve model deployment and inference efficiency.
- Extend the web interface with richer interactive visualizations.
