# Traffic Intersection Video Analysis

A computer vision module for detecting and counting vehicles at traffic intersections
using a trained YOLOv8 model on aerial/top-view footage.

## Project Structure

```
traffic_video_analysis/
├── models/
│   └── best.pt              # trained YOLO model
├── videos/
│   └── test1.mp4            # input video
├── outputs/                 # processed output videos saved here
├── dataset/
│   ├── images/              # extracted frames for training
│   └── labels/              # auto-generated YOLO label files
├── src/
│   ├── main.py              # entry point — runs the detection pipeline
│   ├── detector.py          # VehicleDetector class
│   ├── utils.py             # drawing utilities
│   ├── frame_extractor.py   # extract frames from video for dataset prep
│   ├── auto_label.py        # auto-label images using pretrained YOLO
│   ├── counter.py           # (future) vehicle counting logic
│   └── regions.py           # (future) direction region definitions (N/S/E/W)
├── config.py                # central configuration (paths, thresholds)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Detection

```bash
python src/main.py
```

- Displays live annotated video (press **Q** to quit early)
- Saves result to `outputs/output.mp4`

## Dataset Tools

Extract frames from video:
```bash
python src/frame_extractor.py
```

Auto-label extracted frames:
```bash
python src/auto_label.py
```
