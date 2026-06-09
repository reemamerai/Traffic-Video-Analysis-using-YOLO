# src/detector.py — Vehicle detection and tracking using YOLOv8 + ByteTrack.

import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """Detects and tracks vehicles in video frames using a trained YOLO model."""

    def __init__(self, model_path: str, confidence: float = 0.5,
                 classes: list = None,
                 min_area_ratio: float = 0.0005,
                 max_area_ratio: float = 0.20):
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = classes
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def detect(self, frame: np.ndarray) -> list:
        """Run tracking inference on one frame. Returns a list of detection dicts
        with keys: bbox, track_id, class_id, class_name, confidence."""
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if self.classes is not None and class_id not in self.classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Discard boxes that are too small (noise) or too large (false positives)
            box_area = (x2 - x1) * (y2 - y1)
            if not (self.min_area_ratio <= box_area / frame_area <= self.max_area_ratio):
                continue

            track_id = int(box.id[0]) if box.id is not None else None

            detections.append({
                "bbox":       (x1, y1, x2, y2),
                "track_id":   track_id,
                "class_id":   class_id,
                "class_name": results.names[class_id],
                "confidence": float(box.conf[0]),
            })

        return detections
