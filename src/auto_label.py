# src/auto_label.py
# Auto-labels images using a pretrained YOLO model.
#
# Detects vehicles (car, bus, truck, motorcycle) and saves YOLO-format
# label files with a single unified class: vehicle (class 0).
#
# Output label format (one line per detection):
#   <class_id> <cx> <cy> <w> <h>   (all values normalized 0–1)

import os
import sys
import cv2
from ultralytics import YOLO

# Project root is one level above src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config

# COCO class IDs that map to "vehicle" (class 0 in our dataset)
VEHICLE_CLASSES = set(config.VEHICLE_CLASSES)  # {2, 3, 5, 7}


def auto_label(images_dir: str, labels_dir: str, model_path: str,
               confidence: float = 0.25):
    """
    Run YOLO on every image in `images_dir` and write YOLO-format label files
    to `labels_dir`. All vehicle types are saved as class 0.

    Args:
        images_dir:  folder containing input images
        labels_dir:  folder where .txt label files will be saved
        model_path:  path to pretrained YOLO weights
        confidence:  minimum detection confidence
    """
    os.makedirs(labels_dir, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        print(f"No images found in: {images_dir}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Images found : {len(image_files)}")
    print(f"Labels output: {labels_dir}")
    print("Auto-labeling...")

    labeled = 0

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping unreadable image: {img_name}")
            continue

        img_h, img_w = img.shape[:2]
        results = model(img, conf=confidence, verbose=False)[0]

        lines = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue

            # Convert xyxy → normalized cx, cy, w, h
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            # All vehicle classes map to class 0
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Write label file (even if empty, so the image is tracked)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        labeled += 1

    print(f"Done. Images processed: {labeled} | Label files created: {labeled}")
    print(f"Labels saved to: {labels_dir}")


if __name__ == "__main__":
    images_dir = os.path.join(PROJECT_ROOT, config.DATASET_IMAGES_DIR)
    labels_dir = os.path.join(PROJECT_ROOT, config.DATASET_LABELS_DIR)
    model_path = os.path.join(PROJECT_ROOT, config.MODEL_PATH)

    auto_label(
        images_dir=images_dir,
        labels_dir=labels_dir,
        model_path=model_path,
        confidence=0.25,
    )
