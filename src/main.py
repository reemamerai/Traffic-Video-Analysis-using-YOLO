# src/main.py
# Main entry point for traffic intersection vehicle detection and waiting-zone counting.
#
# Run from the project root:
#   python src/main.py

import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import config
from src.detector import VehicleDetector
from src.regions import WAITING_ZONES, ZONE_COLORS
from src.counter import ZoneCounter
from src.utils import draw_detections, draw_zones, draw_counts


def main():
    video_path  = os.path.join(PROJECT_ROOT, config.VIDEO_PATH)
    output_path = os.path.join(PROJECT_ROOT, config.OUTPUT_VIDEO_PATH)
    csv_path    = os.path.join(PROJECT_ROOT, config.OUTPUT_CSV_PATH)
    model_path  = os.path.join(PROJECT_ROOT, config.MODEL_PATH)

    # --- Setup ---
    detector = VehicleDetector(
        model_path,
        confidence=config.CONFIDENCE_THRESHOLD,
        classes=config.VEHICLE_CLASSES,
        min_area_ratio=config.MIN_BOX_AREA_RATIO,
        max_area_ratio=config.MAX_BOX_AREA_RATIO,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    zone_names = list(WAITING_ZONES.keys())
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["frame_index"] + zone_names)
    csv_writer.writeheader()

    print(f"Video : {width}x{height} @ {fps:.1f} fps — {total} frames")
    print(f"Output video : {output_path}")
    print(f"Output CSV   : {csv_path}")
    print("Processing... (press Q to quit early)")

    window = "Traffic Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    counter = ZoneCounter(WAITING_ZONES)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect + track vehicles
        detections = detector.detect(frame)

        # 2. Live waiting counts for this frame
        counts = counter.update(detections)

        # 3. Draw — zones, boxes, count overlay
        output_frame = draw_zones(frame, WAITING_ZONES, ZONE_COLORS)
        output_frame = draw_detections(output_frame, detections)
        draw_counts(output_frame, counts, ZONE_COLORS)

        # 4. Save annotated frame to video
        writer.write(output_frame)

        # 5. Save per-frame counts to CSV
        csv_writer.writerow({"frame_index": frame_idx, **counts})

        cv2.imshow(window, output_frame)

        key = cv2.waitKey(1) & 0xFF
        window_closed = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
        if key == ord("q") or window_closed:
            print("Quit early by user.")
            break

        frame_idx += 1
        if total > 0 and frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total}")

    # --- Cleanup ---
    cap.release()
    writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"\n--- Summary ---")
    print(f"Frames processed : {frame_idx}")
    print(f"Annotated video  : {output_path}")
    print(f"Waiting counts   : {csv_path}")


if __name__ == "__main__":
    main()
