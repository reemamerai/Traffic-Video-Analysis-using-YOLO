# src/zone_calibrator.py
# Interactive tool to visually define waiting zone coordinates.
#
# Usage (run from project root):
#   python src/zone_calibrator.py
#
# Controls:
#   Click + drag  →  draw a rectangle
#   R             →  redo current zone (clear last drawn box)
#   S             →  save current zone and move to the next one
#   Q             →  quit early
#
# After all 4 zones are drawn, the script prints ready-to-paste
# coordinates for regions.py.

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import config

ZONE_NAMES = ["N_WAIT", "S_WAIT", "E_WAIT", "W_WAIT"]
COLORS = [(255, 100, 0), (0, 200, 255), (0, 255, 100), (180, 0, 255)]

# Mouse state
drawing = False
start_x = start_y = 0
current_rect = None   # (x1, y1, x2, y2) being drawn


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        current_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_rect = (min(start_x, x), min(start_y, y),
                        max(start_x, x), max(start_y, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect = (min(start_x, x), min(start_y, y),
                        max(start_x, x), max(start_y, y))


def load_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise FileNotFoundError(f"Could not read video: {video_path}")
    return frame


def main():
    global current_rect

    video_path = os.path.join(PROJECT_ROOT, config.VIDEO_PATH)
    base_frame = load_first_frame(video_path)
    h, w = base_frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    print("Draw each zone by clicking and dragging. Press S to confirm, R to redo.\n")

    saved_zones = {}
    window = "Zone Calibrator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    for i, zone_name in enumerate(ZONE_NAMES):
        color = COLORS[i]
        current_rect = None

        while True:
            display = base_frame.copy()

            # Draw already confirmed zones
            for j, (name, rect) in enumerate(saved_zones.items()):
                rx1, ry1, rx2, ry2 = rect
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), COLORS[j], 2)
                cv2.putText(display, name, (rx1 + 4, ry1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[j], 2)

            # Draw current in-progress rectangle
            if current_rect:
                rx1, ry1, rx2, ry2 = current_rect
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 2)

            # Instruction overlay
            instruction = f"Drawing: {zone_name}  |  S=save  R=redo  Q=quit"
            cv2.putText(display, instruction, (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("s") and current_rect:
                saved_zones[zone_name] = current_rect
                print(f'  "{zone_name}": {current_rect},')
                break

            elif key == ord("r"):
                current_rect = None

            elif key == ord("q"):
                cv2.destroyAllWindows()
                print("\nCalibration cancelled.")
                return

    cv2.destroyAllWindows()

    # Print ready-to-paste output
    print("\n── Paste into src/regions.py ──────────────────────────────────")
    print("WAITING_ZONES = {")
    labels = {
        "N_WAIT": "North approach",
        "S_WAIT": "South approach",
        "E_WAIT": "East approach",
        "W_WAIT": "West approach",
    }
    for name, rect in saved_zones.items():
        print(f'    "{name}": {rect},  # {labels.get(name, "")}')
    print("}")


if __name__ == "__main__":
    main()
