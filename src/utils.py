# src/utils.py — Drawing utilities for frame annotation.

import cv2
import numpy as np

BOX_COLOR  = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw a bounding box and track ID label for each detection."""
    output = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det.get("track_id")
        label = f"{'ID' + str(track_id) if track_id is not None else '?'} {det['confidence']:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLOR, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(output, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    return output


def draw_zones(frame: np.ndarray, zones: dict, colors: dict) -> np.ndarray:
    """Draw semi-transparent waiting zone rectangles with name labels."""
    output = frame.copy() if not frame.flags["WRITEABLE"] else frame

    for name, (x1, y1, x2, y2) in zones.items():
        color = colors.get(name, (200, 200, 200))

        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, name, (x1 + 6, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return output


def draw_counts(frame: np.ndarray, counts: dict, colors: dict) -> np.ndarray:
    """Render a dark panel in the top-left corner showing live zone counts."""
    x, y = 16, 30
    line_h = 32

    panel_h = line_h * len(counts) + 12
    cv2.rectangle(frame, (x - 8, y - 22), (x + 200, y + panel_h - 10), (20, 20, 20), -1)

    for name, count in counts.items():
        color = colors.get(name, (200, 200, 200))
        cv2.putText(frame, f"{name}: {count}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        y += line_h

    return frame
