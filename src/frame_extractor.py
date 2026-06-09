# src/frame_extractor.py
# Extracts frames from a video at a fixed interval and saves them as images.

import os
import cv2

# Project root is one level above this file (src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_frames(video_path: str, output_dir: str, interval: int = 30):
    """
    Extract one frame every `interval` frames from a video file.

    Args:
        video_path: path to the input video (relative to project root or absolute)
        output_dir: output directory (relative to project root or absolute)
        interval:   save one frame every this many frames (default: 30)
    """
    # Resolve paths relative to the project root
    if not os.path.isabs(video_path):
        video_path = os.path.join(PROJECT_ROOT, video_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")
    print(f"Extracting every {interval} frames...")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            saved += 1
            filename = os.path.join(output_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(filename, frame)

        frame_idx += 1

    cap.release()
    print(f"Done. Frames processed: {frame_idx} | Images saved: {saved}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    extract_frames(
        video_path="videos/video.mp4",
        output_dir="dataset/images",
        interval=30,
    )
