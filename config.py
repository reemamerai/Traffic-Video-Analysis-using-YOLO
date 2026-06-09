# config.py
# Central configuration for the traffic video analysis module.

# --- Paths ---
VIDEO_PATH = "videos/video.mp4"
VIDEO_FPS  = 30.0   # used to convert frame index → time in analytics
OUTPUT_VIDEO_PATH = "outputs/annotated_output.mp4"
OUTPUT_CSV_PATH   = "outputs/waiting_counts.csv"
MODEL_PATH = "models/best.pt"

# --- Detection ---
CONFIDENCE_THRESHOLD = 0.5

# Bounding box size filters (as fraction of total frame area)
# Boxes outside this range are treated as false positives and discarded.
MIN_BOX_AREA_RATIO = 0.0005   # ignore boxes smaller than 0.05% of frame
MAX_BOX_AREA_RATIO = 0.20     # ignore boxes larger than 20% of frame

# --- Vehicle class IDs ---
# When using best.pt (fine-tuned): class 0 = vehicle
# When using a pretrained COCO model: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [0]

# --- Tile-based detection (used by auto-labeling pipeline) ---
INFERENCE_IMAGE_SIZE = 640
TILE_SIZE = 640
TILE_OVERLAP = 100
NMS_IOU_THRESHOLD = 0.4

# --- Dataset ---
DATASET_IMAGES_DIR = "dataset/images"
DATASET_LABELS_DIR = "dataset/labels"
