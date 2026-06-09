# webapp/app.py
# Flask web dashboard for traffic intersection analysis.
#
# Run from the project root:
#   python webapp/app.py
# Then open: http://localhost:5000

import os
import sys
import csv
import json
import shutil
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify

import config
from src.detector import VehicleDetector
from src.regions import WAITING_ZONES, ZONE_COLORS
from src.counter import ZoneCounter
from src.utils import draw_detections, draw_zones, draw_counts

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

WEBAPP_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR    = os.path.join(WEBAPP_DIR, "static")
UPLOADS_DIR   = os.path.join(STATIC_DIR, "uploads")
OUTPUTS_DIR   = os.path.join(STATIC_DIR, "outputs")
PLOTS_DIR     = os.path.join(STATIC_DIR, "plots")
PREVIEW_PATH  = os.path.join(OUTPUTS_DIR, "frame.jpg")
OUTPUT_VIDEO  = os.path.join(OUTPUTS_DIR, "annotated_output.mp4")
OUTPUT_CSV    = os.path.join(OUTPUTS_DIR, "waiting_counts.csv")

MODEL_PATH    = os.path.join(PROJECT_ROOT, config.MODEL_PATH)

# ── Shared processing state ───────────────────────────────────────────────────

state = {
    "status":       "idle",     # idle | processing | done | error
    "frame_idx":    0,
    "total_frames": 0,
    "counts":       {z: 0 for z in WAITING_ZONES},
    "error":        "",
}

# ── Processing thread ─────────────────────────────────────────────────────────

def run_pipeline(video_path: str):
    """Run detection + tracking pipeline in a background thread."""
    global state
    state["status"]    = "processing"
    state["frame_idx"] = 0
    state["error"]     = ""

    try:
        detector = VehicleDetector(
            MODEL_PATH,
            confidence=config.CONFIDENCE_THRESHOLD,
            classes=config.VEHICLE_CLASSES,
            min_area_ratio=config.MIN_BOX_AREA_RATIO,
            max_area_ratio=config.MAX_BOX_AREA_RATIO,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        state["total_frames"] = total

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        writer = cv2.VideoWriter(
            OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        csv_file   = open(OUTPUT_CSV, "w", newline="")
        zone_names = list(WAITING_ZONES.keys())
        csv_writer = csv.DictWriter(csv_file, fieldnames=["frame_index"] + zone_names)
        csv_writer.writeheader()

        counter   = ZoneCounter(WAITING_ZONES)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections   = detector.detect(frame)
            counts       = counter.update(detections)
            output_frame = draw_zones(frame, WAITING_ZONES, ZONE_COLORS)
            output_frame = draw_detections(output_frame, detections)
            draw_counts(output_frame, counts, ZONE_COLORS)

            writer.write(output_frame)
            csv_writer.writerow({"frame_index": frame_idx, **counts})

            # Update shared state every 5 frames
            if frame_idx % 5 == 0:
                cv2.imwrite(PREVIEW_PATH, output_frame)
                state["frame_idx"] = frame_idx
                state["counts"]    = counts.copy()

            frame_idx += 1

        cap.release()
        writer.release()
        csv_file.close()

        # Run analytics and copy plots
        _generate_plots()

        state["frame_idx"] = frame_idx
        state["status"]    = "done"

    except Exception as exc:
        state["status"] = "error"
        state["error"]  = str(exc)
        raise


def _generate_plots():
    """Run analyze_results logic and copy PNGs to static/plots/."""
    import importlib.util, types

    # Temporarily redirect CSV and plots paths into our webapp dirs
    orig_csv   = config.OUTPUT_CSV_PATH
    config.OUTPUT_CSV_PATH = OUTPUT_CSV

    os.makedirs(PLOTS_DIR, exist_ok=True)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")   # no display needed in background thread
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    ZONES = list(WAITING_ZONES.keys())
    ZONE_C = {"N_WAIT": "#4A90D9", "S_WAIT": "#E8A838",
               "E_WAIT": "#4CAF50", "W_WAIT": "#9C5DDB"}
    SMOOTH = 30

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15,
                          "figure.facecolor": "#F8F9FA"})

    df  = pd.read_csv(OUTPUT_CSV)
    fps = float(getattr(config, "VIDEO_FPS", 30))
    df["time_sec"] = df["frame_index"] / fps

    def save_plot(fig, name):
        fig.savefig(os.path.join(PLOTS_DIR, name), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Time-series
    fig, ax = plt.subplots(figsize=(13, 4))
    for z in ZONES:
        smooth = df[z].rolling(SMOOTH, center=True, min_periods=1).mean()
        ax.plot(df["time_sec"], df[z], color=ZONE_C[z], alpha=0.2, linewidth=0.8)
        ax.plot(df["time_sec"], smooth, color=ZONE_C[z], linewidth=2, label=z)
        peak_i = smooth.idxmax()
        ax.scatter(df["time_sec"].iloc[peak_i], smooth.iloc[peak_i],
                   color=ZONE_C[z], s=55, zorder=5)
    ax.set_title("Waiting Vehicles Over Time (smoothed)", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vehicles")
    ax.legend()
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save_plot(fig, "timeseries.png")

    # Stats bar
    stats = df[ZONES].agg(["mean", "max", "std"]).T
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ZONES, stats["mean"],
                  yerr=stats["std"],
                  color=[ZONE_C[z] for z in ZONES],
                  edgecolor="white", width=0.55, capsize=5)
    for bar, val in zip(bars, stats["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stats["std"].max()*0.1 + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Average Waiting Vehicles per Direction", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_plot(fig, "averages.png")

    # Peak bar
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ZONES, stats["max"],
                  color=[ZONE_C[z] for z in ZONES],
                  edgecolor="white", width=0.55)
    for bar, val in zip(bars, stats["max"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(val)), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Peak Congestion per Direction", fontweight="bold")
    ax.set_ylabel("Max Vehicles")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_plot(fig, "peaks.png")

    config.OUTPUT_CSV_PATH = orig_csv

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or f.filename == "":
        return "No file selected.", 400

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(UPLOADS_DIR, "input_video.mp4")
    f.save(save_path)

    # Reset state
    state.update({"status": "idle", "frame_idx": 0,
                  "total_frames": 0, "counts": {z: 0 for z in WAITING_ZONES}})

    return redirect(url_for("processing"))


@app.route("/process")
def processing():
    if state["status"] == "idle":
        video_path = os.path.join(UPLOADS_DIR, "input_video.mp4")
        thread = threading.Thread(target=run_pipeline, args=(video_path,), daemon=True)
        thread.start()
    return render_template("processing.html")


@app.route("/status")
def status():
    """JSON endpoint polled by the frontend for live state."""
    progress = 0
    if state["total_frames"] > 0:
        progress = int(state["frame_idx"] / state["total_frames"] * 100)
    return jsonify({
        "status":    state["status"],
        "frame":     state["frame_idx"],
        "total":     state["total_frames"],
        "progress":  progress,
        "counts":    state["counts"],
        "error":     state["error"],
    })


@app.route("/results")
def results():
    if state["status"] != "done":
        return redirect(url_for("processing"))

    import pandas as pd
    df    = pd.read_csv(OUTPUT_CSV)
    zones = list(WAITING_ZONES.keys())
    summary = {z: {"avg": round(df[z].mean(), 2),
                   "max": int(df[z].max())} for z in zones}

    plots = ["timeseries.png", "averages.png", "peaks.png"]
    plots = [p for p in plots if os.path.exists(os.path.join(PLOTS_DIR, p))]

    return render_template("results.html",
                           summary=summary,
                           plots=plots,
                           total_frames=state["frame_idx"])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
