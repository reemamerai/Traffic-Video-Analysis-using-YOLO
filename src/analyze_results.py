# src/analyze_results.py
# Generates advanced visual analytics from waiting_counts.csv.
#
# Run from the project root:
#   python src/analyze_results.py
#
# Outputs saved to: outputs/plots/

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import config

# ── Constants ─────────────────────────────────────────────────────────────────

CSV_PATH  = os.path.join(PROJECT_ROOT, config.OUTPUT_CSV_PATH)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
ZONES     = ["N_WAIT", "S_WAIT", "E_WAIT", "W_WAIT"]

ZONE_COLORS = {
    "N_WAIT": "#4A90D9",
    "S_WAIT": "#E8A838",
    "E_WAIT": "#4CAF50",
    "W_WAIT": "#9C5DDB",
}

SMOOTH_WINDOW = 30   # frames for moving-average smoothing

# Global style
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#CCCCCC",
    "grid.color":       "#E0E0E0",
    "grid.linestyle":   "--",
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, filename: str):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def to_time_str(frame_idx: int, fps: float) -> str:
    total_sec = frame_idx / fps if fps > 0 else frame_idx
    m, s = divmod(int(total_sec), 60)
    return f"{m:02d}:{s:02d}" if fps > 0 else f"frame {frame_idx}"


def annotate_peak(ax, x_vals, y_vals, color: str):
    """Place a dot + label at the maximum point of a series."""
    peak_idx = y_vals.idxmax()
    px = x_vals.iloc[peak_idx]
    py = y_vals.iloc[peak_idx]
    ax.scatter(px, py, color=color, s=60, zorder=5)
    ax.annotate(
        f"  peak: {int(py)}",
        xy=(px, py),
        xytext=(px, py + 0.6),
        fontsize=9,
        color=color,
        fontweight="bold",
    )

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_timeseries(df: pd.DataFrame, x_col: str, x_label: str):
    """Smoothed time-series with raw data faintly in the background."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_facecolor("#FAFAFA")

    for zone in ZONES:
        color = ZONE_COLORS[zone]
        raw    = df[zone]
        smooth = raw.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()

        # Faint raw signal
        ax.plot(df[x_col], raw, color=color, linewidth=0.8, alpha=0.25)
        # Bold smoothed signal
        ax.plot(df[x_col], smooth, color=color, linewidth=2.2, alpha=0.95, label=zone)
        # Annotate peak on smoothed
        annotate_peak(ax, df[x_col], smooth, color)

    ax.set_title("Waiting Vehicle Counts Over Time  (smoothed)", fontweight="bold", pad=16)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Vehicles in Waiting Zone")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    save(fig, "timeseries_smoothed.png")


def plot_stats_bar(stats: pd.DataFrame):
    """Grouped bar chart: average + std-dev error bars."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ZONES))
    bars = ax.bar(
        x,
        stats["mean"],
        yerr=stats["std"],
        color=[ZONE_COLORS[z] for z in ZONES],
        edgecolor="white",
        width=0.55,
        capsize=6,
        error_kw={"elinewidth": 1.8, "ecolor": "#555555"},
    )

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["std"] + 0.15,
            f"μ={row['mean']:.1f}\nσ={row['std']:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(ZONES, fontsize=13)
    ax.set_title("Average Waiting Vehicles per Direction  (± std dev)", fontweight="bold", pad=16)
    ax.set_xlabel("Waiting Zone")
    ax.set_ylabel("Vehicle Count")
    ax.set_ylim(0, (stats["mean"] + stats["std"]).max() * 1.35)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "average_stddev.png")


def plot_peak_bar(stats: pd.DataFrame, peak_times: dict):
    """Peak congestion per direction with time annotation."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ZONES))
    bars = ax.bar(
        x,
        stats["max"],
        color=[ZONE_COLORS[z] for z in ZONES],
        edgecolor="white",
        width=0.55,
    )

    for bar, zone in zip(bars, ZONES):
        peak_val  = int(stats.loc[zone, "max"])
        time_str  = peak_times[zone]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{peak_val}\n@ {time_str}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(ZONES, fontsize=13)
    ax.set_title("Peak Congestion per Direction", fontweight="bold", pad=16)
    ax.set_xlabel("Waiting Zone")
    ax.set_ylabel("Maximum Vehicle Count")
    ax.set_ylim(0, stats["max"].max() * 1.3)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "peak_congestion.png")


def plot_heatmap(df: pd.DataFrame):
    """Zone-vs-zone correlation heatmap."""
    import numpy as np
    corr = df[ZONES].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(range(len(ZONES)))
    ax.set_yticks(range(len(ZONES)))
    ax.set_xticklabels(ZONES, rotation=30, ha="right")
    ax.set_yticklabels(ZONES)

    for i in range(len(ZONES)):
        for j in range(len(ZONES)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Zone Congestion Correlation", fontweight="bold", pad=14)
    fig.tight_layout()
    save(fig, "correlation_heatmap.png")

# ── Report ────────────────────────────────────────────────────────────────────

def print_report(stats: pd.DataFrame, peak_times: dict):
    busiest_zone  = stats["mean"].idxmax()
    most_stable   = stats["std"].idxmin()
    overall_peak  = stats["max"].idxmax()

    print()
    print("=" * 56)
    print("   TRAFFIC ANALYSIS REPORT")
    print("=" * 56)
    print(f"  {'Zone':<10} {'Avg':>6}  {'Max':>5}  {'Std':>6}  {'Peak at':>8}")
    print(f"  {'-'*10} {'-'*6}  {'-'*5}  {'-'*6}  {'-'*8}")
    for zone in ZONES:
        r = stats.loc[zone]
        print(f"  {zone:<10} {r['mean']:>6.2f}  {int(r['max']):>5}  "
              f"{r['std']:>6.2f}  {peak_times[zone]:>8}")
    print()
    print(f"  Most congested  : {busiest_zone}  (highest average: {stats.loc[busiest_zone,'mean']:.2f})")
    print(f"  Most stable     : {most_stable}  (lowest std dev: {stats.loc[most_stable,'std']:.2f})")
    print(f"  Peak moment     : {overall_peak} reached {int(stats.loc[overall_peak,'max'])} vehicles "
          f"at {peak_times[overall_peak]}")
    print("=" * 56)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df  = pd.read_csv(CSV_PATH)
    fps = float(getattr(config, "VIDEO_FPS", 0))

    if fps > 0:
        df["time_sec"] = df["frame_index"] / fps
        x_col, x_label = "time_sec", "Time (seconds)"
    else:
        x_col, x_label = "frame_index", "Frame"

    print(f"Loaded {len(df)} rows  |  {x_label} axis  |  smooth window: {SMOOTH_WINDOW}\n")

    # Compute stats
    stats = df[ZONES].agg(["mean", "max", "std"]).T
    peak_frames = {zone: int(df[zone].idxmax()) for zone in ZONES}
    peak_times  = {zone: to_time_str(peak_frames[zone], fps) for zone in ZONES}

    # Generate charts
    plot_timeseries(df, x_col, x_label)
    plot_stats_bar(stats)
    plot_peak_bar(stats, peak_times)
    plot_heatmap(df)

    print_report(stats, peak_times)
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
