"""
Day 13 — Rally Segmentation & Combined Analysis.

Loads raw ball detections and player positions, segments the clip into
individual rallies based on ball-lost gaps, computes per-rally stats for both
players, and generates a combined court diagram overlaying player movement
with the ball trajectory coloured by rally.

Input:
  output/ball_positions.npz        — raw ball detections (detect_ball.py)
  output/last_positions_yolo.npz   — raw player positions (extract_pose_yolo.py)
  assets/homography.npy            — court homography (calibrate.py)

Output:
  output/rally_boundaries.csv  — rally_id, start_frame, end_frame, duration_s
  output/rally_stats.csv       — per-rally: duration_s, shots, dist_p1_m, dist_p2_m
  output/combined_court.png    — player positions + ball trajectory, coloured by rally
  output/rally_timeline.png    — horizontal bar chart of rally durations

Run:
  python src/segment_rallies.py [--calibrate] [--min-gap N]
"""

import os
import sys
import argparse
import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.calibrate import get_homography, apply_homography
from utils.video_utils import load_video
from analysis.plot_utils import draw_court
from config import (
    VIDEO_PATH, VIDEO_FPS, OUTPUT_DIR,
    FRAME_SKIP, SMOOTH_WINDOW,
    RALLY_END_MIN_FRAMES,
    PLAYER_COLORS, PLAYER_LABELS,
)
from tracking.detect_ball import BALL_POSITIONS_PATH

_PLAYER_POSITIONS_PATH = os.path.join(OUTPUT_DIR, "last_positions_yolo.npz")
_BOUNDARIES_CSV  = os.path.join(OUTPUT_DIR, "rally_boundaries.csv")
_STATS_CSV       = os.path.join(OUTPUT_DIR, "rally_stats.csv")
_COMBINED_PNG    = os.path.join(OUTPUT_DIR, "combined_court.png")
_TIMELINE_PNG    = os.path.join(OUTPUT_DIR, "rally_timeline.png")


# ── Rally segmentation ─────────────────────────────────────────────────────────

def segment_rallies(frame_idx: np.ndarray, min_gap_frames: int = RALLY_END_MIN_FRAMES):
    """Split a ball detection sequence into rallies based on gap size.

    A consecutive gap >= min_gap_frames between detections marks an inter-rally
    boundary.  Everything between two such gaps is one rally.

    Parameters
    ----------
    frame_idx       : 1-D int array of frame numbers for each ball detection
    min_gap_frames  : minimum gap to count as an inter-rally boundary

    Returns
    -------
    list of (rally_id, start_frame, end_frame)  — 1-indexed, inclusive
    """
    if len(frame_idx) < 2:
        return []

    rallies = []
    rally_id = 1
    rally_start = int(frame_idx[0])

    for i in range(1, len(frame_idx)):
        gap = int(frame_idx[i]) - int(frame_idx[i - 1])
        if gap >= min_gap_frames:
            rallies.append((rally_id, rally_start, int(frame_idx[i - 1])))
            rally_id += 1
            rally_start = int(frame_idx[i])

    # Final rally
    rallies.append((rally_id, rally_start, int(frame_idx[-1])))
    return rallies


# ── Per-rally ball stats ───────────────────────────────────────────────────────

def _count_shots(xs_m: np.ndarray, ys_m: np.ndarray) -> int:
    """Approximate shot count by counting velocity direction reversals.

    A reversal is when consecutive velocity vectors have a negative dot product
    (the ball turned around).  Over-counts slightly on noisy frames, but is a
    consistent relative measure.
    """
    if len(xs_m) < 3:
        return 0
    dx = np.diff(xs_m)
    dy = np.diff(ys_m)
    dots = dx[:-1] * dx[1:] + dy[:-1] * dy[1:]
    return int(np.sum(dots < 0))


def _ball_stats_for_rally(start_f, end_f, ball_frames, ball_xs_m, ball_ys_m):
    """Extract ball detections within [start_f, end_f] and compute stats."""
    mask = (ball_frames >= start_f) & (ball_frames <= end_f)
    rxs = ball_xs_m[mask]
    rys = ball_ys_m[mask]
    n_shots = _count_shots(rxs, rys)
    return int(mask.sum()), n_shots, rxs, rys


# ── Per-rally player stats ─────────────────────────────────────────────────────

def _player_distance_for_rally(court_xs, court_ys, start_f, end_f, frame_skip):
    """Approximate court-space distance a player covered during a rally.

    Player position index i corresponds to roughly frame i * frame_skip.
    """
    i_start = max(0, start_f // frame_skip)
    i_end   = min(len(court_xs) - 1, end_f // frame_skip)
    if i_start >= i_end:
        return 0.0
    xs = np.array(court_xs[i_start : i_end + 1], dtype=float)
    ys = np.array(court_ys[i_start : i_end + 1], dtype=float)
    dists = np.hypot(np.diff(xs), np.diff(ys))
    return float(np.nansum(dists))


# ── CSV export ─────────────────────────────────────────────────────────────────

def _write_boundaries(rallies, fps):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(_BOUNDARIES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rally_id", "start_frame", "end_frame", "duration_s"])
        for rid, sf, ef in rallies:
            w.writerow([rid, sf, ef, f"{(ef - sf) / fps:.2f}"])
    print(f"Saved: {_BOUNDARIES_CSV}")


def _write_rally_stats(rally_stats_rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(_STATS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rally_id", "start_frame", "end_frame",
                    "duration_s", "n_ball_detections", "approx_shots",
                    "dist_p1_m", "dist_p2_m"])
        for row in rally_stats_rows:
            w.writerow([
                row["rally_id"], row["start_frame"], row["end_frame"],
                f"{row['duration_s']:.2f}",
                row["n_ball_det"],
                row["approx_shots"],
                f"{row['dist_p1_m']:.2f}",
                f"{row['dist_p2_m']:.2f}",
            ])
    print(f"Saved: {_STATS_CSV}")


# ── Combined court plot ─────────────────────────────────────────────────────────

def plot_combined_court(court_xs1, court_ys1, court_xs2, court_ys2,
                        ball_frames, ball_xs_m, ball_ys_m, rallies):
    """Overlay player positions + ball trajectory (coloured by rally) on one court."""
    fig, ax = plt.subplots(figsize=(7, 10))
    draw_court(ax)
    ax.set_title("Combined Court — Players & Ball Trajectory by Rally", fontsize=11, pad=10)

    # Player positions — small translucent dots
    if len(court_xs1) > 0:
        ax.scatter(court_xs1, court_ys1, s=4, alpha=0.18, color=PLAYER_COLORS[0],
                   zorder=3, label=PLAYER_LABELS[0])
    if len(court_xs2) > 0:
        ax.scatter(court_xs2, court_ys2, s=4, alpha=0.18, color=PLAYER_COLORS[1],
                   zorder=3, label=PLAYER_LABELS[1])

    # Ball trajectory — one scatter per rally, coloured by rally number
    n_rallies = len(rallies)
    cmap = cm.get_cmap("tab20", max(n_rallies, 1))

    for i, (rid, sf, ef) in enumerate(rallies):
        mask = (ball_frames >= sf) & (ball_frames <= ef)
        rxs = ball_xs_m[mask]
        rys = ball_ys_m[mask]
        if len(rxs) == 0:
            continue
        color = cmap(i / max(n_rallies - 1, 1))
        ax.scatter(rxs, rys, s=7, alpha=0.7, color=color, zorder=5)
        # Label rally number at the midpoint of the ball detections
        mid = len(rxs) // 2
        ax.text(rxs[mid], rys[mid], str(rid), fontsize=6, color=color,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.5, lw=0))

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=PLAYER_COLORS[0], alpha=0.6, label=PLAYER_LABELS[0]),
        mpatches.Patch(color=PLAYER_COLORS[1], alpha=0.6, label=PLAYER_LABELS[1]),
        mpatches.Patch(color="grey", alpha=0.6, label=f"Ball ({n_rallies} rallies, coloured)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(_COMBINED_PNG, dpi=150)
    print(f"Saved: {_COMBINED_PNG}")
    plt.close(fig)


# ── Rally timeline plot ─────────────────────────────────────────────────────────

def plot_rally_timeline(rally_stats_rows, fps):
    """Horizontal bar chart: one bar per rally showing its time span and duration."""
    if not rally_stats_rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(rally_stats_rows) * 0.35 + 1)))

    durations = [r["duration_s"] for r in rally_stats_rows]
    starts    = [r["start_frame"] / fps for r in rally_stats_rows]
    rids      = [r["rally_id"] for r in rally_stats_rows]

    cmap = cm.get_cmap("tab20", max(len(rids), 1))
    for i, (rid, s, d) in enumerate(zip(rids, starts, durations)):
        color = cmap(i / max(len(rids) - 1, 1))
        ax.barh(rid, d, left=s, height=0.6, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.text(s + d + 0.1, rid, f"{d:.1f}s", va="center", fontsize=7, color="black")

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Rally #", fontsize=10)
    ax.set_title("Rally Timeline", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(_TIMELINE_PNG, dpi=150)
    print(f"Saved: {_TIMELINE_PNG}")
    plt.close(fig)


# ── Console summary ────────────────────────────────────────────────────────────

def _print_summary(rally_stats_rows, fps):
    if not rally_stats_rows:
        print("No rallies detected.")
        return

    durations = [r["duration_s"] for r in rally_stats_rows]
    shots     = [r["approx_shots"] for r in rally_stats_rows]
    dp1       = [r["dist_p1_m"] for r in rally_stats_rows]
    dp2       = [r["dist_p2_m"] for r in rally_stats_rows]

    print(f"\n{'─' * 70}")
    print(f"  Rally Segmentation  —  {len(rally_stats_rows)} rallies detected")
    print(f"{'─' * 70}")
    print(f"  {'Rally':>6}  {'Start(s)':>8}  {'End(s)':>7}  {'Dur(s)':>7}  "
          f"{'Shots':>6}  {'P1 dist':>8}  {'P2 dist':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*8}")
    for r in rally_stats_rows:
        print(f"  {r['rally_id']:>6}  "
              f"{r['start_frame']/fps:>8.1f}  "
              f"{r['end_frame']/fps:>7.1f}  "
              f"{r['duration_s']:>7.1f}  "
              f"{r['approx_shots']:>6}  "
              f"{r['dist_p1_m']:>7.2f}m  "
              f"{r['dist_p2_m']:>7.2f}m")
    print(f"{'─' * 70}")
    print(f"  Mean rally duration : {np.mean(durations):.1f}s  "
          f"  Longest: {max(durations):.1f}s  "
          f"  Shortest: {min(durations):.1f}s")
    print(f"  Mean approx shots   : {np.mean(shots):.1f}")
    print(f"  Total P1 distance   : {sum(dp1):.1f}m")
    print(f"  Total P2 distance   : {sum(dp2):.1f}m")
    print(f"{'─' * 70}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(calibrate=False, min_gap=RALLY_END_MIN_FRAMES):
    # ── Load ball positions ────────────────────────────────────────────────────
    if not os.path.exists(BALL_POSITIONS_PATH):
        print(f"[Error] Ball positions not found: {BALL_POSITIONS_PATH}")
        print("Run detect_ball.py first.")
        return

    ball_data   = np.load(BALL_POSITIONS_PATH)
    ball_frames = ball_data["frame_idx"].astype(int)
    ball_xs_m   = ball_data["xs_m"].astype(float)
    ball_ys_m   = ball_data["ys_m"].astype(float)

    order       = np.argsort(ball_frames)
    ball_frames = ball_frames[order]
    ball_xs_m   = ball_xs_m[order]
    ball_ys_m   = ball_ys_m[order]

    print(f"Loaded {len(ball_frames)} raw ball detections.")

    # ── Load player positions ──────────────────────────────────────────────────
    if not os.path.exists(_PLAYER_POSITIONS_PATH):
        print(f"[Warning] Player positions not found: {_PLAYER_POSITIONS_PATH}")
        print("Run extract_pose_yolo.py first.  Player distance per rally will be 0.")
        xs1 = ys1 = xs2 = ys2 = np.array([])
    else:
        pdata = np.load(_PLAYER_POSITIONS_PATH)
        xs1, ys1 = pdata["xs1"].astype(float), pdata["ys1"].astype(float)
        xs2, ys2 = pdata["xs2"].astype(float), pdata["ys2"].astype(float)
        print(f"Loaded player positions: P1={len(xs1)}, P2={len(xs2)} frames.")

    # ── Homography ────────────────────────────────────────────────────────────
    cap, _, _ = load_video(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first video frame.")

    H = get_homography(first_frame, force_recalibrate=calibrate)

    # ── Smooth + project player positions ─────────────────────────────────────
    def _smooth_project(xs, ys):
        if len(xs) < 2:
            return np.array([]), np.array([])
        xs_s = median_filter(xs, size=SMOOTH_WINDOW)
        ys_s = median_filter(ys, size=SMOOTH_WINDOW)
        cxs, cys = apply_homography(xs_s.tolist(), ys_s.tolist(), H)
        return np.array(cxs, dtype=float), np.array(cys, dtype=float)

    court_xs1, court_ys1 = _smooth_project(xs1, ys1)
    court_xs2, court_ys2 = _smooth_project(xs2, ys2)

    # ── Segment rallies ────────────────────────────────────────────────────────
    rallies = segment_rallies(ball_frames, min_gap_frames=min_gap)
    print(f"Detected {len(rallies)} rallies  (min gap threshold = {min_gap} frames = "
          f"{min_gap / VIDEO_FPS:.2f}s)")

    # ── Per-rally stats ────────────────────────────────────────────────────────
    rally_stats_rows = []
    for rid, sf, ef in rallies:
        duration_s  = (ef - sf) / VIDEO_FPS
        n_ball, shots, _, _ = _ball_stats_for_rally(sf, ef, ball_frames, ball_xs_m, ball_ys_m)
        dist_p1 = _player_distance_for_rally(court_xs1, court_ys1, sf, ef, FRAME_SKIP)
        dist_p2 = _player_distance_for_rally(court_xs2, court_ys2, sf, ef, FRAME_SKIP)
        rally_stats_rows.append({
            "rally_id": rid, "start_frame": sf, "end_frame": ef,
            "duration_s": duration_s, "n_ball_det": n_ball,
            "approx_shots": shots, "dist_p1_m": dist_p1, "dist_p2_m": dist_p2,
        })

    # ── Outputs ────────────────────────────────────────────────────────────────
    _print_summary(rally_stats_rows, VIDEO_FPS)
    _write_boundaries(rallies, VIDEO_FPS)
    _write_rally_stats(rally_stats_rows)

    plot_combined_court(court_xs1, court_ys1, court_xs2, court_ys2,
                        ball_frames, ball_xs_m, ball_ys_m, rallies)
    plot_rally_timeline(rally_stats_rows, VIDEO_FPS)

    print("\nDay 13 complete.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description="Day 13 — Rally Segmentation & Combined Analysis")
    parser.add_argument("--calibrate", action="store_true",
                        help="Force recalibration of court homography.")
    parser.add_argument("--min-gap", type=int, default=RALLY_END_MIN_FRAMES,
                        dest="min_gap",
                        help=f"Minimum ball-lost gap (frames) to mark a rally boundary. "
                             f"Default: {RALLY_END_MIN_FRAMES}")
    args = parser.parse_args()
    main(calibrate=args.calibrate, min_gap=args.min_gap)
