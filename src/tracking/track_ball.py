"""
Day 11 — Ball Tracking & Court-Space Trajectory.

Loads the raw ball detections produced by Day 10 (detect_ball.py), applies a
Kalman filter to smooth pixel-space positions and fill short gaps, projects the
smoothed positions through the homography matrix, flags "ball lost" segments,
and writes a static trajectory PNG plus an animated GIF to output/.

Input:  output/ball_positions.npz  (produced by detect_ball.py)
Output:
  output/ball_trajectory_smooth.png   — smoothed court-space trajectory
  output/ball_trajectory_anim.gif     — animated path on the court diagram

Run:
  python src/track_ball.py [--debug] [--calibrate] [--no-anim]
"""

import os
import sys
import argparse
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

# Allow running as  python src/tracking/track_ball.py  from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.calibrate import get_homography, apply_homography
from analysis.plot_utils import draw_court
from utils.video_utils import load_video
from config import (
    VIDEO_PATH, OUTPUT_DIR,
    COURT_WIDTH_M, COURT_LENGTH_M,
    KALMAN_GAP_FILL, BALL_FRAME_SKIP,
)
from tracking.detect_ball import BALL_POSITIONS_PATH

# ── Tunables ───────────────────────────────────────────────────────────────────
# Maximum number of consecutive missing frames that Kalman fill will bridge.
# Longer gaps are flagged as "ball lost".  Matches the Day 11 spec (5 frames).
_GAP_FILL_MAX = KALMAN_GAP_FILL      # default: 5 (set in config.py)

# Minimum gap length (in original frame indices) to label as a "ball lost" event
_LOST_MIN_FRAMES = _GAP_FILL_MAX + 1

# Animation: number of frames shown in the trailing "ghost" path
_TRAIL_LEN = 60

# Output paths
_SMOOTH_PNG  = os.path.join(OUTPUT_DIR, "ball_trajectory_smooth.png")
_ANIM_GIF    = os.path.join(OUTPUT_DIR, "ball_trajectory_anim.gif")
_SEGMENTS_TXT = os.path.join(OUTPUT_DIR, "ball_lost_segments.txt")


# ── Kalman filter ──────────────────────────────────────────────────────────────

def _build_kalman() -> cv2.KalmanFilter:
    """4-state (x, y, vx, vy) constant-velocity Kalman filter in pixel space."""
    kf = cv2.KalmanFilter(4, 2)

    # State transition: x' = x + vx, y' = y + vy, vx'=vx, vy'=vy
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # Measurement maps (x, y) from state
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    # Noise covariances — tuned for a fast-moving squash ball
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)

    return kf


def smooth_and_fill(frame_idx: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                    gap_fill_max: int = _GAP_FILL_MAX):
    """Apply Kalman filter and fill gaps up to gap_fill_max frames.

    Parameters
    ----------
    frame_idx : 1-D int array — original frame numbers of each detection
    xs, ys    : 1-D float arrays — pixel-space x/y of each detection
    gap_fill_max : int — maximum consecutive missing frames to fill

    Returns
    -------
    out_frame : 1-D int array — frame numbers of all output positions
    out_x, out_y : smoothed + filled pixel-space coordinates
    filled_mask  : bool array — True where the position was Kalman-filled (no raw detection)
    lost_segments : list of (start_frame, end_frame) tuples where ball was lost
    """
    if len(frame_idx) == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=bool),
                [])

    kf = _build_kalman()

    # Initialise state with the first detection
    kf.statePost = np.array([[xs[0]], [ys[0]], [0.0], [0.0]], dtype=np.float32)

    out_frame: list[int]   = []
    out_x:     list[float] = []
    out_y:     list[float] = []
    filled:    list[bool]  = []
    lost_segments: list[tuple[int, int]] = []

    prev_frame = frame_idx[0]

    for i, fi in enumerate(frame_idx):
        gap = fi - prev_frame if i > 0 else 0

        if gap > 1:
            # There is a gap between the previous detection and this one.
            if gap - 1 <= gap_fill_max:
                # Fill the gap with Kalman predictions
                for g in range(1, gap):
                    pred = kf.predict()
                    out_frame.append(prev_frame + g)
                    out_x.append(float(pred[0, 0]))
                    out_y.append(float(pred[1, 0]))
                    filled.append(True)
            else:
                # Gap too large — flag as "ball lost" and re-initialise
                lost_segments.append((prev_frame + 1, fi - 1))
                # Still predict forward so filter state stays warm
                for _ in range(min(gap - 1, 30)):
                    kf.predict()

        # Correct with actual measurement
        meas = np.array([[xs[i]], [ys[i]]], dtype=np.float32)
        kf.correct(meas)
        pred = kf.predict()

        out_frame.append(int(fi))
        out_x.append(float(pred[0, 0]))
        out_y.append(float(pred[1, 0]))
        filled.append(False)

        prev_frame = fi

    out_frame_arr = np.array(out_frame, dtype=int)
    out_x_arr     = np.array(out_x,     dtype=float)
    out_y_arr     = np.array(out_y,     dtype=float)
    filled_arr    = np.array(filled,    dtype=bool)

    return out_frame_arr, out_x_arr, out_y_arr, filled_arr, lost_segments


# ── Plotting ───────────────────────────────────────────────────────────────────

def _plot_static(xs_m: np.ndarray, ys_m: np.ndarray, filled: np.ndarray,
                 lost_segments: list) -> None:
    """Save a static PNG showing raw vs smoothed trajectory and lost zones."""
    fig, ax = plt.subplots(figsize=(7, 10))
    draw_court(ax)
    ax.set_title("Ball Trajectory — Kalman smoothed (Day 11)", fontsize=12, pad=10)

    # Draw lost-segment bands as semi-transparent rectangles spanning full width
    for start_f, end_f in lost_segments:
        # We can't easily shade by frame without mapping frame→court-y, so we
        # annotate with a text label placed at the top of the court instead.
        pass  # annotation is done textually in the saved .txt file

    raw_mask = ~filled
    if np.any(raw_mask):
        ax.scatter(xs_m[raw_mask], ys_m[raw_mask],
                   s=6, alpha=0.55, color="yellow", edgecolors="darkorange",
                   linewidths=0.4, label=f"Detected ({raw_mask.sum()})", zorder=5)

    if np.any(filled):
        ax.scatter(xs_m[filled], ys_m[filled],
                   s=6, alpha=0.45, color="deepskyblue", edgecolors="steelblue",
                   linewidths=0.4, label=f"Kalman filled ({filled.sum()})", zorder=4)

    # Draw trajectory line (connect consecutive points in time order)
    ax.plot(xs_m, ys_m, color="white", linewidth=0.5, alpha=0.3, zorder=3)

    ax.legend(loc="upper right", fontsize=8)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(_SMOOTH_PNG, dpi=150)
    print(f"Saved: {_SMOOTH_PNG}")
    plt.close(fig)


def _make_animation(xs_m: np.ndarray, ys_m: np.ndarray,
                    filled: np.ndarray) -> None:
    """Save an animated GIF of the ball tracing its path across the court."""
    # Downsample if too many points (GIF would be huge)
    step = max(1, len(xs_m) // 400)
    xs_s  = xs_m[::step]
    ys_s  = ys_m[::step]
    fill_s = filled[::step]
    n = len(xs_s)

    fig, ax = plt.subplots(figsize=(7, 10))
    draw_court(ax)
    ax.set_title("Ball trajectory (animated)", fontsize=12)

    trail_line, = ax.plot([], [], color="yellow", linewidth=2.0, alpha=0.8, zorder=4)
    ball_dot,   = ax.plot([], [], "o", color="yellow", markersize=10,
                          markeredgecolor="darkorange", markeredgewidth=1.5, zorder=5)
    fill_dot,   = ax.plot([], [], "o", color="deepskyblue", markersize=9,
                          markeredgecolor="steelblue", markeredgewidth=1.2, zorder=5)

    def _init():
        trail_line.set_data([], [])
        ball_dot.set_data([], [])
        fill_dot.set_data([], [])
        return trail_line, ball_dot, fill_dot

    def _update(i):
        start = max(0, i - _TRAIL_LEN)
        trail_line.set_data(xs_s[start:i], ys_s[start:i])

        if not fill_s[i]:
            ball_dot.set_data([xs_s[i]], [ys_s[i]])
            fill_dot.set_data([], [])
        else:
            ball_dot.set_data([], [])
            fill_dot.set_data([xs_s[i]], [ys_s[i]])

        return trail_line, ball_dot, fill_dot

    ani = animation.FuncAnimation(fig, _update, frames=n, init_func=_init,
                                  interval=80, blit=True)

    writer = animation.PillowWriter(fps=12)
    ani.save(_ANIM_GIF, writer=writer)
    print(f"Saved: {_ANIM_GIF}")
    plt.close(fig)


def _write_lost_segments(lost_segments: list, fps: float, frame_skip: int) -> None:
    """Write a human-readable log of ball-lost events."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    effective_fps = fps / frame_skip
    with open(_SEGMENTS_TXT, "w") as f:
        if not lost_segments:
            f.write("No ball-lost segments detected.\n")
        else:
            f.write(f"{'Start frame':>12}  {'End frame':>10}  {'Duration (frames)':>18}  {'Duration (s)':>12}\n")
            f.write("-" * 60 + "\n")
            for s, e in lost_segments:
                dur_f = e - s + 1
                dur_s = dur_f / effective_fps
                f.write(f"{s:>12}  {e:>10}  {dur_f:>18}  {dur_s:>12.2f}\n")
    print(f"Lost segments: {_SEGMENTS_TXT}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(debug: bool = False, calibrate: bool = False, make_anim: bool = True) -> None:
    # Load raw detections
    if not os.path.exists(BALL_POSITIONS_PATH):
        print(f"[Error] Ball positions file not found: {BALL_POSITIONS_PATH}")
        print("Run detect_ball.py first to generate it.")
        return

    data = np.load(BALL_POSITIONS_PATH)
    frame_idx = data["frame_idx"].astype(int)
    xs_px     = data["xs"].astype(float)
    ys_px     = data["ys"].astype(float)

    print(f"Loaded {len(frame_idx)} raw detections from {BALL_POSITIONS_PATH}")

    if len(frame_idx) == 0:
        print("[Error] No ball detections to process.")
        return

    # Sort by frame index (should already be sorted, but be safe)
    order     = np.argsort(frame_idx)
    frame_idx = frame_idx[order]
    xs_px     = xs_px[order]
    ys_px     = ys_px[order]

    # ── Kalman filter ───────────────────────────────────────────────────────────
    print(f"Applying Kalman filter (gap fill ≤ {_GAP_FILL_MAX} frames) …")
    sm_frames, sm_xs_px, sm_ys_px, filled, lost_segs = smooth_and_fill(
        frame_idx, xs_px, ys_px, gap_fill_max=_GAP_FILL_MAX
    )

    n_filled = int(filled.sum())
    n_lost   = len(lost_segs)
    print(f"  Output positions : {len(sm_frames)}  (original {len(frame_idx)} + {n_filled} filled)")
    print(f"  Ball-lost events : {n_lost}" + (" (none)" if n_lost == 0 else ""))
    for s, e in lost_segs:
        print(f"    frames {s} – {e}  ({e - s + 1} frames)")

    # ── Project through homography ──────────────────────────────────────────────
    # Load first frame just to get/confirm homography
    cap, detected_fps, _ = load_video(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame for homography.")

    H = get_homography(first_frame, force_recalibrate=calibrate)

    sm_xs_m_list, sm_ys_m_list = apply_homography(sm_xs_px.tolist(), sm_ys_px.tolist(), H)
    sm_xs_m = np.array(sm_xs_m_list, dtype=float)
    sm_ys_m = np.array(sm_ys_m_list, dtype=float)

    # Mask out any positions that projected outside sensible court bounds
    in_bounds = (
        (sm_xs_m >= -1.0) & (sm_xs_m <= COURT_WIDTH_M  + 1.0) &
        (sm_ys_m >= -1.0) & (sm_ys_m <= COURT_LENGTH_M + 1.0) &
        np.isfinite(sm_xs_m) & np.isfinite(sm_ys_m)
    )
    if not np.all(in_bounds):
        n_oob = int((~in_bounds).sum())
        print(f"  Clamped {n_oob} out-of-bounds smoothed positions.")
        sm_xs_m = sm_xs_m[in_bounds]
        sm_ys_m = sm_ys_m[in_bounds]
        filled  = filled[in_bounds]

    # ── Save smoothed positions ─────────────────────────────────────────────────
    from config import VIDEO_FPS

    smooth_path = os.path.join(OUTPUT_DIR, "ball_positions_smooth.npz")
    np.savez(
        smooth_path,
        frame_idx = sm_frames[in_bounds] if not np.all(in_bounds) else sm_frames,
        xs_m      = sm_xs_m,
        ys_m      = sm_ys_m,
        filled    = filled.astype(np.uint8),
    )
    print(f"Saved smoothed positions: {smooth_path}")

    # ── Outputs ─────────────────────────────────────────────────────────────────
    _write_lost_segments(lost_segs, fps=VIDEO_FPS, frame_skip=BALL_FRAME_SKIP)
    _plot_static(sm_xs_m, sm_ys_m, filled, lost_segs)

    if make_anim:
        print("Generating animation (this may take ~10–20 s) …")
        _make_animation(sm_xs_m, sm_ys_m, filled)

    print("\nDay 11 complete.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description="Day 11 — Ball Tracking & Court-Space Trajectory")
    parser.add_argument("--debug",     action="store_true", help="Extra debug output.")
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration of court homography.")
    parser.add_argument("--no-anim",   action="store_true", help="Skip GIF animation (faster).")
    args = parser.parse_args()

    main(debug=args.debug, calibrate=args.calibrate, make_anim=not args.no_anim)
