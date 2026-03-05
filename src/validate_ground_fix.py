"""
validate_ground_fix.py

Objectively compares hip-based (old) vs heel/ankle-based (new) player position
detection on a sample of frames from the configured video.

Outputs
-------
    output/validation/court_comparison.png   — scatter plots side by side
    output/validation/y_timeseries.png       — Y position over time (jump spikes visible)
    Console metrics table

Run from src/:
    python validate_ground_fix.py [--frames N]
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

from video_utils import load_video
from calibrate import get_homography, apply_homography
from plot_utils import draw_court, plot_heatmap_comparison
from config import (
    VIDEO_PATH, VIDEO_FPS, FRAME_SKIP, CROP_MARGIN,
    MODEL_COMPLEXITY, OUTPUT_DIR, FOOT_VISIBILITY_MIN,
)

VALIDATION_DIR = os.path.join(OUTPUT_DIR, "validation")
_FALLBACK_VIS  = 0.35
mp_pose = mp.solutions.pose

_pose_kwargs = dict(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ── Landmark extractors ───────────────────────────────────────────────────────

def _midpoint(landmarks, w, h, threshold):
    """Return (x, y) midpoint of all landmarks whose visibility >= threshold."""
    visible = [lm for lm in landmarks if lm.visibility >= threshold]
    if not visible:
        return None
    return (
        sum(lm.x for lm in visible) / len(visible) * w,
        sum(lm.y for lm in visible) / len(visible) * h,
    )


def hip_position(landmarks, w, h):
    """Old method: hip midpoint only."""
    lms = [landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP]]
    return _midpoint(lms, w, h, _FALLBACK_VIS)


def ground_position(landmarks, w, h):
    """New method: heels → any lower-body → hip fallback."""
    lheel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    rheel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
    lank  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    rank  = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Tier 0: both heels confident
    if lheel.visibility >= FOOT_VISIBILITY_MIN and rheel.visibility >= FOOT_VISIBILITY_MIN:
        return ((lheel.x + rheel.x) / 2 * w, (lheel.y + rheel.y) / 2 * h)

    # Tier 1: any heel or ankle visible
    lower = [lm for lm in (lheel, rheel, lank, rank) if lm.visibility >= _FALLBACK_VIS]
    if lower:
        return _midpoint(lower, w, h, _FALLBACK_VIS)

    # Tier 2: hip fallback
    return hip_position(landmarks, w, h)


# ── Detection helper ──────────────────────────────────────────────────────────

def detect(frame, last_pos, pose_model, landmark_fn):
    """Detect player in a crop; return position in full-frame coords or None."""
    fh, fw = frame.shape[:2]
    cx, cy = int(last_pos[0]), int(last_pos[1])
    for margin in (CROP_MARGIN, CROP_MARGIN * 2):
        x1 = max(0, cx - margin); y1 = max(0, cy - margin)
        x2 = min(fw, cx + margin); y2 = min(fh, cy + margin)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)
        if res.pose_landmarks:
            pos = landmark_fn(res.pose_landmarks.landmark, crop.shape[1], crop.shape[0])
            if pos is not None:
                return pos[0] + x1, pos[1] + y1
    return None


def auto_detect(frame):
    """Rough initial positions by splitting frame into top/bottom halves."""
    h, w = frame.shape[:2]
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                        min_detection_confidence=0.4)
    positions = []
    for y0, y1 in [(0, h // 2), (h // 2, h)]:
        crop = frame[y0:y1, :]
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res  = pose.process(rgb)
        if res.pose_landmarks:
            pos = hip_position(res.pose_landmarks.landmark, w, y1 - y0)
            if pos:
                positions.append((pos[0], pos[1] + y0))
    pose.close()
    if len(positions) == 2:
        return positions[0], positions[1]
    elif len(positions) == 1:
        other_y = 3 * h // 4 if positions[0][1] < h // 2 else h // 4
        return positions[0], (w // 2, other_y)
    return (w // 2, h // 4), (w // 2, 3 * h // 4)


# ── Metrics ───────────────────────────────────────────────────────────────────

def count_jump_artifacts(hip_y, grnd_y, move_thresh=0.3, stable_thresh=0.1):
    """Frames where hip Y changed > move_thresh m but ground Y stayed < stable_thresh m.

    These are frames the hip method counted as court movement that the ground
    method correctly identified as vertical (jump/tiptoe) motion.
    """
    dh = np.abs(np.diff(hip_y))
    dg = np.abs(np.diff(grnd_y))
    return int(np.sum((dh > move_thresh) & (dg < stable_thresh)))


def print_metrics(label, hip_y, grnd_y):
    mean_h, mean_g   = np.mean(hip_y),  np.mean(grnd_y)
    std_h,  std_g    = np.std(hip_y),   np.std(grnd_y)
    jitter_change    = (std_g - std_h) / std_h * 100 if std_h > 0 else 0
    bias_correction  = mean_g - mean_h          # +ve = moved toward back wall (correct direction)
    artifacts        = count_jump_artifacts(hip_y, grnd_y)

    col = 22
    print(f"\n  ── {label} ──")
    print(f"  {'Metric':<{col}} {'Hip (old)':>{col}} {'Heel/Ankle (new)':>{col}}")
    print(f"  {'-'*66}")
    print(f"  {'Mean court Y (m)':<{col}} {mean_h:>{col}.2f} {mean_g:>{col}.2f}  "
            f"(Δ {bias_correction:+.2f} m {'toward back ✓' if bias_correction > 0 else 'toward front'})")
    print(f"  {'Y std dev / jitter':<{col}} {std_h:>{col}.3f} {std_g:>{col}.3f}  "
            f"({jitter_change:+.1f}%)")
    print(f"  {'Jump artifacts':<{col}} {'—':>{col}} {artifacts:>{col}}  "
            f"frames where hip moved but feet didn't")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(sample_frames):
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    cap, _, _ = load_video(VIDEO_PATH)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    H = get_homography(first_frame, force_recalibrate=False)
    init1, init2 = auto_detect(first_frame)
    print(f"Initial positions: P1={init1}, P2={init2}")

    # Four independent pose models: hip×2 + ground×2
    # (separate models so each method's temporal state stays isolated)
    ph1, ph2 = mp_pose.Pose(**_pose_kwargs), mp_pose.Pose(**_pose_kwargs)
    pg1, pg2 = mp_pose.Pose(**_pose_kwargs), mp_pose.Pose(**_pose_kwargs)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lp_h1, lp_h2 = init1, init2   # last positions for hip method
    lp_g1, lp_g2 = init1, init2   # last positions for ground method

    hip_px1,  hip_px2  = [], []
    grnd_px1, grnd_px2 = [], []
    frame_idx = 0

    with tqdm(total=sample_frames, unit="frame", desc="Running comparison") as pbar:
        while cap.isOpened() and frame_idx < sample_frames:
            for _ in range(FRAME_SKIP - 1):
                if not cap.grab():
                    break
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += FRAME_SKIP

            # ── Hip method ──────────────────────────────────────────────────
            p = detect(frame, lp_h1, ph1, hip_position)
            if p is not None:
                lp_h1 = p
            hip_px1.append(lp_h1)

            p = detect(frame, lp_h2, ph2, hip_position)
            if p is not None:
                lp_h2 = p
            hip_px2.append(lp_h2)

            # ── Ground method ────────────────────────────────────────────────
            p = detect(frame, lp_g1, pg1, ground_position)
            if p is not None:
                lp_g1 = p
            grnd_px1.append(lp_g1)

            p = detect(frame, lp_g2, pg2, ground_position)
            if p is not None:
                lp_g2 = p
            grnd_px2.append(lp_g2)

            pbar.update(FRAME_SKIP)

    cap.release()
    for m in (ph1, ph2, pg1, pg2):
        m.close()

    # ── Convert pixel → court coordinates ────────────────────────────────────
    def to_court(px_list):
        xs, ys = zip(*px_list)
        cx, cy = apply_homography(list(xs), list(ys), H)
        return np.array(cx), np.array(cy)

    hxs1, hys1 = to_court(hip_px1)
    hxs2, hys2 = to_court(hip_px2)
    gxs1, gys1 = to_court(grnd_px1)
    gxs2, gys2 = to_court(grnd_px2)

    # ── Metrics table ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  GROUND POSITION FIX — VALIDATION RESULTS")
    print("=" * 70)
    print_metrics("Player 1", hys1, gys1)
    print_metrics("Player 2", hys2, gys2)
    print()

    # ── Plot 1: Court scatter comparison ─────────────────────────────────────
    fig, (ax_hip, ax_grnd) = plt.subplots(1, 2, figsize=(14, 9))
    for ax, xs1, ys1, xs2, ys2, title in [
        (ax_hip,  hxs1, hys1, hxs2, hys2, "Hip method (old)"),
        (ax_grnd, gxs1, gys1, gxs2, gys2, "Heel/Ankle method (new)"),
    ]:
        draw_court(ax)
        ax.scatter(xs1, ys1, s=4, alpha=0.4, color="red",        label="Player 1")
        ax.scatter(xs2, ys2, s=4, alpha=0.4, color="dodgerblue", label="Player 2")
        ax.set_title(title)
        ax.legend(loc="upper right")
    fig.suptitle("Court Position: Hip vs Heel/Ankle landmark")
    plt.tight_layout()
    path = os.path.join(VALIDATION_DIR, "court_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Plot 2: Y time series — makes jump spikes visible ────────────────────
    t = np.arange(len(hys1)) * FRAME_SKIP / VIDEO_FPS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, hy, gy, label in [
        (axes[0], hys1, gys1, "Player 1"),
        (axes[1], hys2, gys2, "Player 2"),
    ]:
        ax.plot(t, hy, alpha=0.7, lw=0.9, color="coral",     label="Hip (old)")
        ax.plot(t, gy, alpha=0.7, lw=0.9, color="steelblue", label="Heel/Ankle (new)")
        ax.set_ylabel("Court Y (m)")
        ax.set_title(label)
        ax.legend(loc="upper right")
        ax.invert_yaxis()  # front wall at top

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Y Position Over Time — Hip vs Heel/Ankle\n"
        "Spikes in the hip line that are absent in the heel line = jump artifacts"
    )
    plt.tight_layout()
    path = os.path.join(VALIDATION_DIR, "y_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Plot 3: Histograms — distribution shift and jitter reduction ──────────
    # 2×2 grid: rows = players, columns = Y distribution / X distribution.
    # Y histograms show the bias correction (distribution shifts toward back wall)
    # and jitter reduction (narrower spread). X histograms are a sanity check
    # that lateral tracking is broadly unchanged by the fix.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Position Distributions: Hip vs Heel/Ankle\n"
                 "Y: should shift toward back wall and narrow  |  X: should be similar")

    datasets = [
        ("Player 1", hxs1, hys1, gxs1, gys1),
        ("Player 2", hxs2, hys2, gxs2, gys2),
    ]

    for row, (label, hx, hy, gx, gy) in enumerate(datasets):
        # ── Y distribution ────────────────────────────────────────────────────
        ax = axes[row, 0]
        bins = np.linspace(0, 9.75, 50)  # full court depth
        ax.hist(hy, bins=bins, alpha=0.55, color="coral",     label=f"Hip  μ={np.mean(hy):.2f}m")
        ax.hist(gy, bins=bins, alpha=0.55, color="steelblue", label=f"Heel μ={np.mean(gy):.2f}m")
        ax.axvline(np.mean(hy), color="coral",     lw=1.5, ls="--")
        ax.axvline(np.mean(gy), color="steelblue", lw=1.5, ls="--")
        ax.set_xlabel("Court Y (m)  ←front  |  back→")
        ax.set_ylabel("Frame count")
        ax.set_title(f"{label} — Y distribution")
        ax.legend(fontsize=8)

        # ── X distribution ────────────────────────────────────────────────────
        ax = axes[row, 1]
        bins = np.linspace(0, 6.4, 40)   # full court width
        ax.hist(hx, bins=bins, alpha=0.55, color="coral",     label=f"Hip  μ={np.mean(hx):.2f}m")
        ax.hist(gx, bins=bins, alpha=0.55, color="steelblue", label=f"Heel μ={np.mean(gx):.2f}m")
        ax.axvline(np.mean(hx), color="coral",     lw=1.5, ls="--")
        ax.axvline(np.mean(gx), color="steelblue", lw=1.5, ls="--")
        ax.set_xlabel("Court X (m)  ←left  |  right→")
        ax.set_ylabel("Frame count")
        ax.set_title(f"{label} — X distribution")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(VALIDATION_DIR, "histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Heatmap comparison ────────────────────────────────────────────────────
    plot_heatmap_comparison(hxs1, hys1, hxs2, hys2, gxs1, gys1, gxs2, gys2,
                             output_dir=VALIDATION_DIR)

    # ── Combined summary — all plots in one window ────────────────────────────
    saved_names = [
        "court_comparison.png", "y_timeseries.png",
        "histograms.png",       "heatmap_comparison.png",
    ]
    saved_titles = [
        "Court Comparison (Hip vs Heel/Ankle)",
        "Y Position Over Time",
        "Position Distributions",
        "Heatmap Comparison (Hip vs Heel/Ankle)",
    ]
    fig_all, axes_all = plt.subplots(4, 1, figsize=(14, 40))
    fig_all.suptitle(
        "Validation Summary — Hip vs Heel/Ankle Tracking",
        fontsize=14, fontweight="bold",
    )
    for ax, name, title in zip(axes_all, saved_names, saved_titles):
        img = plt.imread(os.path.join(VALIDATION_DIR, name))
        ax.imshow(img, aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare hip vs heel/ankle tracking.")
    parser.add_argument("--frames", type=int, default=500,
                        help="Number of video frames to process (default: 500)")
    args = parser.parse_args()
    run(args.frames)
