"""
Benchmark player position accuracy against ground truth annotations.

Usage
-----
    # Run YOLO on the ground truth frames and print metrics
    python src/benchmark.py

    # Compare a saved positions file against ground truth (for TRACE etc.)
    python src/benchmark.py --positions output/trace_positions.json

    # Specify a different ground truth file
    python src/benchmark.py --gt assets/ground_truth.json

Output
------
    Prints a metrics table.  Optionally saves results JSON with --save.

Metrics
-------
    MAE (m)             Mean absolute Euclidean error across all annotated frames
    MAE front (m)       Same, restricted to frames where GT player y < 3 m
                        (front court — where the back-wall bias is worst)
    p90 error (m)       90th-percentile error (catches catastrophic failures)
    Detection rate      % of GT frames where the model found both players
                        (rather than holding a stale last-position)
    Speed violations    % of consecutive frame pairs implying > 8 m/s movement
                        (a proxy for ghost / wildly wrong positions)
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.calibrate import load_best_calibration, project_to_court, apply_homography, CalibData
from config import VIDEO_PATH, FRAME_CAP, VIDEO_FPS, FRAME_SKIP, FOOT_VISIBILITY_MIN
from utils.video_utils import load_video

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_PATH = os.path.join(_PROJECT_ROOT, "assets", "ground_truth.json")

_MAX_SPEED_MS = 8.0          # m/s — human sprint limit
_FRONT_COURT_Y = 3.0         # metres from front wall — "front court" threshold

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

# Import keypoint constants from the tracker
from tracking.extract_pose_yolo import (
    get_ground_position_coco,
    _YOLO_CONF, _MODEL_NAME,
    _Z_ANKLE, _Z_KNEE, _Z_HIP,
)


# ── projection helper ──────────────────────────────────────────────────────────

def _project_pt(px, py, z_hint, calib_or_H):
    if isinstance(calib_or_H, CalibData):
        xs, ys = project_to_court([px], [py], calib_or_H, zs=[z_hint])
    else:
        xs, ys = apply_homography([px], [py], calib_or_H)
    if not xs:
        return None
    x, y = xs[0], ys[0]
    if np.isnan(x) or np.isnan(y):
        return None
    return x, y


# ── YOLO single-frame inference ────────────────────────────────────────────────

def _yolo_predict_frame(frame, model, calib_or_H):
    """Run model.predict() (no tracker) on a single frame.

    Returns ((p1_court, p2_court), detected_both) where positions may be None.
    Uses the same keypoint-tier logic as the main tracker.
    """
    results = model.predict(frame, conf=_YOLO_CONF, verbose=False, classes=[0])[0]

    if results.keypoints is None or len(results.keypoints.data) == 0:
        return (None, None), False

    kps_all = results.keypoints.data.cpu().numpy()   # (N, 17, 3)
    boxes   = results.boxes.xyxy.cpu().numpy()       # (N, 4) — x1,y1,x2,y2

    positions = []
    for kps in kps_all:
        pos = get_ground_position_coco(kps)          # (x_px, y_px, z_hint) or None
        if pos is None:
            # fallback: bounding box bottom-centre
            continue
        court = _project_pt(pos[0], pos[1], pos[2], calib_or_H)
        if court is not None:
            positions.append(court)

    if len(positions) == 0:
        return (None, None), False
    if len(positions) == 1:
        return (positions[0], None), False

    # With ≥2 detections pick the two with maximum separation
    best_i, best_j, best_sep = 0, 1, -1.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            sep = np.hypot(positions[i][0] - positions[j][0],
                           positions[i][1] - positions[j][1])
            if sep > best_sep:
                best_sep = sep
                best_i, best_j = i, j

    return (positions[best_i], positions[best_j]), True


# ── metrics ────────────────────────────────────────────────────────────────────

def _euclidean(pred, gt):
    """Euclidean distance between two (x, y) court positions in metres."""
    return float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))


def compute_metrics(pred_p1: list, pred_p2: list,
                    gt_p1: list,   gt_p2: list,
                    detected: list,
                    frame_dt_s: float) -> dict:
    """
    pred_p1/p2  : list of (x, y) or None
    gt_p1/p2    : list of (x, y)  (always present)
    detected    : list of bool — True if model found both players this frame
    frame_dt_s  : time between consecutive analysed frames (seconds)
    """
    n = len(gt_p1)
    errors_p1, errors_p2 = [], []
    front_p1,  front_p2  = [], []

    for i in range(n):
        for pred, gt, err_list, front_list in [
            (pred_p1[i], gt_p1[i], errors_p1, front_p1),
            (pred_p2[i], gt_p2[i], errors_p2, front_p2),
        ]:
            if pred is None:
                continue
            e = _euclidean(pred, gt)
            err_list.append(e)
            if gt[1] < _FRONT_COURT_Y:   # near front wall
                front_list.append(e)

    all_errors = errors_p1 + errors_p2
    if not all_errors:
        return {"error": "No valid predictions"}

    # speed violations — consecutive non-None predictions
    violations = 0
    pairs = 0
    for pred_list in (pred_p1, pred_p2):
        prev = None
        for pos in pred_list:
            if pos is not None and prev is not None:
                speed = _euclidean(pos, prev) / frame_dt_s
                if speed > _MAX_SPEED_MS:
                    violations += 1
                pairs += 1
            prev = pos

    return {
        "n_frames":          n,
        "n_with_both":       sum(detected),
        "detection_rate":    round(sum(detected) / n * 100, 1),
        "mae_m":             round(float(np.mean(all_errors)), 3),
        "mae_front_m":       round(float(np.mean(front_p1 + front_p2)), 3) if (front_p1 or front_p2) else None,
        "p90_error_m":       round(float(np.percentile(all_errors, 90)), 3),
        "median_error_m":    round(float(np.median(all_errors)), 3),
        "p1_mae_m":          round(float(np.mean(errors_p1)), 3) if errors_p1 else None,
        "p2_mae_m":          round(float(np.mean(errors_p2)), 3) if errors_p2 else None,
        "speed_violations":  round(violations / max(pairs, 1) * 100, 1),
        "n_front_frames":    len(front_p1) + len(front_p2),
    }


def print_metrics(metrics: dict, label: str = "Model"):
    w = 46
    print(f"\n{'─' * w}")
    print(f"  {label}")
    print(f"{'─' * w}")
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    rows = [
        ("Frames evaluated",      f"{metrics['n_frames']}"),
        ("Detection rate",        f"{metrics['detection_rate']} %"),
        ("MAE (all frames)",      f"{metrics['mae_m']} m"),
        ("MAE (front court)",     f"{metrics['mae_front_m']} m  [{metrics['n_front_frames']} pts]"
                                  if metrics['mae_front_m'] is not None else "n/a"),
        ("Median error",          f"{metrics['median_error_m']} m"),
        ("90th-pct error",        f"{metrics['p90_error_m']} m"),
        ("P1 MAE",                f"{metrics['p1_mae_m']} m" if metrics['p1_mae_m'] is not None else "n/a"),
        ("P2 MAE",                f"{metrics['p2_mae_m']} m" if metrics['p2_mae_m'] is not None else "n/a"),
        ("Speed violations",      f"{metrics['speed_violations']} %"),
    ]
    for name, val in rows:
        print(f"  {name:<26} {val}")
    print(f"{'─' * w}\n")


# ── load external positions file (for TRACE / other models) ───────────────────

def load_positions_file(path: str) -> tuple[dict, dict]:
    """Load a positions JSON produced by another model.

    Expected format:
    {
      "p1": { "<frame_idx>": [court_x, court_y], ... },
      "p2": { "<frame_idx>": [court_x, court_y], ... }
    }
    Returns (p1_dict, p2_dict) keyed by integer frame index.
    """
    with open(path) as f:
        data = json.load(f)
    p1 = {int(k): tuple(v) for k, v in data["p1"].items()}
    p2 = {int(k): tuple(v) for k, v in data["p2"].items()}
    return p1, p2


# ── main ───────────────────────────────────────────────────────────────────────

def run_benchmark(gt_path: str, positions_path: str | None,
                  video_path: str, save_path: str | None):

    # ── load ground truth ──────────────────────────────────────────────────────
    if not os.path.exists(gt_path):
        print(f"Ground truth file not found: {gt_path}")
        print("Run annotate_ground_truth.py first.")
        return

    with open(gt_path) as f:
        gt = json.load(f)

    annotations = gt["annotations"]
    if not annotations:
        print("No annotations found in ground truth file.")
        return

    frame_indices = sorted(int(k) for k in annotations)
    print(f"Ground truth: {len(frame_indices)} annotated frames")

    gt_p1 = [(annotations[str(fi)]["p1_court"][0],
              annotations[str(fi)]["p1_court"][1]) for fi in frame_indices]
    gt_p2 = [(annotations[str(fi)]["p2_court"][0],
              annotations[str(fi)]["p2_court"][1]) for fi in frame_indices]

    frame_dt = FRAME_SKIP / VIDEO_FPS   # seconds between tracked frames

    # ── branch: external positions file vs live YOLO inference ────────────────
    if positions_path:
        if not os.path.exists(positions_path):
            print(f"Positions file not found: {positions_path}")
            return
        p1_dict, p2_dict = load_positions_file(positions_path)
        label = os.path.splitext(os.path.basename(positions_path))[0]

        pred_p1  = [p1_dict.get(fi) for fi in frame_indices]
        pred_p2  = [p2_dict.get(fi) for fi in frame_indices]
        detected = [p1_dict.get(fi) is not None and p2_dict.get(fi) is not None
                    for fi in frame_indices]

    else:
        # Live YOLO inference on GT frames
        if not _YOLO_OK:
            print("ultralytics not installed. Use --positions to benchmark from a file.")
            return

        cap, _, _ = load_video(video_path)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read video")

        calib, H = load_best_calibration(first_frame)
        calib_or_H = calib if calib is not None else H

        model = YOLO(_MODEL_NAME)
        label = _MODEL_NAME.replace(".pt", "")
        print(f"Model: {label}")
        print(f"Running inference on {len(frame_indices)} frames…\n")

        pred_p1, pred_p2, detected = [], [], []
        t0 = time.time()
        for i, fi in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                pred_p1.append(None)
                pred_p2.append(None)
                detected.append(False)
                continue

            (p1, p2), both = _yolo_predict_frame(frame, model, calib_or_H)
            pred_p1.append(p1)
            pred_p2.append(p2)
            detected.append(both)

            if (i + 1) % 50 == 0 or (i + 1) == len(frame_indices):
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                print(f"  {i + 1}/{len(frame_indices)}  ({fps:.1f} frames/s)")

        cap.release()

    # ── compute and print metrics ──────────────────────────────────────────────
    metrics = compute_metrics(pred_p1, pred_p2, gt_p1, gt_p2, detected, frame_dt)
    print_metrics(metrics, label=label)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"label": label, "metrics": metrics}, f, indent=2)
        print(f"Metrics saved → {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark player position accuracy.")
    ap.add_argument("--gt",        type=str, default=GT_PATH,
                    help="Ground truth JSON (default: assets/ground_truth.json)")
    ap.add_argument("--positions", type=str, default=None,
                    help="External positions JSON {p1: {frame: [x,y]}, p2: ...}. "
                         "If omitted, runs live YOLO inference.")
    ap.add_argument("--video",     type=str, default=VIDEO_PATH)
    ap.add_argument("--save",      type=str, default=None,
                    help="Save metrics to this JSON path")
    args = ap.parse_args()

    run_benchmark(args.gt, args.positions, args.video, args.save)
