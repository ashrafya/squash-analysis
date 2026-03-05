"""
Day 10 — Ball Detection (YOLOv8 primary + motion fallback).

Primary:  YOLOv8n COCO sports-ball class 32 — full-frame inference.
Fallback: MOG2 background subtraction + circularity/size filter — catches
          fast-moving small blobs that YOLO misses.  Squash balls are black,
          ~40 mm diameter, and absent from COCO training data, so YOLO recall
          is low; the motion detector handles the majority of frames.

Merge rule: if both methods fire within MERGE_DIST_PX of each other, the YOLO
position is used (higher quality bbox centre).  Motion-only detections receive
a fixed confidence of MOTION_CONF × circularity_score.

Output: output/ball_positions.npz
  frame_idx   — video frame number of each detection
  xs, ys      — pixel-space position
  xs_m, ys_m  — court-space position (metres, via homography)
  confidence  — detection confidence [0, 1]
  method      — 0 = YOLO, 1 = motion

Run standalone:
  python src/detect_ball.py [--debug] [--calibrate] [--reuse]
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from video_utils import load_video
from calibrate import get_homography, apply_homography
from plot_utils import draw_court
from config import (
    VIDEO_PATH, VIDEO_FPS, FRAME_CAP, BALL_FRAME_SKIP,
    OUTPUT_DIR, COURT_WIDTH_M, COURT_LENGTH_M, COURT_BOUNDS_MARGIN_M,
    DEBUG_VIZ_EVERY, ANGLE_MATCH_THRESHOLD,
)

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    YOLO = None  # type: ignore

# ── Tunables ───────────────────────────────────────────────────────────────────
_YOLO_DETECT_MODEL = "yolov8n.pt"   # standard detection model (auto-downloaded first run)
_SPORTS_BALL_CLASS = 32             # COCO class 32 = sports ball
_YOLO_CONF_THRESH  = 0.15           # deliberately low — squash balls are hard for COCO YOLO
_MOTION_CONF       = 0.50           # base confidence assigned to motion-only detections
_MERGE_DIST_PX     = 60             # max pixel distance to merge YOLO + motion hits
_BALL_AREA_MIN     = 4              # px² — smallest plausible ball contour
_BALL_AREA_MAX     = 600            # px² — largest plausible ball contour (players >> this)
_BALL_CIRCULARITY  = 0.45           # min circularity (4π·area/perimeter²); circle = 1.0
_IDEAL_AREA_PX     = 50.0           # px² — expected area at typical camera distance

# Detection method codes stored in output
METHOD_YOLO   = 0
METHOD_MOTION = 1

BALL_POSITIONS_PATH = os.path.join(OUTPUT_DIR, "ball_positions.npz")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _in_court_bounds(pixel_pos, H):
    cx, cy = apply_homography([pixel_pos[0]], [pixel_pos[1]], H)
    if not cx:
        return True
    x, y = cx[0], cy[0]
    if np.isnan(x) or np.isnan(y):
        return True
    return (
        -COURT_BOUNDS_MARGIN_M <= x <= COURT_WIDTH_M  + COURT_BOUNDS_MARGIN_M
        and -COURT_BOUNDS_MARGIN_M <= y <= COURT_LENGTH_M + COURT_BOUNDS_MARGIN_M
    )


def _compute_hist(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h


def _circularity(contour):
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perim = cv2.arcLength(contour, True)
    return (4.0 * np.pi * area) / (perim * perim) if perim > 0 else 0.0


# ── Detection methods ──────────────────────────────────────────────────────────

def detect_ball_yolo(frame, model, H):
    """Run YOLOv8n on the full frame and return (x, y, conf) or None.

    Takes the highest-confidence sports-ball detection that maps within
    court bounds.  Confidence threshold is intentionally low because squash
    balls are not well-represented in COCO training data.
    """
    results = model.predict(frame, conf=_YOLO_CONF_THRESH, verbose=False,
                            classes=[_SPORTS_BALL_CLASS])[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None

    best = None
    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if _in_court_bounds((cx, cy), H):
            if best is None or conf > best[2]:
                best = (cx, cy, conf)

    return best


def detect_ball_motion(frame, bg_sub, H):
    """Use MOG2 background subtraction to find the single best ball candidate.

    Returns (x, y, score) or None.
    score = 0.7 × circularity + 0.3 × size_match — higher is more ball-like.
    Only one candidate is returned (highest score) because there is only one ball.
    """
    fg_mask = bg_sub.apply(frame)

    # Threshold out MOG2 shadow pixels (labelled 127) — keep only hard foreground
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Small morphological close to fill single-pixel gaps in ball blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (_BALL_AREA_MIN <= area <= _BALL_AREA_MAX):
            continue

        circ = _circularity(cnt)
        if circ < _BALL_CIRCULARITY:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        if not _in_court_bounds((cx, cy), H):
            continue

        size_match = 1.0 / (1.0 + abs(area - _IDEAL_AREA_PX) / _IDEAL_AREA_PX)
        score = 0.7 * circ + 0.3 * size_match
        if best is None or score > best[2]:
            best = (cx, cy, score)

    return best


def _merge(yolo_det, motion_det):
    """Merge YOLO and motion results into (x, y, conf, method) or None.

    Priority: YOLO > motion.  If both fire within MERGE_DIST_PX the YOLO
    position is used with a small confidence boost (both detectors agree).
    """
    if yolo_det is not None and motion_det is not None:
        dist = np.hypot(yolo_det[0] - motion_det[0], yolo_det[1] - motion_det[1])
        if dist < _MERGE_DIST_PX:
            return yolo_det[0], yolo_det[1], min(1.0, yolo_det[2] * 1.2), METHOD_YOLO

    if yolo_det is not None:
        return yolo_det[0], yolo_det[1], yolo_det[2], METHOD_YOLO

    if motion_det is not None:
        mx, my, score = motion_det
        return mx, my, _MOTION_CONF * score, METHOD_MOTION

    return None


# ── Plot ───────────────────────────────────────────────────────────────────────

def _plot_trajectory(xs_m, ys_m, methods):
    fig, ax = plt.subplots(figsize=(6, 9))
    draw_court(ax)
    ax.set_title("Ball Trajectory (court space)")

    methods = np.array(methods)
    yolo_mask   = methods == METHOD_YOLO
    motion_mask = methods == METHOD_MOTION

    if np.any(yolo_mask):
        ax.scatter(
            np.array(xs_m)[yolo_mask], np.array(ys_m)[yolo_mask],
            s=14, alpha=0.8, color="yellow", edgecolors="darkorange", linewidths=0.6,
            label=f"YOLO ({yolo_mask.sum()})", zorder=5,
        )
    if np.any(motion_mask):
        ax.scatter(
            np.array(xs_m)[motion_mask], np.array(ys_m)[motion_mask],
            s=8, alpha=0.5, color="cyan", edgecolors="steelblue", linewidths=0.4,
            label=f"Motion ({motion_mask.sum()})", zorder=4,
        )

    ax.legend(loc="upper right")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "ball_trajectory.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Ball trajectory saved: {out_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(debug=False, calibrate=False, reuse=False):
    if not _YOLO_AVAILABLE:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    cap, detected_fps, frame_count = load_video(VIDEO_PATH)
    if detected_fps and abs(detected_fps - VIDEO_FPS) > 1:
        print(f"[Warning] cv2 reports {detected_fps:.2f} fps but VIDEO_FPS={VIDEO_FPS} in config.")
    fps = VIDEO_FPS  # noqa: F841 — kept for future per-frame timing use

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    H = get_homography(first_frame, force_recalibrate=calibrate)

    # ── Reuse ──────────────────────────────────────────────────────────────────
    if reuse:
        try:
            data = np.load(BALL_POSITIONS_PATH)
            xs_m      = data["xs_m"].tolist()
            ys_m      = data["ys_m"].tolist()
            methods   = data["method"].tolist()
            n_yolo    = int((data["method"] == METHOD_YOLO).sum())
            n_motion  = int((data["method"] == METHOD_MOTION).sum())
            print(f"Reused saved ball positions: {len(xs_m)} detections "
                  f"(YOLO={n_yolo}, motion={n_motion})")
            cap.release()
            _plot_trajectory(xs_m, ys_m, methods)
            return
        except FileNotFoundError:
            print("No saved ball positions found — running full detection.")
            reuse = False

    model  = YOLO(_YOLO_DETECT_MODEL)
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=50, detectShadows=True
    )
    print(f"Loaded YOLOv8 detection model: {_YOLO_DETECT_MODEL}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ref_hist = _compute_hist(first_frame)

    total = min(frame_count, FRAME_CAP) if FRAME_CAP else frame_count

    frame_idx_arr  = []
    xs, ys         = [], []
    xs_m, ys_m     = [], []
    confidence_arr = []
    method_arr     = []

    frame_idx      = 0
    frames_checked = 0
    skipped        = 0

    if debug:
        plt.ion()
        fig_d, (ax_vid, ax_court) = plt.subplots(
            1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [2, 1]}
        )
        fig_d.suptitle("Debug — ball detection  (yellow=YOLO  cyan=motion  green=all motion candidates)")
        ax_vid.axis("off")
        _im = ax_vid.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        draw_court(ax_court)
        _sc_y = ax_court.scatter([], [], s=15, color="yellow", edgecolors="darkorange",
                                 linewidths=0.8, label="YOLO",   zorder=5)
        _sc_m = ax_court.scatter([], [], s=10, color="cyan",   edgecolors="steelblue",
                                 linewidths=0.5, label="Motion", zorder=4)
        ax_court.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    with tqdm(total=total, unit="frame") as pbar:
        while cap.isOpened():
            video_ok = True
            for _ in range(BALL_FRAME_SKIP - 1):
                if not cap.grab():
                    video_ok = False
                    break
                frame_idx += 1
            if not video_ok:
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            corr = cv2.compareHist(ref_hist, _compute_hist(frame), cv2.HISTCMP_CORREL)
            if corr < ANGLE_MATCH_THRESHOLD:
                skipped += 1
                pbar.update(BALL_FRAME_SKIP)
                continue

            frames_checked += 1

            yolo_det   = detect_ball_yolo(frame, model, H)
            motion_det = detect_ball_motion(frame, bg_sub, H)
            result     = _merge(yolo_det, motion_det)

            if result is not None:
                bx, by, conf, method = result
                bx_m_list, by_m_list = apply_homography([bx], [by], H)
                if bx_m_list and not np.isnan(bx_m_list[0]):
                    frame_idx_arr.append(frame_idx)
                    xs.append(bx);            ys.append(by)
                    xs_m.append(bx_m_list[0]); ys_m.append(by_m_list[0])
                    confidence_arr.append(conf)
                    method_arr.append(method)

            pbar.update(BALL_FRAME_SKIP)
            pbar.set_postfix(detected=len(xs), skipped=skipped)

            if debug and frame_idx % (BALL_FRAME_SKIP * DEBUG_VIZ_EVERY) == 0:
                vis = frame.copy()
                # Best motion candidate in green (before merge decision)
                if motion_det is not None:
                    cv2.circle(vis, (int(motion_det[0]), int(motion_det[1])), 5, (0, 255, 0), 1)
                # Accepted detection
                if result is not None:
                    bx, by, conf, method = result
                    color = (0, 255, 255) if method == METHOD_YOLO else (255, 255, 0)
                    label = f"{'YOLO' if method == METHOD_YOLO else 'motion'} {conf:.2f}"
                    cv2.circle(vis, (int(bx), int(by)), 10, color, 2)
                    cv2.putText(vis, label, (int(bx) + 12, int(by) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                _im.set_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                if xs_m:
                    y_xs = [xs_m[i] for i, m in enumerate(method_arr) if m == METHOD_YOLO]
                    y_ys = [ys_m[i] for i, m in enumerate(method_arr) if m == METHOD_YOLO]
                    m_xs = [xs_m[i] for i, m in enumerate(method_arr) if m == METHOD_MOTION]
                    m_ys = [ys_m[i] for i, m in enumerate(method_arr) if m == METHOD_MOTION]
                    if y_xs: _sc_y.set_offsets(np.c_[y_xs, y_ys])
                    if m_xs: _sc_m.set_offsets(np.c_[m_xs, m_ys])
                fig_d.canvas.draw_idle()
                plt.pause(0.001)

            if FRAME_CAP and frame_idx >= FRAME_CAP:
                break

    if debug:
        plt.ioff()
        plt.close(fig_d)

    cap.release()

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(
        BALL_POSITIONS_PATH,
        frame_idx=np.array(frame_idx_arr),
        xs=np.array(xs),   ys=np.array(ys),
        xs_m=np.array(xs_m), ys_m=np.array(ys_m),
        confidence=np.array(confidence_arr),
        method=np.array(method_arr),
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    n_yolo   = sum(1 for m in method_arr if m == METHOD_YOLO)
    n_motion = sum(1 for m in method_arr if m == METHOD_MOTION)
    detect_pct = len(xs) / max(1, frames_checked) * 100

    print(f"\nBall detection complete:")
    print(f"  Frames checked : {frames_checked}  (skipped {skipped} camera-cut frames)")
    print(f"  Detections     : {len(xs)} ({detect_pct:.1f}% of checked frames)")
    print(f"  YOLO           : {n_yolo}")
    print(f"  Motion         : {n_motion}")
    print(f"  Saved          : {BALL_POSITIONS_PATH}")

    if xs_m:
        _plot_trajectory(xs_m, ys_m, method_arr)


if __name__ == "__main__":
    import argparse
    import warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

    parser = argparse.ArgumentParser(description="Day 10 — Ball Detection (YOLOv8 + motion fallback)")
    parser.add_argument("--debug",     action="store_true", help="Show live detection overlay.")
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration.")
    parser.add_argument("--reuse",     action="store_true", help="Skip detection and reload saved positions.")
    args = parser.parse_args()
    main(debug=args.debug, calibrate=args.calibrate, reuse=args.reuse)
