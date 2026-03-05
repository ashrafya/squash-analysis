"""
Option B — YOLOv8-pose player tracker (Day 12).

Single full-frame forward pass per analysed frame → all people detected
simultaneously → Hungarian matching assigns detections to (P1, P2).

Player identity coupling is architecturally eliminated: unlike the MediaPipe
crop-based trackers, YOLO never locks onto a region — it always sees the full
frame and re-assigns from scratch each time via Hungarian matching.

COCO-17 keypoint indices used for ground-position estimation:
  11=L_hip  12=R_hip  13=L_knee  14=R_knee  15=L_ankle  16=R_ankle
  (COCO has no "heel" landmark — ankles serve as the primary floor contact)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from video_utils import load_video
from plot_utils import (
    plot_positions, plot_court_positions, plot_heatmap, plot_histograms,
    draw_court, show_summary, plot_timeseries,
)
from calibrate import get_homography, apply_homography
from stats import (
    compute_movement_stats, print_stats_table, save_run_history,
    compute_zone_stats, print_zone_table, compute_timeseries,
)
from config import (
    VIDEO_PATH, VIDEO_FPS, FRAME_CAP, FRAME_SKIP,
    SMOOTH_WINDOW, FOOT_VISIBILITY_MIN, MAX_JUMP_PX,
    ANGLE_MATCH_THRESHOLD, OUTPUT_DIR,
    COURT_WIDTH_M, COURT_LENGTH_M, COURT_BOUNDS_MARGIN_M,
    DEBUG_VIZ_EVERY,
)

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    YOLO = None  # type: ignore

# ── YOLO settings ──────────────────────────────────────────────────────────────
_MODEL_NAME  = "yolov8n-pose.pt"  # smallest / fastest pose model; auto-downloaded on first run
_YOLO_CONF   = 0.35               # min person-detection confidence
_FALLBACK_VIS = 0.35              # min COCO keypoint confidence for fallback tiers

# COCO-17 keypoint indices
_L_HIP,   _R_HIP   = 11, 12
_L_KNEE,  _R_KNEE  = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

POSITIONS_PATH = os.path.join(OUTPUT_DIR, "last_positions_yolo.npz")


# ── Ground position from COCO keypoints ───────────────────────────────────────

def get_ground_position_coco(kps: np.ndarray):
    """Return player's floor-contact point from COCO-17 keypoints.

    kps: np.ndarray shape (17, 3) — columns are [x_px, y_px, confidence].

    Mirrors the MediaPipe tier logic in extract_pose.py, adapted for COCO indices.
    COCO has no heel landmark, so ankles are used as the primary floor contact.

    Tier 0: both ankles ≥ FOOT_VISIBILITY_MIN → ankle midpoint    (floor plane, best)
    Tier 1: any ankle or knee ≥ _FALLBACK_VIS → lower-body avg    (near floor)
    Tier 2: any hip ≥ _FALLBACK_VIS           → hip midpoint       (biased fallback)
    """
    l_ank = kps[_L_ANKLE]
    r_ank = kps[_R_ANKLE]
    l_kne = kps[_L_KNEE]
    r_kne = kps[_R_KNEE]
    l_hip = kps[_L_HIP]
    r_hip = kps[_R_HIP]

    # Tier 0 — both ankles confident
    if l_ank[2] >= FOOT_VISIBILITY_MIN and r_ank[2] >= FOOT_VISIBILITY_MIN:
        return (l_ank[0] + r_ank[0]) / 2.0, (l_ank[1] + r_ank[1]) / 2.0

    # Tier 1 — any ankle or knee visible
    lower = [kp for kp in (l_ank, r_ank, l_kne, r_kne) if kp[2] >= _FALLBACK_VIS]
    if lower:
        return (
            sum(kp[0] for kp in lower) / len(lower),
            sum(kp[1] for kp in lower) / len(lower),
        )

    # Tier 2 — hip fallback
    hips = [kp for kp in (l_hip, r_hip) if kp[2] >= _FALLBACK_VIS]
    if hips:
        return (
            sum(kp[0] for kp in hips) / len(hips),
            sum(kp[1] for kp in hips) / len(hips),
        )

    return None


# ── Court-bounds filter ────────────────────────────────────────────────────────

def _in_court_bounds(pixel_pos, H):
    """True if pixel_pos maps within court + COURT_BOUNDS_MARGIN_M via H."""
    cx, cy = apply_homography([pixel_pos[0]], [pixel_pos[1]], H)
    if not cx:
        return True
    x, y = cx[0], cy[0]
    if np.isnan(x) or np.isnan(y):
        return True  # can't verify — pass through
    return (
        -COURT_BOUNDS_MARGIN_M <= x <= COURT_WIDTH_M  + COURT_BOUNDS_MARGIN_M
        and -COURT_BOUNDS_MARGIN_M <= y <= COURT_LENGTH_M + COURT_BOUNDS_MARGIN_M
    )


# ── YOLO inference ─────────────────────────────────────────────────────────────

def _detect_all_players(frame, model, H):
    """Run YOLOv8-pose on the full frame.

    Returns a list of (x, y) ground positions (pixel space) for all detected
    people whose ground position maps within court bounds.
    """
    results = model.predict(frame, conf=_YOLO_CONF, verbose=False, classes=[0])[0]

    positions = []
    if results.keypoints is None or results.keypoints.data is None:
        return positions

    for kps_tensor in results.keypoints.data:
        kps = kps_tensor.cpu().numpy()  # (17, 3)
        pos = get_ground_position_coco(kps)
        if pos is None:
            continue
        if not _in_court_bounds(pos, H):
            continue
        positions.append(pos)

    return positions


def _hungarian_assign(detections, last_pos_1, last_pos_2):
    """Assign detections to (P1, P2) via Hungarian minimum-cost matching.

    Returns (new_pos_1, new_pos_2) where either may be None if no detection
    was close enough to be matched to that player.
    """
    n = len(detections)
    if n == 0:
        return None, None

    if n == 1:
        d = detections[0]
        dist1 = np.hypot(d[0] - last_pos_1[0], d[1] - last_pos_1[1])
        dist2 = np.hypot(d[0] - last_pos_2[0], d[1] - last_pos_2[1])
        return (d, None) if dist1 <= dist2 else (None, d)

    # Build 2 × n cost matrix and run Hungarian algorithm
    cost = np.array([
        [np.hypot(d[0] - p[0], d[1] - p[1]) for d in detections]
        for p in (last_pos_1, last_pos_2)
    ])  # shape (2, n)

    row_ind, col_ind = linear_sum_assignment(cost)

    new_pos_1 = new_pos_2 = None
    for r, c in zip(row_ind, col_ind):
        if r == 0:
            new_pos_1 = detections[c]
        else:
            new_pos_2 = detections[c]

    return new_pos_1, new_pos_2


# ── Camera-cut filter ──────────────────────────────────────────────────────────

def _compute_hist(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h


# ── Smoothing ──────────────────────────────────────────────────────────────────

def smooth_positions(xs, ys, window):
    xs_smooth = median_filter(np.array(xs), size=window)
    ys_smooth = median_filter(np.array(ys), size=window)
    return xs_smooth.tolist(), ys_smooth.tolist()


# ── Initial player detection ───────────────────────────────────────────────────

def _initial_detect(first_frame, model, H):
    """Detect starting (P1, P2) positions from the first frame using YOLO.

    Returns two pixel-space positions.  Falls back to frame-quarter defaults
    if fewer than two players are found.
    """
    h, w = first_frame.shape[:2]
    positions = _detect_all_players(first_frame, model, H)

    if len(positions) >= 2:
        # Pick the most spatially separated pair
        best_dist = -1.0
        best_pair = (positions[0], positions[1])
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = np.hypot(
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                )
                if d > best_dist:
                    best_dist = d
                    best_pair = (positions[i], positions[j])
        print(f"YOLO auto-detected 2 players: P1={best_pair[0]}, P2={best_pair[1]}")
        return best_pair[0], best_pair[1]

    if len(positions) == 1:
        p = positions[0]
        other_y = int(3 * h / 4) if p[1] < h / 2 else int(h / 4)
        fallback = (w // 2, other_y)
        print(f"[Warning] YOLO found only 1 player in first frame — P2 placed at default {fallback}.")
        return p, fallback

    print("[Warning] YOLO found no players in first frame — using default positions.")
    return (w // 4, h // 2), (3 * w // 4, h // 2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(debug=False, calibrate=False, reuse=False):
    if not _YOLO_AVAILABLE:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )

    cap, detected_fps, frame_count = load_video(VIDEO_PATH)

    if detected_fps and abs(detected_fps - VIDEO_FPS) > 1:
        print(f"[Warning] cv2 reports {detected_fps:.2f} fps but VIDEO_FPS={VIDEO_FPS} in config. Using config value.")
    elif not detected_fps:
        print(f"[Warning] cv2 could not determine fps. Using VIDEO_FPS={VIDEO_FPS} from config.")
    fps = VIDEO_FPS

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    H = get_homography(first_frame, force_recalibrate=calibrate)

    model = YOLO(_MODEL_NAME)
    print(f"Loaded YOLOv8-pose model: {_MODEL_NAME}")

    # ── Reuse saved positions ──────────────────────────────────────────────────
    if reuse:
        try:
            data = np.load(POSITIONS_PATH)
            xs1, ys1 = data["xs1"].tolist(), data["ys1"].tolist()
            xs2, ys2 = data["xs2"].tolist(), data["ys2"].tolist()
            print(f"Reused saved YOLO positions: {len(xs1)} for P1, {len(xs2)} for P2")
            cap.release()
        except FileNotFoundError:
            print("No saved YOLO positions found — running full tracking.")
            reuse = False

    # ── Full tracking loop ─────────────────────────────────────────────────────
    if not reuse:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ref_hist = _compute_hist(first_frame)

        last_pos_1, last_pos_2 = _initial_detect(first_frame, model, H)

        total = min(frame_count, FRAME_CAP) if FRAME_CAP else frame_count
        xs1, ys1 = [], []
        xs2, ys2 = [], []
        frame_idx = 0
        skipped = 0

        if debug:
            plt.ion()
            fig, (ax_vid, ax_court) = plt.subplots(
                1, 2, figsize=(18, 9),
                gridspec_kw={"width_ratios": [2, 1]},
            )
            fig.suptitle("Debug — YOLO live tracking")
            ax_vid.axis("off")
            ax_vid.set_title("Video frame  (green = raw YOLO detections)")
            _im = ax_vid.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            draw_court(ax_court)
            ax_court.set_title("Court positions (live)")
            _sc1 = ax_court.scatter([], [], s=8, alpha=0.6, color="red",        label="Player 1", zorder=5)
            _sc2 = ax_court.scatter([], [], s=8, alpha=0.6, color="dodgerblue", label="Player 2", zorder=5)
            ax_court.legend(loc="upper right")
            plt.tight_layout()
            plt.show()

        with tqdm(total=total, unit="frame") as pbar:
            while cap.isOpened():
                # Advance FRAME_SKIP-1 frames without decoding
                video_ok = True
                for _ in range(FRAME_SKIP - 1):
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

                # Camera-cut filter
                corr = cv2.compareHist(ref_hist, _compute_hist(frame), cv2.HISTCMP_CORREL)
                if corr < ANGLE_MATCH_THRESHOLD:
                    skipped += 1
                    pbar.update(FRAME_SKIP)
                    pbar.set_postfix(p1=len(xs1), p2=len(xs2), skipped=skipped)
                    continue

                # Full-frame YOLO inference → Hungarian assign to (P1, P2)
                detections = _detect_all_players(frame, model, H)
                pos1, pos2 = _hungarian_assign(detections, last_pos_1, last_pos_2)

                # Jump cap + hold-last-value (same semantics as MediaPipe tracker)
                if pos1 is not None and np.hypot(pos1[0] - last_pos_1[0], pos1[1] - last_pos_1[1]) <= MAX_JUMP_PX:
                    last_pos_1 = pos1
                xs1.append(last_pos_1[0])
                ys1.append(last_pos_1[1])

                if pos2 is not None and np.hypot(pos2[0] - last_pos_2[0], pos2[1] - last_pos_2[1]) <= MAX_JUMP_PX:
                    last_pos_2 = pos2
                xs2.append(last_pos_2[0])
                ys2.append(last_pos_2[1])

                pbar.update(FRAME_SKIP)
                pbar.set_postfix(p1=len(xs1), p2=len(xs2), dets=len(detections), skipped=skipped)

                if debug and frame_idx % (FRAME_SKIP * DEBUG_VIZ_EVERY) == 0:
                    vis = frame.copy()
                    # Raw YOLO detections in green
                    for d in detections:
                        cv2.circle(vis, (int(d[0]), int(d[1])), 6, (0, 255, 0), 2)
                    # Tracked positions
                    cv2.circle(vis, (int(last_pos_1[0]), int(last_pos_1[1])), 9, (0, 0, 255),   -1)
                    cv2.circle(vis, (int(last_pos_2[0]), int(last_pos_2[1])), 9, (255, 100, 0), -1)
                    _im.set_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                    if xs1:
                        cxs1, cys1 = apply_homography(xs1, ys1, H)
                        _sc1.set_offsets(np.c_[cxs1, cys1])
                    if xs2:
                        cxs2, cys2 = apply_homography(xs2, ys2, H)
                        _sc2.set_offsets(np.c_[cxs2, cys2])
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

                if FRAME_CAP and frame_idx >= FRAME_CAP:
                    break

        if debug:
            plt.ioff()
            plt.close(fig)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez(POSITIONS_PATH, xs1=np.array(xs1), ys1=np.array(ys1),
                 xs2=np.array(xs2), ys2=np.array(ys2))
        print(f"YOLO positions saved: {POSITIONS_PATH}")

        cap.release()

    # ── Post-processing (identical to extract_pose.py) ────────────────────────
    if xs1:
        xs1, ys1 = smooth_positions(xs1, ys1, SMOOTH_WINDOW)
    if xs2:
        xs2, ys2 = smooth_positions(xs2, ys2, SMOOTH_WINDOW)

    print(f"Extracted {len(xs1)} positions for Player 1, {len(xs2)} for Player 2")

    court_xs1, court_ys1 = apply_homography(xs1, ys1, H) if xs1 else ([], [])
    court_xs2, court_ys2 = apply_homography(xs2, ys2, H) if xs2 else ([], [])

    stats1 = compute_movement_stats(court_xs1, court_ys1, fps, FRAME_SKIP)
    stats2 = compute_movement_stats(court_xs2, court_ys2, fps, FRAME_SKIP)
    print_stats_table(stats1, stats2)

    zone1 = compute_zone_stats(court_xs1, court_ys1)
    zone2 = compute_zone_stats(court_xs2, court_ys2)
    print_zone_table(zone1, zone2)

    save_run_history(stats1, stats2, len(xs1), len(xs2), VIDEO_PATH, FRAME_CAP, FRAME_SKIP,
                     zone_stats1=zone1, zone_stats2=zone2)

    ts1 = compute_timeseries(court_xs1, court_ys1, fps, FRAME_SKIP)
    ts2 = compute_timeseries(court_xs2, court_ys2, fps, FRAME_SKIP)

    plot_positions(xs1, ys1, xs2, ys2, background=first_frame,
                   title="Player Movement (Pixel Space) — YOLOv8-pose")
    plot_court_positions(court_xs1, court_ys1, court_xs2, court_ys2,
                         title="Player Movement (Court Space) — YOLOv8-pose")
    plot_heatmap(court_xs1, court_ys1, court_xs2, court_ys2, zone_stats1=zone1, zone_stats2=zone2)
    plot_histograms(court_xs1, court_ys1, court_xs2, court_ys2)
    plot_timeseries(ts1, ts2)

    show_summary(court_xs1, court_ys1, court_xs2, court_ys2,
                 zone_stats1=zone1, zone_stats2=zone2,
                 ts1=ts1, ts2=ts2)
