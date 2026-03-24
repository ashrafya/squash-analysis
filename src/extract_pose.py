import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import median_filter
from tqdm import tqdm

from video_utils import load_video
from plot_utils import plot_positions, plot_court_positions, plot_heatmap, plot_histograms, plot_zone_breakdown, draw_court, show_summary, plot_timeseries
from calibrate import (
    load_best_calibration, get_homography,
    project_to_court, apply_homography, CalibData,
)
from stats import compute_movement_stats, print_stats_table, save_run_history, compute_zone_stats, print_zone_table, compute_timeseries
from config import (
    VIDEO_PATH,
    VIDEO_FPS,
    FRAME_CAP,
    FRAME_SKIP,
    MODEL_COMPLEXITY,
    SMOOTH_WINDOW,
    FOOT_VISIBILITY_MIN,
    MAX_JUMP_PX,
    CROP_MARGIN,
    DEBUG_VIZ_EVERY,
    ANGLE_MATCH_THRESHOLD,
    OUTPUT_DIR,
    MIN_SEPARATION_PX,       # kept for _try_reassign (Day 12)
    COUPLING_FRAMES_THRESHOLD,
    COURT_WIDTH_M,
    COURT_LENGTH_M,
    MIN_SEPARATION_M,
    COURT_BOUNDS_MARGIN_M,
)

try:
    mp_pose = mp.solutions.pose
except AttributeError:
    mp_pose = None  # mediapipe >= 0.10.30 removed solutions; use YOLO tracker instead

# Two independent pose detectors so each player's temporal tracking stays separate
_pose_kwargs = dict(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
if mp_pose is not None:
    pose1 = mp_pose.Pose(**_pose_kwargs)
    pose2 = mp_pose.Pose(**_pose_kwargs)
else:
    pose1 = pose2 = None

POSITIONS_PATH = os.path.join(OUTPUT_DIR, "last_positions.npz")

# Looser visibility threshold used for single-landmark and shoulder fallbacks.
# Only kicks in when both hips fail the primary HIP_VISIBILITY_MIN check.
_FALLBACK_VIS = 0.35


def get_ground_position(landmarks, frame_width, frame_height):
    """Return player's floor-contact position from pose landmarks.

    Heels and ankles lie on (or very near) the floor plane, so the homography
    maps them accurately with no parallax bias. Hips are kept as a last resort
    only — they introduce a forward-bias and jump artifacts but beat losing the
    position entirely.

    Tier 0: both heels ≥ FOOT_VISIBILITY_MIN → heel midpoint       (on floor, best)
    Tier 1: any heel or ankle ≥ _FALLBACK_VIS → lower-body average  (near floor)
    Tier 2: any hip ≥ _FALLBACK_VIS           → hip midpoint         (biased fallback)
    """
    lheel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    rheel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
    lank  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    rank  = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Tier 0: both heels confident
    if lheel.visibility >= FOOT_VISIBILITY_MIN and rheel.visibility >= FOOT_VISIBILITY_MIN:
        x = (lheel.x + rheel.x) / 2 * frame_width
        y = (lheel.y + rheel.y) / 2 * frame_height
        return x, y

    # Tier 1: any heel or ankle visible
    lower = [lm for lm in (lheel, rheel, lank, rank) if lm.visibility >= _FALLBACK_VIS]
    if lower:
        x = sum(lm.x for lm in lower) / len(lower) * frame_width
        y = sum(lm.y for lm in lower) / len(lower) * frame_height
        return x, y

    # Tier 2: hip fallback — introduces parallax bias but beats losing position
    lhip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rhip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    visible_hips = [lm for lm in (lhip, rhip) if lm.visibility >= _FALLBACK_VIS]
    if visible_hips:
        x = sum(lm.x for lm in visible_hips) / len(visible_hips) * frame_width
        y = sum(lm.y for lm in visible_hips) / len(visible_hips) * frame_height
        return x, y

    return None


def _detect_in_crop(frame, last_pos, margin, pose_model):
    """Run pose detection inside a cropped region; return hip-centre in full-frame coords."""
    h, w = frame.shape[:2]
    cx, cy = int(last_pos[0]), int(last_pos[1])
    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(w, cx + margin)
    y2 = min(h, cy + margin)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    if results.pose_landmarks:
        pos = get_ground_position(results.pose_landmarks.landmark, crop.shape[1], crop.shape[0])
        if pos is not None:
            return pos[0] + x1, pos[1] + y1

    return None


def detect_in_region(frame, last_pos, pose_model):
    """Detect player position with progressively wider crop fallbacks.

    Tries CROP_MARGIN → 2× → 3× on successive failures.  The wider crops
    catch frames where the player moved faster than expected or was temporarily
    occluded and re-appears outside the narrow window.
    """
    for multiplier in (1, 2, 3):
        pos = _detect_in_crop(frame, last_pos, CROP_MARGIN * multiplier, pose_model)
        if pos is not None:
            return pos
    return None


def _in_court_bounds(pixel_pos, calib_or_H):
    """Return True if pixel_pos maps to within the court (+COURT_BOUNDS_MARGIN_M).

    Accepts CalibData (3-D) or ndarray (2-D homography H).
    """
    if isinstance(calib_or_H, CalibData):
        cx, cy = project_to_court([pixel_pos[0]], [pixel_pos[1]], calib_or_H)
    else:
        cx, cy = apply_homography([pixel_pos[0]], [pixel_pos[1]], calib_or_H)
    if not cx:
        return True
    x, y = cx[0], cy[0]
    if np.isnan(x) or np.isnan(y):
        return True  # can't verify — pass through rather than reject
    lo_x = -COURT_BOUNDS_MARGIN_M
    hi_x =  COURT_WIDTH_M  + COURT_BOUNDS_MARGIN_M
    lo_y = -COURT_BOUNDS_MARGIN_M
    hi_y =  COURT_LENGTH_M + COURT_BOUNDS_MARGIN_M
    return lo_x <= x <= hi_x and lo_y <= y <= hi_y


def auto_detect_players(frame):
    """Detect two players by scanning four half-frame regions.

    Scans top, bottom, left, and right halves independently, deduplicates
    near-identical hits, then returns the two most spatially separated
    detections.  This is more robust than a pure top/bottom split for videos
    where both players appear in the same half or where the camera angle is
    not strictly top-to-bottom.
    """
    h, w = frame.shape[:2]
    pose_static = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                                min_detection_confidence=0.4)

    # (crop, y_offset, x_offset)
    regions = [
        (frame[0:h//2, :],    0,     0),     # top half
        (frame[h//2:h, :],    h//2,  0),     # bottom half
        (frame[:, 0:w//2],    0,     0),     # left half
        (frame[:, w//2:w],    0,     w//2),  # right half
    ]

    raw = []
    for crop, y_off, x_off in regions:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose_static.process(rgb)
        if results.pose_landmarks:
            pos = get_ground_position(results.pose_landmarks.landmark,
                                      crop.shape[1], crop.shape[0])
            if pos is not None:
                raw.append((pos[0] + x_off, pos[1] + y_off))

    pose_static.close()

    # Deduplicate: merge positions within 120px of an already-kept position
    unique = []
    for p in raw:
        if not any(np.hypot(p[0] - u[0], p[1] - u[1]) < 120 for u in unique):
            unique.append(p)

    if len(unique) >= 2:
        # Return the most spatially separated pair
        best = max(
            ((unique[i], unique[j])
             for i in range(len(unique))
             for j in range(i + 1, len(unique))),
            key=lambda pair: np.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]),
        )
        return best[0], best[1]

    if len(unique) == 1:
        p = unique[0]
        other_y = 3 * h // 4 if p[1] < h // 2 else h // 4
        print(f"[Warning] Only one player auto-detected — P2 placed at default ({w // 2}, {other_y}). "
              "Use --calibrate to re-run if tracking looks wrong.")
        return p, (w // 2, other_y)

    print("Auto-detection failed, using default positions.")
    return (w // 2, h // 4), (w // 2, 3 * h // 4)


def _try_reassign(frame, last_pos_1, last_pos_2):
    """Full-frame re-detection with optimal re-assignment (2×2 Hungarian).

    Runs auto_detect_players on the full frame and assigns the two candidate
    positions to (P1, P2) by minimum-cost matching.  Returns the original
    positions unchanged if auto-detect looks unreliable (candidates too far
    from both expected positions — indicates a fallback-to-default result).
    """
    cand_a, cand_b = auto_detect_players(frame)

    # Sanity check: each candidate must be close to at least one tracker AND
    # the two candidates must be well-separated from each other (rules out the
    # case where auto-detect found only one player and fabricated the second).
    max_accept = CROP_MARGIN  # tighter than 2× — avoids accepting fallback defaults

    def _near_either(c):
        return (np.hypot(c[0] - last_pos_1[0], c[1] - last_pos_1[1]) < max_accept
                or np.hypot(c[0] - last_pos_2[0], c[1] - last_pos_2[1]) < max_accept)

    candidates_separated = np.hypot(cand_a[0] - cand_b[0], cand_a[1] - cand_b[1]) > MIN_SEPARATION_PX

    if not (candidates_separated and _near_either(cand_a) and _near_either(cand_b)):
        return last_pos_1, last_pos_2  # unreliable — keep current trackers

    # Compare the two possible pairings and pick the lower-cost one
    cost_ab = (np.hypot(cand_a[0] - last_pos_1[0], cand_a[1] - last_pos_1[1])
               + np.hypot(cand_b[0] - last_pos_2[0], cand_b[1] - last_pos_2[1]))
    cost_ba = (np.hypot(cand_b[0] - last_pos_1[0], cand_b[1] - last_pos_1[1])
               + np.hypot(cand_a[0] - last_pos_2[0], cand_a[1] - last_pos_2[1]))

    return (cand_a, cand_b) if cost_ab <= cost_ba else (cand_b, cand_a)


def _compute_hist(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h


def smooth_positions(xs, ys, window):
    xs_smooth = median_filter(np.array(xs), size=window)
    ys_smooth = median_filter(np.array(ys), size=window)
    return xs_smooth.tolist(), ys_smooth.tolist()


def main(debug=False, calibrate=False, reuse=False):
    cap, detected_fps, frame_count = load_video(VIDEO_PATH)

    if detected_fps and abs(detected_fps - VIDEO_FPS) > 1:
        print(f"[Warning] cv2 reports {detected_fps:.2f} fps but VIDEO_FPS={VIDEO_FPS} in config. Using config value.")
    elif not detected_fps:
        print(f"[Warning] cv2 could not determine fps. Using VIDEO_FPS={VIDEO_FPS} from config.")
    fps = VIDEO_FPS

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    calib, H = load_best_calibration(first_frame, force_recalibrate=calibrate)
    _to_court = (lambda xs, ys: project_to_court(xs, ys, calib)
                 if calib is not None else
                 lambda xs, ys: apply_homography(xs, ys, H))

    # ── Reuse saved positions from last run ───────────────────────────────────
    if reuse:
        try:
            data = np.load(POSITIONS_PATH)
            xs1, ys1 = data["xs1"].tolist(), data["ys1"].tolist()
            xs2, ys2 = data["xs2"].tolist(), data["ys2"].tolist()
            print(f"Reused saved positions: {len(xs1)} for P1, {len(xs2)} for P2")
            cap.release()
        except FileNotFoundError:
            print("No saved positions found — running full tracking.")
            reuse = False

    # ── Full tracking loop ────────────────────────────────────────────────────
    if not reuse:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ref_hist = _compute_hist(first_frame)

        last_pos_1, last_pos_2 = auto_detect_players(first_frame)
        print(f"Auto-detected players: P1={last_pos_1}, P2={last_pos_2}")

        total = min(frame_count, FRAME_CAP) if FRAME_CAP else frame_count
        xs1, ys1 = [], []
        xs2, ys2 = [], []
        frame_idx = 0
        skipped = 0
        coupling_streak = 0
        max_coupling_streak = 0

        if debug:
            plt.ion()
            fig, (ax_vid, ax_court) = plt.subplots(
                1, 2, figsize=(18, 9),
                gridspec_kw={"width_ratios": [2, 1]},
            )
            fig.suptitle("Debug — live tracking")
            ax_vid.axis("off")
            ax_vid.set_title("Video frame")
            _im = ax_vid.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            draw_court(ax_court)
            ax_court.set_title("Court positions (live)")
            _sc1 = ax_court.scatter([], [], s=8, alpha=0.6, color="red",        label="Player 1", zorder=5)
            _sc2 = ax_court.scatter([], [], s=8, alpha=0.6, color="dodgerblue", label="Player 2", zorder=5)
            ax_court.legend(loc="upper right")
            plt.tight_layout()
            plt.show()

        with ThreadPoolExecutor(max_workers=2) as pool:
          with tqdm(total=total, unit="frame") as pbar:
            while cap.isOpened():
                # Advance FRAME_SKIP-1 frames without decoding (much faster than cap.read)
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

                corr = cv2.compareHist(ref_hist, _compute_hist(frame), cv2.HISTCMP_CORREL)
                if corr < ANGLE_MATCH_THRESHOLD:
                    skipped += 1
                    pbar.update(FRAME_SKIP)
                    pbar.set_postfix(p1=len(xs1), p2=len(xs2), skipped=skipped)
                    continue  # different camera angle — discard entirely

                # Detect both players in parallel (MediaPipe releases the GIL)
                fut1 = pool.submit(detect_in_region, frame, last_pos_1, pose1)
                fut2 = pool.submit(detect_in_region, frame, last_pos_2, pose2)
                pos1 = fut1.result()
                pos2 = fut2.result()

                # Reject detections that project outside the squash court
                if pos1 is not None and not _in_court_bounds(pos1, H):
                    pos1 = None
                if pos2 is not None and not _in_court_bounds(pos2, H):
                    pos2 = None

                # Pixel-space jump cap (homography extrapolation is unreliable near court edges,
                # so court-space velocity checks are not used here)
                if pos1 is not None and np.hypot(pos1[0] - last_pos_1[0], pos1[1] - last_pos_1[1]) <= MAX_JUMP_PX:
                    last_pos_1 = pos1
                xs1.append(last_pos_1[0])  # always record — hold last known position on failure
                ys1.append(last_pos_1[1])

                if pos2 is not None and np.hypot(pos2[0] - last_pos_2[0], pos2[1] - last_pos_2[1]) <= MAX_JUMP_PX:
                    last_pos_2 = pos2
                xs2.append(last_pos_2[0])
                ys2.append(last_pos_2[1])

                # Coupling diagnostic (court-space): 0.2 m threshold — players can
                # legitimately occlude each other on a squash court so we only flag
                # them as coupled when they are extremely close in real-world metres.
                # Active correction needs a real multi-person detector — see Day 12 Option B/C.
                _cp1 = _to_court([last_pos_1[0]], [last_pos_1[1]])
                _cp2 = _to_court([last_pos_2[0]], [last_pos_2[1]])
                if (_cp1[0] and _cp2[0]
                        and not np.isnan(_cp1[0][0]) and not np.isnan(_cp2[0][0])):
                    court_sep_m = np.hypot(_cp1[0][0] - _cp2[0][0], _cp1[1][0] - _cp2[1][0])
                    if court_sep_m < MIN_SEPARATION_M:
                        coupling_streak += 1
                        max_coupling_streak = max(max_coupling_streak, coupling_streak)
                    else:
                        coupling_streak = 0

                pbar.update(FRAME_SKIP)
                pbar.set_postfix(p1=len(xs1), p2=len(xs2), skipped=skipped)

                if debug and frame_idx % (FRAME_SKIP * DEBUG_VIZ_EVERY) == 0:
                    vis = frame.copy()
                    if pos1 is not None:
                        cv2.circle(vis, (int(pos1[0]), int(pos1[1])), 8, (0, 0, 255), -1)
                    if pos2 is not None:
                        cv2.circle(vis, (int(pos2[0]), int(pos2[1])), 8, (255, 100, 0), -1)
                    _im.set_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                    if xs1:
                        cxs1, cys1 = _to_court(xs1, ys1)
                        _sc1.set_offsets(np.c_[cxs1, cys1])
                    if xs2:
                        cxs2, cys2 = _to_court(xs2, ys2)
                        _sc2.set_offsets(np.c_[cxs2, cys2])
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

                if FRAME_CAP and frame_idx >= FRAME_CAP:
                    break

        if debug:
            plt.ioff()
            plt.close(fig)

        if max_coupling_streak >= COUPLING_FRAMES_THRESHOLD:
            print(f"[Warning] Longest coupling streak: {max_coupling_streak} frames — possible identity swap (see Day 12 for fix).")

        # Save raw positions (pre-smoothing) so --reuse can reload them
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez(POSITIONS_PATH, xs1=np.array(xs1), ys1=np.array(ys1),
                 xs2=np.array(xs2), ys2=np.array(ys2))
        print(f"Positions saved: {POSITIONS_PATH}")

        cap.release()

    pose1.close()
    pose2.close()

    # ── Post-processing ───────────────────────────────────────────────────────
    if xs1:
        xs1, ys1 = smooth_positions(xs1, ys1, SMOOTH_WINDOW)
    if xs2:
        xs2, ys2 = smooth_positions(xs2, ys2, SMOOTH_WINDOW)

    print(f"Extracted {len(xs1)} positions for Player 1, {len(xs2)} for Player 2")

    court_xs1, court_ys1 = _to_court(xs1, ys1) if xs1 else ([], [])
    court_xs2, court_ys2 = _to_court(xs2, ys2) if xs2 else ([], [])

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

    plot_positions(xs1, ys1, xs2, ys2, background=first_frame, title="Player Movement (Pixel Space)")
    plot_court_positions(court_xs1, court_ys1, court_xs2, court_ys2, title="Player Movement (Court Space)")
    plot_heatmap(court_xs1, court_ys1, court_xs2, court_ys2, zone_stats1=zone1, zone_stats2=zone2)
    plot_zone_breakdown(zone1, zone2)
    plot_histograms(court_xs1, court_ys1, court_xs2, court_ys2)
    plot_timeseries(ts1, ts2)

    show_summary(court_xs1, court_ys1, court_xs2, court_ys2,
                 zone_stats1=zone1, zone_stats2=zone2,
                 ts1=ts1, ts2=ts2)


if __name__ == "__main__":
    main()
