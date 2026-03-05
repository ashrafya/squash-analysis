import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import median_filter
from tqdm import tqdm

from video_utils import load_video
from plot_utils import plot_positions, plot_court_positions, plot_heatmap, plot_histograms, draw_court
from calibrate import get_homography, apply_homography
from stats import compute_movement_stats, print_stats_table, save_run_history
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
)

mp_pose = mp.solutions.pose

# Two independent pose detectors so each player's temporal tracking stays separate
_pose_kwargs = dict(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose1 = mp_pose.Pose(**_pose_kwargs)
pose2 = mp_pose.Pose(**_pose_kwargs)

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
    """Detect player position with a wider-crop fallback on initial failure.

    Tries CROP_MARGIN first; if detection fails, widens to 2× CROP_MARGIN.
    This catches frames where the player moved faster than expected and sits
    near the edge of the normal crop window.
    """
    pos = _detect_in_crop(frame, last_pos, CROP_MARGIN, pose_model)
    if pos is not None:
        return pos
    return _detect_in_crop(frame, last_pos, CROP_MARGIN * 2, pose_model)


def auto_detect_players(frame):
    """Detect two players by running pose detection on the top and bottom halves of the frame."""
    h, w = frame.shape[:2]
    pose_static = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                                min_detection_confidence=0.4)
    positions = []
    for y_start, y_end in [(0, h // 2), (h // 2, h)]:
        crop = frame[y_start:y_end, :]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose_static.process(rgb)
        if results.pose_landmarks:
            pos = get_ground_position(results.pose_landmarks.landmark, w, y_end - y_start)
            if pos is not None:
                positions.append((pos[0], pos[1] + y_start))
    pose_static.close()

    if len(positions) == 2:
        return positions[0], positions[1]
    elif len(positions) == 1:
        other_y = 3 * h // 4 if positions[0][1] < h // 2 else h // 4
        return positions[0], (w // 2, other_y)
    else:
        print("Auto-detection failed, using default positions.")
        return (w // 2, h // 4), (w // 2, 3 * h // 4)


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

    H = get_homography(first_frame, force_recalibrate=calibrate)

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
        swap_risk_frames = 0

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

                if pos1 is not None and np.hypot(pos1[0] - last_pos_1[0], pos1[1] - last_pos_1[1]) <= MAX_JUMP_PX:
                    last_pos_1 = pos1  # update only on a valid detection
                xs1.append(last_pos_1[0])  # always record — hold last known position on failure
                ys1.append(last_pos_1[1])

                if pos2 is not None and np.hypot(pos2[0] - last_pos_2[0], pos2[1] - last_pos_2[1]) <= MAX_JUMP_PX:
                    last_pos_2 = pos2
                xs2.append(last_pos_2[0])
                ys2.append(last_pos_2[1])

                # Player proximity check — flag frames where identities could swap
                if np.hypot(last_pos_1[0] - last_pos_2[0], last_pos_1[1] - last_pos_2[1]) < 50:
                    swap_risk_frames += 1

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

        if swap_risk_frames:
            print(f"[Warning] Players within 50px in {swap_risk_frames} frame(s) — possible identity swap.")

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

    court_xs1, court_ys1 = apply_homography(xs1, ys1, H) if xs1 else ([], [])
    court_xs2, court_ys2 = apply_homography(xs2, ys2, H) if xs2 else ([], [])

    stats1 = compute_movement_stats(court_xs1, court_ys1, fps, FRAME_SKIP)
    stats2 = compute_movement_stats(court_xs2, court_ys2, fps, FRAME_SKIP)
    print_stats_table(stats1, stats2)

    save_run_history(stats1, stats2, len(xs1), len(xs2), VIDEO_PATH, FRAME_CAP, FRAME_SKIP)

    plot_positions(xs1, ys1, xs2, ys2, background=first_frame, title="Player Movement (Pixel Space)")
    plot_court_positions(court_xs1, court_ys1, court_xs2, court_ys2, title="Player Movement (Court Space)")
    plot_heatmap(court_xs1, court_ys1, court_xs2, court_ys2)
    plot_histograms(court_xs1, court_ys1, court_xs2, court_ys2)


if __name__ == "__main__":
    main()
