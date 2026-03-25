"""
YOLOv8-pose player tracker — ByteTrack edition.

Architecture
------------
* model.track(persist=True) — YOLO's built-in ByteTrack assigns each person
  a persistent track-ID and uses a Kalman filter to predict positions when
  detections are temporarily missed.  This is far more robust than running
  model.predict() and re-assigning from scratch every frame.

* ID→player mapping — we maintain a small dict {track_id: 0|1} so that
  "P1" and "P2" labels survive across the whole clip.  When ByteTrack
  recycles an ID after a long occlusion, the mapping is refreshed by
  matching the new detection to the closest last-known player position.

COCO-17 keypoint indices used for ground-position estimation:
  11=L_hip  12=R_hip  13=L_knee  14=R_knee  15=L_ankle  16=R_ankle
  (COCO has no heel landmark — ankles are the primary floor-contact proxy)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment  # kept for potential future use
from tqdm import tqdm

from utils.video_utils import load_video
from analysis.plot_utils import (
    plot_positions, plot_court_positions, plot_heatmap, plot_histograms,
    plot_zone_breakdown, draw_court, show_summary, plot_timeseries,
)
from calibration.calibrate import (
    get_calibration, get_homography,
    project_to_court, apply_homography,
    load_best_calibration, CalibData,
)
from analysis.stats import (
    compute_movement_stats, print_stats_table, save_run_history,
    compute_zone_stats, print_zone_table, compute_timeseries,
)
from config import (
    VIDEO_PATH, VIDEO_FPS, FRAME_CAP, FRAME_SKIP,
    SMOOTH_WINDOW, FOOT_VISIBILITY_MIN,
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
_MODEL_NAME   = "yolov8m-pose.pt"  # medium model — noticeably better than nano for partially occluded players
_YOLO_CONF    = 0.25               # lower threshold catches more partially-visible players
_FALLBACK_VIS = 0.25               # COCO keypoint confidence for lower-body fallback tiers
_MAX_LOST_FRAMES = 60              # frames before a stale track-ID is evicted from the ID→player map

# COCO-17 keypoint indices
_L_HIP,   _R_HIP   = 11, 12
_L_KNEE,  _R_KNEE  = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

# Expected world heights (metres) for each keypoint tier used as a floor proxy.
# project_to_court() intersects the ray at this height to recover the correct
# (x, y) floor position even when ankles are not visible.
_Z_ANKLE = 0.0    # ankles are on the floor
_Z_KNEE  = 0.50   # knees ~0.5 m above floor
_Z_HIP   = 0.92   # hips  ~0.9 m above floor

POSITIONS_PATH = os.path.join(OUTPUT_DIR, "last_positions_yolo.npz")


# ── Ground position from COCO keypoints ───────────────────────────────────────

def get_ground_position_coco(kps: np.ndarray):
    """Return player's floor-contact point from COCO-17 keypoints.

    kps: np.ndarray shape (17, 3) — columns are [x_px, y_px, confidence].

    Returns (x_px, y_px, z_world_hint) where z_world_hint is the expected
    world height (metres) of the detected keypoint used.  The caller should
    pass this to project_to_court() so it intersects the ray at the correct
    height plane and recovers the true (x, y) floor position.

    Tier 0: both ankles ≥ FOOT_VISIBILITY_MIN → ankle midpoint   z=0.0  (floor)
    Tier 1: any ankle or knee ≥ _FALLBACK_VIS → lower-body avg   z=weighted mean
    Tier 2: any hip ≥ _FALLBACK_VIS           → hip midpoint      z=_Z_HIP
    """
    l_ank = kps[_L_ANKLE]
    r_ank = kps[_R_ANKLE]
    l_kne = kps[_L_KNEE]
    r_kne = kps[_R_KNEE]
    l_hip = kps[_L_HIP]
    r_hip = kps[_R_HIP]

    # Tier 0 — both ankles confident
    if l_ank[2] >= FOOT_VISIBILITY_MIN and r_ank[2] >= FOOT_VISIBILITY_MIN:
        return (l_ank[0] + r_ank[0]) / 2.0, (l_ank[1] + r_ank[1]) / 2.0, _Z_ANKLE

    # Tier 1 — any ankle or knee visible; z is a weighted mean of their heights
    lower = [(kp, _Z_ANKLE if i < 2 else _Z_KNEE)
             for i, kp in enumerate((l_ank, r_ank, l_kne, r_kne))
             if kp[2] >= _FALLBACK_VIS]
    if lower:
        kps_only, zs_only = zip(*lower)
        return (
            sum(kp[0] for kp in kps_only) / len(kps_only),
            sum(kp[1] for kp in kps_only) / len(kps_only),
            sum(zs_only) / len(zs_only),
        )

    # Tier 2 — hip fallback
    hips = [kp for kp in (l_hip, r_hip) if kp[2] >= _FALLBACK_VIS]
    if hips:
        return (
            sum(kp[0] for kp in hips) / len(hips),
            sum(kp[1] for kp in hips) / len(hips),
            _Z_HIP,
        )

    return None


# ── Court-bounds filter ────────────────────────────────────────────────────────

def _in_court_bounds(pixel_pos, calib_or_H):
    """True if pixel_pos maps within court + COURT_BOUNDS_MARGIN_M.

    Accepts either a CalibData (3-D) or a numpy ndarray (2-D homography H).
    """
    if isinstance(calib_or_H, CalibData):
        cx, cy = project_to_court([pixel_pos[0]], [pixel_pos[1]], calib_or_H)
    else:
        cx, cy = apply_homography([pixel_pos[0]], [pixel_pos[1]], calib_or_H)
    if not cx:
        return True
    x, y = cx[0], cy[0]
    if np.isnan(x) or np.isnan(y):
        return True  # can't verify — pass through
    return (
        -COURT_BOUNDS_MARGIN_M <= x <= COURT_WIDTH_M  + COURT_BOUNDS_MARGIN_M
        and -COURT_BOUNDS_MARGIN_M <= y <= COURT_LENGTH_M + COURT_BOUNDS_MARGIN_M
    )


# ── ByteTrack ID → player mapping (module-level state, reset in main()) ────────

_id_to_player: dict  = {}   # {yolo_track_id: 0 or 1}
_id_last_seen: dict  = {}   # {yolo_track_id: frame_idx}


def _reset_tracker():
    """Clear tracker state (called at the start of each main() run)."""
    _id_to_player.clear()
    _id_last_seen.clear()


# ── YOLO ByteTrack inference ───────────────────────────────────────────────────

def _track_frame(frame, model, calib_or_H, frame_idx: int,
                 last_pos_1, last_pos_2):
    """Run YOLOv8-pose with persist=True (ByteTrack) on one frame.

    Returns (pos_p1, pos_p2) where each pos is (x_px, y_px, z_hint) or None.
    z_hint is the expected world height of the detected keypoint — passed on to
    project_to_court() so it corrects for the parallax from non-ankle fallbacks.

    ID → player mapping logic
    -------------------------
    1. ByteTrack assigns each person a persistent integer track_id that
       survives temporary occlusions via Kalman prediction.
    2. The first time we see a track_id we assign it to the free player
       slot whose last-known position it is closest to.
    3. Stale IDs (not seen for > _MAX_LOST_FRAMES) are evicted so the
       slot can be re-assigned when the player reappears with a new ID.
    """
    results = model.track(frame, persist=True,
                           conf=_YOLO_CONF, verbose=False, classes=[0])[0]

    if results.boxes.id is None or results.keypoints is None:
        return None, None

    track_ids = results.boxes.id.cpu().numpy().astype(int)
    kps_all   = results.keypoints.data.cpu().numpy()   # (N, 17, 3)

    # ── collect in-court detections keyed by track_id ─────────────────────────
    detections: dict = {}    # track_id → (x_px, y_px, z_hint)
    for tid, kps in zip(track_ids, kps_all):
        pos = get_ground_position_coco(kps)   # (x, y, z_hint) or None
        if pos is None:
            continue
        if not _in_court_bounds(pos, calib_or_H):   # uses pos[0], pos[1]
            continue
        detections[tid] = pos
        _id_last_seen[tid] = frame_idx

    # ── evict stale IDs from the mapping ──────────────────────────────────────
    stale = [tid for tid, last in _id_last_seen.items()
             if frame_idx - last > _MAX_LOST_FRAMES]
    for tid in stale:
        _id_to_player.pop(tid, None)
        _id_last_seen.pop(tid, None)

    if not detections:
        return None, None

    # ── resolve known IDs first ────────────────────────────────────────────────
    p1_pos = p2_pos = None
    for tid, pos in detections.items():
        if tid in _id_to_player:
            if _id_to_player[tid] == 0:
                p1_pos = pos
            else:
                p2_pos = pos

    # ── assign unknown IDs to free player slots ────────────────────────────────
    unknown = {tid: pos for tid, pos in detections.items()
               if tid not in _id_to_player}

    for tid, pos in unknown.items():
        occupied = set(_id_to_player.values())
        free = [s for s in (0, 1) if s not in occupied]

        if not free:
            # Both slots taken by currently-active IDs — ignore extra person
            continue

        if len(free) == 1:
            slot = free[0]
        else:
            # Two free slots — assign to closest last-known position (use x,y only)
            refs = [last_pos_1, last_pos_2]
            dists = [
                np.hypot(pos[0] - refs[s][0], pos[1] - refs[s][1])
                if refs[s] is not None else float("inf")
                for s in free
            ]
            slot = free[int(np.argmin(dists))]

        _id_to_player[tid] = slot
        if slot == 0:
            p1_pos = pos
        else:
            p2_pos = pos

    return p1_pos, p2_pos


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

def _initial_detect(cap, model, calib_or_H, first_frame):
    """Scan up to 60 frames to find a frame with two clearly separated players.

    Checking multiple frames handles cold-start scenarios where the first frame
    has only one visible player (e.g. mid-rally, one player out of shot).

    Returns (last_pos_1, last_pos_2) in pixel space.
    """
    h, w = first_frame.shape[:2]
    best_pair   = None
    best_sep    = -1.0
    scan_frames = 60

    # Include the already-read first_frame in the scan
    frames_to_check = [first_frame]
    for _ in range(scan_frames - 1):
        ret, f = cap.read()
        if not ret:
            break
        frames_to_check.append(f)

    for frame in frames_to_check:
        result = model.track(frame, persist=True,
                              conf=_YOLO_CONF, verbose=False, classes=[0])[0]
        if result.boxes.id is None or result.keypoints is None:
            continue

        tids = result.boxes.id.cpu().numpy().astype(int)
        kps_all = result.keypoints.data.cpu().numpy()

        positions = []
        for _, kps in zip(tids, kps_all):
            pos = get_ground_position_coco(kps)
            if pos is None or not _in_court_bounds(pos, calib_or_H):
                continue
            positions.append(pos)

        if len(positions) < 2:
            continue

        # Pick the most spatially separated in-court pair
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                sep = np.hypot(positions[i][0] - positions[j][0],
                               positions[i][1] - positions[j][1])
                if sep > best_sep:
                    best_sep  = sep
                    best_pair = (positions[i], positions[j])

    # Rewind cap so the main loop starts from the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if best_pair is not None:
        print(f"Init: found 2 players (max separation {best_sep:.0f} px)")
        return best_pair

    print("[Warning] Could not detect 2 players in first 60 frames — using defaults.")
    return (w // 4, h // 2, _Z_ANKLE), (3 * w // 4, h // 2, _Z_ANKLE)


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

    calib, H = load_best_calibration(first_frame, force_recalibrate=calibrate)
    calib_or_H = calib if calib is not None else H
    if calib is not None:
        _to_court = lambda xs, ys, zs=None: project_to_court(xs, ys, calib, zs=zs)
    else:
        _to_court = lambda xs, ys, _zs=None: apply_homography(xs, ys, H)

    model = YOLO(_MODEL_NAME)
    print(f"Loaded YOLOv8-pose model: {_MODEL_NAME}")

    # ── Reuse saved positions ──────────────────────────────────────────────────
    if reuse:
        try:
            data = np.load(POSITIONS_PATH)
            xs1, ys1 = data["xs1"].tolist(), data["ys1"].tolist()
            xs2, ys2 = data["xs2"].tolist(), data["ys2"].tolist()
            # zs were added later — default to ankle height for old saves
            zs1 = data["zs1"].tolist() if "zs1" in data else [_Z_ANKLE] * len(xs1)
            zs2 = data["zs2"].tolist() if "zs2" in data else [_Z_ANKLE] * len(xs2)
            print(f"Reused saved YOLO positions: {len(xs1)} for P1, {len(xs2)} for P2")
            cap.release()
        except FileNotFoundError:
            print("No saved YOLO positions found — running full tracking.")
            reuse = False

    # ── Full tracking loop ─────────────────────────────────────────────────────
    if not reuse:
        _reset_tracker()
        ref_hist = _compute_hist(first_frame)

        # Scan first 60 frames to find reliable starting positions; rewinds cap
        last_pos_1, last_pos_2 = _initial_detect(cap, model, calib_or_H, first_frame)

        total = min(frame_count, FRAME_CAP) if FRAME_CAP else frame_count
        xs1, ys1, zs1 = [], [], []
        xs2, ys2, zs2 = [], [], []
        frame_idx = 0
        skipped   = 0

        if debug:
            plt.ion()
            fig, (ax_vid, ax_court) = plt.subplots(
                1, 2, figsize=(18, 9),
                gridspec_kw={"width_ratios": [2, 1]},
            )
            fig.suptitle("Debug — ByteTrack live tracking")
            ax_vid.axis("off")
            ax_vid.set_title("Video frame  (P1=red  P2=orange)")
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

                # ByteTrack inference — returns None when a player is not detected
                pos1, pos2 = _track_frame(frame, model, calib_or_H,
                                           frame_idx, last_pos_1, last_pos_2)

                # Hold last-known position when ByteTrack loses a player
                if pos1 is not None:
                    last_pos_1 = pos1
                if pos2 is not None:
                    last_pos_2 = pos2

                xs1.append(last_pos_1[0])
                ys1.append(last_pos_1[1])
                zs1.append(last_pos_1[2])
                xs2.append(last_pos_2[0])
                ys2.append(last_pos_2[1])
                zs2.append(last_pos_2[2])

                both = (pos1 is not None) + (pos2 is not None)
                pbar.update(FRAME_SKIP)
                pbar.set_postfix(p1=len(xs1), p2=len(xs2),
                                  both=both, skipped=skipped)

                if debug and frame_idx % (FRAME_SKIP * DEBUG_VIZ_EVERY) == 0:
                    vis = frame.copy()
                    cv2.circle(vis, (int(last_pos_1[0]), int(last_pos_1[1])), 9, (0,   0, 255), -1)
                    cv2.circle(vis, (int(last_pos_2[0]), int(last_pos_2[1])), 9, (0, 140, 255), -1)
                    _im.set_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                    if xs1:
                        cxs1, cys1 = _to_court(xs1, ys1, zs1)
                        _sc1.set_offsets(np.c_[cxs1, cys1])
                    if xs2:
                        cxs2, cys2 = _to_court(xs2, ys2, zs2)
                        _sc2.set_offsets(np.c_[cxs2, cys2])
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

                if FRAME_CAP and frame_idx >= FRAME_CAP:
                    break

        if debug:
            plt.ioff()
            plt.close(fig)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez(POSITIONS_PATH,
                 xs1=np.array(xs1), ys1=np.array(ys1), zs1=np.array(zs1),
                 xs2=np.array(xs2), ys2=np.array(ys2), zs2=np.array(zs2))
        print(f"YOLO positions saved: {POSITIONS_PATH}")

        cap.release()

    # ── Post-processing (identical to extract_pose.py) ────────────────────────
    if xs1:
        xs1, ys1 = smooth_positions(xs1, ys1, SMOOTH_WINDOW)
    if xs2:
        xs2, ys2 = smooth_positions(xs2, ys2, SMOOTH_WINDOW)

    print(f"Extracted {len(xs1)} positions for Player 1, {len(xs2)} for Player 2")

    court_xs1, court_ys1 = _to_court(xs1, ys1, zs1) if xs1 else ([], [])
    court_xs2, court_ys2 = _to_court(xs2, ys2, zs2) if xs2 else ([], [])

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
    plot_zone_breakdown(zone1, zone2)
    plot_histograms(court_xs1, court_ys1, court_xs2, court_ys2)
    plot_timeseries(ts1, ts2)

    show_summary(court_xs1, court_ys1, court_xs2, court_ys2,
                 zone_stats1=zone1, zone_stats2=zone2,
                 ts1=ts1, ts2=ts2)
