import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from video_utils import load_video
from plot_utils import plot_positions, plot_court_positions, plot_heatmap, draw_court
from calibrate import get_homography, apply_homography

VIDEO_PATH = "../assets/video/sample.mp4"

FRAME_CAP = 5000          # max frames to process (for testing; set to None to process entire video)
FRAME_SKIP = 5            # process every Nth frame
SMOOTH_WINDOW = 9         # rolling median window for noise filtering
HIP_VISIBILITY_MIN = 0.6  # discard if either hip landmark is below this confidence
MAX_JUMP_PX = 150         # discard if position jumps more than this many pixels
CROP_MARGIN = 200         # pixel radius around last known position to crop for detection
DEBUG_VIZ_EVERY = FRAME_SKIP     # update live debug plot every N processed frames
ANGLE_MATCH_THRESHOLD = 0.7      # min histogram correlation to reference frame; below = different camera angle

mp_pose = mp.solutions.pose

# Two independent pose detectors so each player's temporal tracking stays separate
_pose_kwargs = dict(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose1 = mp_pose.Pose(**_pose_kwargs)
pose2 = mp_pose.Pose(**_pose_kwargs)


def get_hip_center(landmarks, frame_width, frame_height):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    if left_hip.visibility < HIP_VISIBILITY_MIN or right_hip.visibility < HIP_VISIBILITY_MIN:
        return None

    x = (left_hip.x + right_hip.x) / 2 * frame_width
    y = (left_hip.y + right_hip.y) / 2 * frame_height

    return x, y


def detect_in_region(frame, last_pos, pose_model):
    """Crop frame around last_pos, run pose detection, return hip centre in frame coords."""
    h, w = frame.shape[:2]
    cx, cy = int(last_pos[0]), int(last_pos[1])
    x1 = max(0, cx - CROP_MARGIN)
    y1 = max(0, cy - CROP_MARGIN)
    x2 = min(w, cx + CROP_MARGIN)
    y2 = min(h, cy + CROP_MARGIN)

    crop = frame[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    if results.pose_landmarks:
        pos = get_hip_center(results.pose_landmarks.landmark, crop_w, crop_h)
        if pos is not None:
            return pos[0] + x1, pos[1] + y1

    return None


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
            pos = get_hip_center(results.pose_landmarks.landmark, w, y_end - y_start)
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


def main(debug=False, calibrate=False):
    cap, _, frame_count = load_video(VIDEO_PATH)

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_pos_1, last_pos_2 = auto_detect_players(first_frame)
    print(f"Auto-detected players: P1={last_pos_1}, P2={last_pos_2}")

    H = get_homography(first_frame, force_recalibrate=calibrate)
    ref_hist = _compute_hist(first_frame)

    total = min(frame_count, FRAME_CAP)
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

    with tqdm(total=total, unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % FRAME_SKIP != 0:
                continue

            corr = cv2.compareHist(ref_hist, _compute_hist(frame), cv2.HISTCMP_CORREL)
            if corr < ANGLE_MATCH_THRESHOLD:
                skipped += 1
                pbar.update(FRAME_SKIP)
                pbar.set_postfix(p1=len(xs1), p2=len(xs2), skipped=skipped)
                continue  # different camera angle — discard entirely

            pos1 = detect_in_region(frame, last_pos_1, pose1)
            if pos1 is not None and np.hypot(pos1[0] - last_pos_1[0], pos1[1] - last_pos_1[1]) <= MAX_JUMP_PX:
                xs1.append(pos1[0])
                ys1.append(pos1[1])
                last_pos_1 = pos1

            pos2 = detect_in_region(frame, last_pos_2, pose2)
            if pos2 is not None and np.hypot(pos2[0] - last_pos_2[0], pos2[1] - last_pos_2[1]) <= MAX_JUMP_PX:
                xs2.append(pos2[0])
                ys2.append(pos2[1])
                last_pos_2 = pos2

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

            if frame_idx >= FRAME_CAP:
                break

    cap.release()

    if debug:
        plt.ioff()
        plt.close(fig)

    if xs1:
        xs1, ys1 = smooth_positions(xs1, ys1, SMOOTH_WINDOW)
    if xs2:
        xs2, ys2 = smooth_positions(xs2, ys2, SMOOTH_WINDOW)

    print(f"Extracted {len(xs1)} positions for Player 1, {len(xs2)} for Player 2")

    court_xs1, court_ys1 = apply_homography(xs1, ys1, H) if xs1 else ([], [])
    court_xs2, court_ys2 = apply_homography(xs2, ys2, H) if xs2 else ([], [])

    plot_positions(xs1, ys1, xs2, ys2, background=first_frame, title="Player Movement (Pixel Space)")
    plot_court_positions(court_xs1, court_ys1, court_xs2, court_ys2, title="Player Movement (Court Space)")
    plot_heatmap(court_xs1, court_ys1, court_xs2, court_ys2)


if __name__ == "__main__":
    main()
