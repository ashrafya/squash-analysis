"""
TrackNetV4 inference — produces ball_positions.npz compatible with track_ball.py.

Slides a 3-frame window through the video.  For each triplet (t-1, t, t+1) the
model outputs three heatmaps; only the centre frame's heatmap is used as the
ball position for frame t.

Output: output/ball_positions.npz  (same format as detect_ball.py so that
        track_ball.py can consume it unchanged)
  frame_idx   — video frame numbers
  xs, ys      — pixel-space position (original video resolution)
  xs_m, ys_m  — court-space position (metres, via homography)
  confidence  — peak heatmap value [0, 1]
  method      — always 2 (METHOD_TRACKNET)

Usage:
  python src/tracknet_infer.py [--debug] [--calibrate] [--no-track]
                                [--ckpt PATH] [--threshold T]

  --ckpt       : checkpoint to use (default: assets/tracknet_ckpt/best.pt)
  --threshold  : heatmap threshold for detection (default 0.5)
  --debug      : show live detection overlay
  --calibrate  : force court recalibration
  --no-track   : skip running track_ball.py after inference
"""

import os
import sys
import argparse
import warnings

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

_PKG = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_PKG, "..", "..")
sys.path.insert(0, _PKG)   # for tracknet_model import
sys.path.insert(0, _SRC)   # for config / core imports
from config import (
    VIDEO_PATH, OUTPUT_DIR, FRAME_CAP, BALL_FRAME_SKIP,
    ANGLE_MATCH_THRESHOLD, VIDEO_FPS,
)
if not os.path.isabs(VIDEO_PATH):
    VIDEO_PATH = os.path.normpath(os.path.join(_SRC, VIDEO_PATH))
from calibrate import get_homography, apply_homography
from video_utils import load_video
from plot_utils import draw_court
from tracknet_model import TrackNetV4, build_input_tensor, heatmap_to_pixel, HEIGHT, WIDTH

_ROOT = os.path.join(_PKG, "..", "..", "..")
DEFAULT_CKPT = os.path.join(_ROOT, "assets", "tracknet_ckpt", "best.pt")
BALL_POSITIONS_PATH = os.path.join(OUTPUT_DIR, "ball_positions.npz")

METHOD_TRACKNET = 2   # stored in the 'method' array alongside YOLO=0, motion=1


# ── Camera-cut filter (reused from detect_ball.py) ────────────────────────────

def _compute_hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_trajectory(xs_m: list, ys_m: list, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 9))
    draw_court(ax)
    ax.scatter(xs_m, ys_m, s=10, alpha=0.7, color="yellow",
               edgecolors="darkorange", linewidths=0.5, zorder=5,
               label=f"TrackNetV4 ({len(xs_m)})")
    ax.legend(loc="upper right")
    ax.set_title("Ball Trajectory — TrackNetV4")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Trajectory plot saved: {save_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(ckpt_path: str = DEFAULT_CKPT, threshold: float = 0.5,
         debug: bool = False, calibrate: bool = False,
         run_tracker: bool = True):

    # ── Model ────────────────────────────────────────────────────────────────
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        print("Train the model first:  python src/tracknet_train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt.get("model_params", {"in_dim": 9, "out_dim": 3, "n_diff": 2})
    model = TrackNetV4(**params).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    # ── Video + homography ────────────────────────────────────────────────────
    cap, detected_fps, frame_count = load_video(VIDEO_PATH)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    orig_h, orig_w = first_frame.shape[:2]
    H_mat = get_homography(first_frame, force_recalibrate=calibrate)

    # Reference histogram for camera-cut detection
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ref_hist = _compute_hist(first_frame)

    total = min(frame_count, FRAME_CAP) if FRAME_CAP else frame_count

    # ── Output containers ────────────────────────────────────────────────────
    frame_idx_arr  = []
    xs, ys         = [], []
    xs_m, ys_m     = [], []
    confidence_arr = []
    method_arr     = []

    # Rolling buffer: keep the last 3 decoded frames
    buf: list[np.ndarray | None] = [None, None, None]
    frame_idx = 0
    frames_checked = 0
    skipped = 0

    # ── Debug window ─────────────────────────────────────────────────────────
    if debug:
        plt.ion()
        fig_d, (ax_vid, ax_court) = plt.subplots(
            1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [2, 1]}
        )
        fig_d.suptitle("Debug — TrackNetV4 inference  (yellow=detection)")
        ax_vid.axis("off")
        _im = ax_vid.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        draw_court(ax_court)
        _sc = ax_court.scatter([], [], s=15, color="yellow",
                               edgecolors="darkorange", linewidths=0.8, zorder=5)
        plt.tight_layout()
        plt.show()

    print(f"Running TrackNetV4 inference on {total} frames …")

    with tqdm(total=total, unit="frame") as pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        buf = [None, None, None]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Frame skip
            if BALL_FRAME_SKIP > 1 and frame_idx % BALL_FRAME_SKIP != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # Camera-cut filter
            corr = cv2.compareHist(ref_hist, _compute_hist(frame), cv2.HISTCMP_CORREL)
            if corr < ANGLE_MATCH_THRESHOLD:
                frame_idx += 1
                pbar.update(1)
                skipped += 1
                # Don't update buffer on camera cuts
                continue

            # Shift buffer
            buf[0] = buf[1]
            buf[1] = buf[2]
            buf[2] = frame.copy()
            frame_idx += 1
            pbar.update(1)

            # Need all 3 frames to run inference
            if buf[0] is None:
                continue

            frames_checked += 1
            center_fi = frame_idx - 1   # index of buf[1]

            # Build input tensor
            inp = build_input_tensor(buf, device=device)   # (1, 11, 288, 512)

            with torch.no_grad():
                out = model(inp)   # (1, 3, 288, 512)

            # Use centre frame heatmap (channel 1)
            hm = out[0, 1]   # (288, 512)

            det = heatmap_to_pixel(hm, orig_h, orig_w, threshold)

            if det is not None:
                bx, by, conf = det
                bx_m_list, by_m_list = apply_homography([bx], [by], H_mat)
                if bx_m_list and not np.isnan(bx_m_list[0]):
                    frame_idx_arr.append(center_fi)
                    xs.append(bx);              ys.append(by)
                    xs_m.append(bx_m_list[0]);  ys_m.append(by_m_list[0])
                    confidence_arr.append(conf)
                    method_arr.append(METHOD_TRACKNET)

            pbar.set_postfix(detected=len(xs), skipped=skipped)

            if debug and frames_checked % 10 == 0:
                vis = buf[1].copy()
                if det is not None:
                    bx, by, conf = det
                    cv2.circle(vis, (int(bx), int(by)), 10, (0, 255, 255), 2)
                    cv2.putText(vis, f"TNv4 {conf:.2f}",
                                (int(bx) + 12, int(by) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                _im.set_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                if xs_m:
                    _sc.set_offsets(np.c_[xs_m, ys_m])
                fig_d.canvas.draw_idle()
                plt.pause(0.001)

            if FRAME_CAP and frame_idx >= FRAME_CAP:
                break

    if debug:
        plt.ioff()
        plt.close(fig_d)

    cap.release()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(
        BALL_POSITIONS_PATH,
        frame_idx  = np.array(frame_idx_arr),
        xs         = np.array(xs),
        ys         = np.array(ys),
        xs_m       = np.array(xs_m),
        ys_m       = np.array(ys_m),
        confidence = np.array(confidence_arr),
        method     = np.array(method_arr),
    )

    detect_pct = len(xs) / max(1, frames_checked) * 100
    print(f"\nTrackNetV4 inference complete:")
    print(f"  Frames checked : {frames_checked}  (skipped {skipped} camera-cut frames)")
    print(f"  Detections     : {len(xs)} ({detect_pct:.1f}%)")
    print(f"  Saved          : {BALL_POSITIONS_PATH}")

    if xs_m:
        traj_path = os.path.join(OUTPUT_DIR, "ball_trajectory.png")
        _plot_trajectory(xs_m, ys_m, traj_path)

    # ── Optionally run Kalman smoother ────────────────────────────────────────
    if run_tracker and len(xs) > 0:
        print("\nRunning Kalman smoother (track_ball.py) …")
        import track_ball
        track_ball.main(debug=False, calibrate=False, make_anim=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description="TrackNetV4 inference")
    parser.add_argument("--ckpt",      default=DEFAULT_CKPT, help="Checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap threshold")
    parser.add_argument("--debug",     action="store_true", help="Live debug overlay")
    parser.add_argument("--calibrate", action="store_true", help="Force court recalibration")
    parser.add_argument("--no-track",  action="store_true", help="Skip track_ball.py after inference")
    args = parser.parse_args()

    main(ckpt_path=args.ckpt, threshold=args.threshold,
         debug=args.debug, calibrate=args.calibrate,
         run_tracker=not args.no_track)
