"""
debug_3d_player.py — side-by-side video + live 3D court view.

Loads saved player positions and plays them against the video so you can see
exactly where the tracker thinks each player is in 3D court space.

Usage
-----
    python src/evaluation/debug_3d_player.py
    python src/evaluation/debug_3d_player.py --positions output/last_positions_yolo.npz
    python src/evaluation/debug_3d_player.py --video assets/video/myvideo.mp4

Controls
--------
    SPACE       pause / resume
    ← / →       step one tracked frame while paused
    Q / ESC     quit
    +  /  -     speed up / slow down playback
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from calibration.calibrate import (
    load_best_calibration,
    TIN_HEIGHT_M, SERVICE_LINE_HEIGHT_M, OUT_LINE_FRONT_M, OUT_LINE_BACK_M,
)
from config import (
    VIDEO_PATH, OUTPUT_DIR, VIDEO_FPS, FRAME_SKIP,
    COURT_WIDTH_M, COURT_LENGTH_M, SHORT_LINE_M, HALF_COURT_M, SERVICE_BOX_M,
)
from utils.video_utils import load_video

_POSITIONS_PATH = os.path.join(OUTPUT_DIR, "last_positions_yolo.npz")

# ── court geometry ─────────────────────────────────────────────────────────────
_W      = COURT_WIDTH_M
_L      = COURT_LENGTH_M
_SHORT  = _L - SHORT_LINE_M          # 5.49 m from front wall
_BOX_Y  = _SHORT + SERVICE_BOX_M     # 7.09 m
_RIGHT  = _W - SERVICE_BOX_M         # 4.80 m

# ── colours ────────────────────────────────────────────────────────────────────
_C_P1_MPL = (0.95, 0.35, 0.35)
_C_P2_MPL = (0.35, 0.65, 1.00)
_C_P1_BGR = (60,  80,  240)
_C_P2_BGR = (220, 160,  60)

def _set_equal_aspect(ax):
    """Force equal axis scaling on a 3D matplotlib axis."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans  = limits[:, 1] - limits[:, 0]
    centres = limits.mean(axis=1)
    radius = spans.max() / 2
    ax.set_xlim3d(centres[0] - radius, centres[0] + radius)
    ax.set_ylim3d(centres[1] - radius, centres[1] + radius)
    ax.set_zlim3d(centres[2] - radius, centres[2] + radius)


_TRAIL_LEN  = 50    # frames of trail to show
_3D_EVERY   = 4     # re-render 3D every N position-frames (performance)
_3D_SIZE    = 520   # pixels for the 3D window


# ── pre-built court wireframe (list of segment endpoints) ──────────────────────

def _court_lines():
    segs = []

    def seg(x0, y0, z0, x1, y1, z1, style="solid"):
        segs.append(((x0, y0, z0), (x1, y1, z1), style))

    # Floor outline
    seg(0,  0,  0,  _W, 0,  0)
    seg(_W, 0,  0,  _W, _L, 0)
    seg(_W, _L, 0,  0,  _L, 0)
    seg(0,  _L, 0,  0,  0,  0)
    # Short line
    seg(0,  _SHORT, 0,  _W, _SHORT, 0)
    # Half-court (T to back)
    seg(HALF_COURT_M, _SHORT, 0, HALF_COURT_M, _L, 0)
    # Service box lines
    seg(SERVICE_BOX_M, _SHORT, 0, SERVICE_BOX_M, _BOX_Y, 0)
    seg(_RIGHT,        _SHORT, 0, _RIGHT,        _BOX_Y, 0)
    seg(0, _BOX_Y, 0, _W, _BOX_Y, 0)
    # Front wall face
    seg(0, 0, 0, _W, 0, 0)
    seg(0, 0, TIN_HEIGHT_M,          _W, 0, TIN_HEIGHT_M,          "dashed")
    seg(0, 0, SERVICE_LINE_HEIGHT_M, _W, 0, SERVICE_LINE_HEIGHT_M, "dashed")
    seg(0, 0, OUT_LINE_FRONT_M,      _W, 0, OUT_LINE_FRONT_M)
    seg(0, 0, 0, 0, 0, OUT_LINE_FRONT_M)
    seg(_W, 0, 0, _W, 0, OUT_LINE_FRONT_M)
    # Side wall out-lines
    seg(0,  0, OUT_LINE_FRONT_M,  0,  _L, OUT_LINE_BACK_M)
    seg(_W, 0, OUT_LINE_FRONT_M,  _W, _L, OUT_LINE_BACK_M)

    return segs


_COURT_SEGS = _court_lines()


# ── 3D renderer ────────────────────────────────────────────────────────────────

def _render_3d(p1, p2, trail1, trail2, cam_pos, size=_3D_SIZE):
    dpi = 80
    fig    = Figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor="#0e0e0e")
    canvas = FigureCanvasAgg(fig)
    ax     = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0e0e0e")
    fig.patch.set_facecolor("#0e0e0e")

    # Court wireframe
    for (x0, y0, z0), (x1, y1, z1), style in _COURT_SEGS:
        ls  = "--" if style == "dashed" else "-"
        col = "#444444" if style == "dashed" else "#606060"
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=col, lw=0.8, ls=ls)

    # Trails
    for trail, col in [(trail1, _C_P1_MPL), (trail2, _C_P2_MPL)]:
        if len(trail) < 2:
            continue
        txs = [p[0] for p in trail]
        tys = [p[1] for p in trail]
        n   = len(trail)
        for i in range(n - 1):
            alpha = 0.08 + 0.72 * (i / n)
            ax.plot(txs[i:i+2], tys[i:i+2], [0, 0],
                    color=col, lw=1.8, alpha=alpha, solid_capstyle="round")

    # Current player positions
    for pos, col, label in [(p1, _C_P1_MPL, "P1"), (p2, _C_P2_MPL, "P2")]:
        if pos is None or any(np.isnan(pos)):
            continue
        ax.scatter([pos[0]], [pos[1]], [0],
                   c=[col], s=110, zorder=10,
                   edgecolors="white", linewidths=0.6)
        ax.text(pos[0], pos[1], 0.15, label,
                color="white", fontsize=6, ha="center", va="bottom")

    # Camera position
    if cam_pos is not None:
        ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                   c="white", s=35, marker="^", alpha=0.4, zorder=5)

    # Axes & style
    ax.set_xlim3d(-0.3, _W + 0.3)
    ax.set_ylim3d(-0.3, _L + 0.3)
    ax.set_zlim3d(0, 5.5)
    _set_equal_aspect(ax)
    ax.set_xlabel("X (m)", fontsize=5, color="#666666", labelpad=1)
    ax.set_ylabel("Y (m)", fontsize=5, color="#666666", labelpad=1)
    ax.set_zlabel("Z (m)", fontsize=5, color="#666666", labelpad=1)
    ax.tick_params(labelsize=4, colors="#555555")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#222222")
    ax.grid(True, color="#222222", lw=0.3)
    ax.view_init(elev=22, azim=-115)

    fig.tight_layout(pad=0.2)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)


# ── main ───────────────────────────────────────────────────────────────────────

def run(positions_path: str, video_path: str):
    if not os.path.exists(positions_path):
        print(f"Positions file not found: {positions_path}")
        print("Run the tracker first:  python src/main.py")
        return

    # Load positions
    data   = np.load(positions_path)
    xs1    = data["xs1"].tolist()
    ys1    = data["ys1"].tolist()
    xs2    = data["xs2"].tolist()
    ys2    = data["ys2"].tolist()
    n_pos  = len(xs1)
    print(f"Loaded {n_pos} tracked frames from {positions_path}")

    # Load video
    cap, fps, _ = load_video(video_path)
    ret, first  = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not ret:
        print("Could not read video.")
        return

    # Camera position for 3D display
    calib, _ = load_best_calibration(first)
    cam_pos  = tuple(calib.C) if calib is not None else None

    fh, fw  = first.shape[:2]
    scale   = min(1.0, 860 / fw)
    dw, dh  = int(fw * scale), int(fh * scale)

    WIN_VID = "Video — player tracking"
    WIN_3D  = "3D Court View"
    cv2.namedWindow(WIN_VID, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_3D,  cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_VID, dw, dh + 44)
    cv2.resizeWindow(WIN_3D,  _3D_SIZE, _3D_SIZE)
    cv2.moveWindow(WIN_VID,  40,         50)
    cv2.moveWindow(WIN_3D,   dw + 60,    50)

    paused      = False
    idx         = 0
    delay_ms    = max(1, int(1000 / fps * FRAME_SKIP))
    trail1: list = []
    trail2: list = []
    last_3d     = -999
    vid_pos     = 0   # which video frame the cap is currently pointing at
    last_frame  = None

    while idx < n_pos:
        target = idx * FRAME_SKIP

        # ── advance video to the target frame ──────────────────────────────
        if target < vid_pos:
            # Backward seek (only happens when stepping left while paused)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            vid_pos = target

        # Fast-forward by grabbing (no decode) until we're one frame before target
        while vid_pos < target:
            if not cap.grab():
                break
            vid_pos += 1

        ret, frame = cap.read()
        if ret:
            vid_pos += 1
            last_frame = frame
        elif last_frame is not None:
            frame = last_frame   # use last good frame if video lags behind positions
        else:
            print(f"Could not read any video frame at position {target}. "
                  f"Check VIDEO_PATH in config.py.")
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (dw, dh))

        p1 = (xs1[idx], ys1[idx])
        p2 = (xs2[idx], ys2[idx])

        # Update trails
        trail1.append(p1)
        trail2.append(p2)
        if len(trail1) > _TRAIL_LEN:
            trail1.pop(0)
        if len(trail2) > _TRAIL_LEN:
            trail2.pop(0)

        # Status bar on video
        bar = np.zeros((44, dw, 3), dtype=np.uint8)
        bar[:] = (18, 18, 18)
        cv2.putText(bar,
                    f"P1  x={p1[0]:.2f}m  y={p1[1]:.2f}m",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _C_P1_BGR, 1)
        cv2.putText(bar,
                    f"P2  x={p2[0]:.2f}m  y={p2[1]:.2f}m",
                    (dw // 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _C_P2_BGR, 1)
        cv2.putText(bar,
                    f"frame {idx * FRAME_SKIP}  {'[PAUSED]' if paused else ''}",
                    (dw - 180, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
        cv2.putText(bar,
                    "SPC=pause  </>=step  +/-=speed  Q=quit",
                    (dw - 300, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)
        cv2.imshow(WIN_VID, np.vstack([frame, bar]))

        # 3D update (only every _3D_EVERY frames for performance)
        if idx - last_3d >= _3D_EVERY:
            img3d = _render_3d(p1, p2, trail1, trail2, cam_pos)
            cv2.imshow(WIN_3D, img3d)
            last_3d = idx

        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 81 and paused:        # ← arrow
            idx = max(0, idx - 1)
            vid_pos = (idx + 1) * FRAME_SKIP + 1  # force re-seek on next iteration
            trail1.clear(); trail2.clear()
        elif key == 83 and paused:        # → arrow
            idx = min(n_pos - 1, idx + 1)
        elif key in (ord('+'), ord('=')):
            delay_ms = max(1, delay_ms - 10)
            print(f"  delay {delay_ms} ms")
        elif key in (ord('-'), ord('_')):
            delay_ms = min(500, delay_ms + 10)
            print(f"  delay {delay_ms} ms")
        elif not paused:
            idx += 1

    cv2.destroyAllWindows()
    cap.release()
    print("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Side-by-side video + 3D court player view.")
    ap.add_argument("--positions", default=_POSITIONS_PATH,
                    help="NPZ positions file (default: output/last_positions_yolo.npz)")
    ap.add_argument("--video",     default=VIDEO_PATH,
                    help="Video file (default: from config)")
    args = ap.parse_args()
    run(args.positions, args.video)
