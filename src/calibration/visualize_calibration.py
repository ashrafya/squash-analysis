"""
visualize_calibration.py — interactive 3-D calibration viewer

Shows the squash court and recovered camera pose in a rotatable 3-D plot.

Usage:
    python src/visualize_calibration.py          # load saved calibration
    python src/visualize_calibration.py --video  # pick a specific frame first

Controls (matplotlib 3-D):
    Left-drag   — rotate
    Right-drag  — zoom
    Middle-drag — pan
    R           — reset view
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D                          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.calibrate import (
    load_best_calibration, CalibData,
    FLOOR_POINTS, WALL_POINTS,
    TIN_HEIGHT_M, SERVICE_LINE_HEIGHT_M,
    OUT_LINE_FRONT_M, OUT_LINE_BACK_M,
    _side_out_at,
)
from config import (
    COURT_WIDTH_M, COURT_LENGTH_M,
    SHORT_LINE_M, HALF_COURT_M, SERVICE_BOX_M,
    VIDEO_PATH, CALIBRATION_3D_PATH,
)

# ── Court geometry constants ───────────────────────────────────────────────────
W   = COURT_WIDTH_M
L   = COURT_LENGTH_M
SY  = L - SHORT_LINE_M          # 5.49 m — short line y
BOX = SERVICE_BOX_M             # 1.60 m
BX  = W - BOX                   # 4.80 m — right service-box x
BY  = SY + BOX                  # 7.09 m — service box back y

# ── Colour palette ─────────────────────────────────────────────────────────────
COL_FLOOR     = "#2a4a2a"
COL_FLOOR_A   = 0.25
COL_WALL      = "#1a3055"
COL_WALL_A    = 0.18
COL_LINE      = "#88cc88"
COL_LINE_W    = 1.2
COL_MARKING   = "#cc8844"   # wall markings (tin, service line, out-lines)
COL_CAM       = "#ff4444"
COL_FRUSTUM   = "#ff8888"
COL_AXIS_X    = "#ff4444"
COL_AXIS_Y    = "#44ff44"
COL_AXIS_Z    = "#4488ff"
COL_PT_FLOOR  = "#00ff88"
COL_PT_WALL   = "#ffaa00"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _poly(ax, verts, color, alpha):
    ax.add_collection3d(
        Poly3DCollection([verts], facecolor=color, edgecolor="none", alpha=alpha)
    )

def _line(ax, xs, ys, zs, color=COL_LINE, lw=COL_LINE_W, ls="-"):
    ax.plot(xs, ys, zs, color=color, linewidth=lw, linestyle=ls)

def _arrow(ax, origin, vec, color, length=1.0, lw=2):
    o = np.asarray(origin, dtype=float)
    v = np.asarray(vec,    dtype=float)
    v = v / (np.linalg.norm(v) + 1e-9) * length
    ax.quiver(*o, *v, color=color, linewidth=lw,
              arrow_length_ratio=0.25, normalize=False)


# ── Court drawing ──────────────────────────────────────────────────────────────

def draw_floor(ax):
    """Draw floor surface."""
    _poly(ax,
          [(0,0,0),(W,0,0),(W,L,0),(0,L,0)],
          COL_FLOOR, COL_FLOOR_A)

def draw_walls(ax):
    """Draw the four walls as semi-transparent surfaces up to the out-line."""
    # Front wall (y=0)
    _poly(ax,
          [(0,0,0),(W,0,0),(W,0,OUT_LINE_FRONT_M),(0,0,OUT_LINE_FRONT_M)],
          COL_WALL, COL_WALL_A)
    # Back wall (y=L)
    _poly(ax,
          [(0,L,0),(W,L,0),(W,L,OUT_LINE_BACK_M),(0,L,OUT_LINE_BACK_M)],
          COL_WALL, COL_WALL_A)
    # Left wall (x=0) — out-line slopes from front to back
    _poly(ax,
          [(0,0,0),(0,L,0),(0,L,OUT_LINE_BACK_M),(0,0,OUT_LINE_FRONT_M)],
          COL_WALL, COL_WALL_A)
    # Right wall (x=W)
    _poly(ax,
          [(W,0,0),(W,L,0),(W,L,OUT_LINE_BACK_M),(W,0,OUT_LINE_FRONT_M)],
          COL_WALL, COL_WALL_A)

def draw_floor_lines(ax):
    """Floor markings: court outline, short line, T, service boxes."""
    # Court outline
    for x0,x1,y0,y1 in [(0,W,0,0),(0,W,L,L),(0,0,0,L),(W,W,0,L)]:
        _line(ax, [x0,x1], [y0,y1], [0,0])
    # Short line
    _line(ax, [0,W], [SY,SY], [0,0])
    # Half-court line (T to back wall)
    _line(ax, [W/2,W/2], [SY,L], [0,0])
    # Service box lines
    _line(ax, [BOX,BOX], [SY,BY], [0,0])    # left box inner
    _line(ax, [BX, BX ], [SY,BY], [0,0])    # right box inner
    _line(ax, [0,  W  ], [BY,BY], [0,0])    # service box back line

def draw_wall_markings(ax):
    """Tin, service line, and out-lines on walls."""
    # Tin (front wall horizontal)
    _line(ax, [0,W], [0,0], [TIN_HEIGHT_M]*2,           color=COL_MARKING, lw=1.5)
    # Service line (front wall)
    _line(ax, [0,W], [0,0], [SERVICE_LINE_HEIGHT_M]*2,  color=COL_MARKING, lw=1.5, ls="--")
    # Out-line on front wall
    _line(ax, [0,W], [0,0], [OUT_LINE_FRONT_M]*2,       color=COL_MARKING, lw=1.5)
    # Out-line on back wall
    _line(ax, [0,W], [L,L], [OUT_LINE_BACK_M]*2,        color=COL_MARKING, lw=1.5)
    # Out-line on side walls (diagonal left)
    _line(ax, [0,0], [0,L], [OUT_LINE_FRONT_M, OUT_LINE_BACK_M], color=COL_MARKING, lw=1.5)
    # Out-line on side walls (diagonal right)
    _line(ax, [W,W], [0,L], [OUT_LINE_FRONT_M, OUT_LINE_BACK_M], color=COL_MARKING, lw=1.5)

def draw_calibration_points(ax, include_walls: bool = True):
    """Plot the 3-D world positions of every calibration point."""
    pts = FLOOR_POINTS + (WALL_POINTS if include_walls else [])
    for i, (label, (x, y, z)) in enumerate(pts):
        col = COL_PT_WALL if z > 0 else COL_PT_FLOOR
        ax.scatter([x], [y], [z], c=col, s=30, zorder=5)
        ax.text(x + 0.05, y, z + 0.08, str(i + 1),
                fontsize=6, color=col, zorder=6)


# ── Camera drawing ─────────────────────────────────────────────────────────────

def _frustum_floor_pts(calib: CalibData, img_w: int, img_h: int):
    """Project image corners through camera to z=0 floor plane."""
    corners = np.array([[0,0],[img_w,0],[img_w,img_h],[0,img_h]], dtype=np.float64)
    norm = cv2.undistortPoints(corners.reshape(-1,1,2), calib.K, calib.dist).reshape(-1,2)
    rays_cam   = np.column_stack([norm, np.ones(4)])
    rays_world = (calib.R.T @ rays_cam.T).T         # (4,3)
    C = calib.C
    pts = []
    for ray in rays_world:
        if abs(ray[2]) < 1e-8:
            pts.append(None)
            continue
        t = -C[2] / ray[2]
        pts.append(C + t * ray if t > 0 else None)
    return pts

def draw_camera(ax, calib: CalibData, img_w: int = 640, img_h: int = 360):
    """Draw camera centre, coordinate axes, and viewing frustum."""
    C = calib.C    # (3,) world position

    # Camera centre
    ax.scatter([C[0]], [C[1]], [C[2]], c=COL_CAM, s=80, zorder=10, label="Camera centre")

    # Camera coordinate axes in world space
    axis_len = 0.6
    for col, cam_vec in [(COL_AXIS_X, [1,0,0]),
                          (COL_AXIS_Y, [0,1,0]),
                          (COL_AXIS_Z, [0,0,1])]:
        world_vec = calib.R.T @ np.array(cam_vec, dtype=float)
        _arrow(ax, C, world_vec, col, length=axis_len, lw=2)

    # Viewing direction (camera -Z = principal axis, points toward court)
    view_dir = calib.R.T @ np.array([0, 0, 1], dtype=float)
    _arrow(ax, C, view_dir, COL_CAM, length=2.0, lw=3)

    # Frustum: lines from camera to floor corners
    floor_pts = _frustum_floor_pts(calib, img_w, img_h)
    for fp in floor_pts:
        if fp is not None:
            _line(ax, [C[0], fp[0]], [C[1], fp[1]], [C[2], fp[2]],
                  color=COL_FRUSTUM, lw=0.8, ls="--")

    # Frustum floor outline
    valid = [fp for fp in floor_pts if fp is not None]
    if len(valid) == 4:
        xs = [p[0] for p in valid] + [valid[0][0]]
        ys = [p[1] for p in valid] + [valid[0][1]]
        zs = [p[2] for p in valid] + [valid[0][2]]
        _line(ax, xs, ys, zs, color=COL_FRUSTUM, lw=1.5)

    # Camera info text
    ax.text(C[0], C[1], C[2] + 0.2,
            f"Cam  ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}) m",
            fontsize=7, color=COL_CAM)


# ── Axis limits ────────────────────────────────────────────────────────────────

def _set_equal_tight_limits(ax, calib: "CalibData"):
    """Equal scale on all three axes, court-centred.

    The court is the reference object (9.75 × 6.4 m, up to 4.57 m tall).
    The camera is included in the bounding box only if it sits within a
    plausible range (≤ 6 m behind the back wall, ≤ 8 m above floor) so
    that a bad solvePnP result cannot push the camera to y=20 m and make
    the court look tiny / wrong.
    """
    C   = calib.C
    pad = 1.0   # metres of whitespace around the scene

    # Court bounds
    x_lo, x_hi = -pad,                  W + pad
    y_lo, y_hi = -pad,                  L + pad
    z_lo, z_hi = -0.3,                  OUT_LINE_FRONT_M + pad

    # Expand to include camera only if it looks physically plausible
    cam_y_max = L + 6.0   # back wall + 6 m
    cam_z_max = 8.0       # ceiling height with margin
    if C[0] > x_lo and C[0] < W + 8:
        x_lo = min(x_lo, C[0] - pad)
        x_hi = max(x_hi, C[0] + pad)
    if C[1] < cam_y_max:
        y_hi = max(y_hi, C[1] + pad)
    if C[1] < 0:
        y_lo = min(y_lo, C[1] - pad)
    if C[2] < cam_z_max:
        z_hi = max(z_hi, C[2] + pad)

    # Equal scale: largest span sets the cube size
    span = max(x_hi - x_lo, y_hi - y_lo, z_hi - z_lo) / 2.0
    mx   = (x_lo + x_hi) / 2.0
    my   = (y_lo + y_hi) / 2.0
    mz   = (z_lo + z_hi) / 2.0

    ax.set_xlim3d(mx - span, mx + span)
    ax.set_ylim3d(my - span, my + span)
    ax.set_zlim3d(mz - span, mz + span)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive 3-D calibration viewer")
    parser.add_argument("--no-walls",   action="store_true",
                        help="Skip wall calibration points")
    parser.add_argument("--img-size",   nargs=2, type=int, metavar=("W","H"),
                        default=None,
                        help="Image resolution for frustum (default: read from video)")
    args = parser.parse_args()

    # ── Load calibration ───────────────────────────────────────────────────────
    if not os.path.exists(CALIBRATION_3D_PATH):
        print("ERROR: No 3-D calibration found.")
        print("  Run:  python src/run_calibrate.py")
        sys.exit(1)

    calib, _ = load_best_calibration()
    if calib is None:
        print("ERROR: Could not load CalibData.")
        sys.exit(1)

    # ── Image size for frustum ─────────────────────────────────────────────────
    if args.img_size:
        img_w, img_h = args.img_size
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if img_w == 0:
            img_w, img_h = 640, 360
        print(f"Video resolution: {img_w}×{img_h}")

    # ── Print camera summary ───────────────────────────────────────────────────
    C = calib.C
    print(f"\nCamera centre (world):")
    print(f"  x = {C[0]:.3f} m  (0=left wall, {W:.1f}=right wall)")
    print(f"  y = {C[1]:.3f} m  (0=front wall, {L:.2f}=back wall)")
    print(f"  z = {C[2]:.3f} m  (height above floor)")

    view = calib.R.T @ np.array([0,0,1], dtype=float)
    print(f"Viewing direction: ({view[0]:.3f}, {view[1]:.3f}, {view[2]:.3f})")

    # ── Build figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#111111")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("#333333")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # ── Draw everything ────────────────────────────────────────────────────────
    draw_floor(ax)
    draw_walls(ax)
    draw_floor_lines(ax)
    draw_wall_markings(ax)
    draw_calibration_points(ax, include_walls=not args.no_walls)
    draw_camera(ax, calib, img_w, img_h)

    # ── Labels ─────────────────────────────────────────────────────────────────
    ax.set_xlabel("x  (m)  ←left  right→", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("y  (m)  front→back",    color="#aaaaaa", fontsize=8)
    ax.set_zlabel("z  (m)  height",        color="#aaaaaa", fontsize=8)

    # Annotate court corners
    for lbl, (x,y,z) in [("Front-L",(0,0,0)), ("Front-R",(W,0,0)),
                           ("Back-L", (0,L,0)), ("Back-R", (W,L,0))]:
        ax.text(x, y, z - 0.15, lbl, fontsize=6, color="#666666")

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=COL_FLOOR,   alpha=0.7, label="Court floor"),
        mpatches.Patch(facecolor=COL_WALL,    alpha=0.7, label="Walls"),
        mpatches.Patch(facecolor=COL_MARKING, alpha=0.9, label="Wall markings"),
        mpatches.Patch(facecolor=COL_PT_FLOOR,alpha=0.9, label="Floor calib pts"),
        mpatches.Patch(facecolor=COL_PT_WALL, alpha=0.9, label="Wall calib pts"),
        mpatches.Patch(facecolor=COL_CAM,     alpha=0.9, label="Camera"),
        mpatches.Patch(facecolor=COL_AXIS_X,  alpha=0.9, label="Cam X-axis"),
        mpatches.Patch(facecolor=COL_AXIS_Y,  alpha=0.9, label="Cam Y-axis"),
        mpatches.Patch(facecolor=COL_AXIS_Z,  alpha=0.9, label="Cam Z-axis (view)"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              fontsize=7, facecolor="#222222", labelcolor="#cccccc",
              edgecolor="#444444", framealpha=0.85)

    ax.set_title(
        "Squash court — 3-D calibration view\n"
        "Left-drag: rotate  |  Right-drag: zoom  |  Middle-drag: pan",
        color="#cccccc", fontsize=9, pad=10,
    )

    _set_equal_tight_limits(ax, calib)

    # ── Initial viewpoint: from behind and above (like a coach watching) ───────
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
