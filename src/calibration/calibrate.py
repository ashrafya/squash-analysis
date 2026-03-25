"""
calibrate.py — court calibration (2D homography + 3D solvePnP)

Two modes are supported, chosen automatically at load time:

  3D mode  (preferred)
    Uses cv2.solvePnP to recover the camera's intrinsic matrix, rotation
    vector and translation vector from up to 22 clicked points across three
    planes (floor z=0, front wall y=0, side walls x=0/6.4).
    Any pixel (u,v) is mapped to court coordinates by casting a ray through
    the camera centre and intersecting it with the court floor plane (z=0).
    This corrects the parallax error that the 2D homography introduces when
    hip or knee keypoints are used instead of ankles.

  2D mode  (legacy / fallback)
    Plain cv2.findHomography mapping pixel → court (metres).

Collection is split into two phases:
  Phase 1: 14 floor points      (z = 0, same as the original calibration)
  Phase 2:  8 wall points       (y = 0 front wall / x=0,6.4 side walls, z > 0)
             — front wall tin corners
             — front wall service line corners
             — side wall out-line at short line
             — back wall out-line corners (side wall meets back wall)
  Phase 2 is OPTIONAL and skipped if the user presses 'S' at the prompt.

Public API (unchanged for callers):
    get_homography(frame, force_recalibrate=False)  → H            (2D)
    get_calibration(frame, force_recalibrate=False) → CalibData    (3D)
    apply_homography(xs, ys, H)                     → (xs_m, ys_m)
    project_to_court(xs, ys, calib)                 → (xs_m, ys_m)
    load_best_calibration()  → (calib_3d or None, H_2d or None)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from config import (
    COURT_WIDTH_M,
    COURT_LENGTH_M,
    SHORT_LINE_M,
    HALF_COURT_M,
    SERVICE_BOX_M,
    CALIBRATION_PATH,
    CALIBRATION_3D_PATH,
)

# ── WSF standard wall heights (metres) ────────────────────────────────────────
TIN_HEIGHT_M          = 0.48   # top of tin (bottom of playable front wall)
SERVICE_LINE_HEIGHT_M = 1.78   # front wall service line
OUT_LINE_FRONT_M      = 4.57   # out line on front wall
OUT_LINE_BACK_M       = 2.13   # out line at back wall / side wall junction

# Out-line height on side walls at a given y (linear from front to back)
def _side_out_at(y: float) -> float:
    return OUT_LINE_FRONT_M + (OUT_LINE_BACK_M - OUT_LINE_FRONT_M) * y / COURT_LENGTH_M

# ── Derived floor positions ────────────────────────────────────────────────────
_SHORT_Y = COURT_LENGTH_M - SHORT_LINE_M        # 5.49 m
_BOX_Y   = _SHORT_Y + SERVICE_BOX_M             # 7.09 m
_RIGHT_X = COURT_WIDTH_M - SERVICE_BOX_M        # 4.80 m

# ── Derived mid-court positions ────────────────────────────────────────────────
_MID_Y = COURT_LENGTH_M / 2          # 4.875 m — halfway down court

# ── Phase-1: 19 floor points (z = 0) ─────────────────────────────────────────
FLOOR_POINTS = [
    # ── corners ────────────────────────────────────────────────────────────────
    ("Front-left corner",              (0,             0,              0.0)),
    ("Front-right corner",             (COURT_WIDTH_M, 0,              0.0)),
    ("Back-left corner",               (0,             COURT_LENGTH_M, 0.0)),
    ("Back-right corner",              (COURT_WIDTH_M, COURT_LENGTH_M, 0.0)),
    # ── short line & T ─────────────────────────────────────────────────────────
    ("Short line - left wall",         (0,             _SHORT_Y,       0.0)),
    ("Short line - right wall",        (COURT_WIDTH_M, _SHORT_Y,       0.0)),
    ("T-junction",                     (HALF_COURT_M,  _SHORT_Y,       0.0)),
    ("Half-court - back wall",         (HALF_COURT_M,  COURT_LENGTH_M, 0.0)),
    # ── service boxes ──────────────────────────────────────────────────────────
    ("Left box inner-top",             (SERVICE_BOX_M, _SHORT_Y,       0.0)),
    ("Left box back-inner",            (SERVICE_BOX_M, _BOX_Y,         0.0)),
    ("Left box back-outer",            (0,             _BOX_Y,         0.0)),
    ("Right box inner-top",            (_RIGHT_X,      _SHORT_Y,       0.0)),
    ("Right box back-inner",           (_RIGHT_X,      _BOX_Y,         0.0)),
    ("Right box back-outer",           (COURT_WIDTH_M, _BOX_Y,         0.0)),
    # ── extra front-court & mid-court coverage ─────────────────────────────────
    ("Front wall - mid floor",         (HALF_COURT_M,  0,              0.0)),
    ("Half-court at service box line", (HALF_COURT_M,  _BOX_Y,         0.0)),
    # side walls — front court (before short line)
    ("Left wall - near front",         (0,             _SHORT_Y * 0.25, 0.0)),
    ("Right wall - near front",        (COURT_WIDTH_M, _SHORT_Y * 0.25, 0.0)),
    ("Left wall - mid front court",    (0,             _SHORT_Y * 0.5,  0.0)),
    ("Right wall - mid front court",   (COURT_WIDTH_M, _SHORT_Y * 0.5,  0.0)),
    # side walls — mid court
    ("Left wall - mid court floor",    (0,             _MID_Y,         0.0)),
    ("Right wall - mid court floor",   (COURT_WIDTH_M, _MID_Y,         0.0)),
    # side walls — back court (between service box and back wall)
    ("Left wall - back court",         (0,             (_BOX_Y + COURT_LENGTH_M) * 0.5, 0.0)),
    ("Right wall - back court",        (COURT_WIDTH_M, (_BOX_Y + COURT_LENGTH_M) * 0.5, 0.0)),
]

# Legacy (x,y) 2-tuple list for backward-compat callers that do CALIBRATION_POINTS[i][1]
CALIBRATION_POINTS = [(lbl, (x, y)) for lbl, (x, y, _) in FLOOR_POINTS]

# ── Phase-2: 17 wall points (z > 0) ───────────────────────────────────────────
WALL_POINTS = [
    # Front wall — tin (bottom of playable area, 0.48 m)
    ("Front wall tin - left",              (0,             0, TIN_HEIGHT_M         )),
    ("Front wall tin - mid",               (HALF_COURT_M,  0, TIN_HEIGHT_M         )),
    ("Front wall tin - right",             (COURT_WIDTH_M, 0, TIN_HEIGHT_M         )),
    # Front wall — service line (1.78 m)
    ("Front wall service line - left",     (0,             0, SERVICE_LINE_HEIGHT_M)),
    ("Front wall service line - mid",      (HALF_COURT_M,  0, SERVICE_LINE_HEIGHT_M)),
    ("Front wall service line - right",    (COURT_WIDTH_M, 0, SERVICE_LINE_HEIGHT_M)),
    # Front wall — out line (4.57 m — top of front wall)
    ("Front wall out-line - left",         (0,             0, OUT_LINE_FRONT_M     )),
    ("Front wall out-line - mid",          (HALF_COURT_M,  0, OUT_LINE_FRONT_M     )),
    ("Front wall out-line - right",        (COURT_WIDTH_M, 0, OUT_LINE_FRONT_M     )),
    # Side wall out-line at the short line
    ("Left wall out-line at short line",   (0,             _SHORT_Y,       _side_out_at(_SHORT_Y))),
    ("Right wall out-line at short line",  (COURT_WIDTH_M, _SHORT_Y,       _side_out_at(_SHORT_Y))),
    # Side wall out-line at mid-court
    ("Left wall out-line at mid court",    (0,             _MID_Y,         _side_out_at(_MID_Y)  )),
    ("Right wall out-line at mid court",   (COURT_WIDTH_M, _MID_Y,         _side_out_at(_MID_Y)  )),
    # Side wall out-line at the back wall (2.13 m)
    ("Left wall out-line at back wall",    (0,             COURT_LENGTH_M, OUT_LINE_BACK_M       )),
    ("Right wall out-line at back wall",   (COURT_WIDTH_M, COURT_LENGTH_M, OUT_LINE_BACK_M       )),
]

# ── Combined 3-D world points ──────────────────────────────────────────────────
def _make_world_pts(include_walls: bool) -> np.ndarray:
    pts = FLOOR_POINTS + (WALL_POINTS if include_walls else [])
    return np.array([[x, y, z] for _, (x, y, z) in pts], dtype=np.float64)


# ── Calibration data container ─────────────────────────────────────────────────

@dataclass
class CalibData:
    """Camera calibration for 3D ray-plane projection."""
    K:    np.ndarray  # (3,3) intrinsic matrix
    dist: np.ndarray  # (5,)  distortion coefficients
    rvec: np.ndarray  # (3,1) rotation vector
    tvec: np.ndarray  # (3,1) translation vector
    # Derived once at construction for speed
    R:    np.ndarray = field(default=None, init=False)
    C:    np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.C    = (-self.R.T @ self.tvec).flatten()


# ── Per-point hints (shown as a second line in the calibration window) ─────────

_FLOOR_HINTS = [
    # corners
    "Where the front wall meets the left side wall at the floor",
    "Where the front wall meets the right side wall at the floor",
    "Where the back wall meets the left side wall at the floor",
    "Where the back wall meets the right side wall at the floor",
    # short line & T
    "Where the short line meets the left side wall",
    "Where the short line meets the right side wall",
    "Centre of the short line — where the half-court line meets it (the T)",
    "Where the half-court line meets the back wall",
    # service boxes
    "Left service box — inner corner on the short line (1.6m from left wall)",
    "Left service box — back-inner corner (1.6m from left wall, behind short line)",
    "Left service box — back-outer corner (left wall, behind short line)",
    "Right service box — inner corner on the short line (1.6m from right wall)",
    "Right service box — back-inner corner (1.6m from right wall, behind short line)",
    "Right service box — back-outer corner (right wall, behind short line)",
    # extra front-court & mid-court coverage
    "Centre of the front wall at floor level (3.2m from each side wall)",
    "Half-court line meets the service box back line (3.2m from each wall, behind short line)",
    "Left side wall floor ~1.37m from the front wall (quarter of short-line depth)",
    "Right side wall floor ~1.37m from the front wall (quarter of short-line depth)",
    "Left side wall floor ~2.75m from the front wall (halfway to short line)",
    "Right side wall floor ~2.75m from the front wall (halfway to short line)",
    "Left side wall floor at half-court depth (4.875m from front wall)",
    "Right side wall floor at half-court depth (4.875m from front wall)",
    "Left side wall floor midway between service box line and back wall (~8.4m)",
    "Right side wall floor midway between service box line and back wall (~8.4m)",
]

_WALL_HINTS = [
    # tin (0.48 m)
    "Top edge of the tin strip — left end of front wall (0.48 m high)",
    "Top edge of the tin strip — centre of front wall (0.48 m high)",
    "Top edge of the tin strip — right end of front wall (0.48 m high)",
    # service line (1.78 m)
    "Service line on the front wall — left end (1.78 m high)",
    "Service line on the front wall — centre (1.78 m high)",
    "Service line on the front wall — right end (1.78 m high)",
    # out-line on front wall (4.57 m)
    "Out-line on the front wall — left end (4.57 m high, top of front wall)",
    "Out-line on the front wall — centre (4.57 m high)",
    "Out-line on the front wall — right end (4.57 m high)",
    # side wall out-line at short line
    "Out-line on the left side wall where the short line meets it (~3.2 m high)",
    "Out-line on the right side wall where the short line meets it (~3.2 m high)",
    # side wall out-line at mid-court
    "Out-line on the left side wall at mid-court depth (~2.7 m high)",
    "Out-line on the right side wall at mid-court depth (~2.7 m high)",
    # side wall out-line at back wall
    "Out-line on the left side wall at the back wall (2.13 m high)",
    "Out-line on the right side wall at the back wall (2.13 m high)",
]


# ── Court diagram panel ────────────────────────────────────────────────────────

def _draw_court_diagram(point_list: list,
                         done_count: int,
                         panel_w: int = 200,
                         panel_h: int = 310) -> np.ndarray:
    """Return a top-down court diagram image highlighting calibration points.

    Colors: green = already clicked, yellow = click next, grey = pending.
    Only floor points (z=0) are drawn; wall points get a simple legend row.
    """
    img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    # Court area inside the panel (leave 15px margin)
    M = 15
    cw = panel_w - 2 * M   # 170
    ch = panel_h - 2 * M   # 280

    def to_px(x_m, y_m):
        px = int(M + x_m / COURT_WIDTH_M  * cw)
        py = int(M + y_m / COURT_LENGTH_M * ch)
        return px, py

    # Court outline
    cv2.rectangle(img, (M, M), (M + cw, M + ch), (180, 180, 180), 1)

    # Short line
    sl_y = int(M + _SHORT_Y / COURT_LENGTH_M * ch)
    cv2.line(img, (M, sl_y), (M + cw, sl_y), (120, 120, 120), 1)

    # Half-court line (T to back wall only)
    hc_x = int(M + HALF_COURT_M / COURT_WIDTH_M * cw)
    cv2.line(img, (hc_x, sl_y), (hc_x, M + ch), (100, 100, 100), 1)

    # Service box lines
    box_x_l = int(M + SERVICE_BOX_M / COURT_WIDTH_M * cw)
    box_x_r = int(M + _RIGHT_X     / COURT_WIDTH_M * cw)
    box_y   = int(M + _BOX_Y       / COURT_LENGTH_M * ch)
    cv2.line(img, (box_x_l, sl_y), (box_x_l, box_y), (80, 80, 80), 1)
    cv2.line(img, (box_x_r, sl_y), (box_x_r, box_y), (80, 80, 80), 1)
    cv2.line(img, (M,       box_y), (M + cw, box_y), (80, 80, 80), 1)

    # Draw calibration points
    for i, (_, coords) in enumerate(point_list):
        x_m, y_m, z_m = coords
        if z_m > 0:
            continue   # skip wall points from floor diagram
        px, py = to_px(x_m, y_m)
        if i < done_count:
            color, r = (0, 220, 80), 5       # green — done
        elif i == done_count:
            color, r = (0, 220, 255), 7      # yellow — click next
        else:
            color, r = (100, 100, 100), 4    # grey — pending

        cv2.circle(img, (px, py), r, color, -1)
        cv2.putText(img, str(i + 1), (px + 5, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Legend
    lx, ly = M, panel_h - 8
    for col, txt in [((0, 220, 80), "done"), ((0, 220, 255), "next"), ((100, 100, 100), "pending")]:
        cv2.circle(img, (lx + 5, ly - 3), 4, col, -1)
        cv2.putText(img, txt, (lx + 12, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
        lx += 55

    return img


# ── Live 3-D calibration preview ───────────────────────────────────────────────

def _render_3d_preview(pixel_pts: list,
                        point_list: list,
                        frame_w: int,
                        frame_h: int,
                        prior_pixel_pts: Optional[np.ndarray] = None,
                        prior_point_list: Optional[list] = None,
                        out_size: int = 500) -> np.ndarray:
    """Render a 3-D court + camera preview.  Returns a BGR numpy image.

    Uses matplotlib's non-interactive Agg backend so it never blocks.
    All collected points (floor + any prior phase) are used for the solvePnP estimate.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 80
    fig = Figure(figsize=(out_size / dpi, out_size / dpi), dpi=dpi,
                 facecolor="#111111")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    W = COURT_WIDTH_M
    L = COURT_LENGTH_M
    kw = dict(color="#555555", lw=0.8)

    # ── court wireframe ────────────────────────────────────────────────────────
    # Floor outline
    ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], [0]*5, **kw)
    # Short line + half-court
    ax.plot([0, W], [_SHORT_Y, _SHORT_Y], [0, 0], **kw)
    ax.plot([HALF_COURT_M]*2, [_SHORT_Y, L], [0, 0], **kw)
    # Service box lines
    ax.plot([SERVICE_BOX_M]*2, [_SHORT_Y, _BOX_Y], [0, 0], color="#444444", lw=0.5)
    ax.plot([_RIGHT_X]*2,      [_SHORT_Y, _BOX_Y], [0, 0], color="#444444", lw=0.5)
    ax.plot([0, W], [_BOX_Y, _BOX_Y], [0, 0], color="#444444", lw=0.5)
    # Front wall face
    ax.plot([0, W], [0, 0], [0, 0], color="#777777", lw=1.0)
    ax.plot([0, W], [0, 0], [OUT_LINE_FRONT_M]*2, color="#888888", lw=1.2)
    ax.plot([0, 0], [0, 0], [0, OUT_LINE_FRONT_M], color="#666666", lw=0.8)
    ax.plot([W, W], [0, 0], [0, OUT_LINE_FRONT_M], color="#666666", lw=0.8)
    ax.plot([0, W], [0, 0], [TIN_HEIGHT_M]*2,          color="#555555", lw=0.6, ls="--")
    ax.plot([0, W], [0, 0], [SERVICE_LINE_HEIGHT_M]*2,  color="#555555", lw=0.6, ls="--")
    # Side wall out-lines (slope from front to back)
    for x in (0, W):
        ax.plot([x, x], [0, L],
                [OUT_LINE_FRONT_M, OUT_LINE_BACK_M],
                color="#666666", lw=0.8)

    # ── build combined world / pixel point sets for solve ─────────────────────
    all_world = np.array([[cx, cy, cz] for _, (cx, cy, cz) in point_list],
                          dtype=np.float64)
    n_current  = len(pixel_pts)
    all_pixels = list(pixel_pts)

    if prior_pixel_pts is not None and len(prior_pixel_pts) > 0:
        prior_world = np.array([[cx, cy, cz] for _, (cx, cy, cz) in prior_point_list],
                                dtype=np.float64)
        combined_world  = np.vstack([prior_world,  all_world[:n_current]])
        combined_pixels = np.vstack([prior_pixel_pts,
                                     np.array(pixel_pts, dtype=np.float64)
                                     ]) if n_current > 0 else prior_pixel_pts
        n_combined = len(combined_pixels)
    else:
        combined_world  = all_world[:n_current]
        combined_pixels = np.array(pixel_pts, dtype=np.float64) if n_current > 0 else None
        n_combined = n_current

    # ── draw clicked points ────────────────────────────────────────────────────
    # Prior-phase points (floor when we're in wall phase)
    if prior_pixel_pts is not None and prior_point_list is not None:
        for _, (cx, cy, cz) in prior_point_list:
            ax.scatter([cx], [cy], [cz], c="#00cc55", s=18, zorder=4, alpha=0.7)

    # Current-phase points
    for i, (_, (cx, cy, cz)) in enumerate(point_list[:n_current]):
        col = "#ffee00" if i == n_current - 1 else "#00aaff"
        ax.scatter([cx], [cy], [cz], c=col, s=30, zorder=5)

    # ── attempt live solvePnP ──────────────────────────────────────────────────
    rmse_str  = f"{n_combined} pts  |  need 6 to solve"
    cam_label = ""
    if n_combined >= 6 and combined_pixels is not None:
        try:
            K_est = _estimate_camera_matrix(frame_w, frame_h)
            ok, rvec, tvec = cv2.solvePnP(
                combined_world,
                combined_pixels.reshape(-1, 1, 2).astype(np.float64),
                K_est, np.zeros(5),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                R_mat, _ = cv2.Rodrigues(rvec)
                C_pos    = (-R_mat.T @ tvec).flatten()

                # Draw thin rays from camera to each world point
                for (wx, wy, wz) in combined_world:
                    ax.plot([C_pos[0], wx], [C_pos[1], wy], [C_pos[2], wz],
                            color="#ff4444", lw=0.25, alpha=0.25)

                # Camera marker
                ax.scatter([C_pos[0]], [C_pos[1]], [C_pos[2]],
                           c="#ff2222", s=120, marker="^", zorder=10)

                # Reprojection error
                proj, _ = cv2.projectPoints(
                    combined_world, rvec, tvec, K_est, np.zeros(5))
                err  = np.linalg.norm(
                    proj.reshape(-1, 2) - combined_pixels.reshape(-1, 2), axis=1)
                rmse = float(np.sqrt(np.mean(err ** 2)))
                rmse_str  = (f"{n_combined} pts  |  RMSE {rmse:.1f} px  |  "
                             f"cam ({C_pos[0]:.1f}, {C_pos[1]:.1f}, {C_pos[2]:.1f}) m")
        except Exception:
            rmse_str = f"{n_combined} pts  |  solve failed"

    # ── axes & title ───────────────────────────────────────────────────────────
    ax.set_xlim3d(-0.5, W + 0.5)
    ax.set_ylim3d(-0.5, L + 0.5)
    ax.set_zlim3d(0,    6)
    # equal aspect so court proportions look correct
    limits  = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centres = limits.mean(axis=1)
    radius  = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim3d(centres[0] - radius, centres[0] + radius)
    ax.set_ylim3d(centres[1] - radius, centres[1] + radius)
    ax.set_zlim3d(centres[2] - radius, centres[2] + radius)
    ax.set_xlabel("X (m)", fontsize=5, color="#aaaaaa", labelpad=1)
    ax.set_ylabel("Y (m)", fontsize=5, color="#aaaaaa", labelpad=1)
    ax.set_zlabel("Z (m)", fontsize=5, color="#aaaaaa", labelpad=1)
    ax.tick_params(labelsize=4, colors="#888888")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333333")
    ax.grid(True, color="#333333", lw=0.3)
    ax.set_title(rmse_str, fontsize=6, color="#cccccc", pad=2)
    ax.view_init(elev=22, azim=-115)

    fig.tight_layout(pad=0.3)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())   # (h, w, 4) RGBA uint8
    return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)


# ── Interactive point collection ───────────────────────────────────────────────

def _collect_points_phase(frame: np.ndarray,
                           point_list: list,
                           hints: list,
                           title: str,
                           prior_pixel_pts: Optional[np.ndarray] = None,
                           prior_point_list: Optional[list] = None) -> np.ndarray:
    """Collect one click per entry in point_list. Returns (N,2) float32.

    Three windows:
      LEFT   (WIN_VID)   — the video frame; click here to place points
      CENTRE (WIN_GUIDE) — court diagram + current point info; read-only
      RIGHT  (WIN_3D)    — live 3-D court + camera preview; updates on each click

    Controls:
      Left-click in video window  — record point
      U (in either window)        — undo last click
      SPACE or ENTER              — finish (only when all points clicked)
    """
    n_total   = len(point_list)
    pixel_pts: List[Tuple[int, int]] = []
    mouse_pos = [0, 0]
    done      = [False]

    WIN_VID   = title
    WIN_GUIDE = title + " — Guide"
    WIN_3D    = title + " — 3D Preview"

    _last_3d_n = [-1]   # track when pixel_pts length changes

    C_CYAN  = (255, 220, 0)
    C_GREEN = (0, 220, 80)
    C_WHITE = (230, 230, 230)
    C_GREY  = (150, 150, 150)
    C_RED   = (60, 60, 220)

    fh, fw = frame.shape[:2]
    GUIDE_W = max(320, fw // 3)
    GUIDE_H = fh

    # ── guide window renderer ──────────────────────────────────────────────────
    def _render_guide():
        idx    = len(pixel_pts)
        guide  = np.zeros((GUIDE_H, GUIDE_W, 3), dtype=np.uint8)
        guide[:] = (20, 20, 20)

        # Header
        cv2.rectangle(guide, (0, 0), (GUIDE_W, 55), (35, 35, 35), -1)
        cv2.putText(guide, f"Point {idx + 1} / {n_total}" if idx < n_total else "Done!",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GREY, 1)
        if idx < n_total:
            label = point_list[idx][0]
            cv2.putText(guide, label, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_CYAN, 2)
        else:
            cv2.putText(guide, "Press SPACE or ENTER to finish",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GREEN, 1)

        # Hint text (word-wrap into ~45 chars)
        if idx < n_total:
            hint = hints[idx] if idx < len(hints) else ""
            words = hint.split()
            lines, cur = [], ""
            for w in words:
                if len(cur) + len(w) + 1 > 42:
                    lines.append(cur.strip())
                    cur = w + " "
                else:
                    cur += w + " "
            if cur.strip():
                lines.append(cur.strip())
            y = 72
            for ln in lines[:3]:
                cv2.putText(guide, ln, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)
                y += 18

        # Court diagram
        diag_top  = 130
        diag_h    = GUIDE_H - diag_top - 60
        diag_w    = GUIDE_W - 20
        diagram   = _draw_court_diagram(point_list, idx, diag_w, diag_h)
        guide[diag_top:diag_top + diag_h, 10:10 + diag_w] = diagram

        # Footer controls
        fy = GUIDE_H - 40
        cv2.rectangle(guide, (0, fy - 10), (GUIDE_W, GUIDE_H), (30, 30, 30), -1)
        cv2.putText(guide, "Click video to place  |  U = undo",
                    (10, fy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)
        cv2.putText(guide, "SPACE/ENTER = finish when all done",
                    (10, fy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)

        cv2.imshow(WIN_GUIDE, guide)

    # ── video window renderer ──────────────────────────────────────────────────
    def _render_video():
        idx  = len(pixel_pts)
        disp = frame.copy()

        # Previously clicked dots
        for i, (px, py) in enumerate(pixel_pts):
            col = C_GREEN if i < idx - 1 else (0, 200, 255)
            cv2.circle(disp, (px, py), 7, col, -1)
            cv2.circle(disp, (px, py), 7, (0, 0, 0), 1)
            cv2.putText(disp, str(i + 1), (px + 9, py - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

        # Crosshair
        mx, my = mouse_pos
        if idx < n_total:
            L = 18
            cv2.line(disp, (mx - L, my), (mx + L, my), C_CYAN, 1)
            cv2.line(disp, (mx, my - L), (mx, my + L), C_CYAN, 1)
            cv2.circle(disp, (mx, my), 5, C_CYAN, 1)

        # Minimal status strip at bottom (non-intrusive)
        if idx < n_total:
            label = point_list[idx][0]
            msg   = f"  {idx + 1}/{n_total}: {label}"
        else:
            msg   = "  All points done — press SPACE or ENTER"
        bh = 28
        bar = np.zeros((bh, fw, 3), dtype=np.uint8)
        bar[:] = (25, 25, 25)
        col_msg = C_GREEN if idx == n_total else C_CYAN
        cv2.putText(bar, msg, (4, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col_msg, 1)
        cv2.putText(bar, "U=undo", (fw - 70, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)

        combined = np.vstack([disp, bar])
        cv2.imshow(WIN_VID, combined)

    _3d_size = min(fh, 520)

    def _update_3d():
        """Re-render the 3-D preview only when the point count changes."""
        n = len(pixel_pts)
        if n == _last_3d_n[0]:
            return
        _last_3d_n[0] = n
        img3d = _render_3d_preview(
            pixel_pts, point_list, fw, fh,
            prior_pixel_pts=prior_pixel_pts,
            prior_point_list=prior_point_list,
            out_size=_3d_size,
        )
        cv2.imshow(WIN_3D, img3d)

    def _render():
        _render_video()
        _render_guide()
        _update_3d()

    def on_mouse_vid(event, x, y, flags, param):
        mouse_pos[0] = x
        mouse_pos[1] = min(y, fh - 1)    # clamp to video area (exclude status bar)
        if event == cv2.EVENT_LBUTTONDOWN and len(pixel_pts) < n_total:
            pixel_pts.append((x, mouse_pos[1]))
        _render()

    # ── open windows ──────────────────────────────────────────────────────────
    cv2.namedWindow(WIN_VID,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_GUIDE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_3D,    cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_GUIDE, GUIDE_W, GUIDE_H)
    cv2.resizeWindow(WIN_3D,    _3d_size, _3d_size)

    # Layout: video (left) | guide (right-top) | 3D (right-bottom)
    # This keeps everything on one screen regardless of width.
    cv2.moveWindow(WIN_VID,   40,              50)
    cv2.moveWindow(WIN_GUIDE, fw + 60,         50)
    cv2.moveWindow(WIN_3D,    fw + 60,         GUIDE_H + 80)

    cv2.setMouseCallback(WIN_VID, on_mouse_vid)
    _render()

    print(f"\n{title}")
    for i, (label, _) in enumerate(point_list, 1):
        h_str = hints[i - 1] if (i - 1) < len(hints) else ""
        print(f"  {i:2d}. {label:40s}  — {h_str}")
    print("\nControls: left-click video to place  |  U = undo  |  SPACE/ENTER = finish\n")

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('u'), ord('U')):
            if pixel_pts:
                pixel_pts.pop()
                _render()
        elif key in (13, 32) and len(pixel_pts) == n_total:   # ENTER or SPACE
            break

    cv2.destroyWindow(WIN_VID)
    cv2.destroyWindow(WIN_GUIDE)
    cv2.destroyWindow(WIN_3D)
    return np.float32(pixel_pts)


def _collect_all_points(frame: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Run Phase-1 (floor) and optionally Phase-2 (walls).

    Returns (combined_pixel_pts, walls_included).
    Phase-2 prompt appears inside an OpenCV window so no terminal input is needed.
    """
    # ── Phase 1: floor ─────────────────────────────────────────────────────────
    print("\n=== Phase 1 of 2: Floor points (24 points, z = 0) ===")
    floor_pts = _collect_points_phase(
        frame, FLOOR_POINTS, _FLOOR_HINTS,
        "Phase 1 — Floor points  (left-click | U=undo | any key when done)",
    )

    # ── Phase-2 prompt inside an OpenCV window ─────────────────────────────────
    h, w = frame.shape[:2]
    prompt_img = frame.copy()
    lines = [
        ("Phase 2: Wall points  (optional but recommended)", (0, 220, 255), 0.75),
        ("Adds 8 height-referenced points: tin, service line, side-wall out-lines.", (200, 200, 200), 0.45),
        ("These break the floor-plane degeneracy and improve 3-D accuracy.", (200, 200, 200), 0.45),
        ("", None, 0),
        ("Press  W  to continue with wall calibration", (0, 220, 80), 0.60),
        ("Press  S  to skip  (floor-only is fine too)", (160, 160, 160), 0.60),
    ]
    y = 60
    for text, color, scale in lines:
        if text and color:
            cv2.putText(prompt_img, text, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        y += 40 if scale > 0.5 else 28

    WIN = "Phase 2 — Wall points"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.imshow(WIN, prompt_img)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('s') or key == ord('S'):
            cv2.destroyWindow(WIN)
            print("  Skipping wall points — floor-only calibration.")
            return floor_pts, False
        if key == ord('w') or key == ord('W'):
            cv2.destroyWindow(WIN)
            break

    # ── Phase 2: wall ──────────────────────────────────────────────────────────
    print("\n=== Phase 2 of 2: Wall points (15 points, z > 0) ===")
    wall_pts = _collect_points_phase(
        frame, WALL_POINTS, _WALL_HINTS,
        "Phase 2 — Wall points  (left-click | U=undo | any key when done)",
        prior_pixel_pts=floor_pts,
        prior_point_list=FLOOR_POINTS,
    )

    combined = np.vstack([floor_pts, wall_pts])
    return combined, True


# ── Camera intrinsic estimation ────────────────────────────────────────────────

def _estimate_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    """Estimate K assuming ~70° diagonal FOV (typical squash back-wall camera)."""
    diag_px = np.sqrt(frame_width ** 2 + frame_height ** 2)
    focal   = diag_px / (2.0 * np.tan(np.radians(70.0) / 2.0))
    return np.array([
        [focal, 0.0,   frame_width  / 2.0],
        [0.0,   focal, frame_height / 2.0],
        [0.0,   0.0,   1.0               ],
    ], dtype=np.float64)


# ── Reprojection error helper ──────────────────────────────────────────────────

def _reprojection_rmse(calib: CalibData,
                        world_pts: np.ndarray,
                        pixel_pts: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(
        world_pts, calib.rvec, calib.tvec, calib.K, calib.dist
    )
    projected = projected.reshape(-1, 2)
    err = np.linalg.norm(projected - pixel_pts.astype(np.float64), axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


# ── 3-D solve ─────────────────────────────────────────────────────────────────

def _solve_3d(pixel_pts: np.ndarray,
              world_pts: np.ndarray,
              frame_width: int,
              frame_height: int) -> CalibData:
    """Run solvePnP + LM refinement and return CalibData.

    Tries both estimated K and self-calibrating calibrateCamera; keeps the
    lower reprojection RMSE result.
    """
    image_pts = pixel_pts.astype(np.float64).reshape(-1, 1, 2)
    K_est     = _estimate_camera_matrix(frame_width, frame_height)
    dist_z    = np.zeros(5, dtype=np.float64)

    # ── Attempt 1: solvePnP with estimated K ───────────────────────────────────
    ok, rvec, tvec = cv2.solvePnP(
        world_pts, image_pts, K_est, dist_z,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed — check that calibration points are correct.")

    rvec, tvec = cv2.solvePnPRefineLM(world_pts, image_pts, K_est, dist_z, rvec, tvec)
    calib_est  = CalibData(K=K_est, dist=dist_z.copy(), rvec=rvec, tvec=tvec)
    rmse_est   = _reprojection_rmse(calib_est, world_pts, pixel_pts)
    print(f"  solvePnP (estimated K)   reprojection RMSE: {rmse_est:.2f} px")

    best = calib_est

    # ── Attempt 2: solvePnP with distortion refinement ────────────────────────
    # Refine the estimated-K result but also let LM fit radial distortion (k1, k2).
    # This handles barrel distortion from wide-angle squash cameras without the
    # instability of fully free calibrateCamera on a single view.
    try:
        dist_rd  = np.zeros(5, dtype=np.float64)
        rvec2, tvec2 = rvec.copy(), tvec.copy()
        K_rd = K_est.copy()
        # Iterative refinement: alternate solvePnP (fixes K, solves pose) with
        # solvePnPRefineLM (holds K, refines pose + distortion).
        for _ in range(3):
            ok2, rvec2, tvec2 = cv2.solvePnP(
                world_pts, image_pts, K_rd, dist_rd,
                rvec2, tvec2, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            rvec2, tvec2 = cv2.solvePnPRefineLM(
                world_pts, image_pts, K_rd, dist_rd, rvec2, tvec2)
        calib_rd  = CalibData(K=K_rd, dist=dist_rd.copy(), rvec=rvec2, tvec=tvec2)
        rmse_rd   = _reprojection_rmse(calib_rd, world_pts, pixel_pts)
        print(f"  solvePnP + distortion     reprojection RMSE: {rmse_rd:.2f} px")
        if rmse_rd < rmse_est * 0.98:   # only switch if meaningfully better
            best = calib_rd
            print("  Using distortion-refined result.")
    except cv2.error as exc:
        print(f"  Distortion refinement failed ({exc}), keeping solvePnP result.")

    # ── Attempt 3: self-calibrating via calibrateCamera ────────────────────────
    try:
        ret, K_cal, dist_cal, rvecs, tvecs = cv2.calibrateCamera(
            [world_pts.astype(np.float32)],
            [pixel_pts.reshape(-1, 1, 2)],
            (frame_width, frame_height),
            None, None,
        )
        calib_cal = CalibData(
            K=K_cal, dist=dist_cal.flatten(),
            rvec=rvecs[0], tvec=tvecs[0],
        )
        rmse_cal = _reprojection_rmse(calib_cal, world_pts, pixel_pts)
        print(f"  calibrateCamera (free K)  reprojection RMSE: {rmse_cal:.2f} px")
        best_rmse = _reprojection_rmse(best, world_pts, pixel_pts)
        if rmse_cal < best_rmse * 0.98:
            best = calib_cal
            print("  Using calibrateCamera result (lower RMSE).")
        else:
            print("  Keeping previous best result.")
    except cv2.error as exc:
        print(f"  calibrateCamera failed ({exc}), keeping best result so far.")

    return best


# ── Public: 3-D calibration I/O ───────────────────────────────────────────────

def get_calibration(frame: np.ndarray, force_recalibrate: bool = False) -> CalibData:
    """Return 3-D CalibData, loading from disk when possible.

    Saves to CALIBRATION_3D_PATH and simultaneously writes H to
    CALIBRATION_PATH so legacy callers keep working.
    """
    if not force_recalibrate:
        try:
            data  = np.load(CALIBRATION_3D_PATH)
            calib = CalibData(K=data["K"], dist=data["dist"],
                              rvec=data["rvec"], tvec=data["tvec"])
            print("Loaded saved 3-D calibration.")
            return calib
        except (FileNotFoundError, KeyError):
            pass

    print("Running 3-D calibration…")
    h, w = frame.shape[:2]
    pixel_pts, walls_included = _collect_all_points(frame)
    world_pts = _make_world_pts(include_walls=walls_included)

    calib = _solve_3d(pixel_pts, world_pts, w, h)

    # Save 3-D calibration (include a flag so we know which points were used)
    np.savez(CALIBRATION_3D_PATH,
             K=calib.K, dist=calib.dist, rvec=calib.rvec, tvec=calib.tvec,
             walls_used=np.array([walls_included]))
    print("3-D calibration saved.")

    # Also regenerate the legacy 2-D homography from the floor points only
    floor_px  = pixel_pts[:len(FLOOR_POINTS)]
    real_pts  = np.float32([[x, y] for _, (x, y, _) in FLOOR_POINTS])
    H, _      = cv2.findHomography(floor_px, real_pts)
    np.save(CALIBRATION_PATH, H)
    print("Legacy 2-D homography also saved.")

    return calib


# ── Public: 2-D homography (legacy) ───────────────────────────────────────────

def get_homography(frame: np.ndarray, force_recalibrate: bool = False) -> np.ndarray:
    """Return 2-D homography matrix mapping pixel → court metres.

    Prefer get_calibration() for new code.
    """
    if not force_recalibrate:
        try:
            H = np.load(CALIBRATION_PATH)
            print("Loaded saved 2-D homography.")
            return H
        except (FileNotFoundError, OSError):
            pass

    # Collect floor points only
    pixel_pts = _collect_points_phase(
        frame, FLOOR_POINTS, "Calibration — click each floor point in order"
    )
    real_pts  = np.float32([[x, y] for _, (x, y, _) in FLOOR_POINTS])
    H, _      = cv2.findHomography(pixel_pts, real_pts)
    np.save(CALIBRATION_PATH, H)
    print("2-D homography saved.")
    return H


# ── Public: convenience loader (tries 3-D first, falls back to 2-D) ──────────

def load_best_calibration(frame: Optional[np.ndarray] = None,
                           force_recalibrate: bool = False,
                           ) -> Tuple[Optional[CalibData], Optional[np.ndarray]]:
    """Return (CalibData, H) where the preferred mode is non-None.

    If a saved 3-D calibration exists → (calib, None).
    Else falls back to 2-D homography → (None, H).
    Interactive calibration runs if neither file exists and frame is provided.
    """
    if not force_recalibrate:
        try:
            data  = np.load(CALIBRATION_3D_PATH)
            calib = CalibData(K=data["K"], dist=data["dist"],
                              rvec=data["rvec"], tvec=data["tvec"])
            print("Loaded saved 3-D calibration.")
            return calib, None
        except (FileNotFoundError, KeyError):
            pass

    if force_recalibrate and frame is not None:
        calib = get_calibration(frame, force_recalibrate=True)
        return calib, None

    try:
        H = np.load(CALIBRATION_PATH)
        print("Loaded saved 2-D homography (no 3-D calibration found).")
        return None, H
    except (FileNotFoundError, OSError):
        pass

    if frame is not None:
        print("No saved calibration found — running interactive 3-D calibration.")
        calib = get_calibration(frame)
        return calib, None

    raise FileNotFoundError(
        "No calibration files found and no frame provided for interactive calibration."
    )


# ── Projection functions ───────────────────────────────────────────────────────

def project_to_court(xs: list, ys: list, calib: CalibData,
                     zs: Optional[List[float]] = None,
                     ) -> Tuple[List[float], List[float]]:
    """Map pixel coordinates to court floor via ray-plane intersection.

    For each pixel (u, v) with expected world height z_target:
      1. Undistort to normalised image coords using K and dist.
      2. Express the ray direction in world space: d = R^T · [u', v', 1].
      3. Intersect the ray with the horizontal plane z = z_target:
           t = (z_target - C_z) / d_z,  P = C + t·d
         The returned (x, y) is the floor position of the person —
         which equals the hip/knee (x, y) because a person stands vertically.

    zs : per-point expected world height of the detected keypoint (metres).
         Pass 0.0 for ankles (the floor), ~0.5 for knees, ~0.9 for hips.
         Defaults to 0.0 for all points if omitted.

    Why this matters: intersecting a hip-pixel ray with z=0 gives the WRONG
    floor position because the ray continues past the hip toward the ground,
    landing up to ~2 m away from the player's actual floor contact point.
    By intersecting at z=z_hip ≈ 0.9 m instead, we recover the correct (x,y).
    """
    if not xs:
        return [], []

    if zs is None:
        zs = [0.0] * len(xs)

    pts  = np.array([[x, y] for x, y in zip(xs, ys)], dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, calib.K, calib.dist).reshape(-1, 2)  # (N, 2)

    rays_cam   = np.column_stack([norm, np.ones(len(norm))])   # (N, 3)
    rays_world = (calib.R.T @ rays_cam.T).T                    # (N, 3)

    C = calib.C  # (3,)
    result_x, result_y = [], []
    for ray, z_target in zip(rays_world, zs):
        if abs(ray[2]) < 1e-8:
            result_x.append(float("nan"))
            result_y.append(float("nan"))
            continue
        t     = (z_target - C[2]) / ray[2]   # intersect z = z_target plane
        world = C + t * ray
        result_x.append(float(world[0]))
        result_y.append(float(world[1]))

    return result_x, result_y


def apply_homography(xs: list, ys: list, H: np.ndarray
                     ) -> Tuple[List[float], List[float]]:
    """Transform pixel (x, y) lists to court (x, y) via 2-D homography H.

    Kept for backward compatibility.  Prefer project_to_court() with CalibData.
    """
    if not xs:
        return [], []
    pts         = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed[:, 0, 0].tolist(), transformed[:, 0, 1].tolist()
