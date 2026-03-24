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

# ── Phase-1: 14 floor points (z = 0) ─────────────────────────────────────────
FLOOR_POINTS = [
    ("Front-left corner",         (0,             0,             0.0)),
    ("Front-right corner",        (COURT_WIDTH_M, 0,             0.0)),
    ("Back-left corner",          (0,             COURT_LENGTH_M, 0.0)),
    ("Back-right corner",         (COURT_WIDTH_M, COURT_LENGTH_M, 0.0)),
    ("Short line - left wall",    (0,             _SHORT_Y,       0.0)),
    ("Short line - right wall",   (COURT_WIDTH_M, _SHORT_Y,       0.0)),
    ("T-junction",                (HALF_COURT_M,  _SHORT_Y,       0.0)),
    ("Half-court - back wall",    (HALF_COURT_M,  COURT_LENGTH_M, 0.0)),
    ("Left box inner-top",        (SERVICE_BOX_M, _SHORT_Y,       0.0)),
    ("Left box back-inner",       (SERVICE_BOX_M, _BOX_Y,         0.0)),
    ("Left box back-outer",       (0,             _BOX_Y,         0.0)),
    ("Right box inner-top",       (_RIGHT_X,      _SHORT_Y,       0.0)),
    ("Right box back-inner",      (_RIGHT_X,      _BOX_Y,         0.0)),
    ("Right box back-outer",      (COURT_WIDTH_M, _BOX_Y,         0.0)),
]

# Legacy (x,y) 2-tuple list for backward-compat callers that do CALIBRATION_POINTS[i][1]
CALIBRATION_POINTS = [(lbl, (x, y)) for lbl, (x, y, _) in FLOOR_POINTS]

# ── Phase-2: 8 wall points (z > 0) ────────────────────────────────────────────
WALL_POINTS = [
    # Front wall — tin corners (bottom of playable area)
    ("Front wall tin - left",          (0,             0,             TIN_HEIGHT_M         )),
    ("Front wall tin - right",         (COURT_WIDTH_M, 0,             TIN_HEIGHT_M         )),
    # Front wall — service line corners
    ("Front wall service line - left", (0,             0,             SERVICE_LINE_HEIGHT_M)),
    ("Front wall service line - right",(COURT_WIDTH_M, 0,             SERVICE_LINE_HEIGHT_M)),
    # Side wall out-line where it crosses the short line
    ("Left wall out-line at short line",  (0,             _SHORT_Y,      _side_out_at(_SHORT_Y))),
    ("Right wall out-line at short line", (COURT_WIDTH_M, _SHORT_Y,      _side_out_at(_SHORT_Y))),
    # Side wall out-line at the back wall (z = OUT_LINE_BACK_M)
    ("Left wall out-line at back wall",   (0,             COURT_LENGTH_M, OUT_LINE_BACK_M      )),
    ("Right wall out-line at back wall",  (COURT_WIDTH_M, COURT_LENGTH_M, OUT_LINE_BACK_M      )),
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


# ── Interactive point collection ───────────────────────────────────────────────

def _collect_points_phase(frame: np.ndarray,
                           point_list: list,
                           title: str) -> np.ndarray:
    """Collect one click per entry in point_list. Returns (N,2) float32."""
    n_total = len(point_list)
    pixel_pts: List[Tuple[int, int]] = []
    clone = frame.copy()

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(pixel_pts) >= n_total:
            return
        idx = len(pixel_pts)
        pixel_pts.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(clone, str(idx + 1), (x + 7, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _update_prompt()

    def _update_prompt():
        idx = len(pixel_pts)
        overlay = clone.copy()
        if idx < n_total:
            label, _ = point_list[idx]
            text = f"Click {idx + 1}/{n_total}: {label}"
        else:
            text = "All points clicked — press any key to finish."
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, clone, 0.45, 0, clone)
        cv2.putText(clone, text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(title, clone)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    _update_prompt()

    print(f"\n{title}")
    for i, (label, coords) in enumerate(point_list, 1):
        xyz = coords if len(coords) == 3 else (*coords, 0.0)
        print(f"  {i:2d}. {label:40s}  ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) m")
    print("Press any key after all points are clicked.\n")

    while True:
        key = cv2.waitKey(50)
        if key != -1 and len(pixel_pts) == n_total:
            break

    cv2.destroyWindow(title)
    return np.float32(pixel_pts)


def _collect_all_points(frame: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Run Phase-1 (floor) and optionally Phase-2 (walls).

    Returns (combined_pixel_pts, walls_included).
    """
    # Phase 1: floor
    print("\n=== Phase 1: Floor points (14 points, z = 0) ===")
    floor_pts = _collect_points_phase(
        frame, FLOOR_POINTS, "Phase 1 — Floor points (click each in order)"
    )

    # Ask whether to do Phase 2
    print("\n=== Phase 2: Wall points (8 points, z > 0) ===")
    print("  These points (tin, service line, side-wall out-lines) break the")
    print("  coplanar degeneracy and significantly improve solvePnP accuracy.")
    print("  Press ENTER to continue with wall calibration, or 'S' + ENTER to skip.")
    choice = input("  Choice [Enter / S]: ").strip().lower()

    if choice == "s":
        print("  Skipping wall points — using floor-only calibration.")
        return floor_pts, False

    print("\n  Clicking wall points.  For each point click the pixel where the")
    print("  line meets the wall (the corner or line endpoint shown on court).")
    wall_pts = _collect_points_phase(
        frame, WALL_POINTS, "Phase 2 — Wall points (click each in order)"
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

    # ── Attempt 2: self-calibrating via calibrateCamera ────────────────────────
    try:
        ret, K_cal, dist_cal, rvecs, tvecs = cv2.calibrateCamera(
            [world_pts.astype(np.float32)],
            [pixel_pts.reshape(-1, 1, 2)],
            (frame_width, frame_height),
            None, None,
            flags=cv2.CALIB_FIX_ASPECT_RATIO,
        )
        calib_cal = CalibData(
            K=K_cal, dist=dist_cal.flatten(),
            rvec=rvecs[0], tvec=tvecs[0],
        )
        rmse_cal = _reprojection_rmse(calib_cal, world_pts, pixel_pts)
        print(f"  calibrateCamera          reprojection RMSE: {rmse_cal:.2f} px")
        if rmse_cal < rmse_est:
            best = calib_cal
            print("  Using calibrateCamera result (lower RMSE).")
        else:
            print("  Keeping solvePnP result.")
    except cv2.error as exc:
        print(f"  calibrateCamera failed ({exc}), keeping solvePnP result.")

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

def project_to_court(xs: list, ys: list, calib: CalibData
                     ) -> Tuple[List[float], List[float]]:
    """Map pixel coordinates to court floor (z = 0) via ray-plane intersection.

    For each pixel (u, v):
      1. Undistort to normalised image coords using K and dist.
      2. Express the ray direction in world space: d = R^T · [u', v', 1].
      3. Intersect the ray with z = 0: t = -C_z / d_z, P = C + t·d.

    This correctly projects any body keypoint (hips, knees, ankles) to the
    floor plane regardless of its apparent image height.
    """
    if not xs:
        return [], []

    pts  = np.array([[x, y] for x, y in zip(xs, ys)], dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, calib.K, calib.dist).reshape(-1, 2)  # (N, 2)

    rays_cam   = np.column_stack([norm, np.ones(len(norm))])   # (N, 3)
    rays_world = (calib.R.T @ rays_cam.T).T                    # (N, 3)

    C = calib.C  # (3,)
    result_x, result_y = [], []
    for ray in rays_world:
        if abs(ray[2]) < 1e-8:
            result_x.append(float("nan"))
            result_y.append(float("nan"))
            continue
        t     = -C[2] / ray[2]
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
