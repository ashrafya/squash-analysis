import cv2
import numpy as np

from court import COURT_WIDTH_M, COURT_LENGTH_M, SHORT_LINE_M, HALF_COURT_M, SERVICE_BOX_M

CALIBRATION_PATH = "../assets/homography.npy"

# Derived court positions (y measured from front wall)
_SHORT_Y  = COURT_LENGTH_M - SHORT_LINE_M        # 5.49 m
_BOX_Y    = _SHORT_Y + SERVICE_BOX_M             # 7.09 m
_RIGHT_X  = COURT_WIDTH_M - SERVICE_BOX_M        # 4.80 m

# All known court markings with their real-world positions (x, y) in metres.
# y=0 = front wall, y=9.75 = back wall.
# Click them in this order during calibration.
CALIBRATION_POINTS = [
    ("Front-left corner",         (0,             0             )),
    ("Front-right corner",        (COURT_WIDTH_M, 0             )),
    ("Back-left corner",          (0,             COURT_LENGTH_M)),
    ("Back-right corner",         (COURT_WIDTH_M, COURT_LENGTH_M)),
    ("Short line - left wall",    (0,             _SHORT_Y      )),
    ("Short line - right wall",   (COURT_WIDTH_M, _SHORT_Y      )),
    ("T-junction",                (HALF_COURT_M,  _SHORT_Y      )),
    ("Half-court - back wall",    (HALF_COURT_M,  COURT_LENGTH_M)),
    ("Left box inner-top",        (SERVICE_BOX_M, _SHORT_Y      )),
    ("Left box back-inner",       (SERVICE_BOX_M, _BOX_Y        )),
    ("Left box back-outer",       (0,             _BOX_Y        )),
    ("Right box inner-top",       (_RIGHT_X,      _SHORT_Y      )),
    ("Right box back-inner",      (_RIGHT_X,      _BOX_Y        )),
    ("Right box back-outer",      (COURT_WIDTH_M, _BOX_Y        )),
]


def _collect_points(frame):
    """Interactively collect pixel clicks for each calibration point in order."""
    n_total = len(CALIBRATION_POINTS)
    pixel_pts = []
    clone = frame.copy()

    def _redraw():
        cv2.imshow(WIN, clone)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(pixel_pts) >= n_total:
            return
        idx = len(pixel_pts)
        pixel_pts.append((x, y))
        label, _ = CALIBRATION_POINTS[idx]
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(clone, f"{idx + 1}", (x + 7, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _update_prompt()

    def _update_prompt():
        idx = len(pixel_pts)
        overlay = clone.copy()
        if idx < n_total:
            label, _ = CALIBRATION_POINTS[idx]
            text = f"Click {idx + 1}/{n_total}: {label}"
        else:
            text = "All points clicked — press any key to finish."
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, clone, 0.45, 0, clone)
        cv2.putText(clone, text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(WIN, clone)

    WIN = "Calibration — click each marked point in order"
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)
    _update_prompt()

    print("\nCalibration: click each court marking in order (label shown on screen).")
    print("Points to click:")
    for i, (label, real) in enumerate(CALIBRATION_POINTS, 1):
        print(f"  {i:2d}. {label:30s}  real=({real[0]:.2f}, {real[1]:.2f}) m")
    print("Press any key after all points are clicked.\n")

    while True:
        key = cv2.waitKey(50)
        if key != -1 and len(pixel_pts) == n_total:
            break

    cv2.destroyAllWindows()
    return np.float32(pixel_pts)


def get_homography(frame, force_recalibrate=False):
    """Return homography matrix mapping pixel coords -> court coords (metres).

    Loads from disk if a saved calibration exists, unless force_recalibrate=True.
    """
    if not force_recalibrate:
        try:
            H = np.load(CALIBRATION_PATH)
            print("Loaded saved calibration.")
            return H
        except (FileNotFoundError, OSError):
            pass

    pixel_pts = _collect_points(frame)
    real_pts = np.float32([pt for _, pt in CALIBRATION_POINTS])
    H, _ = cv2.findHomography(pixel_pts, real_pts)
    np.save(CALIBRATION_PATH, H)
    print("Calibration saved.")
    return H


def apply_homography(xs, ys, H):
    """Transform pixel (x, y) lists to court (x, y) lists using homography H."""
    pts = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed[:, 0, 0].tolist(), transformed[:, 0, 1].tolist()
