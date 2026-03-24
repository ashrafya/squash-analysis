# =============================================================================
#  config.py — single source of truth for all tunable constants
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
import os as _os
_PROJECT_ROOT    = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
VIDEO_PATH           = _os.path.join(_PROJECT_ROOT, "assets", "video", "women360.mp4")
CALIBRATION_PATH     = _os.path.join(_PROJECT_ROOT, "assets", "homography.npy")
CALIBRATION_3D_PATH  = _os.path.join(_PROJECT_ROOT, "assets", "calibration_3d.npz")
OUTPUT_DIR           = _os.path.join(_PROJECT_ROOT, "output")

# ── WSF Official Court Dimensions (metres) ────────────────────────────────────
COURT_WIDTH_M  = 6.4
COURT_LENGTH_M = 9.75
SHORT_LINE_M   = 4.26   # short line distance from back wall
HALF_COURT_M   = 3.2    # half-court line x position (centre of width)
SERVICE_BOX_M  = 1.6    # service box side length

# ── Derived Court Positions (metres, y measured from front wall) ──────────────
T_X = HALF_COURT_M                   # 3.20 m — T junction x position
T_Y = COURT_LENGTH_M - SHORT_LINE_M  # 5.49 m — T junction y position

# ── Video Processing ──────────────────────────────────────────────────────────
VIDEO_FPS        = 25   # known frame rate of the source videos — used for all timing math
FRAME_CAP        = 2000   # max frames to process (set to None to process full video)
FRAME_SKIP       = 20   # process every Nth frame; time between analysed frames = FRAME_SKIP / VIDEO_FPS
MODEL_COMPLEXITY = 0    # MediaPipe model complexity: 0=fastest, 1=balanced, 2=most accurate

# ── Player Tracking ───────────────────────────────────────────────────────────
SMOOTH_WINDOW         = 9     # rolling median window for noise filtering
FOOT_VISIBILITY_MIN   = 0.6   # min confidence for heel landmarks (primary ground-position source)
MAX_JUMP_PX           = 150   # discard detection if player jumps more than this many pixels
CROP_MARGIN           = 200   # pixel radius around last known position for cropped detection
ANGLE_MATCH_THRESHOLD = 0.7   # min histogram correlation to reference frame (filters camera cuts)

# ── Option A: Tracking Decoupling ─────────────────────────────────────────────
MAX_SPEED_MS              = 10.0   # (reserved) court-space speed cap in m/s
MIN_SEPARATION_PX         = 80     # (reserved) pixel-space separation for _try_reassign sanity check
MIN_SEPARATION_M          = 0.2    # court-space distance (m) below which trackers are flagged as coupled
COUPLING_FRAMES_THRESHOLD = 25     # consecutive coupling frames before warning is printed
VERIFY_EVERY_N            = 150    # (reserved) periodic verification interval for Day 12
COURT_BOUNDS_MARGIN_M     = 1.0    # tolerance beyond court edges for detection validation (metres)

# ── Ball Detection & Tracking ─────────────────────────────────────────────────
BALL_FRAME_SKIP = 1   # process every Nth frame for ball detection (1 = every frame)
                      # keep at 1: at 25fps a hard drive (~150 km/h) travels 1.67m/frame —
                      # at FRAME_SKIP=5 it travels 8.3m, making motion detection useless
KALMAN_GAP_FILL = 5   # max consecutive missing frames Kalman filter will bridge
RALLY_END_MIN_FRAMES = 20  # ball-lost gap >= this many frames marks an inter-rally boundary
                           # at BALL_FRAME_SKIP=1 and 25fps: 20 frames = 0.8 s

# ── Stats ─────────────────────────────────────────────────────────────────────
T_RADIUS_M = 1.25   # distance from T within which a position counts as "at the T"

# ── Zone Grid (3 columns × 3 rows) ───────────────────────────────────────────
ZONE_COL_EDGES = [0.0, COURT_WIDTH_M / 3, 2 * COURT_WIDTH_M / 3, COURT_WIDTH_M]
ZONE_ROW_EDGES = [0.0, COURT_LENGTH_M / 3, 2 * COURT_LENGTH_M / 3, COURT_LENGTH_M]
ZONE_NAMES = [
    ["Front-L", "Front-C", "Front-R"],
    ["Mid-L",   "T",       "Mid-R"  ],
    ["Back-L",  "Back-C",  "Back-R" ],
]

# ── Visualisation ─────────────────────────────────────────────────────────────
HEATMAP_GRID_X  = 64    # heatmap cells across court width  (6.4 m → ~0.1 m/cell)
HEATMAP_GRID_Y  = 100   # heatmap cells across court length (9.75 m → ~0.1 m/cell)
HEATMAP_GAMMA   = 0.25   # gamma < 1 boosts mid-density areas; lower = more spread
PLAYER_COLORS   = ["red", "dodgerblue"]
PLAYER_LABELS   = ["Player 1", "Player 2"]
DEBUG_VIZ_EVERY = FRAME_SKIP   # update live debug plot every N processed frames
