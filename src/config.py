# =============================================================================
#  config.py — single source of truth for all tunable constants
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_PATH       = "../assets/video/men.mp4"
CALIBRATION_PATH = "../assets/homography.npy"
OUTPUT_DIR       = "../output"

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
FRAME_CAP  = 2000   # max frames to process (set to None to process full video)
FRAME_SKIP = 5      # process every Nth frame

# ── Player Tracking ───────────────────────────────────────────────────────────
SMOOTH_WINDOW         = 9     # rolling median window for noise filtering
HIP_VISIBILITY_MIN    = 0.6   # discard pose if either hip landmark is below this confidence
MAX_JUMP_PX           = 150   # discard detection if player jumps more than this many pixels
CROP_MARGIN           = 200   # pixel radius around last known position for cropped detection
ANGLE_MATCH_THRESHOLD = 0.7   # min histogram correlation to reference frame (filters camera cuts)

# ── Stats ─────────────────────────────────────────────────────────────────────
T_RADIUS_M = 1.0   # distance from T within which a position counts as "at the T"

# ── Visualisation ─────────────────────────────────────────────────────────────
HEATMAP_GRID_X  = 64    # heatmap cells across court width  (6.4 m → ~0.1 m/cell)
HEATMAP_GRID_Y  = 100   # heatmap cells across court length (9.75 m → ~0.1 m/cell)
HEATMAP_GAMMA   = 0.4   # gamma < 1 boosts mid-density areas; lower = more spread
PLAYER_COLORS   = ["red", "dodgerblue"]
PLAYER_LABELS   = ["Player 1", "Player 2"]
DEBUG_VIZ_EVERY = FRAME_SKIP   # update live debug plot every N processed frames
