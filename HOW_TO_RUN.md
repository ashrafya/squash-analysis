# How to Run — Squash Analysis Pipeline

Everything runs from the project root (`squash-analysis/`).
Scripts must be run from inside `src/` so relative imports work correctly.

---

## 1. Setup (one-time)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate        # Windows (Git Bash / MSYS)
# source venv/bin/activate           # macOS / Linux

# Install dependencies
pip install opencv-python mediapipe ultralytics scipy numpy matplotlib tqdm
```

**Verified working versions:**
| Package | Version |
|---|---|
| opencv-python | 4.13.0 |
| mediapipe | 0.10.13 |
| ultralytics | 8.4.19 |
| scipy | 1.17.1 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.3 |

---

## 2. Add your video

Place your `.mp4` file at:
```
assets/video/your_match.mp4
```

Then update `src/config.py`:
```python
VIDEO_PATH = "../assets/video/your_match.mp4"
VIDEO_FPS  = 25   # match your video's actual frame rate
```

> **Camera setup:** Fixed camera behind the back wall, capturing the full court.
> The pipeline is tested on 360p and 720p footage.

---

## 3. Pipeline steps

Run each step in order. Every step saves its output to `output/` so you can re-run any step in isolation with `--reuse`.

---

### Step 1 — Court Calibration (one-time per court)

```bash
cd src
python main.py --calibrate
```

A window opens showing the first video frame. **Click 14 court points** in the order shown on screen. The homography matrix is saved to `assets/homography.npy` and reused on all future runs.

> Skip `--calibrate` on subsequent runs — the saved calibration is loaded automatically.

---

### Step 2 — Player Tracking

```bash
cd src
python main.py
```

Tracks both players using YOLOv8-pose (full-frame detection + Hungarian matching).
Produces heatmaps, zone breakdown, movement stats, and court position plots.

**Key flags:**
| Flag | Effect |
|---|---|
| `--debug` | Show live tracking animation while processing |
| `--calibrate` | Force recalibration of court homography |
| `--reuse` | Skip tracking and reload saved positions from last run |
| `--tracker mediapipe` | Use MediaPipe crop-based tracker instead of YOLO |

**Outputs:**
```
output/last_positions_yolo.npz   # raw pixel-space positions (both players)
output/pixel_positions.png       # player paths overlaid on video frame
output/court_positions.png       # player paths on top-down court diagram
output/heatmap_player1.png       # Gaussian heatmap — Player 1
output/heatmap_player2.png       # Gaussian heatmap — Player 2
output/zone_breakdown.png        # 9-zone court breakdown chart
output/histograms.png            # X/Y position distributions
output/timeseries.png            # speed over time
output/run_history.json          # cumulative run stats log
```

**Console output:** stats table with distance (m), avg speed, peak speed, T-time %, zone %s.

---

### Step 3 — Ball Detection

```bash
cd src
python detect_ball.py
```

Detects the squash ball each frame using YOLOv8n (COCO sports-ball) + MOG2 motion fallback.
Processes every frame (`BALL_FRAME_SKIP = 1`) for maximum recall.

**Key flags:**
| Flag | Effect |
|---|---|
| `--debug` | Show live detection overlay (YOLO=yellow, motion=cyan) |
| `--calibrate` | Force recalibration |
| `--reuse` | Skip detection and reload saved ball positions |

**Output:**
```
output/ball_positions.npz        # frame_idx, xs, ys, xs_m, ys_m, confidence, method
output/ball_trajectory.png       # raw ball detections on court diagram
```

---

### Step 4 — Ball Tracking (Kalman smoothing)

```bash
cd src
python track_ball.py
```

Applies a Kalman filter to the raw ball detections, fills short gaps (≤ 5 frames), flags "ball lost" segments, and generates a smoothed trajectory.

**Key flags:**
| Flag | Effect |
|---|---|
| `--calibrate` | Force recalibration |
| `--no-anim` | Skip GIF generation (faster) |

**Outputs:**
```
output/ball_positions_smooth.npz     # smoothed positions + filled flag
output/ball_trajectory_smooth.png    # smoothed trajectory on court
output/ball_trajectory_anim.gif      # animated ball path
output/ball_lost_segments.txt        # log of frames where ball was not found
```

---

### Step 5 — Rally Segmentation & Combined Analysis

```bash
cd src
python segment_rallies.py
```

Uses gaps in ball detections to segment the clip into individual rallies, computes per-rally stats, and generates a combined court diagram.

**Key flags:**
| Flag | Effect |
|---|---|
| `--min-gap N` | Minimum ball-lost gap (frames) to mark a rally boundary. Default: 20 (= 0.8 s at 25fps). Increase if rallies are being split too aggressively. |
| `--calibrate` | Force recalibration |

**Console output:** table of rally #, start/end time, duration, approx shot count, P1/P2 distance.

**Outputs:**
```
output/rally_boundaries.csv      # rally_id, start_frame, end_frame, duration_s
output/rally_stats.csv           # per-rally: duration, shots, dist_p1_m, dist_p2_m
output/combined_court.png        # player positions + ball trajectory coloured by rally
output/rally_timeline.png        # horizontal bar chart of rally durations
```

---

## 4. Full pipeline — quick reference

```bash
cd src

# First run (with calibration):
python main.py --calibrate         # Step 2: player tracking + calibrate
python detect_ball.py              # Step 3: ball detection
python track_ball.py               # Step 4: ball smoothing
python segment_rallies.py          # Step 5: rally segmentation

# Subsequent runs (reuse calibration, re-run tracking):
python main.py                     # Step 2
python detect_ball.py              # Step 3
python track_ball.py               # Step 4
python segment_rallies.py          # Step 5

# Re-run analysis only (skip heavy tracking steps):
python main.py --reuse             # reloads last_positions_yolo.npz
python detect_ball.py --reuse      # reloads ball_positions.npz
python track_ball.py               # always fast (reads from npz)
python segment_rallies.py          # always fast (reads from npz)
```

---

## 5. Tuning

All tunables are in [`src/config.py`](src/config.py). Common ones to adjust:

| Constant | Default | Effect |
|---|---|---|
| `FRAME_CAP` | `5000` | Max frames to process. Set to `None` for full video. |
| `FRAME_SKIP` | `5` | Process every Nth frame for player tracking. Higher = faster but less accurate. |
| `BALL_FRAME_SKIP` | `1` | Process every Nth frame for ball detection. Keep at 1. |
| `ANGLE_MATCH_THRESHOLD` | `0.7` | Camera-cut filter sensitivity. Lower = accept more frames. |
| `MAX_JUMP_PX` | `150` | Discard player detection if it jumps more than this many pixels. |
| `SMOOTH_WINDOW` | `9` | Rolling median window for player position smoothing. |
| `RALLY_END_MIN_FRAMES` | `20` | Ball-lost gap length to mark a rally boundary. |
| `KALMAN_GAP_FILL` | `5` | Max frames Kalman filter will bridge in ball trajectory. |

---

## 6. Output directory structure

```
output/
├── last_positions_yolo.npz       # player tracking (YOLO)
├── pixel_positions.png
├── court_positions.png
├── heatmap_player1.png
├── heatmap_player2.png
├── zone_breakdown.png
├── histograms.png
├── timeseries.png
├── run_history.json
│
├── ball_positions.npz            # raw ball detections
├── ball_trajectory.png
├── ball_positions_smooth.npz     # Kalman-smoothed
├── ball_trajectory_smooth.png
├── ball_trajectory_anim.gif
├── ball_lost_segments.txt
│
├── rally_boundaries.csv          # rally segmentation
├── rally_stats.csv
├── combined_court.png
└── rally_timeline.png
```
