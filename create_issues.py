"""
create_issues.py — Creates GitHub issues for Days 1–21 of the squash analysis build plan.

Usage:
    Set your GitHub personal access token as an environment variable, then run:

        GITHUB_TOKEN=ghp_your_token_here python create_issues.py

    Or on Windows (Command Prompt):
        set GITHUB_TOKEN=ghp_your_token_here
        python create_issues.py

    Or on Windows (PowerShell):
        $env:GITHUB_TOKEN="ghp_your_token_here"
        python create_issues.py

Token needs: repo scope (read + write issues).
Create one at: https://github.com/settings/tokens
"""

import os
import sys
import time
import urllib.request
import urllib.error
import json

REPO  = "ashrafya/squash-analysis"
TOKEN = os.environ.get("GITHUB_TOKEN")

if not TOKEN:
    sys.exit("ERROR: GITHUB_TOKEN environment variable not set. See instructions at the top of this file.")

BASE_URL = "https://api.github.com"
HEADERS  = {
    "Authorization": f"token {TOKEN}",
    "Accept":        "application/vnd.github+json",
    "Content-Type":  "application/json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def api(method, path, data=None):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    req  = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  HTTP {e.code} on {method} {path}: {err[:200]}")
        return None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

LABELS = [
    {"name": "week-1",   "color": "22c55e", "description": "Week 1 — Player Tracking & Heatmap MVP"},
    {"name": "week-2",   "color": "3b82f6", "description": "Week 2 — Ball Tracking, Rally Intelligence & Player Metrics"},
    {"name": "week-3",   "color": "a855f7", "description": "Week 3 — Shot Intelligence, Front Wall Heatmap & Web UI"},
    {"name": "complete", "color": "6b7280", "description": "Day completed"},
]

def ensure_labels():
    print("Creating labels...")
    existing = {l["name"] for l in (api("GET", f"/repos/{REPO}/labels?per_page=100") or [])}
    for label in LABELS:
        if label["name"] in existing:
            print(f"  label '{label['name']}' already exists — skipping")
        else:
            result = api("POST", f"/repos/{REPO}/labels", label)
            if result:
                print(f"  created label '{label['name']}'")
    print()


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------

ISSUES = [
    # ── Week 1 ────────────────────────────────────────────────────────────
    {
        "day":    1,
        "title":  "Day 1 — Video Ingestion & Pose Detection",
        "labels": ["week-1", "complete"],
        "closed": True,
        "body": """\
## Objective
Extract player body position from video frames.

## Tasks
- [ ] Load `.mp4` video using OpenCV
- [ ] Run pose detection on each frame (MediaPipe Pose)
- [ ] Compute player hip centre `(left_hip + right_hip) / 2`
- [ ] Store raw pixel positions per frame

## Deliverable
Script that plots player movement in pixel space.
""",
    },
    {
        "day":    2,
        "title":  "Day 2 — Player Selection & Tracking",
        "labels": ["week-1", "complete"],
        "closed": True,
        "body": """\
## Objective
Track a single player consistently.

## Tasks
- [ ] Detect both players in frame
- [ ] Separate players by position (top/bottom half of frame)
- [ ] Keep only the selected player
- [ ] Filter noisy detections with jump threshold and visibility check

## Deliverable
Clean, continuous player path in pixel coordinates.
""",
    },
    {
        "day":    3,
        "title":  "Day 3 — Court Calibration & Mapping",
        "labels": ["week-1", "complete"],
        "closed": True,
        "body": """\
## Objective
Map camera pixels to court coordinates.

## Tasks
- [ ] Allow user to click court markings in an OpenCV window
- [ ] Compute homography matrix from clicked pixel points to real-world metres
- [ ] Convert pixel positions to court coordinates
- [ ] Save calibration to disk so it only needs to be done once per court

## Deliverable
Player movement correctly mapped within court bounds (metres).
""",
    },
    {
        "day":    4,
        "title":  "Day 4 — Heatmap Generation",
        "labels": ["week-1", "complete"],
        "closed": True,
        "body": """\
## Objective
Convert player positions into a density map.

## Tasks
- [ ] Bin player positions into a 2D grid
- [ ] Accumulate time spent per cell
- [ ] Apply Gaussian smoothing
- [ ] Normalise values and apply gamma compression

## Deliverable
Raw heatmap array representing player presence across the court.
""",
    },
    {
        "day":    5,
        "title":  "Day 5 — Visualisation",
        "labels": ["week-1", "complete"],
        "closed": True,
        "body": """\
## Objective
Create a clear, readable heatmap output.

## Tasks
- [ ] Generate top-down squash court diagram (matplotlib)
- [ ] Overlay per-player heatmap onto court with transparent colourmap
- [ ] Add contour lines at 50% and 80% density
- [ ] Export result as PNG image

## Deliverable
Presentation-ready heatmap PNG with both players overlaid on the court.
""",
    },
    {
        "day":    6,
        "title":  "Day 6 — CLI Integration",
        "labels": ["week-1", "complete"],
        "closed": True,
        "body": """\
## Objective
Make the tool usable without code changes.

## Tasks
- [ ] Add `--debug` flag to show live tracking animation while processing
- [ ] Add `--calibrate` flag to force recalibration of the court homography
- [ ] Handle common failure cases (bad video, missing player)

## Deliverable
End-to-end usable tool via `python main.py [--debug] [--calibrate]`.
""",
    },
    {
        "day":    7,
        "title":  "Day 7 — Testing & Polish",
        "labels": ["week-1", "complete"],
        "closed": True,
        "body": """\
## Objective
MVP readiness.

## Tasks
- [ ] Test on at least two different match videos
- [ ] Handle common failure cases (bad video, missing player)
- [ ] Clean up code
- [ ] Add camera-angle filter to skip replays and cut scenes

## Deliverable
Stable MVP ready for demo to players or coaches.
""",
    },
    # ── Week 2 ────────────────────────────────────────────────────────────
    {
        "day":    8,
        "title":  "Day 8 — Distance, Speed & T-Position Stats",
        "labels": ["week-2", "complete"],
        "closed": True,
        "body": """\
## Objective
Give players the numbers coaches actually care about.

## Tasks
- [ ] Convert court coordinates to real-world metres using WSF court dimensions (9.75 m × 6.4 m)
- [ ] Compute frame-to-frame displacement → total distance covered
- [ ] Compute average speed and peak speed in m/s using frame rate
- [ ] Compute T-position time: percentage of frames within 1 m of the T
- [ ] Print a per-player stats table at end of each run

## Deliverable
Console stats table: distance (m), avg speed, peak speed, T-time %.

## Implementation
- New `src/stats.py` module: `compute_movement_stats()` + `print_stats_table()`
- T position defined as `(3.2 m, 5.49 m)` in court space
""",
    },
    {
        "day":    9,
        "title":  "Day 9 — Court Zone Breakdown",
        "labels": ["week-2"],
        "closed": False,
        "body": """\
## Objective
Show exactly which court zones a player dominates or neglects.

## Tasks
- [ ] Divide the court into 9 standard zones: Front-L, Front-C, Front-R, Mid-L, T (highlighted), Mid-R, Back-L, Back-C, Back-R
- [ ] Compute frames and percentage of time in each zone per player
- [ ] Overlay zone labels with percentages on the heatmap PNG

## Deliverable
Annotated zone heatmap — directly comparable to SmartSquash's court region output.
""",
    },
    {
        "day":    10,
        "title":  "Day 10 — Ball Detection",
        "labels": ["week-2"],
        "closed": False,
        "body": """\
## Objective
Detect the squash ball in video frames reliably.

## Tasks
- [ ] Colour-based detection: isolate yellow/white ball using HSV thresholds + contour circularity filter
- [ ] Benchmark accuracy on 200 manually labelled frames
- [ ] If <70% recall, switch to a fine-tuned YOLOv8-nano model (fast enough for real-time)
- [ ] Store raw ball pixel positions and detection confidence per frame

## Deliverable
Ball detection script with ≥70% recall on test frames.
""",
    },
    {
        "day":    11,
        "title":  "Day 11 — Ball Tracking & Court-Space Trajectory",
        "labels": ["week-2"],
        "closed": False,
        "body": """\
## Objective
Convert noisy ball detections into a smooth trajectory.

## Tasks
- [ ] Apply a Kalman filter to smooth detections and fill short gaps (<5 frames)
- [ ] Project ball positions through the homography matrix into court space
- [ ] Detect "ball lost" segments and flag them
- [ ] Visualise trajectory as an animated path on the court diagram

## Deliverable
Smooth ball trajectory overlaid on the top-down court view.
""",
    },
    {
        "day":    12,
        "title":  "Day 12 — Player Tracking: Option B & C Experiments",
        "labels": ["week-2"],
        "closed": False,
        "body": """\
## Objective
Eliminate player identity coupling at its root by replacing (or augmenting) the
MediaPipe dual-crop trackers with true multi-person detection.

Day 11 shipped **Option A** (velocity cap + coupling counter + periodic re-assignment),
which reduces but does not eliminate coupling. This day experiments with two
architecturally stronger alternatives.

---

## Option B — Full rewrite with YOLOv8-pose (recommended)

YOLOv8-pose detects all people in a single forward pass and assigns keypoints
per bounding box, making coupling architecturally impossible.

### Tasks
- [ ] `pip install ultralytics` and smoke-test `yolov8n-pose.pt` on a sample frame
- [ ] Replace `extract_pose.py` tracking loop with YOLOv8-pose inference
  - Full frame → YOLOv8-pose → `[{bbox, keypoints_17}, ...]`
  - Hungarian matching (scipy `linear_sum_assignment`) on Euclidean displacement
    from last known positions to assign detections to (P1, P2) each frame
- [ ] Port `get_ground_position` heel/ankle fallback tiers to COCO keypoint indices
  (COCO 15=L_ankle, 16=R_ankle; no heel landmark — use ankles only)
- [ ] Validate: run both pipelines on the same clip and compare court-space traces
- [ ] Benchmark: measure frames/second on CPU vs Option A baseline

### Expected outcome
No more coupling events; cleaner traces near the T junction where players cross.

---

## Option C — Hybrid (MediaPipe crop + YOLO verifier)

Keep the fast MediaPipe crop trackers for per-frame speed but run YOLOv8-pose
every `VERIFY_EVERY_N` frames as a ground-truth anchor.

### Tasks
- [ ] Add YOLOv8-pose as an optional dependency (`pip install ultralytics`)
- [ ] Implement `_yolo_verify(frame, last_pos_1, last_pos_2)` — runs full-frame
  YOLO, Hungarian-matches detections, returns corrected (P1, P2) positions
- [ ] Replace `_try_reassign` (which uses the fragile top/bottom split) with
  `_yolo_verify` in the periodic verification sweep
- [ ] Guard with `try/except ImportError` so the pipeline still runs without
  `ultralytics` installed (falls back to Option A behaviour)
- [ ] Compare coupling event counts: Option A baseline vs Option C on full match

### Expected outcome
Near-zero coupling at a fraction of Option B's compute cost.

---

## Deliverable
At least one of Option B or C fully working on a complete match video, with a
side-by-side court-trace comparison showing the improvement over Option A.
""",
    },
    {
        "day":    13,
        "title":  "Day 13 — Per-Rally Stats & Winner/Error Tagging",
        "labels": ["week-2"],
        "closed": False,
        "body": """\
## Objective
Know who won each rally and why.

## Tasks
- [ ] Compute rally length in shots and seconds
- [ ] Compute player distance covered per rally
- [ ] Tag rally outcome: Player 1 winner / Player 2 winner / ambiguous
- [ ] Compute winner and error rates per player
- [ ] Export `rally_stats.csv`

## Deliverable
`rally_stats.csv` with outcome, duration, distance, and shot count per rally.
""",
    },
    {
        "day":    14,
        "title":  "Day 14 — Batch Processing & Week 2 Testing",
        "labels": ["week-2"],
        "closed": False,
        "body": """\
## Objective
Process a full match split across multiple files in one command.

## Tasks
- [ ] Accept a directory as input — process all video segments in sequence
- [ ] Accumulate positions and rally data across segments before building outputs
- [ ] Cache homography per court to skip re-calibration on known courts
- [ ] Validate zone percentages and distances against manual estimates on 3 match videos

## Deliverable
Single command processes an entire match; all Week 2 stats validated.
""",
    },
    # ── Week 3 ────────────────────────────────────────────────────────────
    {
        "day":    15,
        "title":  "Day 15 — Shot Classification",
        "labels": ["week-3"],
        "closed": False,
        "body": """\
## Objective
Identify the type of every shot from ball trajectory alone.

## Tasks
- [ ] Extract shot features from ball trajectory segments: angle, speed, height, court origin/destination
- [ ] Train or adapt a lightweight classifier to label: Drive, Drop, Lob, Boast, Volley, Serve
- [ ] Label confidence threshold — flag uncertain shots rather than guess
- [ ] Output shot type per rally event

## Deliverable
Every detected shot labelled with type and confidence score.
""",
    },
    {
        "day":    16,
        "title":  "Day 16 — Front Wall Heatmap",
        "labels": ["week-3"],
        "closed": False,
        "body": """\
## Objective
Show where on the front wall shots are landing — a feature Rally Vision charges $99/match for.

## Tasks
- [ ] Project ball trajectory forward to estimate front wall impact point
- [ ] Accumulate impact points into a front wall 2D grid
- [ ] Overlay density heatmap onto a front wall diagram (nick areas highlighted)
- [ ] Separate front wall heatmap per shot type (drives vs drops vs lobs)

## Deliverable
Front wall shot placement heatmap — shows whether a player is hitting nicks or telegraphing shots.
""",
    },
    {
        "day":    17,
        "title":  "Day 17 — Shot Mix & Effectiveness Report",
        "labels": ["week-3"],
        "closed": False,
        "body": """\
## Objective
Tell a player not just what shots they hit but which ones work.

## Tasks
- [ ] Cross-reference shot type with rally outcome (winner/error/neutral)
- [ ] Compute shot effectiveness rate per type: % of rallies won after this shot type
- [ ] Compute shot mix: how often each shot type is used relative to total shots
- [ ] Surface "most effective shot" and "highest error rate shot" as headline insights

## Deliverable
Shot mix report with effectiveness rates — directly actionable for a coach session.
""",
    },
    {
        "day":    18,
        "title":  "Day 18 — Automatic Video Highlights",
        "labels": ["week-3"],
        "closed": False,
        "body": """\
## Objective
Auto-clip the best and worst moments without manual scrubbing.

## Tasks
- [ ] Extract video clips for: winners (last 5 s of winning rallies), errors (last 5 s of errors), longest rallies, fastest rallies
- [ ] Compile a highlights reel (concatenated clips) using OpenCV or FFmpeg
- [ ] Save individual clips to `output/highlights/` folder

## Deliverable
Auto-generated highlights folder after every analysis run.
""",
    },
    {
        "day":    19,
        "title":  "Day 19 — Streamlit Web UI",
        "labels": ["week-3"],
        "closed": False,
        "body": """\
## Objective
Make everything accessible without the command line.

## Tasks
- [ ] Build a single-page Streamlit app: video upload, player selection, court calibration, analysis trigger
- [ ] Progress bar with estimated time remaining during processing
- [ ] Display heatmap (floor + front wall), stats cards, zone breakdown, shot mix chart
- [ ] Download buttons for PNG heatmap, CSV stats, highlights folder ZIP

## Deliverable
`streamlit run app.py` launches a fully working web UI.
""",
    },
    {
        "day":    20,
        "title":  "Day 20 — In-Browser Court Calibration",
        "labels": ["week-3"],
        "closed": False,
        "body": """\
## Objective
Replace the OpenCV click-window with something coaches can actually use.

## Tasks
- [ ] Display first video frame as a static image in the browser
- [ ] User clicks four court corners directly in the UI using `streamlit-drawable-canvas`
- [ ] Pass clicked coordinates to the existing homography pipeline
- [ ] Show calibration preview overlay before confirming

## Deliverable
Court calibration that works entirely inside the browser — no desktop popup windows.
""",
    },
    {
        "day":    21,
        "title":  "Day 21 — PDF Match Report",
        "labels": ["week-3"],
        "closed": False,
        "body": """\
## Objective
Give coaches a shareable, professional match report.

## Tasks
- [ ] One-page PDF: player name, match date, floor heatmap, front wall heatmap, stats table, zone breakdown, shot mix, top insights
- [ ] "Export PDF" button in the web UI
- [ ] Use `reportlab` for layout control
- [ ] Auto-generate after every analysis run (also saved to `output/`)

## Deliverable
Professional PDF report downloadable from the app in one click.
""",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_issues():
    ensure_labels()

    print(f"Creating {len(ISSUES)} issues on {REPO}...")
    print()

    for issue in ISSUES:
        day    = issue["day"]
        title  = issue["title"]
        labels = issue["labels"]
        body   = issue["body"]
        closed = issue["closed"]

        print(f"  [{day:02d}] {title}")

        payload = {"title": title, "body": body, "labels": labels}
        result  = api("POST", f"/repos/{REPO}/issues", payload)

        if result and closed:
            number = result["number"]
            api("PATCH", f"/repos/{REPO}/issues/{number}", {"state": "closed"})
            print(f"       → #{number} created and closed ✓")
        elif result:
            print(f"       → #{result['number']} created ✓")
        else:
            print(f"       → FAILED")

        time.sleep(0.5)  # stay well inside GitHub's rate limit (5000 req/hr)

    print()
    print(f"Done! View issues at: https://github.com/{REPO}/issues")


if __name__ == "__main__":
    create_issues()
