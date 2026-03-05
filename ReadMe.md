# Squash Analysis — 4-Week Build Plan

## What We're Building

A self-hosted, open-source squash analytics tool that gives players and coaches the same
insights as Rally Vision ($99/match) and SmartSquash ($50/month) — for free, on their own
hardware, with no video upload to a third party.

---

## Competitive Landscape

| Tool | Price | Key Strength | Weakness |
|---|---|---|---|
| SmartSquash AI | $50/month | Heatmaps, T-position, AI recommendations | No shot-level data, SaaS only |
| Rally Vision | $99/match | Shot-by-shot rating, front wall heatmaps, video highlights | Extremely expensive, SaaS |
| Cross Court Analytics | Enterprise | 10k data points/match, opponent scouting, federation-level | Human taggers, not automated |
| **This project** | **Free / self-hosted** | All of the above, automated, runs locally | In progress |

**Our differentiation:**
- Fully automated — no human taggers, no manual shot logging
- Runs on your own machine — no video leaves your premises
- Free and open-source — every feature available to every player
- Front wall shot heatmap — know not just where you stand but where you aim
- Opponent scouting — build a data dossier on any opponent before a match

---

## Week 1 — Player Tracking & Heatmap MVP ✅ COMPLETE

### Goals
- Upload a squash match video (fixed camera behind back wall)
- Select one player (left or right)
- Calibrate the court by clicking four corners
- Automatically track the selected player's position
- Generate a top-down heatmap showing where the player spent time on the court
- Export the heatmap as a static image

---

### Day 1 — Video Ingestion & Pose Detection ✅

**Objective:** Extract player body position from video frames

Tasks:
- Load `.mp4` video using OpenCV
- Run pose detection on each frame (MediaPipe Pose)
- Compute player hip centre `(left_hip + right_hip) / 2`
- Store raw pixel positions per frame

Deliverable:
- Script that plots player movement in pixel space

---

### Day 2 — Player Selection & Tracking ✅

**Objective:** Track a single player consistently

Tasks:
- Detect both players in frame
- Separate players by horizontal position (left vs right)
- Keep only the selected player
- Filter noisy detections

Deliverable:
- Clean, continuous player path in pixel coordinates

---

### Day 3 — Court Calibration & Mapping ✅

**Objective:** Map camera pixels to court coordinates

Tasks:
- Allow user to click four court corners
- Compute homography matrix
- Convert pixel positions to court coordinates

Deliverable:
- Player movement correctly mapped within court bounds

---

### Day 4 — Heatmap Generation ✅

**Objective:** Convert positions into a density map

Tasks:
- Bin player positions into a 2D grid
- Accumulate time spent per cell
- Apply Gaussian smoothing
- Normalise values

Deliverable:
- Raw heatmap array representing player presence

---

### Day 5 — Visualisation ✅

**Objective:** Create a clear, readable heatmap output

Tasks:
- Generate top-down squash court diagram
- Overlay heatmap onto court
- Export result as PNG image

Deliverable:
- Presentation-ready heatmap image

---

### Day 6 — CLI Integration ✅

**Objective:** Make the tool usable without code changes

Tasks:
- Add `--debug` flag to show live tracking animation while processing
- Add `--calibrate` flag to force recalibration of the court homography
- Handle common failure cases (bad video, missing player)

Deliverable:
- End-to-end usable tool via command line

---

### Day 7 — Testing & Polish ✅

**Objective:** MVP readiness

Tasks:
- Test on at least two different match videos
- Handle common failure cases (bad video, missing player)
- Clean up code

Deliverable:
- Stable MVP ready for demo to players or coaches

---

### Definition of Done ✅
- Player tracked end-to-end without code changes
- Heatmap overlaid on court diagram and exported as PNG
- Works on multiple videos

---

## Week 2 — Ball Tracking, Rally Intelligence & Player Metrics

### Goals
Match SmartSquash's movement stats and add rally-level insight they do not offer.

---

### Day 8 — Distance, Speed & T-Position Stats

**Objective:** Give players the numbers coaches actually care about

Tasks:
- Convert court coordinates to real-world metres using known court dimensions (9.75 m × 6.4 m)
- Compute frame-to-frame displacement → total distance covered
- Compute average speed and peak speed in m/s using frame rate
- Compute T-position time: percentage of frames within 1 m of the T
- Print a per-player stats table at end of each run

Deliverable:
- Console stats table: distance (m), avg speed, peak speed, T-time %

---

### Day 9 — Court Zone Breakdown

**Objective:** Show exactly which court zones a player dominates or neglects

Tasks:
- Divide the court into 9 standard zones: Front-L, Front-C, Front-R, Mid-L, T (highlighted), Mid-R, Back-L, Back-C, Back-R
- Compute frames and percentage of time in each zone per player
- Overlay zone labels with percentages on the heatmap PNG

Deliverable:
- Annotated zone heatmap — directly comparable to SmartSquash's court region output

---

### Day 10 — Ball Detection

**Objective:** Detect the squash ball in video frames reliably

Tasks:
- Colour-based detection: isolate yellow/white ball using HSV thresholds + contour circularity filter
- Benchmark accuracy on 200 manually labelled frames
- If <70% recall, switch to a fine-tuned YOLOv8-nano model (fast enough for real-time)
- Store raw ball pixel positions and detection confidence per frame

Deliverable:
- Ball detection script with ≥70% recall on test frames

---

### Day 11 — Ball Tracking & Court-Space Trajectory

**Objective:** Convert noisy ball detections into a smooth trajectory

Tasks:
- Apply a Kalman filter to smooth detections and fill short gaps (<5 frames)
- Project ball positions through the homography matrix into court space
- Detect "ball lost" segments and flag them
- Visualise trajectory as an animated path on the court diagram

Deliverable:
- Smooth ball trajectory overlaid on the top-down court view

---

### Day 12 — Player Tracking: Option B & C Experiments

**Objective:** Eliminate player identity coupling at its root by replacing (or augmenting) the MediaPipe dual-crop trackers with true multi-person detection

Day 11 shipped **Option A** (velocity cap + coupling counter + periodic re-assignment via 2×2 Hungarian matching), which reduces but does not eliminate coupling. This day experiments with two architecturally stronger alternatives.

**Option B — Full rewrite with YOLOv8-pose (recommended)**

YOLOv8-pose detects all people in a single forward pass, making coupling architecturally impossible.

Tasks:
- `pip install ultralytics` and smoke-test `yolov8n-pose.pt` on a sample frame
- Replace `extract_pose.py` tracking loop with YOLOv8-pose inference: full frame → detections → Hungarian matching to (P1, P2) by minimum displacement
- Port `get_ground_position` heel/ankle fallback tiers to COCO keypoint indices (COCO 15=L_ankle, 16=R_ankle)
- Validate: run both pipelines on the same clip and compare court-space traces
- Benchmark: measure frames/second on CPU vs Option A baseline

**Option C — Hybrid (MediaPipe crop + YOLO verifier)**

Keep the fast MediaPipe crop trackers but use YOLOv8-pose every `VERIFY_EVERY_N` frames as a ground-truth anchor.

Tasks:
- Implement `_yolo_verify(frame, last_pos_1, last_pos_2)` — full-frame YOLO + Hungarian re-assignment
- Replace `_try_reassign` (fragile top/bottom split) with `_yolo_verify` in the periodic verification sweep
- Guard with `try/except ImportError` so the pipeline still runs without `ultralytics` installed (falls back to Option A)
- Compare coupling event counts: Option A baseline vs Option C on full match

Deliverable:
- At least one of Option B or C fully working on a complete match video, with a side-by-side court-trace comparison showing the improvement over Option A

---

### Day 13 — Per-Rally Stats & Winner/Error Tagging

**Objective:** Know who won each rally and why

Tasks:
- Compute rally length in shots and seconds
- Compute player distance covered per rally
- Tag rally outcome: Player 1 winner / Player 2 winner / ambiguous
- Compute winner and error rates per player
- Export `rally_stats.csv`

Deliverable:
- `rally_stats.csv` with outcome, duration, distance, and shot count per rally

---

### Day 14 — Batch Processing & Week 2 Testing

**Objective:** Process a full match split across multiple files in one command

Tasks:
- Accept a directory as input — process all video segments in sequence
- Accumulate positions and rally data across segments before building outputs
- Cache homography per court to skip re-calibration on known courts
- Validate zone percentages and distances against manual estimates on 3 match videos

Deliverable:
- Single command processes an entire match; all Week 2 stats validated

---

### Week 2 Definition of Done
- Distance, speed, T-time, and zone stats generated for every run
- Ball tracked reliably enough to segment rallies with >80% accuracy
- `rally_stats.csv` exported automatically
- Batch processing works on a full match folder

---

## Week 3 — Shot Intelligence, Front Wall Heatmap & Web UI

### Goals
Match and exceed Rally Vision's shot-level analysis — at zero cost to the user.

---

### Day 15 — Shot Classification

**Objective:** Identify the type of every shot from ball trajectory alone

Tasks:
- Extract shot features from ball trajectory segments: angle, speed, height, court origin/destination
- Train or adapt a lightweight classifier to label: Drive, Drop, Lob, Boast, Volley, Serve
- Label confidence threshold — flag uncertain shots rather than guess
- Output shot type per rally event

Deliverable:
- Every detected shot labelled with type and confidence

---

### Day 16 — Front Wall Heatmap

**Objective:** Show where on the front wall shots are landing — a feature Rally Vision charges for

Tasks:
- Project ball trajectory forward to estimate front wall impact point
- Accumulate impact points into a front wall 2D grid
- Overlay density heatmap onto a front wall diagram (nick areas highlighted)
- Separate front wall heatmap per shot type (drives vs drops vs lobs)

Deliverable:
- Front wall shot placement heatmap — shows whether a player is hitting nicks or telegraphing shots

---

### Day 17 — Shot Mix & Effectiveness Report

**Objective:** Tell a player not just what shots they hit but which ones work

Tasks:
- Cross-reference shot type with rally outcome (winner/error/neutral)
- Compute shot effectiveness rate per type: % of rallies won after this shot type
- Compute shot mix: how often each shot type is used relative to total shots
- Surface "most effective shot" and "highest error rate shot" as headline insights

Deliverable:
- Shot mix report with effectiveness rates — directly actionable for a coach session

---

### Day 18 — Automatic Video Highlights

**Objective:** Auto-clip the best and worst moments without manual scrubbing

Tasks:
- Extract video clips for: winners (last 5 s of winning rallies), errors (last 5 s of errors), longest rallies, fastest rallies
- Compile a highlights reel (concatenated clips) using OpenCV or FFmpeg
- Save individual clips to `output/highlights/` folder

Deliverable:
- Auto-generated highlights folder after every analysis run

---

### Day 19 — Streamlit Web UI

**Objective:** Make everything accessible without the command line

Tasks:
- Build a single-page Streamlit app: video upload, player selection, court calibration, analysis trigger
- Progress bar with estimated time remaining during processing
- Display heatmap (floor + front wall), stats cards, zone breakdown, shot mix chart
- Download buttons for PNG heatmap, CSV stats, highlights folder ZIP

Deliverable:
- `streamlit run app.py` launches a fully working web UI

---

### Day 20 — In-Browser Court Calibration

**Objective:** Replace the OpenCV click-window with something coaches can actually use

Tasks:
- Display first video frame as a static image in the browser
- User clicks four court corners directly in the UI using `streamlit-drawable-canvas`
- Pass clicked coordinates to the existing homography pipeline
- Show calibration preview overlay before confirming

Deliverable:
- Court calibration that works entirely inside the browser — no desktop popup windows

---

### Day 21 — PDF Match Report

**Objective:** Give coaches a shareable, professional match report

Tasks:
- One-page PDF: player name, match date, floor heatmap, front wall heatmap, stats table, zone breakdown, shot mix, top insights
- "Export PDF" button in the web UI
- Use `reportlab` for layout control
- Auto-generate after every analysis run (also saved to `output/`)

Deliverable:
- Professional PDF report downloadable from the app in one click

---

### Week 3 Definition of Done
- Every shot classified with type and effectiveness rate
- Front wall heatmap generated per match
- Auto video highlights exported
- Streamlit UI runs end-to-end from upload to PDF report
- Non-technical coach can use the app without instructions

---

## Week 4 — Opponent Scouting, Progress Tracking & Deployment

### Goals
Add the features that Cross Court Analytics charges national federations for — automated,
free, and self-hosted. A player who has used this tool for a month should walk into a
match knowing their opponent's tendencies cold.

---

### Day 22 — Match History & Player Profiles

**Objective:** Build persistent player profiles across multiple matches

Tasks:
- Save each analysis run (heatmap PNGs, rally CSV, shot CSV, stats JSON) to `output/{player_name}/{date}/`
- Build a player profile that aggregates stats across all saved matches
- Display a session history sidebar in the Streamlit UI — click any past match to reload it
- Show trend lines for distance covered, T-time %, error rate over time

Deliverable:
- Player profile with multi-match history and trend charts

---

### Day 23 — Progress Dashboard

**Objective:** Show a player whether they are actually improving week over week

Tasks:
- Line charts for key metrics over time: distance per match, T-time %, shot error rate, shot mix evolution
- Highlight statistically significant improvements or regressions (>10% change flagged)
- "Biggest improvement this month" and "Area needing most work" headline cards
- Export a monthly progress PDF report

Deliverable:
- Progress dashboard showing measurable improvement (or decline) trend lines

---

### Day 24 — Opponent Profiling

**Objective:** Let a player analyse an opponent's video to prepare for a match

Tasks:
- Run the full analysis pipeline on opponent footage (same tool, different player label)
- Store opponent profile under `output/opponents/{opponent_name}/`
- Aggregate opponent tendencies: preferred shot mix, weakest zone, back court vs front court bias, serve patterns
- Identify exploitable patterns: e.g., "rarely plays backhand boast", "error rate spikes in back-left"

Deliverable:
- Opponent profile with tendencies and exploitable patterns — equivalent to Cross Court Analytics' scouting feature

---

### Day 25 — Pre-Match Scouting Report

**Objective:** Generate a one-page battle plan before a match

Tasks:
- Combine player's own profile (strengths/weaknesses) with opponent profile
- Generate a pre-match PDF: opponent shot mix, opponent's weak zones, recommended shot strategy, "play to your strengths" vs "exploit their gaps" recommendations
- Surface top 3 tactical recommendations as bullet points on the first page

Deliverable:
- Pre-match scouting PDF that a player can review the night before a tournament match

---

### Day 26 — Benchmarking Against Elite Players

**Objective:** Give amateur players context for their numbers

Tasks:
- Curate a small benchmark dataset of elite player heatmaps and stats (publicly available match footage)
- Show percentile rankings: "Your T-time is in the 60th percentile for club-level players"
- Show "Elite comparison" overlay: elite player heatmap ghosted behind the user's heatmap
- Flag the single biggest gap between user's patterns and elite patterns

Deliverable:
- Benchmark comparison view with percentile rankings and elite overlay

---

### Day 27 — Docker Deployment & Performance Optimisation

**Objective:** Make the app runnable by anyone on any machine

Tasks:
- Profile the pipeline — identify bottlenecks (expected: MediaPipe inference)
- Parallelise frame decoding and pose inference using `concurrent.futures`
- Target ≤2× real-time processing (a 60-min match processes in ≤30 min on a laptop)
- Write a `Dockerfile` that installs all dependencies and launches Streamlit
- Test the Docker image on a clean machine
- Pin all dependency versions in `requirements.txt`

Deliverable:
- `docker run` launches the full app; processing time halved vs Week 1 baseline

---

### Day 28 — Final Polish, Accuracy Validation & Demo

**Objective:** Ship a demo-ready tool that can be shown to players, coaches, and investors

Tasks:
- Ground truth accuracy test: manually label 100 frames, measure court mapping error → target <0.3 m
- End-to-end test on five different match videos across different courts and lighting conditions
- Fix all known edge cases (player occlusion, bad lighting, camera shake)
- Record a 5-minute demo: upload → calibrate → analyse → heatmaps → shot report → scouting → PDF
- Update README with architecture diagram, screenshots, and setup instructions

Deliverable:
- Polished, demo-ready application with validated accuracy and a recorded walkthrough

---

## Week 4 Definition of Done
- Player profile persists across multiple matches with trend charts
- Opponent profiling and pre-match scouting report generated automatically
- Benchmark percentile rankings shown for key metrics
- Docker deployment tested and documented
- Processing time ≤2× real-time on a standard laptop
- Court mapping error <0.3 m validated on 100 labelled frames
- Demo recorded and shareable

---

## Full 4-Week Summary

| Week | Theme | Key Outputs |
|---|---|---|
| 1 ✅ | Player tracking + heatmap MVP | Floor heatmap, court mapping, CLI |
| 2 | Ball tracking + rally intelligence | Speed/distance stats, rally CSV, zone breakdown |
| 3 | Shot intelligence + web UI | Shot classification, front wall heatmap, video highlights, Streamlit app, PDF report |
| 4 | Scouting + progress + deployment | Player profiles, opponent scouting, benchmark ranking, Docker |

## Final Definition of Done

- Processes a full squash match end-to-end from a browser UI in ≤30 min on a laptop
- Generates floor heatmap, front wall heatmap, shot mix, effectiveness rates, and PDF report
- Automatically clips video highlights for winners and errors
- Player progress tracked across multiple sessions with trend charts
- Opponent profiling and pre-match scouting report generated from opponent footage
- Benchmark percentile rankings give context vs club-level and elite players
- Runs locally via Docker — no video ever leaves the user's machine
- Tested on five different match videos across different courts
- Court mapping error <0.3 m
- Demoed end-to-end in under 5 minutes
