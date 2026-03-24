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
- **Basic court mapping sanity check**: click the T junction and one service box corner on each test video; verify the projected court-space coordinates match WSF ground truth (T = 3.2 m, 5.49 m) to within 0.3 m — catches calibration errors before Week 2 builds on top of them

Deliverable:
- Stable MVP ready for demo to players or coaches
- Court mapping error confirmed < 0.3 m on at least two courts

---

### Definition of Done ✅
- Player tracked end-to-end without code changes
- Heatmap overlaid on court diagram and exported as PNG
- Works on multiple videos

---

## Week 2 — Player Metrics, YOLO Tracking & Video Overlay

### Goals
Complete player tracking, compute the core movement stats, and render results onto video.
Ball tracking is deferred to Week 4 — everything in this week works from player positions alone.

---

### Day 8 — Distance, Speed & T-Position Stats ✅

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

### Day 9 — Court Zone Breakdown ✅

**Objective:** Show exactly which court zones a player dominates or neglects

Tasks:
- Divide the court into 9 standard zones: Front-L, Front-C, Front-R, Mid-L, T (highlighted), Mid-R, Back-L, Back-C, Back-R
- Compute frames and percentage of time in each zone per player
- Overlay zone labels with percentages on the heatmap PNG

Deliverable:
- Annotated zone heatmap — directly comparable to SmartSquash's court region output

---

### Day 10 — SQLite Database & Player Identity

**Objective:** Replace flat JSON run history with a proper database before building any profile features on top of it

Tasks:
- Design schema with `src/db.py`:
  - `players (id, name, created_at)`
  - `matches (id, player_id, opponent_id, match_date, venue, court_id, video_path)`
  - `match_stats (match_id, duration_s, distance_m, avg_speed_ms, peak_speed_ms, t_time_pct, front_pct, back_pct, zone_json, coverage_pct, fatigue_index, lateral_bias)`
  - `rallies (id, match_id, start_frame, end_frame, winner, duration_s, distance_p1_m, distance_p2_m, n_shots)`
  - `shots (id, rally_id, frame_idx, player_idx, shot_type, confidence, origin_zone, wall_x, wall_y)`
- Write `db.py`: `init_db()`, `save_match()`, `get_player_history(player_id)`, `get_match_stats(match_id)`, `get_all_matches()`
- Migrate any existing `run_history.json` entries into the new database automatically on first run
- Add `--player1-name` / `--player2-name` CLI flags to `main.py`; players are looked up or created by name in the `players` table
- Store raw smoothed position arrays in `output/{player_name}/{match_date}/` — stats can be recomputed without re-running video analysis when formulas improve
- Keep `run_history.json` as a human-readable export, regenerated from the database on each run

Deliverable:
- `src/db.py` with full CRUD; all future runs persist automatically to SQLite; `--player1-name` flag working end-to-end

---

### Day 11 — Advanced Movement Analytics

**Objective:** Replace simple distance/speed stats with metrics that actually tell a coach something new

Tasks:

**Better velocity estimation — Savitzky-Golay filter**
- Replace frame-to-frame Euclidean distance with `scipy.signal.savgol_filter` applied to position arrays
- Re-derive `avg_speed_ms` and `peak_speed_ms` from the filtered velocity signal; update `stats.py`

**Court coverage — convex hull area**
- Compute the convex hull of all court-space positions using `scipy.spatial.ConvexHull`
- Report `coverage_pct` = hull area / court area; overlay the hull outline on the heatmap PNG

**Dominant movement axis — PCA**
- Stack all frame-to-frame displacement vectors `(dx, dy)` and run `sklearn.decomposition.PCA(n_components=2)`
- PC1 is the player's dominant movement axis; overlay a small arrow on the court diagram
- Report `lateral_bias`: ratio of lateral (x) to depth (y) variance

**Fatigue index**
- Divide the match into 2-minute rolling windows; compute mean speed per window
- Report `fatigue_index` = speed drop from first quartile to last quartile, normalised by first-quartile speed

Deliverable:
- Updated stats table: convex hull coverage %, dominant axis angle, lateral/depth bias, fatigue index
- Heatmap PNG annotated with convex hull outline and PC1 arrow

---

### Day 12 — Player Tracking: YOLOv8-pose ✅

**Objective:** Eliminate player identity coupling by replacing the MediaPipe dual-crop trackers with YOLOv8-pose full-frame detection

Implemented **Option B**: YOLOv8-pose detects all people in a single forward pass; Hungarian minimum-cost matching assigns detections to (P1, P2) each frame — coupling is architecturally impossible.

Tasks completed:
- Replaced `extract_pose.py` tracking loop with `extract_pose_yolo.py` — full-frame YOLOv8n-pose inference
- Ported ground-position fallback tiers to COCO-17 keypoint indices (ankles → knees → hips)
- Hungarian assignment via `scipy.optimize.linear_sum_assignment`
- Camera-cut filter, jump cap, hold-last-value semantics carried over
- `--tracker yolo` (default) / `--tracker mediapipe` flag added to `main.py`
- Raw pixel positions saved to `output/last_positions_yolo.npz`

Deliverable:
- `src/extract_pose_yolo.py` fully working; `python src/main.py` defaults to YOLO tracker

---

### Day 13 — Movement Intelligence Visualisations

**Objective:** Give coaches visual tools that go beyond what any current commercial product offers — all derived from player positions, no ball needed

Tasks:

**Movement vector field**
- Divide the court into a 6×9 grid; for each cell compute the mean displacement vector of positions passing through it
- Overlay arrows on the court diagram showing direction and magnitude of movement in each zone
- Save to `output/movement_vector_field.png`

**Speed distribution comparison**
- Violin plot of instantaneous speed distributions for both players on the same axis
- Save to `output/speed_distributions.png`

**Return-to-T tendency** *(approximate without ball — uses midpoint between last and next positions as proxy for shot event)*
- Record player position 1 s and 2 s after each approximate shot event
- `T_recovery_rate` = % of events after which player is within T_RADIUS_M of the T — a key coaching metric
- Save to `output/t_recovery.png`

Deliverable:
- All three visualisations saved to `output/` automatically at end of each run

---

### Day 14 — Testing, Validation & Video Overlay

**Objective:** Validate all Week 2 outputs and render tracking results back onto the original video

**Part A — Testing & Validation**

Tasks:
- Run the full pipeline (player tracking → stats → zone breakdown → movement visualisations) end-to-end
- Verify zone percentages sum to 100%; check front/back split is plausible
- Verify convex hull area is within court bounds
- Fix any bugs found

**Part B — Video Overlay** (`src/render_overlay.py`)

Tasks:
- Render tracked player positions as coloured dots on each video frame
- Display live stats (distance, speed, zone) as a HUD in the corner
- Project a mini top-down court diagram into a corner of the frame showing live positions
- Export annotated video to `output/annotated_video.mp4`

Deliverable:
- All Week 2 stats validated
- `output/annotated_video.mp4` with player tracking and HUD overlaid on the original footage

---

### Week 2 Definition of Done
- Distance, speed, T-time, and zone stats generated for every run ✅
- YOLOv8-pose tracker with Hungarian assignment working as default ✅
- SQLite database stores all match data; `--player1-name` flag working
- Advanced movement metrics (convex hull, PCA, fatigue index) computed per match
- Annotated video output with player tracking HUD

---

## Week 3 — Web UI, Court Calibration & Match Reports

### Goals
Make the tool usable by coaches without touching the command line. All features in this week
run on player tracking data alone — no ball tracking required.

---

### Day 15 — Streamlit Web UI

**Objective:** Make everything accessible without the command line

Tasks:
- Build a single-page Streamlit app: video upload, player selection, court calibration, analysis trigger
- Progress bar with estimated time remaining during processing
- Display heatmap (floor), stats cards, zone breakdown, movement visualisations
- Download buttons for PNG heatmap, CSV stats

Deliverable:
- `streamlit run app.py` launches a fully working web UI

---

### Day 16 — In-Browser Court Calibration

**Objective:** Replace the OpenCV click-window with something coaches can actually use

Tasks:
- Display first video frame as a static image in the browser
- User clicks court corners directly in the UI using `streamlit-drawable-canvas`
- Pass clicked coordinates to the existing homography pipeline
- Show calibration preview overlay before confirming

Deliverable:
- Court calibration that works entirely inside the browser — no desktop popup windows

---

### Day 17 — PDF Match Report

**Objective:** Give coaches a shareable, professional match report

Tasks:
- One-page PDF: player name, match date, floor heatmap, stats table, zone breakdown, movement visualisations, top insights
- "Export PDF" button in the web UI
- Use `reportlab` for layout control
- Auto-generate after every analysis run (also saved to `output/`)

Deliverable:
- Professional PDF report downloadable from the app in one click

---

### Day 18 — Match History & Progress Dashboard

**Objective:** Show a player whether they are improving week over week

Tasks:
- Session history sidebar in the Streamlit UI — click any past match to reload its heatmaps and stats
- Line charts for key metrics over time: distance per match, T-time %, fatigue index, convex hull coverage
- Highlight statistically significant improvements or regressions (> 10 % change flagged with context)
- "Biggest improvement this month" and "Area needing most work" headline cards from the database
- Export a monthly progress PDF report

Deliverable:
- Progress dashboard showing measurable trend lines across all stored matches

---

### Day 19 — Benchmarking & Context

**Objective:** Give players context for their numbers so stats feel meaningful rather than abstract

**Core tasks (required):**
- Define a hardcoded `BENCHMARK_REFERENCE` dict in `src/benchmarks.py` with club-level and recreational ranges for each key metric, sourced from published squash science literature
- In the Streamlit UI, display each stat alongside its benchmark range with a traffic-light colour: green = within elite range, amber = club level, red = below club level
- Add a "Context" card: one sentence per metric explaining what the number means for a squash player

**Optional stretch goals:**
- Curate actual stats from 3–5 publicly available PSA match analysis papers and replace hardcoded ranges with real measured values
- Show a percentile ranking: "Your T-time is in the 60th percentile for club-level players"

Deliverable:
- Traffic-light benchmark comparison shown in the Streamlit UI for every key metric; context sentence per metric

---

### Day 20 — Ground Truth Accuracy Validation Suite (player tracking)

**Objective:** Produce a repeatable, automated accuracy report for player tracking so numbers shown to coaches can be trusted

Tasks:
- **Court mapping error**: project the 9 known court line intersections through the homography; compare against WSF ground-truth coordinates; report per-point error and RMSE
- **Player position accuracy**: manually label foot/heel position in 100 evenly-sampled frames; compare against homography-projected pose landmark; report RMSE in metres
- Save all results to `output/accuracy_report.json` and a printed summary table

Target metrics:
- Court mapping RMSE < 0.3 m
- Player position RMSE < 0.5 m

Deliverable:
- `python src/validate_accuracy.py` produces the player-tracking accuracy report in under 5 minutes

---

### Week 3 Definition of Done
- Streamlit UI runs end-to-end from upload to PDF report using player tracking data
- In-browser court calibration replaces OpenCV click window
- Progress dashboard working across at least 3 stored matches
- Benchmark percentile rankings shown for key movement metrics
- Batch processing handles a multi-file match in one command

---

## Week 4 — Ball Tracking & Rally Intelligence

### Goals
Solve ball detection properly, then build all the downstream features that depend on it:
rally segmentation, shot classification, front wall heatmap, and video highlights.

---

### Day 22 — Ball Detection

**Objective:** Detect the squash ball in video frames reliably

Tasks:
- Colour-based detection: isolate yellow/white ball using HSV thresholds + contour circularity filter
- Benchmark accuracy on 200 manually labelled frames
- If < 70 % recall, switch to a fine-tuned YOLOv8-nano model
- Store raw ball pixel positions and detection confidence per frame

Target: ≥ 70 % recall on labelled test frames

Deliverable:
- Ball detection script with measured recall/precision against a labelled test set

---

### Day 23 — Ball Tracking & Court-Space Trajectory

**Objective:** Convert noisy ball detections into a smooth trajectory

Tasks:
- Apply a Kalman filter to smooth detections and fill short gaps (< 5 frames)
- Project ball positions through the homography matrix into court space
- Detect "ball lost" segments and flag them
- Visualise trajectory as an animated path on the court diagram

Deliverable:
- Smooth ball trajectory overlaid on the top-down court view

---

### Day 24 — Rally Segmentation & Combined Analysis

**Objective:** Segment the match into individual rallies and compute per-rally statistics

Tasks:
- `segment_rallies(frame_idx, min_gap_frames)` — gaps ≥ `RALLY_END_MIN_FRAMES` (default 20 frames = 0.8 s) mark inter-rally boundaries; configurable via `--min-gap`
- Per-rally stats: duration, approx shot count (velocity direction reversals), P1/P2 court-space distance
- Combined court plot — player scatter (translucent) + ball trajectory coloured by rally number → `output/combined_court.png`
- Rally timeline bar chart → `output/rally_timeline.png`
- Exports `output/rally_boundaries.csv` and `output/rally_stats.csv`

Deliverable:
- `src/segment_rallies.py` — run as `python src/segment_rallies.py [--min-gap N]`
- `rally_boundaries.csv`, `rally_stats.csv`, `combined_court.png`, `rally_timeline.png`

---

### Day 25 — Shot Classification

**Objective:** Identify the type of every shot from ball trajectory alone

Tasks:
- Extract shot features from ball trajectory segments: angle, speed, height, court origin/destination
- Train or adapt a lightweight classifier to label: Drive, Drop, Lob, Boast, Volley, Serve
- Label confidence threshold — flag uncertain shots rather than guess

Deliverable:
- Every detected shot labelled with type and confidence

---

### Day 26 — Front Wall Heatmap & Shot Mix Report

**Objective:** Show where shots land on the front wall and which ones work

Tasks:

**Front wall heatmap**
- Project ball trajectory forward to estimate front wall impact point
- Accumulate impact points into a front wall 2D grid; overlay density heatmap onto a front wall diagram
- Separate front wall heatmap per shot type (drives vs drops vs lobs)

**Shot mix & effectiveness**
- Cross-reference shot type with rally outcome (winner/error/neutral)
- Compute shot effectiveness rate per type: % of rallies won after this shot type
- Surface "most effective shot" and "highest error rate shot" as headline insights

Deliverable:
- Front wall shot placement heatmap
- Shot mix report with effectiveness rates

---

### Day 27 — Automatic Video Highlights & Per-Rally Analytics

**Objective:** Auto-clip key moments and break down stats by rally

Tasks:

**Video highlights**
- Extract clips for: winners (last 5 s of winning rallies), errors, longest rallies, fastest rallies
- Compile a highlights reel using OpenCV or FFmpeg; save to `output/highlights/`

**Per-rally heatmaps**
- For each segmented rally, compute a separate 2D position density for each player
- Generate a small-multiple grid PNG: one mini court per rally, coloured by density

**Shot tempo**
- From the smoothed ball trajectory, measure time between consecutive ball impacts
- Compute mean and standard deviation of inter-shot interval per player

Deliverable:
- `output/highlights/` folder with auto-generated clips
- `output/per_rally_heatmaps.png`
- Shot tempo stats added to match summary table

---

### Day 28 — Ball Tracking Accuracy Validation & Ablation

**Objective:** Quantify ball detection and rally segmentation accuracy; measure each pipeline component's contribution

Tasks:

**Accuracy validation**
- Manually annotate ball position (or absence) in 200 frames; compute recall, precision, F1
- Manually mark rally start/end timestamps on a 5-minute clip; compare against `segment_rallies()` output; compute F1

Target metrics:
- Ball detection recall ≥ 70 %
- Rally segmentation F1 ≥ 0.80

**Ablation study**
- Run the pipeline with one component disabled at a time; report detection rate and RMSE for each condition:
  - No camera-cut filter
  - No Kalman fill
  - No heel/ankle fallback (hip only)
  - BALL_FRAME_SKIP=5 vs BALL_FRAME_SKIP=1
  - `yolov8n` vs `yolov8s` vs `yolov8m` — accuracy vs fps tradeoff

Deliverable:
- `src/ablation.py` runs all conditions automatically; results saved to `output/ablation_results.json`

---

### Week 4 Definition of Done
- Ball detection recall ≥ 70 % measured against a labelled test set
- Rally segmented from ball-lost gaps; `rally_stats.csv` exported automatically
- Every shot classified with type and effectiveness rate
- Front wall heatmap generated per match
- Auto video highlights exported
- Ablation study quantifies every major pipeline component with measured numbers

---

## Week 5 — Deployment & Final Polish

### Goals
Shot outcome analysis, Docker deployment, and a demo-ready finish.

---

### Day 29 — Shot Placement Outcome Map

**Objective:** Show not just where shots land but whether they work

Tasks:

**Shot placement outcome heatmap**
- Take the front wall heatmap and colour each cell by rally outcome after that shot:
  - Green = point won within 2 shots, Red = error within 2 shots, Grey = rally continues
- Save to `output/shot_placement_outcomes.png`

**Shot chain analysis**
- Find the most common 2-shot and 3-shot sequences; flag sequences that disproportionately end in errors
- Print top 5 chains to console

Deliverable:
- Shot placement outcome heatmap and shot chain summary generated automatically

---

### Day 30 — Match Momentum & Rally Speed Trends

**Objective:** Visualise momentum and energy output across the match

Tasks:

**Match momentum timeline**
- A horizontal strip chart showing who was winning the last 5 rallies at each point in the match
- Overlay speed lines per player — shows whether momentum correlates with energy output
- Save to `output/momentum_timeline.png`

**Rally-level fatigue**
- Plot player speed per rally (x = rally number, y = mean speed during rally)
- Overlay a trend line; flag when speed drops more than 15 % below the session mean
- Save to `output/rally_speed_trend.png`

Deliverable:
- `output/momentum_timeline.png` and `output/rally_speed_trend.png` generated automatically

---

### Day 31 — Docker Deployment & Performance Optimisation

**Objective:** Make the app runnable by anyone on any machine with a single command

Tasks:
- Profile the pipeline end-to-end — identify bottlenecks (expected: YOLO inference)
- Parallelise frame decoding and pose inference using `concurrent.futures`
- Target ≤ 2× real-time processing (a 60-min match processes in ≤ 30 min on a laptop)
- Write a `Dockerfile` that installs all dependencies and launches Streamlit
- Test the Docker image on a clean machine or clean VM
- Pin all dependency versions in `requirements.txt`

Deliverable:
- `docker run` launches the full app; processing time at or below 2× real-time benchmark

---

### Day 32 — Batch Processing & Multi-Video Matches

**Objective:** Process a full match split across multiple video files in one command

Tasks:
- Accept a directory as input (`--video-dir`) — process all video segments in alphabetical order
- Accumulate player positions across segments before computing outputs
- Cache homography per court by name (`--court-name`) — skip re-calibration if the court has been seen before
- Validate that all segments have the same resolution and frame rate before processing; warn if not

Deliverable:
- Single command processes an entire multi-file match; all outputs identical to a single-file run

---

### Day 33 — Final Polish & Demo

**Objective:** Ship a demo-ready tool that can be shown to players, coaches, and potential contributors

Tasks:
- End-to-end test on five different match videos across different courts and lighting conditions
- Fix all known edge cases surfaced during testing (player occlusion, bad lighting, camera shake, unusual court colours)
- Record a 5-minute demo: upload → calibrate → analyse → floor heatmap → shot report → scouting report → PDF → progress dashboard
- Update README with architecture diagram, screenshots, and setup instructions

Deliverable:
- Polished, demo-ready application tested on five videos; recorded walkthrough shareable with coaches and players

---

### Week 5 Definition of Done
- Shot placement outcome map generated from ball tracking data
- Match momentum timeline and rally speed trends generated automatically
- Batch processing handles a multi-file match in one command
- Docker deployment tested; processing time ≤ 2× real-time
- Demo recorded and shareable

---

## Full 5-Week Summary

| Week | Theme | Key Outputs |
|---|---|---|
| 1 ✅ | Player tracking + heatmap MVP | Floor heatmap, court mapping, CLI |
| 2 | Player metrics + YOLO tracking + video overlay | Speed/distance stats, zone breakdown, SQLite DB, advanced movement metrics, annotated video |
| 3 | Web UI + match reports + benchmarking | Streamlit UI, in-browser calibration, PDF report, progress dashboard, accuracy validation |
| 4 | Ball tracking + rally intelligence | Ball detection, rally segmentation, shot classification, front wall heatmap, highlights |
| 5 | Shot outcomes + deployment + final polish | Shot placement outcomes, momentum timeline, batch processing, Docker, demo |

## Final Definition of Done

- Processes a full squash match end-to-end from a browser UI in ≤ 30 min on a laptop
- Generates floor heatmap, front wall heatmap, shot mix, effectiveness rates, and PDF report
- Automatically clips video highlights for winners and errors
- Player progress tracked across multiple sessions with trend charts and fatigue index
- Benchmark percentile rankings give context vs club-level and elite players
- Runs locally via Docker — no video ever leaves the user's machine
- Tested on five different match videos across different courts
- Court mapping RMSE < 0.3 m validated by automated accuracy suite
- Demoed end-to-end in under 5 minutes

---

## Known Gaps

These are real limitations in the current implementation that affect accuracy or usability.
They differ from the research directions below, which are improvements beyond the current design.

### Tracking

- **Player auto-detection is fragile** — `auto_detect_players` splits the frame top/bottom to assign P1/P2 on the first frame. If both players are on the same half of the frame, identity is assigned wrong and stays wrong for the rest of the run.
- **No identity-swap correction** — a proximity warning fires when players are close, but there is no automatic correction. Once identities swap, all subsequent stats for that run are wrong.
- **MediaPipe hip-only fallback is biased** — if heels and ankles are invisible (e.g. player crouching behind the tin), the hip midpoint is used as the ground position. This sits ~1 m above the floor plane, introducing systematic Y-position error on low shots and lunges.

### Ball Tracking

- **Ball tracking is not working reliably** — colour-based HSV detection and TrackNetV4 have both been attempted; neither achieves acceptable recall on match footage. All downstream features that depend on ball position (rally segmentation, shot classification, front wall heatmap) are blocked until this is resolved.
- **No validated ball detection baseline** — recall/precision have not been formally measured against a labelled test set. Target is ≥ 70 % recall (Day 23 accuracy suite).

### Rally Segmentation

- **Segmentation depends on ball tracking** — `segment_rallies()` uses gaps in ball detection as rally boundaries. With unreliable ball tracking, rally boundaries are unreliable too.
- **Stats are per-clip, not per-rally** — all movement stats (distance, speed, T-time) are aggregated across the full video clip. Per-rally breakdowns are planned for Day 25 but not yet implemented.

### Calibration & Mapping

- **Manual recalibration required per session** — homography is cached per file path; if the camera shifts even slightly between sessions on the same court, accuracy degrades silently. No automatic drift detection.
- **14-point calibration is time-consuming** — the click UI requires the user to identify 14 specific court line intersections, which takes 3–5 minutes and is error-prone on low-resolution footage.

### Pipeline

- **No end-to-end accuracy report yet** — court mapping RMSE, player position RMSE, ball detection F1, and rally segmentation F1 have not been formally measured. Planned for Day 23.
- **`run_history.json` does not scale** — flat JSON file works for one player/one machine but will break for multi-player lookups and trend queries. SQLite migration planned for Day 22.

---

## Beyond the Plan — Research Directions & Long-Term Vision

These are open-ended improvements that go beyond the 5-week plan.
They are not scheduled but are listed here so nothing is forgotten.

---

### Mathematical & Algorithmic Improvements

**Physics-based ball model (UKF)**
The current Kalman filter uses a constant-velocity state. A squash ball obeys physics:
- Gravity curves the ball on lobs when the camera has any vertical parallax
- Ball decelerates due to air drag between bounces (~exponential decay)
- Ball velocity direction inverts at wall impacts

An Unscented Kalman Filter (UKF) with a physics state `(x, y, vx, vy, ax, ay)` and a bounce-detection trigger would handle these nonlinearities. `filterpy` implements UKF and is already available.

**PELT changepoint detection for shots**
`ruptures.Pelt(model="rbf").fit_predict(velocity_signal)` finds statistically optimal breakpoints in the ball velocity signal — more robust than fixed-threshold detection when the ball is briefly occluded mid-rally.

**Court coverage — entropy metric**
Shannon entropy of the position distribution over the zone grid:
`H = -sum(p_i * log(p_i))`. Maximum entropy = perfectly uniform coverage; low entropy = concentrated in a few zones. More informative than a simple zone percentage and directly comparable across players.

**Speed validation against published radar data**
PSA World Tour match reports occasionally publish serve and drive speeds. Download 2–3 matches where radar gun data is public, run the pipeline, compare the 95th-percentile speed distribution. If the pipeline reads 30 % lower than radar, there is a calibration error to diagnose (likely a homography scale issue).

---

### Long-Term Profile Architecture

As this becomes a multi-year tracking app, a few architectural decisions made in Day 22 will prevent painful rewrites:

**SQLite from Day 22 onwards** (not JSON). The schema in Day 22 above is designed to support everything in Week 5 and beyond without migration. Key tables: `players`, `matches`, `match_stats`, `positions`, `shots`, `rallies`.

**Store raw position arrays per match** in `output/{player_name}/{match_date}/positions.npz`. Stats formulas will improve over the life of the project; you want to recompute all historical stats without re-running video analysis on old footage.

**Player identity is a first-class concept.** Every analysis run should have `--player1-name` and `--player2-name` flags. Internally, players are looked up or created by name in the `players` table. This is the minimum required to power opponent profiling, progress dashboards, and scouting reports.

**Match metadata matters.** Store: opponent name/ID, match date, venue/court name (for homography reuse), score (optional), and tournament/league. Without this, trend analysis is "match #1, #2, #3" rather than "before vs after training block" or "home court vs away court".

**Recompute-safe design.** Never store derived values (zone percentages, speed stats) as the only copy. Always store raw positions and recompute stats at query time, or invalidate the stats cache when the algorithm changes. This allows a formula fix in `stats.py` to automatically propagate to all historical matches.

---

### Additional Experiments to Run

| Experiment | Method | What you learn |
|---|---|---|
| **Resolution sweep** | Downsample 720p → 480p → 360p; measure ball detection recall at each | Quantifies the 720p recommendation with measured data |
| **Frame-skip sweep** | Run ball detection at skip=1,2,3,5; plot recall curve | Finds the minimum skip that still gives acceptable recall |
| **Model size tradeoff** | `yolov8n` vs `yolov8s` vs `yolov8m` on 200-frame test set | Accuracy vs speed Pareto curve — pick the right model for the user's hardware |
| **Lighting robustness** | Run on clips with different lighting (sunny vs overcast, day vs indoor artificial) | Identifies which conditions break the detector and why |
| **Player occlusion handling** | Find clips where players cross paths; measure identity-swap rate | Quantifies how often the tracker couples and how long it takes to recover |
| **Homography stability** | Re-calibrate the same court 5 times; measure variance in projected positions | Quantifies how much calibration noise contributes to position error |
