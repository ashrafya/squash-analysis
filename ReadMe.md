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

### Day 13 — Rally Segmentation, Per-Rally Stats & Winner/Error Tagging

**Objective:** Segment the match into individual rallies and compute per-rally statistics

Tasks:
- **Rally segmentation** (prerequisite): implement `segment_rallies(ball_frames, ball_lost_segments)` in a new `src/segment_rallies.py` — a ball-lost event lasting > `RALLY_END_MIN_FRAMES` (configurable, default 15 frames ≈ 0.6 s) marks an inter-rally gap; everything between two such gaps is one rally; export `output/rally_boundaries.csv` with `(rally_id, start_frame, end_frame)`
- Compute rally length in shots and seconds
- Compute player distance covered per rally
- Tag rally outcome: Player 1 winner / Player 2 winner / ambiguous (based on which player the ball lands near at rally end)
- Compute winner and error rates per player
- Export `rally_stats.csv`

Deliverable:
- `rally_boundaries.csv` with start/end frame of every detected rally
- `rally_stats.csv` with outcome, duration, distance, and shot count per rally

---

### Day 14 — Week 2 Testing & Validation Buffer

**Objective:** Validate all Week 2 outputs against manual estimates before moving on

Tasks:
- Run the full pipeline (player tracking → ball detection → ball tracking → rally segmentation → stats) end-to-end on at least 2 different match videos
- Manually estimate total distance and rally count for a 3-minute clip; compare against pipeline output; flag any discrepancy > 20%
- Verify zone percentages sum to 100% and front/back split is plausible given the video
- Fix any bugs found — this day is intentionally a buffer; do not start Week 3 work here

Note: Batch processing (accepting a directory of video segments for one match) is a deployment-level feature that belongs in Week 5 (Day 33), once the full pipeline including shot classification and PDF reports is in place.

Deliverable:
- All Week 2 stats validated on 2+ videos; known bugs logged or fixed

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


## Week 4 — Foundation, Validation & Advanced Analytics

### Goals
Before building the app's most complex user-facing features, validate that the underlying
pipeline is accurate enough to trust, replace the temporary JSON storage with a database
that scales to years of match history, and compute the advanced metrics that make the tool
genuinely superior to commercial alternatives.
All of Week 5 (scouting, progress tracking, deployment) builds on this foundation.

---

### Day 22 — SQLite Database & Player Identity

**Objective:** Replace flat JSON run history with a proper database from day one of profile work — do not build opponent profiling and progress dashboards on a system you know you will have to replace

Why SQLite now: `output/run_history.json` works for a single player on a single machine but cannot power multi-player opponent lookups, trend queries across seasons, or cross-match aggregations. SQLite is a single file, ships with Python, has zero server setup, and handles every query needed through Week 5.

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
- Store raw smoothed position arrays in `output/{player_name}/{match_date}/` — stats can be recomputed from raw positions without re-running video analysis when formulas improve
- Keep `run_history.json` as a human-readable export, regenerated from the database on each run

Deliverable:
- `src/db.py` with full CRUD; all future runs persist automatically to SQLite; `--player1-name` flag working end-to-end

---

### Day 23 — Ground Truth Accuracy Validation Suite

**Objective:** Produce a repeatable, automated accuracy report so every future change can be measured against a baseline — and so the numbers shown to coaches can be trusted

Doing this in Week 4 (not at the very end) means problems can still be fixed before complex Week 5 features are built on top of them.

Tasks:
- **Court mapping error**: on a calibrated test frame, project the 9 known court line intersections (T junction, four short-line ends, four service box corners) through the homography; compare against WSF ground-truth coordinates; report per-point error and RMSE
- **Player position accuracy**: manually label foot/heel position in 100 evenly-sampled frames; compare against homography-projected pose landmark; report RMSE in metres
- **Ball detection recall & precision**: manually annotate ball position (or absence) in 200 frames; run detector; compute recall, precision, F1 by method (YOLO vs motion vs merged)
- **Rally segmentation F1**: manually mark rally start/end timestamps on a 5-minute clip; compare against `segment_rallies()` output; compute F1 score
- Save all results to `output/accuracy_report.json` and a printed summary table

Target metrics:
- Court mapping RMSE < 0.3 m
- Player position RMSE < 0.5 m
- Ball detection recall ≥ 70 %
- Rally segmentation F1 ≥ 0.80

Deliverable:
- `python src/validate_accuracy.py` produces a full accuracy report in under 5 minutes; all four targets met before proceeding to Week 5

---

### Day 24 — Advanced Movement Analytics

**Objective:** Replace simple distance/speed stats with metrics that actually tell a coach something new

Tasks:

**Better velocity estimation — Savitzky-Golay filter**
- Replace frame-to-frame Euclidean distance with `scipy.signal.savgol_filter` applied to position arrays
- A polynomial least-squares smoother that simultaneously smooths positions and produces clean first and second derivatives (velocity, acceleration) without the lag of a rolling mean
- Re-derive `avg_speed_ms` and `peak_speed_ms` from the filtered velocity signal; update `stats.py`

**Court coverage — convex hull area**
- Compute the convex hull of all court-space positions using `scipy.spatial.ConvexHull`
- Report `coverage_pct` = hull area / court area; overlay the hull outline on the heatmap PNG
- Two players can have identical zone percentages but very different hulls — one is in corners, the other covers the full court

**Dominant movement axis — PCA**
- Stack all frame-to-frame displacement vectors `(dx, dy)` and run `sklearn.decomposition.PCA(n_components=2)`
- PC1 is the player's dominant movement axis; PC2 is perpendicular
- Overlay a small arrow on the court diagram showing PC1 direction and magnitude
- Report `lateral_bias`: ratio of lateral (x) to depth (y) variance

**Fatigue index**
- Divide the match into 2-minute rolling windows; compute mean speed per window
- Fit a linear regression (slope + p-value) to the speed-over-time series
- Report `fatigue_index` = speed drop from first quartile to last quartile, normalised by first-quartile speed

Deliverable:
- Updated stats table: convex hull coverage %, dominant axis angle, lateral/depth bias, fatigue index
- Heatmap PNG annotated with convex hull outline and PC1 arrow

---

### Day 25 — Per-Rally Analytics & Shot Tempo

**Objective:** Break aggregate stats down by rally to reveal patterns that disappear in averages

Tasks:

**Per-rally heatmaps**
- For each segmented rally, compute a separate 2D position density for each player
- Generate a small-multiple grid PNG: one mini court per rally, coloured by density
- Immediately shows whether court coverage narrows under pressure in long rallies

**Rally-level fatigue**
- Plot player speed per rally (x = rally number, y = mean speed during rally)
- Overlay a trend line; flag when speed drops more than 15 % below the session mean
- Save to `output/rally_speed_trend.png`

**Shot tempo**
- From the smoothed ball trajectory, measure time between consecutive ball impacts
- Compute mean and standard deviation of inter-shot interval per player — low variance = consistent rhythm; high variance = reactive, scrambling play

**Changepoint detection for shot segmentation**
- Replace hard velocity-threshold shot detection with PELT changepoint detection using the `ruptures` library (`pip install ruptures`)
- More robust to noisy frames than a fixed threshold

Deliverable:
- Per-rally heatmap grid saved to `output/per_rally_heatmaps.png`
- Shot tempo stats added to match summary table
- Rally speed trend chart saved to `output/rally_speed_trend.png`

---

### Day 26 — Movement Intelligence Visualisations

**Objective:** Give coaches visual tools that go beyond what any current commercial product offers

Tasks:

**Movement vector field**
- Divide the court into a 6×9 grid; for each cell compute the mean displacement vector of positions passing through it
- Overlay arrows on the court diagram showing direction and magnitude of movement in each zone
- Save to `output/movement_vector_field.png`

**Return-to-T tendency**
- After every shot event, record the player's position 1 s and 2 s later
- Scatter plot: shot origin → 2 s post-shot position, connected by an arrow coloured green (reached T) or red (did not)
- `T_recovery_rate` = % of shots after which player recovers the T within 2 s — a key coaching metric
- Save to `output/t_recovery.png`

**Match momentum timeline**
- A horizontal strip chart showing who was winning the last 5 rallies at each point in the match
- Overlay speed lines per player — shows whether momentum correlates with energy output
- Save to `output/momentum_timeline.png`

**Speed distribution comparison**
- Violin plot of instantaneous speed distributions for both players on the same axis; fatter upper tail = more explosive burst movement
- Save to `output/speed_distributions.png`

Deliverable:
- All four visualisations saved to `output/` automatically at end of each run

---

### Day 27 — Shot Placement Outcome Colouring & Opponent Response Map

**Objective:** Show not just where shots land but whether they work — and what the opponent does next

Tasks:

**Shot placement outcome heatmap**
- Take the front wall heatmap from Day 16 and colour each cell by rally outcome after that shot:
  - Green = point won within 2 shots
  - Red = error committed within 2 shots
  - Grey = rally continues (neutral)
- Reveals which front wall areas are high-value for this player vs which generate errors
- Save to `output/shot_placement_outcomes.png`

**Opponent response map**
- For each shot from court zone X to front wall zone Y, record which court zone the opponent occupies on their next shot
- Build a conditional probability matrix; display as a court diagram with response arrows
- Example insight: "tight backhand drive from back-right almost always pushes opponent into back-left"
- Save to `output/opponent_response_map.png`

**Shot chain analysis**
- Find the most common 2-shot and 3-shot sequences using n-gram counting over the shot-type sequence
- Flag sequences that disproportionately end in errors vs winners
- Print top 5 chains to console

Deliverable:
- Shot placement outcome heatmap, opponent response map, and shot chain summary all generated automatically

---

### Day 28 — Ablation Studies & Systematic Benchmarking

**Objective:** Quantify the contribution of every pipeline component so future changes are evidence-based

Tasks:

**Ablation study — tracking pipeline**

Run the full pipeline with one component disabled at a time; report detection rate and player position RMSE vs ground truth for each condition:

| Condition | Expected impact |
|---|---|
| Baseline (all components on) | — |
| No camera-cut filter | More false positives from replays and ads |
| No Kalman fill | More fragmented trajectories, more lost-segment events |
| No heel/ankle fallback (hip only) | Higher Y-position bias on lunges |
| BALL_FRAME_SKIP=5 vs BALL_FRAME_SKIP=1 | Ball detection recall drops sharply at skip=5 |

**Ablation study — ball detection**

| Condition | Recall | Precision |
|---|---|---|
| YOLO only | baseline | baseline |
| Motion only | expected higher recall | expected lower precision |
| Merged (current) | expected best F1 | — |
| 360p vs 720p source | expected −20 pp recall at 360p | — |

**Model size and resolution experiments**
- Run `yolov8n.pt` vs `yolov8s.pt` vs `yolov8m.pt` on the same 200-frame test set; plot accuracy vs fps tradeoff curve
- Downsample 720p to 480p and 360p using OpenCV; plot ball detection recall curve vs resolution

Deliverable:
- `src/ablation.py` runs all conditions automatically and prints a comparison table
- Results saved to `output/ablation_results.json`

---

### Week 4 Definition of Done
- SQLite database stores all match data; `--player1-name` and `--player2-name` flags tag every run
- Automated accuracy report passes all four target metrics (court RMSE < 0.3 m, player RMSE < 0.5 m, ball recall ≥ 70 %, rally F1 ≥ 0.80)
- Savitzky-Golay velocity, convex hull coverage, PCA movement axis, and fatigue index computed and stored per match
- Per-rally heatmaps, shot tempo, movement visualisations, and shot placement outcome map all generated automatically
- Ablation study quantifies every major pipeline component with measured numbers

---

## Week 5 — Scouting, Progress Tracking & Deployment

### Goals
Build the user-facing features that turn validated analytics into a usable product for players
and coaches over a full season. All of these features build directly on the SQLite database,
accurate pipeline, and advanced metrics established in Week 4.

---

### Day 29 — Match History & Progress Dashboard

**Objective:** Show a player whether they are improving week over week, backed by data from the database built in Day 22

Tasks:
- Display a session history sidebar in the Streamlit UI — click any past match to reload its heatmaps, stats, and visualisations
- Line charts for key metrics over time: distance per match, T-time %, fatigue index, shot error rate, shot mix evolution, convex hull coverage, T-recovery rate
- Highlight statistically significant improvements or regressions (> 10 % change flagged with context)
- "Biggest improvement this month" and "Area needing most work" headline cards generated from the database
- Export a monthly progress PDF report

Deliverable:
- Progress dashboard showing measurable improvement (or decline) trend lines across all stored matches

---

### Day 30 — Opponent Profiling

**Objective:** Let a player analyse opponent footage to identify exploitable tendencies

Tasks:
- Run the full analysis pipeline on opponent footage (same tool, `--player1-name` set to opponent's name)
- All opponent data stored in the same SQLite database under the opponent's player record
- Aggregate opponent tendencies from all stored matches: preferred shot mix, weakest zone, front/back bias, serve patterns, fatigue profile (when in the match do they slow down?)
- Identify exploitable patterns: e.g., "rarely plays backhand boast", "error rate spikes in back-left after long rallies", "T-recovery rate drops below 50 % late in games"

Deliverable:
- Opponent profile page in the Streamlit UI with tendencies and exploitable patterns — equivalent to Cross Court Analytics' scouting feature

---

### Day 31 — Pre-Match Scouting Report

**Objective:** Generate a one-page battle plan a player can review the night before a match

Tasks:
- Query the opponent profile (Day 30) and the player's own profile (Day 29) from the database
- Generate a pre-match PDF:
  - Opponent shot mix heatmap and weak zones (uses shot placement outcome map from Day 27)
  - Opponent fatigue profile — when in the match do they slow down?
  - Opponent response map — which shot from which zone puts them under most pressure?
  - Player's own strengths vs opponent's weaknesses — where do they align?
- Surface top 3 tactical recommendations as bullet points on the first page

Deliverable:
- Pre-match scouting PDF generated in one click from the Streamlit UI

---

### Day 32 — Benchmarking & Context

**Objective:** Give players context for their numbers so stats feel meaningful rather than abstract

The core deliverable is a simple hardcoded reference table — no data curation required. The optional stretch goals add richer comparisons if time and data allow.

**Core tasks (required):**
- Define a hardcoded `BENCHMARK_REFERENCE` dict in `src/benchmarks.py` with reasonable club-level and recreational ranges for each key metric, sourced from published squash science literature (several peer-reviewed studies report elite and sub-elite movement profiles):
  - `distance_m`: typical range for a competitive match clip
  - `avg_speed_ms`, `peak_speed_ms`: published ranges for club vs elite
  - `t_time_pct`: expected T-position percentage for different levels
  - `t_recovery_rate`: typical recovery rates
- In the Streamlit UI, display each stat alongside its benchmark range with a simple traffic-light colour: green = within elite range, amber = club level, red = below club level
- Add a "Context" card: one sentence per metric explaining what the number means for a squash player (e.g. "Elite players spend > 60 % of rally time within 1.25 m of the T")

**Optional stretch goals (do if time allows):**
- Curate actual stats from 3–5 publicly available PSA match analysis papers or videos and replace the hardcoded ranges with real measured values
- Show a percentile ranking: "Your T-time is in the 60th percentile for club-level players" — requires having enough reference data points to form a distribution
- "Elite overlay": ghost an elite player's heatmap behind the user's heatmap — requires sourcing and processing actual elite match footage, which is a significant data collection effort

Deliverable:
- Traffic-light benchmark comparison shown in the Streamlit UI for every key metric; context sentence per metric; stretch goals clearly marked as optional in the code

---

### Day 33 — Batch Processing & Multi-Video Matches

**Objective:** Process a full match split across multiple video files in one command

Batch processing is placed here (not Day 14) because the full pipeline — shot classification, rally stats, PDF reports, SQLite storage — is now complete. Batching earlier would have produced only partial outputs.

Tasks:
- Accept a directory as input (`--video-dir`) — process all video segments in alphabetical order
- Accumulate player positions, rally data, and shot events across segments before computing outputs (no artificial match boundary between files)
- Cache homography per court by name (`--court-name`) — skip re-calibration if the court has been seen before
- Validate that all segments have the same resolution and frame rate before processing; warn if not

Deliverable:
- Single command processes an entire multi-file match; all outputs identical to a single-file run

---

### Day 34 — Docker Deployment & Performance Optimisation

**Objective:** Make the app runnable by anyone on any machine with a single command

Tasks:
- Profile the pipeline end-to-end — identify bottlenecks (expected: MediaPipe and YOLO inference)
- Parallelise frame decoding and pose inference using `concurrent.futures`
- Target ≤ 2× real-time processing (a 60-min match processes in ≤ 30 min on a laptop)
- Write a `Dockerfile` that installs all dependencies and launches Streamlit
- Test the Docker image on a clean machine or clean VM
- Pin all dependency versions in `requirements.txt`

Deliverable:
- `docker run` launches the full app; processing time at or below 2× real-time benchmark

---

### Day 35 — Final Polish & Demo

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
- Match history and progress dashboard working across at least 3 stored matches
- Opponent profiling and pre-match scouting PDF generated automatically
- Benchmark percentile rankings shown for key metrics
- Batch processing handles a multi-file match in one command
- Docker deployment tested; processing time ≤ 2× real-time
- Demo recorded and shareable

---

## Full 5-Week Summary

| Week | Theme | Key Outputs |
|---|---|---|
| 1 ✅ | Player tracking + heatmap MVP | Floor heatmap, court mapping, CLI |
| 2 | Ball tracking + rally intelligence | Speed/distance stats, rally segmentation, zone breakdown |
| 3 | Shot intelligence + web UI | Shot classification, front wall heatmap, highlights, Streamlit UI, PDF report |
| 4 | Foundation + validation + advanced analytics | SQLite DB, accuracy suite, advanced movement metrics, ablation studies |
| 5 | Scouting + progress + deployment | Player profiles, opponent scouting, benchmarking, batch processing, Docker |

## Final Definition of Done

- Processes a full squash match end-to-end from a browser UI in ≤ 30 min on a laptop
- Generates floor heatmap, front wall heatmap, shot mix, effectiveness rates, and PDF report
- Automatically clips video highlights for winners and errors
- Player progress tracked across multiple sessions with trend charts and fatigue index
- Opponent profiling and pre-match scouting report generated from opponent footage
- Benchmark percentile rankings give context vs club-level and elite players
- Runs locally via Docker — no video ever leaves the user's machine
- Tested on five different match videos across different courts
- Court mapping RMSE < 0.3 m validated by automated accuracy suite
- Demoed end-to-end in under 5 minutes

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
