"""
FastAPI backend — wraps the squash analysis pipeline.

Endpoints:
  POST /analyse           — upload MP4, returns {job_id}
  GET  /jobs/{id}/status  — poll job progress
  GET  /jobs/{id}/files/{filename} — serve output files
"""

import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # squash-analysis/
SRC  = ROOT / "src"
JOBS_DIR = ROOT / "website" / "api" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Squash Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (replace with Redis/DB for production) ─────────────
jobs: dict[str, dict] = {}

# ── Schemas ────────────────────────────────────────────────────────────────
class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    error: Optional[str] = None
    result: Optional[dict] = None
    stats: Optional[dict] = None


# ── Pipeline steps ─────────────────────────────────────────────────────────
STEPS = [
    ("uploading",           0,   None),
    ("player_tracking",     15,  ["python", "main.py", "--reuse"]),         # reuse if already calibrated
    ("ball_detection",      40,  ["python", "detect_ball.py"]),
    ("ball_tracking",       65,  ["python", "track_ball.py", "--no-anim"]),
    ("rally_segmentation",  80,  ["python", "segment_rallies.py"]),
    ("generating_report",   90,  None),                                     # PDF generation inline
    ("done",                100, None),
]

OUTPUT_FILES = {
    "heatmap_p1":     "heatmap_player1.png",
    "heatmap_p2":     "heatmap_player2.png",
    "zone_breakdown": "zone_breakdown.png",
    "combined_court": "combined_court.png",
    "timeseries":     "timeseries.png",
    "rally_timeline": "rally_timeline.png",
    "histograms":     "histograms.png",
    "report_pdf":     "report.pdf",
}


async def run_pipeline(job_id: str, video_path: Path, output_dir: Path):
    """Run the pipeline steps and update job state."""
    job = jobs[job_id]

    try:
        # ── Step 1: player tracking ────────────────────────────────────────
        job["status"] = "player_tracking"
        job["progress"] = 15
        env = os.environ.copy()
        env["SQUASH_VIDEO_PATH"] = str(video_path)
        env["SQUASH_OUTPUT_DIR"] = str(output_dir)

        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "main.py"],
            cwd=str(SRC),
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Player tracking failed: {result.stderr[:500]}")

        # ── Step 2: ball detection ─────────────────────────────────────────
        job["status"] = "ball_detection"
        job["progress"] = 40
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "detect_ball.py"],
            cwd=str(SRC),
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ball detection failed: {result.stderr[:500]}")

        # ── Step 3: Kalman smoothing ───────────────────────────────────────
        job["status"] = "ball_tracking"
        job["progress"] = 65
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "track_ball.py", "--no-anim"],
            cwd=str(SRC),
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ball tracking failed: {result.stderr[:500]}")

        # ── Step 4: rally segmentation ─────────────────────────────────────
        job["status"] = "rally_segmentation"
        job["progress"] = 80
        result = await asyncio.to_thread(
            subprocess.run,
            ["python", "segment_rallies.py"],
            cwd=str(SRC),
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Rally segmentation failed: {result.stderr[:500]}")

        # ── Step 5: collect outputs & generate PDF ─────────────────────────
        job["status"] = "generating_report"
        job["progress"] = 90

        # Copy output files to job directory
        pipeline_output = ROOT / "output"
        for key, filename in OUTPUT_FILES.items():
            src_file = pipeline_output / filename
            if key == "report_pdf":
                continue  # generated below
            if src_file.exists():
                shutil.copy2(src_file, output_dir / filename)

        _generate_pdf(output_dir)

        # Read rally stats if available
        stats = _read_stats(pipeline_output)

        # Build result file map
        result_files = {
            key: filename
            for key, filename in OUTPUT_FILES.items()
            if (output_dir / filename).exists()
        }

        job["status"] = "done"
        job["progress"] = 100
        job["result"] = result_files
        job["stats"] = stats

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


def _generate_pdf(output_dir: Path):
    """Compile output images into a simple PDF using fpdf2."""
    try:
        from fpdf import FPDF  # type: ignore

        pdf = FPDF(orientation="L", unit="mm", format="A4")
        pdf.set_auto_page_break(False)

        images = [
            ("Player 1 Heatmap",   "heatmap_player1.png"),
            ("Player 2 Heatmap",   "heatmap_player2.png"),
            ("Zone Breakdown",     "zone_breakdown.png"),
            ("Combined Court",     "combined_court.png"),
            ("Speed Over Time",    "timeseries.png"),
            ("Rally Timeline",     "rally_timeline.png"),
            ("Position Histograms","histograms.png"),
        ]

        for title, filename in images:
            img_path = output_dir / filename
            if not img_path.exists():
                continue
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, title, ln=True, align="C")
            pdf.image(str(img_path), x=10, y=20, w=277)  # A4 landscape width = 297, leave margins

        pdf.output(str(output_dir / "report.pdf"))
    except ImportError:
        # fpdf2 not installed — skip PDF generation silently
        pass


def _read_stats(output_dir: Path) -> dict | None:
    """Parse rally_stats.csv and return summary stats."""
    import csv

    stats_path = output_dir / "rally_stats.csv"
    if not stats_path.exists():
        return None

    rows = []
    with open(stats_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    try:
        rally_count = len(rows)
        total_dur = sum(float(r.get("duration_s", 0)) for r in rows)
        p1_dist = sum(float(r.get("dist_p1_m", 0)) for r in rows)
        p2_dist = sum(float(r.get("dist_p2_m", 0)) for r in rows)
        p1_speed = (p1_dist / total_dur * 3.6) if total_dur else 0
        p2_speed = (p2_dist / total_dur * 3.6) if total_dur else 0
        return {
            "rally_count": rally_count,
            "total_duration_s": round(total_dur, 1),
            "p1_distance_m": round(p1_dist, 1),
            "p2_distance_m": round(p2_dist, 1),
            "p1_avg_speed_kmh": round(p1_speed, 1),
            "p2_avg_speed_kmh": round(p2_speed, 1),
        }
    except (ValueError, KeyError):
        return None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted.")

    job_id = str(uuid.uuid4())
    output_dir = JOBS_DIR / job_id
    output_dir.mkdir(parents=True)

    # Save uploaded file
    video_path = output_dir / "input.mp4"
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    jobs[job_id] = {"status": "uploading", "progress": 5, "error": None, "result": None, "stats": None}

    # Run pipeline in background
    asyncio.create_task(run_pipeline(job_id, video_path, output_dir))

    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        error=job.get("error"),
        result=job.get("result"),
        stats=job.get("stats"),
    )


@app.get("/jobs/{job_id}/files/{filename}")
async def serve_file(job_id: str, filename: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Sanitise filename to prevent path traversal
    safe_name = Path(filename).name
    file_path = JOBS_DIR / job_id / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(file_path))


@app.get("/health")
async def health():
    return {"status": "ok"}
