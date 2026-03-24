"use client";
import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { ArrowLeft, AlertCircle } from "lucide-react";
import UploadZone from "@/components/upload/UploadZone";
import ProcessingStatus, { type PipelineStep } from "@/components/upload/ProcessingStatus";
import ResultsPanel, { type AnalysisResult } from "@/components/upload/ResultsPanel";
import { submitJob, pollJob, type JobStatus } from "@/lib/api";

type PageState = "idle" | "processing" | "done" | "error";

const INITIAL_STEPS: PipelineStep[] = [
  { id: "upload",       label: "Uploading video",       detail: "Transferring MP4 to analysis server",               status: "pending" },
  { id: "player",       label: "Player tracking",        detail: "YOLOv8-pose detecting both players frame-by-frame",  status: "pending" },
  { id: "ball",         label: "Ball detection",         detail: "YOLOv8 + MOG2 motion fallback per frame",            status: "pending" },
  { id: "smooth",       label: "Kalman smoothing",       detail: "Filtering ball trajectory and filling short gaps",   status: "pending" },
  { id: "rally",        label: "Rally segmentation",     detail: "Segmenting rallies from ball-lost gaps",             status: "pending" },
  { id: "report",       label: "Generating PDF report",  detail: "Compiling all charts into a shareable PDF",          status: "pending" },
];

const STATUS_TO_STEP: Record<JobStatus["status"], number> = {
  queued: -1,
  uploading: 0,
  player_tracking: 1,
  ball_detection: 2,
  ball_tracking: 3,
  rally_segmentation: 4,
  generating_report: 5,
  done: 6,
  error: -1,
};

function stepsFromStatus(status: JobStatus["status"]): PipelineStep[] {
  const currentIdx = STATUS_TO_STEP[status] ?? -1;
  return INITIAL_STEPS.map((s, i) => ({
    ...s,
    status:
      status === "error" ? (i < currentIdx ? "done" : i === currentIdx ? "error" : "pending")
      : i < currentIdx ? "done"
      : i === currentIdx ? "running"
      : "pending",
  }));
}

export default function AnalyzePage() {
  const [pageState, setPageState] = useState<PageState>("idle");
  const [steps, setSteps] = useState<PipelineStep[]>(INITIAL_STEPS);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => { if (pollRef.current) clearTimeout(pollRef.current); }, []);

  async function handleFile(file: File) {
    setPageState("processing");
    setErrorMsg(null);
    setSteps(stepsFromStatus("uploading"));
    setProgress(5);

    let jobId: string;
    try {
      jobId = await submitJob(file);
    } catch (e) {
      setPageState("error");
      setErrorMsg(e instanceof Error ? e.message : "Upload failed");
      return;
    }

    function poll() {
      pollRef.current = setTimeout(async () => {
        try {
          const status = await pollJob(jobId);
          setSteps(stepsFromStatus(status.status));
          setProgress(status.progress);

          if (status.status === "done") {
            setResult({
              jobId,
              files: status.result ?? {},
              stats: status.stats,
            });
            setPageState("done");
          } else if (status.status === "error") {
            setErrorMsg(status.error ?? "Analysis failed");
            setPageState("error");
          } else {
            poll();
          }
        } catch {
          setErrorMsg("Lost connection to server");
          setPageState("error");
        }
      }, 2500);
    }
    poll();
  }

  return (
    <div className="min-h-screen bg-bg-dark font-body">
      {/* Top coloured strip */}
      <div className="h-1 w-full flex fixed top-0 z-50">
        <div className="flex-1 bg-lime"/><div className="flex-1 bg-coral"/>
        <div className="flex-1 bg-purple"/><div className="flex-1 bg-cyan"/>
      </div>

      <div className="max-w-3xl mx-auto px-4 sm:px-6 pt-12 pb-10">
        {/* Back link */}
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-muted hover:text-lime transition-colors text-sm font-heading mb-8 cursor-pointer"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to home
        </Link>

        {/* Header */}
        <div className="mb-8 border-b-2 border-lime pb-8">
          <div className="inline-block bg-lime border-2 border-ink px-3 py-1 mb-4 shadow-block">
            <span className="section-label text-ink">Analysis Tool</span>
          </div>
          <h1 className="font-heading font-extrabold text-4xl sm:text-5xl text-foreground leading-tight">
            Analyse a Match.
          </h1>
          <p className="text-muted font-body mt-3">
            Upload an MP4 from behind the back wall. The pipeline runs fully
            automatically — no manual tagging required.
          </p>
        </div>

        {/* Idle: upload zone */}
        {pageState === "idle" && (
          <div className="border-2 border-white/15 bg-surface-dark p-6">
            <UploadZone onFile={handleFile} />
            <div className="mt-5 pt-5 border-t border-white/10">
              <p className="text-xs text-muted font-body">
                <strong className="text-lime/70 font-heading">Requirements:</strong>{" "}
                Fixed camera behind the back wall · MP4 format · Up to 500 MB ·
                360p–1080p supported.
              </p>
            </div>
          </div>
        )}

        {/* Processing */}
        {pageState === "processing" && (
          <div className="border-2 border-lime/40 bg-surface-dark p-6 shadow-block-lime">
            <ProcessingStatus steps={steps} progress={progress} />
          </div>
        )}

        {/* Error */}
        {pageState === "error" && (
          <div className="border-2 border-coral bg-surface-dark p-6 shadow-block-coral space-y-4">
            <div className="flex items-start gap-3 text-coral" role="alert">
              <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-heading font-semibold">Analysis failed</p>
                <p className="text-sm font-body mt-1">{errorMsg}</p>
              </div>
            </div>
            <button
              onClick={() => { setPageState("idle"); setSteps(INITIAL_STEPS); setProgress(0); }}
              className="text-sm text-lime hover:text-chalk transition-colors cursor-pointer font-heading"
            >
              ← Try again
            </button>
          </div>
        )}

        {/* Done */}
        {pageState === "done" && result && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-success font-heading font-semibold">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Analysis complete
            </div>
            <ResultsPanel result={result} />
          </div>
        )}
      </div>
    </div>
  );
}
