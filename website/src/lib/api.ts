import type { AnalysisResult } from "@/components/upload/ResultsPanel";

export interface JobStatus {
  job_id: string;
  status: "queued" | "uploading" | "player_tracking" | "ball_detection" | "ball_tracking" | "rally_segmentation" | "generating_report" | "done" | "error";
  progress: number; // 0-100
  error?: string;
  result?: AnalysisResult["files"];
  stats?: AnalysisResult["stats"];
}

export async function submitJob(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/api/analyse", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail ?? `Upload failed (${res.status})`);
  }
  const data = await res.json();
  return data.job_id as string;
}

export async function pollJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`/api/jobs/${jobId}/status`);
  if (!res.ok) throw new Error(`Status check failed (${res.status})`);
  return res.json();
}
