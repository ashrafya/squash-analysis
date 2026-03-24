"use client";
import { useState } from "react";
import Image from "next/image";
import { Download, ZoomIn, X, FileDown } from "lucide-react";
import Button from "@/components/ui/Button";
import clsx from "clsx";

export interface AnalysisResult {
  jobId: string;
  files: { heatmap_p1?: string; heatmap_p2?: string; zone_breakdown?: string; combined_court?: string; timeseries?: string; rally_timeline?: string; histograms?: string; report_pdf?: string; };
  stats?: { rally_count: number; total_duration_s: number; p1_distance_m: number; p2_distance_m: number; p1_avg_speed_kmh: number; p2_avg_speed_kmh: number; };
}

const chartGroups = [
  { id: "heatmaps",   label: "Heatmaps",       active: "bg-lime text-ink border-lime",    keys: ["heatmap_p1","heatmap_p2"] as const,              titles: ["Player 1","Player 2"] },
  { id: "court",      label: "Court View",      active: "bg-coral text-chalk border-coral", keys: ["combined_court","zone_breakdown"] as const,       titles: ["Combined Court","Zone Breakdown"] },
  { id: "stats",      label: "Speed & Rallies", active: "bg-purple text-chalk border-purple", keys: ["timeseries","rally_timeline","histograms"] as const, titles: ["Speed","Rally Timeline","Histograms"] },
];

const statColors = [
  "bg-lime text-ink","bg-coral text-chalk","bg-purple text-chalk",
  "bg-cyan text-ink","bg-ink text-lime","bg-lime/30 text-ink",
];

export default function ResultsPanel({ result }: { result: AnalysisResult }) {
  const [activeTab, setActiveTab] = useState("heatmaps");
  const [zoomSrc,   setZoomSrc]   = useState<string | null>(null);
  const group   = chartGroups.find(g => g.id === activeTab)!;
  const baseUrl = `/api/jobs/${result.jobId}/files/`;

  return (
    <div className="space-y-5">
      {result.stats && (
        <div className="grid grid-cols-3 sm:grid-cols-6 border-2 border-lime">
          {[
            { label: "Rallies",     value: String(result.stats.rally_count) },
            { label: "P1 Distance", value: `${result.stats.p1_distance_m.toFixed(1)}m` },
            { label: "P2 Distance", value: `${result.stats.p2_distance_m.toFixed(1)}m` },
            { label: "P1 Speed",    value: `${result.stats.p1_avg_speed_kmh.toFixed(1)}` },
            { label: "P2 Speed",    value: `${result.stats.p2_avg_speed_kmh.toFixed(1)}` },
            { label: "Duration",    value: `${Math.round(result.stats.total_duration_s)}s` },
          ].map((s, i) => (
            <div key={s.label} className={clsx("p-4 text-center border-r-2 border-lime/30 last:border-r-0", statColors[i])}>
              <p className="font-heading font-extrabold text-2xl leading-none">{s.value}</p>
              <p className="section-label mt-1.5 opacity-70">{s.label}</p>
            </div>
          ))}
        </div>
      )}

      <div className="flex border-2 border-lime w-fit">
        {chartGroups.map(g => (
          <button key={g.id} onClick={() => setActiveTab(g.id)}
            className={clsx(
              "px-5 py-2.5 font-heading font-semibold text-sm border-r-2 border-lime last:border-r-0 cursor-pointer transition-colors duration-100",
              activeTab === g.id ? g.active : "bg-surface-dark text-muted hover:text-foreground"
            )}>
            {g.label}
          </button>
        ))}
      </div>

      <div className={clsx("border-2 border-lime grid", group.keys.length > 1 ? "grid-cols-1 sm:grid-cols-2 divide-x-2 divide-lime/30" : "grid-cols-1")}>
        {group.keys.map((key, i) => {
          const src = result.files[key as keyof typeof result.files];
          if (!src) return null;
          const full = `${baseUrl}${src}`;
          return (
            <div key={key} className="bg-surface-dark">
              <div className="flex items-center justify-between px-4 py-2 border-b border-lime/20">
                <span className="section-label text-muted">{group.titles[i]}</span>
                <button onClick={() => setZoomSrc(full)} className="text-muted hover:text-lime transition-colors cursor-pointer" aria-label="Zoom">
                  <ZoomIn className="w-4 h-4"/>
                </button>
              </div>
              <div className="relative aspect-video"><Image src={full} alt={group.titles[i]} fill className="object-contain p-3"/></div>
              <div className="px-4 py-2 border-t border-lime/20">
                <a href={full} download>
                  <Button size="sm" variant="ghost" className="text-xs text-muted hover:text-lime gap-1">
                    <Download className="w-3.5 h-3.5"/> Save PNG
                  </Button>
                </a>
              </div>
            </div>
          );
        })}
      </div>

      {result.files.report_pdf && (
        <a href={`${baseUrl}${result.files.report_pdf}`} download>
          <Button size="lg" variant="lime" className="w-full gap-2">
            <FileDown className="w-5 h-5"/> Download Full PDF Report
          </Button>
        </a>
      )}

      {zoomSrc && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-4" onClick={() => setZoomSrc(null)}>
          <button onClick={() => setZoomSrc(null)} className="absolute top-4 right-4 text-white/60 hover:text-white cursor-pointer" aria-label="Close">
            <X className="w-7 h-7"/>
          </button>
          <div className="relative max-w-4xl max-h-[80vh] w-full h-full">
            <Image src={zoomSrc} alt="Zoom" fill className="object-contain"/>
          </div>
        </div>
      )}
    </div>
  );
}
