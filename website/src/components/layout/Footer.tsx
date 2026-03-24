import Link from "next/link";
import { Github } from "lucide-react";

const ticker = ["Player Heatmaps","Ball Tracking","Rally Segmentation","Zone Breakdown","Speed Charts","PDF Reports","YOLOv8 Powered","Open Source"];

export default function Footer() {
  const items = [...ticker, ...ticker];
  return (
    <footer className="bg-ink border-t-2 border-lime">
      <div className="border-b-2 border-lime py-3 overflow-hidden">
        <div className="marquee-track">
          {items.map((t, i) => (
            <span key={i} className="font-heading font-bold text-lime text-xs uppercase tracking-widest mx-8 whitespace-nowrap">
              {t} ✦
            </span>
          ))}
        </div>
      </div>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-lime border-2 border-lime flex items-center justify-center">
            <svg viewBox="0 0 20 20" fill="none" className="w-3.5 h-3.5" stroke="#0A0A0A" strokeWidth={2.5}>
              <circle cx="10" cy="10" r="3"/>
              <path d="M10 2v3M10 15v3M3.5 3.5l2 2M14.5 14.5l2 2M2 10h3M15 10h3"/>
            </svg>
          </div>
          <span className="font-heading font-bold text-chalk">SquashAnalysis</span>
        </div>
        <p className="text-xs text-muted font-body text-center">Free & open-source. No subscription, no data uploaded to third parties.</p>
        <div className="flex items-center gap-4">
          <a href="https://github.com/" target="_blank" rel="noopener noreferrer"
            className="text-muted hover:text-lime transition-colors cursor-pointer" aria-label="GitHub">
            <Github className="w-5 h-5"/>
          </a>
          <Link href="/analyze" className="font-heading font-semibold text-xs text-lime hover:text-chalk transition-colors">
            Try it →
          </Link>
        </div>
      </div>
    </footer>
  );
}
