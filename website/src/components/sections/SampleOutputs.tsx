"use client";
import { useState } from "react";
import Image from "next/image";
import clsx from "clsx";

const tabs = [
  { id: "heatmap",    label: "Heatmaps",       active: "bg-lime text-ink border-lime",    images: [{ src: "/demo/heatmap_player1.png", alt: "Player 1 heatmap" }, { src: "/demo/heatmap_player2.png", alt: "Player 2 heatmap" }] },
  { id: "zones",      label: "Zone Breakdown", active: "bg-coral text-chalk border-coral", images: [{ src: "/demo/zone_breakdown.png",  alt: "Zone breakdown" }] },
  { id: "trajectory", label: "Trajectory",     active: "bg-purple text-chalk border-purple",images: [{ src: "/demo/combined_court.png", alt: "Combined court" }] },
  { id: "charts",     label: "Speed & Stats",  active: "bg-cyan text-ink border-cyan",    images: [{ src: "/demo/timeseries.png",      alt: "Speed" }, { src: "/demo/rally_timeline.png", alt: "Rallies" }] },
];

export default function SampleOutputs() {
  const [active, setActive] = useState("heatmap");
  const tab = tabs.find(t => t.id === active)!;

  return (
    <section id="demo" className="bg-chalk border-b-2 border-ink py-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="mb-12">
          <div className="inline-block bg-purple border-2 border-ink px-3 py-1 mb-4 shadow-block">
            <span className="section-label text-chalk">Sample Outputs</span>
          </div>
          <h2 className="font-heading font-extrabold text-4xl sm:text-5xl text-ink">
            See what you get.
          </h2>
        </div>

        {/* Tab bar */}
        <div className="flex flex-wrap border-2 border-ink w-fit mb-0">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setActive(t.id)}
              className={clsx(
                "px-5 py-3 font-heading font-semibold text-sm border-r-2 border-ink last:border-r-0 cursor-pointer transition-colors duration-100",
                active === t.id ? t.active : "bg-chalk text-ink/50 hover:bg-ink/5"
              )}>
              {t.label}
            </button>
          ))}
        </div>

        {/* Panel */}
        <div className="border-2 border-t-0 border-ink shadow-[4px_4px_0px_#0A0A0A]">
          <div className={clsx("grid", tab.images.length > 1 ? "grid-cols-1 sm:grid-cols-2 divide-x-2 divide-ink" : "grid-cols-1")}>
            {tab.images.map((img, i) => (
              <div key={img.src} className="relative aspect-video bg-ink/4">
                <Image src={img.src} alt={img.alt} fill className="object-contain p-4"
                  onError={e => { (e.currentTarget as HTMLImageElement).style.opacity = "0"; }}/>
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <p className="font-heading text-xs text-ink/15">{img.alt}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
