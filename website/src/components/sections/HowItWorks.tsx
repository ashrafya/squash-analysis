import { Upload, Cpu, Download } from "lucide-react";

const steps = [
  { icon: Upload,   num: "01", title: "Upload your MP4",        bg: "bg-lime",   text: "text-ink",   desc: "Drop your match recording. Fixed camera behind the back wall. 360p–1080p supported." },
  { icon: Cpu,      num: "02", title: "AI analyses the match",  bg: "bg-coral",  text: "text-chalk", desc: "YOLOv8 tracks both players and the ball. Kalman smoothing. Rallies segmented automatically." },
  { icon: Download, num: "03", title: "Download your report",   bg: "bg-purple", text: "text-chalk", desc: "Heatmaps, zone breakdowns, speed charts, rally timelines, and a PDF — ready in minutes." },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="bg-ink py-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="mb-12">
          <div className="inline-block border-2 border-lime px-3 py-1 mb-4">
            <span className="section-label text-lime">How It Works</span>
          </div>
          <h2 className="font-heading font-extrabold text-4xl sm:text-5xl text-chalk leading-tight">
            Three steps to match intelligence.
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 border-2 border-lime">
          {steps.map((step, i) => {
            const Icon = step.icon;
            return (
              <div key={step.num}
                className={`p-8 ${i < steps.length - 1 ? "border-b-2 md:border-b-0 md:border-r-2 border-lime" : ""}`}>
                <div className={`inline-flex items-center justify-center w-14 h-14 border-2 border-lime ${step.bg} mb-6`}>
                  <Icon className={`w-6 h-6 ${step.text}`} strokeWidth={2}/>
                </div>
                <p className="section-label text-lime/50 mb-1">{step.num}</p>
                <h3 className="font-heading font-bold text-xl text-chalk mb-3">{step.title}</h3>
                <p className="font-body text-sm text-muted leading-relaxed">{step.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
