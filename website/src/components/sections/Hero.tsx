import Link from "next/link";
import Button from "@/components/ui/Button";
import { ArrowRight, Play } from "lucide-react";

export default function Hero() {
  return (
    <section className="bg-chalk border-b-2 border-ink pt-16 overflow-hidden">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">

        {/* ── Top label strip ── */}
        <div className="flex items-center gap-0 mt-6 mb-10 border-2 border-ink w-fit shadow-block">
          <span className="bg-lime px-4 py-1.5 section-label text-ink border-r-2 border-ink">Open-source</span>
          <span className="bg-coral px-4 py-1.5 section-label text-chalk border-r-2 border-ink">Self-hosted</span>
          <span className="bg-purple px-4 py-1.5 section-label text-chalk">Free forever</span>
        </div>

        {/* ── Main hero grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-0 border-2 border-ink shadow-[8px_8px_0px_#0A0A0A]">

          {/* Left — lime block with headline */}
          <div className="bg-lime border-r-2 border-ink p-10 flex flex-col justify-between min-h-[420px]">
            <h1 className="font-heading font-extrabold text-5xl sm:text-6xl xl:text-7xl leading-[0.92] text-ink tracking-tight">
              ANALYSE<br/>
              YOUR<br/>
              SQUASH<br/>
              MATCH.
            </h1>

            <div className="mt-8">
              <p className="font-body text-base text-ink/70 max-w-xs leading-relaxed mb-6">
                Upload a recording. Get player heatmaps, ball trajectories, rally
                segmentation, and a full PDF report — automated, on your machine.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link href="/analyze">
                  <Button size="lg" variant="dark">
                    Analyse a Match <ArrowRight className="w-4 h-4"/>
                  </Button>
                </Link>
                <a href="#demo">
                  <Button size="lg" variant="secondary">
                    <Play className="w-4 h-4"/> Demo
                  </Button>
                </a>
              </div>
            </div>
          </div>

          {/* Right — dark block with mockup */}
          <div className="bg-ink p-8 flex flex-col gap-4">

            {/* Price callout */}
            <div className="flex gap-3 flex-wrap">
              {[
                { label: "Rally Vision", price: "$99/match", strike: true },
                { label: "SmartSquash",  price: "$50/mo",    strike: true },
                { label: "Ours",         price: "$0",        strike: false, accent: "bg-lime text-ink" },
              ].map((p) => (
                <div key={p.label} className={`border-2 border-white/10 px-3 py-2 ${p.accent ?? "bg-white/5"}`}>
                  <p className="section-label text-white/40">{p.label}</p>
                  <p className={`font-heading font-bold text-lg mt-0.5 ${p.strike ? "line-through text-white/25" : "text-ink"}`}>
                    {p.price}
                  </p>
                </div>
              ))}
            </div>

            {/* Heatmap mockup */}
            <div className="border-2 border-white/10 flex-1">
              <div className="bg-white/5 px-3 py-2 border-b border-white/10 flex items-center gap-2">
                <div className="w-2 h-2 bg-lime animate-pulse"/>
                <span className="section-label text-white/40">heatmap_player1.png</span>
              </div>
              <div className="relative p-3 h-36">
                <div className="absolute top-1/3 left-2/5 w-16 h-12 rounded-full bg-coral/40 blur-2xl"/>
                <div className="absolute top-1/2 left-1/4 w-12 h-8 rounded-full bg-lime/30 blur-xl"/>
                <div className="absolute top-1/4 left-3/5 w-10 h-8 rounded-full bg-purple/35 blur-xl"/>
                <div className="absolute inset-x-4 top-1/2 h-px bg-white/10"/>
                <div className="absolute bottom-2 right-3 section-label text-white/20">Player 1 Position</div>
              </div>
            </div>

            {/* Trajectory strip */}
            <div className="border-2 border-coral bg-white/5">
              <div className="bg-coral px-3 py-2 border-b-2 border-ink flex justify-between items-center">
                <span className="section-label text-chalk">Ball Trajectory</span>
                <span className="font-heading font-bold text-xs text-chalk">5 rallies</span>
              </div>
              <div className="h-12 px-3 flex items-center">
                <svg viewBox="0 0 300 30" className="w-full">
                  <path d="M5,22 Q40,4 80,20 Q120,2 160,18 Q200,4 240,14 Q265,8 295,12"
                    fill="none" stroke="#C8FF00" strokeWidth="2" strokeDasharray="4,3"/>
                  <path d="M5,12 Q50,24 90,8 Q130,22 170,10 Q220,20 290,6"
                    fill="none" stroke="#FF3D5A" strokeWidth="2" strokeDasharray="4,3"/>
                  {[80,160,240].map(x=><circle key={x} cx={x} cy={20} r="3" fill="#C8FF00"/>)}
                  {[90,170].map(x=><circle key={x} cx={x} cy={8} r="3" fill="#FF3D5A"/>)}
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* ── Bottom stat ribbon ── */}
        <div className="grid grid-cols-4 border-x-2 border-b-2 border-ink">
          {[
            { v: "5K+",  l: "Frames/match" },
            { v: "98%",  l: "Tracking accuracy" },
            { v: "5",    l: "Pipeline steps" },
            { v: "Free", l: "Always" },
          ].map((s, i) => {
            const bgs = ["bg-lime text-ink","bg-coral text-chalk","bg-purple text-chalk","bg-cyan text-ink"];
            return (
              <div key={s.v} className={`${bgs[i]} px-4 py-4 border-r-2 border-ink last:border-r-0 text-center`}>
                <p className="font-heading font-extrabold text-2xl leading-none">{s.v}</p>
                <p className="section-label mt-1 opacity-70">{s.l}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
