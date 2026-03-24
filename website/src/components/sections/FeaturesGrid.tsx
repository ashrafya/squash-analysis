import { Activity, MapPin, Target, BarChart2, FileText, Zap } from "lucide-react";

const features = [
  { icon: MapPin,    title: "Player Heatmaps",   iconBg: "bg-lime",   iconText: "text-ink",   shadow: "shadow-block",          border: "border-ink",   desc: "Gaussian heatmaps showing where each player spent time. T-zone dominance and court coverage." },
  { icon: Target,    title: "Ball Tracking",      iconBg: "bg-coral",  iconText: "text-chalk", shadow: "shadow-block-coral",    border: "border-coral", desc: "YOLOv8 + MOG2 motion detection + Kalman smoothing for robust ball trajectories across every rally." },
  { icon: Activity,  title: "Rally Segmentation", iconBg: "bg-purple", iconText: "text-chalk", shadow: "shadow-block-purple",   border: "border-purple",desc: "Automatic rally detection from ball-lost gaps. Per-rally stats: duration, shots, distance covered." },
  { icon: BarChart2, title: "Zone Breakdown",     iconBg: "bg-cyan",   iconText: "text-ink",   shadow: "shadow-block-cyan",     border: "border-cyan",  desc: "9-zone court grid analysis. See which zones you dominate and which you neglect." },
  { icon: Zap,       title: "Speed & Distance",   iconBg: "bg-lime",   iconText: "text-ink",   shadow: "shadow-block",          border: "border-ink",   desc: "Per-player distance, average speed, peak speed, and time-series speed charts." },
  { icon: FileText,  title: "PDF Report",         iconBg: "bg-coral",  iconText: "text-chalk", shadow: "shadow-block-coral",    border: "border-coral", desc: "All charts compiled into a shareable, printable PDF. Show your coach or track progress." },
];

export default function FeaturesGrid() {
  return (
    <section id="features" className="bg-chalk border-b-2 border-ink py-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="mb-14">
          <div className="inline-block bg-ink border-2 border-ink px-3 py-1 mb-4">
            <span className="section-label text-lime">Features</span>
          </div>
          <h2 className="font-heading font-extrabold text-4xl sm:text-5xl text-ink">
            Everything you need to improve.
          </h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f) => {
            const Icon = f.icon;
            return (
              <div key={f.title}
                className={`bg-chalk p-6 border-2 transition-all duration-150 hover:-translate-x-0.5 hover:-translate-y-0.5 ${f.border} ${f.shadow}`}>
                <div className={`w-10 h-10 border-2 border-ink flex items-center justify-center mb-4 ${f.iconBg}`}>
                  <Icon className={`w-5 h-5 ${f.iconText}`} strokeWidth={2}/>
                </div>
                <h3 className="font-heading font-semibold text-base text-ink mb-2">{f.title}</h3>
                <p className="font-body text-sm text-ink/55 leading-relaxed">{f.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
