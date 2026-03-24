import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import clsx from "clsx";

export type StepStatus = "pending" | "running" | "done" | "error";
export interface PipelineStep { id: string; label: string; detail: string; status: StepStatus; }

export default function ProcessingStatus({ steps, progress }: { steps: PipelineStep[]; progress?: number }) {
  return (
    <div className="space-y-5">
      {progress !== undefined && (
        <div>
          <div className="flex justify-between text-xs font-heading text-muted mb-2">
            <span className="section-label">Processing</span>
            <span className="text-lime font-bold">{Math.round(progress)}%</span>
          </div>
          <div className="h-2 border-2 border-lime/40 bg-surface-dark">
            <div className="h-full bg-lime transition-all duration-500" style={{ width: `${progress}%` }}/>
          </div>
        </div>
      )}
      <div className="border-2 border-white/10">
        {steps.map((step, i) => (
          <div key={step.id} className={clsx(
            "flex items-start gap-4 px-5 py-4 border-b border-white/8 last:border-b-0 transition-all duration-150",
            step.status === "running" && "bg-lime/8 border-l-2 border-lime",
            step.status === "done"    && "opacity-50",
            step.status === "pending" && "opacity-30",
            step.status === "error"   && "bg-coral/8 border-l-2 border-coral"
          )}>
            <div className="mt-0.5 flex-shrink-0">
              {step.status === "done"    && <CheckCircle2 className="w-4 h-4 text-success"/>}
              {step.status === "running" && <Loader2 className="w-4 h-4 text-lime animate-spin"/>}
              {step.status === "pending" && <Circle className="w-4 h-4 text-muted"/>}
              {step.status === "error"   && <Circle className="w-4 h-4 text-coral"/>}
            </div>
            <div>
              <p className={clsx("font-heading font-semibold text-sm", step.status === "running" ? "text-lime" : "text-foreground")}>
                {i + 1}. {step.label}
              </p>
              <p className="font-body text-xs text-muted mt-0.5">{step.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
