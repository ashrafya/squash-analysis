import clsx from "clsx";

type Accent = "none" | "lime" | "coral" | "purple" | "cyan";

const shadows: Record<Accent, string> = {
  none:   "shadow-block hover:shadow-[6px_6px_0px_#0A0A0A]",
  lime:   "shadow-block-lime hover:shadow-[6px_6px_0px_#C8FF00]",
  coral:  "shadow-block-coral hover:shadow-[6px_6px_0px_#FF3D5A]",
  purple: "shadow-block-purple hover:shadow-[6px_6px_0px_#7C3AED]",
  cyan:   "shadow-block-cyan hover:shadow-[6px_6px_0px_#00D4FF]",
};

const borders: Record<Accent, string> = {
  none:   "border-ink",
  lime:   "border-lime",
  coral:  "border-coral",
  purple: "border-purple",
  cyan:   "border-cyan",
};

interface CardProps {
  children: React.ReactNode;
  className?: string;
  dark?: boolean;
  accent?: Accent;
}

export default function Card({ children, className, dark = false, accent = "none" }: CardProps) {
  return (
    <div className={clsx(
      "p-6 border-2 transition-all duration-150 hover:-translate-x-0.5 hover:-translate-y-0.5",
      dark ? "bg-surface-dark text-foreground" : "bg-chalk text-ink",
      borders[dark ? "lime" : accent],
      shadows[dark ? "lime" : accent],
      className
    )}>
      {children}
    </div>
  );
}
