import clsx from "clsx";

type BadgeVariant = "default" | "success" | "warning" | "info";

const variants: Record<BadgeVariant, string> = {
  default: "bg-ink text-chalk border-2 border-ink",
  success: "bg-lime text-ink border-2 border-ink",
  warning: "bg-coral text-chalk border-2 border-ink",
  info:    "bg-purple text-chalk border-2 border-ink",
};

export default function Badge({ children, variant = "default" }: { children: React.ReactNode; variant?: BadgeVariant }) {
  return (
    <span className={clsx("inline-flex items-center px-3 py-1 text-xs font-heading font-semibold tracking-wide", variants[variant])}>
      {children}
    </span>
  );
}
