"use client";
import { Loader2 } from "lucide-react";
import clsx from "clsx";

type Variant = "lime" | "coral" | "dark" | "secondary" | "ghost";
type Size    = "sm" | "md" | "lg";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?:    Size;
  loading?: boolean;
  children: React.ReactNode;
}

const variantClasses: Record<Variant, string> = {
  lime:      "bg-lime text-ink border-2 border-ink shadow-block hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[6px_6px_0px_#0A0A0A]",
  coral:     "bg-coral text-chalk border-2 border-ink shadow-block hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[6px_6px_0px_#0A0A0A]",
  dark:      "bg-ink text-chalk border-2 border-ink shadow-block-lime hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[6px_6px_0px_#C8FF00]",
  secondary: "bg-chalk text-ink border-2 border-ink shadow-block hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[6px_6px_0px_#0A0A0A]",
  ghost:     "bg-transparent text-ink border-2 border-transparent hover:border-ink",
};

const sizeClasses: Record<Size, string> = {
  sm: "px-4 py-2 text-sm",
  md: "px-5 py-2.5 text-base",
  lg: "px-8 py-4 text-base",
};

export default function Button({
  variant = "lime",
  size    = "md",
  loading = false,
  disabled,
  children,
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-btn font-heading font-semibold",
        "transition-all duration-150 cursor-pointer",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lime focus-visible:ring-offset-2",
        "disabled:opacity-40 disabled:cursor-not-allowed disabled:transform-none",
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
      {...props}
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin"/>}
      {children}
    </button>
  );
}
