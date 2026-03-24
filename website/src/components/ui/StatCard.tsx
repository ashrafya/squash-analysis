import clsx from "clsx";

const colors = [
  { bg: "bg-lime",   text: "text-ink",   shadow: "shadow-block" },
  { bg: "bg-coral",  text: "text-chalk", shadow: "shadow-block" },
  { bg: "bg-purple", text: "text-chalk", shadow: "shadow-block" },
  { bg: "bg-cyan",   text: "text-ink",   shadow: "shadow-block" },
];

export default function StatCard({ value, label, sub, index = 0 }: { value: string; label: string; sub?: string; index?: number }) {
  const c = colors[index % colors.length];
  return (
    <div className={clsx("p-6 border-2 border-ink", c.bg, c.shadow)}>
      <p className={clsx("font-heading font-extrabold text-4xl leading-none", c.text)}>{value}</p>
      <p className={clsx("font-heading font-semibold text-sm mt-3", c.text)}>{label}</p>
      {sub && <p className={clsx("text-xs mt-1 opacity-60", c.text)}>{sub}</p>}
    </div>
  );
}
