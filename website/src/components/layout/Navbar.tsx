"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import Button from "@/components/ui/Button";
import { Menu, X } from "lucide-react";
import clsx from "clsx";

const links = [
  { href: "#features",     label: "Features" },
  { href: "#how-it-works", label: "How It Works" },
  { href: "#demo",         label: "Demo" },
  { href: "https://github.com/ashrafya/squash-analysis", label: "GitHub" },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", h, { passive: true });
    return () => window.removeEventListener("scroll", h);
  }, []);

  return (
    <header className={clsx(
      "fixed top-0 left-0 right-0 z-50 transition-all duration-150",
      scrolled ? "bg-chalk border-b-2 border-ink" : "bg-transparent border-b-2 border-transparent"
    )}>
      {/* Coloured top strip */}
      <div className="h-1 w-full flex">
        <div className="flex-1 bg-lime"/>
        <div className="flex-1 bg-coral"/>
        <div className="flex-1 bg-purple"/>
        <div className="flex-1 bg-cyan"/>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 bg-ink border-2 border-ink flex items-center justify-center group-hover:bg-lime transition-colors duration-150">
            <svg viewBox="0 0 20 20" fill="none" className="w-4 h-4" stroke="#FAFAFA" strokeWidth={2.5}>
              <circle cx="10" cy="10" r="3"/>
              <path d="M10 2v3M10 15v3M3.5 3.5l2 2M14.5 14.5l2 2M2 10h3M15 10h3M3.5 16.5l2-2M14.5 5.5l2-2"/>
            </svg>
          </div>
          <span className="font-heading font-bold text-ink tracking-tight">SquashAnalysis</span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-7">
          {links.map((l) => (
            <a key={l.href} href={l.href}
              className="font-heading font-medium text-sm text-ink/50 hover:text-ink transition-colors cursor-pointer">
              {l.label}
            </a>
          ))}
        </nav>

        <div className="hidden md:block">
          <Link href="/analyze">
            <Button size="sm" variant="lime">Analyse a Match</Button>
          </Link>
        </div>

        <button className="md:hidden p-2 cursor-pointer" onClick={() => setOpen(!open)} aria-label="Toggle menu">
          {open ? <X className="w-5 h-5 text-ink"/> : <Menu className="w-5 h-5 text-ink"/>}
        </button>
      </div>

      {open && (
        <div className="md:hidden bg-chalk border-t-2 border-ink px-4 py-4 flex flex-col gap-4">
          {links.map((l) => (
            <a key={l.href} href={l.href} onClick={() => setOpen(false)}
              className="font-heading font-medium text-sm text-ink hover:text-coral transition-colors cursor-pointer">
              {l.label}
            </a>
          ))}
          <Link href="/analyze"><Button size="sm" variant="lime" className="w-full">Analyse a Match</Button></Link>
        </div>
      )}
    </header>
  );
}
