import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Squash Analysis — Analyse your game like a pro",
  description:
    "Free, self-hosted squash analytics. Player heatmaps, ball tracking, rally segmentation, and PDF reports — automated from a single MP4.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-body">{children}</body>
    </html>
  );
}
