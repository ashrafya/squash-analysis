import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        lime:    "#C8FF00",
        coral:   "#FF3D5A",
        purple:  "#7C3AED",
        cyan:    "#00D4FF",
        ink:     "#0A0A0A",
        chalk:   "#FAFAFA",
        "bg-dark":      "#0A0A0A",
        "surface-dark": "#141414",
        foreground:     "#F0F0F0",
        muted:          "#737373",
        success:        "#22C55E",
      },
      fontFamily: {
        heading: ["var(--font-space)", "sans-serif"],
        body:    ["var(--font-inter)",  "sans-serif"],
      },
      borderRadius: {
        btn:   "4px",
        card:  "0px",
        modal: "4px",
      },
      boxShadow: {
        block:          "4px 4px 0px #0A0A0A",
        "block-lime":   "4px 4px 0px #C8FF00",
        "block-coral":  "4px 4px 0px #FF3D5A",
        "block-purple": "4px 4px 0px #7C3AED",
        "block-cyan":   "4px 4px 0px #00D4FF",
      },
      animation: {
        "fade-up": "fadeUp 0.4s ease-out forwards",
        shimmer:   "shimmer 1.6s infinite",
        marquee:   "marquee 20s linear infinite",
      },
      keyframes: {
        fadeUp: {
          "0%":   { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition:  "200% 0" },
        },
        marquee: {
          "0%":   { transform: "translateX(0%)" },
          "100%": { transform: "translateX(-50%)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
