/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/squash-analysis",
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_BASE_PATH: "/squash-analysis",
  },
};

export default nextConfig;
