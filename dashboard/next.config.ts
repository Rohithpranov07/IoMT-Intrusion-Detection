import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: this dashboard reads from a compiled-in dataset, never a server.
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
