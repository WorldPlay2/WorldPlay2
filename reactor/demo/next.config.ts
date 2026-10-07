import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // ReactorProvider (@reactor-team/js-sdk 3.0.2) is not Strict-Mode safe: the
  // simulated unmount disposes its first Reactor, and the cleanup that follows
  // calls disconnect() on that disposed instance, which rejects with "This
  // Reactor was disposed" as an unhandled error. Dev-only; `next build` output
  // never double-invokes effects.
  reactStrictMode: false,
};

export default nextConfig;
