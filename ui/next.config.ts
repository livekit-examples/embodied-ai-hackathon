import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Disabled so dev-only double-invocation doesn't trigger duplicate calls
  // into stateful APIs (e.g. the agent dispatch in `/api/token`).
  reactStrictMode: false,
};

export default nextConfig;
