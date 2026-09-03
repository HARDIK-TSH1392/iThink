import path from 'node:path'
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Enable React strict mode
  reactStrictMode: true,
  turbopack: {
    root: path.resolve(__dirname, '..'),
  },
  // Lets a teammate on the same LAN load the dev server (and its HMR
  // websocket) from this machine's local IP instead of localhost. This IP
  // is machine- and network-specific (DHCP can reassign it) -- update it
  // locally to whatever `ipconfig`/`hostname -I` reports for your own
  // machine's LAN adapter rather than relying on this committed value.
  allowedDevOrigins: ['192.168.1.8'],

  // Optimize images
  images: {
    unoptimized: true,
  },

  async rewrites() {
    const backendUrl = process.env.AGENT_BACKEND_URL?.replace(/\/$/, '')
    if (!backendUrl) {
      return []
    }

    return [
      {
        source: '/api/get_config',
        destination: `${backendUrl}/get_config`,
      },
      {
        source: '/api/startAgent',
        destination: `${backendUrl}/startAgent`,
      },
      {
        source: '/api/stopAgent',
        destination: `${backendUrl}/stopAgent`,
      },
      {
        source: '/api/setName',
        destination: `${backendUrl}/setName`,
      },
      {
        source: '/api/getNames',
        destination: `${backendUrl}/getNames`,
      },
    ]
  },
}

export default nextConfig
