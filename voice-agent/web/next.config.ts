import path from 'node:path'
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Enable React strict mode
  reactStrictMode: true,
  turbopack: {
    root: path.resolve(__dirname, '..'),
  },
  // Lets a teammate on the same LAN load the dev server (and its HMR
  // websocket) from this machine's local IP instead of localhost.
  allowedDevOrigins: ['192.168.1.8'],

  // Optimize images
  images: {
    unoptimized: true,
  },

  async rewrites() {
    const backendUrl = process.env.AGENT_BACKEND_URL?.replace(/\/$/, '')
    const ithinkUrl = process.env.ITHINK_BACKEND_URL?.replace(/\/$/, '')

    const rewrites = []

    if (backendUrl) {
      rewrites.push(
        { source: '/api/get_config', destination: `${backendUrl}/get_config` },
        { source: '/api/startAgent', destination: `${backendUrl}/startAgent` },
        { source: '/api/stopAgent', destination: `${backendUrl}/stopAgent` },
        { source: '/api/setName', destination: `${backendUrl}/setName` },
        { source: '/api/getNames', destination: `${backendUrl}/getNames` },
      )
    }

    if (ithinkUrl) {
      rewrites.push(
        {
          source: '/api/recordUtterance/:channel',
          destination: `${ithinkUrl}/icall/channel/:channel/utterances`,
        },
        {
          source: '/api/callStatus/:channel',
          destination: `${ithinkUrl}/icall/channel/:channel/status`,
        },
        {
          source: '/api/chatNotes/:channel',
          destination: `${ithinkUrl}/icall/channel/:channel/chat-notes`,
        },
      )
    }

    return rewrites
  },
}

export default nextConfig
