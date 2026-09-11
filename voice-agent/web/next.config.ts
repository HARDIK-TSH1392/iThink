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
  // '*' alone doesn't match here (confirmed live: still blocked) --
  // Next.js's allowedDevOrigins wants a hostname pattern, not a bare
  // wildcard. cloudflared quick tunnels get a new random subdomain every
  // restart, so a specific hostname broke every time this was a literal
  // subdomain -- '*.trycloudflare.com' covers any of them without needing
  // to update it per-restart (the previous value here was a stale,
  // one-off subdomain, not actually a wildcard, despite this comment
  // already describing the wildcard as the intent).
  allowedDevOrigins: ['172.25.231.35', '*.trycloudflare.com'],

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
        { source: '/api/removeName', destination: `${backendUrl}/removeName` },
        { source: '/api/sendChatMessage', destination: `${backendUrl}/sendChatMessage` },
        { source: '/api/chatMessages', destination: `${backendUrl}/chatMessages` },
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
        {
          source: '/api/recap/:channel',
          destination: `${ithinkUrl}/icall/channel/:channel/recap`,
        },
        {
          source: '/api/languageStatus/:channel',
          destination: `${ithinkUrl}/icall/channel/:channel/language-status`,
        },
      )
    }

    return rewrites
  },
}

export default nextConfig
