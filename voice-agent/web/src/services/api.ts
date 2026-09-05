const API_BASE_URL = '/api'

export interface GetConfigResponse {
  app_id: string
  token: string
  uid: string
  channel_name: string
  agent_uid: string
}

export async function getConfig(options?: { channel?: string; uid?: string | number }): Promise<GetConfigResponse> {
  const params = new URLSearchParams()
  if (options?.channel !== undefined && options.channel !== '') {
    params.set('channel', options.channel)
  }
  if (options?.uid !== undefined && options.uid !== '') {
    params.set('uid', String(options.uid))
  }

  const query = params.toString()
  const response = await fetch(`${API_BASE_URL}/get_config${query ? `?${query}` : ''}`, {
    method: 'GET',
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  const result = await response.json()
  if (result.code !== 0 || !result.data) {
    throw new Error(result.msg || 'Failed to get configuration')
  }
  return result.data
}

export interface StartAgentResponse {
  agentId: string
  // The uid actually running the agent in this channel -- may differ from
  // the rtcUid this call passed in, if another participant already started
  // it (see server/src/agent.py's per-channel dedup).
  agentUid: string
}

export async function startAgent(
  channelName: string,
  rtcUid: number,
  userUid: number,
): Promise<StartAgentResponse> {
  const payload = { channelName, rtcUid, userUid }

  const response = await fetch(`${API_BASE_URL}/startAgent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  const result = await response.json()
  if (result.code !== 0 || !result.data?.agent_id) {
    throw new Error(result.msg || 'Failed to start agent')
  }
  return { agentId: result.data.agent_id, agentUid: String(result.data.agent_uid) }
}

export async function stopAgent(agentId: string): Promise<void> {
  if (!agentId) return

  const response = await fetch(`${API_BASE_URL}/stopAgent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agentId }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
}

export type GithubCommit = {
  sha: string
  message: string
  author: string
  date: string
}

export type LogEntry = {
  id: number
  severity: string
  message: string
  timestamp: string
  source_id: string
}

// Pushed to everyone on the call (see broadcast_shared_screen in the
// backend) when someone asks to see GitHub commits or server logs -- the
// discriminant `type` field picks which of the two item arrays is present.
export type SharedScreen =
  | { type: 'github_commits'; title: string; commits: GithubCommit[]; timestamp: string }
  | { type: 'logs'; title: string; logs: LogEntry[]; timestamp: string }

export async function setName(
  channelName: string,
  uid: string,
  name: string,
): Promise<{ recap: string | null; sharedScreens: SharedScreen[] }> {
  const response = await fetch(`${API_BASE_URL}/setName`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ channelName, uid, name }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  // Present only for a genuine late join -- a private catch-up (recap text
  // + past shared screens) for this caller alone, never broadcast to the
  // shared meet chat.
  const result = await response.json()
  return {
    recap: result.data?.recap ?? null,
    sharedScreens: result.data?.sharedScreens ?? [],
  }
}

export async function removeName(channelName: string, uid: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/removeName`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ channelName, uid }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
}

export interface MeetChatMessage {
  uid: string
  name: string
  text: string
  timestamp: number
}

export async function sendMeetChatMessage(
  channelName: string,
  uid: string,
  name: string,
  text: string,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/sendChatMessage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ channelName, uid, name, text }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
}

export async function getMeetChatMessages(channelName: string): Promise<MeetChatMessage[]> {
  const response = await fetch(
    `${API_BASE_URL}/chatMessages?channel=${encodeURIComponent(channelName)}`,
  )

  if (!response.ok) {
    return []
  }

  const result = await response.json()
  return result.data ?? []
}

export async function getNames(channelName: string): Promise<Record<string, string>> {
  const response = await fetch(
    `${API_BASE_URL}/getNames?channel=${encodeURIComponent(channelName)}`,
  )

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  const result = await response.json()
  return result.data ?? {}
}

// Below: the main iThink backend (iCall), not the Agora agent server --
// routed through /api/recordUtterance and /api/callStatus (see
// next.config.ts's ITHINK_BACKEND_URL rewrites), a separate service from
// everything above this line.

export async function recordUtterance(
  channelName: string,
  speakerUid: string,
  speakerName: string | undefined,
  text: string,
  turnIndex: number,
  timestamp: number,
): Promise<void> {
  const response = await fetch(`/api/recordUtterance/${encodeURIComponent(channelName)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      speaker_uid: speakerUid,
      speaker_name: speakerName,
      text,
      turn_index: turnIndex,
      timestamp: new Date(timestamp).toISOString(),
    }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
}

export async function markCallCompleted(channelName: string): Promise<void> {
  const response = await fetch(`/api/callStatus/${encodeURIComponent(channelName)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status: 'completed' }),
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
}

export interface ChatNote {
  text: string
  timestamp: string
}

export async function getChatNotes(channelName: string): Promise<ChatNote[]> {
  const response = await fetch(`/api/chatNotes/${encodeURIComponent(channelName)}`)

  if (!response.ok) {
    return []
  }

  const result = await response.json()
  return result.data ?? []
}

export interface LiveRecap {
  recap: string | null
  sharedScreens: unknown[]
}

// Backs the live timeline panel -- the brief's "a continuously updated
// incident timeline" was only ever visible server-side before this
// (queryable via curl, never shown to anyone on the actual call). Polled,
// not pushed -- this is a plain-text recap of already-extracted state
// (see iCall_utils.format_live_recap), cheap enough that a poll interval
// is simpler than wiring another RTM message type for it.
export async function getRecap(channelName: string): Promise<LiveRecap> {
  const response = await fetch(`/api/recap/${encodeURIComponent(channelName)}`)

  if (!response.ok) {
    return { recap: null, sharedScreens: [] }
  }

  const result = await response.json()
  return {
    recap: result.data?.recap ?? null,
    sharedScreens: result.data?.shared_screens ?? [],
  }
}
