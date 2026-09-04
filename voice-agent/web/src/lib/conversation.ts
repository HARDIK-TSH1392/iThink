import {
  type AgentState,
  type AgentTranscription,
  TurnStatus,
  type TranscriptHelperItem,
  type UserTranscription,
} from 'agora-agent-client-toolkit'
import {
  type AgentVisualizerState,
  EMessageStatus,
  type IMessageListItem,
} from 'agora-agent-uikit'

// First letter of a person's display name, uppercased -- shared avatar
// content across the participant grid, transcript, and chat panel. Falls
// back to "?" for an empty/unnamed string rather than rendering nothing.
export function getInitial(name: string): string {
  const trimmed = name.trim()
  return trimmed ? trimmed[0].toUpperCase() : '?'
}

export function normalizeTranscriptSpacing(text: string): string {
  return text
    .replace(/([.!?])([A-Za-z])/g, '$1 $2')
    .replace(/,([A-Za-z])/g, ', $1')
    .replace(/\s{2,}/g, ' ')
    .trim()
}

export function normalizeTimestampMs(timestamp: number): number {
  return timestamp > 1e12 ? timestamp : timestamp * 1000
}

export function mapAgentVisualizerState(
  agentState: AgentState | null,
  isAgentConnected: boolean,
  connectionState: string,
): AgentVisualizerState {
  if (connectionState === 'DISCONNECTED' || connectionState === 'DISCONNECTING') {
    return 'disconnected'
  }

  if (connectionState === 'CONNECTING' || connectionState === 'RECONNECTING') {
    return 'joining'
  }

  if (!isAgentConnected) {
    return 'not-joined'
  }

  switch (agentState) {
    case 'listening':
      return 'listening'
    case 'thinking':
      return 'analyzing'
    case 'speaking':
      return 'talking'
    case 'idle':
    case 'silent':
    default:
      return 'ambient'
  }
}

function toMessageListItem(
  item: TranscriptHelperItem<Partial<UserTranscription | AgentTranscription>>,
): IMessageListItem {
  return {
    turn_id: item.turn_id,
    uid: Number(item.uid) || 0,
    text: typeof item.text === 'string' ? item.text : '',
    status: item.status as unknown as IMessageListItem['status'],
    createdAt:
      typeof item._time === 'number'
        ? normalizeTimestampMs(item._time)
        : undefined,
  }
}

export function normalizeTranscript(
  transcript: TranscriptHelperItem<Partial<UserTranscription | AgentTranscription>>[],
  localUid: string,
) {
  return transcript.map((item) => {
    // agora-agent-client-toolkit hardcodes uid to "0" for every human
    // speaker (it assumes a single-remote-user call), but the underlying
    // STT payload still carries the real Agora RTC uid in metadata.user_id
    // -- use that when present so multiple humans get attributed correctly
    // instead of every speaker collapsing onto the local viewer.
    const realUid = item.metadata?.user_id
    const nextUid =
      realUid && realUid !== '0' ? realUid : item.uid === '0' ? localUid : item.uid
    const nextText =
      typeof item.text === 'string' ? normalizeTranscriptSpacing(item.text) : item.text

    return { ...item, uid: nextUid, text: nextText }
  })
}

export function getMessageList(
  transcript: TranscriptHelperItem<Partial<UserTranscription | AgentTranscription>>[],
) {
  return transcript
    .filter((item) => item.status !== TurnStatus.IN_PROGRESS)
    .map(toMessageListItem)
}

// TurnStatus has three values, not two: IN_PROGRESS, END, and INTERRUPTED
// (agora-agent-client-toolkit's TranscriptionBase). getMessageList only
// filters out IN_PROGRESS, so INTERRUPTED turns -- a turn the VAD cut off
// before the speaker actually finished -- still reach the caller. Once the
// speaker continues, that continuation lands in a *new* turn_id, so an
// INTERRUPTED turn and the END turn that follows it can carry overlapping
// or duplicate-looking text for what was really one sentence. Only an END
// turn is genuinely finished and safe to persist as a distinct utterance.
export function isFinishedTurn(item: IMessageListItem): boolean {
  return item.status === EMessageStatus.END
}

export function getCurrentInProgressMessage(
  transcript: TranscriptHelperItem<Partial<UserTranscription | AgentTranscription>>[],
) {
  const item = transcript.find((entry) => entry.status === TurnStatus.IN_PROGRESS)
  return item ? toMessageListItem(item) : null
}
