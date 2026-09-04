import type { RTMClient } from 'agora-rtm';
import type { SharedScreen } from '@/services/api';

/** Session bootstrap from GET /api/get_config (channel + tokens + agent identity). */
export interface AgoraTokenData {
  token: string;
  uid: string;
  channel: string;
  agentId?: string;
  appId?: string; // `app_id` returned by backend
  agentUid?: string; // `NEXT_PUBLIC_AGENT_UID`
}

export interface AgoraRenewalTokens {
  rtcToken: string;
  rtmToken: string;
}

export interface ConversationComponentProps {
  agoraData: AgoraTokenData;
  rtmClient: RTMClient;
  localName: string;
  /** Private "what did I miss" recap for this viewer alone, when they joined an in-progress call. */
  lateJoinRecap: string | null;
  /** Shared screens (GitHub commits / server logs) already shown before this viewer joined. */
  initialSharedScreens: SharedScreen[];
  onTokenWillExpire: (uid: string) => Promise<AgoraRenewalTokens>;
  onEndConversation: () => void;
}
