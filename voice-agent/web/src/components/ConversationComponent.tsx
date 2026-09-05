"use client";

import { MicOff, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConnectionStatusPanel } from "@/components/ConnectionStatusPanel";
import {
	type ConnectionIssue,
	getConversationIssueSeverity,
} from "@/components/ConversationErrorCard";
import { LiveRecapPanel } from "@/components/LiveRecapPanel";
import { MeetChatPanel } from "@/components/MeetChatPanel";
import { MicrophoneSelector } from "@/components/MicrophoneSelector";
import { QuickstartConversationLayout } from "@/components/QuickstartConversationLayout";
import {
	type QuickstartAgentMetric,
	QuickstartPipelineMetrics,
} from "@/components/QuickstartPipelineMetrics";
import { QuickstartTranscriptPanel } from "@/components/QuickstartTranscriptPanel";
import { McpResponseTile } from "@/components/McpResponseTile";
import { DEFAULT_AGENT_UID } from "@/lib/agora";
import { playJoinChime } from "@/lib/joinChime";
import {
	type ChatNote,
	type SharedScreen,
	getChatNotes,
	getNames,
	recordUtterance,
} from "@/services/api";
import {
	getCurrentInProgressMessage,
	getInitial,
	getMessageList,
	mapAgentVisualizerState,
	normalizeTimestampMs,
	normalizeTranscript,
} from "@/lib/conversation";
import type { ConversationComponentProps } from "@/types/conversation";
import {
	type AgentState,
	type AgentTranscription,
	AgoraVoiceAI,
	AgoraVoiceAIEvents,
	MessageSalStatus,
	type TranscriptHelperItem,
	TranscriptHelperMode,
	type UserTranscription,
} from "agora-agent-client-toolkit";
import { MicButtonWithVisualizer } from "agora-agent-uikit/rtc";
import AgoraRTC, {
	RemoteUser,
	type UID,
	useClientEvent,
	useJoin,
	useLocalMicrophoneTrack,
	usePublish,
	useRTCClient,
	useRemoteUsers,
} from "agora-rtc-react";
import { setParameter } from "agora-rtc-sdk-ng/esm";

const MAX_CONNECTION_ISSUES = 6;

type RtmMessageErrorPayload = {
	object: "message.error";
	module?: string;
	code?: number;
	message?: string;
	send_ts?: number;
};

type RtmSalStatusPayload = {
	object: "message.sal_status";
	status?: string;
	timestamp?: number;
};

function isRtmMessageErrorPayload(
	value: unknown,
): value is RtmMessageErrorPayload {
	return (
		!!value &&
		typeof value === "object" &&
		(value as { object?: unknown }).object === "message.error"
	);
}

function isRtmSalStatusPayload(value: unknown): value is RtmSalStatusPayload {
	return (
		!!value &&
		typeof value === "object" &&
		(value as { object?: unknown }).object === "message.sal_status"
	);
}

// Pushed server-side via Agora's Signaling REST API (see
// broadcast_shared_screen in the backend) when someone asks iThink to show
// GitHub commits or server logs -- every participant's RTM client is
// already subscribed to the channel, so this arrives live with no
// polling. Distinct `type` discriminant keeps it from ever being confused
// with Agora's own message.error/message.sal_status payloads above.
type SharedScreenBroadcast = {
	type: "ithink_shared_screen";
	screen: SharedScreen;
};

function isSharedScreenBroadcast(value: unknown): value is SharedScreenBroadcast {
	return (
		!!value &&
		typeof value === "object" &&
		(value as { type?: unknown }).type === "ithink_shared_screen" &&
		!!(value as { screen?: unknown }).screen
	);
}

export default function ConversationComponent({
	agoraData,
	rtmClient,
	localName,
	lateJoinRecap,
	initialSharedScreens,
	onTokenWillExpire,
	onEndConversation,
}: ConversationComponentProps) {
	const client = useRTCClient();
	const remoteUsers = useRemoteUsers();
	const [isEnabled, setIsEnabled] = useState(true);
	const [isAgentConnected, setIsAgentConnected] = useState(false);
	const [isConnectionDetailsOpen, setIsConnectionDetailsOpen] = useState(false);

	const [connectionState, setConnectionState] = useState<string>("CONNECTING");
	const agentUID =
		agoraData.agentUid ??
		process.env.NEXT_PUBLIC_AGENT_UID ??
		String(DEFAULT_AGENT_UID);
	const [joinedUID, setJoinedUID] = useState<UID>(0);

	const [rawTranscript, setRawTranscript] = useState<
		TranscriptHelperItem<Partial<UserTranscription | AgentTranscription>>[]
	>([]);
	// Own accumulator, keyed by (uid, turn_id), independent of the toolkit's
	// internal chatHistory -- confirmed live (incident-38) that unsubscribe()
	// (called by resubscribeTranscript below, itself added to recover from a
	// stalled transcript feed) hard-resets that internal history to []
	// (SubRenderQueue.reset() in the installed toolkit's source, called from
	// CovSubRenderController.cleanup()). Every resubscribe -- including a
	// perfectly healthy one firing on ordinary silence, which the time-based
	// watchdog can't tell apart from a genuine stall -- was therefore wiping
	// everything already on screen, even though the backend had the full
	// transcript intact the whole time (call_utterances never lost anything,
	// only the live display did). Merging every TRANSCRIPT_UPDATED payload
	// into this ref instead of replacing rawTranscript wholesale means a
	// reset toolkit history only ever adds to what's already shown, never
	// erases it.
	const transcriptByKeyRef = useRef<
		Map<string, TranscriptHelperItem<Partial<UserTranscription | AgentTranscription>>>
	>(new Map());
	const [agentState, setAgentState] = useState<AgentState | null>(null);
	const [agentMetrics, setAgentMetrics] = useState<QuickstartAgentMetric[]>([]);
	const [connectionIssues, setConnectionIssues] = useState<ConnectionIssue[]>(
		[],
	);
	const addConnectionIssue = useCallback((issue: ConnectionIssue) => {
		setConnectionIssues((prev) => {
			const isDuplicate = prev.some(
				(x) =>
					x.agentUserId === issue.agentUserId &&
					x.code === issue.code &&
					x.message === issue.message &&
					Math.abs(x.timestamp - issue.timestamp) < 1500,
			);
			if (isDuplicate) return prev;
			return [issue, ...prev].slice(0, MAX_CONNECTION_ISSUES);
		});
	}, []);

	useEffect(() => {
		if (connectionIssues.length > 0) {
			setIsConnectionDetailsOpen(true);
		}
	}, [connectionIssues.length]);

	// uid -> display name, sourced from the backend's channel name registry
	// (see LandingPage's setName call and services/api.ts's getNames). Polled
	// rather than pushed in real time -- simpler and easier to verify than
	// RTM presence, and a few seconds' staleness on a name label is fine.
	const [participantNames, setParticipantNames] = useState<Record<string, string>>(
		() => (localName ? { [agoraData.uid]: localName } : {}),
	);

	useEffect(() => {
		let cancelled = false;

		const refresh = () => {
			getNames(agoraData.channel)
				.then((names) => {
					if (cancelled) return;
					setParticipantNames((prev) => {
						const merged = { ...prev, ...names };
						const changed = Object.keys(merged).some((k) => merged[k] !== prev[k]);
						return changed ? merged : prev;
					});
				})
				.catch((error) => {
					console.error("Failed to fetch participant names:", error);
				});
		};

		refresh();
		const interval = setInterval(refresh, 3000);
		return () => {
			cancelled = true;
			clearInterval(interval);
		};
	}, [agoraData.channel]);

	// Written notes the agent chose not to interrupt the call to say out
	// loud (StructuringUpdate.agent_chat_note) -- same polling pattern as
	// participantNames above, for the same reasons.
	const [chatNotes, setChatNotes] = useState<ChatNote[]>([]);

	useEffect(() => {
		let cancelled = false;

		const refresh = () => {
			getChatNotes(agoraData.channel)
				.then((notes) => {
					if (cancelled) return;
					setChatNotes((prev) => (notes.length !== prev.length ? notes : prev));
				})
				.catch((error) => {
					console.error("Failed to fetch chat notes:", error);
				});
		};

		refresh();
		const interval = setInterval(refresh, 3000);
		return () => {
			cancelled = true;
			clearInterval(interval);
		};
	}, [agoraData.channel]);

	// GitHub commits / server logs pushed to everyone on the call. Seeded
	// once from what a late joiner's setName response already caught them
	// up on, then appended to live via the RTM broadcast listener below --
	// never re-fetched by polling, since the whole point of the RTM push is
	// to avoid that round trip.
	const [sharedScreens, setSharedScreens] = useState<SharedScreen[]>(initialSharedScreens);

	useEffect(() => {
		const handleSharedScreenMessage = (event: { message: string | Uint8Array }) => {
			const payloadText =
				typeof event.message === "string" ? event.message : new TextDecoder().decode(event.message);

			let parsed: unknown;
			try {
				parsed = JSON.parse(payloadText);
			} catch {
				return;
			}

			if (!isSharedScreenBroadcast(parsed)) return;

			setSharedScreens((prev) => [...prev, parsed.screen]);
		};

		rtmClient.addEventListener("message", handleSharedScreenMessage);
		return () => {
			rtmClient.removeEventListener("message", handleSharedScreenMessage);
		};
	}, [rtmClient]);

	const [isReady, setIsReady] = useState(false);
	useEffect(() => {
		let cancelled = false;
		const id = setTimeout(() => {
			if (!cancelled) setIsReady(true);
		}, 0);
		return () => {
			cancelled = true;
			clearTimeout(id);
			setIsReady(false);
		};
	}, []);

	const appId = agoraData.appId ?? "";

	const { isConnected: joinSuccess } = useJoin(
		{
			appid: appId,
			channel: agoraData.channel,
			token: agoraData.token,
			uid: Number.parseInt(agoraData.uid, 10),
		},
		isReady,
	);

	// Agora's SDK defaults an unconfigured mic track to "music_standard"
	// (48kHz, 32Kbps) -- a music-fidelity target, not a voice one. Confirmed
	// live across every call this session (incident-14/16/17/19/20/21):
	// SEND_AUDIO_BITRATE_TOO_LOW fired constantly for whichever participant
	// was on a phone, which matches -- phones/cellular routinely can't
	// sustain that target consistently, while a human ear tolerates the dip
	// fine but Deepgram's STT doesn't, silently producing nothing usable.
	// "speech_standard" only needs 24Kbps (25% less) and is Agora's own
	// documented recommendation for voice calls specifically -- real
	// population here is always a laptop-plus-phone mix, so the track
	// needs to be built for the weaker device's network, not the default.
	// { ANS: true, AEC: true } is the hook's own default, but only when no
	// config object is passed at all -- passing one here to add
	// encoderConfig replaces that default outright (it's a plain JS default
	// parameter, not a merge), so both are restated explicitly to avoid
	// silently losing echo cancellation and noise suppression.
	const { localMicrophoneTrack } = useLocalMicrophoneTrack(isReady, {
		ANS: true,
		AEC: true,
		encoderConfig: "speech_standard",
	});

	useEffect(() => {
		if (!client) return;
		try {
			setParameter("ENABLE_AUDIO_PTS", true);
		} catch (error) {
			console.warn("Could not set ENABLE_AUDIO_PTS:", error);
		}
	}, [client]);

	useEffect(() => {
		if (joinSuccess && client) {
			const uid = client.uid;
			if (uid !== null && uid !== undefined) {
				setJoinedUID(uid);
			}
		}
	}, [joinSuccess, client]);

	useEffect(() => {
		if (!isReady || !joinSuccess) return;

		let cancelled = false;
		(async () => {
			try {
				const ai = await AgoraVoiceAI.init({
					rtcEngine: client,
					rtmConfig: { rtmEngine: rtmClient },
					renderMode: TranscriptHelperMode.TEXT,
					enableLog: true,
				});

				if (cancelled) {
					try {
						if (AgoraVoiceAI.getInstance() === ai) {
							ai.unsubscribe();
							ai.destroy();
						}
					} catch {}
					return;
				}

				ai.on(AgoraVoiceAIEvents.TRANSCRIPT_UPDATED, (t) => {
					for (const item of t) {
						transcriptByKeyRef.current.set(`${item.uid}-${item.turn_id}`, item);
					}
					setRawTranscript(
						Array.from(transcriptByKeyRef.current.values()).sort((a, b) => a._time - b._time),
					);
				});
				ai.on(AgoraVoiceAIEvents.AGENT_STATE_CHANGED, (_, event) =>
					setAgentState(event.state),
				);
				ai.on(AgoraVoiceAIEvents.AGENT_METRICS, (_, metrics) => {
					setAgentMetrics((prev) => [...prev, metrics].slice(-8));
				});
				ai.on(AgoraVoiceAIEvents.MESSAGE_ERROR, (agentUserId, error) => {
					addConnectionIssue({
						id: `${Date.now()}-${agentUserId}-message-error-${error.code}`,
						source: "rtm",
						agentUserId,
						code: error.code,
						message: error.message,
						timestamp: normalizeTimestampMs(error.timestamp),
					});
				});
				ai.on(
					AgoraVoiceAIEvents.MESSAGE_SAL_STATUS,
					(agentUserId, salStatus) => {
						if (
							salStatus.status === MessageSalStatus.VP_REGISTER_FAIL ||
							salStatus.status === MessageSalStatus.VP_REGISTER_DUPLICATE
						) {
							addConnectionIssue({
								id: `${Date.now()}-${agentUserId}-sal-${salStatus.status}`,
								source: "rtm",
								agentUserId,
								code: salStatus.status,
								message: `SAL status: ${salStatus.status}`,
								timestamp: normalizeTimestampMs(salStatus.timestamp),
							});
						}
					},
				);
				ai.on(AgoraVoiceAIEvents.AGENT_ERROR, (agentUserId, error) => {
					addConnectionIssue({
						id: `${Date.now()}-${agentUserId}-agent-error-${error.code}`,
						source: "agent",
						agentUserId,
						code: error.code,
						message: `${error.type}: ${error.message}`,
						timestamp: normalizeTimestampMs(error.timestamp),
					});
				});
				ai.subscribeMessage(agoraData.channel);
			} catch (error) {
				if (!cancelled) {
					console.error("[AgoraVoiceAI] init failed:", error);
				}
			}
		})();

		return () => {
			cancelled = true;
			try {
				const ai = AgoraVoiceAI.getInstance();
				if (ai) {
					ai.unsubscribe();
					ai.destroy();
				}
			} catch {}
		};
	}, [
		isReady,
		joinSuccess,
		client,
		rtmClient,
		agoraData.channel,
		addConnectionIssue,
	]);

	useEffect(() => {
		const handleRtmMessage = (event: {
			message: string | Uint8Array;
			publisher: string;
		}) => {
			const payloadText =
				typeof event.message === "string"
					? event.message
					: new TextDecoder().decode(event.message);

			let parsed: unknown;
			try {
				parsed = JSON.parse(payloadText);
			} catch {
				return;
			}

			if (isRtmMessageErrorPayload(parsed)) {
				const p = parsed;
				addConnectionIssue({
					id: `${Date.now()}-${event.publisher}-rtm-msg-error-${p.code ?? "unknown"}`,
					source: "rtm-signaling",
					agentUserId: event.publisher,
					code: p.code ?? "unknown",
					message: `${p.module ?? "unknown"}: ${p.message ?? "Unknown signaling error"}`,
					timestamp: normalizeTimestampMs(p.send_ts ?? Date.now()),
				});
				return;
			}

			if (isRtmSalStatusPayload(parsed)) {
				const p = parsed;
				if (
					p.status === "VP_REGISTER_FAIL" ||
					p.status === "VP_REGISTER_DUPLICATE"
				) {
					addConnectionIssue({
						id: `${Date.now()}-${event.publisher}-rtm-sal-${p.status}`,
						source: "rtm-signaling",
						agentUserId: event.publisher,
						code: p.status,
						message: `SAL status: ${p.status}`,
						timestamp: normalizeTimestampMs(p.timestamp ?? Date.now()),
					});
				}
			}
		};

		rtmClient.addEventListener("message", handleRtmMessage);
		return () => {
			rtmClient.removeEventListener("message", handleRtmMessage);
		};
	}, [rtmClient, addConnectionIssue]);

	const transcript = useMemo(() => {
		return normalizeTranscript(rawTranscript, String(client.uid));
	}, [rawTranscript, client.uid]);

	const messageList = useMemo(() => getMessageList(transcript), [transcript]);

	// Persists each finalized human line to the main backend (iCall), keyed
	// by the corrected per-speaker uid from normalizeTranscript -- this is
	// what post-call role inference reads (see iCall_service.
	// infer_and_store_participant_roles). Agent lines are skipped -- role
	// inference is about the humans on the call, not the agent itself.
	//
	// Does NOT filter on turn status beyond what getMessageList already
	// excludes (IN_PROGRESS) -- a prior version also skipped INTERRUPTED
	// turns on the theory that they were superseded duplicates, but real
	// call data (incident-17) showed the opposite: INTERRUPTED turns often
	// carry real speech that's never repeated in any later turn, and
	// excluding them was silently dropping content, not just duplicates.
	// The actual duplicate-post race (two independent mounts of this same
	// effect both passing the check below) is handled server-side instead,
	// by a unique constraint on (call_id, turn_index) in record_utterance --
	// a guarantee that holds regardless of what status a turn carries.
	//
	// Tracks the *text already posted* per key, not just whether the key
	// was ever posted -- confirmed live (incident-25): a turn can still get
	// revised/extended by the transcript source after its first appearance
	// (the live panel showed Bag's full sentence, but the stored row was
	// stuck at "Actually, Rahul", the first, incomplete snapshot). Posting
	// once per key permanently locked in whatever text was present at that
	// first post. Re-posting when the text for an already-seen key changes
	// lets record_utterance's upsert-on-conflict keep the stored row
	// current instead of stuck on a stale fragment.
	const postedTurnText = useRef<Map<string | number, string>>(new Map());
	useEffect(() => {
		for (const message of messageList) {
			const key = message.turn_id ?? `${message.uid}-${message.createdAt}`;
			if (String(message.uid) === agentUID) continue;
			const text = message.text?.trim();
			if (!text) continue;
			if (postedTurnText.current.get(key) === text) continue;

			postedTurnText.current.set(key, text);
			recordUtterance(
				agoraData.channel,
				String(message.uid),
				participantNames[String(message.uid)],
				text,
				typeof message.turn_id === "number" ? message.turn_id : postedTurnText.current.size,
				message.createdAt ?? Date.now(),
			).catch((error) => {
				console.error("Failed to record utterance:", error);
			});
		}
	}, [messageList, agentUID, agoraData.channel, participantNames]);

	const currentInProgressMessage = useMemo(() => {
		return getCurrentInProgressMessage(transcript);
	}, [transcript]);

	usePublish([localMicrophoneTrack]);

	useClientEvent(client, "user-joined", (user) => {
		if (user.uid.toString() === agentUID) setIsAgentConnected(true);
		playJoinChime();
	});

	useClientEvent(client, "user-left", (user) => {
		if (user.uid.toString() === agentUID) setIsAgentConnected(false);
	});

	// Remote mic mute/unmute for the participant tiles below. Muting here
	// (see handleMicToggle) uses track.setEnabled, not unpublish/publish --
	// that surfaces to other clients as "user-info-updated" with a
	// "mute-audio"/"unmute-audio" message, a different event than
	// user-published/user-unpublished (which is for a track being
	// attached/detached entirely, not just muted).
	const [remoteMutedUids, setRemoteMutedUids] = useState<Set<string>>(new Set());

	useClientEvent(client, "user-info-updated", (uid, msg) => {
		if (msg !== "mute-audio" && msg !== "unmute-audio") return;
		setRemoteMutedUids((prev) => {
			const next = new Set(prev);
			if (msg === "mute-audio") next.add(String(uid));
			else next.delete(String(uid));
			return next;
		});
	});

	// Per-participant speaking indicator for the grid view -- Agora reports
	// volume levels for every uid in the channel (local + remote) on each
	// tick; anything above a small threshold counts as "speaking" this tick.
	const [speakingUids, setSpeakingUids] = useState<Set<string>>(new Set());

	useEffect(() => {
		client.enableAudioVolumeIndicator();
	}, [client]);

	// Detects a mic that's gone silent while still "on" -- observed live on
	// mobile (incident-14, incident-16): a participant's audio stops
	// reaching the pipeline entirely partway through the call (OS/browser
	// suspending the mic on backgrounding, a permission getting revoked,
	// etc.) with nothing in the UI showing it. Tracked from the same
	// volume-indicator ticks already used for the speaking-ring indicator,
	// so this adds no extra polling.
	const MIC_SILENCE_WARNING_MS = 45_000;
	const lastLocalAudioAtRef = useRef<number>(Date.now());
	const [micSilenceWarning, setMicSilenceWarning] = useState(false);

	useClientEvent(client, "volume-indicator", (volumes) => {
		const SPEAKING_THRESHOLD = 15;
		const next = new Set<string>();
		const localUidStr = String(agoraData.uid);
		for (const v of volumes) {
			if (v.level > SPEAKING_THRESHOLD) next.add(String(v.uid));
			// Any non-trivial level counts as "the mic is producing audio" --
			// this is about total silence, not about whether they're speaking
			// loud enough to show the speaking ring.
			if (String(v.uid) === localUidStr && v.level > 2) {
				lastLocalAudioAtRef.current = Date.now();
			}
		}
		setSpeakingUids(next);
	});

	useEffect(() => {
		if (!isEnabled) {
			setMicSilenceWarning(false);
			return;
		}
		const interval = setInterval(() => {
			setMicSilenceWarning(Date.now() - lastLocalAudioAtRef.current > MIC_SILENCE_WARNING_MS);
		}, 5_000);
		return () => clearInterval(interval);
	}, [isEnabled]);

	// Best-effort recovery for the same failure: some mobile browsers
	// silently suspend an active mic track while the tab is backgrounded
	// (screen lock, app switch) without ever erroring or firing a "muted"
	// event -- re-asserting enabled state on return at least gives Agora a
	// chance to resume a track that's still alive but stalled.
	useEffect(() => {
		const handleVisibilityChange = () => {
			if (document.visibilityState === "visible" && localMicrophoneTrack && isEnabled) {
				localMicrophoneTrack.setEnabled(true).catch(() => {});
				lastLocalAudioAtRef.current = Date.now();
			}
		};
		document.addEventListener("visibilitychange", handleVisibilityChange);
		return () => document.removeEventListener("visibilitychange", handleVisibilityChange);
	}, [localMicrophoneTrack, isEnabled]);

	useEffect(() => {
		const isAgentInRemoteUsers = remoteUsers.some(
			(user) => user.uid.toString() === agentUID,
		);
		setIsAgentConnected(isAgentInRemoteUsers);
	}, [remoteUsers, agentUID]);

	useClientEvent(client, "connection-state-change", (curState) => {
		setConnectionState(curState);
	});

	// Confirmed live (incident-20): a signaling-layer hiccup ("ws request
	// timeout" on the RTC client, not the audio media path) can leave
	// transcript delivery permanently stalled even after the RTC client's
	// own connection-state cycles back to CONNECTED and audio keeps working
	// fine -- the only thing that fixed it was a full manual page rejoin.
	// AgoraVoiceAI's stream-message handling (how transcripts actually
	// arrive -- see agora-agent-client-toolkit's useTranscript) only resets
	// its internal state (chunked-message reassembly cache, its own raw
	// event bindings) on unsubscribe()/destroy(), never automatically on a
	// mid-session reconnect. unsubscribe() then subscribeMessage() again is
	// the toolkit's own public, documented pair for exactly this -- it
	// explicitly preserves the ai.on(...) consumer callbacks registered
	// above (destroy() would remove those instead) and doesn't touch the
	// live RTC/RTM connection, so this is much lower-risk than forcing an
	// actual leave+rejoin of the call.
	const lastResubscribeAtRef = useRef(0);
	const resubscribeTranscript = useCallback(() => {
		// Guards against re-triggering on rapid reconnect/connect flapping --
		// one resubscribe per interruption is enough, and doing it too often
		// risks racing a message that arrives in the brief unsubscribed gap.
		const RESUBSCRIBE_COOLDOWN_MS = 5_000;
		if (Date.now() - lastResubscribeAtRef.current < RESUBSCRIBE_COOLDOWN_MS) return;
		lastResubscribeAtRef.current = Date.now();

		const ai = AgoraVoiceAI.getInstance();
		if (!ai) return;
		try {
			ai.unsubscribe();
			ai.subscribeMessage(agoraData.channel);
		} catch (error) {
			console.error("[AgoraVoiceAI] Failed to resubscribe transcript delivery:", error);
		}
	}, [agoraData.channel]);

	const prevConnectionStateRef = useRef(connectionState);
	useEffect(() => {
		const prevState = prevConnectionStateRef.current;
		prevConnectionStateRef.current = connectionState;

		const wasInterrupted = prevState === "RECONNECTING" || prevState === "DISCONNECTED";
		if (connectionState !== "CONNECTED" || !wasInterrupted) return;
		resubscribeTranscript();
	}, [connectionState, resubscribeTranscript]);

	// The precise version of the fix above -- confirmed live (incident-34)
	// that the RTC-connectionState trigger and the watchdog below aren't
	// enough on their own. Traced unsubscribe()/subscribeMessage() into the
	// installed toolkit's source: they only call bindRtmEvents()/
	// unbindRtmEvents(), i.e. attach or detach listeners on whatever
	// rtmEngine instance was set at init() -- neither one reconnects
	// anything. So when the failure is the RTM transport itself ("ws open
	// error"), re-subscribing to a still-broken engine does nothing, which
	// is exactly what incident-34 showed: the watchdog fired and forced a
	// resubscribe, and the stall continued anyway.
	//
	// The RTM SDK has its own connection lifecycle, separate from the RTC
	// client's, and rtmClient.addEventListener("status", ...) is public API
	// (already used once, at initial login, in LandingPage.tsx's
	// waitForRtmConnected) -- not something AgoraVoiceAI exposes a hook for
	// internally (checked: _handleRtmStatus only logs at debug level and
	// never re-emits). Listening to it directly here means reacting to the
	// moment the RTM transport itself -- not just the RTC client -- actually
	// comes back, which is the one thing that can make re-subscribing
	// meaningful again.
	const prevRtmStateRef = useRef<string | undefined>(undefined);
	useEffect(() => {
		const onRtmStatus = (
			connectionStatus: { newState?: string } | { state?: string } | Record<string, unknown>,
		) => {
			const nextState =
				typeof connectionStatus === "object" && connectionStatus !== null
					? "newState" in connectionStatus
						? connectionStatus.newState
						: "state" in connectionStatus
							? connectionStatus.state
							: undefined
					: undefined;
			const prevState = prevRtmStateRef.current;
			prevRtmStateRef.current = typeof nextState === "string" ? nextState : prevState;

			const wasInterrupted = prevState === "RECONNECTING" || prevState === "DISCONNECTED";
			if (nextState !== "CONNECTED" || !wasInterrupted) return;
			resubscribeTranscript();
		};
		rtmClient.addEventListener("status", onRtmStatus);
		return () => rtmClient.removeEventListener("status", onRtmStatus);
	}, [rtmClient, resubscribeTranscript]);

	// Watchdog for the same stall as a backstop -- confirmed live
	// (incident-33): a "ws open error" on the RTM/transcript channel left
	// transcript delivery dead for an entire call while the RTC client's own
	// connectionState sat at CONNECTED throughout (never cycled through
	// RECONNECTING/DISCONNECTED), so the RTC-based effect above never fired.
	// Real speech WAS happening -- the backend logged real structuring turns
	// for that call -- only the browser-side transcript feed was dead. Kept
	// even now that the RTM-status trigger above exists: it catches
	// whatever the RTM SDK's own reconnect logic doesn't announce cleanly,
	// same reasoning as keeping the RTC-based trigger alongside it.
	//
	// agora-agent-client-toolkit already detects this exact condition (zero
	// TRANSCRIPT_UPDATED events) and logs it, but only once, 15s after
	// joining, as a console.warn with no consumer-facing event to hook into
	// (checked the installed package directly). This reimplements the same
	// idea ourselves, running for the whole call rather than once at join,
	// since the observed stall started mid-call, not at the start.
	const lastTranscriptAtRef = useRef(Date.now());
	useEffect(() => {
		lastTranscriptAtRef.current = Date.now();
	}, [messageList.length]);

	useEffect(() => {
		const TRANSCRIPT_STALL_MS = 25_000;
		const CHECK_INTERVAL_MS = 5_000;
		const interval = setInterval(() => {
			if (!isAgentConnected) return;
			if (Date.now() - lastTranscriptAtRef.current < TRANSCRIPT_STALL_MS) return;
			console.warn(
				`[AgoraVoiceAI] No transcript activity for ${TRANSCRIPT_STALL_MS}ms while the agent is connected -- forcing a resubscribe`,
			);
			resubscribeTranscript();
			// Give the resubscribe a full window to take effect before
			// considering it stalled again, rather than retrying every tick.
			lastTranscriptAtRef.current = Date.now();
		}, CHECK_INTERVAL_MS);
		return () => clearInterval(interval);
	}, [isAgentConnected, resubscribeTranscript]);

	// Agora's own uplink/downlink quality signal (0=unknown, 1=excellent,
	// ..., 6=disconnected), fired ~every 2s once joined. Surfaced so a poor
	// connection (weak wifi/cellular) shows up as a visible warning instead
	// of silently degrading STT/transcription with no indication to anyone
	// on the call that they might not be getting through -- see incident-14's
	// call, where SEND_AUDIO_BITRATE_TOO_LOW cycled the whole call with no
	// visible signal of it anywhere in the UI.
	const [networkQuality, setNetworkQuality] = useState({ uplink: 0, downlink: 0 });

	useClientEvent(client, "network-quality", (stats) => {
		setNetworkQuality({
			uplink: stats.uplinkNetworkQuality,
			downlink: stats.downlinkNetworkQuality,
		});
	});

	const connectionSeverity = useMemo<"normal" | "warning" | "error">(() => {
		if (
			connectionState === "DISCONNECTED" ||
			connectionState === "DISCONNECTING"
		) {
			return "error";
		}
		if (
			connectionState === "CONNECTING" ||
			connectionState === "RECONNECTING"
		) {
			return "warning";
		}
		const issueSeverity =
			connectionIssues.length === 0
				? "normal"
				: connectionIssues.some(
							(issue) => getConversationIssueSeverity(issue) === "error",
						)
					? "error"
					: "warning";

		const worstNetworkQuality = Math.max(networkQuality.uplink, networkQuality.downlink);
		const networkSeverity =
			worstNetworkQuality >= 4 ? "error" : worstNetworkQuality === 3 ? "warning" : "normal";

		if (issueSeverity === "error" || networkSeverity === "error") return "error";
		if (issueSeverity === "warning" || networkSeverity === "warning") return "warning";
		return "normal";
	}, [connectionState, connectionIssues, networkQuality]);

	const visualizerState = useMemo(
		() =>
			mapAgentVisualizerState(agentState, isAgentConnected, connectionState),
		[agentState, isAgentConnected, connectionState],
	);

	const handleMicToggle = useCallback(async () => {
		const next = !isEnabled;
		const track = localMicrophoneTrack;
		if (!track) {
			setIsEnabled(next);
			return;
		}
		try {
			await track.setEnabled(next);
			setIsEnabled(next);
		} catch (error) {
			console.error("Failed to toggle microphone:", error);
		}
	}, [isEnabled, localMicrophoneTrack]);

	const handleTokenWillExpire = useCallback(async () => {
		if (!onTokenWillExpire || !joinedUID) return;
		try {
			const { rtcToken, rtmToken } = await onTokenWillExpire(
				joinedUID.toString(),
			);
			await client?.renewToken(rtcToken);
			await rtmClient.renewToken(rtmToken);
		} catch (error) {
			console.error("Failed to renew Agora token:", error);
		}
	}, [client, onTokenWillExpire, joinedUID, rtmClient]);

	useClientEvent(client, "token-privilege-will-expire", handleTokenWillExpire);

	// Chrome/Safari block audio playback that isn't triggered by a user
	// gesture -- RemoteUser's automatic play() can silently lose this race
	// (observed live: agent TTS never audible, only a console warning).
	// onAutoplayFailed is a single global hook (fires once even if several
	// tracks failed at once), not a per-client event -- registering it here
	// is safe since only one ConversationComponent is ever mounted at a time.
	const [audioPlaybackBlocked, setAudioPlaybackBlocked] = useState(false);

	useEffect(() => {
		AgoraRTC.onAutoplayFailed = () => setAudioPlaybackBlocked(true);
	}, []);

	const handleResumeAudio = useCallback(() => {
		for (const user of remoteUsers) {
			if (user.audioTrack && !user.audioTrack.isPlaying) {
				user.audioTrack.play();
			}
		}
		setAudioPlaybackBlocked(false);
	}, [remoteUsers]);

	const handleEndConversation = useCallback(async () => {
		const track = localMicrophoneTrack;
		if (track) {
			try {
				await client?.unpublish(track);
			} catch (error) {
				console.warn("Failed to unpublish microphone track:", error);
			}

			try {
				track.stop();
				track.close();
			} catch (error) {
				console.warn("Failed to release microphone track:", error);
			}
		}

		onEndConversation();
	}, [client, localMicrophoneTrack, onEndConversation]);

	return (
		<>
			{(audioPlaybackBlocked || micSilenceWarning) && (
				<div className="fixed inset-x-0 top-0 z-50 flex flex-col items-center gap-2 p-3">
					{audioPlaybackBlocked && (
						<button
							type="button"
							onClick={handleResumeAudio}
							className="rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-lg"
						>
							Click to enable audio playback
						</button>
					)}
					{micSilenceWarning && (
						<div className="rounded-full bg-destructive px-4 py-2 text-sm font-medium text-destructive-foreground shadow-lg">
							No audio detected from your microphone -- check it's not muted or blocked
						</div>
					)}
				</div>
			)}
			<QuickstartConversationLayout
				statusPanel={
					<ConnectionStatusPanel
						connectionState={connectionState}
						connectionSeverity={connectionSeverity}
						connectionIssues={connectionIssues}
						networkQuality={networkQuality}
						isOpen={isConnectionDetailsOpen}
						onToggle={() => setIsConnectionDetailsOpen((open) => !open)}
					/>
				}
				pipelineMetrics={<QuickstartPipelineMetrics metrics={agentMetrics} />}
				transcriptPanel={
					<QuickstartTranscriptPanel
						messageList={messageList}
						currentInProgressMessage={currentInProgressMessage}
						agentUID={agentUID}
						localUid={agoraData.uid}
						participantNames={participantNames}
						chatNotes={chatNotes}
					/>
				}
				visualizer={
					<section
						className="relative flex h-full min-h-[20rem] w-full max-w-4xl flex-col items-center justify-center gap-6"
						aria-label="AI agent status visualization"
					>
						{remoteUsers.map((user) => (
							<div key={user.uid} className="hidden">
								<RemoteUser user={user} />
							</div>
						))}

						{/* Meet/Zoom-style participant grid. No video (audio-only call) --
						    each tile is an avatar circle + label, with a pulsing ring while
						    that participant's volume level is above the speaking threshold
						    (see the volume-indicator listener above). Names come from RTM
						    presence state (participantNames); falls back to a bare UID
						    label if a participant hasn't published a name yet. */}
						<div
							className="grid w-full max-w-4xl grid-cols-2 gap-6 sm:grid-cols-3"
							aria-label="Call participants"
						>
							{(() => {
								const localUid = agoraData.uid;
								const tiles = [
									{
										uid: localUid,
										label: "You",
										avatarName: localName || "You",
										isAgent: false,
										speaking: speakingUids.has(String(localUid)) && isEnabled,
										muted: !isEnabled,
									},
									...remoteUsers.map((user) => {
										const isAgent = String(user.uid) === String(agentUID);
										const remoteLabel = isAgent
											? "Watcher"
											: (participantNames[String(user.uid)] ??
												`Participant ${user.uid}`);
										return {
											uid: user.uid,
											label: remoteLabel,
											avatarName: remoteLabel,
											isAgent,
											speaking: isAgent
												? visualizerState === "talking"
												: speakingUids.has(String(user.uid)),
											// The agent has no manual mute toggle from a participant's
											// perspective -- the indicator only means something for
											// actual meeting members.
											muted: isAgent ? false : remoteMutedUids.has(String(user.uid)),
										};
									}),
								];

								return tiles.map((tile) => (
									<div
										key={tile.uid}
										className="flex flex-col items-center gap-3 rounded-2xl border border-border bg-card/60 px-5 py-8"
									>
										<div className="relative">
											<div
												className={`flex h-24 w-24 items-center justify-center rounded-full font-medium transition-shadow ${
													tile.isAgent ? "bg-primary/15 text-primary" : "bg-muted text-foreground"
												} ${tile.speaking ? "ring-4 ring-primary/70 animate-pulse" : ""}`}
											aria-hidden="true"
										>
											{tile.isAgent ? (
												<Sparkles className="h-9 w-9" strokeWidth={1.75} />
											) : (
												<span className="text-3xl">{getInitial(tile.avatarName)}</span>
											)}
										</div>
										{tile.muted ? (
											<span
												className="absolute -bottom-1 -right-1 flex h-7 w-7 items-center justify-center rounded-full bg-destructive text-destructive-foreground ring-2 ring-card"
												title={`${tile.label} is muted`}
											>
												<MicOff className="h-3.5 w-3.5" aria-hidden="true" />
												<span className="sr-only">{tile.label} is muted</span>
											</span>
										) : null}
									</div>
									<span className="max-w-full truncate text-sm font-medium text-foreground">
										{tile.label}
									</span>
								</div>
								));
							})()}
							{/* A special, always-present tile (not a participant) -- shows
							    the most recent GitHub/logs/MCP lookup directly in the grid,
							    live, with nothing to click or switch to. */}
							<McpResponseTile screens={sharedScreens} />
						</div>
					</section>
				}
				controls={
					<fieldset className="flex items-center gap-3" aria-label="Audio controls">
						<div className="conversation-mic-host flex items-center justify-center">
							<MicButtonWithVisualizer
								isEnabled={isEnabled}
								setIsEnabled={setIsEnabled}
								track={localMicrophoneTrack}
								onToggle={handleMicToggle}
								className="overflow-visible"
								aria-label={isEnabled ? "Mute microphone" : "Unmute microphone"}
								enabledColor="hsl(var(--primary))"
								disabledColor="hsl(var(--destructive))"
							/>
						</div>
						<MicrophoneSelector localMicrophoneTrack={localMicrophoneTrack} />
					</fieldset>
				}
				chatPanel={
					<MeetChatPanel
						channel={agoraData.channel}
						localUid={agoraData.uid}
						localName={localName}
						lateJoinRecap={lateJoinRecap}
					/>
				}
				timelinePanel={<LiveRecapPanel channelName={agoraData.channel} />}
				chatHasUnread={!!lateJoinRecap}
				onEndConversation={handleEndConversation}
			/>
		</>
	);
}
