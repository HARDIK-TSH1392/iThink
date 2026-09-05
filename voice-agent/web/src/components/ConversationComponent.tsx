"use client";

import { Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConnectionStatusPanel } from "@/components/ConnectionStatusPanel";
import {
	type ConnectionIssue,
	getConversationIssueSeverity,
} from "@/components/ConversationErrorCard";
import { MeetChatPanel } from "@/components/MeetChatPanel";
import { MicrophoneSelector } from "@/components/MicrophoneSelector";
import { QuickstartConversationLayout } from "@/components/QuickstartConversationLayout";
import {
	type QuickstartAgentMetric,
	QuickstartPipelineMetrics,
} from "@/components/QuickstartPipelineMetrics";
import { QuickstartTranscriptPanel } from "@/components/QuickstartTranscriptPanel";
import { SharedScreenPanel } from "@/components/SharedScreenPanel";
import { DEFAULT_AGENT_UID } from "@/lib/agora";
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
	// Increments on every new live broadcast so the layout can force its
	// panel open even if the viewer currently has a different one selected
	// -- a plain boolean wouldn't re-trigger on a second screen in a row.
	const [screenBroadcastSignal, setScreenBroadcastSignal] = useState(0);

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
			setScreenBroadcastSignal((count) => count + 1);
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
					setRawTranscript([...t]);
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
	// infer_and_store_participant_roles). Posted once per turn_id (a Set
	// ref, not state, since this is a side effect with nothing to render).
	// Agent lines are skipped -- role inference is about the humans on the
	// call, not the agent itself.
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
	const postedTurnIds = useRef<Set<string | number>>(new Set());
	useEffect(() => {
		for (const message of messageList) {
			const key = message.turn_id ?? `${message.uid}-${message.createdAt}`;
			if (postedTurnIds.current.has(key)) continue;
			if (String(message.uid) === agentUID) continue;
			if (!message.text?.trim()) continue;

			postedTurnIds.current.add(key);
			recordUtterance(
				agoraData.channel,
				String(message.uid),
				participantNames[String(message.uid)],
				message.text,
				typeof message.turn_id === "number" ? message.turn_id : postedTurnIds.current.size,
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
	});

	useClientEvent(client, "user-left", (user) => {
		if (user.uid.toString() === agentUID) setIsAgentConnected(false);
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

	// Real-time, room-wide warning for the exact failure confirmed live
	// across incident-14/16/17/19: someone's outgoing audio degrades badly
	// enough (low input level, or the encoder dropping bitrate) that real,
	// substantive speech never produces usable STT output at all -- not a
	// visible error, just silence where a transcript line should be. The
	// periodic network-quality score (below) is too coarse/laggy to catch
	// this moment-to-moment; these specific exception codes are Agora's own
	// direct signal for it. Room-wide (not just the affected person) since
	// the point is for everyone to know a gap might be happening right now,
	// not just to help the affected person fix their own connection.
	const AUDIO_QUALITY_PROBLEM_CODES = new Set([2001, 2003]); // AUDIO_INPUT_LEVEL_TOO_LOW, SEND_AUDIO_BITRATE_TOO_LOW
	const AUDIO_QUALITY_RECOVER_CODES = new Set([4001, 4003]); // matching *_RECOVER codes
	const [degradedAudioUids, setDegradedAudioUids] = useState<Set<string>>(new Set());

	useClientEvent(client, "exception", (event) => {
		const uid = String(event.uid);
		if (AUDIO_QUALITY_PROBLEM_CODES.has(event.code)) {
			setDegradedAudioUids((prev) => new Set(prev).add(uid));
		} else if (AUDIO_QUALITY_RECOVER_CODES.has(event.code)) {
			setDegradedAudioUids((prev) => {
				if (!prev.has(uid)) return prev;
				const next = new Set(prev);
				next.delete(uid);
				return next;
			});
		}
	});

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
	const prevConnectionStateRef = useRef(connectionState);
	const lastResubscribeAtRef = useRef(0);
	useEffect(() => {
		const prevState = prevConnectionStateRef.current;
		prevConnectionStateRef.current = connectionState;

		const wasInterrupted = prevState === "RECONNECTING" || prevState === "DISCONNECTED";
		if (connectionState !== "CONNECTED" || !wasInterrupted) return;

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
			console.error("[AgoraVoiceAI] Failed to resubscribe after reconnect:", error);
		}
	}, [connectionState, agoraData.channel]);

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

	// Human participants only -- the agent's own audio quality doesn't
	// affect whether *human* speech gets transcribed, which is what this
	// warning is about.
	const degradedAudioNames = Array.from(degradedAudioUids)
		.filter((uid) => uid !== agentUID)
		.map((uid) => participantNames[uid] ?? `Participant ${uid}`);

	return (
		<>
			{(audioPlaybackBlocked || micSilenceWarning || degradedAudioNames.length > 0) && (
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
					{degradedAudioNames.length > 0 && (
						<div className="rounded-full bg-amber-600 px-4 py-2 text-sm font-medium text-white shadow-lg">
							Poor connection for {degradedAudioNames.join(", ")} -- recent speech may not be
							transcribed accurately
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
									},
									...remoteUsers.map((user) => {
										const isAgent = String(user.uid) === String(agentUID);
										const remoteLabel = isAgent
											? "iThink Agent"
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
										};
									}),
								];

								return tiles.map((tile) => (
									<div
										key={tile.uid}
										className="flex flex-col items-center gap-3 rounded-2xl border border-border bg-card/60 px-5 py-8"
									>
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
										<span className="max-w-full truncate text-sm font-medium text-foreground">
											{tile.label}
										</span>
									</div>
								));
							})()}
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
				screensPanel={<SharedScreenPanel screens={sharedScreens} />}
				autoOpenScreensSignal={screenBroadcastSignal}
				onEndConversation={handleEndConversation}
			/>
		</>
	);
}
