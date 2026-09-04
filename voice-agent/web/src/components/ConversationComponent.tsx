"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConnectionStatusPanel } from "@/components/ConnectionStatusPanel";
import {
	type ConnectionIssue,
	getConversationIssueSeverity,
} from "@/components/ConversationErrorCard";
import { MicrophoneSelector } from "@/components/MicrophoneSelector";
import { QuickstartConversationLayout } from "@/components/QuickstartConversationLayout";
import {
	type QuickstartAgentMetric,
	QuickstartPipelineMetrics,
} from "@/components/QuickstartPipelineMetrics";
import { QuickstartTranscriptPanel } from "@/components/QuickstartTranscriptPanel";
import { DEFAULT_AGENT_UID } from "@/lib/agora";
import { type ChatNote, getChatNotes, getNames, recordUtterance } from "@/services/api";
import {
	getCurrentInProgressMessage,
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
import {
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

export default function ConversationComponent({
	agoraData,
	rtmClient,
	localName,
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

	const { localMicrophoneTrack } = useLocalMicrophoneTrack(isReady);

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

	useClientEvent(client, "volume-indicator", (volumes) => {
		const SPEAKING_THRESHOLD = 15;
		const next = new Set<string>();
		for (const v of volumes) {
			if (v.level > SPEAKING_THRESHOLD) next.add(String(v.uid));
		}
		setSpeakingUids(next);
	});

	useEffect(() => {
		const isAgentInRemoteUsers = remoteUsers.some(
			(user) => user.uid.toString() === agentUID,
		);
		setIsAgentConnected(isAgentInRemoteUsers);
	}, [remoteUsers, agentUID]);

	useClientEvent(client, "connection-state-change", (curState) => {
		setConnectionState(curState);
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
		if (connectionIssues.length === 0) {
			return "normal";
		}
		return connectionIssues.some(
			(issue) => getConversationIssueSeverity(issue) === "error",
		)
			? "error"
			: "warning";
	}, [connectionState, connectionIssues]);

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
		<QuickstartConversationLayout
			statusPanel={
				<ConnectionStatusPanel
					connectionState={connectionState}
					connectionSeverity={connectionSeverity}
					connectionIssues={connectionIssues}
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
									isAgent: false,
									speaking: speakingUids.has(String(localUid)) && isEnabled,
								},
								...remoteUsers.map((user) => {
									const isAgent = String(user.uid) === String(agentUID);
									return {
										uid: user.uid,
										label: isAgent
											? "iThink Agent"
											: (participantNames[String(user.uid)] ??
												`Participant ${user.uid}`),
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
										className={`flex h-24 w-24 items-center justify-center rounded-full text-4xl transition-shadow ${
											tile.isAgent ? "bg-primary/15 text-primary" : "bg-muted text-foreground"
										} ${tile.speaking ? "ring-4 ring-primary/70 animate-pulse" : ""}`}
										aria-hidden="true"
									>
										{tile.isAgent ? "\u{1F916}" : "\u{1F464}"}
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
				<fieldset
					className="mx-auto flex w-fit items-center gap-3 rounded-full border border-border bg-card/80 px-4 py-2 backdrop-blur-md"
					aria-label="Audio controls"
				>
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
			onEndConversation={handleEndConversation}
		/>
	);
}
