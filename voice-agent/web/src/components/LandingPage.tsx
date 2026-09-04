"use client";

import type { RTMClient } from "agora-rtm";
import dynamic from "next/dynamic";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";

import { ErrorBoundary } from "@/components/ErrorBoundary";
import { LoadingSkeleton } from "@/components/LoadingSkeleton";
import { PreCallBackdrop } from "@/components/PreCallBackdrop";
import { QuickstartPreCallCard } from "@/components/QuickstartPreCallCard";
import { ShareButton } from "@/components/share-button";
import { getConfig, markCallCompleted, removeName, setName, startAgent } from "@/services/api";
import type { AgoraRenewalTokens, AgoraTokenData } from "@/types/conversation";

const ConversationComponent = dynamic(
	() => import("@/components/ConversationComponent"),
	{
		ssr: false,
	},
);

function waitForRtmConnected(rtmClient: RTMClient, timeoutMs = 600): Promise<void> {
	return new Promise((resolve) => {
		let settled = false;
		let timer: ReturnType<typeof setTimeout> | null = null;

		const finish = () => {
			if (settled) return;
			settled = true;
			if (timer) clearTimeout(timer);
			rtmClient.removeEventListener("status", onStatus);
			resolve();
		};

		const onStatus = (
			connectionStatus:
				| { newState?: string }
				| { state?: string }
				| Record<string, unknown>,
		) => {
			const nextState =
				typeof connectionStatus === "object" && connectionStatus !== null
					? "newState" in connectionStatus
						? connectionStatus.newState
						: "state" in connectionStatus
							? connectionStatus.state
							: undefined
					: undefined;
			if (nextState === "CONNECTED") {
				finish();
			}
		};

		rtmClient.addEventListener("status", onStatus);
		timer = setTimeout(finish, timeoutMs);
	});
}

const AgoraProvider = dynamic(
	async () => {
		const { AgoraRTCProvider, default: AgoraRTC } = await import(
			"agora-rtc-react"
		);

		return {
			default: function AgoraProviders({
				children,
			}: { children: React.ReactNode }) {
				const clientRef = useRef<ReturnType<
					typeof AgoraRTC.createClient
				> | null>(null);
				if (!clientRef.current) {
					clientRef.current = AgoraRTC.createClient({
						mode: "rtc",
						codec: "vp8",
					});
				}
				return (
					<AgoraRTCProvider client={clientRef.current}>
						{children}
					</AgoraRTCProvider>
				);
			},
		};
	},
	{ ssr: false },
);

export default function LandingPage() {
	// iThink demo hook: ?channel=incident-3 joins that specific incident's
	// room instead of a randomly generated one, so the room created by
	// POST /icall/incidents/{id}/call is the one a human actually joins.
	const searchParams = useSearchParams();
	const channelFromUrl = searchParams.get("channel") ?? undefined;
	// Lets two people share a join link with fixed UIDs (e.g. ?uid=1001 for
	// one teammate, ?uid=1002 for the other) so they land in the same room
	// instead of each getting a randomly generated UID.
	const uidFromUrl = searchParams.get("uid") ?? undefined;

	const [showConversation, setShowConversation] = useState(false);
	const [agoraData, setAgoraData] = useState<AgoraTokenData | null>(null);
	const [rtmClient, setRtmClient] = useState<RTMClient | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [agentJoinError, setAgentJoinError] = useState(false);
	const [localName, setLocalName] = useState("");
	const [lateJoinRecap, setLateJoinRecap] = useState<string | null>(null);

	useEffect(() => {
		import("agora-rtc-react").catch(() => {});
		import("agora-rtm").catch(() => {});
	}, []);

	// Covers closing the tab / navigating away without clicking "End
	// Conversation" -- handleEndConversation's removeName call never runs
	// in that case, so the registry would otherwise keep this uid listed
	// as "still on the call" indefinitely. sendBeacon (not fetch) is what
	// actually survives a page already being torn down.
	useEffect(() => {
		if (!agoraData?.channel) return;
		const { channel, uid } = agoraData;

		const handlePageHide = () => {
			navigator.sendBeacon?.(
				"/api/removeName",
				new Blob([JSON.stringify({ channelName: channel, uid })], {
					type: "application/json",
				}),
			);
		};

		window.addEventListener("pagehide", handlePageHide);
		return () => window.removeEventListener("pagehide", handlePageHide);
	}, [agoraData]);

	const handleStartConversation = async (name: string) => {
		setIsLoading(true);
		setError(null);
		setAgentJoinError(false);
		setLocalName(name);

		try {
			const config = await getConfig({ channel: channelFromUrl, uid: uidFromUrl });
			const appId = config.app_id;

			const [agentIdResult, rtm, setNameResult] = await Promise.all([
				startAgent(
					config.channel_name,
					Number(config.agent_uid),
					Number(config.uid),
				).catch((err) => {
					console.error("Failed to start conversation with agent:", err);
					setAgentJoinError(true);
					return undefined;
				}),
				(async () => {
					const { default: AgoraRTM } = await import("agora-rtm");
					const nextRtm: RTMClient = new AgoraRTM.RTM(appId, config.uid);
					await nextRtm.login({ token: config.token });
					await waitForRtmConnected(nextRtm);
					await nextRtm.subscribe(config.channel_name);
					return nextRtm;
				})(),
				// Best-effort: publish our display name so other participants can
				// label us properly. Shouldn't block joining the call if it fails.
				// When we're a late joiner, the response also carries a private
				// catch-up recap meant for us alone -- see setLateJoinRecap below.
				setName(config.channel_name, config.uid, name).catch((err) => {
					console.error("Failed to publish display name:", err);
					return { recap: null };
				}),
			]);

			setRtmClient(rtm);
			setLateJoinRecap(setNameResult.recap);
			setAgoraData({
				token: config.token,
				uid: config.uid,
				channel: config.channel_name,
				appId: config.app_id,
				// Use the uid the backend confirms is actually running the
				// agent, not our own pre-join guess (config.agent_uid) --
				// they differ whenever someone else already started it.
				agentUid: agentIdResult?.agentUid ?? config.agent_uid,
				agentId: agentIdResult?.agentId,
			});
			setShowConversation(true);
		} catch (nextError) {
			setError("Failed to start conversation. Please try again.");
			console.error("Error starting conversation:", nextError);
		} finally {
			setIsLoading(false);
		}
	};

	const handleTokenWillExpire = useCallback(
		async (uid: string): Promise<AgoraRenewalTokens> => {
			try {
				const channel = agoraData?.channel;
				if (!channel) {
					throw new Error("Missing channel for token renewal");
				}

				// Python get_config issues RTM-capable tokens for the configured account,
				// so renew RTM with the same UID used by the RTM client login.
				const [rtcConfig, rtmConfig] = await Promise.all([
					getConfig({ channel, uid }),
					getConfig({ channel, uid: agoraData.uid }),
				]);

				return {
					rtcToken: rtcConfig.token,
					rtmToken: rtmConfig.token,
				};
			} catch (error) {
				console.error("Error renewing token:", error);
				throw error;
			}
		},
		[agoraData],
	);

	const handleEndConversation = async () => {
		// Deliberately doesn't call stopAgent: with remote_rtc_uids: ["*"] one
		// agent is shared by everyone on the call, so "end conversation" must
		// only leave the channel for *this* participant, not kill the agent
		// for whoever's still on the call. The agent's own idle_timeout (see
		// agent.py) stops it a few seconds after the last human leaves.

		// Marks the call completed and triggers post-call role inference
		// (backend/app/iCall/iCall_service.infer_and_store_participant_roles).
		// Safe to call once per person leaving -- the backend only fires
		// Slack/Jira notifications on the first transition into "completed"
		// and just refreshes role inference on any repeat call.
		if (agoraData?.channel) {
			markCallCompleted(agoraData.channel).catch((err) =>
				console.error("Failed to mark call completed:", err),
			);
			// Drops this uid from the name registry so the pre-call "N
			// already on the call" count (and getNames generally) reflects
			// who's actually still there, not everyone who's ever joined.
			removeName(agoraData.channel, agoraData.uid).catch((err) =>
				console.error("Failed to remove name on leave:", err),
			);
		}

		rtmClient?.logout().catch((err) => console.error("RTM logout error:", err));
		setRtmClient(null);
		setAgoraData(null);
		setShowConversation(false);
	};

	return (
		<div className="relative flex h-dvh min-h-screen flex-col overflow-hidden bg-background text-foreground">
			{!showConversation ? <PreCallBackdrop /> : null}
			<div
				className={`flex min-h-0 flex-1 flex-col ${
					showConversation
						? "items-stretch justify-start"
						: "items-center justify-center"
				}`}
			>
				<div
					className={`z-10 flex min-h-0 flex-1 flex-col ${
						showConversation
							? "h-full w-full max-w-none items-stretch gap-0 px-0 text-left"
							: "w-full max-w-none items-center justify-center px-4 text-center"
					}`}
				>
					{!showConversation ? (
						<QuickstartPreCallCard
							isLoading={isLoading}
							error={error}
							onStartConversation={handleStartConversation}
							channel={channelFromUrl}
						/>
					) : agoraData && rtmClient ? (
						<>
							{agentJoinError ? (
								<div className="max-w-sm rounded-md bg-destructive/10 p-3 text-sm text-destructive">
									Failed to connect with AI agent. The conversation may not work
									as expected.
								</div>
							) : null}
							<Suspense fallback={<LoadingSkeleton />}>
								<ErrorBoundary>
									<AgoraProvider>
										<ConversationComponent
											agoraData={agoraData}
											rtmClient={rtmClient}
											localName={localName}
											lateJoinRecap={lateJoinRecap}
											onTokenWillExpire={handleTokenWillExpire}
											onEndConversation={handleEndConversation}
										/>
									</AgoraProvider>
								</ErrorBoundary>
							</Suspense>
						</>
					) : (
						<p className="text-sm text-muted-foreground">
							Failed to load conversation data.
						</p>
					)}
				</div>
			</div>

			<footer
				className={`fixed inset-x-0 bottom-0 z-40 flex items-center gap-4 px-4 py-4 md:px-6 md:py-6 ${
					showConversation ? "justify-end" : "justify-between"
				}`}
			>
				{!showConversation ? <ShareButton menuPlacement="top" /> : null}
				<div className="flex items-center justify-end gap-2 text-muted-foreground">
					<span className="text-xs font-medium uppercase tracking-wide">
						Powered by
					</span>
					<a
						href="https://agora.io/en/"
						target="_blank"
						rel="noopener noreferrer"
						className="transition-colors hover:text-primary"
						aria-label="Visit Agora's website"
					>
						<Image
							src="/agora-logo-rgb-blue.svg"
							alt="Agora"
							width={86}
							height={24}
							priority
							className="h-6 w-auto translate-y-1 transition-opacity hover:opacity-80"
						/>
						<span className="sr-only">Agora</span>
					</a>
				</div>
			</footer>
		</div>
	);
}
