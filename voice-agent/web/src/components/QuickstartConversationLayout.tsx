"use client";

import { ClipboardList, FileText, MessageSquare, PhoneOff } from "lucide-react";
import Image from "next/image";
import { useState, type ReactNode } from "react";

type SidePanel = "transcript" | "chat" | "timeline" | null;

type QuickstartConversationLayoutProps = {
	statusPanel: ReactNode;
	pipelineMetrics: ReactNode;
	transcriptPanel: ReactNode;
	chatPanel: ReactNode;
	// The brief's "a continuously updated incident timeline" -- previously
	// only queryable server-side, never shown to anyone actually on the
	// call. A third tab alongside transcript/chat, since it's a different
	// kind of content (derived/structured state, not a message log). The
	// old fourth "screens" tab is gone -- superseded by McpResponseTile's
	// always-visible grid tile (see ConversationComponent), not folded
	// into this one.
	timelinePanel: ReactNode;
	visualizer: ReactNode;
	controls: ReactNode;
	onEndConversation: () => void;
	/** True when this viewer has an unseen message waiting in chat (e.g. a late-join recap) -- shows a dot on the Chat button until they open it. */
	chatHasUnread: boolean;
};

export function QuickstartConversationLayout({
	statusPanel,
	// Intentionally unused -- the pipeline/latency badges (STT/LLM/TTS +
	// timings) were removed from view per iThink's demo needs, but the
	// underlying metrics tracking in ConversationComponent stays intact in
	// case they're wanted again later (e.g. a debug view).
	pipelineMetrics: _pipelineMetrics,
	transcriptPanel,
	chatPanel,
	timelinePanel,
	visualizer,
	controls,
	onEndConversation,
	chatHasUnread,
}: QuickstartConversationLayoutProps) {
	const [activePanel, setActivePanel] = useState<SidePanel>("transcript");
	// Tracks locally once the viewer has actually opened chat, independent
	// of whether chatHasUnread itself ever changes -- a late-join recap is
	// a one-time thing, there's no "mark as read" signal to send back up.
	const [hasSeenChat, setHasSeenChat] = useState(false);

	const togglePanel = (panel: SidePanel) => {
		setActivePanel((current) => (current === panel ? null : panel));
		if (panel === "chat") setHasSeenChat(true);
	};

	const showChatDot = chatHasUnread && !hasSeenChat;

	return (
		<div className="flex min-h-0 flex-1 flex-col text-left">
			<header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-4 md:h-[76px] md:px-6 md:py-0">
				<div className="flex min-w-0 items-center gap-3">
					<Image
						src="/watcher-logo.jpeg"
						alt="Watcher"
						width={40}
						height={40}
						className="h-10 w-10 shrink-0 rounded-lg object-contain"
					/>
					<div className="flex min-w-0 flex-col justify-center gap-1">
						<span className="truncate text-lg font-semibold leading-none tracking-[-0.025em] text-foreground">
							Watcher
						</span>
					</div>
				</div>

				{statusPanel}
			</header>

			<div className="flex min-h-0 w-full flex-1 flex-col gap-4 px-4 pb-4 pt-4 md:px-6 lg:flex-row lg:gap-0">
				{activePanel ? (
					<aside className="order-2 h-64 min-h-0 w-full shrink-0 lg:order-1 lg:h-full lg:w-[26rem]">
						{activePanel === "transcript"
							? transcriptPanel
							: activePanel === "chat"
								? chatPanel
								: timelinePanel}
					</aside>
				) : null}

				<main
					className={`order-1 flex min-h-0 flex-1 flex-col lg:order-2 ${
						activePanel ? "lg:border-l lg:border-border/80 lg:pl-6" : ""
					}`}
				>
					<div className="flex min-h-0 flex-1 flex-col pb-2 pt-3 md:pb-6">
						<div className="flex min-h-0 flex-1 items-center justify-center">
							{visualizer}
						</div>

						{/* One unified control bar: transcript, chat, mic/settings
						    (passed in as `controls`), end call -- Meet/Zoom-style,
						    bottom-center. */}
						<div className="mx-auto mt-4 flex w-fit items-center gap-2 rounded-full border border-border bg-card/80 px-3 py-2 backdrop-blur-md">
							<button
								type="button"
								onClick={() => togglePanel("transcript")}
								aria-pressed={activePanel === "transcript"}
								aria-label={activePanel === "transcript" ? "Hide transcript" : "Show transcript"}
								title={activePanel === "transcript" ? "Hide transcript" : "Show transcript"}
								className={`flex h-10 w-10 items-center justify-center rounded-full transition-colors ${
									activePanel === "transcript"
										? "bg-primary/15 text-primary"
										: "text-muted-foreground hover:bg-muted hover:text-foreground"
								}`}
							>
								<FileText className="h-[18px] w-[18px]" />
							</button>

							<button
								type="button"
								onClick={() => togglePanel("chat")}
								aria-pressed={activePanel === "chat"}
								aria-label={
									showChatDot
										? "Show chat (new message)"
										: activePanel === "chat"
											? "Hide chat"
											: "Show chat"
								}
								title={showChatDot ? "New message in chat" : activePanel === "chat" ? "Hide chat" : "Show chat"}
								className={`relative flex h-10 w-10 items-center justify-center rounded-full transition-colors ${
									activePanel === "chat"
										? "bg-primary/15 text-primary"
										: "text-muted-foreground hover:bg-muted hover:text-foreground"
								}`}
							>
								<MessageSquare className="h-[18px] w-[18px]" />
								{showChatDot ? (
									<span
										className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-destructive ring-2 ring-card"
										aria-hidden="true"
									/>
								) : null}
							</button>

							<button
								type="button"
								onClick={() => togglePanel("timeline")}
								aria-pressed={activePanel === "timeline"}
								aria-label={activePanel === "timeline" ? "Hide incident timeline" : "Show incident timeline"}
								title={activePanel === "timeline" ? "Hide incident timeline" : "Show incident timeline"}
								className={`flex h-10 w-10 items-center justify-center rounded-full transition-colors ${
									activePanel === "timeline"
										? "bg-primary/15 text-primary"
										: "text-muted-foreground hover:bg-muted hover:text-foreground"
								}`}
							>
								<ClipboardList className="h-[18px] w-[18px]" />
							</button>

							<div className="mx-1 h-6 w-px bg-border" aria-hidden="true" />

							{controls}

							<div className="mx-1 h-6 w-px bg-border" aria-hidden="true" />

							<button
								type="button"
								onClick={onEndConversation}
								aria-label="End conversation with AI agent"
								title="End conversation"
								className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive text-destructive-foreground transition-colors hover:bg-destructive/85"
							>
								<PhoneOff className="h-[18px] w-[18px]" />
							</button>
						</div>
					</div>
				</main>
			</div>
		</div>
	);
}
