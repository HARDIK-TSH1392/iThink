"use client";

import { FileText, MessageSquare, PhoneOff } from "lucide-react";
import Image from "next/image";
import { useState, type ReactNode } from "react";

type SidePanel = "transcript" | "chat" | null;

type QuickstartConversationLayoutProps = {
	statusPanel: ReactNode;
	pipelineMetrics: ReactNode;
	transcriptPanel: ReactNode;
	chatPanel: ReactNode;
	visualizer: ReactNode;
	controls: ReactNode;
	onEndConversation: () => void;
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
	visualizer,
	controls,
	onEndConversation,
}: QuickstartConversationLayoutProps) {
	const [activePanel, setActivePanel] = useState<SidePanel>("transcript");

	const togglePanel = (panel: SidePanel) => {
		setActivePanel((current) => (current === panel ? null : panel));
	};

	return (
		<div className="flex min-h-0 flex-1 flex-col text-left">
			<header className="flex shrink-0 items-center justify-between border-b border-border px-4 py-4 md:h-[76px] md:px-6 md:py-0">
				<div className="flex min-w-0 items-center gap-3">
					<Image
						src="/agora-logo-mark.svg"
						alt="Agora"
						width={40}
						height={40}
						className="h-10 w-10 shrink-0 object-contain"
					/>
					<div className="flex min-w-0 flex-col justify-center gap-1">
						<span className="truncate text-lg font-semibold leading-none tracking-[-0.025em] text-foreground">
							iThink
						</span>
					</div>
				</div>

				{statusPanel}
			</header>

			<div className="flex min-h-0 w-full flex-1 flex-col gap-4 px-4 pb-4 pt-4 md:px-6 lg:flex-row lg:gap-0">
				{activePanel ? (
					<aside className="order-2 h-64 min-h-0 w-full shrink-0 lg:order-1 lg:h-full lg:w-[26rem]">
						{activePanel === "transcript" ? transcriptPanel : chatPanel}
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
								aria-label={activePanel === "chat" ? "Hide chat" : "Show chat"}
								title={activePanel === "chat" ? "Hide chat" : "Show chat"}
								className={`flex h-10 w-10 items-center justify-center rounded-full transition-colors ${
									activePanel === "chat"
										? "bg-primary/15 text-primary"
										: "text-muted-foreground hover:bg-muted hover:text-foreground"
								}`}
							>
								<MessageSquare className="h-[18px] w-[18px]" />
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
