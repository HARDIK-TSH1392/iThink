"use client";

import { FileText, MessageSquareDashed, Sparkles, StickyNote } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { getInitial } from "@/lib/conversation";
import { getLanguageStatus, type LanguageStatus } from "@/services/api";

const LANGUAGE_DISPLAY_NAMES: Record<string, string> = {
	multi: "English/Hindi",
	"ta-IN": "Tamil", "te-IN": "Telugu", "kn-IN": "Kannada", "bn-IN": "Bengali",
	"mr-IN": "Marathi", "gu-IN": "Gujarati", "pa-IN": "Punjabi",
	"ml-IN": "Malayalam", "or-IN": "Odia",
};

const LANGUAGE_STATUS_POLL_MS = 3_000;

type TranscriptMessage = {
	turn_id?: string | number;
	uid: number;
	text?: string;
	createdAt?: number;
};

type ChatNote = {
	text: string;
	timestamp: string;
};

type QuickstartTranscriptPanelProps = {
	messageList: TranscriptMessage[];
	currentInProgressMessage: TranscriptMessage | null;
	agentUID: string;
	localUid: string;
	participantNames: Record<string, string>;
	chatNotes: ChatNote[];
	channel: string;
};

// Deepgram mishears "Watcher" often enough that the backend now recognizes
// these as valid wake-word attempts too (see iCall_utils.py's
// _KNOWN_MISHEARINGS -- keep both lists in sync). That fix makes the agent
// respond correctly, but doesn't change the text Deepgram actually
// produced -- without this, the transcript panel would keep showing
// "Voucher" on screen even while the agent visibly answers as if it heard
// "Watcher". Display-only: doesn't touch persisted transcripts or
// anything the structuring pipeline sees, just what's rendered here.
// Extended with Sarvam-observed mishearings (vachar, vache) -- same root
// W-to-V acoustic confusion pattern, independently confirmed on a second,
// unrelated STT vendor. Keep in sync with iCall_utils.py's
// _KNOWN_MISHEARINGS.
const KNOWN_MISHEARINGS = /\b(voucher|vajar|vucher|voacher|vacher|varcher|vachar|vache)\b/gi;

function displayText(text: string): string {
	return text.replace(KNOWN_MISHEARINGS, "Watcher");
}

function formatMessageTime(createdAt?: number) {
	if (!createdAt) return null;
	return new Intl.DateTimeFormat(undefined, {
		hour: "numeric",
		minute: "2-digit",
	}).format(new Date(createdAt));
}

export function QuickstartTranscriptPanel({
	messageList,
	currentInProgressMessage,
	agentUID,
	localUid,
	participantNames,
	chatNotes,
	channel,
}: QuickstartTranscriptPanelProps) {
	const scrollRef = useRef<HTMLDivElement>(null);

	// Polled, not pushed -- the handoff itself has real, measured latency
	// (voice-agent server's switch_language does a stop -> poll -> start
	// round trip), so this banner exists specifically to not make that
	// gap look like the agent went silent.
	const [languageStatus, setLanguageStatus] = useState<LanguageStatus | null>(null);
	useEffect(() => {
		if (!channel) return;
		let cancelled = false;
		const poll = () => {
			getLanguageStatus(channel).then((status) => {
				if (!cancelled) setLanguageStatus(status);
			});
		};
		poll();
		const interval = setInterval(poll, LANGUAGE_STATUS_POLL_MS);
		return () => {
			cancelled = true;
			clearInterval(interval);
		};
	}, [channel]);
	// Sticky-scroll, not force-scroll: without tracking this, the effect
	// below re-ran on every render (new turns, streaming token updates) and
	// unconditionally snapped scrollTop back to the bottom, fighting any
	// attempt to scroll up to read earlier turns. Threshold-based "was the
	// user already at the bottom" check, same pattern as any chat UI --
	// only auto-follow new messages when they hadn't scrolled away.
	const isAtBottomRef = useRef(true);
	const NEAR_BOTTOM_THRESHOLD_PX = 80;

	const handleScroll = () => {
		const node = scrollRef.current;
		if (!node) return;
		const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
		isAtBottomRef.current = distanceFromBottom < NEAR_BOTTOM_THRESHOLD_PX;
	};

	const messages = useMemo(
		() =>
			currentInProgressMessage
				? [...messageList, currentInProgressMessage]
				: messageList,
		[currentInProgressMessage, messageList],
	);

	useEffect(() => {
		const node = scrollRef.current;
		if (!node || !isAtBottomRef.current) return;
		node.scrollTop = node.scrollHeight;
	});

	return (
		<section
			className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#141414] shadow-[0_12px_32px_rgba(0,0,0,0.4)]"
			aria-label="Transcription panel"
		>
			<div className="flex h-14 shrink-0 items-center gap-3 border-b border-white/10 bg-white/[0.03] px-4">
				<div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
					<FileText className="h-4 w-4" />
				</div>
				<div>
					<h2 className="text-sm font-semibold text-white">Transcript</h2>
					<p className="text-xs text-white/60">Live voice turns</p>
				</div>
			</div>

			{languageStatus?.switchPending ? (
				<div
					className="flex shrink-0 items-center gap-2 border-b border-primary/20 bg-primary/10 px-4 py-2 text-xs font-medium text-primary"
					aria-live="polite"
				>
					<span className="h-2 w-2 animate-pulse rounded-full bg-primary" aria-hidden="true" />
					Switching to {LANGUAGE_DISPLAY_NAMES[languageStatus.switchTarget ?? ""] ?? languageStatus.switchTarget}...
				</div>
			) : null}

			{chatNotes.length > 0 ? (
				<div
					className="flex max-h-32 shrink-0 flex-col gap-2 overflow-y-auto border-b border-amber-500/20 bg-amber-500/10 px-4 py-3"
					aria-label="Agent notes (not spoken aloud)"
				>
					{chatNotes.map((note, index) => (
						<div key={index} className="flex items-start gap-2 text-xs leading-5 text-white/90">
							<StickyNote className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" aria-hidden="true" />
							<span>{note.text}</span>
						</div>
					))}
				</div>
			) : null}

			<div
				ref={scrollRef}
				onScroll={handleScroll}
				className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 py-4"
			>
				{messages.length === 0 ? (
					<div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm text-white/60">
						<MessageSquareDashed className="h-9 w-9 text-white/25" aria-hidden="true" />
						<span>Start speaking to see the live transcript here.</span>
					</div>
				) : (
					messages.map((message, index) => {
						const uidStr = String(message.uid);
						const isAgent = uidStr === agentUID;
						const isLocal = uidStr === localUid;
						const label = isAgent
							? "Watcher"
							: isLocal
								? "You"
								: (participantNames[uidStr] ?? `Participant ${uidStr}`);
						const rawText = message.text?.trim();
						const text = isAgent || !rawText ? rawText : displayText(rawText);
						const time = formatMessageTime(message.createdAt);

						return (
							<article
								key={`${message.turn_id ?? message.uid}-${index}`}
								className={`flex items-end gap-2 ${isAgent ? "flex-row" : "flex-row-reverse"}`}
							>
								<div
									className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
										isAgent ? "bg-primary/20 text-primary" : "bg-muted text-foreground"
									}`}
									aria-hidden="true"
								>
									{isAgent ? <Sparkles className="h-3.5 w-3.5" /> : getInitial(label)}
								</div>
								<div className={`flex max-w-[80%] flex-col ${isAgent ? "items-start" : "items-end"}`}>
									<div className="mb-1 flex items-center gap-2 px-1 text-xs font-semibold text-white/60">
										<span>{label}</span>
										{time ? <span className="font-normal">{time}</span> : null}
									</div>
									<div
										className={`whitespace-pre-wrap rounded-xl border px-3 py-2 text-sm leading-6 ${
											isAgent
												? "border-primary/25 bg-primary/10 text-white"
												: "border-border bg-foreground text-background"
										}`}
									>
										{text || "..."}
									</div>
								</div>
							</article>
						);
					})
				)}
			</div>
		</section>
	);
}
