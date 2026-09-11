"use client";

import { MessageSquare, MessageSquareDashed, Send, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { getInitial } from "@/lib/conversation";
import { type MeetChatMessage, getMeetChatMessages, sendMeetChatMessage } from "@/services/api";

// Sentinel uid for the agent's late-join recap banner below -- local-only,
// never sent to or read from the server, so it can't collide with a real
// participant's uid.
const AGENT_CHAT_UID = "agent";

type MeetChatPanelProps = {
	channel: string;
	localUid: string;
	localName: string;
	/** Private "what did I miss" recap for this viewer alone, from setName's response. */
	lateJoinRecap: string | null;
};

function formatMessageTime(timestamp: number) {
	return new Intl.DateTimeFormat(undefined, {
		hour: "numeric",
		minute: "2-digit",
	}).format(new Date(timestamp));
}

export function MeetChatPanel({ channel, localUid, localName, lateJoinRecap }: MeetChatPanelProps) {
	const [messages, setMessages] = useState<MeetChatMessage[]>([]);
	const [draft, setDraft] = useState("");
	const [sending, setSending] = useState(false);
	const scrollRef = useRef<HTMLDivElement>(null);

	// Captured once, from a prop that came from this client's own setName
	// response -- never fetched from or merged into the polled/shared
	// `messages` list above, so it stays visible to this viewer only and
	// survives every poll tick untouched.
	const [recap] = useState<MeetChatMessage | null>(() =>
		lateJoinRecap
			? { uid: AGENT_CHAT_UID, name: "Watcher", text: lateJoinRecap, timestamp: Date.now() }
			: null,
	);
	const displayMessages = recap ? [recap, ...messages] : messages;

	useEffect(() => {
		let cancelled = false;

		const refresh = () => {
			getMeetChatMessages(channel)
				.then((next) => {
					if (!cancelled) setMessages((prev) => (next.length !== prev.length ? next : prev));
				})
				.catch(() => {});
		};

		refresh();
		const interval = setInterval(refresh, 2000);
		return () => {
			cancelled = true;
			clearInterval(interval);
		};
	}, [channel]);

	useEffect(() => {
		const node = scrollRef.current;
		if (!node) return;
		node.scrollTop = node.scrollHeight;
	}, [displayMessages]);

	const handleSend = async () => {
		const text = draft.trim();
		if (!text || sending) return;
		setSending(true);
		setDraft("");
		try {
			await sendMeetChatMessage(channel, localUid, localName || "You", text);
			// Optimistic append -- don't wait for the next poll tick to see
			// your own message land.
			setMessages((prev) => [
				...prev,
				{ uid: localUid, name: localName || "You", text, timestamp: Date.now() },
			]);
		} catch (error) {
			console.error("Failed to send chat message:", error);
		} finally {
			setSending(false);
		}
	};

	return (
		<section
			className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#141414] shadow-[0_12px_32px_rgba(0,0,0,0.4)]"
			aria-label="In-call chat"
		>
			<div className="flex h-14 shrink-0 items-center gap-3 border-b border-white/10 bg-white/[0.03] px-4">
				<div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
					<MessageSquare className="h-4 w-4" />
				</div>
				<div>
					<h2 className="text-sm font-semibold text-white">Chat</h2>
					<p className="text-xs text-white/60">Message everyone on the call</p>
				</div>
			</div>

			<div
				ref={scrollRef}
				className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 py-4"
			>
				{displayMessages.length === 0 ? (
					<div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm text-white/60">
						<MessageSquareDashed className="h-9 w-9 text-white/25" aria-hidden="true" />
						<span>No messages yet -- say hi.</span>
					</div>
				) : (
					displayMessages.map((message, index) => {
						const isLocal = message.uid === localUid;
						const isAgent = message.uid === AGENT_CHAT_UID;
						const label = isAgent ? message.name : isLocal ? "You" : message.name;
						const highlighted = isAgent || isLocal;
						return (
							<article
								key={index}
								className={`flex items-end gap-2 ${isLocal ? "flex-row-reverse" : "flex-row"}`}
							>
								<div
									className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
										highlighted ? "bg-primary/20 text-primary" : "bg-muted text-foreground"
									}`}
									aria-hidden="true"
								>
									{isAgent ? <Sparkles className="h-3.5 w-3.5" /> : getInitial(label)}
								</div>
								<div className={`flex max-w-[80%] flex-col ${isLocal ? "items-end" : "items-start"}`}>
									<div className="mb-1 flex items-center gap-2 px-1 text-xs font-semibold text-white/60">
										<span>{label}</span>
										<span className="font-normal">{formatMessageTime(message.timestamp)}</span>
									</div>
									<div
										className={`whitespace-pre-wrap rounded-xl border px-3 py-2 text-sm leading-6 ${
											highlighted
												? "border-primary/25 bg-primary/10 text-white"
												: "border-border bg-foreground text-background"
										}`}
									>
										{message.text}
									</div>
								</div>
							</article>
						);
					})
				)}
			</div>

			<form
				className="flex shrink-0 items-center gap-2 border-t border-border p-3"
				onSubmit={(e) => {
					e.preventDefault();
					handleSend();
				}}
			>
				<input
					type="text"
					value={draft}
					onChange={(e) => setDraft(e.target.value)}
					placeholder="Type a message"
					maxLength={500}
					aria-label="Chat message"
					className="h-9 flex-1 rounded-lg border border-border bg-transparent px-3 text-sm text-white placeholder:text-white/40 focus:border-primary focus:outline-none"
				/>
				<Button
					type="submit"
					size="sm"
					disabled={!draft.trim() || sending}
					aria-label="Send message"
					className="h-9 w-9 shrink-0 rounded-lg p-0"
				>
					<Send className="h-4 w-4" />
				</Button>
			</form>
		</section>
	);
}
