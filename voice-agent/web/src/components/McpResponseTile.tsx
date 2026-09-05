"use client";

import { GitCommitHorizontal, ScrollText, Wrench } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { SharedScreen } from "@/services/api";

type McpResponseTileProps = {
	/** Most recent GitHub/logs/MCP tool result, or null if nothing's been looked up yet this call. */
	screen: SharedScreen | null;
};

function formatTimestamp(iso: string) {
	return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date(iso));
}

function TileIcon({ type }: { type: SharedScreen["type"] }) {
	if (type === "github_commits") return <GitCommitHorizontal className="h-4 w-4" />;
	if (type === "logs") return <ScrollText className="h-4 w-4" />;
	return <Wrench className="h-4 w-4" />;
}

// Flattens whichever shape (commits/logs/raw tool text) into one block of
// plain lines, so the whole thing can stream in as continuous text the same
// way an LLM response streams -- rather than structured list items just
// appearing all at once.
function flattenScreenText(screen: SharedScreen): string {
	if (screen.type === "github_commits") {
		if (screen.commits.length === 0) return "No commits found.";
		return screen.commits
			.slice(0, 3)
			.map((commit) => `${commit.author}: ${commit.message}`)
			.join("\n");
	}
	if (screen.type === "logs") {
		if (screen.logs.length === 0) return "No log entries found.";
		return screen.logs
			.slice(0, 3)
			.map((log) => `[${log.severity.toUpperCase()}] ${log.message}`)
			.join("\n");
	}
	return screen.text;
}

const WORD_INTERVAL_MS = 45;

/**
 * Reveals `text` word by word (newlines and spacing preserved exactly),
 * restarting only when `streamKey` changes -- so a genuinely new lookup
 * re-streams from the top, but re-renders from unrelated state changes
 * don't reset an already-finished (or in-progress) reveal.
 */
function useWordStream(text: string, streamKey: string) {
	// Captures word+trailing-whitespace pairs together (the capturing group
	// keeps separators in the split result) so revealing two tokens at a
	// time -- one word, one separator -- reconstructs the original spacing
	// and line breaks exactly, instead of collapsing them like a naive
	// split(" ") would.
	const tokens = useMemo(() => text.split(/(\s+)/), [text]);
	const [revealCount, setRevealCount] = useState(0);
	const previousKey = useRef<string | null>(null);

	useEffect(() => {
		if (previousKey.current === streamKey) return;
		previousKey.current = streamKey;
		setRevealCount(0);
	}, [streamKey]);

	useEffect(() => {
		if (revealCount >= tokens.length) return;
		const id = setTimeout(() => setRevealCount((count) => Math.min(count + 2, tokens.length)), WORD_INTERVAL_MS);
		return () => clearTimeout(id);
	}, [tokens, revealCount]);

	return tokens.slice(0, revealCount).join("");
}

/**
 * Lives directly in the participant grid, alongside the You/Watcher/other-
 * participant tiles -- always visible, no tab to open or switch to. Shows
 * whatever the most recent GitHub/logs/MCP tool lookup turned up, updated
 * live as new ones come in (same sharedScreens state ConversationComponent
 * already tracks for late-joiner catch-up + the RTM broadcast listener),
 * streaming in word by word rather than popping in all at once.
 */
export function McpResponseTile({ screen }: McpResponseTileProps) {
	const bodyText = screen ? flattenScreenText(screen) : "";
	const streamedBody = useWordStream(bodyText, screen?.timestamp ?? "");

	return (
		<div className="col-span-2 flex max-h-64 flex-col gap-2 overflow-hidden rounded-2xl border border-primary/20 bg-primary/[0.04] px-5 py-4 sm:col-span-1">
			<div className="flex shrink-0 items-center gap-2 text-primary">
				{screen ? <TileIcon type={screen.type} /> : <GitCommitHorizontal className="h-4 w-4 opacity-40" />}
				<span className="text-xs font-semibold uppercase tracking-wide">MCP lookups</span>
				{screen ? (
					<span className="ml-auto text-[10px] font-normal text-muted-foreground">
						{formatTimestamp(screen.timestamp)}
					</span>
				) : null}
			</div>
			{screen ? (
				<div className="flex min-h-0 flex-1 flex-col gap-2">
					<p className="shrink-0 break-words text-xs font-medium text-foreground">{screen.title}</p>
					<p className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap break-words text-xs leading-5 text-foreground">
						{streamedBody}
					</p>
				</div>
			) : (
				<p className="text-xs text-muted-foreground">Nothing looked up yet -- ask Watcher about commits or logs.</p>
			)}
		</div>
	);
}
