"use client";

import { GitCommitHorizontal, ScrollText, Wrench } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { SharedScreen } from "@/services/api";

type McpResponseTileProps = {
	/** Every GitHub/logs/MCP tool result from this call so far, oldest first. */
	screens: SharedScreen[];
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
// plain lines, so it can stream in as continuous text the same way an LLM
// response streams -- rather than structured list items just appearing all
// at once.
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
 * restarting only when `streamKey` changes. Combined with a stable `key`
 * per entry in the list below, each entry only ever streams once, the
 * first time it's rendered -- later entries being appended doesn't
 * remount (or re-animate) the ones already shown.
 */
function useWordStream(text: string, streamKey: string) {
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

function HistoryEntry({ screen }: { screen: SharedScreen }) {
	const streamedBody = useWordStream(flattenScreenText(screen), screen.timestamp);

	return (
		<div className="flex shrink-0 flex-col gap-1 border-b border-primary/10 pb-2 last:border-b-0 last:pb-0">
			<div className="flex items-center gap-1.5 text-primary/80">
				<TileIcon type={screen.type} />
				<span className="truncate text-[11px] font-medium text-foreground">{screen.title}</span>
				<span className="ml-auto shrink-0 text-[10px] font-normal text-muted-foreground">
					{formatTimestamp(screen.timestamp)}
				</span>
			</div>
			<p className="whitespace-pre-wrap break-words text-xs leading-5 text-foreground">{streamedBody}</p>
		</div>
	);
}

// Grows smoothly as more lookups come in (rather than snapping to a new
// size), up to a cap -- beyond that it scrolls internally instead of
// pushing the rest of the call UI around indefinitely.
const BASE_HEIGHT_PX = 130;
const HEIGHT_PER_ENTRY_PX = 76;
const MAX_HEIGHT_PX = 420;

/**
 * Lives directly in the participant grid, alongside the You/Watcher/other-
 * participant tiles -- always visible, no tab to open or switch to. Keeps
 * every GitHub/logs/MCP lookup from the call, not just the latest one, so
 * asking a second question doesn't make the first answer disappear.
 */
export function McpResponseTile({ screens }: McpResponseTileProps) {
	const scrollRef = useRef<HTMLDivElement>(null);
	const targetHeight = Math.min(BASE_HEIGHT_PX + screens.length * HEIGHT_PER_ENTRY_PX, MAX_HEIGHT_PX);

	useEffect(() => {
		const node = scrollRef.current;
		if (!node) return;
		node.scrollTop = node.scrollHeight;
	}, [screens.length]);

	return (
		<div
			className="col-span-2 flex flex-col gap-2 overflow-hidden rounded-2xl border border-primary/20 bg-primary/[0.04] px-5 py-4 transition-[height] duration-300 ease-in-out sm:col-span-1"
			style={{ height: targetHeight }}
		>
			<div className="flex shrink-0 items-center gap-2 text-primary">
				<GitCommitHorizontal className={`h-4 w-4 ${screens.length === 0 ? "opacity-40" : ""}`} />
				<span className="text-xs font-semibold uppercase tracking-wide">MCP lookups</span>
			</div>
			{screens.length > 0 ? (
				<div ref={scrollRef} className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
					{screens.map((screen) => (
						<HistoryEntry key={screen.timestamp} screen={screen} />
					))}
				</div>
			) : (
				<p className="text-xs text-muted-foreground">Nothing looked up yet -- ask Watcher about commits or logs.</p>
			)}
		</div>
	);
}
