"use client";

import { GitCommitHorizontal, ScrollText, ScreenShareOff } from "lucide-react";
import { useState } from "react";

import type { SharedScreen } from "@/services/api";

type SharedScreenPanelProps = {
	screens: SharedScreen[];
};

function formatTimestamp(iso: string) {
	return new Intl.DateTimeFormat(undefined, {
		hour: "numeric",
		minute: "2-digit",
		month: "short",
		day: "numeric",
	}).format(new Date(iso));
}

const SEVERITY_STYLES: Record<string, string> = {
	prod_down: "border-red-500/30 bg-red-500/10 text-red-400",
	critical: "border-red-500/30 bg-red-500/10 text-red-400",
	error: "border-amber-500/30 bg-amber-500/10 text-amber-400",
	warning: "border-yellow-500/30 bg-yellow-500/10 text-yellow-400",
};

function ScreenBody({ screen }: { screen: SharedScreen }) {
	if (screen.type === "github_commits") {
		if (screen.commits.length === 0) {
			return <p className="text-sm text-muted-foreground">No commits found.</p>;
		}
		return (
			<ul className="flex flex-col gap-2">
				{screen.commits.map((commit) => (
					<li
						key={commit.sha}
						className="flex flex-col gap-1 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2"
					>
						<div className="flex items-center justify-between gap-2">
							<span className="truncate text-sm font-medium text-foreground">{commit.message}</span>
							<code className="shrink-0 rounded bg-white/10 px-1.5 py-0.5 text-xs text-muted-foreground">
								{commit.sha}
							</code>
						</div>
						<div className="flex items-center gap-2 text-xs text-muted-foreground">
							<span>{commit.author}</span>
							<span aria-hidden="true">&middot;</span>
							<span>{formatTimestamp(commit.date)}</span>
						</div>
					</li>
				))}
			</ul>
		);
	}

	if (screen.logs.length === 0) {
		return <p className="text-sm text-muted-foreground">No log entries found since the incident started.</p>;
	}
	return (
		<ul className="flex flex-col gap-2">
			{screen.logs.map((log) => (
				<li
					key={log.id}
					className="flex flex-col gap-1 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2"
				>
					<div className="flex items-center justify-between gap-2">
						<span
							className={`shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide ${
								SEVERITY_STYLES[log.severity] ?? "border-border bg-muted text-muted-foreground"
							}`}
						>
							{log.severity}
						</span>
						<span className="text-xs text-muted-foreground">{formatTimestamp(log.timestamp)}</span>
					</div>
					<span className="text-sm text-foreground">{log.message}</span>
				</li>
			))}
		</ul>
	);
}

export function SharedScreenPanel({ screens }: SharedScreenPanelProps) {
	const [selectedIndex, setSelectedIndex] = useState(0);
	const ordered = [...screens].reverse();
	const selected = ordered[Math.min(selectedIndex, ordered.length - 1)];

	return (
		<section
			className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#141414] shadow-[0_12px_32px_rgba(0,0,0,0.4)]"
			aria-label="Shared screens"
		>
			<div className="flex h-14 shrink-0 items-center gap-3 border-b border-white/10 bg-white/[0.03] px-4">
				<div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
					<GitCommitHorizontal className="h-4 w-4" />
				</div>
				<div>
					<h2 className="text-sm font-semibold text-foreground">Shared with everyone</h2>
					<p className="text-xs text-muted-foreground">GitHub commits &amp; server logs pulled up on this call</p>
				</div>
			</div>

			{ordered.length === 0 ? (
				<div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm text-muted-foreground">
					<ScreenShareOff className="h-9 w-9 text-muted-foreground/40" aria-hidden="true" />
					<span>Nothing shared yet -- ask Watcher to show commits or logs.</span>
				</div>
			) : (
				<div className="flex min-h-0 flex-1 flex-col">
					{ordered.length > 1 ? (
						<div className="flex shrink-0 gap-1 overflow-x-auto border-b border-white/10 px-3 py-2">
							{ordered.map((screen, index) => (
								<button
									key={`${screen.type}-${screen.timestamp}`}
									type="button"
									onClick={() => setSelectedIndex(index)}
									className={`flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
										index === selectedIndex
											? "border-primary/40 bg-primary/15 text-primary"
											: "border-white/10 text-muted-foreground hover:bg-white/5"
									}`}
								>
									{screen.type === "github_commits" ? (
										<GitCommitHorizontal className="h-3 w-3" />
									) : (
										<ScrollText className="h-3 w-3" />
									)}
									{screen.type === "github_commits" ? "Commits" : "Logs"}
								</button>
							))}
						</div>
					) : null}

					<div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 py-4">
						<div>
							<h3 className="text-sm font-semibold text-foreground">{selected.title}</h3>
							<p className="text-xs text-muted-foreground">{formatTimestamp(selected.timestamp)}</p>
						</div>
						<ScreenBody screen={selected} />
					</div>
				</div>
			)}
		</section>
	);
}
