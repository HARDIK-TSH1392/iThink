"use client";

import { ClipboardList, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { getRecap } from "@/services/api";

type LiveRecapPanelProps = {
	channelName: string;
};

// Polling interval for the live timeline -- the brief's "a continuously
// updated incident timeline" only ever existed server-side before this
// panel (queryable via curl, never shown to anyone actually on the call).
// Polled rather than pushed over RTM: this is a plain-text rendering of
// already-extracted structured_state (see iCall_utils.format_live_recap),
// so a short poll is far simpler than adding another RTM message type for
// content that changes at most once per turn.
const POLL_INTERVAL_MS = 7_000;

export function LiveRecapPanel({ channelName }: LiveRecapPanelProps) {
	const [recap, setRecap] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		let cancelled = false;

		const fetchRecap = async () => {
			try {
				const result = await getRecap(channelName);
				if (!cancelled) {
					setRecap(result.recap);
					setLoading(false);
				}
			} catch {
				// Best-effort, same reasoning as everywhere else this call polls
				// the backend -- a slow/unreachable backend shouldn't crash the
				// panel, just leave it showing whatever it last had.
				if (!cancelled) setLoading(false);
			}
		};

		fetchRecap();
		const interval = setInterval(fetchRecap, POLL_INTERVAL_MS);
		return () => {
			cancelled = true;
			clearInterval(interval);
		};
	}, [channelName]);

	return (
		<section
			className="flex h-full min-h-0 w-full flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#141414] shadow-[0_12px_32px_rgba(0,0,0,0.4)]"
			aria-label="Incident timeline"
		>
			<div className="flex h-14 shrink-0 items-center gap-3 border-b border-white/10 bg-white/[0.03] px-4">
				<div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
					<ClipboardList className="h-4 w-4" />
				</div>
				<div>
					<h2 className="text-sm font-semibold text-white">Incident timeline</h2>
					<p className="text-xs text-white/60">
						Facts, decisions &amp; action items Watcher has recorded so far
					</p>
				</div>
			</div>

			{loading ? (
				<div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm text-white/60">
					<RefreshCw className="h-9 w-9 animate-spin text-white/25" aria-hidden="true" />
					<span>Loading...</span>
				</div>
			) : !recap ? (
				<div className="flex h-full flex-col items-center justify-center gap-3 px-4 text-center text-sm text-white/60">
					<ClipboardList className="h-9 w-9 text-white/25" aria-hidden="true" />
					<span>Nothing recorded yet -- this fills in as the call progresses.</span>
				</div>
			) : (
				<div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
					<pre className="whitespace-pre-wrap break-words font-sans text-sm text-white">
						{recap}
					</pre>
				</div>
			)}
		</section>
	);
}
