"use client";

import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { getNames } from "@/services/api";

type QuickstartPreCallCardProps = {
	isLoading: boolean;
	error: string | null;
	onStartConversation: (name: string) => void;
	channel?: string;
};

// Parses "incident-73" -> "73" for display; falls back to the raw channel
// name for anything that doesn't match that convention.
function incidentLabel(channel?: string): string | null {
	if (!channel) return null;
	const match = channel.match(/^incident-(\d+)$/);
	return match ? `Incident #${match[1]}` : channel;
}

export function QuickstartPreCallCard({
	isLoading,
	error,
	onStartConversation,
	channel,
}: QuickstartPreCallCardProps) {
	const label = incidentLabel(channel);
	const [name, setName] = useState("");
	const trimmedName = name.trim();

	// How many people have already joined this incident's call, polled from
	// the same name registry the transcript uses (setName/getNames) --
	// reflects who's actually shown up, not just who was invited. Note:
	// entries aren't removed when someone leaves, so this can overcount a
	// long-running channel where earlier joiners have already left.
	const [participantCount, setParticipantCount] = useState<number | null>(null);
	useEffect(() => {
		if (!channel) return;
		let cancelled = false;

		const refresh = () => {
			getNames(channel)
				.then((names) => {
					if (!cancelled) setParticipantCount(Object.keys(names).length);
				})
				.catch(() => {});
		};

		refresh();
		const interval = setInterval(refresh, 3000);
		return () => {
			cancelled = true;
			clearInterval(interval);
		};
	}, [channel]);

	return (
		<div
			className="relative mx-auto flex w-[min(92vw,26.25rem)] animate-fade-up flex-col items-center overflow-hidden rounded-[20px] border border-white/15 px-10 py-10 text-center shadow-[0_8px_32px_rgba(0,0,0,0.45)] backdrop-blur-2xl backdrop-saturate-150"
			style={{
				backgroundImage:
					"linear-gradient(164.988deg, rgba(255,255,255,0.08) 1.0596%, rgba(255,255,255,0) 60%), linear-gradient(90deg, rgba(16,16,16,0.45) 0%, rgba(16,16,16,0.45) 100%)",
			}}
		>
			{/* Glass edge highlight -- a thin light gradient along the top,
			    the detail that reads as "glass" rather than just "blurred". */}
			<div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/40 to-transparent" />
			<h1 className="text-[28px] font-medium leading-[1.2] text-white">
				{label ? `Join ${label}` : "iThink Incident Room"}
			</h1>
			<p className="mt-[14px] text-sm font-medium leading-6 text-muted-foreground">
				{label ? (
					participantCount === null ? (
						"Checking who's already on the call…"
					) : participantCount === 0 ? (
						"No one has joined yet — you'll be first."
					) : (
						`${participantCount} participant${participantCount === 1 ? "" : "s"} already on the call`
					)
				) : (
					"iThink's AI incident commander, powered by Agora Conversational AI."
				)}
			</p>

			<input
				type="text"
				value={name}
				onChange={(e) => setName(e.target.value)}
				placeholder="Your name"
				disabled={isLoading}
				maxLength={40}
				aria-label="Your name"
				className="mt-8 h-10 w-full rounded-lg border border-[#3a3a3a] bg-transparent px-3 text-sm text-white placeholder:text-muted-foreground focus:border-primary focus:outline-none"
			/>

			<Button
				onClick={() => onStartConversation(trimmedName)}
				disabled={isLoading || !trimmedName}
				className="mt-4 h-10 w-full rounded-lg border border-primary bg-primary text-sm font-medium text-black hover:border-white hover:bg-white hover:text-black disabled:hover:border-primary disabled:hover:bg-primary disabled:hover:text-black"
				aria-label={
					isLoading ? "Joining incident call" : "Join incident call"
				}
			>
				{isLoading ? (
					<>
						<Loader2 className="h-4 w-4 animate-spin" />
						Joining...
					</>
				) : (
					"Join Call"
				)}
			</Button>
			{error ? <p className="mt-3 text-xs text-destructive">{error}</p> : null}
		</div>
	);
}
