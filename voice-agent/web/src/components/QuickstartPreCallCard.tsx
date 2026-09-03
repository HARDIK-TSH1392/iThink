"use client";

import { Loader2 } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";

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

	return (
		<div
			className="mx-auto flex w-[min(92vw,26.25rem)] animate-fade-up flex-col items-center rounded-[20px] border border-[#2b2b2b] px-10 py-10 text-center shadow-[0_10px_24px_rgba(0,0,0,0.28)]"
			style={{
				backgroundImage:
					"linear-gradient(164.988deg, rgba(54,54,54,0.2) 1.0596%, rgba(0,0,0,0) 96.089%), linear-gradient(90deg, rgb(16,16,16) 0%, rgb(16,16,16) 100%)",
			}}
		>
			<h1 className="text-[28px] font-medium leading-[1.2] text-white">
				{label ? `Join ${label}` : "iThink Incident Room"}
			</h1>
			<p className="mt-[14px] text-sm font-medium leading-6 text-muted-foreground">
				{label
					? "iThink will join the call, keep a live record of facts and decisions, and flag anything unclear."
					: "iThink's AI incident commander, powered by Agora Conversational AI."}
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
