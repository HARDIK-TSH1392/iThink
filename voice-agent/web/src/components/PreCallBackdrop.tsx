"use client";

import {
	Activity,
	AlertTriangle,
	Bell,
	ClipboardCheck,
	GitBranch,
	Headphones,
	MessageSquare,
	Radio,
	Server,
	ShieldCheck,
	Siren,
	Terminal,
} from "lucide-react";
import { useEffect, useRef } from "react";

const ICONS = [
	AlertTriangle,
	Activity,
	ShieldCheck,
	Headphones,
	ClipboardCheck,
	Terminal,
	GitBranch,
	Bell,
	Radio,
	Siren,
	Server,
	MessageSquare,
];

// Deterministic pseudo-random (mulberry32) rather than Math.random() --
// server-rendered HTML and the client's first render must produce
// identical output, or Next.js flags a hydration mismatch. A fixed seed
// means "random-looking" but exactly reproducible every render.
function mulberry32(seed: number) {
	let a = seed;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const SCATTER_COUNT = 89;
const REPEL_RADIUS = 130; // px
const REPEL_STRENGTH = 55; // px, max push at zero distance

const random = mulberry32(1993);
const SCATTERED_ICONS = Array.from({ length: SCATTER_COUNT }, (_, i) => {
	const Icon = ICONS[i % ICONS.length];
	return {
		Icon,
		key: i,
		topPct: 4 + random() * 90,
		size: 16 + random() * 20,
		opacity: 0.06 + random() * 0.09,
		duration: 26 + random() * 34, // s, full right-to-left crossing
		// Fraction of the drift cycle already elapsed at t=0, so icons start
		// scattered mid-drift instead of all bunched at the right edge.
		phase: random(),
	};
});

export function PreCallBackdrop() {
	const containerRef = useRef<HTMLDivElement>(null);
	const iconRefs = useRef<(HTMLDivElement | null)[]>([]);
	const mouseRef = useRef<{ x: number; y: number } | null>(null);
	const reducedMotionRef = useRef(false);

	useEffect(() => {
		reducedMotionRef.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

		const handleMouseMove = (e: MouseEvent) => {
			mouseRef.current = { x: e.clientX, y: e.clientY };
		};
		const handleMouseLeave = () => {
			mouseRef.current = null;
		};
		window.addEventListener("mousemove", handleMouseMove);
		window.addEventListener("mouseleave", handleMouseLeave);

		const startTime = performance.now();
		let frameId: number;

		const tick = (now: number) => {
			frameId = requestAnimationFrame(tick);

			const container = containerRef.current;
			if (!container) return;
			const rect = container.getBoundingClientRect();
			const elapsedSeconds = (now - startTime) / 1000;
			const driftDistance = rect.width * 1.6;
			const mouse = mouseRef.current;

			SCATTERED_ICONS.forEach((cfg, i) => {
				const el = iconRefs.current[i];
				if (!el) return;

				// Base drift position (right edge -> fully off-screen left),
				// looping seamlessly via modulo.
				const progress = reducedMotionRef.current
					? cfg.phase
					: ((elapsedSeconds / cfg.duration + cfg.phase) % 1);
				// The element's untransformed position is the container's right
				// edge (it's laid out with `right-0`), so this is the delta from
				// there, not an absolute page position.
				const driftX = -progress * driftDistance;
				// Absolute on-screen position, only for measuring distance to
				// the cursor -- not written back to the DOM directly.
				const baseX = rect.right + driftX;
				const baseY = rect.top + (cfg.topPct / 100) * rect.height;

				let offsetX = 0;
				let offsetY = 0;
				if (mouse) {
					const dx = baseX - mouse.x;
					const dy = baseY - mouse.y;
					const dist = Math.hypot(dx, dy);
					if (dist < REPEL_RADIUS && dist > 0.01) {
						const falloff = 1 - dist / REPEL_RADIUS;
						const push = REPEL_STRENGTH * falloff * falloff;
						offsetX = (dx / dist) * push;
						offsetY = (dy / dist) * push;
					}
				}

				el.style.transform = `translate3d(${driftX + offsetX}px, ${offsetY}px, 0)`;
			});
		};

		frameId = requestAnimationFrame(tick);
		return () => {
			cancelAnimationFrame(frameId);
			window.removeEventListener("mousemove", handleMouseMove);
			window.removeEventListener("mouseleave", handleMouseLeave);
		};
	}, []);

	return (
		<div
			ref={containerRef}
			className="pointer-events-none absolute inset-0 overflow-hidden"
			aria-hidden="true"
		>
			{/* Tutti-frutti color wash -- heavily blurred, low opacity so it stays
			    an ambient glow rather than competing with the card. */}
			<div className="absolute -left-24 -top-24 h-72 w-72 rounded-full bg-pink-500/25 blur-[100px]" />
			<div className="absolute -right-16 top-10 h-64 w-64 rounded-full bg-amber-400/20 blur-[100px]" />
			<div className="absolute bottom-0 left-1/4 h-80 w-80 rounded-full bg-lime-400/15 blur-[110px]" />
			<div className="absolute -bottom-20 -right-20 h-72 w-72 rounded-full bg-sky-400/20 blur-[100px]" />
			<div className="absolute right-1/3 top-1/2 h-56 w-56 -translate-y-1/2 rounded-full bg-fuchsia-500/15 blur-[100px]" />

			{/* Grey professional icons: drift right to left, and get pushed
			    away from wherever the cursor currently is. */}
			{SCATTERED_ICONS.map(({ Icon, key, topPct, size, opacity }, i) => (
				<div
					key={key}
					ref={(el) => {
						iconRefs.current[i] = el;
					}}
					className="absolute right-0 will-change-transform"
					style={{ top: `${topPct}%` }}
				>
					<Icon size={size} strokeWidth={1.5} className="text-foreground" style={{ opacity }} />
				</div>
			))}
		</div>
	);
}
