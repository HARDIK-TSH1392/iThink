// Synthesized via the Web Audio API rather than an audio file -- no
// external asset to fetch, host, or bundle, and it stays a few lines of
// code instead of a binary in the repo. A short, soft two-note chime,
// gentle enough not to startle anyone mid-call.

let sharedAudioContext: AudioContext | null = null;

function getAudioContext(): AudioContext | null {
	if (typeof window === "undefined") return null;
	if (!sharedAudioContext) {
		const AudioContextCtor =
			window.AudioContext ||
			(window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
		if (!AudioContextCtor) return null;
		sharedAudioContext = new AudioContextCtor();
	}
	return sharedAudioContext;
}

const CHIME_NOTES: Array<{ freq: number; startOffset: number }> = [
	{ freq: 660, startOffset: 0 },
	{ freq: 880, startOffset: 0.12 },
];
const NOTE_DURATION_S = 0.22;
const PEAK_GAIN = 0.12;

/** Plays a brief, mild join chime. Safe to call from any event handler --
 * silently does nothing if Web Audio isn't available or the context is
 * blocked (never worth surfacing an error over a notification sound). */
export function playJoinChime() {
	const ctx = getAudioContext();
	if (!ctx) return;

	try {
		if (ctx.state === "suspended") {
			void ctx.resume();
		}

		const now = ctx.currentTime;
		for (const { freq, startOffset } of CHIME_NOTES) {
			const oscillator = ctx.createOscillator();
			const gainNode = ctx.createGain();
			oscillator.type = "sine";
			oscillator.frequency.value = freq;

			const noteStart = now + startOffset;
			const noteEnd = noteStart + NOTE_DURATION_S;

			// Quick fade-in then an exponential decay -- avoids the audible
			// click a hard on/off transition produces, and reads as "soft"
			// rather than a harsh beep.
			gainNode.gain.setValueAtTime(0, noteStart);
			gainNode.gain.linearRampToValueAtTime(PEAK_GAIN, noteStart + 0.02);
			gainNode.gain.exponentialRampToValueAtTime(0.0001, noteEnd);

			oscillator.connect(gainNode);
			gainNode.connect(ctx.destination);
			oscillator.start(noteStart);
			oscillator.stop(noteEnd + 0.02);
		}
	} catch {
		// Never let a notification sound break the call.
	}
}
