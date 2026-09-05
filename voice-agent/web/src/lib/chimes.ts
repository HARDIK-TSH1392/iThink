// Synthesized via the Web Audio API rather than audio files -- nothing to
// fetch, host, or bundle, and each chime stays a few lines of code instead
// of a binary in the repo. Shares one AudioContext across every chime
// (creating a fresh one per call is wasteful and some browsers cap how
// many can exist at once).

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

type ChimeNote = { freq: number; startOffset: number };

function playTones(notes: ChimeNote[], waveform: OscillatorType, noteDurationS: number, peakGain: number) {
	const ctx = getAudioContext();
	if (!ctx) return;

	try {
		if (ctx.state === "suspended") {
			void ctx.resume();
		}

		const now = ctx.currentTime;
		for (const { freq, startOffset } of notes) {
			const oscillator = ctx.createOscillator();
			const gainNode = ctx.createGain();
			oscillator.type = waveform;
			oscillator.frequency.value = freq;

			const noteStart = now + startOffset;
			const noteEnd = noteStart + noteDurationS;

			// Quick fade-in then an exponential decay -- avoids the audible
			// click a hard on/off transition produces, and reads as "soft"
			// rather than a harsh beep.
			gainNode.gain.setValueAtTime(0, noteStart);
			gainNode.gain.linearRampToValueAtTime(peakGain, noteStart + 0.02);
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

const JOIN_NOTES: ChimeNote[] = [
	{ freq: 660, startOffset: 0 },
	{ freq: 880, startOffset: 0.12 },
];

/** Plays a brief, mild join chime -- a soft, mellow two-note rise. */
export function playJoinChime() {
	playTones(JOIN_NOTES, "sine", 0.22, 0.12);
}

// Deliberately a different timbre (triangle, brighter/more textured than
// the join chime's sine) and a different melodic shape (three quick
// ascending notes vs. two slower ones) -- easy to tell apart by ear from
// the join chime without either one being harsh or jarring.
const HAND_RAISE_NOTES: ChimeNote[] = [
	{ freq: 784, startOffset: 0 }, // G5
	{ freq: 988, startOffset: 0.09 }, // B5
	{ freq: 1175, startOffset: 0.18 }, // D6
];

/** Plays a brief, mild "hand raised" chime -- a bright, quick three-note rise. */
export function playHandRaiseChime() {
	playTones(HAND_RAISE_NOTES, "triangle", 0.16, 0.1);
}
