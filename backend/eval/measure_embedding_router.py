"""
Real embedding-based router, tested against the exact same real transcript
turns and expected labels the LLM-based harnesses used -- built and
validated with real data now that we have it, not deferred on the
"unvalidated" objection.

Uses Gemini's own embedding model (gemini-embedding-001) via the same
client/auth already configured -- no new vendor, no new infra beyond one
more Gemini API call type.

Measures BOTH latency (is it actually near-instant) and accuracy (does it
correctly separate our real categories) so the decision is made on
evidence, not on the industry number from someone else's domain.
"""
import asyncio
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.iCall.iCall_utils import _get_client
from fixtures import FIXTURES

EMBED_MODEL = "gemini-embedding-001"

EXAMPLES = {
    "correction": [
        "Actually confirmed, it's in the AC region, not US East.",
        "My mistake, it's actually the EU region.",
        "Correction: it's not the database, it's the API layer.",
        "Scratch that, it's actually a network issue.",
        "It's confirmed the outage is in EU Central, not US East.",
    ],
    "conflict": [
        "Are you sure? I thought it was US East.",
        "Wait, that contradicts what was said earlier.",
        "Hold on, didn't we already say it was resolved?",
        "That doesn't match what Sarah said five minutes ago.",
    ],
    "root_cause_question": [
        "What caused this outage?",
        "Why did this happen in the first place?",
        "Do we know the root cause yet?",
        "What's actually causing the errors?",
    ],
    "resolution_declared": [
        "Consider it resolved, the database team confirmed it's fine.",
        "That's been ruled out, it's not the database.",
        "We've confirmed this is fixed now.",
        "For now, consider it ruled out, we got confirmation.",
    ],
    "wrap_up": [
        "I think that covers everything, thanks.",
        "Let's wrap this up for now.",
        "Okay, let's reconvene in thirty minutes.",
        "That's all for now, thanks everyone.",
    ],
}

# Ground truth per turn, in the same order as fixtures.py -- derived from
# the same human labels used in fixtures.py's `expect` dicts (not a
# separate, looser labeling standard).
EXPECTED_TRIGGERS = {
    21: ["none", "none", "none", "none", "conflict", "conflict", "correction",
         "none", "none", "wrap_up"],
    32: ["none", "none", "none", "none", "conflict", "resolution_declared", "none",
         "none", "correction", "none", "none", "none", "none", "none",
         "none", "wrap_up"],
    19: ["none", "none", "none", "none", "none", "none", "none", "none",
         "none", "correction", "none", "none", "none"],
}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def embed(client, texts: list[str]) -> list[list[float]]:
    response = await asyncio.to_thread(client.models.embed_content, model=EMBED_MODEL, contents=texts)
    return [e.values for e in response.embeddings]


async def main():
    client = _get_client()

    print("Pre-encoding example phrases...")
    category_embeddings = {}
    for category, phrases in EXAMPLES.items():
        category_embeddings[category] = await embed(client, phrases)

    THRESHOLD = 0.60
    latencies = []
    correct = 0
    total = 0

    for fixture in FIXTURES:
        expected_list = EXPECTED_TRIGGERS[fixture["call_id"]]
        for turn, expected in zip(fixture["turns"], expected_list):
            start = time.monotonic()
            [turn_vec] = await embed(client, [turn["text"]])
            elapsed = time.monotonic() - start
            latencies.append(elapsed)

            best_category, best_score = "none", THRESHOLD
            for category, vecs in category_embeddings.items():
                score = max(cosine(turn_vec, v) for v in vecs)
                if score > best_score:
                    best_category, best_score = category, score

            total += 1
            is_correct = best_category == expected
            correct += is_correct
            mark = "OK" if is_correct else "WRONG"
            print(f"  {elapsed:.3f}s  [{mark:5s}] predicted={best_category:20s} expected={expected:20s} "
                  f"(score={best_score:.2f})  {turn['text'][:40]!r}")

    latencies.sort()
    n = len(latencies)
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Latency: min={min(latencies):.3f}s max={max(latencies):.3f}s "
          f"mean={sum(latencies)/n:.3f}s median={latencies[n//2]:.3f}s p90={latencies[int(n*0.9)]:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
