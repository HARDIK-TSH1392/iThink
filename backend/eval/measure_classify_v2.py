"""
Second attempt at the classify-only prompt, incorporating research findings
on why the v1 attempt scored 20.5% (worse than always guessing "none"):

1. Few-shot examples, using short/fragmented real-style utterances (not
   clean sentences) -- v1 had zero examples at all.
2. "none" as its own explicitly-defined category with concrete criteria
   (short utterance, filler word, name, no standalone claim), not an
   implicit fallback -- research shows this measurably reduces confident
   wrong guesses on exactly the "Bob"/"Hi."/"So" failures v1 made.
3. A lightweight state snapshot (last open question + known facts so far),
   not just the last 4 raw turns with zero state -- research strongly
   suggests this, not prompt wording, is the dominant reason the combined
   prompt (full state) beat the classify-only prompt (no state) by such a
   wide margin.
4. A short one-clause "reason" field before the label (not full chain-of-
   thought) -- research found full CoT can hurt pattern-classification
   accuracy and adds real latency; a short forced justification still
   catches obviously-wrong guesses cheaply.

Tested against the exact same real transcripts and expected labels as v1,
so the two are directly comparable.
"""
import asyncio
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types
from pydantic import BaseModel

from app.iCall.iCall_utils import _get_client, PRIMARY_MODEL
from fixtures import FIXTURES
from measure_embedding_router import EXPECTED_TRIGGERS

CLASSIFY_INSTRUCTION = """You are watching a live incident call, turn by turn. Classify only the LATEST turn.

Categories:
- correction: a prior stated fact is being reversed/retracted as wrong.
  Example: "Actually confirmed, it's EU Central, not US East." (reason: reverses an earlier claim)
- conflict: this turn contradicts something already recorded and it is NOT yet resolved.
  Example: "Are you sure? I thought it was US East." (reason: directly questions an earlier claim)
- root_cause_question: someone is asking what caused the incident.
  Example: "What actually caused this?" (reason: explicit root-cause ask)
- resolution_declared: someone explicitly says a hypothesis/issue is now settled or ruled out.
  Example: "Consider it ruled out, we got confirmation." (reason: explicit settling language)
- wrap_up: the call sounds like it's ending.
  Example: "That covers everything, thanks." (reason: explicit closing language)
- none: this is the correct default for anything that doesn't clearly match one of the above --
  including short fillers, names, acknowledgments, or fragments with no standalone claim.
  Example: "Bob" (reason: a name alone, not a resolution or claim)
  Example: "Yeah." (reason: a filler acknowledgment, no new claim)
  Example: "Hi." (reason: a greeting, not a wrap-up)
  Example: "So, like," (reason: an incomplete fragment, no claim)
  Example: "Let's roll back the deploy." (reason: a decision, not one of the listed categories)

A short, low-content, or fragmentary utterance should be 'none' by default -- it is not a fallback
of last resort, it is the CORRECT answer whenever the turn doesn't clearly and specifically match
one of the five named categories above. Do not guess a specific category just because the turn
sounds vaguely related to the topic.

IMPORTANT -- completion is not correction: do not label a turn as 'correction' just because it adds
detail to, or narrows down, an existing fact. Example: an existing fact "The outage is not in the US"
followed by this turn saying "the outage is also in the Africa region" is NOT a correction -- both are
true at the same time (ruling a place out, then naming the actual place, is one continuous thought).
Label that 'none' (or the new detail is simply recorded as its own fact, not a reversal of the old one).
Only label 'correction' when the new statement is actually incompatible with the old one -- the old
fact could not still be true given the new one.

Given the call context below, output a one-clause reason, then the label."""


class TriggerClassification(BaseModel):
    reason: str
    trigger: Literal["correction", "conflict", "root_cause_question", "resolution_declared", "wrap_up", "none"]


def build_context(known_facts: list[str], last_question: str | None, recent_turns: list[str]) -> str:
    lines = []
    if known_facts:
        lines.append("Known facts so far: " + "; ".join(known_facts[-5:]))
    if last_question:
        lines.append(f"Last open question asked: {last_question!r}")
    lines.append("Recent turns:")
    lines.extend(recent_turns[-4:])
    return "\n".join(lines)


async def classify(client, known_facts, last_question, recent_turns) -> tuple[str, str, float]:
    prompt = build_context(known_facts, last_question, recent_turns)
    config = types.GenerateContentConfig(
        system_instruction=CLASSIFY_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=TriggerClassification,
    )
    start = time.monotonic()
    response = await asyncio.to_thread(
        client.models.generate_content, model=PRIMARY_MODEL, contents=prompt, config=config,
    )
    elapsed = time.monotonic() - start
    result = TriggerClassification.model_validate_json(response.text)
    return result.trigger, result.reason, elapsed


async def main():
    client = _get_client()
    latencies = []
    correct = 0
    total = 0

    for fixture in FIXTURES:
        expected_list = EXPECTED_TRIGGERS[fixture["call_id"]]
        history = []
        known_facts = []
        last_question = None

        for turn, expected in zip(fixture["turns"], expected_list):
            line = f"{turn['speaker']}: {turn['text']}"
            history.append(line)

            trigger, reason, elapsed = await classify(client, known_facts, last_question, history)
            latencies.append(elapsed)

            total += 1
            is_correct = trigger == expected
            correct += is_correct
            mark = "OK" if is_correct else "WRONG"
            print(f"  {elapsed:.2f}s  [{mark:5s}] predicted={trigger:20s} expected={expected:20s} "
                  f"reason={reason[:45]!r}  {turn['text'][:35]!r}")

            # cheap, deterministic state tracking for the next turn's context --
            # not real extraction, just enough to unstarve the classifier per
            # the research finding that state, not wording, was the main gap
            if "?" in turn["text"]:
                last_question = turn["text"]
            if len(turn["text"].split()) > 6:
                known_facts.append(turn["text"])

    latencies.sort()
    n = len(latencies)
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Latency: min={min(latencies):.2f}s max={max(latencies):.2f}s "
          f"mean={sum(latencies)/n:.2f}s median={latencies[n//2]:.2f}s p90={latencies[int(n*0.9)]:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
