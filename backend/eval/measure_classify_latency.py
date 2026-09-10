"""
Real, measured latency of a genuinely minimal classify-only Gemini call --
tiny prompt (recent turns only, no full structured_state/fact-reconciliation
context), tiny output (one enum field) -- against the exact same real
transcript turns the combined-call benchmark used, so the two numbers are
directly comparable.

This is not a design decision, it's the missing data point the design
decision depends on: how much latency is actually available to save by
splitting classify from extract, measured, not assumed.
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

CLASSIFY_INSTRUCTION = (
    "You are watching a live incident call. Given only the most recent "
    "turns, classify the LATEST turn as exactly one of: 'correction' (a "
    "prior fact is being reversed/retracted), 'conflict' (this turn "
    "contradicts something said earlier and it's unresolved), "
    "'root_cause_question' (someone is fishing for what caused the "
    "incident), 'resolution_declared' (someone explicitly says a "
    "hypothesis/issue is now settled/ruled out), 'wrap_up' (the call "
    "sounds like it's ending), or 'none' if it's none of these. Output "
    "only the classification, do not explain."
)


class TriggerClassification(BaseModel):
    trigger: Literal["correction", "conflict", "root_cause_question", "resolution_declared", "wrap_up", "none"]


async def classify(client, recent_turns: list[str]) -> tuple[str, float]:
    prompt = "\n".join(recent_turns[-4:])  # only last few turns -- classify doesn't need full history
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
    return result.trigger, elapsed


async def main():
    client = _get_client()
    latencies = []
    for fixture in FIXTURES:
        history = []
        for turn in fixture["turns"]:
            history.append(f"{turn['speaker']}: {turn['text']}")
            trigger, elapsed = await classify(client, history)
            latencies.append(elapsed)
            print(f"  {elapsed:.2f}s  [{trigger:20s}]  {turn['text'][:45]!r}")

    latencies.sort()
    n = len(latencies)
    print(f"\n{'='*60}")
    print(f"n={n}  min={min(latencies):.2f}s  max={max(latencies):.2f}s  "
          f"mean={sum(latencies)/n:.2f}s  median={latencies[n//2]:.2f}s  "
          f"p90={latencies[int(n*0.9)]:.2f}s")

asyncio.run(main())
