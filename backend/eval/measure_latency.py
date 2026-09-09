"""
Real per-call latency, measured against the actual live Gemini API using
the exact production function (generate_structuring_update), on the same
real transcripts the accuracy harness uses -- not a synthetic benchmark.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.iCall.iCall_schema import ChatMessage
from app.iCall.iCall_utils import generate_structuring_update
from fixtures import FIXTURES


async def main():
    latencies = []
    for fixture in FIXTURES:
        messages = []
        state = {}
        for turn in fixture["turns"]:
            messages.append(ChatMessage(role="user", content=turn["text"]))
            start = time.monotonic()
            result = await generate_structuring_update(messages, state)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            print(f"  {elapsed:.2f}s  call_id={fixture['call_id']}  {turn['text'][:50]!r}")
            if result.update:
                # roughly mirror real state growth turn to turn
                state = {**state, "facts": (state.get("facts", []) + result.update.facts)}

    latencies.sort()
    n = len(latencies)
    print(f"\n{'='*60}")
    print(f"n={n}  min={min(latencies):.2f}s  max={max(latencies):.2f}s  "
          f"mean={sum(latencies)/n:.2f}s  median={latencies[n//2]:.2f}s  "
          f"p90={latencies[int(n*0.9)]:.2f}s")

asyncio.run(main())
