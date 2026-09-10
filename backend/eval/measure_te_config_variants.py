"""
Real A/B/C latency test for two candidate Thinking Engine config changes
(unset max_output_tokens -> capped, and unset thinking_config -> explicit
budget=0), against the live Gemini API using the exact production prompt
builder and schema -- not a synthetic benchmark, not a guess.

Scratch/one-off: does not touch iCall_utils.py, just reuses its prompt
builder and client so the only thing varying between arms is the
GenerateContentConfig itself.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types
from app.iCall.iCall_schema import ChatMessage, StructuringUpdate
from app.iCall.iCall_utils import (
    _get_client, _build_structuring_prompt, STRUCTURING_SYSTEM_INSTRUCTION, PRIMARY_MODEL,
)
from fixtures import FIXTURES


def build_config(variant: str) -> types.GenerateContentConfig:
    kwargs = dict(
        system_instruction=STRUCTURING_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=StructuringUpdate,
    )
    if variant == "capped_tokens":
        kwargs["max_output_tokens"] = 512
    elif variant == "thinking_budget_0":
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    elif variant == "both":
        kwargs["max_output_tokens"] = 512
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


async def run_variant(variant: str):
    client = _get_client()
    latencies = []
    truncated = 0
    for fixture in FIXTURES:
        messages = []
        state = {}
        for turn in fixture["turns"]:
            messages.append(ChatMessage(role="user", content=turn["text"]))
            prompt = _build_structuring_prompt(messages, state, None, None, service=None, region=None, incident_age_minutes=None)
            config = build_config(variant)
            start = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(client.models.generate_content, model=PRIMARY_MODEL, contents=prompt, config=config),
                    timeout=30,
                )
                elapsed = time.monotonic() - start
                latencies.append(elapsed)
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                if str(finish_reason) not in ("FinishReason.STOP", "1", "STOP"):
                    truncated += 1
                    print(f"    [{variant}] finish_reason={finish_reason} (possible truncation) turn={turn['text'][:40]!r}")
                parsed = response.parsed
                if parsed:
                    state = {**state, "facts": (state.get("facts", []) + parsed.facts)}
            except Exception as exc:
                print(f"    [{variant}] ERROR: {exc}")
    n = len(latencies)
    if n == 0:
        print(f"{variant}: no successful calls")
        return
    latencies.sort()
    print(
        f"{variant:20s} n={n} min={min(latencies):.2f}s max={max(latencies):.2f}s "
        f"mean={sum(latencies)/n:.2f}s median={latencies[n//2]:.2f}s "
        f"p90={latencies[int(n*0.9)]:.2f}s truncated={truncated}"
    )


async def main():
    for variant in ["baseline", "capped_tokens"]:
        await run_variant(variant)


asyncio.run(main())
