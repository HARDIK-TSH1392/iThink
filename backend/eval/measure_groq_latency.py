"""
Real, direct latency benchmark for Groq's openai/gpt-oss-120b, using the
EXACT same real transcripts, system instruction, and prompt-building logic
as the Gemini benchmark -- not a synthetic/toy prompt -- so the comparison
is genuinely apples-to-apples against our own measured Gemini numbers
(mean 1.27s, median 1.21s, p90 1.57s, max 2.21s).

Uses strict, schema-constrained JSON output (Groq's documented constrained-
decoding mode), matching the same guarantee google-genai's response_schema
gives us today.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.iCall.iCall_utils import _build_structuring_prompt, STRUCTURING_SYSTEM_INSTRUCTION
from app.iCall.iCall_schema import ChatMessage
from fixtures import FIXTURES

GROQ_ENV_PATH = "/home/obito/coding/aivatar/pilot-bot-v1/.env"


def load_groq_key() -> str:
    for line in open(GROQ_ENV_PATH):
        if line.startswith("GROQ_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("GROQ_API_KEY not found")


GROQ_API_KEY = load_groq_key()
GROQ_MODEL = "openai/gpt-oss-120b"

# Strict-mode-compliant JSON schema, hand-adapted from StructuringUpdate --
# Pydantic's auto-generated schema uses anyOf-with-null for optional fields,
# which Groq's strict constrained decoding does not accept as-is.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "structuring_update",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "facts": {"type": "array", "items": {"type": "string"}},
                "hypotheses": {"type": "array", "items": {"type": "string"}},
                "decisions": {"type": "array", "items": {"type": "string"}},
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}, "owner": {"type": ["string", "null"]}},
                        "required": ["text", "owner"],
                        "additionalProperties": False,
                    },
                },
                "missing_info": {"type": "array", "items": {"type": "string"}},
                "conflict": {"type": ["string", "null"]},
                "corrects_fact": {"type": ["string", "null"]},
                "is_wrapping_up": {"type": "boolean"},
                "spoken_reply": {"type": "string"},
            },
            "required": ["facts", "hypotheses", "decisions", "action_items", "missing_info",
                         "conflict", "corrects_fact", "is_wrapping_up", "spoken_reply"],
            "additionalProperties": False,
        },
    },
}


async def call_groq(client: httpx.AsyncClient, prompt: str) -> tuple[dict | None, float, str | None]:
    start = time.monotonic()
    try:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": STRUCTURING_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "response_format": RESPONSE_SCHEMA,
            },
            timeout=30.0,
        )
        elapsed = time.monotonic() - start
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content), elapsed, None
    except Exception as exc:
        elapsed = time.monotonic() - start
        return None, elapsed, str(exc)[:200]


async def main():
    latencies = []
    errors = []
    async with httpx.AsyncClient() as client:
        for fixture in FIXTURES:
            state = {}
            messages = []
            for turn in fixture["turns"]:
                messages.append(ChatMessage(role="user", content=turn["text"]))
                prompt = _build_structuring_prompt(messages, state)
                result, elapsed, err = await call_groq(client, prompt)
                latencies.append(elapsed)
                status = "OK" if result else f"ERROR: {err}"
                print(f"  {elapsed:.2f}s  [{status}]  {turn['text'][:45]!r}")
                if err:
                    errors.append(err)
                if result:
                    state = {**state, "facts": state.get("facts", []) + result.get("facts", [])}

    latencies.sort()
    n = len(latencies)
    print(f"\n{'='*70}")
    print(f"Groq {GROQ_MODEL} -- n={n}  errors={len(errors)}")
    print(f"min={min(latencies):.2f}s max={max(latencies):.2f}s mean={sum(latencies)/n:.2f}s "
          f"median={latencies[n//2]:.2f}s p90={latencies[int(n*0.9)]:.2f}s")
    if errors:
        print("\nSample errors:")
        for e in errors[:5]:
            print(" ", e)

asyncio.run(main())
