"""
v3: tests the user's proposed design directly -- the REAL structured_state
(produced by the actual, trusted apply_structuring_update merge, not a
hand-rolled approximation) plus ONLY the single latest turn, no raw
multi-turn window at all. Compared against v2 (crude state + last-4-raw-
turns) on the exact same real transcripts and expected labels.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types
from pydantic import BaseModel

from app.database import async_session
from app.iNcidents.iNcidents_model import Incident
from app.iCall.iCall_model import IncidentCall
from app.iCall.iCall_schema import ChatMessage
from app.iCall.iCall_utils import generate_structuring_update, _get_client, PRIMARY_MODEL
from app.iCall.iCall_service import apply_structuring_update

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import FIXTURES
from measure_embedding_router import EXPECTED_TRIGGERS
from measure_classify_v2 import CLASSIFY_INSTRUCTION, TriggerClassification


async def classify_with_real_state(client, structured_state: dict, latest_turn: str) -> tuple[str, str, float]:
    prompt = f"Known state so far: {structured_state}\n\nLatest turn: {latest_turn}"
    config = types.GenerateContentConfig(
        system_instruction=CLASSIFY_INSTRUCTION, response_mime_type="application/json", response_schema=TriggerClassification,
    )
    start = time.monotonic()
    response = await asyncio.to_thread(client.models.generate_content, model=PRIMARY_MODEL, contents=prompt, config=config)
    elapsed = time.monotonic() - start
    result = TriggerClassification.model_validate_json(response.text)
    return result.trigger, result.reason, elapsed


async def main():
    client = _get_client()
    latencies = []
    correct = 0
    total = 0

    async with async_session() as db:
        for fixture in FIXTURES:
            print(f"\n--- call_id={fixture['call_id']} ---")
            expected_list = EXPECTED_TRIGGERS[fixture["call_id"]]

            incident = Incident(title=f"v3-{fixture['channel']}", region="us-east", service="auth-api",
                                 environment="production", status="approved")
            db.add(incident)
            await db.flush()
            call = IncidentCall(incident_id=incident.id, channel_name=f"v3-{fixture['channel']}", structured_state={})
            db.add(call)
            await db.flush()

            messages = []
            try:
                for turn, expected in zip(fixture["turns"], expected_list):
                    messages.append(ChatMessage(role="user", content=turn["text"]))

                    # real state, as it existed BEFORE this turn -- same
                    # snapshot the real combined call would have seen
                    state_before = dict(call.structured_state or {})

                    trigger, reason, elapsed = await classify_with_real_state(client, state_before, turn["text"])
                    latencies.append(elapsed)
                    total += 1
                    is_correct = trigger == expected
                    correct += is_correct
                    mark = "OK" if is_correct else "WRONG"
                    print(f"  {elapsed:.2f}s  [{mark:5s}] predicted={trigger:20s} expected={expected:20s} "
                          f"reason={reason[:40]!r}  {turn['text'][:35]!r}")

                    # advance real state using the real, trusted extraction call --
                    # same ground truth every other test in this session used
                    result = await generate_structuring_update(messages, call.structured_state or {})
                    if result.update:
                        call = await apply_structuring_update(db, call, result.update)
            finally:
                await db.delete(call)
                await db.delete(incident)
                await db.commit()

    latencies.sort()
    n = len(latencies)
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Latency: mean={sum(latencies)/n:.2f}s p90={latencies[int(n*0.9)]:.2f}s")

asyncio.run(main())
