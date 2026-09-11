"""
Tests the actual missing piece from the classify/extract split proposal:
a broader "does this turn contain anything worth extracting" gate --
distinct from the narrow trigger-category classifier (correction/conflict/
root_cause_question/resolution_declared/wrap_up/none) already tested.

Ground truth for "should extraction have run" comes from the REAL combined
call (the proven 96%-accurate one) run turn-by-turn first -- if it produced
ANY new facts/hypotheses/decisions/action_items OR flagged conflict/
corrects_fact/is_wrapping_up, extraction was genuinely needed on that turn.
This is not assumed, it's what the trusted extractor itself did.

Then a separate, cheap classify-style gate call is tested against that same
ground truth: does it correctly predict "yes, extract" whenever the trusted
extractor found something, and "no, skip" whenever it didn't -- measuring
whether gating extraction this way would actually preserve system accuracy,
or silently drop real content (the flaw identified in the trigger-only gate).
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

GATE_INSTRUCTION = """You are watching a live incident call. Decide if the LATEST turn contains
ANYTHING worth recording: a new fact, a hypothesis/guess, a decision, an action item with or
without an owner, a gap/missing piece of information, a contradiction of something said earlier,
or the call sounding like it's wrapping up.

Output extract=true if there is ANY such content, even something small or partial.
Output extract=false ONLY for turns with no informational content at all: greetings ("hi",
"hey"), bare acknowledgments ("yeah", "okay", "got it"), names alone ("Bob"), filler words
("so", "like", "think"), or incomplete fragments with no standalone content ("so, like,").

When genuinely unsure whether a turn has content, prefer extract=true -- a missed extraction
silently loses real information, while an unnecessary extraction only costs a little extra
processing. The two mistakes are not equally bad."""


class ExtractGate(BaseModel):
    extract: bool


async def gate(client, recent_turns: list[str]) -> tuple[bool, float]:
    prompt = "\n".join(recent_turns[-4:])
    config = types.GenerateContentConfig(
        system_instruction=GATE_INSTRUCTION, response_mime_type="application/json", response_schema=ExtractGate,
    )
    start = time.monotonic()
    response = await asyncio.to_thread(client.models.generate_content, model=PRIMARY_MODEL, contents=prompt, config=config)
    elapsed = time.monotonic() - start
    return ExtractGate.model_validate_json(response.text).extract, elapsed


async def get_ground_truth(db, fixture) -> list[bool]:
    """Re-runs the real, trusted combined call turn-by-turn and records
    whether it actually found anything worth recording each turn."""
    incident = Incident(title=f"gate-{fixture['channel']}", region="us-east", service="auth-api",
                         environment="production", status="approved")
    db.add(incident)
    await db.flush()
    call = IncidentCall(incident_id=incident.id, channel_name=f"gate-{fixture['channel']}", structured_state={})
    db.add(call)
    await db.flush()

    truth = []
    messages = []
    try:
        for turn in fixture["turns"]:
            messages.append(ChatMessage(role="user", content=turn["text"]))
            result = await generate_structuring_update(messages, call.structured_state or {})
            update = result.update
            had_content = bool(
                update.facts or update.hypotheses or update.decisions or update.action_items
                or update.missing_info or update.conflict or update.corrects_fact or update.is_wrapping_up
            )
            truth.append(had_content)
            call = await apply_structuring_update(db, call, update)
    finally:
        await db.delete(call)
        await db.delete(incident)
        await db.commit()
    return truth


async def main():
    client = _get_client()
    async with async_session() as db:
        false_negatives = 0  # real content, gate said skip -- the dangerous case
        false_positives = 0  # no content, gate said extract -- costs a little, not dangerous
        correct = 0
        total = 0
        latencies = []

        for fixture in FIXTURES:
            print(f"\n--- call_id={fixture['call_id']} ---")
            truth = await get_ground_truth(db, fixture)
            history = []
            for turn, had_content in zip(fixture["turns"], truth):
                history.append(f"{turn['speaker']}: {turn['text']}")
                predicted, elapsed = await gate(client, history)
                latencies.append(elapsed)
                total += 1
                if predicted == had_content:
                    correct += 1
                    mark = "OK"
                elif had_content and not predicted:
                    false_negatives += 1
                    mark = "MISS (dangerous: real content, gate said skip)"
                else:
                    false_positives += 1
                    mark = "extra (no content, gate said extract -- costs a little)"
                print(f"  ground_truth_had_content={had_content!s:5s} gate_predicted={predicted!s:5s}  [{mark}]  {turn['text'][:45]!r}")

    latencies.sort()
    n = len(latencies)
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"False negatives (dangerous -- lost real content): {false_negatives}/{total}")
    print(f"False positives (safe -- unnecessary extraction): {false_positives}/{total}")
    print(f"Latency: mean={sum(latencies)/n:.2f}s p90={latencies[int(n*0.9)]:.2f}s")

asyncio.run(main())
