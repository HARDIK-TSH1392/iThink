"""
Runs the real structuring pipeline (generate_structuring_update +
apply_structuring_update -- the exact functions the live call uses, not a
reimplementation) turn-by-turn against the labeled fixtures in fixtures.py,
and scores the result against each turn's expected judgment.

Uses one throwaway Incident + IncidentCall row per fixture, created and
deleted around the run -- never touches or leaves behind anything in the
real dev DB. Requires a real GEMINI_API_KEY (this hits the live API,
deliberately -- there is no meaningful way to evaluate prompt accuracy
against a mock).

Usage: backend/.venv/bin/python eval/run_eval.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from app.database import async_session
from app.iNcidents.iNcidents_model import Incident
from app.iCall.iCall_model import IncidentCall
from app.iCall.iCall_schema import ChatMessage
from app.iCall.iCall_utils import generate_structuring_update
from app.iCall.iCall_service import apply_structuring_update

from fixtures import FIXTURES

import re


def _normalize(text: str) -> str:
    """
    Strips punctuation and collapses whitespace before a substring check --
    real STT output (Deepgram's smart_format) inserts sentence-splitting
    periods mid-phrase ("US. East region" for "US East region"), which is a
    transcription artifact, not a difference in the model's actual
    judgment. Comparing normalized text avoids scoring that artifact as an
    extraction failure.
    """
    no_punct = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", no_punct).strip()


def check_turn(update, existing_state_before: dict, expect: dict) -> list[tuple[str, bool, str]]:
    """Returns a list of (check_name, passed, detail) for one turn."""
    results = []

    if "key_terms_in_facts" in expect:
        all_fact_text = _normalize(" ".join(update.facts))
        for term in expect["key_terms_in_facts"]:
            passed = _normalize(term) in all_fact_text
            results.append((f"fact contains '{term}'", passed, all_fact_text[:120]))

    if "hypothesis_expected" in expect:
        passed = bool(update.hypotheses) == expect["hypothesis_expected"]
        results.append(("hypothesis extracted", passed, str(update.hypotheses)))

    if "conflict_expected" in expect:
        passed = bool(update.conflict) == expect["conflict_expected"]
        results.append(("conflict flagged as expected", passed, repr(update.conflict)))

    if "corrects_fact_expected" in expect:
        fired = bool(update.corrects_fact) and update.corrects_fact in existing_state_before.get("facts", [])
        passed = fired == expect["corrects_fact_expected"]
        detail = repr(update.corrects_fact)
        if fired and "corrects_fact_should_reference" in expect:
            ref = expect["corrects_fact_should_reference"]
            passed = passed and _normalize(ref) in _normalize(update.corrects_fact or "")
            detail += f" (should reference '{ref}')"
        results.append(("corrects_fact correct", passed, detail))

    if "decision_expected" in expect:
        passed = bool(update.decisions) == expect["decision_expected"]
        results.append(("decision extracted", passed, str(update.decisions)))

    if "missing_info_expected" in expect:
        passed = bool(update.missing_info) == expect["missing_info_expected"]
        results.append(("missing_info extracted", passed, str(update.missing_info)))

    if "missing_info_should_be_deduped" in expect:
        already_known = set(existing_state_before.get("missing_info", []))
        new_gaps = [g for g in update.missing_info if g not in already_known]
        passed = (len(new_gaps) == 0) == expect["missing_info_should_be_deduped"]
        results.append(("missing_info deduped (no new gap)", passed, str(update.missing_info)))

    if "action_item_owner_expected" in expect:
        owners = [item.owner for item in update.action_items if item.owner]
        passed = any(expect["action_item_owner_expected"].lower() in (o or "").lower() for o in owners)
        results.append((f"action item owner = {expect['action_item_owner_expected']}", passed, str(owners)))

    if "is_wrapping_up_expected" in expect:
        passed = update.is_wrapping_up == expect["is_wrapping_up_expected"]
        results.append(("is_wrapping_up correct", passed, str(update.is_wrapping_up)))

    return results


async def run_fixture(db: AsyncSession, fixture: dict) -> tuple[int, int, list[str]]:
    incident = Incident(
        title=f"eval-{fixture['channel']}", region="us-east", service="auth-api",
        environment="production", status="approved",
    )
    db.add(incident)
    await db.flush()

    call = IncidentCall(incident_id=incident.id, channel_name=f"eval-{fixture['channel']}", structured_state={})
    db.add(call)
    await db.flush()

    passed_count = 0
    total_count = 0
    failures = []
    messages: list[ChatMessage] = []

    try:
        for i, turn in enumerate(fixture["turns"]):
            messages.append(ChatMessage(role="user", content=turn["text"]))
            state_before = dict(call.structured_state or {})

            result = await generate_structuring_update(messages, state_before)
            update = result.update
            if update is None:
                failures.append(f"  turn {i} ({turn['text'][:40]!r}): no update returned (tool_call={result.tool_call})")
                continue

            checks = check_turn(update, state_before, turn.get("expect", {}))
            for name, passed, detail in checks:
                total_count += 1
                if passed:
                    passed_count += 1
                else:
                    failures.append(f"  turn {i} ({turn['text'][:50]!r}): FAILED [{name}] -- got {detail}")

            call = await apply_structuring_update(db, call, update)
    finally:
        await db.delete(call)
        await db.delete(incident)
        await db.commit()

    return passed_count, total_count, failures


async def main():
    grand_passed = 0
    grand_total = 0
    async with async_session() as db:
        for fixture in FIXTURES:
            print(f"\n{'='*70}\ncall_id={fixture['call_id']} ({fixture['channel']}): {fixture['description']}\n{'='*70}")
            passed, total, failures = await run_fixture(db, fixture)
            grand_passed += passed
            grand_total += total
            print(f"Score: {passed}/{total}")
            for f in failures:
                print(f)

    print(f"\n{'='*70}\nTOTAL: {grand_passed}/{grand_total} ({100*grand_passed/grand_total:.1f}%)\n{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
