"""
Regression test for the action-item duplication bug from incident-46
(2026-09-09): "check the port config" got extracted on three separate
turns as STT fragmented one exchange, producing three action_items
entries and four spoken "Got it, Ted's on that." confirmations for what
was really one assignment.

Pure unit test against the real, deterministic functions involved
(apply_structuring_update, should_speak_aloud, describe_speak_reason,
build_gated_spoken_reply) -- no LLM call needed, since dedup is Python
logic operating on already-extracted StructuringUpdate objects, not a
model behavior being measured.

Uses one throwaway Incident + IncidentCall row, created and deleted
around the run -- same discipline as run_eval.py, never leaves anything
behind in the real dev DB.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session, init_db
from app.iCall.iCall_schema import StructuringUpdate, ActionItem
from app.iCall.iCall_service import apply_structuring_update, get_or_create_call
from app.iCall.iCall_utils import should_speak_aloud, describe_speak_reason, build_gated_spoken_reply
from app.iNcidents.iNcidents_crudl import create_incident
from app.iNcidents.iNcidents_schema import IncidentCreate


def _update(action_items, spoken_reply="ack"):
    return StructuringUpdate(
        action_items=[ActionItem(**item) for item in action_items],
        spoken_reply=spoken_reply,
    )


async def main():
    await init_db()
    async with async_session() as db:
        incident = await create_incident(db, IncidentCreate(
            title="dedup test", region="us-east", service="checkout", environment="production",
        ))
        call = await get_or_create_call(db, incident.id)

        results = []

        # Turn 1: unowned mention -- should create ONE entry, should NOT speak.
        u1 = _update([{"text": "check the port config", "owner": None}])
        speak1 = should_speak_aloud(u1, "someone needs to check the port config", call.structured_state)
        call = await apply_structuring_update(db, call, u1)
        results.append(("turn1 speak", speak1, False))
        results.append(("turn1 count", len(call.structured_state["action_items"]), 1))

        # Turn 2: same text, owner now supplied -- should MERGE (still 1
        # entry), owner filled in, and SHOULD speak once.
        u2 = _update([{"text": "check the port config", "owner": "Ted"}])
        speak2 = should_speak_aloud(u2, "Ted covers it", call.structured_state)
        reason2 = describe_speak_reason(u2, "Ted covers it", call.structured_state) if speak2 else None
        spoken2 = build_gated_spoken_reply(u2, reason2, call.structured_state) if reason2 else None
        call = await apply_structuring_update(db, call, u2)
        results.append(("turn2 speak", speak2, True))
        results.append(("turn2 reason", reason2, "action_item_owner"))
        results.append(("turn2 spoken", spoken2, "Got it, Ted's on that."))
        results.append(("turn2 count", len(call.structured_state["action_items"]), 1))
        results.append(("turn2 owner", call.structured_state["action_items"][0].get("owner"), "Ted"))

        # Turn 3: STT re-extracts the SAME text with the SAME owner again
        # (the real incident-46 pattern) -- should be a pure no-op: still
        # 1 entry, and must NOT speak again.
        u3 = _update([{"text": "check the port config", "owner": "Ted"}])
        speak3 = should_speak_aloud(u3, "check the port config", call.structured_state)
        call = await apply_structuring_update(db, call, u3)
        results.append(("turn3 speak (must be False)", speak3, False))
        results.append(("turn3 count (must stay 1)", len(call.structured_state["action_items"]), 1))

        # Turn 4: a genuinely different action item -- must NOT be merged
        # into the unrelated existing one.
        u4 = _update([{"text": "restart the connection pool", "owner": None}])
        call = await apply_structuring_update(db, call, u4)
        results.append(("turn4 count (2 distinct items)", len(call.structured_state["action_items"]), 2))

        ok = True
        for label, actual, expected in results:
            status = "PASS" if actual == expected else "FAIL"
            if actual != expected:
                ok = False
            print(f"  [{status}] {label}: got={actual!r} expected={expected!r}")

        print("\nALL PASS" if ok else "\nFAILURES ABOVE")

        await db.delete(call)
        await db.delete(incident)
        await db.commit()

        return 0 if ok else 1


sys.exit(asyncio.run(main()))
