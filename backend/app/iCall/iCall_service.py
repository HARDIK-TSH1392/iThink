import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, List, Optional

from app.iNcidents.iNcidents_crudl import get_incident

from .iCall_model import IncidentCall, CallUtterance
from .iCall_schema import CallUtteranceCreate, StructuringUpdate
from .iCall_utils import generate_channel_name, CALL_STATUS_SCHEDULED

# Serializes "check for an existing call, else create one" per incident_id,
# the same class of fix as iTriage's per-key lock: without it, concurrent
# requests for the same incident_id can both pass the "no existing call"
# check before either commits, and the loser hits the DB's unique
# constraint on incident_id as an unhandled error instead of just getting
# the winner's row back.
_incident_call_locks: Dict[int, asyncio.Lock] = {}
_locks_registry_guard = asyncio.Lock()


async def _get_incident_lock(incident_id: int) -> asyncio.Lock:
    async with _locks_registry_guard:
        if incident_id not in _incident_call_locks:
            _incident_call_locks[incident_id] = asyncio.Lock()
        return _incident_call_locks[incident_id]


class IncidentNotFoundError(Exception):
    pass


async def get_call_by_incident(db: AsyncSession, incident_id: int) -> Optional[IncidentCall]:
    result = await db.execute(
        select(IncidentCall).where(IncidentCall.incident_id == incident_id)
    )
    return result.scalar_one_or_none()


async def get_or_create_call(db: AsyncSession, incident_id: int) -> IncidentCall:
    """
    The one place a channel name is decided. Orchestration (calendar/email
    invite) and the voice agent (RTC join) both call this instead of
    independently computing a channel name, so there is exactly one source
    of truth per incident rather than two formulas that could drift apart.

    Raises IncidentNotFoundError if incident_id doesn't exist — the API
    layer turns that into a 404 rather than letting a bad foreign key
    surface as a raw database error.
    """
    lock = await _get_incident_lock(incident_id)
    async with lock:
        existing = await get_call_by_incident(db, incident_id)
        if existing is not None:
            return existing

        incident = await get_incident(db, incident_id)
        if incident is None:
            raise IncidentNotFoundError(f"Incident {incident_id} not found")

        call = IncidentCall(
            incident_id=incident_id,
            channel_name=generate_channel_name(incident_id),
            status=CALL_STATUS_SCHEDULED,
            structured_state={},
        )
        db.add(call)
        await db.commit()
        await db.refresh(call)
        return call


async def get_call(db: AsyncSession, call_id: int) -> Optional[IncidentCall]:
    return await db.get(IncidentCall, call_id)


async def get_call_by_channel_name(db: AsyncSession, channel_name: str) -> Optional[IncidentCall]:
    """
    Resolves a call from the Agora channel name alone. This is what the
    custom-LLM endpoint uses to figure out *which* incident's call it's
    serving — Agora's request body carries no incident/call id, only the
    conversation content, so the channel name in the URL path is the only
    thing tying a given request back to a specific IncidentCall row.
    """
    result = await db.execute(
        select(IncidentCall).where(IncidentCall.channel_name == channel_name)
    )
    return result.scalar_one_or_none()


async def apply_structuring_update(
    db: AsyncSession, call: IncidentCall, update: StructuringUpdate
) -> IncidentCall:
    """
    Merges one turn's extraction into structured_state by appending —
    deliberately never overwrites the existing lists wholesale. Mirrors the
    "/update overwrites params entirely" gotcha documented for Agora's own
    agent-update endpoint: the same failure mode (silently losing prior
    state) is just as real here if this merged the LLM's per-turn output
    in as the new state instead of adding to what's already recorded.

    Builds brand-new list objects for every key rather than a shallow
    dict(...) copy — a shallow copy leaves the nested lists shared with
    call.structured_state's existing lists, so mutating them in place also
    mutates the "old" value SQLAlchemy compares against, which makes it
    see old == new and silently skip the UPDATE on second and later calls.
    Caught this via the two-turn accumulation test, not by inspection.
    """
    old = call.structured_state or {}
    state = {
        "facts": list(old.get("facts", [])),
        "hypotheses": list(old.get("hypotheses", [])),
        "decisions": list(old.get("decisions", [])),
        "action_items": list(old.get("action_items", [])),
        "conflicts": list(old.get("conflicts", [])),
        "identified_speakers": list(old.get("identified_speakers", [])),
    }

    state["facts"].extend(update.facts)
    state["hypotheses"].extend(update.hypotheses)
    state["decisions"].extend(update.decisions)
    state["action_items"].extend(item.model_dump() for item in update.action_items)
    if update.conflict:
        state["conflicts"].append(update.conflict)
    for speaker in update.identified_speakers:
        if speaker not in state["identified_speakers"]:
            state["identified_speakers"].append(speaker)

    call.structured_state = state
    await db.commit()
    await db.refresh(call)
    return call


async def update_call_status(db: AsyncSession, call: IncidentCall, status: str) -> IncidentCall:
    call.status = status
    await db.commit()
    await db.refresh(call)
    return call


async def record_utterance(
    db: AsyncSession, call_id: int, event: CallUtteranceCreate
) -> CallUtterance:
    utterance = CallUtterance(call_id=call_id, **event.model_dump())
    db.add(utterance)
    await db.commit()
    await db.refresh(utterance)
    return utterance


async def list_utterances(db: AsyncSession, call_id: int) -> List[CallUtterance]:
    result = await db.execute(
        select(CallUtterance)
        .where(CallUtterance.call_id == call_id)
        .order_by(CallUtterance.turn_index.asc())
    )
    return list(result.scalars().all())
