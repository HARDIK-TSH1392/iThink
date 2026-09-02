from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from .iCall_model import IncidentCall, CallUtterance
from .iCall_schema import CallUtteranceCreate
from .iCall_utils import generate_channel_name, CALL_STATUS_SCHEDULED


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
    """
    existing = await get_call_by_incident(db, incident_id)
    if existing is not None:
        return existing

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
