from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator, List

from app.database import async_session
from .iCall_schema import (
    IncidentCallRead,
    CallStatusUpdate,
    CallUtteranceCreate,
    CallUtteranceRead,
)
from .iCall_service import (
    get_or_create_call,
    get_call,
    update_call_status,
    record_utterance,
    list_utterances,
    IncidentNotFoundError,
)

router = APIRouter(prefix="/icall", tags=["iCall"])


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a DB session per request.
    """
    async with async_session() as session:
        yield session


@router.post("/incidents/{incident_id}/call", response_model=IncidentCallRead)
async def get_or_create_call_endpoint(
    incident_id: int,
    db: AsyncSession = Depends(get_db),
) -> IncidentCallRead:
    """
    Get this incident's call, creating it (and its channel name) if it
    doesn't exist yet. Orchestration calls this to get the channel name for
    the meeting invite; the voice agent calls this to know which channel to
    join. Both get the same answer because there's one row, not two guesses.
    """
    try:
        call = await get_or_create_call(db, incident_id)
    except IncidentNotFoundError:
        raise HTTPException(status_code=404, detail="Incident not found")
    return IncidentCallRead.model_validate(call)


@router.get("/{call_id}", response_model=IncidentCallRead)
async def get_call_endpoint(
    call_id: int,
    db: AsyncSession = Depends(get_db),
) -> IncidentCallRead:
    """
    Get a single call by ID.
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return IncidentCallRead.model_validate(call)


@router.patch("/{call_id}/status", response_model=IncidentCallRead)
async def update_call_status_endpoint(
    call_id: int,
    payload: CallStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> IncidentCallRead:
    """
    Advance a call's lifecycle status (scheduled -> in_progress -> completed).
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    call = await update_call_status(db, call, payload.status)
    return IncidentCallRead.model_validate(call)


@router.post("/{call_id}/utterances", response_model=CallUtteranceRead)
async def record_utterance_endpoint(
    call_id: int,
    payload: CallUtteranceCreate,
    db: AsyncSession = Depends(get_db),
) -> CallUtteranceRead:
    """
    Record one attributed transcript line for a call.
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    utterance = await record_utterance(db, call_id, payload)
    return CallUtteranceRead.model_validate(utterance)


@router.get("/{call_id}/utterances", response_model=List[CallUtteranceRead])
async def list_utterances_endpoint(
    call_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[CallUtteranceRead]:
    """
    List a call's transcript, ordered by turn.
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    utterances = await list_utterances(db, call_id)
    return [CallUtteranceRead.model_validate(u) for u in utterances]
