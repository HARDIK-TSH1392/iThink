from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator, List, Optional

from app.database import async_session
from .iNcidents_schema import (
    IncidentCreate,
    IncidentRead,
    IncidentStatusUpdate,
    IncidentApprovalDecision,
)
from .iNcidents_crudl import (
    create_incident,
    get_incident,
    get_linked_log_ids,
    list_incidents,
    link_log,
    update_status,
    record_approval_decision,
)
from .iNcidents_utils import is_valid_transition, STATUS_AWAITING_APPROVAL


router = APIRouter(prefix="/incidents", tags=["iNcidents"])


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a DB session per request.
    """
    async with async_session() as session:
        yield session


async def _to_read(db: AsyncSession, incident) -> IncidentRead:
    log_ids = await get_linked_log_ids(db, incident.id)
    read = IncidentRead.model_validate(incident)
    read.linked_log_ids = log_ids
    return read


@router.post("/", response_model=IncidentRead)
async def create_incident_endpoint(
    payload: IncidentCreate,
    db: AsyncSession = Depends(get_db),
) -> IncidentRead:
    """
    Create a new incident, optionally linking evidence logs at creation time.
    """
    incident = await create_incident(db, payload)
    return await _to_read(db, incident)


@router.get("/", response_model=List[IncidentRead])
async def list_incidents_endpoint(
    status: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    service: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[IncidentRead]:
    """
    List incidents with optional filters, newest first.
    """
    incidents = await list_incidents(
        db,
        status=status,
        region=region,
        service=service,
        priority=priority,
        limit=limit,
        offset=offset,
    )
    return [await _to_read(db, incident) for incident in incidents]


@router.get("/{incident_id}", response_model=IncidentRead)
async def get_incident_endpoint(
    incident_id: int,
    db: AsyncSession = Depends(get_db),
) -> IncidentRead:
    """
    Get a single incident by ID.
    """
    incident = await get_incident(db, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return await _to_read(db, incident)


@router.post("/{incident_id}/logs/{log_id}", response_model=IncidentRead)
async def link_log_endpoint(
    incident_id: int,
    log_id: int,
    db: AsyncSession = Depends(get_db),
) -> IncidentRead:
    """
    Link an additional evidence log to an incident.
    """
    incident = await get_incident(db, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    await link_log(db, incident_id, log_id)
    return await _to_read(db, incident)


@router.patch("/{incident_id}/status", response_model=IncidentRead)
async def update_status_endpoint(
    incident_id: int,
    payload: IncidentStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> IncidentRead:
    """
    Generic lifecycle transition, validated against the deterministic
    state machine (e.g. resolved -> detected is rejected).
    """
    incident = await get_incident(db, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    if not is_valid_transition(incident.status, payload.status):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot transition incident from '{incident.status}' to '{payload.status}'",
        )

    incident = await update_status(db, incident, payload.status)
    return await _to_read(db, incident)


@router.post("/{incident_id}/decision", response_model=IncidentRead)
async def decide_incident_endpoint(
    incident_id: int,
    payload: IncidentApprovalDecision,
    db: AsyncSession = Depends(get_db),
) -> IncidentRead:
    """
    The single human-approval gate: a team lead approves or rejects an
    incident candidate before any orchestration/external action happens.
    """
    incident = await get_incident(db, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    if incident.status != STATUS_AWAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Incident must be in '{STATUS_AWAITING_APPROVAL}' status to "
                f"record a decision (currently '{incident.status}')"
            ),
        )

    incident = await record_approval_decision(
        db, incident, payload.decision, payload.approved_by
    )
    return await _to_read(db, incident)
