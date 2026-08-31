from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from typing import Sequence, Optional, List

from app.iLogs.iLogs_model import ILog
from .iNcidents_model import Incident, IncidentLog
from .iNcidents_schema import IncidentCreate
from .iNcidents_utils import TERMINAL_STATUSES, STATUS_ORCHESTRATING, STATUS_REJECTED


async def existing_log_ids(db: AsyncSession, log_ids: List[int]) -> List[int]:
    """
    Given candidate log IDs, return the subset that actually exist in iLogs.
    Used to validate evidence links before persisting them, since SQLite
    (unlike Postgres) does not enforce the incident_logs FK by default.
    """
    if not log_ids:
        return []
    result = await db.execute(select(ILog.id).where(ILog.id.in_(log_ids)))
    return [row[0] for row in result.all()]


async def create_incident(db: AsyncSession, payload: IncidentCreate) -> Incident:
    """
    Create a new incident and link any evidence logs passed at creation time.
    """
    data = payload.model_dump(exclude={"log_ids"})
    incident = Incident(**data)
    db.add(incident)
    await db.flush()  # assign incident.id before linking logs

    for log_id in payload.log_ids:
        db.add(IncidentLog(incident_id=incident.id, log_id=log_id))

    await db.commit()
    await db.refresh(incident)
    return incident


async def get_incident(db: AsyncSession, incident_id: int) -> Optional[Incident]:
    """
    Get a single incident by ID. Returns None if not found.
    """
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    return result.scalar_one_or_none()


async def get_linked_log_ids(db: AsyncSession, incident_id: int) -> List[int]:
    """
    Return the IDs of iLogs events linked as evidence to this incident.
    """
    result = await db.execute(
        select(IncidentLog.log_id).where(IncidentLog.incident_id == incident_id)
    )
    return [row[0] for row in result.all()]


async def list_incidents(
    db: AsyncSession,
    status: Optional[str] = None,
    region: Optional[str] = None,
    service: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[Incident]:
    """
    List incidents with optional filters, newest first.
    """
    query = select(Incident)

    if status:
        query = query.where(Incident.status == status)
    if region:
        query = query.where(Incident.region == region)
    if service:
        query = query.where(Incident.service == service)
    if priority:
        query = query.where(Incident.priority == priority)

    query = (
        query
        .order_by(Incident.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(query)
    return result.scalars().all()


async def link_log(db: AsyncSession, incident_id: int, log_id: int) -> bool:
    """
    Link an additional evidence log to an incident. Returns False if
    already linked (idempotent), True if newly linked.
    """
    existing = await db.execute(
        select(IncidentLog).where(
            IncidentLog.incident_id == incident_id,
            IncidentLog.log_id == log_id,
        )
    )
    if existing.scalar_one_or_none():
        return False

    db.add(IncidentLog(incident_id=incident_id, log_id=log_id))
    await db.commit()
    return True


async def update_status(db: AsyncSession, incident: Incident, new_status: str) -> Incident:
    """
    Persist a validated lifecycle transition. Caller is responsible for
    checking iNcidents_utils.is_valid_transition beforehand.
    """
    incident.status = new_status
    if new_status in TERMINAL_STATUSES:
        incident.resolved_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(incident)
    return incident


async def record_approval_decision(
    db: AsyncSession,
    incident: Incident,
    decision: str,
    approved_by: str,
) -> Incident:
    """
    Record the single human-approval gate decision for an incident.
    Approving moves the incident into orchestration; rejecting closes it out.
    """
    incident.approved_by = approved_by
    incident.approved_at = datetime.now(timezone.utc)
    incident.status = STATUS_ORCHESTRATING if decision == "approve" else STATUS_REJECTED

    if decision == "reject":
        incident.resolved_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(incident)
    return incident
