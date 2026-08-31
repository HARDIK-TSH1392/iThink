from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator, List, Optional

from app.database import async_session
from app.iTriage.iTriage_service import run_triage_for_log
from .iLogs_schema import LogEventCreate, LogEventRead
from .iLogs_crudl import create_log, get_log, list_logs, delete_log
from .iLogs_utils import (
    should_trigger_incident,
    infer_incident_priority,
    compute_impact_score,
    is_candidate_for_ai_review,
)


router = APIRouter(prefix="/ilogs", tags=["iLogs"])


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a DB session per request.
    """
    async with async_session() as session:
        yield session


@router.post("/ingest", response_model=LogEventRead)
async def ingest_log(
    event: LogEventCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> LogEventRead:
    """
    Ingest a new log event.

    This is the main entry point for logs from external sources.
    The log is stored, deterministic triage is performed (incident
    candidate flag, priority, impact score), and iTriage is dispatched
    as a background task for any event that qualifies.
    """
    created = await create_log(db, event)

    # Deterministic triage (no AI yet)
    is_incident_candidate = should_trigger_incident(event)
    priority = infer_incident_priority(event) if is_incident_candidate else None
    impact_score = compute_impact_score(event)
    ai_review_candidate = is_candidate_for_ai_review(event)

    if is_incident_candidate:
        print(
            f"[INCIDENT CANDIDATE] priority={priority} | "
            f"impact_score={impact_score} | "
            f"{event.region} | {event.service} | {event.message}"
        )

    if ai_review_candidate and not is_incident_candidate:
        print(
            f"[AI REVIEW CANDIDATE] impact_score={impact_score} | "
            f"{event.region} | {event.service} | {event.message}"
        )

    # Trigger iTriage on the union of both signals: should_trigger_incident
    # alone would miss e.g. a prod_down event in a non-production
    # environment, which is_candidate_for_ai_review doesn't cover.
    if is_incident_candidate or ai_review_candidate:
        background_tasks.add_task(run_triage_for_log, created.id)

    return LogEventRead.model_validate(created)


@router.get("/", response_model=List[LogEventRead])
async def list_ilogs(
    region: Optional[str] = Query(None),
    service: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[LogEventRead]:
    """
    List logs with optional filters.

    Returns logs ordered by timestamp (newest first), with pagination.
    """
    logs = await list_logs(
        db,
        region=region,
        service=service,
        severity=severity,
        limit=limit,
        offset=offset,
    )
    return [LogEventRead.model_validate(log) for log in logs]


@router.get("/{log_id}", response_model=LogEventRead)
async def get_ilog(
    log_id: int,
    db: AsyncSession = Depends(get_db),
) -> LogEventRead:
    """
    Get a single log by ID.
    """
    log = await get_log(db, log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    return LogEventRead.model_validate(log)


@router.delete("/{log_id}")
async def delete_ilog(
    log_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Delete a log by ID.
    """
    deleted = await delete_log(db, log_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Log not found")
    return {"status": "deleted"}