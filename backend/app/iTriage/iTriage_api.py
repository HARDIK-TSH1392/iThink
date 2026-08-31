from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from typing import AsyncGenerator

from app.database import async_session
from app.iLogs.iLogs_model import ILog

from .iTriage_service import run_triage_for_log
from .iTriage_model import TriageResult
from .iTriage_schema import TriageResultRead


router = APIRouter(prefix="/itriage", tags=["iTriage"])


@router.post("/run/{log_id}")
async def run_triage_endpoint(log_id: int) -> dict:
    """
    Manually trigger triage for a given log_id. This exists purely for
    testing/demo purposes — in the real flow, iLogs triggers this itself
    via a BackgroundTask on ingest.
    """
    async with async_session() as db:
        log = await db.get(ILog, log_id)
        if not log:
            raise HTTPException(status_code=404, detail="Log not found")

    return await run_triage_for_log(log_id)


@router.get("/results/{log_id}", response_model=TriageResultRead)
async def get_triage_result_endpoint(log_id: int) -> TriageResultRead:
    """
    Fetch the most recent triage result for a given log_id, if one exists.
    """
    async with async_session() as db:
        result = await db.execute(
            select(TriageResult)
            .where(TriageResult.log_id == log_id)
            .order_by(TriageResult.id.desc())
        )
        triage_result = result.scalars().first()

    if not triage_result:
        raise HTTPException(status_code=404, detail="No triage result found for this log")

    return TriageResultRead(
        id=triage_result.id,
        log_id=triage_result.log_id,
        incident_id=triage_result.incident_id,
        deterministic_confidence=triage_result.deterministic_confidence,
        verdict=triage_result.verdict,
    )
