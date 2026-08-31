from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Sequence, Optional

from .iLogs_model import ILog
from .iLogs_schema import LogEventCreate


async def create_log(db: AsyncSession, event: LogEventCreate) -> ILog:
    """
    Create a new log entry from a validated request schema.
    """
    db_log = ILog(**event.model_dump())
    db.add(db_log)
    await db.commit()
    await db.refresh(db_log)
    return db_log


async def get_log(db: AsyncSession, log_id: int) -> Optional[ILog]:
    """
    Get a single log by ID. Returns None if not found.
    """
    result = await db.execute(select(ILog).where(ILog.id == log_id))
    return result.scalar_one_or_none()


async def list_logs(
    db: AsyncSession,
    region: Optional[str] = None,
    service: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[ILog]:
    """
    List logs with optional filters.
    Returns a sequence of ILog rows.
    """
    query = select(ILog)

    if region:
        query = query.where(ILog.region == region)
    if service:
        query = query.where(ILog.service == service)
    if severity:
        query = query.where(ILog.severity == severity)

    query = (
        query
        .order_by(ILog.timestamp.desc())
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(query)
    return result.scalars().all()


async def delete_log(db: AsyncSession, log_id: int) -> bool:
    """
    Delete a log by ID. Returns True if deleted, False if not found.
    """
    log = await get_log(db, log_id)
    if not log:
        return False

    await db.delete(log)
    await db.commit()
    return True