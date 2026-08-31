from sqlalchemy import String, DateTime, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional

from app.database import Base
from .iNcidents_utils import STATUS_DETECTED


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    title: Mapped[str] = mapped_column(String)
    summary: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Lifecycle
    status: Mapped[str] = mapped_column(String, default=STATUS_DETECTED, index=True)
    priority: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)  # P1/P2/P3

    # Scope identity (mirrors iLogs identity fields for correlation)
    region: Mapped[str] = mapped_column(String, index=True)
    service: Mapped[str] = mapped_column(String, index=True)
    environment: Mapped[str] = mapped_column(String)
    source_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    impact_score: Mapped[Optional[int]] = mapped_column(nullable=True)

    # Human approval gate
    approved_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_incidents_status_priority", "status", "priority"),
        Index("ix_incidents_region_service", "region", "service"),
    )


class IncidentLog(Base):
    """
    Association between an incident and an evidence log event (iLogs.ILog).
    An incident may be backed by multiple correlated log events over time.
    """

    __tablename__ = "incident_logs"

    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), primary_key=True
    )
    log_id: Mapped[int] = mapped_column(
        ForeignKey("ilogs.id", ondelete="CASCADE"), primary_key=True
    )
    linked_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
