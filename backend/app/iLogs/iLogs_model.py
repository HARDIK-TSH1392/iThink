from sqlalchemy import String, DateTime, Index, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional

from app.database import Base


class ILog(Base):
    __tablename__ = "ilogs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Source identity
    source_id: Mapped[str] = mapped_column(String, index=True)
    source_type: Mapped[str] = mapped_column(String)  # k8s_deployment, vm, lambda, database, other
    region: Mapped[str] = mapped_column(String, index=True)
    service: Mapped[str] = mapped_column(String, index=True)
    environment: Mapped[str] = mapped_column(String)  # production, staging, dev

    # Log details
    severity: Mapped[str] = mapped_column(String, index=True)  # warning, error, critical, prod_down
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    message: Mapped[str] = mapped_column(String)

    # Structured data
    attributes: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)
    raw: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_ilogs_source_region", "source_id", "region"),
        Index("ix_ilogs_severity_timestamp", "severity", "timestamp"),
    )