from sqlalchemy import String, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from app.database import Base


class TriageResult(Base):
    """
    Audit record of a single triage_ai run: which log triggered it, which
    incident it was attached to, the deterministically-computed confidence,
    and the raw Gemini verdict (for explainability / demoing the reasoning
    trail, not just the final decision).
    """

    __tablename__ = "triage_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    log_id: Mapped[int] = mapped_column(
        ForeignKey("ilogs.id", ondelete="CASCADE"), index=True
    )
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), index=True
    )

    deterministic_confidence: Mapped[str] = mapped_column(String)  # low/medium/high
    verdict: Mapped[dict] = mapped_column(JSON)  # raw TriageVerdict, serialized

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
