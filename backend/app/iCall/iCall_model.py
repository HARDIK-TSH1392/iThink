from sqlalchemy import String, DateTime, ForeignKey, Integer, JSON, Index, func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional

from app.database import Base
from .iCall_utils import CALL_STATUS_SCHEDULED


class IncidentCall(Base):
    """
    The voice-room record for one incident. Owns the Agora RTC channel name
    (see iCall_utils.generate_channel_name) so orchestration and the voice
    agent both read the same value instead of each deriving their own.

    structured_state is intentionally a loose JSON blob for now: the shape
    of facts/hypotheses/decisions/action_items/timeline/conflicts/missing_info
    is still an open prompt-engineering discussion, not yet a fixed schema.
    Mirrors how iTriage.TriageResult.verdict stores its LLM output as JSON
    before locking in a rigid structure.
    """

    __tablename__ = "incident_calls"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"), index=True, unique=True
    )

    channel_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    status: Mapped[str] = mapped_column(String, default=CALL_STATUS_SCHEDULED, index=True)

    structured_state: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


class CallUtterance(Base):
    """
    One attributed line of transcript within a call. Kept separate from
    IncidentCall.structured_state the same way iLogs (raw events) is kept
    separate from Incident (aggregated state) — raw evidence vs. derived
    summary are different lifecycles and shouldn't share a table.
    """

    __tablename__ = "call_utterances"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    call_id: Mapped[int] = mapped_column(
        ForeignKey("incident_calls.id", ondelete="CASCADE"), index=True
    )

    # Agora RTC participant UID the utterance was attributed to. Resolving
    # this to a human name/role is Orchestration's directory lookup, not
    # this module's job — kept as the raw UID here.
    speaker_uid: Mapped[str] = mapped_column(String, index=True)

    text: Mapped[str] = mapped_column(String)
    turn_index: Mapped[int] = mapped_column(Integer, index=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    __table_args__ = (
        # Common access pattern: "give me this call's transcript in order."
        Index("ix_call_utterances_call_turn", "call_id", "turn_index"),
    )
