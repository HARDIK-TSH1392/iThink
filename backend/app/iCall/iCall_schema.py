from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal

CallStatus = Literal["scheduled", "in_progress", "completed"]


class IncidentCallRead(BaseModel):
    """Schema for reading a call (response)."""

    id: int
    incident_id: int
    channel_name: str
    status: str
    structured_state: dict
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CallStatusUpdate(BaseModel):
    """Advance a call's lifecycle status."""

    status: CallStatus


class CallUtteranceCreate(BaseModel):
    """Schema for recording one attributed transcript line."""

    speaker_uid: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=0)
    timestamp: datetime


class CallUtteranceRead(BaseModel):
    """Schema for reading a transcript line (response)."""

    id: int
    call_id: int
    speaker_uid: str
    text: str
    turn_index: int
    timestamp: datetime

    class Config:
        from_attributes = True
