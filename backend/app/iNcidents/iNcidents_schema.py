from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Literal

IncidentStatus = Literal[
    "detected",
    "triage",
    "awaiting_approval",
    "approved",
    "rejected",
    "orchestrating",
    "in_call",
    "post_call",
    "resolved",
    "closed",
]

IncidentPriority = Literal["P1", "P2", "P3"]


class IncidentCreate(BaseModel):
    """Schema for creating an incident (request body)."""

    title: str
    summary: Optional[str] = None
    region: str
    service: str
    environment: Literal["production", "staging", "dev"]
    source_id: Optional[str] = None
    priority: Optional[IncidentPriority] = None
    impact_score: Optional[int] = None
    log_ids: List[int] = Field(default_factory=list)


class IncidentRead(BaseModel):
    """Schema for reading an incident (response)."""

    id: int
    title: str
    summary: Optional[str] = None
    status: str
    priority: Optional[str] = None
    region: str
    service: str
    environment: str
    source_id: Optional[str] = None
    impact_score: Optional[int] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    linked_log_ids: List[int] = Field(default_factory=list)

    class Config:
        from_attributes = True


class IncidentStatusUpdate(BaseModel):
    """Generic lifecycle transition, validated against the state machine."""

    status: IncidentStatus


class IncidentApprovalDecision(BaseModel):
    """The single human-approval gate before orchestration/external actions."""

    decision: Literal["approve", "reject"]
    approved_by: str
    reason: Optional[str] = None
