from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

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


class ChatMessage(BaseModel):
    """
    One message in the incoming request. Agora's Custom LLM contract adds
    turn_id/timestamp per message on top of the standard OpenAI role/content
    shape — accepted here but not required, since this is a proxy-only
    first version (see iCall_utils.generate_chat_reply).
    """

    role: str
    content: str
    turn_id: Optional[int] = None
    timestamp: Optional[int] = None


class AgoraWebhookEvent(BaseModel):
    """
    Envelope Agora POSTs to a registered webhook URL (Console -> Project ->
    notification config). eventType is one of: 101 agent joined, 102 agent
    left, 103 dialogue history, 110 agent error, 111 performance metrics,
    201/202 call status. payload's shape depends on eventType, so it's kept
    as a raw dict rather than a fixed schema.
    """

    noticeId: str
    productId: int
    eventType: int
    notifyMs: int
    payload: Dict[str, Any]

    class Config:
        extra = "allow"


class ActionItem(BaseModel):
    text: str
    owner: Optional[str] = None


class StructuringUpdate(BaseModel):
    """
    LLM output shape for one turn of live structuring. Deliberately excludes
    any root_cause or recommended_fix field — same discipline as iTriage's
    TriageVerdict — this is a coordination layer, not an autonomous
    diagnostician.

    facts/hypotheses/decisions/action_items are NEW items only for this
    turn, not the full running state — iCall_service merges them into
    IncidentCall.structured_state by appending in code, the same
    "don't let the LLM freely rewrite state" discipline as iTriage's
    deterministic confidence scoring. Never replace structured_state
    wholesale with this object.

    identified_speakers is a best-effort name/role guess from what was said
    (e.g. "I'm Priya, on-call SRE") — NOT tied to Agora's per-participant
    UID. Whether Agora's Custom LLM request carries a UID per message isn't
    confirmed in the docs read so far; real UID-level attribution is a
    separate, still-open investigation (see the per-speaker-transcription
    task), not something to fake here.
    """

    facts: List[str] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    conflict: Optional[str] = None
    identified_speakers: List[str] = Field(default_factory=list)
    spoken_reply: str


class ChatCompletionRequest(BaseModel):
    """
    Request body Agora's Conversational AI Engine sends to a Custom LLM
    base_url. Only the fields this proxy actually reads are declared;
    everything else Agora may send is tolerated and ignored.
    """

    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = True

    class Config:
        extra = "allow"
