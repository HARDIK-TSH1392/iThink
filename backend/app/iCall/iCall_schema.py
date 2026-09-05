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
    participant_roles: dict = Field(default_factory=dict)
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
    # Best-known display name at post time (from the join-screen name
    # registry). Optional since a line could arrive before a name's been
    # published yet -- role-inference just treats those as unnamed.
    speaker_name: Optional[str] = None
    text: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=0)
    timestamp: datetime


class CallUtteranceRead(BaseModel):
    """Schema for reading a transcript line (response)."""

    id: int
    call_id: int
    speaker_uid: str
    speaker_name: Optional[str] = None
    text: str
    turn_index: int
    timestamp: datetime

    class Config:
        from_attributes = True


class AgentUtteranceRead(BaseModel):
    """Schema for reading one line the agent actually spoke (response)."""

    id: int
    call_id: int
    text: str
    reason: str
    timestamp: datetime

    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    """
    One message in the incoming request. Agora's Custom LLM contract adds
    turn_id/timestamp per message on top of the standard OpenAI role/content
    shape — accepted here but not required, since this is a proxy-only
    first version (see iCall_utils.generate_chat_reply).

    tool_calls/tool_call_id/name support native MCP tool-calling (see
    chat_completions_endpoint): an assistant message that called a tool
    has tool_calls and often no content; the tool's result comes back as
    its own role="tool" message with tool_call_id (and sometimes name)
    set, content holding the tool's return value.
    """

    role: str
    content: Optional[str] = None
    turn_id: Optional[int] = None
    timestamp: Optional[int] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


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

    facts/hypotheses/decisions/action_items/missing_info are NEW items only
    for this turn, not the full running state — iCall_service merges them
    into IncidentCall.structured_state by appending in code, the same
    "don't let the LLM freely rewrite state" discipline as iTriage's
    deterministic confidence scoring. Never replace structured_state
    wholesale with this object.

    identified_speakers is a best-effort name/role guess from what was said
    (e.g. "I'm Priya, on-call SRE") — NOT tied to Agora's per-participant
    UID; this live per-turn path still can't attribute speech to a UID
    (Agora's Custom LLM request carries no per-message speaker id, only
    merged role/content). Real UID-level attribution now exists, but as a
    separate pipeline: attributed client-side utterances -> end-of-call
    role classification (see iCall_service.infer_and_store_participant_roles)
    -- deliberately not duplicated here.

    is_wrapping_up flags that this turn sounds like the call concluding
    (e.g. "I think that covers everything," explicit goodbyes). iCall_api's
    endpoint -- not this model, not the LLM -- decides the actual closing
    line spoken to the room when this is true, so what gets promised about
    Slack/Jira is always accurate rather than whatever the model
    paraphrases.
    """

    facts: List[str] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    # Gaps or open questions noticed this turn -- e.g. "no one has confirmed
    # which region is affected." Distinct from a hypothesis (a guess someone
    # floated) and from a conflict (two things said that contradict).
    missing_info: List[str] = Field(default_factory=list)
    conflict: Optional[str] = None
    # Exact text of an existing fact this turn's new information supersedes
    # (e.g. the room confirms an earlier "outage is in the AC region" was
    # wrong). Distinct from conflict: conflict flags a contradiction as
    # still an open question; this is for the later turn that actually
    # resolves it. iCall_service.apply_structuring_update only acts on
    # this when it matches an existing fact's text exactly -- otherwise
    # it's ignored, never used to guess which fact was meant.
    corrects_fact: Optional[str] = None
    identified_speakers: List[str] = Field(default_factory=list)
    is_wrapping_up: bool = False
    spoken_reply: str
    # Optional, additional to spoken_reply -- something worth flagging that
    # doesn't warrant interrupting a conversation that's actively flowing
    # (e.g. a secondary observation, a connection between two facts that
    # isn't urgent). Rendered as a written note in the call UI, never
    # spoken aloud. spoken_reply still always happens; this is a second,
    # non-disruptive channel, not a replacement for it.
    agent_chat_note: Optional[str] = None
    # True when this turn is someone asking to see/pull up the server logs
    # for this incident (any phrasing -- "show me the logs", "what's in the
    # logs since this started", "pull up server logs"), as opposed to just
    # talking about symptoms in prose. iCall_api reacts to this by querying
    # iLogs directly (deterministic DB query, not LLM-guessed data) and
    # broadcasting a shared screen -- this field is only the intent signal.
    wants_log_screen: bool = False


class RoleScore(BaseModel):
    """One category's relevance score for one speaker. Scores are
    independent (not required to sum to 1) -- someone can plausibly score
    high on both "devops" and "backend_engineer"."""

    role: str
    score: float = Field(ge=0, le=1)


class SpeakerRoleClassification(BaseModel):
    """
    One speaker's role inference from everything they said in the call.
    best_role/rationale are what iCall_service actually uses when there's
    no directory match; scores are kept for transparency/debugging, not
    because anything currently consumes the full distribution.
    """

    speaker_uid: str
    speaker_name: str
    scores: List[RoleScore] = Field(default_factory=list)
    best_role: str
    rationale: str


class CallRoleClassification(BaseModel):
    """LLM output shape for one call's end-of-call role inference pass."""

    speakers: List[SpeakerRoleClassification] = Field(default_factory=list)


class ActionItemOwnerAssignment(BaseModel):
    """
    One action item's role-based owner assignment. item_index refers back
    to the position in the unresolved-items list this was asked about, not
    the item's position in the call's full action_items list -- iCall_service
    maps it back.
    """

    item_index: int
    owner_uid: Optional[str] = None
    rationale: str


class ActionItemOwnerAssignments(BaseModel):
    """LLM output shape for the end-of-call role-based ownership pass."""

    assignments: List[ActionItemOwnerAssignment] = Field(default_factory=list)


class UnresolvedRisksSummary(BaseModel):
    """
    LLM output shape for the end-of-call risk synthesis: what's still open
    or uncertain across everything recorded, for the final summary. Not a
    diagnosis or a fix -- see iCall_utils.UNRESOLVED_RISKS_SYSTEM_INSTRUCTION.
    """

    risks: List[str] = Field(default_factory=list)


class ReviewedTicketContent(BaseModel):
    """
    LLM output shape for the pre-Jira-ticket content review (see
    iCall_utils.review_ticket_content). A cleaned-for-external-reading copy
    of the call's recorded state -- never written back to
    IncidentCall.structured_state, only used to build the Jira ticket body,
    so this can never affect the live call's own system-of-record.

    action_items is deliberately excluded: its owner/owner_uid resolution
    already went through a carefully validated two-pass process (see
    iCall_service._reconcile_action_item_owners) and its text is matched
    against elsewhere by exact string (see iCall_utils._stale_unowned_
    action_item) -- letting a review pass reword it risks silently
    breaking that, for no real benefit, so action items go into the ticket
    unedited.
    """

    facts: List[str] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    unresolved_risks: List[str] = Field(default_factory=list)


class ChatCompletionRequest(BaseModel):
    """
    Request body Agora's Conversational AI Engine sends to a Custom LLM
    base_url. Only the fields this proxy actually reads are declared;
    everything else Agora may send is tolerated and ignored.
    """

    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    # Native MCP tool-calling (see voice-agent/server/src/agent.py's
    # mcp_servers config) -- present only when Agora registered MCP
    # server(s) on this agent's LLM. Declared as plain dicts (not typed
    # further) since we only need to hand them to Gemini's function-
    # calling config, not validate their shape ourselves.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None

    class Config:
        extra = "allow"
