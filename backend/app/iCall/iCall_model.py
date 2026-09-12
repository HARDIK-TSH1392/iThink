from sqlalchemy import String, DateTime, ForeignKey, Integer, JSON, UniqueConstraint, func
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

    # "multi" = Tier 1 (Deepgram, English+Hindi native code-switching, no
    # vendor switch needed). Any other value is one of Sarvam's own
    # target_language_code strings (ta-IN, te-IN, kn-IN, bn-IN, mr-IN,
    # gu-IN, pa-IN, ml-IN, or-IN) -- reused as-is so this value flows
    # straight into SarvamSTT/SarvamTTS with no translation table. Must be
    # a real column, not a structured_state key: it has to be known at
    # agent-start time, before structured_state exists.
    language_code: Mapped[str] = mapped_column(String, default="multi")

    # uid -> {name, directory_matched, directory_title, directory_team,
    # directory_org_role, inferred_role, inferred_scores, rationale,
    # final_role, source}. Populated once, after the call ends (see
    # iCall_service.infer_and_store_participant_roles) -- role inference
    # needs everything a person said across the whole call, not a
    # turn-by-turn guess, so this is deliberately not live state.
    participant_roles: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Set to "awaiting_reply" when this call's post-call review DM is sent
    # to a delegating approver (see iOrchestrate.notify_delegate_review),
    # and cleared back to None once they reply "approve" and the Jira
    # ticket is created. Lets the Slack Events handler (a free-text DM
    # reply carries no action_id/value the way a button click does) find
    # which call a given Slack user's message is actually about -- see
    # iOrchestrate_api's find_awaiting_delegate_review_call.
    delegate_review_status: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)

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

    # Best-known display name at the time the client posted this utterance
    # (from the join-screen name registry, a separate service -- see
    # voice-agent/server's setName/getNames). Denormalized onto each row
    # rather than looked up cross-service, since this backend has no other
    # way to resolve a uid to a name.
    speaker_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    text: Mapped[str] = mapped_column(String)
    turn_index: Mapped[int] = mapped_column(Integer, index=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    __table_args__ = (
        # Also the lookup index for "give me this call's transcript in
        # order" -- a UNIQUE constraint creates one same as a plain Index
        # would. Enforces that a given turn_id (Agora's per-call turn
        # counter, see TranscriptHelperItem.turn_id) is recorded at most
        # once: the browser's own de-dupe (postedTurnIds, a useRef) isn't
        # atomic across two independent mounts of the same component (React
        # StrictMode's double-invoke in dev, or a retried request), so two
        # concurrent posts of the same turn both racing past that check is
        # a real, observed failure mode, not hypothetical -- see
        # record_utterance's IntegrityError handling below.
        UniqueConstraint("call_id", "turn_index", name="uq_call_utterances_call_turn"),
    )


class AgentUtterance(Base):
    """
    One line the agent actually spoke -- deliberately a separate table from
    CallUtterance, not the same one with a special speaker_uid. spoken_reply
    was never persisted anywhere before this (confirmed live: there was no
    way to answer "what did the agent actually say" after a call except by
    reconstructing likely content from the deterministic gate rules).

    Also structurally different from CallUtterance: there's no Agora
    turn_id to key on here (the backend's chat_completions_endpoint sees a
    message history, not Agora's own turn counter), so this has no
    turn_index/unique-constraint scheme -- just an append-only log in
    call order via id/timestamp, written once per turn the gate actually
    decided to speak (never for a turn that stayed silent).
    """

    __tablename__ = "agent_utterances"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    call_id: Mapped[int] = mapped_column(
        ForeignKey("incident_calls.id", ondelete="CASCADE"), index=True
    )

    text: Mapped[str] = mapped_column(String)

    # Which branch of chat_completions_endpoint's gate produced this --
    # "conflict", "missing_info", "action_item_owner", "direct_address",
    # "is_wrapping_up", "pattern:<name>", "health_score_drop",
    # "silence_check", or "fallback:<reply-constant-name>". Same
    # explainability goal as record_pattern_nudge's own reasoning: "why
    # did the AI say that" should be answerable after the fact, not just
    # "it said something."
    reason: Mapped[str] = mapped_column(String, index=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
