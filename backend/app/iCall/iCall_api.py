import json
import logging
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any, AsyncGenerator, List, Optional, Tuple

from app.config import get_settings
from app.database import async_session
from .iCall_model import IncidentCall
from .iCall_schema import (
    IncidentCallRead,
    CallStatusUpdate,
    CallUtteranceCreate,
    CallUtteranceRead,
    AgentUtteranceRead,
    ChatCompletionRequest,
    AgoraWebhookEvent,
)
from .iCall_service import (
    get_or_create_call,
    get_call,
    get_call_by_channel_name,
    update_call_status,
    record_utterance,
    list_utterances,
    apply_structuring_update,
    record_health_score,
    record_pattern_nudge,
    record_shared_screen,
    record_agent_utterance,
    list_agent_utterances,
    record_missing_info_nudge,
    record_direct_address_reply,
    record_silence_streak,
    record_wrapped_up,
    get_call_turn_lock,
    infer_and_store_participant_roles,
    IncidentNotFoundError,
)
from .iCall_utils import (
    generate_structuring_update,
    describe_event_type,
    verify_agora_signature,
    format_live_recap,
    build_silence_prompt,
    _call_needs_check_in,
    get_live_participant_count,
    broadcast_shared_screen,
    _is_silence_trigger,
    should_speak_aloud,
    describe_speak_reason,
    build_gated_spoken_reply,
    evaluate_call_patterns,
    compute_coordination_health_score,
    detect_health_score_drop,
    build_health_recap,
    build_correction_callout,
    redact_sensitive_reply,
    build_keyterms,
    transcribe_delegate_voice_note,
    CALL_STATUS_COMPLETED,
    EVENT_AGENT_LEFT,
    CLOSING_LINE,
    FALLBACK_REPLY,
    MODEL_UNAVAILABLE_REPLY,
    MALFORMED_RESPONSE_FALLBACK,
)
from app.iNcidents.iNcidents_crudl import get_incident
from app.iNcidents.iNcidents_utils import STATUS_AWAITING_APPROVAL
from app.iLogs.iLogs_crudl import list_logs
from app.iOrchestrate.iOrchestrate_utils import (
    post_call_summary_notification,
    notify_jira_approval_needed,
    notify_delegate_review_needed,
    approve_incident_with_delegate_notes,
)

router = APIRouter(prefix="/icall", tags=["iCall"])


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a DB session per request.
    """
    async with async_session() as session:
        yield session


@router.post("/incidents/{incident_id}/call", response_model=IncidentCallRead)
async def get_or_create_call_endpoint(
    incident_id: int,
    db: AsyncSession = Depends(get_db),
) -> IncidentCallRead:
    """
    Get this incident's call, creating it (and its channel name) if it
    doesn't exist yet. Orchestration calls this to get the channel name for
    the meeting invite; the voice agent calls this to know which channel to
    join. Both get the same answer because there's one row, not two guesses.
    """
    try:
        call = await get_or_create_call(db, incident_id)
    except IncidentNotFoundError:
        raise HTTPException(status_code=404, detail="Incident not found")
    return IncidentCallRead.model_validate(call)


@router.get("/{call_id}", response_model=IncidentCallRead)
async def get_call_endpoint(
    call_id: int,
    db: AsyncSession = Depends(get_db),
) -> IncidentCallRead:
    """
    Get a single call by ID.
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return IncidentCallRead.model_validate(call)


async def _apply_status_transition(db: AsyncSession, call: IncidentCall, new_status: str) -> IncidentCall:
    """
    Shared by both status endpoints (numeric and channel-keyed). Guards the
    "just reached completed" side effects against firing more than once --
    with a call now shared by multiple participants, "End Conversation"
    fires this once per person leaving, not once for the whole room (see
    voice-agent's LandingPage: killing the agent on the first leave was
    already wrong for the same reason, fixed earlier). Role inference is
    safe to re-run every time (it just recomputes from however much
    transcript exists so far, improving as more people leave); the Slack/
    Jira notifications are not, and must only ever fire on the transition.
    """
    was_already_completed = call.status == CALL_STATUS_COMPLETED
    call = await update_call_status(db, call, new_status)

    if call.status == CALL_STATUS_COMPLETED:
        call = await infer_and_store_participant_roles(db, call)

        if not was_already_completed:
            incident = await get_incident(db, call.incident_id)
            if incident:
                # Public channel summary always posts either way -- only
                # the private Jira-approval path branches on delegate mode.
                await post_call_summary_notification(incident, call)
                if incident.delegate_notes:
                    await notify_delegate_review_needed(db, incident, call)
                else:
                    await notify_jira_approval_needed(db, incident, call)

    return call


@router.patch("/{call_id}/status", response_model=IncidentCallRead)
async def update_call_status_endpoint(
    call_id: int,
    payload: CallStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> IncidentCallRead:
    """
    Advance a call's lifecycle status (scheduled -> in_progress -> completed).
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    call = await _apply_status_transition(db, call, payload.status)
    return IncidentCallRead.model_validate(call)


@router.patch("/channel/{channel_name}/status", response_model=IncidentCallRead)
async def update_call_status_by_channel_endpoint(
    channel_name: str,
    payload: CallStatusUpdate,
    db: AsyncSession = Depends(get_db),
) -> IncidentCallRead:
    """
    Same as update_call_status_endpoint, keyed by channel_name -- the
    client (voice-agent web) only ever knows its Agora channel, never the
    numeric call id.
    """
    call = await get_call_by_channel_name(db, channel_name)
    if not call:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    call = await _apply_status_transition(db, call, payload.status)
    return IncidentCallRead.model_validate(call)


@router.post("/{call_id}/utterances", response_model=CallUtteranceRead)
async def record_utterance_endpoint(
    call_id: int,
    payload: CallUtteranceCreate,
    db: AsyncSession = Depends(get_db),
) -> CallUtteranceRead:
    """
    Record one attributed transcript line for a call.
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    utterance = await record_utterance(db, call_id, payload)
    return CallUtteranceRead.model_validate(utterance)


@router.post("/channel/{channel_name}/utterances", response_model=CallUtteranceRead)
async def record_utterance_by_channel_endpoint(
    channel_name: str,
    payload: CallUtteranceCreate,
    db: AsyncSession = Depends(get_db),
) -> CallUtteranceRead:
    """
    Same as record_utterance_endpoint, keyed by channel_name instead of
    call_id -- the client only ever knows its Agora channel name, the same
    way the chat-completions proxy above resolves calls by channel rather
    than requiring the browser to know a numeric call id.
    """
    call = await get_call_by_channel_name(db, channel_name)
    if not call:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    utterance = await record_utterance(db, call.id, payload)
    return CallUtteranceRead.model_validate(utterance)


@router.get("/channel/{channel_name}/chat-notes")
async def list_chat_notes_endpoint(
    channel_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Written notes the agent chose not to interrupt the call to say out
    loud (StructuringUpdate.agent_chat_note) -- polled by the client to
    render alongside the transcript. Channel-keyed for the same reason as
    everything else the browser calls directly.
    """
    call = await get_call_by_channel_name(db, channel_name)
    if not call:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    notes = (call.structured_state or {}).get("chat_notes", [])
    return {"code": 0, "data": notes, "msg": "success"}


@router.get("/channel/{channel_name}/recap")
async def get_live_recap_endpoint(
    channel_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Plain-text catch-up of the call's structured_state so far, for a
    participant joining mid-discussion. Called by the voice-agent service
    (not the browser) right after a late joiner registers their name --
    see voice-agent/server/src/server.py's set_name. Channel-keyed for the
    same reason as chat-notes: the caller only knows the channel name.

    Also returns shared_screens (GitHub commits / server logs already
    shown to the room, see record_shared_screen) so a late joiner's client
    can render the same screens the room already saw, even if the live RTM
    broadcast happened before they were subscribed to the channel.
    """
    call = await get_call_by_channel_name(db, channel_name)
    if not call:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    state = call.structured_state or {}
    recap = format_live_recap(state)
    shared_screens = state.get("shared_screens", [])
    return {"code": 0, "data": {"recap": recap, "shared_screens": shared_screens}, "msg": "success"}


@router.get("/channel/{channel_name}/keyterms")
async def get_keyterms_endpoint(
    channel_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Deepgram keyterm-prompting string for this call (see
    iCall_utils.build_keyterms). Called by the voice-agent service right
    before it starts the STT vendor for a call -- channel-keyed for the
    same reason as recap/chat-notes, the caller only knows the channel
    name at that point, not the incident id.
    """
    call = await get_call_by_channel_name(db, channel_name)
    if not call:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    incident = await get_incident(db, call.incident_id)
    service = incident.service if incident else None
    return {"code": 0, "data": {"keyterm": build_keyterms(service)}, "msg": "success"}


@router.get("/channel/{channel_name}/delegate")
async def get_delegate_endpoint(
    channel_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delegate-mode notes for this call, if the resolved approver approved
    but couldn't join (see iOrchestrate_api's approve_delegate/modal flow).
    Called by voice-agent/server right before it starts the agent, so it
    can enrich the opening greeting -- and, when an avatar vendor is
    configured, front the session with a visual stand-in -- with what the
    absent lead wants covered. Same best-effort shape as /keyterms: no
    delegate notes is the common case, not an error.
    """
    call = await get_call_by_channel_name(db, channel_name)
    if not call:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    incident = await get_incident(db, call.incident_id)
    notes = incident.delegate_notes if incident else None
    approver_name = None
    if notes:
        from app.iDirectory.iDirectory_crudl import resolve_approver

        approver = await resolve_approver(db, incident.service)
        approver_name = approver.name if approver else None

    return {
        "code": 0,
        "data": {"delegate_notes": notes, "approver_name": approver_name},
        "msg": "success",
    }


@router.post("/incidents/{incident_id}/delegate-voice-note")
async def transcribe_delegate_voice_note_endpoint(
    incident_id: int,
    audio: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Voice alternative to typing delegate notes into the Slack modal (see
    demo/delegate-voice.html, linked from open_slack_delegate_modal).
    Transcribe-only -- no side effects, never touches approval state.
    The page shows the transcript back for review/edit before the human
    explicitly confirms via POST .../delegate-approve below, same "see it
    before it's final" discipline as the Slack modal's own text field.
    """
    incident = await get_incident(db, incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    if incident.status != STATUS_AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"Incident {incident_id} is no longer awaiting approval (status: {incident.status})",
        )

    audio_bytes = await audio.read()
    transcript = await transcribe_delegate_voice_note(audio_bytes, audio.content_type or "audio/webm")
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="Couldn't make out any speech in that recording -- try again, speaking clearly.",
        )
    return {"code": 0, "data": {"transcript": transcript}, "msg": "success"}


@router.post("/incidents/{incident_id}/delegate-approve")
async def delegate_approve_endpoint(
    incident_id: int,
    notes: str = Form(...),
    approved_by: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Confirms delegate notes (typed, spoken-then-transcribed, or edited
    after either) and approves the incident in one step -- the same
    shared action the Slack modal's Submit button triggers (see
    iOrchestrate_utils.approve_incident_with_delegate_notes), just reached
    from a plain web form instead of Slack. Blank notes safely no-op
    rather than approving with nothing -- see that function's own guard.
    """
    ok = await approve_incident_with_delegate_notes(db, incident_id, notes, approved_by)
    if not ok:
        incident = await get_incident(db, incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
        if incident.status != STATUS_AWAITING_APPROVAL:
            raise HTTPException(
                status_code=409,
                detail=f"Incident {incident_id} is no longer awaiting approval (status: {incident.status})",
            )
        raise HTTPException(status_code=422, detail="Notes can't be empty")
    return {"code": 0, "msg": "success"}


@router.get("/{call_id}/utterances", response_model=List[CallUtteranceRead])
async def list_utterances_endpoint(
    call_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[CallUtteranceRead]:
    """
    List a call's transcript, ordered by turn.
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    utterances = await list_utterances(db, call_id)
    return [CallUtteranceRead.model_validate(u) for u in utterances]


@router.get("/{call_id}/agent-utterances", response_model=List[AgentUtteranceRead])
async def list_agent_utterances_endpoint(
    call_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[AgentUtteranceRead]:
    """
    List everything the agent actually said in a call, in order, with why
    (see AgentUtterance.reason) -- the answer to "what did the agent
    actually say" that didn't exist before this endpoint (see
    record_agent_utterance).
    """
    call = await get_call(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    utterances = await list_agent_utterances(db, call_id)
    return [AgentUtteranceRead.model_validate(u) for u in utterances]


async def _push_shared_screen(db: AsyncSession, call: IncidentCall, channel_name: str, screen: dict) -> IncidentCall:
    """Records + broadcasts one shared-screen event; returns the (possibly updated) call."""
    call = await record_shared_screen(db, call, screen)
    await broadcast_shared_screen(channel_name, screen)
    return call


async def _maybe_push_shared_screens(
    db: AsyncSession,
    call: IncidentCall,
    channel_name: str,
    incident,
    update,
    deploy_check_result: Optional[dict],
) -> IncidentCall:
    """
    Reacts to this turn's two "show me something" signals -- a GitHub
    deploy-question that actually got real commit data back (deterministic,
    already computed as part of generate_structuring_update), and
    update.wants_log_screen (LLM-judged intent, deliberately broader/more
    flexible phrasing than the deploy-keyword check since natural ways to
    ask to see logs vary a lot more than "did we deploy/ship/commit...").
    Both push a shared screen to everyone on the call; see
    iCall_utils.broadcast_shared_screen.
    """
    now = datetime.now(timezone.utc).isoformat()

    if deploy_check_result and deploy_check_result.get("commits"):
        screen = {
            "type": "github_commits",
            "title": f"Recent commits: {deploy_check_result.get('repo', '')}",
            "commits": deploy_check_result["commits"],
            "timestamp": now,
        }
        call = await _push_shared_screen(db, call, channel_name, screen)

    if update.wants_log_screen and incident is not None:
        logs = await list_logs(
            db,
            service=incident.service,
            region=incident.region,
            since=incident.created_at,
            limit=50,
        )
        screen = {
            "type": "logs",
            "title": f"Server logs for {incident.service} since the incident started",
            "logs": [
                {
                    "id": log.id,
                    "severity": log.severity,
                    "message": log.message,
                    "timestamp": log.timestamp.isoformat(),
                    "source_id": log.source_id,
                }
                for log in logs
            ],
            "timestamp": now,
        }
        call = await _push_shared_screen(db, call, channel_name, screen)

    return call


async def _decide_spoken_reply(
    db: AsyncSession,
    call: IncidentCall,
    update,
    old_facts: List[str],
    latest_user_message: Optional[str],
    health_score: int,
    force_speak: bool = False,
) -> str:
    """
    Given a StructuringUpdate already merged into call.structured_state,
    decide what (if anything) actually gets spoken. Shared by both a
    normal turn and a tool-result follow-up turn (see _process_turn and
    _process_tool_result_turn) -- the decision logic is identical either
    way, only how `update` got produced differs.
    """
    # The LLM only detects *that* the room sounds like it's wrapping up;
    # the actual words are ours, not a paraphrase, so what's promised about
    # Slack/Jira is always accurate. Diagnostic fallbacks (no API key
    # configured, or a malformed model response) always speak too -- they're
    # meta-signals about the system itself, not ordinary conversational
    # content the gate is meant to quiet down.
    if update.is_wrapping_up:
        # Guard against re-speaking the recap: confirmed live (incident-101,
        # call_id=14) that separate low-content filler turns ("Okay. Nice.",
        # "Okay. Thank you.") each independently get classified as
        # is_wrapping_up by the model, which has no memory across turns of
        # having already said the recap once -- fired 4x identically in one
        # real call. wrapped_up is the durable, code-side memory of "already
        # spoke it," same role record_wrapped_up already plays for the
        # silence-trigger branch (see its wrapped_up check above). Once
        # wrapped_up, later filler is genuinely nothing to say -- the recap
        # already landed, and nudging/pattern logic below is about steering
        # an ongoing incident, not a call that's already been summarized.
        if (call.structured_state or {}).get("wrapped_up"):
            return ""
        # The actually appropriate moment for a real content recap is right
        # here, not 30-60s of post-goodbye silence later (see the
        # silence-trigger branch above) -- this IS both "spoken status
        # summaries at appropriate moments" and "a final incident summary"
        # from the brief, landing in the one moment that's unambiguously
        # right for it. CLOSING_LINE's own promise about Slack/Jira stays
        # exactly as accurate as before; this just says what actually
        # happened before promising where the fuller version goes.
        spoken_reply = redact_sensitive_reply(f"{build_health_recap(call.structured_state)} {CLOSING_LINE}")
        call = await record_wrapped_up(db, call)
        await record_agent_utterance(db, call.id, spoken_reply, "is_wrapping_up")
        return spoken_reply

    if update.spoken_reply in (FALLBACK_REPLY, MODEL_UNAVAILABLE_REPLY, MALFORMED_RESPONSE_FALLBACK):
        spoken_reply = update.spoken_reply
        fallback_names = {
            FALLBACK_REPLY: "FALLBACK_REPLY",
            MODEL_UNAVAILABLE_REPLY: "MODEL_UNAVAILABLE_REPLY",
            MALFORMED_RESPONSE_FALLBACK: "MALFORMED_RESPONSE_FALLBACK",
        }
        await record_agent_utterance(
            db, call.id, spoken_reply, f"fallback:{fallback_names[spoken_reply]}"
        )
        return spoken_reply

    # Fixed, deterministic callout, checked ahead of the generic gate the
    # same way is_wrapping_up/the fallbacks above are -- a state change this
    # significant (the room's shared understanding just flipped) can't be
    # left to update.spoken_reply's free-text phrasing. Only fires when the
    # correction actually applied (old_facts contained the exact text) --
    # see apply_structuring_update's fail-safe exact-match requirement.
    if update.corrects_fact and update.corrects_fact in old_facts:
        spoken_reply = redact_sensitive_reply(build_correction_callout(update))
        await record_agent_utterance(db, call.id, spoken_reply, "correction")
        return spoken_reply

    # A tool-result turn is always the answer to something the model itself
    # just chose to go look up -- confirmed live that the generic gate's
    # direct-address cooldown (meant to stop the SAME utterance getting
    # answered twice by fragmented STT turns) also silently ate the actual
    # answer here: the ack ("Pulling up the logs...") starts the cooldown,
    # and the tool round-trip routinely finishes inside that window, so the
    # real report-back got gated as if it were a duplicate of the ack. Same
    # "bypass the generic gate with a dedicated deterministic path" pattern
    # as is_wrapping_up/corrects_fact above -- this just always speaks
    # whatever the model actually has to report, when it has anything to say.
    if force_speak and update.spoken_reply:
        # Highest-risk path for this: it's a direct readback of a tool/log
        # result the model just fetched, which is exactly the kind of raw
        # content a real credential could be sitting inside.
        spoken_reply = redact_sensitive_reply(update.spoken_reply)
        await record_agent_utterance(db, call.id, spoken_reply, "tool_result")
        return spoken_reply

    if should_speak_aloud(update, latest_user_message, call.structured_state):
        reason = describe_speak_reason(update, latest_user_message, call.structured_state)
        spoken_reply = redact_sensitive_reply(build_gated_spoken_reply(update, reason, call.structured_state))
        if reason == "missing_info":
            call = await record_missing_info_nudge(db, call)
        elif reason == "direct_address":
            call = await record_direct_address_reply(db, call)
        await record_agent_utterance(db, call.id, spoken_reply, reason)
        return spoken_reply

    # Nothing about THIS turn was urgent -- but the accumulated state
    # might still be worth flagging (a pattern across recorded events, or
    # a sharp drop in coordination health). Checked only here, below the
    # turn-level gate, so a pattern nudge never competes with something
    # more directly relevant to what was just said.
    pattern = evaluate_call_patterns(call.structured_state)
    if pattern:
        # record_pattern_nudge's own message param is itself persisted to
        # the timeline (see its docstring) -- pass the redacted text there
        # too, not just what's returned/spoken.
        spoken_reply = redact_sensitive_reply(pattern["message"])
        call = await record_pattern_nudge(db, call, pattern["pattern"], spoken_reply)
        await record_agent_utterance(db, call.id, spoken_reply, f"pattern:{pattern['pattern']}")
        return spoken_reply

    if detect_health_score_drop(call.structured_state):
        spoken_reply = redact_sensitive_reply(build_health_recap(call.structured_state))
        await record_pattern_nudge(db, call, "health_score_drop", spoken_reply, score=health_score)
        await record_agent_utterance(db, call.id, spoken_reply, "health_score_drop")
        return spoken_reply

    # Ordinary turn, nothing urgent -- stay silent. The prompt already
    # asks the model to keep these to "a brief acknowledgment," but that
    # was never enforced; this makes it deterministic. Nothing recorded
    # is affected (spoken_reply was never persisted -- see docstring).
    return ""


def _incident_age_minutes(incident) -> Optional[int]:
    """
    Minutes since this incident was created, or None when there's no
    incident to measure from. See _build_structuring_prompt's age_line --
    a log-lookup tool's own default window is relative to "now," not to
    when the incident actually started, so the model needs this to widen
    since_minutes itself rather than silently missing earlier log entries.
    """
    if incident is None:
        return None
    # Incident.created_at is a naive DateTime column (SQLite has no real
    # tz-aware type) but is always written as UTC (server_default=func.now())
    # -- confirmed live (incident 101: tzinfo=None) -- so it needs stamping
    # as UTC before comparing against an aware now(), or this raises
    # "can't subtract offset-naive and offset-aware datetimes".
    created_at = incident.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - created_at).total_seconds() // 60))


def _find_tool_name(payload: ChatCompletionRequest, tool_call_id: Optional[str]) -> Optional[str]:
    """
    A role="tool" result message doesn't always carry its own tool name --
    look it up from the preceding assistant message's tool_calls entry
    with the matching id, the only place OpenAI's contract guarantees it.
    """
    if not tool_call_id:
        return None
    for message in reversed(payload.messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if tool_call.get("id") == tool_call_id:
                return (tool_call.get("function") or {}).get("name")
    return None


async def _maybe_push_tool_result_screen(
    db: AsyncSession, call: IncidentCall, channel_name: str, tool_name: Optional[str], tool_result_text: str
) -> IncidentCall:
    """
    Shows everyone on the call the actual result of a native MCP tool call
    (see chat_completions_endpoint's tool-result branch). get_recent_logs
    is our own tool (backend/ilogs_mcp_service/) so its exact shape is
    known and rendered as a proper "logs" screen, same as the deterministic
    logs path; any GitHub tool's result is shown as a generic "tool_result"
    screen instead of forcing it into the commits-specific shape, since
    GitHub's MCP surface has ~30 tools with different return shapes and
    only some of them (search_commits, get_commit) are actually commits.
    """
    if not tool_name:
        return call

    now = datetime.now(timezone.utc).isoformat()

    if tool_name == "get_recent_logs":
        try:
            logs = json.loads(tool_result_text)
        except (json.JSONDecodeError, TypeError):
            logs = []
        screen = {
            "type": "logs",
            "title": "Server logs (via native MCP tool call)",
            "logs": logs if isinstance(logs, list) else [],
            "timestamp": now,
        }
    else:
        screen = {
            "type": "tool_result",
            "title": f"Tool result: {tool_name}",
            "text": tool_result_text[:4000],
            "timestamp": now,
        }

    return await _push_shared_screen(db, call, channel_name, screen)


async def _process_tool_result_turn(
    db: AsyncSession,
    call: IncidentCall,
    channel_name: str,
    payload: ChatCompletionRequest,
    tool_message,
) -> Tuple[Optional[str], None]:
    """
    Follow-up request after the model called a native MCP tool on a prior
    turn (see _process_turn) and Agora executed it -- this request's last
    message carries the tool's actual result. Runs the same structuring
    pipeline as an ordinary turn, just with the tool's result injected as
    context instead of new conversation to extract facts from, and pushes
    a shared screen from the real result data.
    """
    incident = await get_incident(db, call.incident_id)
    service = incident.service if incident else None
    region = incident.region if incident else None
    incident_age_minutes = _incident_age_minutes(incident)
    tool_name = tool_message.name or _find_tool_name(payload, tool_message.tool_call_id)
    tool_result_text = tool_message.content or ""

    old_facts = list((call.structured_state or {}).get("facts", []))

    result = await generate_structuring_update(
        payload.messages, call.structured_state or {}, service, tool_result_text=tool_result_text,
        region=region, incident_age_minutes=incident_age_minutes,
    )
    update = result.update
    call = await apply_structuring_update(db, call, update)
    call = await _maybe_push_tool_result_screen(db, call, channel_name, tool_name, tool_result_text)

    health_score = compute_coordination_health_score(call.structured_state)
    call = await record_health_score(db, call, health_score)

    latest_user_message = next(
        (m.content for m in reversed(payload.messages) if m.role == "user"), None
    )
    logger.warning(
        "channel=%s tool-result turn latest_user_message=%r",
        channel_name, latest_user_message,
    )
    spoken_reply = await _decide_spoken_reply(
        db, call, update, old_facts, latest_user_message, health_score, force_speak=True,
    )
    return spoken_reply, None


async def _process_turn(
    db: AsyncSession, call: IncidentCall, channel_name: str, payload: ChatCompletionRequest
) -> Tuple[Optional[str], Optional[Any]]:
    """
    One turn's worth of chat_completions_endpoint's work -- generate (or
    skip) a structuring update, merge it into structured_state, and decide
    what (if anything) gets said. Pulled out of the endpoint so it can run
    entirely inside get_call_turn_lock's per-call lock (see that function's
    docstring for why: overlapping turns racing on the same call's
    structured_state is a real, observed lost-update bug, not
    hypothetical). Returns (spoken_reply, tool_call) -- exactly one of the
    two is set: tool_call when the model decided to call an Agora-native
    MCP tool this turn (see StructuringResult), spoken_reply ("" for
    silence) otherwise.
    """
    last_message = payload.messages[-1] if payload.messages else None
    if last_message is not None and last_message.role == "tool":
        return await _process_tool_result_turn(db, call, channel_name, payload, last_message)

    if _is_silence_trigger(payload.messages):
        # Confirmed live (incident-39): the room's silence timer doesn't
        # know the call already ended. A closing-line spoken 77 seconds
        # earlier didn't stop the room-wide 30s silence timer from firing
        # anyway, so a nudge and then a recap both fired into a call
        # nobody was listening to anymore -- worse than saying nothing,
        # since it looked like the "summary" was random and disconnected
        # rather than the actual wrap-up moment. Once wrapped_up is set
        # (see _decide_spoken_reply's is_wrapping_up branch), there's
        # nothing left to nudge about -- the room's already had its
        # recap, right when it actually mattered.
        if (call.structured_state or {}).get("wrapped_up"):
            return "", None

        # The room's gone quiet -- this isn't real speech to extract facts
        # from, so skip generate_structuring_update entirely. Stay silent
        # when there's nobody to nudge (a lone participant), otherwise say
        # something shaped by how far the call has actually gotten and by
        # how long it's STAYED quiet through previous nudges.
        #
        # Event-driven, not clock-driven: a call that's already in good
        # shape gets no proactive silence nudge at all, no matter how long
        # the room's been quiet -- see _call_needs_check_in. Silence in a
        # healthy, resolved call is fine; silence in a stuck one (nothing
        # figured out yet, or a genuinely open conflict/gap) is the thing
        # worth flagging.
        #
        # Escalates rather than repeating: confirmed live (incident-38) the
        # brief nudge fired 10 times verbatim over 5.5 minutes with nothing
        # else said -- not "spoken status summaries at appropriate moments"
        # from the brief, just a loop. 1st trigger in a quiet streak: the
        # existing brief nudge. 2nd: an actual status recap -- this IS the
        # "appropriate moment" for one, the room's had two chances to speak
        # up and hasn't. 3rd+: stay silent -- by then more nagging doesn't
        # help, and repeating a full recap every 30s would be just as
        # inappropriate as repeating the nudge was.
        participant_count = await get_live_participant_count(channel_name)
        if participant_count is not None and participant_count <= 1:
            return "", None
        if not _call_needs_check_in(call.structured_state or {}):
            return "", None
        streak = (call.structured_state or {}).get("silence_streak", 0) + 1
        if streak == 1:
            spoken_reply = build_silence_prompt(call.structured_state or {})
            reason = "silence_check"
        elif streak == 2:
            spoken_reply = build_health_recap(call.structured_state or {})
            reason = "silence_recap"
        else:
            await record_silence_streak(db, call, streak)
            return "", None
        await record_silence_streak(db, call, streak)
        await record_agent_utterance(db, call.id, spoken_reply, reason)
        return spoken_reply, None

    incident = await get_incident(db, call.incident_id)
    service = incident.service if incident else None
    region = incident.region if incident else None
    incident_age_minutes = _incident_age_minutes(incident)

    # Captured before apply_structuring_update mutates structured_state --
    # need the PRE-merge facts list to know whether corrects_fact actually
    # matched and applied (vs. a hallucinated/non-matching reference that
    # apply_structuring_update silently no-ops on), so the spoken callout
    # below never claims a correction happened when nothing was updated.
    old_facts = list((call.structured_state or {}).get("facts", []))

    result = await generate_structuring_update(
        payload.messages, call.structured_state or {}, service, tools=payload.tools,
        region=region, incident_age_minutes=incident_age_minutes,
    )
    if result.tool_call is not None:
        # The model decided to call a native MCP tool instead of answering
        # directly -- nothing to structure or speak yet, Agora executes
        # the tool and sends the result back as a follow-up request (see
        # _process_tool_result_turn above).
        return None, result.tool_call

    update = result.update
    deploy_check_result = result.deploy_check_result
    call = await apply_structuring_update(db, call, update)
    call = await _maybe_push_shared_screens(db, call, channel_name, incident, update, deploy_check_result)

    # Coordination-health score: a rough, code-computed aggregate over the
    # call's own state (open conflicts, missing info, unowned items, time
    # since the last decision) -- recorded every turn, cheaply, regardless
    # of whether anything below actually speaks. See detect_health_score_drop.
    health_score = compute_coordination_health_score(call.structured_state)
    call = await record_health_score(db, call, health_score)

    latest_user_message = next(
        (m.content for m in reversed(payload.messages) if m.role == "user"), None
    )
    logger.warning(
        "channel=%s real turn latest_user_message=%r facts_this_turn=%r",
        channel_name, latest_user_message, update.facts,
    )
    spoken_reply = await _decide_spoken_reply(db, call, update, old_facts, latest_user_message, health_score)
    return spoken_reply, None


def _sse_chunk(completion_id: str, created: int, model: str, delta: dict, finish_reason) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/channel/{channel_name}/llm/chat/completions")
async def chat_completions_endpoint(
    channel_name: str,
    payload: ChatCompletionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Agora Custom LLM target (properties.llm.base_url in the agent's join
    config) — one URL per channel, since Agora's request body carries the
    conversation but no incident/call id. channel_name in the path is what
    resolves this request back to a specific IncidentCall row (see
    iCall_service.get_call_by_channel_name).

    Runs live structuring: extracts new facts/hypotheses/decisions/action
    items, detects contradictions against what's already recorded, and
    streams back what the agent should say — in standard OpenAI
    chat.completion.chunk SSE format, the wire format Agora's engine
    expects here.

    A silence-triggered turn (see _is_silence_trigger) is handled separately
    and never reaches generate_structuring_update -- it isn't real speech to
    extract facts from, just Agora prompting the room after quiet.

    For an ordinary turn, whether the reply actually reaches the room's
    speakers is a separate, deterministic decision from generating it -- see
    should_speak_aloud for turn-level triggers (a conflict, missing info,
    direct address) and evaluate_call_patterns/detect_health_score_drop for
    patterns noticed across everything recorded so far, not just this turn.
    Everything that gets *recorded* (facts, hypotheses, decisions, action
    items, missing_info, conflicts, chat_notes) happens in
    apply_structuring_update regardless of any of these decisions; only the
    audio output is gated.
    """
    call = await get_call_by_channel_name(db, channel_name)
    if call is None:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    # Serializes per-call turn processing -- see get_call_turn_lock's
    # docstring for the lost-update race this closes. Re-fetch after
    # acquiring the lock: `call` above was read before waiting on it, and
    # a turn that queued ahead of this one may have just committed a
    # newer structured_state.
    lock = await get_call_turn_lock(call.id)
    async with lock:
        call = await get_call(db, call.id)
        spoken_reply, tool_call = await _process_turn(db, call, channel_name, payload)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if tool_call is not None:
        # The model decided to call a native MCP tool -- standard OpenAI
        # streaming tool-call format so Agora's engine actually executes
        # it against the registered MCP server and sends the result back
        # as a follow-up request (see _process_tool_result_turn).
        async def tool_call_stream():
            yield _sse_chunk(
                completion_id, created, payload.model,
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "index": 0,
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {"name": tool_call.name, "arguments": json.dumps(dict(tool_call.args or {}))},
                    }],
                },
                None,
            )
            yield _sse_chunk(completion_id, created, payload.model, {}, "tool_calls")
            yield "data: [DONE]\n\n"

        return StreamingResponse(tool_call_stream(), media_type="text/event-stream")

    async def event_stream():
        yield _sse_chunk(completion_id, created, payload.model, {"role": "assistant"}, None)
        yield _sse_chunk(completion_id, created, payload.model, {"content": spoken_reply}, None)
        yield _sse_chunk(completion_id, created, payload.model, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/webhooks/agora")
async def agora_webhook_endpoint(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Registered in Agora Console -> Project -> notification config (needs a
    public HTTPS URL, so this can't be exercised against a real event until
    this server is actually deployed or tunneled). Verifies Agora-Signature-V2
    against AGORA_WEBHOOK_SECRET when configured; accepts unverified with a
    loud warning when it isn't, so local dev isn't blocked on a secret that
    can't exist yet without a registered webhook.

    EVENT_AGENT_LEFT is the only event wired to real behavior: it's Agora's
    own authoritative "the agent actually stopped" signal, a second, more
    reliable path to _apply_status_transition than the client PATCHing call
    status on /channel/{channel_name}/status. That client path misses a
    crashed tab or a dropped connection entirely -- the agent leaves the
    channel, but nothing ever tells this backend the call ended, so role
    inference, ownership assignment, the Slack summary, and the Jira-
    approval gate all silently never run. This closes that gap without
    touching the existing client path (still fires first in the common
    case; _apply_status_transition's was_already_completed guard makes
    the webhook's later call a no-op rather than a duplicate notification).

    Other event types (dialogue history, agent error, etc.) are only logged
    for now -- same "prove the wire, not the logic yet" shape as before.
    """
    raw_body = await request.body()
    settings = get_settings()

    if settings.agora_webhook_secret:
        signature = request.headers.get("Agora-Signature-V2", "")
        if not verify_agora_signature(raw_body, signature, settings.agora_webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
    else:
        print("[iCall webhook] AGORA_WEBHOOK_SECRET not set — accepting unverified (dev only)")

    try:
        event = AgoraWebhookEvent.model_validate_json(raw_body)
    except ValidationError as exc:
        # Confirmed live: Agora Console's own registration health check
        # POSTs a body that doesn't match this schema, and previously got
        # rejected with 422 -- which would fail the health check and block
        # registration outright. A malformed/unrecognized body is never
        # worth a 4xx here: Agora already expects 200 or it retries the
        # (real) event forever, and the console's own probe isn't a real
        # event to act on anyway.
        print(f"[iCall webhook] payload didn't match AgoraWebhookEvent, acking anyway: {exc}")
        return {"status": "ignored", "reason": "unrecognized payload shape"}

    print(
        f"[iCall webhook] {describe_event_type(event.eventType)} "
        f"noticeId={event.noticeId} payload={event.payload}"
    )

    if event.eventType == EVENT_AGENT_LEFT:
        channel_name = event.payload.get("channel")
        if channel_name:
            call = await get_call_by_channel_name(db, channel_name)
            if call:
                await _apply_status_transition(db, call, CALL_STATUS_COMPLETED)
            else:
                print(f"[iCall webhook] agent_left for unknown channel={channel_name!r} -- no matching call")
        else:
            print(f"[iCall webhook] agent_left payload missing 'channel': {event.payload}")

    # Agora expects 200 OK; an un-acked webhook gets retried.
    return {"status": "received"}
