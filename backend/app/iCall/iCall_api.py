import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator, List

from app.config import get_settings
from app.database import async_session
from .iCall_model import IncidentCall
from .iCall_schema import (
    IncidentCallRead,
    CallStatusUpdate,
    CallUtteranceCreate,
    CallUtteranceRead,
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
    infer_and_store_participant_roles,
    IncidentNotFoundError,
)
from .iCall_utils import (
    generate_structuring_update,
    describe_event_type,
    verify_agora_signature,
    CALL_STATUS_COMPLETED,
    CLOSING_LINE,
)
from app.iNcidents.iNcidents_crudl import get_incident
from app.iOrchestrate.iOrchestrate_utils import post_call_summary_notification, notify_jira_approval_needed

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
                await post_call_summary_notification(incident, call)
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
    """
    call = await get_call_by_channel_name(db, channel_name)
    if call is None:
        raise HTTPException(status_code=404, detail=f"No call found for channel '{channel_name}'")

    incident = await get_incident(db, call.incident_id)
    service = incident.service if incident else None

    update = await generate_structuring_update(payload.messages, call.structured_state or {}, service)
    await apply_structuring_update(db, call, update)

    # The LLM only detects *that* the room sounds like it's wrapping up;
    # the actual words are ours, not a paraphrase, so what's promised about
    # Slack/Jira is always accurate.
    spoken_reply = CLOSING_LINE if update.is_wrapping_up else update.spoken_reply

    async def event_stream():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        yield _sse_chunk(completion_id, created, payload.model, {"role": "assistant"}, None)
        yield _sse_chunk(completion_id, created, payload.model, {"content": spoken_reply}, None)
        yield _sse_chunk(completion_id, created, payload.model, {}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/webhooks/agora")
async def agora_webhook_endpoint(request: Request):
    """
    Registered in Agora Console -> Project -> notification config (needs a
    public HTTPS URL, so this can't be exercised against a real event until
    this server is actually deployed or tunneled). Verifies Agora-Signature-V2
    against AGORA_WEBHOOK_SECRET when configured; accepts unverified with a
    loud warning when it isn't, so local dev isn't blocked on a secret that
    can't exist yet without a registered webhook.

    No structuring logic wired to events yet — this just proves receipt and
    logs what arrived, the same "prove the wire, not the logic yet" shape as
    the chat-completions endpoint above.
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
        raise HTTPException(status_code=422, detail=f"Malformed webhook payload: {exc}")

    print(
        f"[iCall webhook] {describe_event_type(event.eventType)} "
        f"noticeId={event.noticeId} payload={event.payload}"
    )

    # Agora expects 200 OK; an un-acked webhook gets retried.
    return {"status": "received"}
