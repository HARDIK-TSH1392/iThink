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
    update_call_status,
    record_utterance,
    list_utterances,
    IncidentNotFoundError,
)
from .iCall_utils import generate_chat_reply, describe_event_type, verify_agora_signature

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

    call = await update_call_status(db, call, payload.status)
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


@router.post("/llm/chat/completions")
async def chat_completions_endpoint(payload: ChatCompletionRequest):
    """
    Agora Custom LLM target (properties.llm.base_url in the agent's join
    config). Proxy-only for now: forwards the conversation to Gemini and
    streams the reply back in standard OpenAI chat.completion.chunk SSE
    format, which is the wire format Agora's Conversational AI Engine
    expects here. No structuring/conflict-detection logic yet — this
    endpoint exists purely to prove Agora is calling our own server
    instead of a stock provider, before any real analysis is layered in.
    """
    reply_text = await generate_chat_reply(payload.messages)

    async def event_stream():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        yield _sse_chunk(completion_id, created, payload.model, {"role": "assistant"}, None)
        yield _sse_chunk(completion_id, created, payload.model, {"content": reply_text}, None)
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
