# -*- coding: utf-8 -*-
"""
Agora Agent & Token Service

HTTP APIs:
- GET  /get_config     -> Agent.generate_config()
- POST /startAgent     -> Agent.start()
- POST /stopAgent      -> Agent.stop()
"""
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

# The Agora CLI writes the Python quickstart environment to server/.env.
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_base_dir, '.env'), override=True)

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agora_agent.agentkit.token import generate_convo_ai_token
from agent import Agent

logger = logging.getLogger("uvicorn.error")


def _log_route_error(route: str, exc: Exception, **context) -> None:
    """Log route failures with safe request context and a traceback."""
    safe_context = {key: value for key, value in context.items() if value is not None}
    logger.exception(
        "Request failed route=%s context=%s error_type=%s error=%s",
        route,
        safe_context,
        type(exc).__name__,
        exc,
    )


def _to_http_error(exc: Exception) -> HTTPException:
    """Convert SDK exceptions to HTTP errors"""
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Internal error: {exc}")

try:
    agent = Agent()
except ValueError as e:
    logger.exception(
        "Failed to initialize Agora Agent SDK. Service will fail if endpoints are called without proper configuration: %s",
        e,
    )
    agent = None


# FastAPI application
app = FastAPI(
    title="Agora Agent & Token Service",
    version="2.0.0",
    description="Agora Conversational AI service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


# Request models
class StartAgentRequest(BaseModel):
    """Request body for POST /startAgent"""
    channelName: str
    rtcUid: int
    userUid: int
    parameters: Optional[Dict[str, Any]] = None


class StopAgentRequest(BaseModel):
    """Request body for POST /stopAgent"""
    agentId: str


class SetNameRequest(BaseModel):
    """Request body for POST /setName"""
    channelName: str
    uid: str
    name: str


class RemoveNameRequest(BaseModel):
    """Request body for POST /removeName"""
    channelName: str
    uid: str


# channel_name -> {uid: name}. In-memory and best-effort: lets every
# participant on a call see everyone else's display name without needing
# RTM presence or any coordination ahead of time. Cleared implicitly when
# the process restarts -- fine for a live call, not meant to persist.
_channel_names: Dict[str, Dict[str, str]] = {}


class ChatMessageSend(BaseModel):
    """Request body for POST /sendChatMessage"""
    channelName: str
    uid: str
    name: str
    text: str


# channel_name -> [{uid, name, text, timestamp}, ...]. In-call text chat
# between humans (separate from the voice transcript and from the agent's
# own written notes) -- same in-memory, best-effort, polling-based
# discipline as _channel_names, for the same reasons (no RTM plumbing,
# easy to verify, fine to lose on a process restart).
_channel_chat_messages: Dict[str, List[Dict[str, Any]]] = {}


# API endpoints
def _generate_channel_name() -> str:
    return f"ai-conversation-{int(time.time())}-{random.randint(1000, 9999)}"


@router.get("/get_config")
async def get_config(
    channel: Optional[str] = Query(default=None),
    uid: Optional[int] = Query(default=None),
):
    """Generate connection configuration"""
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Service not properly configured. Please check environment variables.",
        )

    try:
        # Agora RTC accepts uid=0 as "auto assign", but RTM token subjects cannot
        # use 0. Replace missing, zero, or negative values with a generated UID.
        user_uid = random.randint(1000, 9999999) if uid is None or uid <= 0 else uid
        agent_uid = str(random.randint(10000000, 99999999))
        channel_name = channel or _generate_channel_name()

        # Get credentials from environment
        app_id = os.getenv("AGORA_APP_ID")
        app_certificate = os.getenv("AGORA_APP_CERTIFICATE")

        # Generate a one-hour RTC+RTM token and renew it client-side as needed.
        token = generate_convo_ai_token(
            app_id=app_id,
            app_certificate=app_certificate,
            channel_name=channel_name,
            uid=user_uid,
            token_expire=3600,
        )

        config_data = {
            "app_id": app_id,
            "token": token,
            "uid": str(user_uid),
            "channel_name": channel_name,
            "agent_uid": agent_uid,
        }

        return {
            "code": 0,
            "data": config_data,
            "msg": "success",
        }
    except Exception as e:
        _log_route_error("/get_config", e, channel=channel, uid=uid)
        raise _to_http_error(e)


@router.post("/startAgent")
async def start_agent(request: StartAgentRequest):
    """Start agent in a channel"""
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Service not properly configured. Please check environment variables.",
        )

    try:
        output_audio_codec = None
        if request.parameters:
            output_audio_codec = request.parameters.get("output_audio_codec")

        result = await agent.start(
            channel_name=request.channelName,
            agent_uid=request.rtcUid,
            user_uid=request.userUid,
            output_audio_codec=output_audio_codec,
        )
        return {"code": 0, "msg": "success", "data": result}
    except Exception as e:
        _log_route_error(
            "/startAgent",
            e,
            channelName=request.channelName,
            rtcUid=request.rtcUid,
            userUid=request.userUid,
        )
        raise _to_http_error(e)


@router.post("/stopAgent")
async def stop_agent(request: StopAgentRequest):
    """Stop agent by ID"""
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Service not properly configured. Please check environment variables.",
        )

    try:
        await agent.stop(request.agentId)
        return {"code": 0, "msg": "success"}
    except Exception as e:
        _log_route_error("/stopAgent", e, agentId=request.agentId)
        raise _to_http_error(e)


async def _fetch_late_joiner_catchup(channel_name: str) -> Dict[str, Any]:
    """
    A participant just joined a channel where others were already present
    -- fetch iThink's live catch-up (text recap + any shared screens shown
    so far, e.g. GitHub commits or server logs pulled up earlier in the
    call) so set_name can hand it straight back to that one joiner (never
    broadcast through _channel_chat_messages, which every participant
    polls -- a late-join catch-up is meant for the joiner alone, not a
    message "sent" to the room). Best-effort throughout: a slow/
    unreachable iThink backend should never block someone from joining.
    """
    ithink_base = os.getenv("ITHINK_BACKEND_BASE_URL", "http://127.0.0.1:8123/api/v1")
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(f"{ithink_base}/icall/channel/{channel_name}/recap")
            response.raise_for_status()
            data = response.json().get("data", {})
            return {
                "recap": data.get("recap"),
                "sharedScreens": data.get("shared_screens", []),
            }
    except Exception:
        logger.warning("Failed to fetch late-joiner catch-up for channel=%s", channel_name, exc_info=True)
        return {"recap": None, "sharedScreens": []}


@router.post("/setName")
async def set_name(request: SetNameRequest):
    """
    Record a participant's display name for a channel. When this is a
    genuine late join, the response also carries a private catch-up (text
    recap + past shared screens) for this caller only -- see
    _fetch_late_joiner_catchup.
    """
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required and cannot be empty")

    existing_names = _channel_names.setdefault(request.channelName, {})
    # A late join: this uid is new to the channel AND someone else was
    # already on the call -- i.e. there's a real chance discussion already
    # happened. A single first joiner naturally has an empty structured_state
    # anyway (format_live_recap returns None), but checking here avoids the
    # round-trip entirely for the common case.
    is_late_join = request.uid not in existing_names and len(existing_names) > 0
    existing_names[request.uid] = name

    catchup = await _fetch_late_joiner_catchup(request.channelName) if is_late_join else {
        "recap": None,
        "sharedScreens": [],
    }

    return {"code": 0, "msg": "success", "data": catchup}


@router.get("/getNames")
async def get_names(channel: str = Query(...)):
    """Return the uid -> name map recorded so far for a channel."""
    return {"code": 0, "data": _channel_names.get(channel, {}), "msg": "success"}


@router.post("/removeName")
async def remove_name(request: RemoveNameRequest):
    """
    Drops a participant from a channel's name registry -- called when
    someone actually leaves, so getNames (and the pre-call "N already on
    the call" count) reflects who's still there instead of accumulating
    everyone who's ever joined. Best-effort, same as setName: silently a
    no-op if the channel or uid was never recorded.
    """
    _channel_names.get(request.channelName, {}).pop(request.uid, None)
    return {"code": 0, "msg": "success"}


@router.post("/sendChatMessage")
async def send_chat_message(request: ChatMessageSend):
    """Append one in-call chat message for a channel."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required and cannot be empty")
    _channel_chat_messages.setdefault(request.channelName, []).append(
        {
            "uid": request.uid,
            "name": request.name,
            "text": text,
            "timestamp": time.time() * 1000,
        }
    )
    return {"code": 0, "msg": "success"}


@router.get("/chatMessages")
async def get_chat_messages(channel: str = Query(...)):
    """Return this channel's in-call chat history so far, oldest first."""
    return {"code": 0, "data": _channel_chat_messages.get(channel, []), "msg": "success"}


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
