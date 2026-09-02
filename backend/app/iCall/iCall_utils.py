import asyncio
import hashlib
import hmac
from typing import List, Set

from google import genai
from google.genai import types

from app.config import get_settings
from .iCall_schema import ChatMessage

# -----------------------------------------------------------------------------
# Call lifecycle status constants
# -----------------------------------------------------------------------------
# Deliberately simpler than iNcidents' state machine: a call only ever moves
# forward (scheduled -> in_progress -> completed), with no approval gate of
# its own — the incident-level approval already happened before a call is
# ever created.

CALL_STATUS_SCHEDULED = "scheduled"
CALL_STATUS_IN_PROGRESS = "in_progress"
CALL_STATUS_COMPLETED = "completed"

ALL_CALL_STATUSES: Set[str] = {
    CALL_STATUS_SCHEDULED,
    CALL_STATUS_IN_PROGRESS,
    CALL_STATUS_COMPLETED,
}


def generate_channel_name(incident_id: int) -> str:
    """
    Deterministic Agora RTC channel name for a given incident.

    This is the single source of truth for "what is this incident's room
    called" — both the voice-agent join code and the orchestration
    (email/calendar invite) code read this same value via the IncidentCall
    row rather than each independently computing their own formula.
    """
    return f"incident-{incident_id}"


# -----------------------------------------------------------------------------
# Custom-LLM proxy: proof-of-wiring only, no structuring/conflict logic yet.
# Mirrors iTriage_utils' Gemini calling pattern (primary/fallback model,
# concurrency cap) for consistency across the two modules that talk to Gemini.
# -----------------------------------------------------------------------------

PRIMARY_MODEL = "gemini-3.7-flash"
FALLBACK_MODEL = "gemini-2.5-flash"

_gemini_semaphore = asyncio.Semaphore(4)

_client: "genai.Client | None" = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_settings().gemini_api_key)
    return _client


FALLBACK_REPLY = (
    "I can hear you, but I'm not wired to a real model yet — "
    "set GEMINI_API_KEY to get an actual response."
)


async def generate_chat_reply(messages: List[ChatMessage]) -> str:
    """
    Proxy-only first version: forward the conversation to Gemini and return
    plain text. No structuring, no conflict detection — this step exists
    only to prove Agora's Custom LLM is actually calling our server. Real
    analysis logic is a separate, not-yet-settled prompt-engineering step.

    Degrades to FALLBACK_REPLY (rather than raising) when no API key is
    configured, so the wiring can be proven end-to-end before a real key
    is available.
    """
    if not get_settings().gemini_api_key:
        return FALLBACK_REPLY

    client = _get_client()
    contents = "\n".join(f"{m.role}: {m.content}" for m in messages)

    async with _gemini_semaphore:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=PRIMARY_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a placeholder voice-agent brain used only to "
                        "prove the Agora Custom LLM wiring works end to end. "
                        "Keep replies to one short sentence."
                    ),
                ),
            )
        except Exception:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=FALLBACK_MODEL,
                contents=contents,
            )

    return response.text or FALLBACK_REPLY


# -----------------------------------------------------------------------------
# Agora platform webhooks (Console -> Project -> notification config).
# Event type numbers per Agora's Conversational AI event-notifications docs.
# -----------------------------------------------------------------------------

EVENT_AGENT_JOINED = 101
EVENT_AGENT_LEFT = 102
EVENT_DIALOGUE_HISTORY = 103
EVENT_AGENT_ERROR = 110
EVENT_PERFORMANCE_METRICS = 111
EVENT_INCOMING_CALL_STATUS = 201
EVENT_OUTGOING_CALL_STATUS = 202

EVENT_TYPE_NAMES = {
    EVENT_AGENT_JOINED: "agent_joined",
    EVENT_AGENT_LEFT: "agent_left",
    EVENT_DIALOGUE_HISTORY: "dialogue_history",
    EVENT_AGENT_ERROR: "agent_error",
    EVENT_PERFORMANCE_METRICS: "performance_metrics",
    EVENT_INCOMING_CALL_STATUS: "incoming_call_status",
    EVENT_OUTGOING_CALL_STATUS: "outgoing_call_status",
}


def describe_event_type(event_type: int) -> str:
    return EVENT_TYPE_NAMES.get(event_type, f"unknown({event_type})")


def verify_agora_signature(raw_body: bytes, signature_header: str, secret: str) -> bool:
    """
    Verify Agora-Signature-V2 (HMAC/SHA256 of the raw request body, hex
    digest) against the secret shown in Agora Console when the webhook was
    registered. Uses hmac.compare_digest to avoid timing attacks.

    NOT independently confirmed against a real Agora webhook yet — hex
    digest is the standard convention, but this endpoint hasn't received a
    real signed request since no webhook is registered (that needs a public
    HTTPS URL, which local dev doesn't have). Re-verify the encoding the
    first time a real webhook actually reaches this server.
    """
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header or "")
