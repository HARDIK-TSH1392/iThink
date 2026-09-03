import asyncio
import hashlib
import hmac
from typing import List, Set

from google import genai
from google.genai import types

from app.config import get_settings
from .iCall_schema import ChatMessage, StructuringUpdate

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
# Live structuring: turns iCall from a proven-wiring proxy into the actual
# incident-commander behavior. Mirrors iTriage_utils' Gemini calling pattern
# (primary/fallback model, concurrency cap, response_schema) for consistency
# across the two modules that talk to Gemini.
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

STRUCTURING_SYSTEM_INSTRUCTION = """You are iThink, a voice participant in a live
incident call. Your job is narrow, the same way it is for a human note-taker
who also happens to be allowed to ask one question: keep the room's shared
understanding straight, don't investigate or diagnose.

Hard constraints:
- Never assert or imply a root cause. Never recommend a specific fix,
  rollback, or remediation action. That judgment belongs to the humans on
  the call, not you.
- Only extract facts/hypotheses/decisions/action items that were actually
  said in this turn. Do not invent details.
- A "fact" is something stated with confidence as already true or already
  observed (e.g. "the DB is fine"). A "hypothesis" is a guess or theory
  being floated, not yet confirmed. Keep them separate — conflating a guess
  with a confirmed fact is exactly the failure mode this system exists to
  prevent.
- Set "conflict" only when something said in this turn contradicts a fact
  or hypothesis already recorded in the state you were given below. Phrase
  it as one short, targeted clarifying question you would ask out loud —
  not a statement, not an accusation.
- identified_speakers: only include a name/role if someone actually
  introduced themselves in this turn (e.g. "this is Priya, on-call SRE").
  Do not guess who is speaking from tone or content alone.
- spoken_reply is what you will say out loud right now. If there's a
  conflict, spoken_reply should be that clarifying question. Otherwise keep
  it to a brief, natural acknowledgment — you are a participant, not a
  narrator repeating back everything you heard.
- Output must strictly match the provided response schema.
"""


def _build_structuring_prompt(messages: List[ChatMessage], existing_state: dict) -> str:
    conversation = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return f"""Incident state recorded so far (facts/hypotheses/decisions already
confirmed in this call — use this to detect contradictions, not to repeat):
{existing_state}

Conversation so far:
{conversation}

Extract only what is new in the latest turn, and produce your structured
update per the response schema.
"""


def _fallback_structuring_update() -> StructuringUpdate:
    return StructuringUpdate(spoken_reply=FALLBACK_REPLY)


async def generate_structuring_update(
    messages: List[ChatMessage], existing_state: dict
) -> StructuringUpdate:
    """
    One turn of live structuring: given the conversation and what's already
    recorded for this call, extract new facts/hypotheses/decisions/action
    items, flag a contradiction if one exists, and produce what the agent
    should say. Degrades to a plain fallback reply (not an error) when no
    API key is configured, same reasoning as the original proxy-only version
    this replaces — prove the wiring survives even without a real key.
    """
    if not get_settings().gemini_api_key:
        return _fallback_structuring_update()

    client = _get_client()
    prompt = _build_structuring_prompt(messages, existing_state)
    config = types.GenerateContentConfig(
        system_instruction=STRUCTURING_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=StructuringUpdate,
    )

    async with _gemini_semaphore:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=PRIMARY_MODEL,
                contents=prompt,
                config=config,
            )
        except Exception:
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=config,
                )
            except Exception:
                return _fallback_structuring_update()

    try:
        return StructuringUpdate.model_validate_json(response.text)
    except Exception:
        # Model returned something that didn't match the schema — don't crash
        # the live call over a malformed extraction, just say something safe
        # and record nothing rather than guessing at a partial parse.
        return StructuringUpdate(spoken_reply="Sorry, could you say that again?")


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
