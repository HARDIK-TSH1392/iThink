import asyncio
import hashlib
import hmac
from typing import Dict, List, Optional, Set

import httpx
from google import genai
from google.genai import types

from app.config import get_settings
from .iCall_schema import (
    ChatMessage,
    StructuringUpdate,
    CallRoleClassification,
    ActionItemOwnerAssignments,
    UnresolvedRisksSummary,
)

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


# See iTriage_utils.py's identical constants for why both are the same
# verified model rather than a primary + a guessed fallback name.
PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

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

# Spoken when StructuringUpdate.is_wrapping_up is true. Deliberately a fixed
# string, not LLM-generated -- see StructuringUpdate.is_wrapping_up's
# docstring: the model only detects the moment, this is the guaranteed-
# accurate line about what actually happens next (Slack summary, Jira
# ticket, both gated on human approval per iOrchestrate).
CLOSING_LINE = (
    "Thank you — I've gathered the facts, roles, and action items from this "
    "call. With your approval, I'll share a full summary on Slack and open "
    "tracking tickets on Jira."
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
- If a "GitHub lookup result" is included below, it DIRECTLY ANSWERS
  whatever this turn asked about deploys/commits/ships/releases -- it is
  not background info, it is the answer. When it's present: state what it
  shows as a fact, answer the question in spoken_reply using its actual
  content (commit messages, who, when), and do NOT add a missing_info
  entry about deploys/commits -- the question is already answered, it is
  not missing. Never say "checking" or "looking into it" when the answer
  is right there in the lookup result. Only ignore the lookup result if
  it's about a genuinely unrelated topic.
- missing_info: note a gap only when the room's own conversation implies it
  needs an answer to move forward (e.g. someone asks "which region is
  this?" and nobody answers, or a decision is made that depends on a fact
  nobody has confirmed yet). Not a general checklist of things you wish
  you knew -- only what the room itself is missing.
- is_wrapping_up: true only when this turn itself sounds like the group is
  concluding (explicit goodbyes, "I think that covers it," "let's
  reconvene later") -- not merely because the conversation has been calm
  for a while.
- spoken_reply is what you will say out loud right now. If there's a
  conflict, spoken_reply should be that clarifying question. Otherwise keep
  it to a brief, natural acknowledgment — you are a participant, not a
  narrator repeating back everything you heard. If is_wrapping_up is true,
  spoken_reply is ignored (the caller substitutes a fixed closing line) --
  don't spend effort crafting one.
- agent_chat_note (optional): use this instead of expanding spoken_reply
  when you have something worth flagging that doesn't need to interrupt
  the room -- e.g. a secondary observation, a connection between two facts
  said minutes apart, something useful but not urgent. The room is
  actively talking; a long spoken interjection breaks their flow, but a
  written note doesn't. Leave it null on ordinary turns -- most turns
  don't need one. Never put something here that's genuinely urgent (a
  contradiction, a safety-relevant gap) -- those still belong in
  spoken_reply, said out loud, because a note is easy to miss mid-call.
- Output must strictly match the provided response schema.
"""


# -----------------------------------------------------------------------------
# GitHub deploy-check: "did we ship anything to this service recently" is one
# of the first questions any real incident call asks. Deliberately a
# deterministic keyword trigger, not another LLM call -- this needs to add
# no noticeable latency to a live turn, and "does this sound like a deploy
# question" is narrow enough that a keyword check is both cheap and reliable
# for it, matching the project's deterministic-first discipline. Calls the
# isolated github_mcp_service (see backend/github_mcp_service/) over plain
# HTTP -- kept out of process because the official mcp SDK's SSE support
# conflicts with this backend's pinned FastAPI/starlette.
# -----------------------------------------------------------------------------

DEPLOY_QUESTION_KEYWORDS = (
    "deploy", "deployed", "deployment", "ship", "shipped", "release",
    "released", "commit", "merge", "merged", "push", "pushed",
)

# incident.service -> "owner/repo" on GitHub. Empty until populated -- a
# service with no mapping just means the deploy-check is silently skipped
# for it, not an error.
SERVICE_TO_GITHUB_REPO: Dict[str, str] = {"auth-api": "HARDIK-TSH1392/stark-payment-global"}


def _mentions_deploy_question(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in DEPLOY_QUESTION_KEYWORDS)


async def _check_recent_deploys(repo: str) -> Optional[str]:
    """
    Calls the isolated GitHub MCP bridge service. Returns None (not an
    error) on any failure -- a missing/unreachable deploy-check service
    should never break the live call's own response.
    """
    url = f"{get_settings().github_mcp_service_url.rstrip('/')}/check-recent-deploys"
    try:
        # A full MCP handshake against GitHub's remote server measured
        # ~7.7s in testing -- 8s was cutting it too close and caused
        # spurious timeouts. 20s gives real margin; the isolated service
        # itself also caches sessions per-repo so repeat calls in the same
        # call are much faster than this worst case.
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(url, json={"repo": repo})
            response.raise_for_status()
            data = response.json()
    except Exception:
        return None

    if not data.get("configured"):
        return None
    return data.get("summary")


async def _maybe_check_recent_deploys(messages: List[ChatMessage], service: Optional[str]) -> Optional[str]:
    if not messages or not service:
        return None
    repo = SERVICE_TO_GITHUB_REPO.get(service)
    if not repo:
        return None

    latest_user_message = next(
        (m.content for m in reversed(messages) if m.role == "user"), None
    )
    if not latest_user_message or not _mentions_deploy_question(latest_user_message):
        return None

    return await _check_recent_deploys(repo)


def _build_structuring_prompt(
    messages: List[ChatMessage], existing_state: dict, deploy_check_result: Optional[str] = None
) -> str:
    conversation = "\n".join(f"{m.role}: {m.content}" for m in messages)
    deploy_section = (
        f"\nGitHub lookup result (only mention this if it's actually relevant to what was just asked):\n{deploy_check_result}\n"
        if deploy_check_result
        else ""
    )
    return f"""Incident state recorded so far (facts/hypotheses/decisions already
confirmed in this call — use this to detect contradictions, not to repeat):
{existing_state}
{deploy_section}
Conversation so far:
{conversation}

Extract only what is new in the latest turn, and produce your structured
update per the response schema.
"""


def _fallback_structuring_update() -> StructuringUpdate:
    return StructuringUpdate(spoken_reply=FALLBACK_REPLY)


async def generate_structuring_update(
    messages: List[ChatMessage], existing_state: dict, service: Optional[str] = None
) -> StructuringUpdate:
    """
    One turn of live structuring: given the conversation and what's already
    recorded for this call, extract new facts/hypotheses/decisions/action
    items, flag a contradiction if one exists, and produce what the agent
    should say. Degrades to a plain fallback reply (not an error) when no
    API key is configured, same reasoning as the original proxy-only version
    this replaces — prove the wiring survives even without a real key.

    service is the incident's own service (a known, deterministic fact --
    not guessed from conversation) used to check GitHub for recent deploys
    when this turn sounds like it's asking about one. See
    _maybe_check_recent_deploys.
    """
    if not get_settings().gemini_api_key:
        return _fallback_structuring_update()

    deploy_check_result = await _maybe_check_recent_deploys(messages, service)

    client = _get_client()
    prompt = _build_structuring_prompt(messages, existing_state, deploy_check_result)
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
# Post-call role inference: given everything each participant said across
# the whole call, classify their likely functional role. Deliberately runs
# once at call-end over the full transcript (iCall_service.record_utterance
# accumulates per-turn) rather than turn-by-turn -- a single utterance is a
# weak, noisy signal, but a whole call's worth of what someone said is a
# much more reliable basis for the same classification. This is the
# fallback signal for participants iDirectory has no answer for; a known
# employee's directory title stays authoritative (see
# iCall_service.infer_and_store_participant_roles).
# -----------------------------------------------------------------------------

ROLE_CATEGORIES = [
    "backend_engineer",
    "frontend_engineer",
    "devops",
    "team_lead",
    "manager",
    "business_analyst",
    "qa_engineer",
    "other",
]

ROLE_CLASSIFICATION_SYSTEM_INSTRUCTION = f"""You are classifying the likely
functional role of each participant on an incident call, based only on
everything they said during the call.

Categories (use exactly these strings for `role`): {", ".join(ROLE_CATEGORIES)}

Rules:
- Judge only from the substance of what each person said -- what they
  investigated, decided, or were responsible for -- never from tone,
  confidence, or how much they talked.
- A person can score meaningfully on more than one category (e.g. someone
  doing both deploys and backend debugging). Scores are independent
  relevance judgments, not a probability distribution -- they do not need
  to sum to 1.
- best_role is your single best guess: the category with the strongest,
  most specific evidence in what they actually said.
- rationale is one short sentence citing the concrete thing they said that
  justifies best_role -- not a restatement of the category name.
- If a participant said too little to judge, use "other" with a low score
  and say so plainly in rationale rather than guessing.
- Output must strictly match the provided response schema.
"""


def _build_role_classification_prompt(speaker_utterances: Dict[str, List[str]], speaker_names: Dict[str, str]) -> str:
    sections = []
    for uid, lines in speaker_utterances.items():
        name = speaker_names.get(uid) or f"Participant {uid}"
        transcript = "\n".join(f"- {line}" for line in lines)
        sections.append(f"Speaker uid={uid} name=\"{name}\":\n{transcript}")
    return "Classify each of the following call participants:\n\n" + "\n\n".join(sections)


async def classify_participant_roles(
    speaker_utterances: Dict[str, List[str]], speaker_names: Dict[str, str]
) -> CallRoleClassification:
    """
    One end-of-call Gemini pass over every participant's full set of
    utterances. Returns an empty classification (not an error) when there's
    nothing to classify or the model call fails -- iCall_service treats a
    missing inferred_role the same as "no directory match either": leave
    that participant's role unresolved rather than guessing further.
    """
    if not speaker_utterances or not get_settings().gemini_api_key:
        return CallRoleClassification()

    client = _get_client()
    prompt = _build_role_classification_prompt(speaker_utterances, speaker_names)
    config = types.GenerateContentConfig(
        system_instruction=ROLE_CLASSIFICATION_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=CallRoleClassification,
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
                return CallRoleClassification()

    try:
        return CallRoleClassification.model_validate_json(response.text)
    except Exception:
        return CallRoleClassification()


# -----------------------------------------------------------------------------
# Role-based ownership assignment: the actual point of role inference isn't
# just knowing who's what -- it's deciding who owns each follow-up. Most
# action items from a live call won't have an explicit named owner (nobody
# says "David will do X" for every item), so this matches an item's actual
# content against the roster's roles, the same judgment a human triaging a
# backlog makes reading "check the connection pool" and thinking
# "that's backend/devops work" before looking at who's free.
# -----------------------------------------------------------------------------

ACTION_ITEM_OWNER_SYSTEM_INSTRUCTION = """You are assigning each unassigned
action item from an incident call to whichever participant's role best
fits the actual work described.

Rules:
- Only assign owner_uid to a uid from the given roster -- never invent a
  person or use a uid that isn't listed.
- Judge purely from what the action item's text says needs to be done,
  matched against each roster member's role.
- If no roster member's role is a reasonable fit for an item, leave
  owner_uid null for that item rather than forcing a guess.
- rationale is one short sentence naming the specific role-to-task match
  (or stating plainly that nothing fit).
- item_index in your response must exactly match the item's index in the
  input list.
- Output must strictly match the provided response schema.
"""


def _build_owner_assignment_prompt(items: List[str], roster: List[dict]) -> str:
    items_block = "\n".join(f"{i}. {text}" for i, text in enumerate(items))
    roster_block = "\n".join(
        f'- uid={r["uid"]} name="{r["name"]}" role={r["role"]}' for r in roster
    )
    return f"""Roster of people actually on this call:
{roster_block}

Unassigned action items (by index):
{items_block}

For each item index, assign the best-fit owner_uid from the roster above, or null if none fits.
"""


async def assign_action_item_owners(
    items: List[str], roster: List[dict]
) -> ActionItemOwnerAssignments:
    """
    One end-of-call Gemini pass matching unassigned action items to the
    roster's roles. Returns no assignments (not an error) when there's
    nothing to assign or the model call fails -- iCall_service leaves
    those items unassigned rather than guessing further.
    """
    if not items or not roster or not get_settings().gemini_api_key:
        return ActionItemOwnerAssignments()

    client = _get_client()
    prompt = _build_owner_assignment_prompt(items, roster)
    config = types.GenerateContentConfig(
        system_instruction=ACTION_ITEM_OWNER_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=ActionItemOwnerAssignments,
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
                return ActionItemOwnerAssignments()

    try:
        return ActionItemOwnerAssignments.model_validate_json(response.text)
    except Exception:
        return ActionItemOwnerAssignments()


# -----------------------------------------------------------------------------
# End-of-call risk synthesis -- "A final incident summary with unresolved
# risks" from the brief. Runs once, over the call's full final state, not
# per-turn: whether something is still unresolved is only knowable once the
# call is actually over, not while it's still being recorded.
# -----------------------------------------------------------------------------

UNRESOLVED_RISKS_SYSTEM_INSTRUCTION = """Given everything recorded during an
incident call, identify unresolved risks the team should keep tracking now
that the call has ended.

Rules:
- A risk is something that could still cause harm or uncertainty: a
  hypothesis never confirmed as a fact, a missing_info gap never answered,
  an action item with no owner, or a conflict never resolved.
- Do not assert a root cause or suggest a fix -- only state plainly what
  remains open or uncertain.
- Each risk is one short, concrete sentence someone could act on or watch
  for -- not a vague restatement of "there are open questions."
- If nothing is genuinely unresolved, return an empty list rather than
  manufacturing filler risks.
- Output must strictly match the provided response schema.
"""


def _build_unresolved_risks_prompt(structured_state: dict) -> str:
    return f"""Final recorded state for this incident call:
{structured_state}

Identify the unresolved risks per the rules above.
"""


async def summarize_unresolved_risks(structured_state: dict) -> UnresolvedRisksSummary:
    """
    One end-of-call Gemini pass. Returns no risks (not an error) when
    there's nothing to summarize or the model call fails -- a missing
    unresolved-risks list is treated as "none surfaced," not a crash.
    """
    if not structured_state or not get_settings().gemini_api_key:
        return UnresolvedRisksSummary()

    client = _get_client()
    prompt = _build_unresolved_risks_prompt(structured_state)
    config = types.GenerateContentConfig(
        system_instruction=UNRESOLVED_RISKS_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=UnresolvedRisksSummary,
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
                return UnresolvedRisksSummary()

    try:
        return UnresolvedRisksSummary.model_validate_json(response.text)
    except Exception:
        return UnresolvedRisksSummary()


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
