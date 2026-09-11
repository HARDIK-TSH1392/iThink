import asyncio
import hashlib
import hmac
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Set

import httpx
from google import genai
from google.genai import types

from app.config import get_settings
from .iCall_schema import (
    ChatMessage,
    StructuringUpdate,
    CallRoleClassification,
    ActionItem,
    ActionItemOwnerAssignments,
    UnresolvedRisksSummary,
    ReviewedTicketContent,
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


# See iTriage_utils.py's constants for the fuller history -- these two
# used to be the same verified model on purpose (avoiding a guessed
# fallback name after a retired-model outage), but that gave zero real
# redundancy: when gemini-3.5-flash-lite itself hit Gemini's "high demand"
# instability live, retrying the identical model failed identically both
# times. Re-verified on 2026-09-04 during that same instability that
# these two are genuinely distinct models with independent capacity, not
# just different names for the same thing.
#
# PRIMARY moved to a lite tier on 2026-09-05 for latency, not carelessly
# repeating the incident above: measured against the actual production
# prompt/schema (not guessed), gemini-flash-lite-latest ran the identical
# structuring call in 1.27s vs 2.69s for gemini-flash-latest -- a 53% cut
# on the single biggest piece of the ~5s turn-to-speech latency -- and it
# correctly handled the hardest live-failing case this session hit (the
# hypothesis-contradiction conflict check) with equivalent output. FALLBACK
# deliberately stays a full-tier model, not another lite one: if the lite
# primary hits the same kind of capacity instability that caused the
# history above, the fallback path needs to actually be different capacity,
# not just a different name for the same risk.
PRIMARY_MODEL = "gemini-flash-lite-latest"
FALLBACK_MODEL = "gemini-3.5-flash"

# None of the four Gemini call sites below had a timeout before this --
# found via a dry run that hit a genuine hang (90+s, zero output, not even
# the fast ~1-3s failure a real 503 returns) on one of the end-of-call
# passes. Without this, a stuck request blocks whatever awaits it
# indefinitely: for the per-turn structuring call, that's the live
# conversation itself (Agora's Custom LLM hook has nothing to send back to
# the room); for the end-of-call passes, that's the client's "End
# Conversation" request just hanging. Bumped from 25s to 30s after
# observing the verified fallback model itself take ~24s under load --
# 25s was cutting that too close.
GEMINI_CALL_TIMEOUT_S = 30

# A hardcoded Semaphore(4) global concurrency cap used to live here --
# removed after confirming live (direct test: 12 concurrent Gemini calls,
# 3x the old limit) that the real API handles it fine with no rate-limit
# errors and no latency degradation. It was never based on a measured rate
# limit or a real observed failure; it just quietly capped real capacity
# with no evidence it was ever needed.

# Hedged requests (fire a duplicate after ~1.6s -- just above our measured
# p90 -- and take whichever finishes first) were tried here and reverted.
# The reasoning going in was sound (our slowest observed turn happened on
# the SMALLEST-context turn of a call, pointing at infra noise rather than
# input size, which is the textbook condition hedging is documented to
# help with) -- but measured live, the wrapper itself added ~180ms of
# overhead to essentially every call, not just the slow tail, making mean
# latency worse (1.41s direct vs 1.59s hedged, same prompt). Not debugged
# further under time pressure rather than shipped on the strength of the
# theory alone. If this is revisited, the actual overhead source (task
# creation/shielding cost vs. something else) needs isolating first, and
# it needs to be re-measured against this same benchmark before trusting
# it again -- the theory being right last time is exactly why the
# implementation still has to be checked, not assumed correct.
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

# Distinct from FALLBACK_REPLY on purpose: that one means no API key is
# configured at all. This means the key IS configured and both the
# primary and fallback model calls still failed (Gemini-side instability,
# not a setup problem) -- telling someone to "set GEMINI_API_KEY" when
# they already have one working most of the time is actively misleading,
# found while verifying the new model pairing above hit a moment where
# both attempts got a 503 back to back.
MODEL_UNAVAILABLE_REPLY = (
    "I'm having trouble reaching my language model right now — "
    "bear with me, I'll catch up once it's back."
)

# Used when the model's own output doesn't validate against the schema --
# a diagnostic signal, not conversational content. Named so the speak-gate
# below can recognize and always pass it through, the same as FALLBACK_REPLY.
MALFORMED_RESPONSE_FALLBACK = "Sorry, could you say that again?"

# The agent's own name, in one place -- everything that needs to recognize
# or say it (direct-address detection here, the structuring/triage prompts,
# the voice-agent's greeting) reads from this constant instead of a
# hardcoded literal, so a future rename touches one line, not a scattered
# grep-and-hope. Previously "iThink" -- a compound word Deepgram had no
# reason to transcribe distinctly from the hedge phrase "I think", which is
# why direct-address detection used to require "hey" alongside it just to
# stay rare enough not to fire on ordinary sentences ("I think the DB is
# fine"). "Watcher" is an ordinary, distinct English word -- but that
# doesn't mean STT transcribes it reliably. Confirmed live across two real
# calls (incident-46, incident-55): asked for roughly 15 times combined,
# it came back correct only ~25-30% of the time, misheard as "Voucher" /
# "Vajar" / "Vucher" / "Voacher" / "Vacher" / "Varcher" the rest. Keyterm
# boosting would fix the acoustic mishearing directly, but re-enabling it
# reproduces a worse bug (see build_keyterms / agent.py's revert history:
# it breaks transcription entirely on this pipeline, empty content on most
# turns). Can't fix Deepgram's ear, so instead of not hearing our own name
# at all, recognize its own most common mistakes -- a real, if inelegant,
# fix for the actual practical problem.
AGENT_NAME = "Watcher"

# The known mis-hearings above, kept as their own list (rather than baked
# into the regex inline) so QuickstartTranscriptPanel.tsx on the frontend
# can mirror the same corrections for DISPLAY purposes -- the backend
# recognizing "Voucher" as a valid wake word doesn't change what text
# Deepgram actually produced, so without a matching frontend fix the
# transcript panel would keep showing the wrong word even while the agent
# correctly responds to it. Keep both lists in sync if this list changes.
_KNOWN_MISHEARINGS = ("voucher", "vajar", "vucher", "voacher", "vacher", "varcher")

# Case-insensitive, whole-word match: also answers to "agent" (the generic,
# obvious way to address an AI assistant on a call -- confirmed live,
# incident-31: "Agent, can you tell me the status?" got zero response
# because neither this nor a bare "hey"/"i think" combo covered it). A
# missed direct address (silently ignoring someone who's genuinely talking
# to it) is a worse failure than an occasional false accept from ordinary
# use of the word "agent" -- see the wake-word FRR/FAR tradeoff this is
# modeled on: in a live incident room, being ignored erodes trust faster
# than an extra reply does. Same asymmetry justifies accepting the known
# mishearings too, despite the (low) risk of one of them appearing in
# unrelated speech.
_ADDRESS_PATTERN = re.compile(
    r"\b(" + re.escape(AGENT_NAME.lower()) + r"|agent|" + "|".join(_KNOWN_MISHEARINGS) + r")\b",
    re.IGNORECASE,
)


def _is_direct_address(text: str) -> bool:
    """True if this turn looks like someone deliberately addressing the agent by name."""
    return bool(_ADDRESS_PATTERN.search(text))

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


def build_correction_callout(update: StructuringUpdate) -> str:
    """
    Deterministic, non-LLM-authored line for when a previously recorded
    fact just got superseded (see apply_structuring_update's corrects_fact
    handling) -- same "don't let the model freely narrate a state change"
    discipline as CLOSING_LINE. Confirmed live (incident-31): the region
    correction (US East -> US West) got applied to structured_state
    entirely silently, with nothing said to the room -- a real gap against
    the brief's "detection of... conflicting information," since updating
    the record without announcing it doesn't actually keep the team's
    shared understanding aligned.

    "Update: <current>. Earlier: <previous>." template, not a full sentence
    -- confirmed live (incident-101) that natural-language phrasing here
    reads with more ambiguity than a flat template needs to carry:
    "Correction" outright implied the old fact was wrong (misleading when
    the two facts were actually compatible, not contradictory -- real
    example: "Outage is not in the US" + "The outage is in the Africa
    region", just STT fragmenting one sentence into two turns), and even
    "previously noted as" still reads as prose that could be parsed
    different ways under time pressure. "Update:" is kept as the lead word
    deliberately -- it signals up front that a change is coming, unlike a
    bare "Now:" -- but the two facts themselves are separated into their
    own labeled clause ("Earlier: ...") rather than run together
    mid-sentence, specifically so TTS doesn't blur them into one
    continuous, seamless string with no audible break between "what's true
    now" and "what was true before." Neither clause asserts the earlier
    statement was false, doesn't editorialize, just states what was true
    then and what's true now. Only reachable when corrects_fact actually
    fired on a genuine change, not a restatement of the same fact in
    different words -- see the corrects_fact prompt guidance in
    STRUCTURING_SYSTEM_INSTRUCTION for that guard; this function only
    controls the wording once a real update is confirmed, it doesn't
    decide whether one happened.

    update.facts is this turn's newly-extracted facts, not the full running
    list -- per the corrects_fact prompt guidance, the turn that resolves a
    contradiction is expected to also state the corrected fact itself, so
    it's normally present here. Falls back to a plainer phrasing on the
    rare turn where the model flagged corrects_fact without a matching new
    fact -- there's no "current" fact to put after "Update:" in that case,
    so it says so directly rather than leaving the clause empty or
    stacking two labels back to back ("Update: Earlier: ...").
    """
    if update.facts:
        return f'Update: {update.facts[0]}. Earlier: {update.corrects_fact}.'
    return f"Update: that's no longer accurate. Earlier: {update.corrects_fact}."

# Matches SILENCE_TRIGGER_MARKER in voice-agent/server/src/agent.py, set as
# parameters.silence_config.content there with action="think" -- Agora
# appends it as the last message and routes the whole conversation through
# our custom LLM (this proxy) instead of speaking it directly. An inert
# control string, never something a real participant would say, so an exact
# match on the last message is a reliable, zero-cost way to tell "the room
# went quiet" apart from a real turn.
SILENCE_TRIGGER_MARKER = "[[ithink-silence-check]]"


def _is_silence_trigger(messages: List[ChatMessage]) -> bool:
    return bool(messages) and messages[-1].content.strip() == SILENCE_TRIGGER_MARKER


# Generic incident-call vocabulary Deepgram has no particular reason to get
# right on its own -- proper nouns, jargon, and hyphenated compounds, not
# ordinary English words it already transcribes fine. Deliberately excludes
# plain phrases like "root cause" or "incident commander": common English
# any general model already handles, and (since Agora's keyterm field is
# one space-separated string, not a list) a multi-word phrase here would
# get split into separate single-word terms anyway, not kept intact. Kept
# small and general (not tied to any one incident) since it applies to
# every call; the service name (the actual per-call variable) is appended
# separately in build_keyterms.
BASE_KEYTERMS = [
    AGENT_NAME,
    "rollback",
    "postmortem",
    "on-call",
    "Jira",
    "Slack",
    "PagerDuty",
]


def build_keyterms(service: Optional[str]) -> str:
    """
    Deepgram keyterm-prompting string for this call -- boosts recognition of
    words STT has no reason to get right on its own (see Deepgram's own
    keyterm docs: brand names, jargon, anything outside common English).
    Confirmed live this session: "auth-api" came back as "OT API" with no
    boosting at all.

    A pure function of what's already known deterministically about the
    call (the incident's own service name), same discipline as everywhere
    else in this file that avoids inventing content -- no LLM call, no
    guessing, just the one real per-call variable appended to a fixed base
    list. Agora's Deepgram integration takes this as a single
    space-separated string (its own documented format), not a list --
    fine here since none of these terms need internal spaces preserved as
    one token (service names in this system are hyphenated, e.g.
    "auth-api", never multi-word).
    """
    terms = list(BASE_KEYTERMS)
    if service:
        terms.append(service)
    return " ".join(terms)


def _call_needs_check_in(structured_state: dict) -> bool:
    """
    Whether room-wide silence is actually worth interrupting -- the
    event-driven counterpart to detect_health_score_drop's own philosophy,
    applied to the silence-trigger path too. 30 seconds of quiet during a
    real incident call is often the room *working*: reading a dashboard,
    checking a rotation tool, messaging another team. That's not a
    stalled conversation, and interrupting it isn't "keeping the team
    aligned" -- it's noise during focused work. A call that's already in
    good shape (real progress recorded, nothing open) shouldn't get any
    proactive silence nudge at all; one that's stuck (no progress yet, or
    a genuinely open conflict/gap) still should.
    """
    has_progress = bool(
        structured_state.get("facts")
        or structured_state.get("hypotheses")
        or structured_state.get("decisions")
    )
    if not has_progress:
        return True
    has_open_issue = bool(structured_state.get("conflicts") or structured_state.get("missing_info"))
    return has_open_issue


def build_silence_prompt(structured_state: dict) -> str:
    """
    Deterministic, transcript-aware line for when the room's gone quiet --
    same "don't let an LLM call reinvent this" discipline as CLOSING_LINE
    and format_call_summary/format_live_recap: the facts/hypotheses/
    decisions were already extracted live by generate_structuring_update,
    so what to say about "how far we've gotten" doesn't need a fresh model
    call, just a read of what's already recorded.

    Nothing recorded yet (no facts, hypotheses, or decisions) means the
    room hasn't actually made progress -- say so plainly and ask for
    something to work with, rather than the generic "anyone have more to
    add" that only makes sense once there's something to add *to*.
    """
    state = structured_state or {}
    has_progress = bool(
        state.get("facts") or state.get("hypotheses") or state.get("decisions")
    )
    if not has_progress:
        return (
            "So far we haven't figured out the root cause or gathered any facts yet -- "
            "can someone walk through what's actually happening?"
        )
    return "Does anyone else have any more points to contribute?"


async def get_live_participant_count(channel_name: str) -> Optional[int]:
    """
    Queries voice-agent/server's own join registry (_channel_names) for how
    many people are actually on the call right now. Used only to suppress
    the silence nudge when there's nobody to nudge (a lone participant
    talking to themselves gets no "does anyone have more to add?").

    Returns None on any failure -- callers should treat that as "unknown"
    and fail open (still nudge) rather than let a slow/unreachable
    voice-agent service silently stop the agent from ever speaking up.
    """
    base = get_settings().voice_agent_server_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{base}/getNames", params={"channel": channel_name})
            response.raise_for_status()
            names = response.json().get("data", {})
    except Exception:
        return None
    return len(names) if isinstance(names, dict) else None


STRUCTURING_SYSTEM_INSTRUCTION = f"""You are {AGENT_NAME}, a voice participant in a live
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
- A "decision" is the room explicitly choosing a course of action or
  agreeing on next steps ("let's roll back the deploy", "we'll page the
  on-call DBA", "agreed, we hold off until QA confirms") — not a fact
  (something already true) and not by itself an action item (a task with
  an owner). If the same turn both decides something and assigns someone
  to carry it out, record both: the decision, and a matching action item
  with that owner. Don't record a decision for someone merely floating an
  option or asking what to do — only once the room actually settles on one.
- Set "conflict" whenever something said in this turn contradicts a fact
  OR a hypothesis already recorded in the state you were given below --
  hypotheses count just as much as facts here, not only direct fact-vs-fact
  contradictions. For example: someone floats "might be a database
  connection issue" (a hypothesis) and a later turn says "the database team
  confirmed everything is fine on their side" -- that directly contradicts
  the hypothesis and must be flagged as a conflict, phrased as a clarifying
  question ("so is the database ruled out, or still a possibility?"), even
  though the later statement also stands on its own as a new fact worth
  recording. Recording the new fact is not a substitute for flagging the
  conflict -- do both when both apply. Phrase the conflict itself as one
  short, targeted clarifying question you would ask out loud -- not a
  statement, not an accusation.
  Do NOT raise a conflict about something the facts list you were given
  below has already settled. STT often splits one continuous correction
  across several turns (e.g. "APAC." then, separately, "hey I mean EU
  Central was wrong too") -- if the facts list already contains the
  corrected answer and no longer contains the old one, a later fragment
  that's just finishing that same thought is not new information to
  question, even if it reads ambiguously on its own. Confirmed live
  (incident-55, 2026-09-10): the room heard "Update: APAC. Earlier: EU
  Central." immediately followed nine seconds later by "So is the region
  EU Central or APAC?" -- re-litigating something already resolved one
  turn earlier, confusing rather than clarifying. Only raise a conflict
  here for a genuinely new contradiction the facts list doesn't already
  reflect.
- corrects_fact (optional): set this to the EXACT text of an existing fact
  from the state given below, only when something said in THIS turn
  directly resolves a contradiction as the room's now-confirmed answer
  (e.g. an earlier "the outage is in the AC region" turns out to be wrong
  once "confirmed, it's only Africa" is said). Must match the existing
  fact's text exactly, character for character, or it's ignored — don't
  paraphrase it. Leave null on ordinary turns. This is different from
  "conflict": conflict is for flagging a contradiction as still an open
  question; corrects_fact is for the later turn where the room has
  actually settled it and the old fact is now simply wrong, not just
  disputed.
  DO NOT set corrects_fact merely because a new fact is more specific than,
  or adds detail to, an existing one -- that's a genuine completion, not a
  contradiction, and the old fact was never wrong. Concretely: an existing
  fact "Outage is not in the US" followed by this turn saying "the outage
  is in the Africa region" is NOT a contradiction -- both are true at the
  same time (ruling a place out, then naming the actual place, is one
  continuous thought, often just split across two turns by the transcript).
  Leave corrects_fact null there; record the Africa detail as a new fact
  instead. Only use corrects_fact when the new statement is actually
  incompatible with the old one (the old fact could not still be true
  given the new one) -- a negative statement ("not X") and a later
  positive statement about something other than X are compatible, not
  contradictory.
  ALSO do not set corrects_fact when the new statement just RESTATES an
  existing fact in different words -- that's not a correction either,
  since nothing actually changed. Concretely: an existing fact "The
  outage is not in the Africa region" followed by this turn saying
  "Africa region is ruled out entirely" is the SAME claim, just reworded
  -- not a correction, and there is no new fact worth recording from it.
  Leave both facts and corrects_fact untouched on a turn like that. Only
  use corrects_fact when the room's answer actually flips (an old fact
  that said one thing is now confirmed false, replaced by a genuinely
  different claim) -- not when someone just says the same confirmed
  thing again in their own words.
- action_items: when a specific person is named as responsible for a task
  in this turn ("Rahul, can you check the logs", "I'll get Priya to look
  at it"), set that item's owner to that name. Leave owner null when no
  specific person was named — don't guess an owner from role, context, or
  who seems most likely responsible.
  This includes an IMPLICIT reference back to something asked earlier in
  the call, not just a task stated fresh in this same turn. Concretely:
  if an earlier turn asked "does anyone know which port is affected?" and
  this turn says "Ted covers it", record an action item for that earlier
  question ("find out which port is affected") with owner "Ted" -- do not
  require the task itself to be restated in full for the owner to count.
  Look at the state given below for what open question or gap "it" is
  most plausibly referring to before deciding there's nothing to record.
- root-cause fishing vs. hypothesis vs. resolution: someone directly
  asking what caused the incident ("what actually caused this", "do we
  know the root cause yet") is is_root_cause_question -- this is a
  question being asked OF the room, not a guess being floated (that's a
  hypothesis) and not the room settling something (that's
  is_resolution_declared, see below). Never let is_root_cause_question
  being true tempt spoken_reply into actually answering with a cause;
  the hard constraint against asserting root cause applies here exactly
  as everywhere else in this prompt.
- is_resolution_declared: true when this turn explicitly settles an open
  hypothesis or conflict as resolved or ruled out ("consider it ruled
  out, we got confirmation", "that's confirmed fixed now", "we can rule
  out the database"). This can fire mid-call, well before the incident
  or the call itself is over -- it is unrelated to is_wrapping_up.
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
- spoken_reply is what you will say out loud right now, shaped by whichever
  specific reason this turn will actually be spoken — never default to a
  generic acknowledgment ("okay, noted") when one of these applies instead:
  - If there's a conflict, spoken_reply should be that clarifying question.
  - If missing_info is set, spoken_reply should directly ask for that
    missing piece, not just acknowledge something's unclear.
  - If this turn recorded a new action item with an owner, spoken_reply
    should briefly confirm the assignment (e.g. "Got it, Rahul's on that.").
  - If this turn was a direct question addressed to you, answer it.
  - Otherwise, this turn likely won't be spoken at all (a deterministic
    gate decides that, not you) — keep spoken_reply minimal rather than
    spending effort on a generic acknowledgment for content that's about
    to be discarded anyway.
  If is_wrapping_up is true, spoken_reply is ignored regardless of the
  above (the caller substitutes a fixed closing line).
- A turn that's just a greeting or social opener ("hello", "hi", "hey",
  "anyone there?") with no actual content yet has nothing to extract --
  leave facts/hypotheses/decisions/action_items/missing_info empty and
  reply naturally ("Hi, go ahead" / "I'm here, listening"). Don't say
  "okay, noted" or similar to a greeting -- there's nothing to have noted
  yet, and it reads as ignoring what was actually said.
- agent_chat_note (optional): use this instead of expanding spoken_reply
  when you have something worth flagging that doesn't need to interrupt
  the room -- e.g. a secondary observation, a connection between two facts
  said minutes apart, something useful but not urgent. The room is
  actively talking; a long spoken interjection breaks their flow, but a
  written note doesn't. Leave it null on ordinary turns -- most turns
  don't need one. Never put something here that's genuinely urgent (a
  contradiction, a safety-relevant gap) -- those still belong in
  spoken_reply, said out loud, because a note is easy to miss mid-call.
- wants_log_screen: true only when this turn is someone asking to see or
  pull up the actual server logs for this incident -- "show me the logs",
  "what's in the logs since this started," "can you pull up the server
  logs," any phrasing with that intent. False for merely talking about
  symptoms or errors in prose (e.g. "the logs are full of 500s" is a fact,
  not a request to see the log lines themselves). When true, spoken_reply
  should acknowledge you're pulling them up, not describe log contents
  yourself -- you don't have the actual log data, the caller queries it
  separately and shows it visually to everyone on the call.
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


async def _check_recent_deploys(repo: str) -> Optional[dict]:
    """
    Calls the isolated GitHub MCP bridge service. Returns None (not an
    error) on any failure -- a missing/unreachable deploy-check service
    should never break the live call's own response. Returns the full
    {"summary", "commits"} dict rather than just the summary text so a
    caller building the GitHub shared-screen (see broadcast_shared_screen)
    has the raw per-commit data too, not just the prose version fed to the
    LLM prompt.
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
    return data


async def _maybe_check_recent_deploys(messages: List[ChatMessage], service: Optional[str]) -> Optional[dict]:
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

    result = await _check_recent_deploys(repo)
    if result is None:
        return None
    result["repo"] = repo
    return result


# -----------------------------------------------------------------------------
# Shared screens: when someone asks to see GitHub commits or server logs,
# push the actual data to everyone on the call at once via Agora's
# Signaling (RTM 2.x) REST API, which can message a channel from the
# server without joining it. Every participant's browser is already
# subscribed to this RTM channel (see LandingPage's RTM login/subscribe),
# so this reaches all of them live with no polling. Best-effort throughout
# -- an unconfigured or unreachable Signaling API should never break the
# turn that triggered it; the data is still recorded (see
# iCall_service.record_shared_screen) so a late joiner can catch up via
# the /recap endpoint even if the live broadcast failed or they weren't
# there for it.
# -----------------------------------------------------------------------------

SHARED_SCREEN_MESSAGE_TYPE = "ithink_shared_screen"


async def broadcast_shared_screen(channel_name: str, screen: dict) -> bool:
    """
    POSTs one channel message to Agora's Signaling REST API. Returns
    whether it actually sent -- callers should still record the screen in
    structured_state regardless of this result (see caller in iCall_api),
    since the DB copy is what late joiners and any client that missed the
    live message fall back on.
    """
    settings = get_settings()
    if not (settings.agora_app_id and settings.agora_customer_key and settings.agora_customer_secret):
        return False

    import base64
    import json as _json

    url = (
        f"https://api.agora.io/dev/v2/project/{settings.agora_app_id}"
        f"/rtm/users/ithink-backend/channel_messages"
    )
    credentials = base64.b64encode(
        f"{settings.agora_customer_key}:{settings.agora_customer_secret}".encode()
    ).decode()
    payload = _json.dumps({"type": SHARED_SCREEN_MESSAGE_TYPE, "screen": screen})

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json;charset=utf-8",
                },
                json={
                    "channel_name": channel_name,
                    "payload": payload,
                    "enable_historical_messaging": False,
                },
            )
            response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[iCall] Failed to broadcast shared screen to channel={channel_name}: {exc}")
        return False


def _build_structuring_prompt(
    messages: List[ChatMessage],
    existing_state: dict,
    deploy_check_result: Optional[dict] = None,
    tool_result_text: Optional[str] = None,
    service: Optional[str] = None,
    region: Optional[str] = None,
    incident_age_minutes: Optional[int] = None,
) -> str:
    conversation = "\n".join(f"{m.role}: {m.content or ''}" for m in messages)
    deploy_summary = deploy_check_result.get("summary") if deploy_check_result else None
    deploy_section = (
        f"\nGitHub lookup result (only mention this if it's actually relevant to what was just asked):\n{deploy_summary}\n"
        if deploy_summary
        else ""
    )
    # Confirmed live: without this, a native-MCP tool call (get_recent_logs
    # or GitHub's commit-listing tools) had no way to know which service/
    # region/repo this incident is actually about -- it only worked when a
    # participant happened to say the exact service name out loud in this
    # call's own conversation, and returned nothing for a service like
    # "cdn-edge" that never got mentioned by name. service/region are
    # already known, deterministic facts (see generate_structuring_update's
    # docstring) -- surfacing them here means a tool call fills in the
    # right arguments instead of guessing from conversation content.
    incident_context_section = ""
    if service:
        repo = SERVICE_TO_GITHUB_REPO.get(service)
        repo_line = (
            f"Linked GitHub repo for this service: {repo} -- use this exact repo when looking up commits/deploys."
            if repo
            else "No GitHub repo is linked for this service -- if asked for commits/deploys, say so rather than guessing a repo."
        )
        # get_recent_logs' own default window is generous (24h, see
        # ilogs_mcp_service/server.py) precisely because logs leading up to
        # an incident typically predate when it was officially detected --
        # this is just context, not a precise value to compute against;
        # asking the model to do that math itself was the fragile version
        # of this fix (it doesn't know how far pre-detection logs go back
        # either).
        age_line = (
            f"This incident was created {incident_age_minutes} minute(s) ago -- if a log lookup "
            "comes back empty, don't assume there's nothing to find; widen since_minutes well beyond "
            "the tool's default before concluding that.\n"
            if incident_age_minutes is not None
            else ""
        )
        incident_context_section = (
            f"\nThis incident's service is `{service}`"
            + (f" in region `{region}`" if region else "")
            + " -- use this exact value (not something inferred from conversation) as the service/region argument for "
            "any log or GitHub lookup tool you call.\n"
            f"{repo_line}\n{age_line}"
        )
    # tool_result_text: this turn is the follow-up after the model itself
    # called an Agora-native MCP tool (see chat_completions_endpoint) --
    # same "this directly answers the question" framing as the deploy
    # lookup above, just for whichever tool was actually called.
    tool_section = (
        f"\nTool result (this DIRECTLY ANSWERS what you just asked to look up -- state it as fact, don't say \"checking\"):\n{tool_result_text}\n"
        if tool_result_text
        else ""
    )
    return f"""Incident state recorded so far (facts/hypotheses/decisions already
confirmed in this call — use this to detect contradictions, not to repeat):
{existing_state}
{incident_context_section}{deploy_section}{tool_section}
Conversation so far:
{conversation}

Extract only what is new in the latest turn, and produce your structured
update per the response schema.
"""


def _fallback_structuring_update(reply: str = FALLBACK_REPLY) -> StructuringUpdate:
    return StructuringUpdate(spoken_reply=reply)


class StructuringResult(NamedTuple):
    """
    update is None exactly when tool_call is set -- the model decided to
    call an Agora-native MCP tool instead of answering this turn, so
    there's nothing to structure yet (see chat_completions_endpoint: the
    real structuring happens on the follow-up request once the tool's
    result comes back).
    """
    update: Optional[StructuringUpdate]
    deploy_check_result: Optional[dict]
    tool_call: Optional[Any]


def _convert_tools_to_gemini(tools: Optional[List[dict]]) -> Optional[types.Tool]:
    """
    Converts Agora's OpenAI-format `tools` (forwarded from its native MCP
    server registration, see voice-agent/server/src/agent.py's mcp_servers
    config) into a Gemini Tool. OpenAI's function.parameters is already
    JSON schema, same shape Gemini's parameters_json_schema expects, so
    this is a straight field remap, not a real format conversion.
    """
    if not tools:
        return None
    declarations = []
    for entry in tools:
        fn = entry.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=fn.get("description") or "",
                parameters_json_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return types.Tool(function_declarations=declarations) if declarations else None


async def generate_structuring_update(
    messages: List[ChatMessage],
    existing_state: dict,
    service: Optional[str] = None,
    tools: Optional[List[dict]] = None,
    tool_result_text: Optional[str] = None,
    region: Optional[str] = None,
    incident_age_minutes: Optional[int] = None,
) -> StructuringResult:
    """
    One turn of live structuring: given the conversation and what's already
    recorded for this call, extract new facts/hypotheses/decisions/action
    items, flag a contradiction if one exists, and produce what the agent
    should say. Degrades to a plain fallback reply (not an error) when no
    API key is configured, same reasoning as the original proxy-only version
    this replaces — prove the wiring survives even without a real key.

    service/region are the incident's own, known-deterministic values --
    used both for the existing GitHub-deploy-check heuristic below AND
    (via _build_structuring_prompt's incident_context_section) surfaced
    directly to the model so a native MCP tool call (get_recent_logs,
    GitHub's commit tools) has the right arguments instead of only working
    when a participant happened to say the service name out loud. Confirmed
    live: a service with no such mention in conversation (e.g. "cdn-edge")
    returned nothing from either tool before this was added.

    service is the incident's own service (a known, deterministic fact --
    not guessed from conversation) used to check GitHub for recent deploys
    when this turn sounds like it's asking about one. See
    _maybe_check_recent_deploys -- this deterministic path stays in place
    alongside native tool-calling below, not replaced by it: it's proven
    and adds no extra round-trip, whereas tools are only used when the
    model itself decides a turn needs one.

    tools, when Agora registered MCP server(s) on this agent (see
    voice-agent/server/src/agent.py's mcp_servers), lets the model call
    them directly (confirmed live: Gemini supports tools + response_schema
    in the same call -- it either returns a function_call part, in which
    case result.update is None and result.tool_call is set, or normal
    structured JSON matching StructuringUpdate).
    """
    if not get_settings().gemini_api_key:
        return StructuringResult(_fallback_structuring_update(), None, None)

    deploy_check_result = await _maybe_check_recent_deploys(messages, service)
    gemini_tool = _convert_tools_to_gemini(tools)

    client = _get_client()
    prompt = _build_structuring_prompt(
        messages, existing_state, deploy_check_result, tool_result_text,
        service=service, region=region, incident_age_minutes=incident_age_minutes,
    )
    config = types.GenerateContentConfig(
        system_instruction=STRUCTURING_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=StructuringUpdate,
        tools=[gemini_tool] if gemini_tool else None,
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=PRIMARY_MODEL,
                contents=prompt,
                config=config,
            ),
            timeout=GEMINI_CALL_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"[iCall] Structuring primary call failed/timed out, trying fallback: {exc}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=config,
                ),
                timeout=GEMINI_CALL_TIMEOUT_S,
            )
        except Exception as exc2:
            # This is the live per-turn call -- a hang here previously
            # had no ceiling at all and would have frozen the actual
            # conversation (Agora's Custom LLM hook has nothing to send
            # back to the room until this returns).
            print(f"[iCall] Structuring failed on both attempts, falling back: {exc2}")
            return StructuringResult(_fallback_structuring_update(MODEL_UNAVAILABLE_REPLY), None, None)

    parts = response.candidates[0].content.parts if response.candidates else []
    function_call = next((p.function_call for p in parts if p.function_call), None)
    if function_call:
        return StructuringResult(None, deploy_check_result, function_call)

    try:
        update = StructuringUpdate.model_validate_json(response.text)
    except Exception:
        # Model returned something that didn't match the schema — don't crash
        # the live call over a malformed extraction, just say something safe
        # and record nothing rather than guessing at a partial parse.
        update = StructuringUpdate(spoken_reply=MALFORMED_RESPONSE_FALLBACK)
    return StructuringResult(update, deploy_check_result, None)


# -----------------------------------------------------------------------------
# Speak gate: StructuringUpdate.spoken_reply is a required field, so the
# model always produces *something* -- the STRUCTURING_SYSTEM_INSTRUCTION
# asks it to keep ordinary turns to "a brief, natural acknowledgment" rather
# than route them to agent_chat_note, but a prompt is a preference, not a
# guarantee. This makes "should this turn actually interrupt the room out
# loud" a deterministic check instead of trusting the model to self-regulate
# on every single turn -- same reasoning as everywhere else in this codebase
# that a fact is computed in code rather than asserted by the LLM.
#
# Deliberately narrow and reusing fields that already exist (conflict,
# missing_info, action_items, is_wrapping_up) rather than inventing new
# signals under time pressure. The action_items check exists specifically
# because the problem statement names it: "Proposes owners in speech" --
# an item with a newly-set owner is speak-worthy on its own, independent
# of whether this turn also happens to have a conflict or missing_info.
# Two known non-goals, on purpose: no periodic "recap" timer here
# (voice-agent's own silence_config already prompts the room after 30s of
# true dead air -- a different, complementary mechanism, not duplicated),
# and no dedup on repeated missing_info gaps (the system prompt already
# scopes missing_info to "the room itself is missing" on THIS turn, not a
# running checklist, so treating a fresh one as speak-worthy is consistent
# with how narrowly the model is already asked to set it).
# -----------------------------------------------------------------------------


# A gap that's still genuinely open doesn't need re-announcing every turn
# just because the model re-extracts it -- confirmed live (incident-31):
# "which model is affected?" got asked 4 times in 17 seconds while the room
# was mid-explanation, the same class of problem detect_health_score_drop's
# cooldown already exists to prevent for a different signal. A genuinely
# NEW gap (not already in structured_state) always still speaks immediately
# regardless of cooldown -- this only throttles re-asking the SAME open item.
MISSING_INFO_NUDGE_COOLDOWN_S = 20

# Confirmed live (incident-38): STT fragmenting one continuous utterance
# ("Watcher." / "you tell me the current status?") into two turns can make
# BOTH independently trip the direct_address gate -- each fragment is
# processed sequentially under the per-call turn lock (no lost-update race
# here, unlike the earlier Bob/rollback bug), and each one's own
# latest_user_message check is individually valid, so the lock alone can't
# catch this. Same class of problem as missing_info's repeat-nudging, same
# fix shape: a short cooldown after actually speaking a direct-address
# reply. Deliberately shorter than MISSING_INFO_NUDGE_COOLDOWN_S -- long
# enough to bridge a fragmentation gap (observed live: 2.2s between the two
# duplicate replies), short enough that a genuinely fast follow-up question
# addressed to the agent still gets answered.
DIRECT_ADDRESS_REPLY_COOLDOWN_S = 8


def _has_speakable_missing_info(update_missing_info: List[str], structured_state: dict) -> bool:
    if not update_missing_info:
        return False
    already_known = set(structured_state.get("missing_info", []))
    if any(gap not in already_known for gap in update_missing_info):
        return True
    last_nudge_at = structured_state.get("last_missing_info_nudge_at")
    if not last_nudge_at:
        return True
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_nudge_at)).total_seconds()
    return elapsed >= MISSING_INFO_NUDGE_COOLDOWN_S


def _direct_address_off_cooldown(structured_state: dict) -> bool:
    last_reply_at = structured_state.get("last_direct_address_reply_at")
    if not last_reply_at:
        return True
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_reply_at)).total_seconds()
    return elapsed >= DIRECT_ADDRESS_REPLY_COOLDOWN_S


def action_item_key(text: str) -> str:
    """Normalized identity for matching the same action item re-extracted
    across STT-fragmented turns -- lowercased, whitespace-collapsed exact
    match, deliberately not fuzzy. Confirmed live (incident-46): the real
    duplicates ("check the port config" extracted three separate times as
    the room circled back to confirm an owner) were byte-identical after
    this normalization, so exact match already covers the observed
    failure without the tuning risk of a similarity threshold nobody has
    measured against real transcripts."""
    return " ".join(text.lower().split())


def _is_new_owner_assignment(item: ActionItem, structured_state: dict) -> bool:
    """
    True only if this item's owner is information the room hasn't already
    heard confirmed out loud -- i.e. either this action item's text isn't
    in state yet, or it is but with no owner recorded yet. False when the
    same (text, owner) pairing is already there, which happens when STT
    fragments one instruction into several turns and the model
    re-extracts the same item, with the same owner, on a later turn (see
    apply_structuring_update's action_items dedup, same normalization).
    Without this check, should_speak_aloud/describe_speak_reason would
    re-fire "action_item_owner" once per re-extraction instead of once
    per actual assignment -- confirmed live (incident-46): "Got it, Ted's
    on that." spoken four times for what was one real assignment.
    """
    key = action_item_key(item.text)
    for existing in structured_state.get("action_items", []) or []:
        if action_item_key(existing.get("text", "")) == key:
            return not existing.get("owner")
    return True


def should_speak_aloud(
    update: StructuringUpdate, latest_user_message: Optional[str], structured_state: dict
) -> bool:
    """
    True if this turn's spoken_reply should actually reach the room's
    speakers. False means the caller sends empty content to Agora's TTS
    instead -- spoken_reply is never persisted anywhere (see
    apply_structuring_update), so suppressing it loses nothing recorded,
    only the audible acknowledgment itself. Anything actually worth
    keeping on a quiet turn belongs in agent_chat_note, which the model
    sets independently of this gate.

    is_wrapping_up and corrects_fact are NOT checked here -- both always
    speak via their own dedicated, deterministic paths in
    iCall_api._process_turn (a fixed closing line, and a fixed correction
    callout respectively), same reasoning both times: what gets said in
    those two moments must be guaranteed-accurate, not left to
    update.spoken_reply's free-text phrasing.
    """
    if update.conflict:
        return True
    if _has_speakable_missing_info(update.missing_info, structured_state):
        return True
    if any(item.owner and _is_new_owner_assignment(item, structured_state) for item in update.action_items):
        return True
    if (
        latest_user_message
        and _is_direct_address(latest_user_message)
        and _direct_address_off_cooldown(structured_state)
    ):
        return True
    return False


def build_gated_spoken_reply(update: StructuringUpdate, reason: str, structured_state: dict) -> str:
    """
    Deterministic spoken text for the three should_speak_aloud reasons that
    have an already-extracted structured field to build from -- conflict,
    missing_info, action_item_owner. direct_address has no such field (it's
    open-ended conversational content), so that reason still uses
    update.spoken_reply as-is.

    structured_state is only actually used by the action_item_owner branch
    (to pick the item whose owner is genuinely new, same
    _is_new_owner_assignment check as should_speak_aloud/
    describe_speak_reason -- see there for why) -- required anyway, rather
    than optional, since reason is only ever "action_item_owner" when
    those two already confirmed a new assignment exists, so a caller that
    somehow got here without state would be masking a real bug, not
    hitting a legitimate no-state case.

    Confirmed live (incident-101, after the gemini-flash-lite-latest swap):
    update.spoken_reply produced generic acknowledgments ("Got it, noting
    that a CDN Edge module failure...") for turns tagged reason=missing_info,
    violating the system prompt's explicit anti-generic-acknowledgment rule
    -- plausibly smaller/faster models degrading on nuanced instruction-
    following (documented in the literature as up to a 61.8% drop in one
    study), though not certain. Rather than keep tightening the prompt and
    hoping the model complies, build the spoken text directly from the
    fields the model already extracted correctly (conflict, missing_info,
    action_items) -- same "wrap the non-deterministic model inside
    deterministic software boundaries" discipline as CLOSING_LINE and
    build_correction_callout. This removes an entire class of prompt-
    compliance risk at zero added latency (no extra model call, just string
    formatting on data already in hand).
    """
    if reason == "conflict":
        # update.conflict is itself specified (see the corrects_fact/conflict
        # prompt guidance) as "one short, targeted clarifying question you
        # would ask out loud" -- already the exact text to speak, not a
        # paraphrase target.
        return update.conflict or ""
    if reason == "missing_info":
        gap = update.missing_info[0] if update.missing_info else ""
        return f"Can you clarify: {gap}?" if gap else ""
    if reason == "action_item_owner":
        owner_item = next(
            (item for item in update.action_items if item.owner and _is_new_owner_assignment(item, structured_state)),
            None,
        )
        return f"Got it, {owner_item.owner}'s on that." if owner_item else ""
    return update.spoken_reply


def describe_speak_reason(
    update: StructuringUpdate, latest_user_message: Optional[str], structured_state: dict
) -> str:
    """
    Same branch order as should_speak_aloud, but names which check fired
    instead of returning a bare bool -- only for AgentUtterance.reason
    (see iCall_api.chat_completions_endpoint), so "why did the AI say
    that" stays answerable after the fact. Only call this when
    should_speak_aloud already returned True; it assumes one of these
    branches fired and doesn't itself re-check that.
    """
    if update.conflict:
        return "conflict"
    if _has_speakable_missing_info(update.missing_info, structured_state):
        return "missing_info"
    if any(item.owner and _is_new_owner_assignment(item, structured_state) for item in update.action_items):
        return "action_item_owner"
    return "direct_address"


# -----------------------------------------------------------------------------
# Pattern watching over the accumulated call state -- the difference between
# a passive log L3 polls and a memory that notices patterns in itself. Every
# check here is deterministic, computed from structured_state alone, no LLM
# call: it's watching for patterns ACROSS already-recorded events (how many
# conflicts, how stale an item is, how many competing theories), not judging
# any single new statement, which is a genuinely different question from
# what should_speak_aloud answers above. Cheap enough to run every turn.
#
# Deliberately scoped to what the current schema actually supports rather
# than the full design: a "previously-confirmed fact gets recontradicted"
# pattern needs tentative/confirmed promotion tracking this schema doesn't
# have yet (facts are a flat list, no confidence state), so it's left out
# here rather than faked with a fragile text-matching heuristic.
# -----------------------------------------------------------------------------

CONFLICT_PILEUP_THRESHOLD = 2
CONFLICT_PILEUP_WINDOW_MINUTES = 5

HYPOTHESIS_CLUSTER_THRESHOLD = 3
HYPOTHESIS_CLUSTER_WINDOW_MINUTES = 5

STALE_ACTION_ITEM_MINUTES = 5

HEALTH_SCORE_DROP_THRESHOLD = 20
HEALTH_SCORE_HISTORY_MAX_LEN = 20


def _recent_timeline_entries(timeline: List[dict], entry_type: str, window_minutes: int) -> List[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    recent = []
    for entry in timeline:
        if entry.get("type") != entry_type:
            continue
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if ts >= cutoff:
            recent.append(entry)
    return recent


def _stale_unowned_action_item(structured_state: dict) -> Optional[dict]:
    """
    Matches an action item back to its timeline entry by text to find when
    it was recorded -- action_items themselves aren't individually
    timestamped, only their timeline entry is. Fragile if two items share
    identical text; acceptable at this scope, not worth a schema change to
    fully solve tonight.
    """
    timeline_ts_by_text = {
        e["text"]: e.get("timestamp")
        for e in structured_state.get("timeline", [])
        if e.get("type") == "action_item"
    }
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_ACTION_ITEM_MINUTES)
    for item in structured_state.get("action_items", []):
        if item.get("owner") or item.get("owner_uid"):
            continue
        ts_str = timeline_ts_by_text.get(item.get("text"))
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if ts <= cutoff:
            return item
    return None


# Confirmed live (incident-101/call_id=14): with no cooldown at all,
# confusion_cluster fired 9x and conflict_pileup fired 4x, identically
# worded, within ~20 seconds -- these are pure window-threshold checks over
# the timeline with zero memory of "did I just say this," so once the
# threshold is crossed every subsequent turn re-trips it again. Same bug
# class already fixed for missing_info/direct_address/health_score_drop
# (see MISSING_INFO_NUDGE_COOLDOWN_S, DIRECT_ADDRESS_REPLY_COOLDOWN_S), just
# never applied here. Cooldown is per-pattern-name (not a single shared
# timestamp), so a suppressed conflict_pileup doesn't also block a
# genuinely separate stale_action_item from being spoken. 90s: long enough
# to kill the observed rapid-fire spam (repeats every 2-20s in the wild),
# short enough that a pattern which is still genuinely unresolved after
# 90s of real conversation gets surfaced again rather than going silent
# for the rest of the call.
PATTERN_NUDGE_COOLDOWN_S = 90


def _pattern_off_cooldown(structured_state: dict, pattern_name: str) -> bool:
    last_at = (structured_state.get("last_pattern_nudge_at") or {}).get(pattern_name)
    if not last_at:
        return True
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_at)).total_seconds()
    return elapsed >= PATTERN_NUDGE_COOLDOWN_S


def evaluate_call_patterns(structured_state: dict) -> Optional[dict]:
    """
    Checked in priority order, first match wins -- returns None if nothing
    fired. Only called when should_speak_aloud's turn-level checks (an
    explicit conflict/missing-info/wake-word this turn) already decided
    not to speak, so a pattern nudge never competes with something more
    directly relevant to what was just said.

    A pattern whose condition is met but is still on its own cooldown
    (see PATTERN_NUDGE_COOLDOWN_S) is skipped, not treated as a match --
    the priority order then falls through to check the next pattern, so
    one pattern being on cooldown never masks a genuinely different one.
    """
    timeline = structured_state.get("timeline", [])

    if (
        len(_recent_timeline_entries(timeline, "conflict", CONFLICT_PILEUP_WINDOW_MINUTES)) >= CONFLICT_PILEUP_THRESHOLD
        and _pattern_off_cooldown(structured_state, "conflict_pileup")
    ):
        return {
            "pattern": "conflict_pileup",
            "message": "We've got a few open questions piling up -- want to pause and reconcile before moving on?",
        }

    stale_item = _stale_unowned_action_item(structured_state)
    if stale_item and _pattern_off_cooldown(structured_state, "stale_action_item"):
        return {
            "pattern": "stale_action_item",
            "message": f"Just checking in -- \"{stale_item['text']}\" still doesn't have an owner. Can someone take that?",
        }

    if (
        len(_recent_timeline_entries(timeline, "hypothesis", HYPOTHESIS_CLUSTER_WINDOW_MINUTES)) >= HYPOTHESIS_CLUSTER_THRESHOLD
        and _pattern_off_cooldown(structured_state, "confusion_cluster")
    ):
        return {
            "pattern": "confusion_cluster",
            "message": "A few different theories have come up in the last few minutes -- want to narrow down to one to test first?",
        }

    return None


def compute_coordination_health_score(structured_state: dict) -> int:
    """
    0-100, higher is healthier -- a rough aggregate over open conflicts,
    missing-info gaps, unowned action items, and time since the last
    decision. The same idea data-observability tools use to reduce many
    raw signals into one number worth acting on, pointed at the room's own
    coordination instead of the payment system it's discussing. Weights
    are a judgment call, not derived from anything measured; deliberately
    simple and code-computed rather than model-judged, so it's at least
    consistent and explainable.
    """
    score = 100
    score -= 15 * len(structured_state.get("conflicts", []))
    score -= 10 * len(structured_state.get("missing_info", []))
    # A fact getting contradicted a SECOND time (settled ground coming loose
    # again, not an ordinary first contradiction) is weighted heavier than a
    # routine conflict -- see apply_structuring_update's
    # previously_corrected_facts chain for how this gets detected.
    score -= 25 * structured_state.get("second_contradiction_count", 0)

    timeline = structured_state.get("timeline", [])
    decision_timestamps = []
    for e in timeline:
        if e.get("type") != "decision":
            continue
        try:
            decision_timestamps.append(datetime.fromisoformat(e["timestamp"]))
        except (KeyError, ValueError, TypeError):
            continue

    if decision_timestamps:
        minutes_since = (datetime.now(timezone.utc) - max(decision_timestamps)).total_seconds() / 60
        if minutes_since > 10:
            score -= 15
    elif timeline:
        # No decision at all yet -- mild penalty, not severe, since that's
        # completely normal early in a call.
        score -= 5

    unowned_items = sum(
        1 for item in structured_state.get("action_items", [])
        if not (item.get("owner") or item.get("owner_uid"))
    )
    score -= 5 * unowned_items

    return max(0, min(100, score))


def detect_health_score_drop(structured_state: dict) -> bool:
    """
    A SHARP DROP OR A SLOW BLEED, not a static low value -- a call that's
    held steady at 60 all along isn't an emergency; one that fell from 90
    to 55 is, whether that happened in one turn or over ten. Needs history
    (see iCall_service.record_health_score), since a single snapshot can't
    tell a drop from a call that started low.

    Compares against the call's own peak so far (bounded by
    HEALTH_SCORE_HISTORY_MAX_LEN, not literally unbounded), not just a
    short recent window -- confirmed live (incident-39): a real 25-point
    gradual decline (100 -> 75 over ~2 minutes, each individual step only
    5-10 points) never crossed HEALTH_SCORE_DROP_THRESHOLD under the old
    5-entry lookback, because no single 5-entry window ever showed more
    than a 10-15 point swing. Comparing against the whole stored history's
    peak catches both shapes of decline with one check.

    Also fires on ANY currently-open conflict alone, regardless of the
    aggregate score threshold -- a real logical gap found by reasoning
    through the scoring weights (compute_coordination_health_score: -15
    per conflict, -10 per missing_info, -15 stale-decision, -5 unowned
    item): a single open conflict only costs 15 points, so it could never
    cross a 20-point threshold on its own. That's backwards -- an
    unresolved contradiction is arguably the single clearest signal
    something's wrong, and this mechanism (built specifically to catch
    "things have gotten bad") shouldn't require a SECOND problem to also
    be true before it can fire. Deliberately not achieved by just
    lowering HEALTH_SCORE_DROP_THRESHOLD instead -- that would also make
    unrelated combinations (e.g. two unowned items early in a call) newly
    qualify, which isn't the same thing as "there's an unresolved
    conflict" and would be a new false positive, not a fix.

    Re-nudges only if things have gotten WORSE since the last nudge, not
    on every turn the score merely stays below some historical peak (and,
    for the conflict-alone path, not on every turn a conflict merely
    remains open with nothing else changed -- the score staying flat is
    exactly that case). Confirmed live (incident-26): with no such check,
    this fired on 2-3 consecutive qualifying turns with nothing new to
    report, repeating the identical recap -- the opposite of "at
    appropriate moments" from the brief. last_health_score_drop_nudge_score
    is set by iCall_service.record_pattern_nudge each time this actually
    speaks.
    """
    history = structured_state.get("health_score_history", [])
    if len(history) < 2:
        return False
    current = history[-1]["score"]
    peak = max(h["score"] for h in history[:-1])
    score_dropped_enough = (peak - current) >= HEALTH_SCORE_DROP_THRESHOLD
    has_open_conflict = bool(structured_state.get("conflicts"))
    if not score_dropped_enough and not has_open_conflict:
        return False
    last_nudge_score = structured_state.get("last_health_score_drop_nudge_score")
    if last_nudge_score is not None and current >= last_nudge_score:
        return False
    return True


def build_health_recap(structured_state: dict) -> str:
    """
    Deterministic, not model-generated -- a status recap triggered by a
    code-computed pattern shouldn't itself depend on another LLM call to
    say something coherent about that same state.

    Names actual conflicts/unowned items rather than only counting them --
    "2 conflicts still unresolved" with no indication of *what* they are
    is technically accurate but not actually useful to someone on the
    call. The content is already sitting in structured_state (extracted
    by the same turn that recorded it); this just uses more of it. Zero
    added latency or cost -- no new LLM call, just a richer deterministic
    template over data already in hand. Facts/hypotheses stay as counts
    since those lists can grow long over a call; conflicts and unowned
    items are usually few and are the actionable part of a recap.
    """
    facts_n = len(structured_state.get("facts", []))
    hyp_n = len(structured_state.get("hypotheses", []))
    items_n = len(structured_state.get("action_items", []))
    conflicts = structured_state.get("conflicts", [])
    conflicts_n = len(conflicts)

    summary = (
        f"Quick status check -- {facts_n} facts confirmed, {hyp_n} open "
        f"theories, {items_n} action items tracked."
    )

    if conflicts:
        named = "; ".join(conflicts[:2])
        more = f" and {conflicts_n - 2} more" if conflicts_n > 2 else ""
        summary += f" Still unresolved: {named}{more}."
    else:
        summary += " Nothing unresolved right now."

    unowned = [item["text"] for item in structured_state.get("action_items", []) if not (item.get("owner") or item.get("owner_uid"))]
    if unowned:
        summary += f' Still needs an owner: "{unowned[0]}"' + (f" and {len(unowned) - 1} more." if len(unowned) > 1 else ".")

    return summary


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
    "ai_engineer",
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

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=PRIMARY_MODEL,
                contents=prompt,
                config=config,
            ),
            timeout=GEMINI_CALL_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"[iCall] Role classification primary call failed/timed out, trying fallback: {exc}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=config,
                ),
                timeout=GEMINI_CALL_TIMEOUT_S,
            )
        except Exception as exc2:
            # Runs once, at call end, with no retry beyond this -- a
            # silent failure here means role recognition for this call
            # is permanently empty with nothing in the logs to explain
            # why. Caught by a dry run that happened to hit a transient
            # failure with no visible cause until this was added.
            print(f"[iCall] Role classification failed on both attempts, leaving roles unresolved: {exc2}")
            return CallRoleClassification()

    try:
        return CallRoleClassification.model_validate_json(response.text)
    except Exception as exc:
        print(f"[iCall] Role classification response didn't match schema, leaving roles unresolved: {exc}")
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

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=PRIMARY_MODEL,
                contents=prompt,
                config=config,
            ),
            timeout=GEMINI_CALL_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"[iCall] Owner assignment primary call failed/timed out, trying fallback: {exc}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=config,
                ),
                timeout=GEMINI_CALL_TIMEOUT_S,
            )
        except Exception as exc2:
            print(f"[iCall] Owner assignment failed on both attempts, leaving items unassigned: {exc2}")
            return ActionItemOwnerAssignments()

    try:
        return ActionItemOwnerAssignments.model_validate_json(response.text)
    except Exception as exc:
        print(f"[iCall] Owner assignment response didn't match schema, leaving items unassigned: {exc}")
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
- An action item counts as having an owner if EITHER its "owner" field OR
  its "owner_uid" field is set (owner_uid means it was matched to someone
  on the call by role, not by name -- that still counts as assigned). Only
  flag an action item as a risk if both are empty/null.
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

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=PRIMARY_MODEL,
                contents=prompt,
                config=config,
            ),
            timeout=GEMINI_CALL_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"[iCall] Unresolved-risks primary call failed/timed out, trying fallback: {exc}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=config,
                ),
                timeout=GEMINI_CALL_TIMEOUT_S,
            )
        except Exception as exc2:
            print(f"[iCall] Unresolved-risks summary failed on both attempts, leaving risks empty: {exc2}")
            return UnresolvedRisksSummary()

    try:
        return UnresolvedRisksSummary.model_validate_json(response.text)
    except Exception as exc:
        print(f"[iCall] Unresolved-risks response didn't match schema, leaving risks empty: {exc}")
        return UnresolvedRisksSummary()


# -----------------------------------------------------------------------------
# Pre-Jira-ticket content review -- confirmed live (incident-101, a solo
# test call): the room's raw recorded state can carry test artifacts
# ("wait for the teammate to join"), and repeated/garbled restatements of
# the same guess across turns (STT noise -- "reverse east" for "US East"),
# that read fine as a live coordination aid but badly as an external ticket
# someone outside the call has to act on. This is a one-time, read-only
# cleanup applied ONLY when building the ticket body -- never written back
# to IncidentCall.structured_state, so it can never affect the live call's
# own system-of-record (corrects_fact matching, health scoring, the stale-
# action-item nudge) the way editing the real state in place would.
# -----------------------------------------------------------------------------

TICKET_CONTENT_REVIEW_SYSTEM_INSTRUCTION = """You are cleaning up a live
incident call's recorded notes before they become a Jira ticket description
that someone outside the call will read and act on. Your job is to make it
read like a real ticket, not to change what actually happened.

Rules:
- Remove an item only when it's clearly not about the incident itself --
  call logistics or filler like "wait for the teammate to join," a greeting
  that got recorded as a fact, or an obvious test/placeholder statement.
  When in doubt, keep it: don't remove anything that could plausibly matter
  to someone investigating this incident later.
- Merge hypotheses that are just restatements of the same underlying guess
  worded differently across turns (common with live speech-to-text noise,
  e.g. "the network is down in a specific region" / "the region is US
  East" / a mis-transcribed variant of the same word) into ONE clearly
  phrased version. Do not merge hypotheses that are actually different
  theories, even if they sound related.
- Fix awkward, garbled, or run-on phrasing into clear, professional
  English -- without changing what was actually asked, decided, or
  observed, and without resolving a question that was never actually
  answered on the call. A still-open conflict must stay phrased as an open
  question, just a clearly worded one.
- Never invent information that wasn't recorded, and never drop a
  genuinely distinct fact, decision, missing-info gap, conflict, or risk --
  only remove true noise and merge true duplicates.
- Output must strictly match the response schema below.
"""


def _build_ticket_content_review_prompt(structured_state: dict) -> str:
    return f"""Recorded state for this incident call, to be cleaned up for
a Jira ticket:
{structured_state}

Produce the cleaned version per the rules above.
"""


def _passthrough_reviewed_ticket_content(structured_state: dict) -> ReviewedTicketContent:
    """
    Safe degrade for review_ticket_content: unlike summarize_unresolved_
    risks (where "no risks" is a valid answer), an empty result here would
    mean a Jira ticket gets created with its description silently wiped --
    much worse than the pre-review, slightly messy text. On any failure,
    fall back to the ORIGINAL unedited lists rather than an empty review.
    """
    state = structured_state or {}
    return ReviewedTicketContent(
        facts=list(state.get("facts", [])),
        hypotheses=list(state.get("hypotheses", [])),
        decisions=list(state.get("decisions", [])),
        missing_info=list(state.get("missing_info", [])),
        conflicts=list(state.get("conflicts", [])),
        unresolved_risks=list(state.get("unresolved_risks", [])),
    )


async def review_ticket_content(structured_state: dict) -> ReviewedTicketContent:
    """
    One pass, run only when a Jira ticket is actually about to be created
    (see iOrchestrate_api._handle_jira_decision) -- never on the live call
    path. Falls back to the unedited original content (not an empty
    result) whenever there's nothing to review or the model call fails.
    """
    fallback = _passthrough_reviewed_ticket_content(structured_state)
    if not structured_state or not get_settings().gemini_api_key:
        return fallback

    client = _get_client()
    prompt = _build_ticket_content_review_prompt(structured_state)
    config = types.GenerateContentConfig(
        system_instruction=TICKET_CONTENT_REVIEW_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=ReviewedTicketContent,
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=PRIMARY_MODEL,
                contents=prompt,
                config=config,
            ),
            timeout=GEMINI_CALL_TIMEOUT_S,
        )
    except Exception as exc:
        print(f"[iCall] Ticket-content review primary call failed/timed out, trying fallback: {exc}")
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=config,
                ),
                timeout=GEMINI_CALL_TIMEOUT_S,
            )
        except Exception as exc2:
            print(f"[iCall] Ticket-content review failed on both attempts, using unedited content: {exc2}")
            return fallback

    try:
        return ReviewedTicketContent.model_validate_json(response.text)
    except Exception as exc:
        print(f"[iCall] Ticket-content review response didn't match schema, using unedited content: {exc}")
        return fallback


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


def format_live_recap(structured_state: dict) -> Optional[str]:
    """
    Plain-text catch-up message for someone joining a call already in
    progress. Deterministic, no LLM call -- the data was already extracted
    live by generate_structuring_update, this just renders it. Unlike
    iOrchestrate's format_call_summary (Slack-flavored *bold* markup, meant
    for the post-call channel post), this is meant to be dropped straight
    into the in-call text chat, so it stays plain and short.

    Returns None when nothing's been recorded yet (call just started, or
    everyone present so far has only just joined) so callers can skip
    sending a pointless "nothing to report" message to a joiner.
    """
    state = structured_state or {}
    facts = state.get("facts", [])
    hypotheses = state.get("hypotheses", [])
    decisions = state.get("decisions", [])
    action_items = state.get("action_items", [])

    if not (facts or hypotheses or decisions or action_items):
        return None

    lines = ["Quick catch-up on what's been discussed so far:"]

    def _section(title: str, items: List[str]) -> None:
        if not items:
            return
        lines.append(f"\n{title}:")
        for item in items:
            lines.append(f"- {item}")

    _section("Facts", facts)
    _section("Working theories", hypotheses)
    _section("Decisions", decisions)

    if action_items:
        lines.append("\nAction items:")
        for item in action_items:
            owner = item.get("owner") or "unassigned"
            lines.append(f"- {item.get('text', '')} (owner: {owner})")

    return "\n".join(lines)


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
