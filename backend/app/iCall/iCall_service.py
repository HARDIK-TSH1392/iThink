import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from typing import Dict, List, Optional

from app.iNcidents.iNcidents_crudl import get_incident, list_incidents
from app.iNcidents.iNcidents_utils import STATUS_RESOLVED
from app.iDirectory.iDirectory_crudl import find_employee_by_name

from .iCall_model import IncidentCall, CallUtterance, AgentUtterance
from .iCall_schema import CallUtteranceCreate, StructuringUpdate
from .iCall_utils import (
    generate_channel_name,
    format_related_incident_note,
    classify_participant_roles,
    assign_action_item_owners,
    summarize_unresolved_risks,
    action_item_key,
    CALL_STATUS_SCHEDULED,
    HEALTH_SCORE_HISTORY_MAX_LEN,
)

# Serializes "check for an existing call, else create one" per incident_id,
# the same class of fix as iTriage's per-key lock: without it, concurrent
# requests for the same incident_id can both pass the "no existing call"
# check before either commits, and the loser hits the DB's unique
# constraint on incident_id as an unhandled error instead of just getting
# the winner's row back.
_incident_call_locks: Dict[int, asyncio.Lock] = {}
_locks_registry_guard = asyncio.Lock()


async def _get_incident_lock(incident_id: int) -> asyncio.Lock:
    async with _locks_registry_guard:
        if incident_id not in _incident_call_locks:
            _incident_call_locks[incident_id] = asyncio.Lock()
        return _incident_call_locks[incident_id]


# Same race as above, different trigger: Agora's Custom LLM webhook fires
# once per completed STT turn, and fragmented speech (a sentence split
# across two turns by a mid-sentence pause) can produce two overlapping
# chat_completions_endpoint requests for the same call. Each reads
# call.structured_state, spends several seconds in a Gemini call, then
# writes back -- without serializing that per call_id, the second commit
# silently clobbers the first's changes (a real lost-update, not
# hypothetical: confirmed live, incident-29 -- an owned action item both
# turns extracted and spoke about ended up recorded only once, from
# whichever request committed last). Locking also gives the second turn
# an accurate existing_state to extract against (see
# _build_structuring_prompt's "use this to detect contradictions, not to
# repeat"), since it now only starts after the first turn's update has
# actually landed, instead of both racing off the same stale snapshot.
_call_turn_locks: Dict[int, asyncio.Lock] = {}
_call_turn_locks_guard = asyncio.Lock()


async def get_call_turn_lock(call_id: int) -> asyncio.Lock:
    async with _call_turn_locks_guard:
        if call_id not in _call_turn_locks:
            _call_turn_locks[call_id] = asyncio.Lock()
        return _call_turn_locks[call_id]


class IncidentNotFoundError(Exception):
    pass


async def get_call_by_incident(db: AsyncSession, incident_id: int) -> Optional[IncidentCall]:
    result = await db.execute(
        select(IncidentCall).where(IncidentCall.incident_id == incident_id)
    )
    return result.scalar_one_or_none()


async def get_or_create_call(
    db: AsyncSession, incident_id: int, language_code: Optional[str] = None
) -> IncidentCall:
    """
    The one place a channel name is decided. Orchestration (calendar/email
    invite) and the voice agent (RTC join) both call this instead of
    independently computing a channel name, so there is exactly one source
    of truth per incident rather than two formulas that could drift apart.

    language_code is only applied on the create branch -- an existing call
    ignores it, same idempotent-creation semantics as channel_name itself.
    "multi" (Deepgram, English+Hindi native code-switching) is the default
    when unset.

    Raises IncidentNotFoundError if incident_id doesn't exist — the API
    layer turns that into a 404 rather than letting a bad foreign key
    surface as a raw database error.
    """
    lock = await _get_incident_lock(incident_id)
    async with lock:
        existing = await get_call_by_incident(db, incident_id)
        if existing is not None:
            return existing

        incident = await get_incident(db, incident_id)
        if incident is None:
            raise IncidentNotFoundError(f"Incident {incident_id} not found")

        # Cross-incident memory: every call starts fresh today even when
        # the exact same service/region had a resolved incident before --
        # looked up once here (not every turn, unlike service/region
        # grounding above which is cheap and already refetched per turn)
        # since a call's whole duration won't change which past incident
        # is most recent, and stored directly on the new row so later
        # turns just read it back with no extra query.
        initial_state: dict = {}
        past_incidents = await list_incidents(
            db, status=STATUS_RESOLVED, region=incident.region, service=incident.service, limit=5,
        )
        related = next((i for i in past_incidents if i.id != incident_id), None)
        if related is not None:
            initial_state["related_incident_note"] = format_related_incident_note(related)

        # Delegate mode: the resolved approver approved but can't
        # personally join (see iOrchestrate_api's approve_delegate/modal
        # flow) -- their notes ride into the call the same way
        # related_incident_note does, read by _build_structuring_prompt
        # and by voice-agent/server's greeting/avatar fetch (see
        # iCall_api's GET .../delegate endpoint).
        if incident.delegate_notes:
            initial_state["delegate_notes"] = incident.delegate_notes

        call = IncidentCall(
            incident_id=incident_id,
            channel_name=generate_channel_name(incident_id),
            status=CALL_STATUS_SCHEDULED,
            structured_state=initial_state,
            language_code=language_code or "multi",
        )
        db.add(call)
        await db.commit()
        await db.refresh(call)
        return call


async def get_call(db: AsyncSession, call_id: int) -> Optional[IncidentCall]:
    return await db.get(IncidentCall, call_id)


async def get_call_by_channel_name(db: AsyncSession, channel_name: str) -> Optional[IncidentCall]:
    """
    Resolves a call from the Agora channel name alone. This is what the
    custom-LLM endpoint uses to figure out *which* incident's call it's
    serving — Agora's request body carries no incident/call id, only the
    conversation content, so the channel name in the URL path is the only
    thing tying a given request back to a specific IncidentCall row.
    """
    result = await db.execute(
        select(IncidentCall).where(IncidentCall.channel_name == channel_name)
    )
    return result.scalar_one_or_none()


async def apply_structuring_update(
    db: AsyncSession, call: IncidentCall, update: StructuringUpdate
) -> IncidentCall:
    """
    Merges one turn's extraction into structured_state by appending —
    deliberately never overwrites the existing lists wholesale. Mirrors the
    "/update overwrites params entirely" gotcha documented for Agora's own
    agent-update endpoint: the same failure mode (silently losing prior
    state) is just as real here if this merged the LLM's per-turn output
    in as the new state instead of adding to what's already recorded.

    Builds brand-new list objects for every key rather than a shallow
    dict(...) copy — a shallow copy leaves the nested lists shared with
    call.structured_state's existing lists, so mutating them in place also
    mutates the "old" value SQLAlchemy compares against, which makes it
    see old == new and silently skip the UPDATE on second and later calls.
    Caught this via the two-turn accumulation test, not by inspection.

    timeline is deliberately not an LLM-extracted field: it's a
    deterministic merge-and-sort of what's already being extracted (facts,
    decisions, action items, conflicts, missing info), each stamped with
    when it was recorded. The LLM decides *what* was said; this function
    decides *when it goes in the timeline* -- same "don't let the model
    freely construct derived state" discipline as everything else here.
    """
    old = call.structured_state or {}
    # Start from a shallow copy so any key this function doesn't know about
    # (health_score_history, unresolved_risks, and anything added later)
    # passes through untouched, instead of being silently dropped by an
    # explicit key list -- found via a dry run that fed health_score_history
    # through record_health_score, then watched apply_structuring_update
    # wipe it on the very next turn. Safe specifically because a shallow
    # copy only breaks when a nested object is later mutated in place; the
    # fields below are all deliberately rebuilt as fresh lists (the actual
    # fix for that original bug), and anything just passed through here is
    # never mutated by this function at all.
    state = dict(old)
    state.update({
        "facts": list(old.get("facts", [])),
        "hypotheses": list(old.get("hypotheses", [])),
        "decisions": list(old.get("decisions", [])),
        "action_items": list(old.get("action_items", [])),
        "missing_info": list(old.get("missing_info", [])),
        "conflicts": list(old.get("conflicts", [])),
        "identified_speakers": list(old.get("identified_speakers", [])),
        "timeline": list(old.get("timeline", [])),
        "chat_notes": list(old.get("chat_notes", [])),
        # Tracks facts that are themselves the PRODUCT of an earlier
        # correction -- not a fixed ID system (facts are plain strings with
        # no stable identity across a correction), just enough of a chain to
        # recognize "the thing being corrected right now was already a
        # correction once" as distinct from an ordinary first-time
        # contradiction. See the corrects_fact handling below for how this
        # gets read and extended.
        "previously_corrected_facts": list(old.get("previously_corrected_facts", [])),
        # This function only ever runs for a real turn (the silence-trigger
        # branch skips it entirely -- see iCall_api._process_turn), so
        # reaching here means the room actually said something. Resets the
        # silence-escalation counter every time, matching
        # detect_health_score_drop's own "only re-fire on genuinely new
        # information" discipline.
        "silence_streak": 0,
    })

    now = datetime.now(timezone.utc).isoformat()

    def _add_timeline_entry(entry_type: str, text: str) -> None:
        state["timeline"].append({"type": entry_type, "text": text, "timestamp": now})

    # Fail safe, not fail silent: only remove the superseded fact when the
    # model's corrects_fact text matches one already in the active list
    # exactly. A hallucinated or slightly-misquoted reference just does
    # nothing here -- better to occasionally leave a stale fact in place
    # than to ever guess-delete the wrong one. The active facts list stays
    # a current-understanding snapshot; the timeline (below) keeps the
    # superseded fact too, since it's the actual historical record.
    if update.corrects_fact and update.corrects_fact in state["facts"]:
        state["facts"].remove(update.corrects_fact)
        # Escalation: is the fact being corrected RIGHT NOW itself the
        # result of an earlier correction? That's settled ground coming
        # loose a second time, not an ordinary first contradiction -- more
        # alarming, and worth a bigger, immediate health-score hit rather
        # than waiting to be noticed the normal way (see
        # compute_coordination_health_score's second_contradiction_count
        # penalty). An ordinary first-time correction still records
        # normally, just without the escalation marker.
        is_second_contradiction = update.corrects_fact in state["previously_corrected_facts"]
        state["second_contradiction_count"] = old.get("second_contradiction_count", 0) + (
            1 if is_second_contradiction else 0
        )
        if update.facts:
            state["previously_corrected_facts"].append(update.facts[0])
        _add_timeline_entry(
            "second_correction" if is_second_contradiction else "correction",
            f'Correction: "{update.corrects_fact}" is superseded by updated information.',
        )

    state["facts"].extend(update.facts)
    for fact in update.facts:
        _add_timeline_entry("fact", fact)

    state["hypotheses"].extend(update.hypotheses)
    for hypothesis in update.hypotheses:
        _add_timeline_entry("hypothesis", hypothesis)

    state["decisions"].extend(update.decisions)
    for decision in update.decisions:
        _add_timeline_entry("decision", decision)

    # Dedup by normalized text (see iCall_utils.action_item_key) before
    # appending -- without this, an STT-fragmented instruction that gets
    # re-extracted across several turns (the room circling back to
    # confirm an owner, say) produced a separate action_items entry each
    # time. Confirmed live (incident-46): "check the port config" ended up
    # recorded three times, once unowned then twice with the same owner.
    # A genuinely new item (new text) still appends normally; a repeat of
    # existing text merges into it instead -- filling in the owner if this
    # turn is the one that supplies it, otherwise a pure no-op. Matching
    # should_speak_aloud/describe_speak_reason/build_gated_spoken_reply use
    # the same normalization to decide whether to announce the assignment,
    # so the timeline and the spoken confirmation never disagree about
    # what's actually new.
    for item in update.action_items:
        key = action_item_key(item.text)
        existing_index = next(
            (i for i, ai in enumerate(state["action_items"]) if action_item_key(ai.get("text", "")) == key),
            None,
        )
        if existing_index is None:
            state["action_items"].append(item.model_dump())
            _add_timeline_entry("action_item", item.text)
        elif item.owner and not state["action_items"][existing_index].get("owner"):
            # Replace with a NEW dict rather than mutating the existing one
            # in place -- state["action_items"] is a shallow copy of
            # old["action_items"], so the entries themselves are still the
            # SAME dict objects as in call.structured_state (the value
            # SQLAlchemy already tracks). Mutating one in place would mutate
            # that old value too, defeating the old != new change-detection
            # this function's own top-of-function docstring already warns
            # about -- caught by the dedup regression test doing exactly
            # this (backend/eval/test_action_item_dedup.py), not by
            # inspection.
            state["action_items"][existing_index] = {
                **state["action_items"][existing_index],
                "owner": item.owner,
            }

    state["missing_info"].extend(update.missing_info)
    for gap in update.missing_info:
        _add_timeline_entry("missing_info", gap)

    if update.conflict:
        state["conflicts"].append(update.conflict)
        _add_timeline_entry("conflict", update.conflict)

    for speaker in update.identified_speakers:
        if speaker not in state["identified_speakers"]:
            state["identified_speakers"].append(speaker)

    if update.agent_chat_note:
        state["chat_notes"].append({"text": update.agent_chat_note, "timestamp": now})
        _add_timeline_entry("chat_note", update.agent_chat_note)

    call.structured_state = state
    await db.commit()
    await db.refresh(call)
    return call


async def record_health_score(db: AsyncSession, call: IncidentCall, score: int) -> IncidentCall:
    """
    Appends this turn's coordination-health score to a rolling history --
    needed because detect_health_score_drop cares about a SHARP DROP, not
    a static low value, which is only knowable with history to compare
    against. Capped at HEALTH_SCORE_HISTORY_MAX_LEN entries since only the
    recent window matters for drop detection; an unbounded list would grow
    for the entire life of a long call for no benefit.

    Same rebuild-fresh-objects discipline as apply_structuring_update --
    dict(old) plus a new list, never mutating call.structured_state's
    existing nested objects in place.
    """
    old = call.structured_state or {}
    history = list(old.get("health_score_history", []))
    history.append({"score": score, "timestamp": datetime.now(timezone.utc).isoformat()})
    history = history[-HEALTH_SCORE_HISTORY_MAX_LEN:]

    new_state = dict(old)
    new_state["health_score_history"] = history
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_missing_info_nudge(db: AsyncSession, call: IncidentCall) -> IncidentCall:
    """
    Stamps when a missing_info gap was last actually spoken about --
    iCall_utils._has_speakable_missing_info reads this to throttle
    re-asking the SAME still-open gap on every qualifying turn. Same
    cooldown-marker pattern as record_pattern_nudge's score param for
    detect_health_score_drop, just for a turn-level gate instead of a CEP
    pattern.
    """
    old = call.structured_state or {}
    new_state = dict(old)
    new_state["last_missing_info_nudge_at"] = datetime.now(timezone.utc).isoformat()
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_last_seen_user_message(db: AsyncSession, call: IncidentCall, text: Optional[str]) -> IncidentCall:
    """
    Stamps the raw text of this turn's latest user message so the NEXT
    turn can diff against it (see iCall_utils._new_content_since) --
    confirmed live (incident-68) that Agora accumulates unanswered speech
    onto one growing message rather than starting a fresh one each turn,
    so a stale Hindi fragment from several exchanges ago can still be
    sitting in the raw text a brand-new English question arrives glued
    onto. Without this stored reference, there's nothing to diff against
    and the whole accumulated blob has to be treated as "the latest
    message," which is exactly what caused that bug.

    A no-op when text is None/empty (e.g. a tool-result follow-up turn
    with no real new user speech) -- never overwrites a real prior value
    with nothing.
    """
    if not text:
        return call
    old = call.structured_state or {}
    new_state = dict(old)
    new_state["last_seen_raw_user_message"] = text
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_direct_address_reply(db: AsyncSession, call: IncidentCall) -> IncidentCall:
    """
    Stamps when a direct-address reply was last actually spoken --
    iCall_utils._direct_address_off_cooldown reads this to stop two STT
    fragments of the same utterance (confirmed live, incident-38: "Watcher."
    / "you tell me the current status?") from each independently answering.
    """
    old = call.structured_state or {}
    new_state = dict(old)
    new_state["last_direct_address_reply_at"] = datetime.now(timezone.utc).isoformat()
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_silence_streak(db: AsyncSession, call: IncidentCall, streak: int) -> IncidentCall:
    """
    Tracks consecutive silence-trigger turns with no real speech in
    between -- iCall_api._process_turn uses this to escalate from a brief
    nudge (1st) to an actual status recap (2nd) to staying silent (3rd+),
    instead of repeating the identical content-free nudge forever.
    Confirmed live (incident-38): the same nudge line fired 10 times
    verbatim over 5.5 minutes with nothing else said -- not "spoken status
    summaries at appropriate moments," just a loop. Reset to 0 by
    apply_structuring_update on any real turn.
    """
    old = call.structured_state or {}
    new_state = dict(old)
    new_state["silence_streak"] = streak
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_wrapped_up(db: AsyncSession, call: IncidentCall) -> IncidentCall:
    """
    Marks that the call has already spoken its wrap-up recap --
    iCall_api._process_turn's silence-trigger branch checks this to stay
    silent afterward instead of continuing to nudge/recap into a call
    that already said its goodbyes. Confirmed live (incident-39): without
    this, a silence_check and then a silence_recap both fired 43s and 77s
    after the closing line, into dead air.
    """
    old = call.structured_state or {}
    new_state = dict(old)
    new_state["wrapped_up"] = True
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_language_switch_pending(
    db: AsyncSession, call: IncidentCall, target_code: str
) -> IncidentCall:
    """
    Marks that a spoken language-switch trigger was detected and a handoff
    was fired at the voice-agent server -- iCall_api._process_turn checks
    this so the frontend's language-status poll can show a "switching..."
    banner instead of the handoff (which has real, measured latency) just
    looking like the agent went silent. Cleared by
    record_language_switch_applied once the voice-agent server confirms
    the new agent is up.
    """
    old = call.structured_state or {}
    new_state = dict(old)
    new_state["language_switch_pending"] = True
    new_state["language_switch_target"] = target_code
    new_state["language_switch_started_at"] = datetime.now(timezone.utc).isoformat()
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_language_switch_applied(
    db: AsyncSession, call: IncidentCall, new_code: str
) -> IncidentCall:
    """
    Called once the voice-agent server confirms the handoff completed --
    updates the durable language_code (read fresh by the next agent start)
    and clears the transient pending flags record_language_switch_pending
    set.
    """
    old = call.structured_state or {}
    new_state = dict(old)
    new_state.pop("language_switch_pending", None)
    new_state.pop("language_switch_target", None)
    new_state.pop("language_switch_started_at", None)
    call.structured_state = new_state
    call.language_code = new_code
    await db.commit()
    await db.refresh(call)
    return call


async def record_language_switch_confirmation_pending(
    db: AsyncSession, call: IncidentCall, target_code: str
) -> IncidentCall:
    """
    Stage 3 of the language-switch detector (see iCall_utils.
    detect_language_switch_trigger): a language was named but no clear
    switch-verb accompanied it, so instead of switching outright or
    silently dropping it, the agent asks and this flag records what it's
    waiting for. Checked (and its expiry enforced) by iCall_api._process_
    turn against LANGUAGE_CONFIRMATION_TIMEOUT_S before trusting a later
    "yes" -- same rebuild-fresh-dict discipline as every other record_*
    function here.
    """
    old = call.structured_state or {}
    new_state = dict(old)
    new_state["pending_language_confirmation"] = {
        "target": target_code,
        "asked_at": datetime.now(timezone.utc).isoformat(),
    }
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def clear_language_switch_confirmation_pending(
    db: AsyncSession, call: IncidentCall
) -> IncidentCall:
    """
    Clears the Stage-3 confirmation flag once it's been resolved (an
    affirmative/negative reply was seen) or has expired -- see
    record_language_switch_confirmation_pending.
    """
    old = call.structured_state or {}
    if "pending_language_confirmation" not in old:
        return call
    new_state = dict(old)
    new_state.pop("pending_language_confirmation", None)
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_pattern_nudge(
    db: AsyncSession, call: IncidentCall, pattern: str, message: str, score: Optional[int] = None
) -> IncidentCall:
    """
    Logs a CEP-pattern-triggered nudge into the timeline -- unlike an
    ordinary spoken_reply (never persisted, see apply_structuring_update's
    docstring), this content didn't come from the model reacting to what
    was just said, it came from code noticing a pattern across everything
    recorded so far. Worth keeping in the auditable record specifically
    because "why did the AI say that" should be answerable after the
    fact, the same explainability the whole pattern-watching design is
    for.

    score, when given (the "health_score_drop" pattern only), is stashed
    as last_health_score_drop_nudge_score so detect_health_score_drop can
    tell "already nudged for this drop" from "it's gotten worse since" --
    without this, that check fired repeatedly on consecutive turns with
    nothing new to report (confirmed live, incident-26).

    Every call also stamps last_pattern_nudge_at[pattern] with now, which
    is what evaluate_call_patterns._pattern_off_cooldown reads -- without
    this, its window-threshold checks have zero memory of "did I just say
    this" and re-fire identically on every subsequent turn (confirmed
    live, incident-101: confusion_cluster x9, conflict_pileup x4 in ~20s).
    """
    old = call.structured_state or {}
    timeline = list(old.get("timeline", []))
    now = datetime.now(timezone.utc)
    timeline.append({
        "type": "pattern_nudge",
        "text": f"[{pattern}] {message}",
        "timestamp": now.isoformat(),
    })

    new_state = dict(old)
    new_state["timeline"] = timeline
    last_pattern_nudge_at = dict(old.get("last_pattern_nudge_at", {}))
    last_pattern_nudge_at[pattern] = now.isoformat()
    new_state["last_pattern_nudge_at"] = last_pattern_nudge_at
    if score is not None:
        new_state["last_health_score_drop_nudge_score"] = score
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def record_shared_screen(db: AsyncSession, call: IncidentCall, screen: dict) -> IncidentCall:
    """
    Appends one shared-screen event (GitHub commits or server logs, see
    iCall_utils.broadcast_shared_screen) to a running list -- kept here
    independent of the live RTM broadcast succeeding or failing, so a late
    joiner (or anyone whose broadcast message got lost) can still catch up
    via GET /icall/channel/{channel_name}/recap, same append-only,
    never-mutate-in-place discipline as record_health_score/
    record_pattern_nudge.
    """
    old = call.structured_state or {}
    screens = list(old.get("shared_screens", []))
    screens.append(screen)

    new_state = dict(old)
    new_state["shared_screens"] = screens
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def update_call_status(db: AsyncSession, call: IncidentCall, status: str) -> IncidentCall:
    call.status = status
    await db.commit()
    await db.refresh(call)
    return call


DELEGATE_REVIEW_AWAITING = "awaiting_reply"


async def start_delegate_review(db: AsyncSession, call: IncidentCall, draft: str) -> IncidentCall:
    """
    Stores the reviewed draft (see iOrchestrate_utils.notify_delegate_review_needed)
    separately from call.structured_state's own recorded fields -- shallow-copied
    dict, so this key survives every later apply_structuring_update call
    exactly like related_incident_note does, without that function needing
    to know anything about it.
    """
    state = dict(call.structured_state or {})
    state["delegate_review_draft"] = draft
    call.structured_state = state
    call.delegate_review_status = DELEGATE_REVIEW_AWAITING
    await db.commit()
    await db.refresh(call)
    return call


async def update_delegate_review_draft(db: AsyncSession, call: IncidentCall, draft: str) -> IncidentCall:
    state = dict(call.structured_state or {})
    state["delegate_review_draft"] = draft
    call.structured_state = state
    await db.commit()
    await db.refresh(call)
    return call


async def clear_delegate_review(db: AsyncSession, call: IncidentCall) -> IncidentCall:
    call.delegate_review_status = None
    await db.commit()
    await db.refresh(call)
    return call


async def find_awaiting_delegate_review_call(db: AsyncSession, slack_user_id: str) -> Optional[IncidentCall]:
    """
    Correlates an incoming Slack DM (which carries only a Slack user id, no
    action_id/value the way a button click does) back to the call it's
    about. Scoped to calls actually awaiting a reply -- small by
    construction (one per delegate incident mid-review at a time), so a
    linear scan + re-resolving each incident's approver is simple and
    plenty fast at this scale; no need for a dedicated index on the join.
    """
    from app.iNcidents.iNcidents_crudl import get_incident
    from app.iDirectory.iDirectory_crudl import resolve_approver

    result = await db.execute(
        select(IncidentCall).where(IncidentCall.delegate_review_status == DELEGATE_REVIEW_AWAITING)
    )
    for call in result.scalars().all():
        incident = await get_incident(db, call.incident_id)
        if not incident:
            continue
        approver = await resolve_approver(db, incident.service)
        if approver and approver.slack_user_id == slack_user_id:
            return call
    return None


async def record_utterance(
    db: AsyncSession, call_id: int, event: CallUtteranceCreate
) -> CallUtterance:
    """
    A duplicate post of a turn_index already recorded for this call isn't
    always a new utterance -- it's the same race as get_or_create_call
    below and TeamService.set_owner (iDirectory_crudl): the browser's own
    de-dupe (postedTurnText, a useRef) isn't atomic across two independent
    mounts of the same effect, so two concurrent posts of identical text
    can both pass it. The unique constraint on (call_id, turn_index) is
    what actually enforces exclusivity there; this turns the loser's
    IntegrityError into "return what's already there" instead of a 500.

    But a second post for the same turn_index isn't always a pure race --
    confirmed live (incident-25): the transcript source can revise a
    turn's text after it's first posted (the live UI showed a full
    sentence while the stored row was stuck on the first, incomplete
    fragment). When the incoming text actually differs from what's
    stored, update the row instead of leaving it stale -- the client only
    sends a second post for a given turn_index when its own tracked text
    for that key has changed, so this isn't reachable from ordinary
    duplicate/race posts of identical text.
    """
    utterance = CallUtterance(call_id=call_id, **event.model_dump())
    db.add(utterance)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        result = await db.execute(
            select(CallUtterance).where(
                CallUtterance.call_id == call_id,
                CallUtterance.turn_index == event.turn_index,
            )
        )
        existing = result.scalar_one()
        if event.text != existing.text:
            existing.text = event.text
            existing.timestamp = event.timestamp
            await db.commit()
            await db.refresh(existing)
        return existing
    await db.refresh(utterance)
    return utterance


async def list_utterances(db: AsyncSession, call_id: int) -> List[CallUtterance]:
    result = await db.execute(
        select(CallUtterance)
        .where(CallUtterance.call_id == call_id)
        .order_by(CallUtterance.turn_index.asc())
    )
    return list(result.scalars().all())


async def record_agent_utterance(
    db: AsyncSession, call_id: int, text: str, reason: str
) -> AgentUtterance:
    """
    Persists one line the agent actually spoke. Call this only when
    spoken_reply ended up non-empty -- a turn the gate decided to stay
    silent on has nothing to record. See AgentUtterance's docstring for
    why this is a separate table from CallUtterance rather than a
    special speaker_uid on it.
    """
    utterance = AgentUtterance(
        call_id=call_id,
        text=text,
        reason=reason,
        timestamp=datetime.now(timezone.utc),
    )
    db.add(utterance)
    await db.commit()
    await db.refresh(utterance)
    return utterance


async def list_agent_utterances(db: AsyncSession, call_id: int) -> List[AgentUtterance]:
    result = await db.execute(
        select(AgentUtterance)
        .where(AgentUtterance.call_id == call_id)
        .order_by(AgentUtterance.timestamp.asc())
    )
    return list(result.scalars().all())


async def infer_and_store_participant_roles(db: AsyncSession, call: IncidentCall) -> IncidentCall:
    """
    Runs once, when a call completes. Hybrid role resolution: a known
    employee's iDirectory title is authoritative -- a real, deterministic
    fact beats a conversation-derived guess every time -- and only
    participants iDirectory has no answer for fall back to the LLM's
    best-effort read of everything they said across the whole call.
    """
    utterances = await list_utterances(db, call.id)
    if not utterances:
        return call

    speaker_texts: Dict[str, List[str]] = defaultdict(list)
    speaker_names: Dict[str, str] = {}
    for u in utterances:
        speaker_texts[u.speaker_uid].append(u.text)
        if u.speaker_name:
            speaker_names[u.speaker_uid] = u.speaker_name  # last known name wins

    classification = await classify_participant_roles(dict(speaker_texts), speaker_names)
    inferred_by_uid = {s.speaker_uid: s for s in classification.speakers}

    participant_roles: Dict[str, dict] = {}
    for uid in speaker_texts:
        name = speaker_names.get(uid)
        inferred = inferred_by_uid.get(uid)
        employee = await find_employee_by_name(db, name) if name else None
        directory_title = employee.title if employee else None

        entry = {
            "name": name,
            "directory_matched": employee is not None,
            "directory_title": directory_title,
            "directory_org_role": employee.role if employee else None,
            "directory_team_id": employee.team_id if employee else None,
            "inferred_role": inferred.best_role if inferred else None,
            "inferred_scores": [s.model_dump() for s in inferred.scores] if inferred else [],
            "inferred_rationale": inferred.rationale if inferred else None,
        }
        if directory_title:
            entry["final_role"] = directory_title
            entry["source"] = "directory"
        elif inferred:
            entry["final_role"] = inferred.best_role
            entry["source"] = "inferred"
        else:
            entry["final_role"] = None
            entry["source"] = "unknown"

        participant_roles[uid] = entry

    call.participant_roles = participant_roles
    await db.commit()
    await db.refresh(call)

    await _reconcile_action_item_owners(db, call, participant_roles)
    await _add_unresolved_risks_summary(db, call)
    return call


async def _add_unresolved_risks_summary(db: AsyncSession, call: IncidentCall) -> None:
    """
    One-shot, end-of-call: "A final incident summary with unresolved
    risks" from the brief. Runs after role inference and owner
    reconciliation so it's synthesizing over the most complete state
    available, not an earlier snapshot.
    """
    state = call.structured_state or {}
    if not state:
        return

    summary = await summarize_unresolved_risks(state)
    if not summary.risks:
        return

    new_state = dict(state)
    new_state["unresolved_risks"] = summary.risks
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)


async def _reconcile_action_item_owners(
    db: AsyncSession, call: IncidentCall, participant_roles: Dict[str, dict]
) -> None:
    """
    Two-pass, best-effort ownership resolution -- this is the actual point
    of role inference, not just a side note: knowing who's on the call in
    what capacity only matters if it decides who owns each follow-up.

    Pass 1 (name match): if the live extraction already named an owner
    (e.g. "David will check the pool"), resolve that name against the
    roster of people actually confirmed on this call.

    Pass 2 (role match): for whatever's left -- most items, typically,
    since a live call rarely has someone named as owner for every item --
    one more Gemini call matches each item's actual content against the
    roster's roles. Never invents a person outside the roster (validated
    against participant_roles below, not just trusted from the model);
    leaves an item unassigned rather than forcing a guess when no role
    fits.

    Rebuilds fresh dict/list objects throughout (never mutates the
    existing structured_state nested objects in place) -- an in-place
    mutation here would leave call.structured_state pointing at the same
    object it started with, and SQLAlchemy would see old == new and
    silently skip the UPDATE, the exact pitfall apply_structuring_update
    already documents.
    """
    state = call.structured_state or {}
    action_items = state.get("action_items") or []
    if not action_items:
        return

    roster_by_lower_name = {
        entry["name"].strip().lower(): (uid, entry)
        for uid, entry in participant_roles.items()
        if entry.get("name")
    }

    new_items = [dict(item) for item in action_items]
    changed = False

    # Pass 1: explicit name match
    for new_item in new_items:
        owner_text = (new_item.get("owner") or "").strip()
        if owner_text and not new_item.get("owner_uid"):
            match = roster_by_lower_name.get(owner_text.lower())
            if match:
                uid, entry = match
                new_item["owner_uid"] = uid
                new_item["owner_role"] = entry.get("final_role")
                new_item["owner_source"] = "name_match"
                # Normalize the display text to the roster's canonical name too --
                # otherwise a stale/raw extraction (e.g. an STT mishearing) stays
                # in owner even though owner_uid now correctly identifies someone
                # else entirely, which reads as a bug in any UI that shows owner.
                if entry.get("name"):
                    new_item["owner"] = entry["name"]
                changed = True

    # Pass 2: role match for whatever's still unassigned
    unresolved_indices = [i for i, item in enumerate(new_items) if not item.get("owner_uid")]
    roster_with_roles = [
        {"uid": uid, "name": entry["name"], "role": entry["final_role"]}
        for uid, entry in participant_roles.items()
        if entry.get("name") and entry.get("final_role")
    ]

    if unresolved_indices and roster_with_roles:
        assignments = await assign_action_item_owners(
            [new_items[i].get("text", "") for i in unresolved_indices],
            roster_with_roles,
        )
        assignment_by_local_index = {a.item_index: a for a in assignments.assignments}
        for local_index, global_index in enumerate(unresolved_indices):
            assignment = assignment_by_local_index.get(local_index)
            if not assignment or not assignment.owner_uid:
                continue
            # Never trust the model's uid blindly -- only apply it if it's
            # actually someone on this call's confirmed roster.
            roster_entry = participant_roles.get(assignment.owner_uid)
            if not roster_entry:
                continue
            new_items[global_index]["owner_uid"] = assignment.owner_uid
            new_items[global_index]["owner_role"] = roster_entry.get("final_role")
            new_items[global_index]["owner_source"] = "role_match"
            new_items[global_index]["owner_rationale"] = assignment.rationale
            # Same normalization as Pass 1 -- role-match still resolves to a
            # real roster person, so owner should say who that actually is.
            if roster_entry.get("name"):
                new_items[global_index]["owner"] = roster_entry["name"]
            changed = True

    if changed:
        new_state = dict(state)
        new_state["action_items"] = new_items
        call.structured_state = new_state
        await db.commit()
        await db.refresh(call)
