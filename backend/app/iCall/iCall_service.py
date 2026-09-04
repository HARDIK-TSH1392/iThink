import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, List, Optional

from app.iNcidents.iNcidents_crudl import get_incident
from app.iDirectory.iDirectory_crudl import find_employee_by_name

from .iCall_model import IncidentCall, CallUtterance
from .iCall_schema import CallUtteranceCreate, StructuringUpdate
from .iCall_utils import (
    generate_channel_name,
    classify_participant_roles,
    assign_action_item_owners,
    summarize_unresolved_risks,
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


class IncidentNotFoundError(Exception):
    pass


async def get_call_by_incident(db: AsyncSession, incident_id: int) -> Optional[IncidentCall]:
    result = await db.execute(
        select(IncidentCall).where(IncidentCall.incident_id == incident_id)
    )
    return result.scalar_one_or_none()


async def get_or_create_call(db: AsyncSession, incident_id: int) -> IncidentCall:
    """
    The one place a channel name is decided. Orchestration (calendar/email
    invite) and the voice agent (RTC join) both call this instead of
    independently computing a channel name, so there is exactly one source
    of truth per incident rather than two formulas that could drift apart.

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

        call = IncidentCall(
            incident_id=incident_id,
            channel_name=generate_channel_name(incident_id),
            status=CALL_STATUS_SCHEDULED,
            structured_state={},
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
    })

    now = datetime.now(timezone.utc).isoformat()

    def _add_timeline_entry(entry_type: str, text: str) -> None:
        state["timeline"].append({"type": entry_type, "text": text, "timestamp": now})

    state["facts"].extend(update.facts)
    for fact in update.facts:
        _add_timeline_entry("fact", fact)

    state["hypotheses"].extend(update.hypotheses)
    for hypothesis in update.hypotheses:
        _add_timeline_entry("hypothesis", hypothesis)

    state["decisions"].extend(update.decisions)
    for decision in update.decisions:
        _add_timeline_entry("decision", decision)

    state["action_items"].extend(item.model_dump() for item in update.action_items)
    for item in update.action_items:
        _add_timeline_entry("action_item", item.text)

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


async def record_pattern_nudge(db: AsyncSession, call: IncidentCall, pattern: str, message: str) -> IncidentCall:
    """
    Logs a CEP-pattern-triggered nudge into the timeline -- unlike an
    ordinary spoken_reply (never persisted, see apply_structuring_update's
    docstring), this content didn't come from the model reacting to what
    was just said, it came from code noticing a pattern across everything
    recorded so far. Worth keeping in the auditable record specifically
    because "why did the AI say that" should be answerable after the
    fact, the same explainability the whole pattern-watching design is
    for.
    """
    old = call.structured_state or {}
    timeline = list(old.get("timeline", []))
    timeline.append({
        "type": "pattern_nudge",
        "text": f"[{pattern}] {message}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    new_state = dict(old)
    new_state["timeline"] = timeline
    call.structured_state = new_state
    await db.commit()
    await db.refresh(call)
    return call


async def update_call_status(db: AsyncSession, call: IncidentCall, status: str) -> IncidentCall:
    call.status = status
    await db.commit()
    await db.refresh(call)
    return call


async def record_utterance(
    db: AsyncSession, call_id: int, event: CallUtteranceCreate
) -> CallUtterance:
    utterance = CallUtterance(call_id=call_id, **event.model_dump())
    db.add(utterance)
    await db.commit()
    await db.refresh(utterance)
    return utterance


async def list_utterances(db: AsyncSession, call_id: int) -> List[CallUtterance]:
    result = await db.execute(
        select(CallUtterance)
        .where(CallUtterance.call_id == call_id)
        .order_by(CallUtterance.turn_index.asc())
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
            changed = True

    if changed:
        new_state = dict(state)
        new_state["action_items"] = new_items
        call.structured_state = new_state
        await db.commit()
        await db.refresh(call)
