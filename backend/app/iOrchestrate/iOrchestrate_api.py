import json
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.database import async_session
from app.iNcidents.iNcidents_crudl import get_incident, record_approval_decision
from app.iNcidents.iNcidents_utils import STATUS_AWAITING_APPROVAL
from app.iCall.iCall_service import (
    get_call,
    find_awaiting_delegate_review_call,
    update_delegate_review_draft,
    clear_delegate_review,
)
from app.iCall.iCall_utils import review_ticket_content, parse_delegate_reply

from .iOrchestrate_utils import (
    verify_slack_signature,
    update_slack_message,
    notify_incident_approved,
    format_call_summary,
    create_jira_ticket,
    post_jira_ticket_created_notification,
    open_slack_delegate_modal,
    reply_to_delegate_dm,
)

router = APIRouter(prefix="/iorchestrate", tags=["iOrchestrate"])


async def _set_delegate_notes(db, incident_id: int, notes: str) -> None:
    incident = await get_incident(db, incident_id)
    if incident:
        incident.delegate_notes = notes
        await db.commit()


async def _handle_incident_decision(db, action_id: str, incident_id: int, user_name: str, response_url: str):
    incident = await get_incident(db, incident_id)
    if not incident:
        await update_slack_message(response_url, f"⚠️ Incident #{incident_id} not found.")
        return

    if incident.status != STATUS_AWAITING_APPROVAL:
        await update_slack_message(
            response_url,
            f"⚠️ Incident #{incident_id} is no longer awaiting approval "
            f"(current status: `{incident.status}`) — no action taken.",
        )
        return

    decision = "approve" if action_id in ("approve_join", "approve_incident") else "reject"
    incident = await record_approval_decision(db, incident, decision, approved_by=user_name)

    if decision == "approve":
        await update_slack_message(
            response_url,
            f"✅ *Incident #{incident_id} approved* by {user_name} — you'll join the call. "
            f"Status: `{incident.status}`.",
        )
        await notify_incident_approved(db, incident)
    else:
        await update_slack_message(response_url, f"❌ *Incident #{incident_id} rejected* by {user_name}.")


async def _handle_approve_delegate(db, incident_id: int, trigger_id: str, response_url: str):
    """
    Just opens the modal -- the incident is NOT approved yet. Approval and
    delegate_notes both get set together on modal submission (see
    _handle_delegate_modal_submission), so a lead who opens this and then
    cancels the modal hasn't approved anything, same as never clicking a
    button at all.
    """
    incident = await get_incident(db, incident_id)
    if not incident:
        await update_slack_message(response_url, f"⚠️ Incident #{incident_id} not found.")
        return
    if incident.status != STATUS_AWAITING_APPROVAL:
        await update_slack_message(
            response_url,
            f"⚠️ Incident #{incident_id} is no longer awaiting approval "
            f"(current status: `{incident.status}`) — no action taken.",
        )
        return
    opened = await open_slack_delegate_modal(trigger_id, incident_id)
    if not opened:
        await update_slack_message(
            response_url,
            f"⚠️ Couldn't open the delegate form for incident #{incident_id} — try again, "
            f"or use \"Approve — I'll join\" instead.",
        )


async def _handle_delegate_modal_submission(db, payload: dict) -> None:
    incident_id = int(payload["view"]["private_metadata"])
    values = payload["view"]["state"]["values"]
    notes = values["delegate_notes_block"]["delegate_notes_input"]["value"] or ""
    user_name = payload.get("user", {}).get("username") or payload.get("user", {}).get("name", "unknown")

    incident = await get_incident(db, incident_id)
    if not incident or incident.status != STATUS_AWAITING_APPROVAL:
        return

    await _set_delegate_notes(db, incident_id, notes)
    incident = await record_approval_decision(db, incident, "approve", approved_by=user_name)
    await notify_incident_approved(db, incident)


async def _handle_jira_decision(db, action_id: str, call_id: int, user_name: str, response_url: str):
    call = await get_call(db, call_id)
    if not call:
        await update_slack_message(response_url, f"⚠️ Call #{call_id} not found.")
        return

    if action_id == "skip_jira_ticket":
        await update_slack_message(response_url, f"Skipped Jira ticket creation for call #{call_id} (by {user_name}).")
        return

    incident = await get_incident(db, call.incident_id)
    if not incident:
        await update_slack_message(response_url, f"⚠️ Incident for call #{call_id} not found.")
        return

    # Confirmed live (incident-101, a solo test call): the room's raw
    # recorded state can carry test artifacts ("wait for the teammate to
    # join") and garbled/repeated restatements of the same guess -- fine
    # for a live coordination aid, bad for an external ticket someone else
    # has to act on. review_ticket_content is a read-only cleanup applied
    # only here, right before the ticket is built -- it never touches
    # call.structured_state itself, so the live call's own record (and
    # Slack's post-call summary, which intentionally stays raw/verbatim)
    # are unaffected. action_items/timeline etc. pass through untouched.
    reviewed = await review_ticket_content(call.structured_state or {})
    ticket_state = dict(call.structured_state or {})
    ticket_state.update(reviewed.model_dump())

    summary_text = format_call_summary(ticket_state, call.participant_roles)
    url = await create_jira_ticket(incident.id, incident.title, summary_text)

    if url:
        await update_slack_message(
            response_url,
            f"📋 *Jira ticket created* by {user_name}: {url}",
        )
        await post_jira_ticket_created_notification(
            incident.id, incident.title, incident.priority, incident.service, incident.region,
            user_name, url, summary_text,
        )
    else:
        await update_slack_message(
            response_url,
            f"⚠️ Jira ticket creation failed for incident #{incident.id} (approved by {user_name}) — "
            f"check JIRA_* config / server logs.",
        )


def _verify_slack_request(raw_body: bytes, request: Request) -> None:
    settings = get_settings()
    if settings.slack_signing_secret:
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")
        if not verify_slack_signature(raw_body, timestamp, signature, settings.slack_signing_secret):
            raise HTTPException(status_code=401, detail="Invalid Slack signature")
    else:
        print("[iOrchestrate] SLACK_SIGNING_SECRET not set — accepting unverified (dev only)")


@router.post("/slack/interact")
async def slack_interactivity_endpoint(request: Request):
    """
    Receives Slack's interactive-component payload for a button click
    (incident approve/reject/delegate, the Jira-creation gate) or a modal
    submission (the delegate-notes form). Verifies the request is
    genuinely from Slack, dispatches to the matching handler.

    Two distinct Slack payload shapes land here, dispatched on
    payload["type"]: "block_actions" (a button click -- carries
    payload["actions"], updates the original message via response_url) and
    "view_submission" (a modal's Submit button -- carries payload["view"],
    no response_url; Slack just wants a 200 to close the modal).
    """
    raw_body = await request.body()
    _verify_slack_request(raw_body, request)

    # Slack sends application/x-www-form-urlencoded with a `payload` field
    # containing the actual JSON.
    form = parse_qs(raw_body.decode("utf-8"))
    payload_raw = form.get("payload", [None])[0]
    if not payload_raw:
        raise HTTPException(status_code=422, detail="Missing Slack interaction payload")

    payload = json.loads(payload_raw)
    payload_type = payload.get("type")

    async with async_session() as db:
        if payload_type == "view_submission":
            if payload.get("view", {}).get("callback_id") == "delegate_notes_modal":
                await _handle_delegate_modal_submission(db, payload)
            # Empty 200 closes the modal (Slack's default when no
            # response_action is returned) -- no response_url exists for
            # this payload type.
            return {}

        action = payload["actions"][0]
        action_id = action["action_id"]
        value = int(action["value"])
        response_url = payload["response_url"]
        user_name = payload.get("user", {}).get("username") or payload.get("user", {}).get("name", "unknown")

        if action_id in ("approve_join", "approve_incident", "reject_incident"):
            await _handle_incident_decision(db, action_id, value, user_name, response_url)
        elif action_id == "approve_delegate":
            trigger_id = payload.get("trigger_id")
            if trigger_id:
                # Open the modal FIRST -- trigger_id expires in ~3s, and
                # this is the only synchronous, immediate thing this
                # handler does; everything else (recording the decision)
                # happens later, on modal submission.
                await _handle_approve_delegate(db, value, trigger_id, response_url)
            else:
                await update_slack_message(response_url, "⚠️ Missing trigger_id — try clicking again.")
        elif action_id in ("create_jira_ticket", "skip_jira_ticket"):
            await _handle_jira_decision(db, action_id, value, user_name, response_url)
        else:
            await update_slack_message(response_url, f"⚠️ Unknown action: {action_id}")

    return {"status": "ok"}


@router.post("/slack/events")
async def slack_events_endpoint(request: Request):
    """
    Slack's Events API -- the free-text half of the delegate-review loop
    (see notify_delegate_review_needed): the lead's DM reply carries no
    action_id/value the way a button click does, so this is a genuinely
    different Slack integration surface from /slack/interact's interactive
    components.

    Requires Event Subscriptions enabled in the Slack App config (api.slack.com
    -> your app -> Event Subscriptions), subscribed to `message.im`, with
    `im:history` added to Bot Token Scopes and the app reinstalled to pick
    up the new scope -- none of this is needed for /slack/interact, and
    isn't set up by anything in this codebase; it's an external dashboard
    step. Request URL verification (the "challenge" handshake below) is
    what that page checks when you first save this endpoint's public URL.
    """
    raw_body = await request.body()
    _verify_slack_request(raw_body, request)

    payload = json.loads(raw_body)

    # One-time handshake when this URL is first registered in Slack's
    # Event Subscriptions page -- echo the challenge back verbatim.
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge")}

    event = payload.get("event", {})
    # Only real human DMs: skip bot messages (including this app's own DM
    # replies, which would otherwise loop back into this same handler),
    # message edits/deletes (message.im also delivers subtype "message_changed"
    # etc.), and anything that isn't a plain DM.
    if (
        payload.get("type") != "event_callback"
        or event.get("type") != "message"
        or event.get("channel_type") != "im"
        or event.get("subtype")
        or event.get("bot_id")
        or not event.get("user")
        or not event.get("text")
    ):
        return {"status": "ok"}

    slack_user_id = event["user"]
    reply_text = event["text"]

    async with async_session() as db:
        call = await find_awaiting_delegate_review_call(db, slack_user_id)
        if call is None:
            # Not a reply we're expecting from this user right now -- most
            # DM traffic to this bot, if any, isn't part of an active
            # delegate review. Silently ignore rather than replying to
            # every stray message.
            return {"status": "ok"}

        current_draft = (call.structured_state or {}).get("delegate_review_draft", "")
        result = await parse_delegate_reply(current_draft, reply_text)

        if result.decision == "approve":
            incident = await get_incident(db, call.incident_id)
            url = await create_jira_ticket(incident.id, incident.title, current_draft) if incident else None
            await clear_delegate_review(db, call)
            if url:
                await reply_to_delegate_dm(slack_user_id, f"{result.acknowledgement}\n📋 {url}")
                if incident:
                    await post_jira_ticket_created_notification(
                        incident.id, incident.title, incident.priority, incident.service,
                        incident.region, "delegate review (Slack DM)", url, current_draft,
                    )
            else:
                await reply_to_delegate_dm(
                    slack_user_id,
                    "Approved, but ticket creation failed — check JIRA_* config or try again shortly.",
                )
        elif result.decision == "edit":
            await update_delegate_review_draft(db, call, result.updated_draft)
            await reply_to_delegate_dm(slack_user_id, result.acknowledgement)
        else:
            await reply_to_delegate_dm(slack_user_id, result.acknowledgement)

    return {"status": "ok"}
