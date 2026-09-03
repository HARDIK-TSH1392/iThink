import json
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.database import async_session
from app.iNcidents.iNcidents_crudl import get_incident, record_approval_decision
from app.iNcidents.iNcidents_utils import STATUS_AWAITING_APPROVAL
from app.iCall.iCall_service import get_call

from .iOrchestrate_utils import (
    verify_slack_signature,
    update_slack_message,
    notify_incident_approved,
    format_call_summary,
    create_jira_ticket,
)

router = APIRouter(prefix="/iorchestrate", tags=["iOrchestrate"])


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

    decision = "approve" if action_id == "approve_incident" else "reject"
    incident = await record_approval_decision(db, incident, decision, approved_by=user_name)

    if decision == "approve":
        await update_slack_message(
            response_url,
            f"✅ *Incident #{incident_id} approved* by {user_name}. Status: `{incident.status}`.",
        )
        await notify_incident_approved(db, incident)
    else:
        await update_slack_message(response_url, f"❌ *Incident #{incident_id} rejected* by {user_name}.")


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

    summary_text = format_call_summary(call.structured_state)
    url = await create_jira_ticket(incident.id, incident.title, summary_text)

    if url:
        await update_slack_message(
            response_url,
            f"📋 *Jira ticket created* by {user_name}: {url}",
        )
    else:
        await update_slack_message(
            response_url,
            f"⚠️ Jira ticket creation failed for incident #{incident.id} (approved by {user_name}) — "
            f"check JIRA_* config / server logs.",
        )


@router.post("/slack/interact")
async def slack_interactivity_endpoint(request: Request):
    """
    Receives Slack's interactive-component payload for any button click --
    incident approve/reject, or the second Jira-creation gate. Verifies the
    request is genuinely from Slack, dispatches to the matching handler,
    and updates the original message in place via its response_url.
    """
    raw_body = await request.body()
    settings = get_settings()

    if settings.slack_signing_secret:
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")
        if not verify_slack_signature(raw_body, timestamp, signature, settings.slack_signing_secret):
            raise HTTPException(status_code=401, detail="Invalid Slack signature")
    else:
        print("[iOrchestrate] SLACK_SIGNING_SECRET not set — accepting unverified (dev only)")

    # Slack sends application/x-www-form-urlencoded with a `payload` field
    # containing the actual JSON.
    form = parse_qs(raw_body.decode("utf-8"))
    payload_raw = form.get("payload", [None])[0]
    if not payload_raw:
        raise HTTPException(status_code=422, detail="Missing Slack interaction payload")

    payload = json.loads(payload_raw)
    action = payload["actions"][0]
    action_id = action["action_id"]
    value = int(action["value"])
    response_url = payload["response_url"]
    user_name = payload.get("user", {}).get("username") or payload.get("user", {}).get("name", "unknown")

    async with async_session() as db:
        if action_id in ("approve_incident", "reject_incident"):
            await _handle_incident_decision(db, action_id, value, user_name, response_url)
        elif action_id in ("create_jira_ticket", "skip_jira_ticket"):
            await _handle_jira_decision(db, action_id, value, user_name, response_url)
        else:
            await update_slack_message(response_url, f"⚠️ Unknown action: {action_id}")

    return {"status": "ok"}
