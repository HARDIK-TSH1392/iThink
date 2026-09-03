import hashlib
import hmac
import time
import httpx
from datetime import datetime
from typing import Optional

from app.config import get_settings


def format_call_summary(structured_state: dict) -> str:
    """
    Deterministic formatting of a call's structured_state into a readable
    brief -- no extra LLM call here. The data (facts/hypotheses/decisions/
    action_items/conflicts) was already extracted live during the call by
    iCall; this just renders it, same "don't let formatting re-invent
    facts" discipline as everything upstream of it.
    """
    state = structured_state or {}
    lines = []

    def _section(title: str, items: list, empty: str = "None recorded"):
        lines.append(f"*{title}:*")
        if items:
            for item in items:
                lines.append(f"  • {item}")
        else:
            lines.append(f"  {empty}")

    _section("Facts", state.get("facts", []))
    _section("Hypotheses", state.get("hypotheses", []), empty="None")
    _section("Decisions", state.get("decisions", []), empty="None recorded")

    action_items = state.get("action_items", [])
    lines.append("*Action Items:*")
    if action_items:
        for item in action_items:
            owner = item.get("owner") or "unassigned"
            lines.append(f"  • {item.get('text', '')} (owner: {owner})")
    else:
        lines.append("  None recorded")

    conflicts = state.get("conflicts", [])
    if conflicts:
        _section("Unresolved conflicts/questions", conflicts)

    return "\n".join(lines)


async def post_call_summary_notification(incident, call) -> bool:
    """
    Posts the post-call brief to the channel once a call is marked
    completed, with a "Create Jira Ticket" / "Skip" action -- the second
    human-approval gate before any Jira action, mirroring the same
    Approve/Reject button mechanism already built for incident approval.
    """
    summary = format_call_summary(call.structured_state)
    text = (
        f"*Post-call summary: Incident #{incident.id}* :memo:\n"
        f"*{incident.title}*  |  `{incident.service}` in `{incident.region}`\n\n"
        f"{summary}"
    )

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "block_id": f"jira_decision_{call.id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📋 Create Jira Ticket"},
                    "style": "primary",
                    "action_id": "create_jira_ticket",
                    "value": str(call.id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Skip"},
                    "action_id": "skip_jira_ticket",
                    "value": str(call.id),
                },
            ],
        },
    ]

    ok = await _post_to_slack(text, blocks=blocks)
    if ok:
        print(f"[iOrchestrate] Post-call summary posted for incident {incident.id} / call {call.id}")
    return ok


def _summary_to_adf(summary_text: str) -> dict:
    """
    Jira Cloud's REST API v3 requires descriptions in Atlassian Document
    Format, not plain text/markdown. Renders each line as its own
    paragraph -- simplest valid ADF that preserves the structure.
    """
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            for line in summary_text.split("\n")
            if line.strip()
        ],
    }


async def create_jira_ticket(incident_id: int, title: str, summary_text: str) -> Optional[str]:
    """
    Creates a Jira issue from the call's captured discussion. Returns the
    created issue's browse URL, or None on failure (logged, never raised --
    same fire-and-forget-from-the-caller discipline as the Slack posts).
    """
    settings = get_settings()
    if not all([settings.jira_site_url, settings.jira_email, settings.jira_api_token, settings.jira_project_key]):
        print("[iOrchestrate] Jira not configured, skipping ticket creation")
        return None

    payload = {
        "fields": {
            "project": {"key": settings.jira_project_key},
            "summary": f"Incident #{incident_id}: {title}",
            "description": _summary_to_adf(summary_text),
            "issuetype": {"name": "Task"},
        }
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{settings.jira_site_url.rstrip('/')}/rest/api/3/issue",
                auth=(settings.jira_email, settings.jira_api_token),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            issue_key = data["key"]
            url = f"{settings.jira_site_url.rstrip('/')}/browse/{issue_key}"
            print(f"[iOrchestrate] Jira ticket {issue_key} created for incident {incident_id}")
            return url
    except Exception as exc:
        print(f"[iOrchestrate] Jira ticket creation failed for incident {incident_id}: {exc}")
        return None


async def _post_to_slack(text: str, blocks: Optional[list] = None) -> bool:
    settings = get_settings()
    if not settings.slack_webhook_url:
        print("[iOrchestrate] SLACK_WEBHOOK_URL not set, skipping notification")
        return False

    payload = {"text": text}
    if blocks:
        payload["blocks"] = blocks

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(settings.slack_webhook_url, json=payload)
            response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[iOrchestrate] Slack post failed: {exc}")
        return False


async def _post_dm_to_slack_user(user_id: str, text: str, blocks: Optional[list] = None) -> bool:
    """
    DMs a specific Slack user via the Bot Token (chat.postMessage) rather
    than posting to the shared channel -- approval requests are private to
    the resolved approver; the channel only sees the outcome, posted
    separately once a decision is made.
    """
    settings = get_settings()
    if not settings.slack_bot_token:
        print("[iOrchestrate] SLACK_BOT_TOKEN not set, cannot DM -- falling back to channel post")
        return False

    payload = {"channel": user_id, "text": text}
    if blocks:
        payload["blocks"] = blocks

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {settings.slack_bot_token}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("ok"):
                print(f"[iOrchestrate] Slack DM API error: {data.get('error')}")
                return False
        return True
    except Exception as exc:
        print(f"[iOrchestrate] Slack DM failed: {exc}")
        return False


async def post_approval_request_notification(
    incident_id: int,
    title: str,
    summary: Optional[str],
    service: str,
    region: str,
    priority: Optional[str],
    detected_at: datetime,
    approver_name: Optional[str],
    approver_email: Optional[str],
    approver_slack_user_id: Optional[str] = None,
) -> bool:
    """
    Notify that an incident needs approval, as soon as it reaches
    awaiting_approval -- the "someone has to know to check" gap. DMs the
    resolved approver directly if their Slack ID is on file; otherwise
    falls back to posting in the channel (still flagging who should act,
    just not privately) -- no owning team or nobody available also falls
    back to the channel, flagged loudly rather than notifying nobody.

    Includes real Approve/Reject buttons (Block Kit) -- clicking one hits
    /iorchestrate/slack/interact, which is verified, applies the decision
    through the same iNcidents logic the API uses, and updates this
    message in place.
    """
    who = f"{approver_name} ({approver_email})" if approver_name else (
        "⚠️ no available approver found — please assign manually"
    )

    # title and summary are often the identical string (iTriage sets both to
    # the same incident_summary) -- only show one copy of that text.
    headline = summary if summary else title

    text = (
        f"*Approval needed: Incident #{incident_id}* :rotating_light:\n"
        f"{headline}\n"
        f"When: {detected_at.isoformat()}  |  Where: `{service}` in `{region}`  |  Priority: *{priority or 'unset'}*\n"
        f"Approver: {who}"
    )

    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "block_id": f"incident_decision_{incident_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Approve"},
                    "style": "primary",
                    "action_id": "approve_incident",
                    "value": str(incident_id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Reject"},
                    "style": "danger",
                    "action_id": "reject_incident",
                    "value": str(incident_id),
                },
            ],
        },
    ]

    if approver_slack_user_id:
        ok = await _post_dm_to_slack_user(approver_slack_user_id, text, blocks=blocks)
        if ok:
            print(f"[iOrchestrate] Approval-request DM sent to {approver_slack_user_id} for incident {incident_id}")
            return True
        print(f"[iOrchestrate] DM failed for incident {incident_id}, falling back to channel post")

    ok = await _post_to_slack(text, blocks=blocks)
    if ok:
        print(f"[iOrchestrate] Approval-request notification posted to channel for incident {incident_id}")
    return ok


async def notify_approval_needed(db, incident) -> bool:
    """
    Single entry point for "an incident just reached awaiting_approval" --
    resolves the approver via iDirectory and posts the notification. Called
    from both the manual status-update endpoint and iTriage's automatic
    flow, so the notification never depends on which path got the incident
    there.
    """
    from app.iDirectory.iDirectory_crudl import resolve_approver

    approver = await resolve_approver(db, incident.service)
    return await post_approval_request_notification(
        incident_id=incident.id,
        title=incident.title,
        summary=incident.summary,
        service=incident.service,
        region=incident.region,
        priority=incident.priority,
        detected_at=incident.created_at,
        approver_name=approver.name if approver else None,
        approver_email=approver.email if approver else None,
        approver_slack_user_id=approver.slack_user_id if approver else None,
    )


async def post_incident_approved_notification(
    incident_id: int,
    title: str,
    priority: str,
    service: str,
    region: str,
    approved_by: str,
    responders: Optional[list] = None,
) -> bool:
    """
    Minimal orchestration action: post a Slack message when an incident is
    approved. Fire-and-forget from the caller's perspective -- failures are
    logged, never raised, since a Slack outage should not block the
    approval flow itself.

    `responders` is a list of Employee-like objects (needs .name and
    .slack_user_id) -- listed by name; anyone with a real Slack ID on file
    gets <@mentioned> too, which pings them if they're already a channel
    member. Nobody is actually invited to the channel here -- that needs a
    Slack scope we deliberately skipped for now (low payoff tonight, since
    only real accounts can be invited at all).
    """
    text = (
        f"*Incident #{incident_id} approved* :rotating_light:\n"
        f"*{title}*\n"
        f"Priority: *{priority}*  |  Service: `{service}`  |  Region: `{region}`\n"
        f"Approved by: {approved_by}"
    )

    if responders:
        names = [
            f"<@{r.slack_user_id}> ({r.name})" if r.slack_user_id else r.name
            for r in responders
        ]
        text += f"\nResponders needed: {', '.join(names)}"

    ok = await _post_to_slack(text)
    if ok:
        print(f"[iOrchestrate] Slack notification sent for incident {incident_id}")
    return ok


async def notify_incident_approved(db, incident) -> bool:
    """
    Single entry point for "an incident was just approved" -- resolves
    responders via iDirectory and posts the announcement. Mirrors
    notify_approval_needed's shape so both approval paths (the manual API
    and the Slack button) call one function instead of duplicating the
    resolve+notify logic.
    """
    from app.iDirectory.iDirectory_crudl import resolve_responders

    responders = await resolve_responders(db, incident.service)
    return await post_incident_approved_notification(
        incident_id=incident.id,
        title=incident.title,
        priority=incident.priority or "unset",
        service=incident.service,
        region=incident.region,
        approved_by=incident.approved_by,
        responders=responders,
    )


# -----------------------------------------------------------------------------
# Slack interactivity: verifying button clicks, updating the message in place
# -----------------------------------------------------------------------------

def verify_slack_signature(raw_body: bytes, timestamp: str, signature: str, secret: str) -> bool:
    """
    Slack's request-signing scheme: v0=HMAC-SHA256("v0:{timestamp}:{body}", secret).
    Rejects requests older than 5 minutes (replay protection).
    """
    try:
        if abs(time.time() - float(timestamp)) > 60 * 5:
            return False
    except (TypeError, ValueError):
        return False

    basestring = f"v0:{timestamp}:".encode() + raw_body
    computed = "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature or "")


async def update_slack_message(response_url: str, text: str) -> bool:
    """
    Replace the original interactive message's content via the response_url
    Slack includes in every interactivity payload -- no bot token needed,
    this URL alone is authorized to update that specific message.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                response_url,
                json={"replace_original": True, "text": text, "blocks": []},
            )
            response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[iOrchestrate] Failed to update Slack message: {exc}")
        return False
