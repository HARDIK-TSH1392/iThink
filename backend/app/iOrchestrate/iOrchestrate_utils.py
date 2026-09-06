import hashlib
import hmac
import time
import httpx
from datetime import datetime
from typing import Optional

from app.config import get_settings


def format_call_summary(structured_state: dict, participant_roles: Optional[dict] = None) -> str:
    """
    Deterministic formatting of a call's structured_state into a readable
    brief -- no extra LLM call here. The data (facts/hypotheses/decisions/
    action_items/conflicts) was already extracted live during the call by
    iCall; this just renders it, same "don't let formatting re-invent
    facts" discipline as everything upstream of it.

    participant_roles is IncidentCall.participant_roles, a separate column
    from structured_state -- passed in explicitly rather than expected
    inside structured_state, to key action-item owners' names/roles.
    """
    state = structured_state or {}
    participant_roles = participant_roles or {}
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
            owner_uid = item.get("owner_uid")
            if owner_uid:
                owner_entry = participant_roles.get(owner_uid, {})
                owner_name = owner_entry.get("name") or item.get("owner") or "unassigned"
                owner_role = item.get("owner_role")
                owner = f"{owner_name} — {owner_role}" if owner_role else owner_name
                if item.get("owner_source") == "role_match":
                    owner += " (matched by role, not named)"
            else:
                owner = item.get("owner") or "unassigned"
            lines.append(f"  • {item.get('text', '')} (owner: {owner})")
    else:
        lines.append("  None recorded")

    _section("Missing information", state.get("missing_info", []), empty="None")

    conflicts = state.get("conflicts", [])
    if conflicts:
        _section("Unresolved conflicts/questions", conflicts)

    risks = state.get("unresolved_risks", [])
    if risks:
        _section("Unresolved risks", risks)

    return "\n".join(lines)


async def post_call_summary_notification(incident, call) -> bool:
    """
    Posts the post-call brief to the channel, purely informational -- no
    buttons here. The Jira approval ask is a separate, private DM to the
    resolved approver (see notify_jira_approval_needed), same "approval
    goes to the team lead, not the channel" rule as incident approval.
    """
    summary = format_call_summary(call.structured_state, call.participant_roles)
    text = (
        f"*Post-call summary: Incident #{incident.id}* :memo:\n"
        f"*{incident.title}*  |  `{incident.service}` in `{incident.region}`\n\n"
        f"{summary}"
    )
    ok = await _post_to_slack(text)
    if ok:
        print(f"[iOrchestrate] Post-call summary posted for incident {incident.id} / call {call.id}")
    return ok


def _build_jira_approval_blocks(call_id: int, text: str) -> list:
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {
            "type": "actions",
            "block_id": f"jira_decision_{call_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📋 Create Jira Ticket"},
                    "style": "primary",
                    "action_id": "create_jira_ticket",
                    "value": str(call_id),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Skip"},
                    "action_id": "skip_jira_ticket",
                    "value": str(call_id),
                },
            ],
        },
    ]


async def notify_jira_approval_needed(db, incident, call) -> bool:
    """
    Second human-approval gate: DMs the resolved approver privately asking
    whether to create a Jira ticket from this call's captured discussion.

    Deliberately does NOT fall back to posting the actionable Create-Ticket
    button in the incident channel the way the first approval gate does --
    unlike an incident's own approve/reject (which is scoped to a specific
    incident either way), anyone in the channel clicking "Create Jira
    Ticket" here would create a real ticket on someone else's behalf with
    no real approval having happened. If the approver can't be reached
    privately (no Slack ID on file, or the DM itself fails), this posts a
    plain, non-actionable notice instead -- the team still finds out
    something needs following up on, but only the resolved approver can
    actually trigger ticket creation, and only via their own DM.
    """
    from app.iDirectory.iDirectory_crudl import resolve_approver

    approver = await resolve_approver(db, incident.service)
    summary = format_call_summary(call.structured_state, call.participant_roles)
    text = (
        f"*Jira ticket approval needed: Incident #{incident.id}* :memo:\n"
        f"*{incident.title}*  |  `{incident.service}` in `{incident.region}`\n\n"
        f"{summary}"
    )

    if approver and approver.slack_user_id:
        blocks = _build_jira_approval_blocks(call.id, text)
        ok = await _post_dm_to_slack_user(approver.slack_user_id, text, blocks=blocks)
        if ok:
            print(f"[iOrchestrate] Jira-approval DM sent to {approver.slack_user_id} for call {call.id}")
            return True
        print(f"[iOrchestrate] Jira-approval DM failed for call {call.id} -- posting a non-actionable notice to the channel instead")
    else:
        print(f"[iOrchestrate] No Slack ID on file for call {call.id}'s resolved approver -- posting a non-actionable notice to the channel instead")

    who = f"{approver.name} ({approver.email})" if approver else "no available approver found -- please assign manually"
    fallback_text = (
        f"*Jira ticket approval needed: Incident #{incident.id}* :memo:\n"
        f"Couldn't reach the approver privately -- {who} should review this and create the ticket manually.\n"
        f"*{incident.title}*  |  `{incident.service}` in `{incident.region}`"
    )
    ok = await _post_to_slack(fallback_text)
    if ok:
        print(f"[iOrchestrate] Non-actionable Jira-approval notice posted to channel for call {call.id}")
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

    site_url = settings.jira_site_url.rstrip("/")
    auth = (settings.jira_email, settings.jira_api_token)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Team-managed ("next-gen") Jira projects reject issue creation
            # without an explicit reporter, even though the field metadata
            # claims hasDefaultValue=true -- confirmed live via a 400 with
            # no explanatory body beyond the status code. Resolved from the
            # authenticated token's own identity rather than hardcoded, so
            # this keeps working if JIRA_EMAIL ever points at a different
            # account.
            me_response = await client.get(f"{site_url}/rest/api/3/myself", auth=auth)
            me_response.raise_for_status()
            reporter_account_id = me_response.json()["accountId"]

            # Jira's summary field has a hard 255-char cap -- confirmed live
            # with a P1 incident title long enough to push "Incident #N: "
            # plus the AI-generated title past it. The full title still
            # reaches the ticket via the description, so truncating here
            # only shortens the headline, not the actual content.
            summary_field = f"Incident #{incident_id}: {title}"
            if len(summary_field) > 255:
                summary_field = summary_field[:254].rstrip() + "…"

            payload = {
                "fields": {
                    "project": {"key": settings.jira_project_key},
                    "summary": summary_field,
                    "description": _summary_to_adf(summary_text),
                    "issuetype": {"name": "Task"},
                    "reporter": {"id": reporter_account_id},
                }
            }
            response = await client.post(
                f"{site_url}/rest/api/3/issue",
                auth=auth,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            issue_key = data["key"]
            url = f"{site_url}/browse/{issue_key}"
            print(f"[iOrchestrate] Jira ticket {issue_key} created for incident {incident_id}")
            return url
    except httpx.HTTPStatusError as exc:
        # str(exc) alone is just the status line -- Jira's actual rejection
        # reason (e.g. errorMessages/errors) is in the response body, which
        # is what actually explains a 400 instead of just confirming one
        # happened.
        print(f"[iOrchestrate] Jira ticket creation failed for incident {incident_id}: {exc}\nResponse body: {exc.response.text}")
        return None
    except Exception as exc:
        print(f"[iOrchestrate] Jira ticket creation failed for incident {incident_id}: {exc}")
        return None


async def post_jira_ticket_created_notification(
    incident_id: int,
    title: str,
    priority: str,
    service: str,
    region: str,
    approved_by: str,
    url: str,
    summary_text: str,
) -> bool:
    """
    Every other outcome in this flow (incident approval, its result,
    post-call summary, even the Jira-approval ask when no private approver
    could be reached) gets echoed to the shared channel -- only the actual
    Jira-ticket-created confirmation didn't, and stayed stuck in the
    approver's own DM (via update_slack_message's response_url) with no
    one else ever seeing it. Mirrors post_incident_approved_notification's
    shape: the *ask* can be private, but the *outcome* is always shared.

    summary_text is the same rendered breakdown that went into the ticket's
    own Jira description (from format_call_summary) -- reused rather than
    recomputed, so the channel post always matches what's actually in the
    ticket, and matches the post-call summary message's level of detail.
    """
    text = (
        f"*Jira ticket created for Incident #{incident_id}* :clipboard:\n"
        f"{title}\n"
        f"Priority: {priority}  |  Service: {service}  |  Region: {region}\n"
        f"Created by: {approved_by}  |  {url}\n\n"
        f"{summary_text}"
    )
    ok = await _post_to_slack(text)
    if ok:
        print(f"[iOrchestrate] Jira-ticket-created notification posted to channel for incident {incident_id}")
    return ok


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
    resolved approver directly if their Slack ID is on file, with real
    Approve/Reject buttons (Block Kit) -- clicking one hits
    /iorchestrate/slack/interact, which is verified, applies the decision
    through the same iNcidents logic the API uses, and updates this
    message in place.

    Deliberately does NOT fall back to posting those buttons in the
    incident channel -- same reasoning as notify_jira_approval_needed's
    identical gate: slack_interactivity_endpoint verifies the request came
    from Slack, but never checks that the clicking user IS the resolved
    approver, so an actionable button visible to the whole channel means
    anyone in it can approve or reject a real incident. When the approver
    can't be reached privately (no Slack ID on file, or the DM itself
    fails), this posts a plain, non-actionable notice instead -- the team
    still finds out an incident needs approval, but only the resolved
    approver's own DM (or a direct API call) can actually decide it.
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
        print(f"[iOrchestrate] DM failed for incident {incident_id} -- posting a non-actionable notice to the channel instead")
    else:
        print(f"[iOrchestrate] No Slack ID on file for incident {incident_id}'s resolved approver -- posting a non-actionable notice to the channel instead")

    fallback_text = (
        f"*Approval needed: Incident #{incident_id}* :rotating_light:\n"
        f"Couldn't reach the approver privately -- {who} should review this and approve/reject manually.\n"
        f"{headline}\n"
        f"When: {detected_at.isoformat()}  |  Where: `{service}` in `{region}`  |  Priority: *{priority or 'unset'}*"
    )
    ok = await _post_to_slack(fallback_text)
    if ok:
        print(f"[iOrchestrate] Non-actionable approval-request notice posted to channel for incident {incident_id}")
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
    join_url: Optional[str] = None,
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

    `join_url` is the voice-agent web client link for this incident's
    channel -- whoever opens it first and clicks "Start Conversation" also
    starts the AI agent for everyone in the room (joining the room and
    starting the agent are independent Agora operations; this UI just
    triggers both from one click). Optional and omitted from the message
    entirely when room creation failed, rather than posting a broken link
    -- see notify_incident_approved, which is the only caller.
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

    if join_url:
        text += f"\nJoin the incident call: {join_url}"

    ok = await _post_to_slack(text)
    if ok:
        print(f"[iOrchestrate] Slack notification sent for incident {incident_id}")
    return ok


async def notify_incident_approved(db, incident) -> bool:
    """
    Single entry point for "an incident was just approved" -- resolves
    responders via iDirectory, ensures the call/channel exists, and posts
    the announcement with a join link. Mirrors notify_approval_needed's
    shape so both approval paths (the manual API and the Slack button)
    call one function instead of duplicating the resolve+notify logic.

    Room creation failing does not block the Slack announcement -- same
    fire-and-forget discipline as everything else here; the message just
    omits the link rather than the whole notification failing.
    """
    from app.iDirectory.iDirectory_crudl import resolve_responders
    from app.iCall.iCall_service import get_or_create_call

    responders = await resolve_responders(db, incident.service)

    join_url = None
    try:
        call = await get_or_create_call(db, incident.id)
        settings = get_settings()
        join_url = f"{settings.voice_agent_web_base_url.rstrip('/')}/?channel={call.channel_name}"
    except Exception as exc:
        print(f"[iOrchestrate] Room creation failed for incident {incident.id}: {exc}")

    return await post_incident_approved_notification(
        incident_id=incident.id,
        title=incident.title,
        priority=incident.priority or "unset",
        service=incident.service,
        region=incident.region,
        approved_by=incident.approved_by,
        responders=responders,
        join_url=join_url,
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
            # Slack's response_url endpoint can return HTTP 200 with a
            # non-"ok" body (e.g. an expired/already-used response_url) --
            # raise_for_status alone would treat that as success. Confirmed
            # worth checking after a ticket-creation success silently didn't
            # show up as updated in Slack with no error in the logs.
            if response.text.strip() != "ok":
                print(f"[iOrchestrate] Slack message update returned non-ok body: {response.text}")
                return False
        return True
    except Exception as exc:
        print(f"[iOrchestrate] Failed to update Slack message: {exc}")
        return False
