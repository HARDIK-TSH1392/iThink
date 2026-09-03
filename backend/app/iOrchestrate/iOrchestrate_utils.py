import httpx
from datetime import datetime
from typing import Optional

from app.config import get_settings


async def _post_to_slack(text: str) -> bool:
    settings = get_settings()
    if not settings.slack_webhook_url:
        print("[iOrchestrate] SLACK_WEBHOOK_URL not set, skipping notification")
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(settings.slack_webhook_url, json={"text": text})
            response.raise_for_status()
        return True
    except Exception as exc:
        print(f"[iOrchestrate] Slack post failed: {exc}")
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
) -> bool:
    """
    Notify that an incident needs approval, as soon as it reaches
    awaiting_approval -- the "someone has to know to check" gap. If no
    approver could be resolved (no owning team, or nobody available), still
    posts -- flagging that loudly rather than silently notifying nobody.
    """
    who = f"{approver_name} ({approver_email})" if approver_name else (
        "⚠️ *no available approver found* — please assign manually"
    )

    # title and summary are often the identical string (iTriage sets both to
    # the same incident_summary) -- only show one copy of that text.
    headline = summary if summary else title

    text = (
        f"*Approval needed: Incident #{incident_id}* :rotating_light:\n"
        f"{headline}\n"
        f"When: {detected_at.isoformat()}  |  Where: `{service}` in `{region}`  |  Priority: *{priority or 'unset'}*\n"
        f"Approver: {who}\n"
        f"Approve or reject via the dashboard/API for incident #{incident_id}."
    )
    ok = await _post_to_slack(text)
    if ok:
        print(f"[iOrchestrate] Approval-request notification sent for incident {incident_id}")
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
    )


async def post_incident_approved_notification(
    incident_id: int,
    title: str,
    priority: str,
    service: str,
    region: str,
    approved_by: str,
) -> bool:
    """
    Minimal orchestration action: post a Slack message when an incident is
    approved. Fire-and-forget from the caller's perspective -- failures are
    logged, never raised, since a Slack outage should not block the
    approval flow itself.
    """
    settings = get_settings()
    if not settings.slack_webhook_url:
        print("[iOrchestrate] SLACK_WEBHOOK_URL not set, skipping notification")
        return False

    text = (
        f"*Incident #{incident_id} approved* :rotating_light:\n"
        f"*{title}*\n"
        f"Priority: *{priority}*  |  Service: `{service}`  |  Region: `{region}`\n"
        f"Approved by: {approved_by}"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(settings.slack_webhook_url, json={"text": text})
            response.raise_for_status()
        print(f"[iOrchestrate] Slack notification sent for incident {incident_id}")
        return True
    except Exception as exc:
        print(f"[iOrchestrate] Slack notification failed for incident {incident_id}: {exc}")
        return False
