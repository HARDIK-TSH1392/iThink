import httpx

from app.config import get_settings


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
