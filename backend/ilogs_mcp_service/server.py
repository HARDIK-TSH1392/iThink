# -*- coding: utf-8 -*-
"""
iLogs MCP server.

Isolated for the same reason as backend/github_mcp_service/: the `mcp`
SDK's server-side pulls in a starlette version too new for this repo's
pinned FastAPI backend. This service doesn't touch the database directly
either -- it's a thin MCP wrapper around the main backend's existing
GET /ilogs/ REST endpoint, so there's exactly one source of truth for how
logs are actually queried/filtered.

Exposes get_recent_logs, get_incident_status, and list_action_items for
Agora's Conversational AI agent to call natively (see
voice-agent/server/src/agent.py's mcp_servers config) when someone on a
call asks about logs, the incident's own status, or who owns what -- the
model decides when to call each, we don't gate on keywords here.
"""
import os

from dotenv import load_dotenv

load_dotenv()

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

ITHINK_BACKEND_BASE_URL = os.getenv("ITHINK_BACKEND_BASE_URL", "http://127.0.0.1:8123/api/v1")
# This service is exposed to the public internet (Agora's cloud calls it
# directly, see voice-agent/server/agent.py's mcp_servers registration) --
# TransportSecuritySettings' Host/Origin allowlisting only guards against
# DNS-rebinding-style browser attacks, not a direct client that already
# has the tunnel URL (its request naturally carries the matching Host
# header). A shared secret is the actual access control: unset means the
# check is skipped entirely, matching dev-without-a-tunnel usage.
ILOGS_SHARED_SECRET = os.getenv("ILOGS_SHARED_SECRET")

app = MCPServer(
    name="ilogs",
    instructions="Query recent server log entries, incident status, and action items for this call.",
)


def _check_shared_secret(ctx: Context) -> None:
    if not ILOGS_SHARED_SECRET:
        return
    headers = ctx.headers or {}
    provided = headers.get("x-ilogs-shared-secret")
    if provided != ILOGS_SHARED_SECRET:
        raise ToolError("Unauthorized: missing or incorrect shared secret.")


async def require_incident_approved(channel_name: str) -> dict:
    """
    Gate for any FUTURE write-capable tool (e.g. a PagerDuty page, a
    GitHub write action) -- raises ToolError unless this channel's
    incident has actually been approved. Neither tool in this file today
    takes a real action (get_recent_logs/get_incident_status/
    list_action_items/search_similar_incidents are all read-only, and
    GitHub's own MCP tools are scoped read-only too -- see agent.py's
    allowed_tools), so nothing calls this yet. It exists so the next
    write-capable tool has a ready, tested "no action without human
    approval" check to call at its very first line, instead of that
    discipline living only in a prompt instruction the model could ignore
    or a human could forget to add.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        call_response = await client.get(f"{ITHINK_BACKEND_BASE_URL}/icall/channel/{channel_name}")
        if call_response.status_code == 404:
            raise ToolError(f"No call found for channel '{channel_name}'.")
        call_response.raise_for_status()
        call = call_response.json()

        incident_response = await client.get(f"{ITHINK_BACKEND_BASE_URL}/incidents/{call['incident_id']}")
        incident_response.raise_for_status()
        incident = incident_response.json()

    if not incident.get("approved_by"):
        raise ToolError(
            f"Incident #{incident['id']} has not been approved yet -- this action requires "
            "human approval first."
        )
    return incident


@app.tool()
async def get_recent_logs(
    service: str,
    ctx: Context,
    region: str = "",
    since_minutes: int = 1440,
    severity: str = "",
    limit: int = 20,
) -> list[dict]:
    """
    Get recent server log entries for a service, optionally filtered by
    region and minimum severity, from the last `since_minutes` minutes.
    Returns each entry's severity, message, timestamp, and source_id,
    newest first.

    Default widened from 60 to 1440 (24h): confirmed live that the model
    calling this tool has no reliable way to know how long ago an incident
    actually started (logs leading up to it typically predate when it was
    officially detected/created), so a tight default silently returned "no
    log entries found" for a real, ongoing incident purely because more
    than an hour of real time had passed since the log was recorded --
    not a data problem, just too narrow a default window for an incident-
    investigation tool. The caller can still pass a smaller value
    explicitly when it actually wants a tight recent window.
    """
    _check_shared_secret(ctx)

    from datetime import datetime, timedelta, timezone

    since = (datetime.now(timezone.utc) - timedelta(minutes=since_minutes)).isoformat()
    params: dict = {"service": service, "since": since, "limit": limit}
    if region:
        params["region"] = region
    if severity:
        params["severity"] = severity

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{ITHINK_BACKEND_BASE_URL}/ilogs/", params=params)
        response.raise_for_status()
        logs = response.json()

    return [
        {
            "severity": log["severity"],
            "message": log["message"],
            "timestamp": log["timestamp"],
            "source_id": log["source_id"],
        }
        for log in logs
    ]


@app.tool()
async def get_incident_status(channel_name: str, ctx: Context) -> dict:
    """
    Get this call's incident-level status: triage/mitigation state,
    priority, whether it's been approved, and who approved it. Use the
    exact channel_name given in your own prompt context, never a guessed
    or numeric id.

    Backed by the same two REST endpoints the dashboard itself uses
    (GET /icall/channel/{channel_name} then GET /incidents/{incident_id})
    -- no separate status tracking invented for this tool.
    """
    _check_shared_secret(ctx)

    async with httpx.AsyncClient(timeout=10) as client:
        call_response = await client.get(f"{ITHINK_BACKEND_BASE_URL}/icall/channel/{channel_name}")
        if call_response.status_code == 404:
            raise ToolError(f"No call found for channel '{channel_name}'.")
        call_response.raise_for_status()
        call = call_response.json()

        incident_response = await client.get(f"{ITHINK_BACKEND_BASE_URL}/incidents/{call['incident_id']}")
        incident_response.raise_for_status()
        incident = incident_response.json()

    return {
        "call_status": call["status"],
        "incident_status": incident["status"],
        "priority": incident["priority"],
        "service": incident["service"],
        "region": incident["region"],
        "approved_by": incident["approved_by"],
        "resolved_at": incident["resolved_at"],
    }


@app.tool()
async def list_action_items(channel_name: str, ctx: Context) -> list[dict]:
    """
    List this call's recorded action items with their text and resolved
    owner (name, role, and how confidently that owner was matched). Use
    the exact channel_name given in your own prompt context.

    There's no separate "done"/"open" tracking on an action item today --
    this returns everything recorded for the call so far, not a filtered
    subset, since inventing a completion status this tool can't actually
    verify would be worse than not claiming one at all.
    """
    _check_shared_secret(ctx)

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{ITHINK_BACKEND_BASE_URL}/icall/channel/{channel_name}")
        if response.status_code == 404:
            raise ToolError(f"No call found for channel '{channel_name}'.")
        response.raise_for_status()
        call = response.json()

    return call.get("structured_state", {}).get("action_items", [])


@app.tool()
async def search_similar_incidents(query: str, ctx: Context, service: str = "", limit: int = 5) -> list[dict]:
    """
    Search past CLOSED incidents by keyword against their title/summary,
    optionally narrowed to one service. Use when someone asks "has this
    happened before" or similar -- not a general incident list.

    Generalizes the existing one-incident "related incident" note (see
    backend's iCall_utils.format_related_incident_note, auto-injected at
    call start for the same service/region) into something the model can
    actively query across more incidents and services. Same discipline as
    that note, deliberately: returns only title/summary/service/region/
    priority/resolved_at, never the past incident's own facts or
    decisions -- pulling those in verbatim risks treating an unrelated
    past cause as this incident's own confirmed fact. Mention a result
    only if it's actually relevant to what's being discussed, never
    assume the same root cause without evidence from this call.

    Filters on "closed", not the STATUS_RESOLVED constant that note uses
    -- confirmed live that no code path ever actually sets an incident's
    status to "resolved" (only "closed" is ever assigned), so filtering
    on it the same way that note does would silently return nothing.
    """
    _check_shared_secret(ctx)

    params: dict = {"status": "closed", "limit": 100}
    if service:
        params["service"] = service

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{ITHINK_BACKEND_BASE_URL}/incidents/", params=params)
        response.raise_for_status()
        incidents = response.json()

    keywords = [w for w in query.lower().split() if len(w) > 2]

    def score(incident: dict) -> int:
        text = f"{incident.get('title', '')} {incident.get('summary') or ''}".lower()
        return sum(1 for kw in keywords if kw in text)

    scored = sorted(incidents, key=score, reverse=True)
    matched = [i for i in scored if score(i) > 0][:limit]
    if not matched:
        # No keyword hit -- fall back to the most recent resolved
        # incidents (already newest-first from the backend) rather than
        # returning nothing, same "something useful beats a false
        # negative" reasoning as get_recent_logs' generous default window.
        matched = incidents[:limit]

    return [
        {
            "title": i["title"],
            "summary": i.get("summary"),
            "service": i["service"],
            "region": i["region"],
            "priority": i["priority"],
            "resolved_at": i["resolved_at"],
        }
        for i in matched
    ]


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8004"))
    app.run(transport="streamable-http", host="0.0.0.0", port=port, streamable_http_path="/mcp")
