# -*- coding: utf-8 -*-
"""
iLogs MCP server.

Isolated for the same reason as backend/github_mcp_service/: the `mcp`
SDK's server-side pulls in a starlette version too new for this repo's
pinned FastAPI backend. This service doesn't touch the database directly
either -- it's a thin MCP wrapper around the main backend's existing
GET /ilogs/ REST endpoint, so there's exactly one source of truth for how
logs are actually queried/filtered.

Exposes one tool, get_recent_logs, for Agora's Conversational AI agent to
call natively (see voice-agent/server/src/agent.py's mcp_servers config)
when someone on a call asks to see server logs -- the model decides when
to call it, we don't gate on keywords here.
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
    instructions="Query recent server log entries recorded for this incident's service.",
)


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
    if ILOGS_SHARED_SECRET:
        headers = ctx.headers or {}
        provided = headers.get("x-ilogs-shared-secret")
        if provided != ILOGS_SHARED_SECRET:
            raise ToolError("Unauthorized: missing or incorrect shared secret.")

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


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8004"))
    app.run(transport="streamable-http", host="0.0.0.0", port=port, streamable_http_path="/mcp")
