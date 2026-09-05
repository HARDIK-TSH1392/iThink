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
from mcp.server.mcpserver import MCPServer

ITHINK_BACKEND_BASE_URL = os.getenv("ITHINK_BACKEND_BASE_URL", "http://127.0.0.1:8123/api/v1")

app = MCPServer(
    name="ilogs",
    instructions="Query recent server log entries recorded for this incident's service.",
)


@app.tool()
async def get_recent_logs(
    service: str,
    region: str = "",
    since_minutes: int = 60,
    severity: str = "",
    limit: int = 20,
) -> list[dict]:
    """
    Get recent server log entries for a service, optionally filtered by
    region and minimum severity, from the last `since_minutes` minutes.
    Returns each entry's severity, message, timestamp, and source_id,
    newest first.
    """
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
