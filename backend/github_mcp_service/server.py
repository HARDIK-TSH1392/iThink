# -*- coding: utf-8 -*-
"""
GitHub MCP bridge.

Isolated on purpose: the official `mcp` SDK's SSE transport depends on
sse-starlette>=0.49.1, which conflicts with the main backend's pinned
FastAPI (needs starlette<0.39.0). Rather than fight that in a shared venv,
this is its own tiny service with its own venv -- the main backend calls
it over a plain HTTP request, same as it calls Slack or Jira.

Not a general-purpose MCP client: exposes exactly one capability iCall
needs (recent commit activity for a service's repo), not a passthrough to
arbitrary GitHub MCP tools.
"""
import asyncio
import contextlib
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import httpx2

GITHUB_MCP_URL = os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="GitHub MCP bridge")

# A full MCP handshake against GitHub's remote server measured ~7-8s in
# testing -- too slow to pay on every call during a live incident call.
# Kept as one long-lived session behind this lock rather than reconnecting
# per-request; only the first call (or a call after a dropped connection)
# pays the handshake cost.
_session: ClientSession | None = None
_session_stack: contextlib.AsyncExitStack | None = None
_session_lock = asyncio.Lock()


async def _get_session() -> ClientSession:
    global _session, _session_stack
    if _session is not None:
        return _session

    async with _session_lock:
        if _session is not None:  # someone else won the race while we waited
            return _session

        stack = contextlib.AsyncExitStack()
        http_client = await stack.enter_async_context(
            httpx2.AsyncClient(headers={"Authorization": f"Bearer {GITHUB_TOKEN}"}, timeout=15)
        )
        read_stream, write_stream = await stack.enter_async_context(
            streamable_http_client(GITHUB_MCP_URL, http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        _session_stack = stack
        _session = session
        logger.info("GitHub MCP session established")
        return _session


async def _reset_session() -> None:
    global _session, _session_stack
    stack, _session_stack = _session_stack, None
    _session = None
    if stack is not None:
        with contextlib.suppress(Exception):
            await stack.aclose()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _reset_session()


class RecentDeploysRequest(BaseModel):
    repo: str  # "owner/name"
    limit: int = 5


class RecentDeploysResponse(BaseModel):
    configured: bool
    summary: str
    commits: list[dict]


@app.get("/health")
async def health():
    return {"status": "ok", "configured": bool(GITHUB_TOKEN)}


@app.post("/check-recent-deploys", response_model=RecentDeploysResponse)
async def check_recent_deploys(payload: RecentDeploysRequest) -> RecentDeploysResponse:
    """
    Calls GitHub's official remote MCP server's list_commits tool for the
    given repo. Degrades to configured=False (not an error) when no token
    is set -- same "prove the wiring, don't crash the call" discipline as
    everything else in iCall when a dependency isn't configured yet.
    """
    if not GITHUB_TOKEN:
        return RecentDeploysResponse(
            configured=False,
            summary="GitHub integration not configured (no GITHUB_TOKEN set).",
            commits=[],
        )

    tool_args = {
        "owner": payload.repo.split("/")[0],
        "repo": payload.repo.split("/")[1],
        "perPage": payload.limit,
    }

    try:
        session = await _get_session()
        try:
            result = await session.call_tool("list_commits", tool_args)
        except Exception:
            # The cached session may have gone stale (idle connection
            # dropped, etc) -- reset and retry exactly once with a fresh
            # one before giving up.
            logger.warning("GitHub MCP call failed on cached session, retrying fresh", exc_info=True)
            await _reset_session()
            session = await _get_session()
            result = await session.call_tool("list_commits", tool_args)

        commits: list[dict] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            for entry in parsed if isinstance(parsed, list) else [parsed]:
                commit_info = entry.get("commit", {})
                commits.append(
                    {
                        "sha": entry.get("sha", "")[:7],
                        "message": commit_info.get("message", ""),
                        "author": commit_info.get("author", {}).get("name", "unknown"),
                        "date": commit_info.get("author", {}).get("date", ""),
                    }
                )

        if not commits:
            summary = f"No recent commits found for {payload.repo}."
        else:
            lines = [
                f'- "{c["message"]}" by {c["author"]} at {c["date"]} ({c["sha"]})'
                for c in commits
            ]
            summary = f"Recent commits to {payload.repo}:\n" + "\n".join(lines)

        return RecentDeploysResponse(configured=True, summary=summary, commits=commits)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GitHub MCP call failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
