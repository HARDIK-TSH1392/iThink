# Watcher

A real-time AI incident commander for the EchoSphere hackathon (team We
thinkED). Joins a live voice incident call, keeps facts separate from
guesses, catches contradictions out loud, assigns and tracks who owns what,
and only takes real actions (a ticket, a Slack post) after a human says yes.

"Watcher" is the agent's name and the user-facing product name; `iThink` is
this repo/backend's original internal codename and still shows up in module
names, package names, and some URLs (`ithink_dev.db`, `ITHINK_BACKEND_BASE_URL`,
etc.) — that's intentional, not a leftover rebrand bug.

This README exists mainly to answer "what does each module actually do" —
the module names (`iLogs`, `iNcidents`, `iTriage`, `iCall`, `iDirectory`,
`iOrchestrate`) don't say that on their own.

## Module map

| Module | Plain-English role | Owns |
|---|---|---|
| `iLogs` | **The ears.** Receives a monitoring/log signal, scores how severe it looks. | Log ingestion, deterministic severity/region/service scoring |
| `iNcidents` | **The incident file + the approval stamp.** Nothing external happens until a human approves here. | Incident lifecycle state machine, the human-approval gate |
| `iTriage` | **The assistant that reads alerts first.** Auto-decides if a log is a real incident and files it for approval — no human has to notice the alert themselves. | Gemini-based triage, automatic incident creation, evidence correlation |
| `iCall` | **The meeting room + the AI notetaker in it.** Creates the voice channel for an approved incident and is the actual brain the voice agent talks to during the call. | Room/channel creation, the custom-LLM endpoint, live structuring (facts/hypotheses/decisions/conflicts/timeline), participant role recognition, two-pass task-ownership assignment, silence/pattern-based spoken nudges, the coordination health score, end-of-call unresolved-risk synthesis, the pre-Jira-ticket content review, the Agora webhook receiver (also the authoritative "call actually ended" signal, independent of the client) |
| `iDirectory` | **The org chart.** Who's on which team, which team owns which service, who's actually available right now. | Teams, employees (with a fixed job-title vocabulary shared with `iCall`'s role classifier), `resolve_approver` / `resolve_responders` |
| `iOrchestrate` | **The messenger.** Turns an incident/call event into a real Slack message or Jira ticket — nothing here happens without a prior human approval upstream, and neither approval gate ever exposes an actionable button in a shared channel (DM-only, or a non-actionable notice if the approver can't be reached privately). | Slack notify-on-approval (with real Approve/Reject buttons), DM-to-approver, post-call summary, the Jira approval gate + ticket creation |

`voice-agent/` (repo root, alongside `backend/`) is **Agora's own official
ConvoAI quickstart**, not code we wrote — see its own section below.

## How a request actually flows, end to end

1. A signal arrives → `POST /api/v1/ilogs/ingest` (**iLogs**).
2. `iTriage` runs as a background task, decides if it's a real incident, and
   if so creates one via `iNcidents`, landing in `awaiting_approval` —
   **iOrchestrate** immediately DMs the resolved approver (via `iDirectory`)
   with real Approve/Reject buttons, falling back to the shared channel if
   nobody's resolvable.
3. A human approves (via the API, or the Slack button itself) →
   `iNcidents` records the decision, **iOrchestrate** announces it and
   @mentions the resolved responders.
4. `POST /api/v1/icall/incidents/{id}/call` (**iCall**) creates (or fetches)
   that incident's room — this is the *only* place the channel name is
   decided; nothing else should compute its own.
5. A human and the Agora voice agent join that channel. Every turn, Agora
   calls `POST /api/v1/icall/channel/{channel_name}/llm/chat/completions`
   (**iCall**) for what to say next — this is where facts/hypotheses/
   decisions/conflicts actually get extracted and saved. The model can also
   call GitHub/log-lookup tools mid-turn (see "Native MCP tool-calling"
   below); the result comes back as a follow-up request on the same
   endpoint before anything gets spoken.
6. Agora also posts session events to `POST /api/v1/icall/webhooks/agora`
   (**iCall**) — in particular, "agent left the channel" is a second,
   authoritative trigger for marking the call `completed`, independent of
   the client's own status update (which a crashed tab or dropped
   connection would otherwise skip entirely).
7. When the call is marked `completed`, role inference and task-ownership
   assignment run once over the full transcript, **iOrchestrate** posts the
   structured summary to Slack, and privately DMs the approver asking
   whether to open a Jira ticket from what was captured — a second,
   separate human-confirmation gate before anything reaches Jira. The
   ticket's content goes through one more LLM-reviewed cleanup pass first
   (see `iCall_utils.review_ticket_content`) to strip call-logistics noise
   and transcription-garbled phrasing before an external ticket is created
   from it — the Slack summary itself stays verbatim/unedited.

## Native MCP tool-calling during a live call

Agora's Conversational AI Engine can call MCP tools directly and forward the
result back to `iCall`'s custom-LLM endpoint — this needs
`advanced_features.enable_tools=True` on the agent (silently a no-op
otherwise) and is configured in `voice-agent/server/src/agent.py`'s
`mcp_servers` list:

- **GitHub** (`GITHUB_TOKEN` in `voice-agent/server/.env`) — registers
  GitHub's own official remote MCP server (`api.githubcopilot.com/mcp/`)
  directly; the model calls it live during a call to answer questions about
  commits/deploys.
- **iLogs** (`ILOGS_MCP_URL` in `voice-agent/server/.env`) — a thin MCP
  wrapper this repo owns (`backend/ilogs_mcp_service/`) around the backend's
  own `GET /ilogs/`, so the model can pull recent server logs for the
  incident's service the same way.

Both need a **publicly reachable HTTPS endpoint** — Agora's cloud calls
them directly, not this local machine — so local dev needs a tunnel
(`cloudflared tunnel --url http://localhost:8004`, etc.) pointed at
whichever of these you're running locally. `iCall`'s structuring prompt
tells the model the incident's actual service/region (and linked GitHub
repo, if any) explicitly rather than expecting it to infer them from
conversation — a service nobody happened to say by name out loud on the
call previously returned nothing from either tool.

Separately, an older, narrower GitHub integration still exists:
`backend/github_mcp_service/` (port 8003) is a deterministic,
keyword-triggered ("did we deploy/ship/commit recently?") lookup used by
`iCall_utils._check_recent_deploys` — a fixed `SERVICE_TO_GITHUB_REPO`
mapping, not the model deciding when to call it. Both this and the native
GitHub MCP tool above can be enabled at once; they don't conflict.

## Setup — backend

```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in at least GEMINI_API_KEY
./.venv/bin/uvicorn app.main:app --reload --port 8123
```

API docs (Swagger, try-it-out for every endpoint): `http://127.0.0.1:8123/docs`

`iOrchestrate` needs its own env vars to do anything beyond logging a skip —
`SLACK_WEBHOOK_URL`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` for Slack;
`JIRA_SITE_URL` / `JIRA_EMAIL` / `JIRA_API_TOKEN` / `JIRA_PROJECT_KEY` for
Jira. All optional — everything degrades to a logged skip, not a crash,
when unset.

A few more, also optional/degrade-safely: `AGORA_WEBHOOK_SECRET` verifies
the Agora webhook above once you've registered one in Agora Console
(unverified-but-accepted with a loud warning otherwise); `AGORA_APP_ID` +
`AGORA_CUSTOMER_KEY` / `AGORA_CUSTOMER_SECRET` (a *different* credential
pair from the voice agent's App ID/Certificate — Console → RESTful API) are
needed to broadcast a GitHub/logs lookup live to everyone already on the
call, not just whoever asked; without them the data still gets recorded,
late joiners still catch up via `/recap`, it just doesn't push live.
`GITHUB_MCP_SERVICE_URL` / `VOICE_AGENT_SERVER_URL` point at the two other
local services this backend calls (see below). See `backend/.env.example`
for the full list.

## Setup — voice agent (`voice-agent/`)

This started from **Agora's official Python ConvoAI quickstart** (`agora
init`) — see its own `README.md` for the full upstream instructions/project
layout. It's grown well past the original two-edit customization since:
custom-LLM wiring, native MCP tool registration, `silence_config`/
`interruption` tuning, and Deepgram STT locale/keyterm tuning all live in
`voice-agent/server/src/agent.py`; the always-visible MCP-results tile,
raise-hand, mute indicators, and join/hand-raise chimes live under
`voice-agent/web/src/components/`. `voice-agent/AGENTS.md` is the actual
up-to-date contributor guide for this directory.

To run it:

```bash
cd voice-agent
bun run setup   # writes server/.env from your Agora CLI project binding
bun run dev     # starts both the Python agent backend (:8000) and the web client (:3000)
```

Needs `AGORA_APP_ID` / `AGORA_APP_CERTIFICATE` in `voice-agent/server/.env`
(written automatically by `bun run setup` if you've run `agora login` +
`agora init`/`agora project use` already) and, for the agent to actually
reach our backend, `ITHINK_BACKEND_BASE_URL` pointed at wherever the
`backend/` service above is running (defaults to `http://127.0.0.1:8123/api/v1`).

For the native MCP tools (see above) to actually register with Agora, also
set `GITHUB_TOKEN` and/or `ILOGS_MCP_URL` in this same `.env` — both
optional, each tool is simply not registered when its var is unset, no
crash either way. `ILOGS_MCP_URL` needs `backend/ilogs_mcp_service/`
running (`python3 server.py`, separate process, own `.venv` — see its
module docstring for why it's isolated) and tunneled publicly.

**Important:** the web client only produces a working voice agent when
joined via a real `?channel=incident-N` link tied to an actual `IncidentCall`
row (see the demo dashboard below) — opening `localhost:3000` bare generates
a random room with no backing incident, which will 404 against `iCall`'s
custom-LLM endpoint. That's expected, not a bug.

## Demo control panel

`demo/dashboard.html` — plain HTML/JS, no build step, open it directly in a
browser. Walks the flow in one screen: simulate a signal → see it triaged →
approve → create the room → get a real link to join it. Requires the
backend running on `127.0.0.1:8123` (edit the `API_BASE` constant at the top
of the file if it's running elsewhere) and the voice agent's web client
running on `localhost:3000` for the join link to actually work.

## What's real vs. not, right now

Being direct about this so nobody assumes more is built than actually is:

**Working and verified live** (not just written — actually exercised
against a real call/incident at least once): all six modules; live
structuring during a call (facts vs. hypotheses vs. decisions, missing-info
vs. conflict detection, a spoken callout — not a silent state edit — when a
later turn corrects an earlier fact); participant role recognition
(directory-authoritative, LLM-inferred fallback) and two-pass task-ownership
assignment (name match, then role match against the call's actual roster —
never invents a person outside it); a real timeline with silence-triggered
and pattern-based spoken nudges (a room gone quiet, a pileup of open
conflicts, a stale unowned action item), gated by a deterministic
coordination-health score, not the model self-regulating; native Agora MCP
tool-calling for live GitHub/log lookups mid-call, with the incident's
service/region/repo told to the model explicitly rather than left to
conversation, shown live in an always-visible grid tile with a full,
scrollable history (not just the latest lookup); a live-updating,
in-call timeline/recap panel rendering `structured_state` itself, not just
a Slack summary after the fact; a raise-hand + mute indicator on every
participant's tile;
Slack notify-on-approval and the Jira approval gate, both DM-only to the
resolved approver — neither ever exposes an actionable button in a shared
channel, even as a fallback; an LLM-reviewed cleanup pass on a call's
content before it becomes a Jira ticket (strips logistics noise, merges
transcription-garbled duplicate hypotheses, never invents a resolution to a
question the call never actually answered); an Agora webhook as a second,
authoritative "call ended" signal independent of the client.

**Real, currently-open gaps:**
- **PagerDuty isn't actually integrated.** It exists only as a Deepgram STT
  keyterm (a word to transcribe correctly), zero real API calls anywhere.
  PagerDuty has an official remote MCP server (`mcp.pagerduty.com/mcp`,
  API-key auth) that would close this the same way the GitHub/iLogs native
  MCP tools work today — not yet wired up.
- **"Monitoring systems" integration is simulated, not real.** `iLogs` is
  this repo's own DB-backed log store/ingestion API, not a connection to an
  actual external tool (Datadog/Grafana/CloudWatch/etc.).
- **No automated test suite.** Every fix described in this repo's commit
  history was verified with a one-off script run once, not a committed
  regression test — nothing currently catches a future change silently
  breaking, say, the ownership two-pass or the speak-gate.
- **Nothing here is actually deployed.** Local processes + `cloudflared`
  quick tunnels for the backend and both MCP services; every tunnel URL is
  ephemeral and needs re-registering (Agora Console's webhook config
  included) whenever the process restarts.
- Calendar/email invites for a call; follow-up meeting scheduling after one.

## Conventions, if you're adding a module

- One folder per module: `backend/app/i<Name>/`, files
  `i<Name>_model.py` / `_schema.py` / `_crudl.py` (or `_service.py` for a
  pipeline-shaped module rather than a plain resource) / `_utils.py` /
  `_api.py`.
- Compute anything resembling a score or confidence level in code — never
  let the LLM self-report one. Keep any LLM-facing schema free of fields
  that look like a diagnosis or a recommended fix; this is a coordination
  tool, not an autonomous one.
- Register new routers in `app/main.py`.
- External-service calls (Slack, Jira, Gemini) fail loudly in logs but
  never raise into the caller — a Slack outage should never block an
  approval or a call from proceeding.
