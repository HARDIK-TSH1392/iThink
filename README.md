# iThink

A real-time AI incident commander for the EchoSphere hackathon. Joins a live
voice incident call, keeps facts separate from guesses, catches contradictions
out loud, and only takes real actions (a ticket, a Slack post) after a human
says yes.

This README exists mainly to answer "what does each module actually do" —
the module names (`iLogs`, `iNcidents`, `iTriage`, `iCall`, `iDirectory`,
`iOrchestrate`) don't say that on their own.

## Module map

| Module | Plain-English role | Owns |
|---|---|---|
| `iLogs` | **The ears.** Receives a monitoring/log signal, scores how severe it looks. | Log ingestion, deterministic severity/region/service scoring |
| `iNcidents` | **The incident file + the approval stamp.** Nothing external happens until a human approves here. | Incident lifecycle state machine, the human-approval gate |
| `iTriage` | **The assistant that reads alerts first.** Auto-decides if a log is a real incident and files it for approval — no human has to notice the alert themselves. | Gemini-based triage, automatic incident creation, evidence correlation |
| `iCall` | **The meeting room + the AI notetaker in it.** Creates the voice channel for an approved incident and is the actual brain the voice agent talks to during the call. | Room/channel creation, the custom-LLM endpoint, live structuring (facts/hypotheses/decisions/conflicts), the Agora webhook receiver |
| `iDirectory` | **The org chart.** Who's on which team, which team owns which service, who's actually available right now. | Teams, employees, `resolve_approver` / `resolve_responders` |
| `iOrchestrate` | **The messenger.** Turns an incident/call event into a real Slack message or Jira ticket — nothing here happens without a prior human approval upstream. | Slack notify-on-approval (with real Approve/Reject buttons), DM-to-approver, post-call summary, the Jira approval gate + ticket creation |

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
   decisions/conflicts actually get extracted and saved.
6. Agora also posts session events to `POST /api/v1/icall/webhooks/agora`
   (**iCall**).
7. When the call is marked `completed`, **iOrchestrate** posts the
   structured summary to Slack and privately DMs the approver asking
   whether to open a Jira ticket from what was captured — a second,
   separate human-confirmation gate before anything reaches Jira.

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
when unset. See `backend/.env.example` for the full list.

## Setup — voice agent (`voice-agent/`)

This is **Agora's official Python ConvoAI quickstart**, cloned via `agora
init`, not written by this team — see its own `README.md` for the full
upstream instructions. What's actually ours: two small edits, documented
inline in the code —

- `voice-agent/server/src/agent.py` — the LLM vendor points at `iCall`'s
  custom-LLM endpoint (`CustomLLM(base_url=...)`) instead of managed OpenAI.
- `voice-agent/web/src/components/LandingPage.tsx` (+ `web/app/page.tsx`) —
  accepts a specific room via a `?channel=` URL parameter, so a human can
  join *this incident's* room instead of a randomly generated one.

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

**Working and tested:** all six modules above, including live structuring
during a call (facts/hypotheses/decisions extraction, conflict detection),
Slack notify-on-approval with interactive buttons, and the Jira approval
gate — though extraction quality and the Slack/Jira integrations themselves
haven't been exercised against real Gemini/Slack/Jira credentials in every
environment; each degrades to a logged skip rather than crashing when a key
is missing, which is what's actually been verified end to end.

**Not built yet:**
- Calendar/email invites for a call.
- Follow-up meeting scheduling after a call.
- A UI for `structured_state` (the facts/hypotheses/decisions data) — it's
  captured, persisted, and summarized to Slack, but nothing renders it as a
  live board.

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
