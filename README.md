# iThink

A real-time AI incident commander for the EchoSphere hackathon. Joins a live
voice incident call, keeps facts separate from guesses, catches contradictions
out loud, and only takes real actions (a ticket, a Slack post) after a human
says yes.

This README exists mainly to answer "what does each module actually do" —
the module names (`iLogs`, `iNcidents`, `iTriage`, `iCall`) don't say that on
their own.

## Module map

| Module | Plain-English role | Owns |
|---|---|---|
| `iLogs` | **The ears.** Receives a monitoring/log signal, scores how severe it looks. | Log ingestion, deterministic severity/region/service scoring |
| `iNcidents` | **The incident file + the approval stamp.** Nothing external happens until a human approves here. | Incident lifecycle state machine, the human-approval gate |
| `iTriage` | **The assistant that reads alerts first.** Auto-decides if a log is a real incident and files it for approval — no human has to notice the alert themselves. | Gemini-based triage, automatic incident creation, evidence correlation |
| `iCall` | **The meeting room + the AI notetaker in it.** Creates the voice channel for an approved incident and is the actual brain the voice agent talks to during the call. | Room/channel creation, the custom-LLM endpoint, live structuring (facts/hypotheses/decisions/conflicts), the Agora webhook receiver |

**Not part of this repo:** the actual voice-calling web app is Agora's own
official ConvoAI quickstart, run separately (see `AGORA_INTEGRATION.md` if
present, or ask whoever set up the local voice agent). This repo is the
backend API surface; the quickstart is a thin client + agent process that
calls into `iCall`.

## How a request actually flows, end to end

1. A signal arrives → `POST /api/v1/ilogs/ingest` (**iLogs**).
2. `iTriage` runs as a background task, decides if it's a real incident, and
   if so creates one via `iNcidents`, landing in `awaiting_approval`.
3. A human approves it → `POST /api/v1/incidents/{id}/decision` (**iNcidents**).
4. `POST /api/v1/icall/incidents/{id}/call` (**iCall**) creates (or fetches)
   that incident's room — this is the *only* place the channel name is
   decided; nothing else should compute its own.
5. A human and the Agora voice agent join that channel. Every turn, Agora
   calls `POST /api/v1/icall/channel/{channel_name}/llm/chat/completions`
   (**iCall**) for what to say next — this is where facts/hypotheses/
   decisions/conflicts actually get extracted and saved.
6. Agora also posts session events to `POST /api/v1/icall/webhooks/agora`
   (**iCall**).

## Setup

```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY at minimum
./.venv/bin/uvicorn app.main:app --reload --port 8123
```

API docs (Swagger, try-it-out for every endpoint): `http://127.0.0.1:8123/docs`

## Demo control panel

`demo/dashboard.html` — plain HTML/JS, no build step, open it directly in a
browser. Walks the flow in one screen: simulate a signal → see it triaged →
approve → create the room → get a real link to join it. Requires the backend
running on `127.0.0.1:8123` (edit the `API_BASE` constant at the top of the
file if it's running elsewhere) and the Agora quickstart's web client running
on `localhost:3000` for the join link to actually work.

## What's real vs. not, right now

Being direct about this so nobody assumes more is built than actually is:

**Working and tested:** all four modules above, including live structuring
during a call (facts/hypotheses/decisions extraction, conflict detection) —
though the *extraction quality* itself hasn't been exercised against a real
Gemini call in this environment (no API key configured here; the fallback
path is what's verified end to end).

**Not built yet:**
- Orchestration — Slack channel creation, email/calendar invites, employee
  directory. A minimal Slack webhook post (on incident approval) is the
  smallest version of this and is next once a webhook URL is available.
- Post-call — the structured board, a second approval gate for Jira
  ticket creation, follow-up meeting scheduling, and the final
  incident summary with unresolved risks.
- A UI for `structured_state` (the facts/hypotheses/decisions data) — it's
  captured and persisted, nothing renders it yet.

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
