from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select

from app.database import async_session
from app.iLogs.iLogs_model import ILog
from app.iLogs.iLogs_utils import compute_impact_score, infer_incident_priority
from app.iNcidents.iNcidents_model import Incident
from app.iNcidents.iNcidents_crudl import create_incident, list_incidents, link_log, update_status
from app.iNcidents.iNcidents_schema import IncidentCreate
from app.iNcidents.iNcidents_utils import (
    TERMINAL_STATUSES,
    STATUS_TRIAGE,
    STATUS_AWAITING_APPROVAL,
    STATUS_CLOSED,
    is_valid_transition,
)

from .iTriage_model import TriageResult
from .iTriage_utils import call_gemini_for_verdict, compute_deterministic_confidence

EVIDENCE_WINDOW_MINUTES = 60


async def _fetch_evidence(db, log: ILog) -> List[ILog]:
    since = log.timestamp - timedelta(minutes=EVIDENCE_WINDOW_MINUTES)
    result = await db.execute(
        select(ILog)
        .where(
            ILog.source_id == log.source_id,
            ILog.service == log.service,
            ILog.region == log.region,
            ILog.timestamp >= since,
            ILog.timestamp <= log.timestamp,
        )
        .order_by(ILog.timestamp.asc())
    )
    return list(result.scalars().all())


async def _find_open_incident(db, log: ILog) -> Optional[Incident]:
    candidates = await list_incidents(db, region=log.region, service=log.service, limit=500)
    for incident in candidates:
        if incident.source_id == log.source_id and incident.status not in TERMINAL_STATUSES:
            return incident
    return None


def _build_prompt(log: ILog, evidence: List[ILog], deterministic_priority: str, deterministic_impact_score: int) -> str:
    evidence_lines = "\n".join(
        f"- [log_id={e.id}] {e.timestamp.isoformat()} severity={e.severity} message={e.message!r}"
        for e in evidence
    )
    return f"""Triggering event:
- source_id: {log.source_id}
- service: {log.service}
- region: {log.region}
- environment: {log.environment}
- severity: {log.severity}
- message: {log.message!r}

Deterministic first-pass assessment (from a rule-based system, not AI):
- suggested_priority: {deterministic_priority}
- impact_score: {deterministic_impact_score}

Correlated evidence from the same source/service/region in the last {EVIDENCE_WINDOW_MINUTES} minutes ({len(evidence)} event(s)):
{evidence_lines}

Confirm or revise the deterministic priority suggestion based on this evidence, and produce your structured verdict.
"""


async def run_triage_for_log(log_id: int) -> dict:
    """
    Entry point invoked as a FastAPI BackgroundTask from iLogs' ingest
    handler (return value ignored there), and directly by the manual
    /itriage/run/{log_id} endpoint for testing (return value used).
    Opens its own DB session since a request's session may be gone by the
    time a background task runs.
    """
    async with async_session() as db:
        log = await db.get(ILog, log_id)
        if log is None:
            return {"action": "error", "detail": "log not found"}

        existing = await _find_open_incident(db, log)
        if existing is not None:
            await link_log(db, existing.id, log.id)
            return {"action": "linked_existing", "incident_id": existing.id}

        evidence = await _fetch_evidence(db, log)
        if not evidence:
            evidence = [log]

        deterministic_confidence = compute_deterministic_confidence(
            evidence_count=len(evidence),
            distinct_severities={e.severity for e in evidence},
            time_span_minutes=(evidence[-1].timestamp - evidence[0].timestamp).total_seconds() / 60,
        )
        deterministic_priority = infer_incident_priority(log)
        deterministic_impact_score = compute_impact_score(log)

        incident = await create_incident(
            db,
            IncidentCreate(
                title=f"Investigating {log.service} in {log.region}",
                region=log.region,
                service=log.service,
                environment=log.environment,
                source_id=log.source_id,
                priority=deterministic_priority,
                impact_score=deterministic_impact_score,
                log_ids=[e.id for e in evidence],
            ),
        )

        if is_valid_transition(incident.status, STATUS_TRIAGE):
            incident = await update_status(db, incident, STATUS_TRIAGE)

        prompt = _build_prompt(log, evidence, deterministic_priority, deterministic_impact_score)

        try:
            verdict = await call_gemini_for_verdict(prompt)
        except Exception as exc:
            print(f"[iTriage] Gemini call failed for log_id={log_id}, incident_id={incident.id}: {exc}")
            return {"action": "gemini_error", "incident_id": incident.id, "detail": str(exc)}

        db.add(
            TriageResult(
                log_id=log.id,
                incident_id=incident.id,
                deterministic_confidence=deterministic_confidence,
                verdict=verdict.model_dump(),
            )
        )

        incident.title = verdict.incident_summary[:255]
        incident.summary = verdict.incident_summary
        incident.priority = verdict.priority_assessment
        await db.commit()
        await db.refresh(incident)

        next_status = STATUS_CLOSED if verdict.incident_likelihood == "low" else STATUS_AWAITING_APPROVAL
        if is_valid_transition(incident.status, next_status):
            incident = await update_status(db, incident, next_status)

        return {
            "action": "triaged",
            "incident_id": incident.id,
            "status": incident.status,
            "deterministic_confidence": deterministic_confidence,
            "verdict": verdict.model_dump(),
        }
