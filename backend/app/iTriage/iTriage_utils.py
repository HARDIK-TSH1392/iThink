import asyncio
from typing import List, Literal, Set

from google import genai
from google.genai import types

from app.config import get_settings
from .iTriage_schema import TriageVerdict

# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------
# Primary model, with a fallback if it's ever unavailable (rate-limited,
# retired, region restrictions, etc). Both are Gemini flash-tier models
# suited to fast structured-output tasks rather than long-form generation.


# Both verified directly against this project's API key on 2026-09-03 --
# gemini-3.7-flash and gemini-2.5-flash (the previous values here) were
# retired/unavailable and caused a real, hard-to-diagnose outage: primary
# failed silently, fell through to a dead fallback name, and the whole
# triage pipeline stalled at "triage" status with the failure hidden below
# an unlogged except. Using the same verified model for both isn't ideal
# redundancy, but a guessed second name reintroduces exactly this bug.
PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

# Neither Gemini call below had a timeout before this -- found live, not
# hypothetically: a plain "say OK" call to this same model hung past 20s
# with zero response while diagnosing incidents stuck at "triage" status.
# Without a ceiling, that hang holds the per-(source_id, service, region)
# key lock in run_triage_for_log open indefinitely, blocking any further
# ingest for that same key too, not just the one incident. Matches the
# fix already applied to iCall's four Gemini call sites for the identical
# underlying issue (Gemini's own "high demand" instability, not a bug
# here) -- same 25s margin.
GEMINI_CALL_TIMEOUT_S = 25

# Caps concurrent Gemini calls so a burst of correlated log events can't fire
# off unbounded parallel API calls (cost + rate-limit protection).
_gemini_semaphore = asyncio.Semaphore(4)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_settings().gemini_api_key)
    return _client


SYSTEM_INSTRUCTION = """You are the triage-analysis component of iThink, an
incident-coordination system. You are given a triggering log event plus
correlated evidence from the same source/service/region.

Your job is narrow: assess whether this evidence indicates a likely real
incident, and produce a structured verdict for a human to review before any
action is taken.

Hard constraints:
- Never assert or imply a definitive root cause. You may note patterns in
  the evidence, but diagnosis is for humans to determine, not you.
- Never recommend a specific technical fix or remediation action (e.g. do
  not suggest rollbacks, config changes, or restarts). That judgment belongs
  to the people who join the incident call.
- Only reason from the evidence provided. Do not invent facts, timestamps,
  or details that are not present in the input.
- Your priority_assessment should explicitly confirm or revise the given
  deterministic priority suggestion, with a rationale grounded in the
  evidence — not an independent guess made in a vacuum.
- incident_summary should be a plain-language description of what is
  happening, suitable for a meeting invite or approval request — not a
  recommendation of what to do about it.
- Output must strictly match the provided response schema.
"""


async def call_gemini_for_verdict(prompt: str) -> TriageVerdict:
    """
    Calls Gemini with the given evidence prompt and returns a validated
    TriageVerdict. Tries the primary model first, falls back once on failure.
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=TriageVerdict,
    )

    async with _gemini_semaphore:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=PRIMARY_MODEL,
                    contents=prompt,
                    config=config,
                ),
                timeout=GEMINI_CALL_TIMEOUT_S,
            )
        except Exception as exc:
            print(f"[iTriage] Verdict primary call failed/timed out, trying fallback: {exc}")
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=FALLBACK_MODEL,
                    contents=prompt,
                    config=config,
                ),
                timeout=GEMINI_CALL_TIMEOUT_S,
            )

    return TriageVerdict.model_validate_json(response.text)


# -----------------------------------------------------------------------------
# Deterministic confidence (NOT self-reported by the LLM)
# -----------------------------------------------------------------------------
# Computed from evidence coverage: how many corroborating events, how
# consistent their severity is, and how tightly clustered in time they are.
# Mirrors the same deterministic, tunable, explainable style as
# iLogs_utils.compute_impact_score.

def compute_deterministic_confidence(
    evidence_count: int,
    distinct_severities: Set[str],
    time_span_minutes: float,
) -> Literal["low", "medium", "high"]:
    score = 0

    if evidence_count >= 5:
        score += 2
    elif evidence_count >= 2:
        score += 1

    if len(distinct_severities) == 1:
        score += 1

    if time_span_minutes <= 15:
        score += 1

    if score >= 3:
        return "high"
    if score >= 1:
        return "medium"
    return "low"
