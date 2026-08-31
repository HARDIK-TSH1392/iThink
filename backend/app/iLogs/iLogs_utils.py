from .iLogs_schema import LogEventCreate
from typing import Set


# -----------------------------------------------------------------------------
# Configuration: Critical services and high-priority regions
# -----------------------------------------------------------------------------

CRITICAL_SERVICES: Set[str] = {
    "auth-api",
    "payment-service",
    "order-service",
    "login-service",
}

HIGH_PRIORITY_REGIONS: Set[str] = {
    "us-east",
    "eu-west",
    "ap-south",
}


# -----------------------------------------------------------------------------
# Impact scoring (deterministic, tunable)
# -----------------------------------------------------------------------------

def compute_impact_score(event: LogEventCreate) -> int:
    """
    Compute a deterministic impact score for a log event.

    This is inspired by standard risk/impact scoring patterns:
    - Severity contributes a base score (higher for more severe events).
    - Production environment multiplies the score (higher business impact).
    - Critical services and high-priority regions add extra weight.

    The exact weights are tunable and should be adjusted based on
    historical incident data and business priorities.

    Higher score = higher expected business impact.
    """
    score = 0

    # ---- Severity base score ----
    # Inspired by standard severity scales (catastrophic → negligible)
    if event.severity == "prod_down":
        score += 40  # catastrophic / service outage
    elif event.severity == "critical":
        score += 30  # major functionality degraded
    elif event.severity == "error":
        score += 20  # partial/non-critical loss
    elif event.severity == "warning":
        score += 10  # minor / early signal

    # ---- Environment multiplier ----
    # Production incidents have higher business impact (standard practice)
    if event.environment == "production":
        score *= 2
    elif event.environment == "staging":
        score *= 1
    elif event.environment == "dev":
        score *= 1

    # ---- Service criticality bonus ----
    # Critical services get additional weight (asset criticality factor)
    if event.service in CRITICAL_SERVICES:
        score += 25

    # ---- Region priority bonus ----
    # High-priority regions get additional weight
    if event.region in HIGH_PRIORITY_REGIONS:
        score += 15

    return score


# -----------------------------------------------------------------------------
# Incident triggering rules (deterministic, multi-factor)
# -----------------------------------------------------------------------------

def should_trigger_incident(event: LogEventCreate) -> bool:
    """
    Determine if a log event should be considered as a potential incident.

    Rules (deterministic, no AI):
    - Any 'prod_down' is always an incident candidate.
    - 'critical' in production environment is always an incident candidate.
    - 'error' in production on a critical service is an incident candidate.
    - Non-production environments are ignored for now (can be changed later).
    """
    # Always consider prod_down in any environment as incident-worthy
    if event.severity == "prod_down":
        return True

    # Only consider production logs for incident creation (for now)
    if event.environment != "production":
        return False

    if event.severity == "critical":
        return True

    if event.severity == "error" and event.service in CRITICAL_SERVICES:
        return True

    return False


# -----------------------------------------------------------------------------
# Priority inference (P1/P2/P3) aligned with business impact
# -----------------------------------------------------------------------------

def infer_incident_priority(event: LogEventCreate) -> str:
    """
    Map log event to an incident priority (P1/P2/P3) using deterministic rules.

    Priority logic (aligned with standard P1–P3 definitions):
    - P1: Core functionality broken or revenue impact; immediate response required.
    - P2: Major functionality degraded; workaround exists.
    - P3: Partial/non-critical loss; lower business impact.

    References:
    - P1/P2/P3 definitions from incident management best practices.
    - Multi-factor prioritization (severity + asset criticality + region).
    """
    # P1: Production down or critical on critical service in high-priority region
    if event.severity == "prod_down" and event.environment == "production":
        return "P1"

    if event.severity == "critical":
        if event.service in CRITICAL_SERVICES and event.region in HIGH_PRIORITY_REGIONS:
            return "P1"
        return "P2"

    if event.severity == "error":
        if event.service in CRITICAL_SERVICES and event.environment == "production":
            return "P2"

    # Fallback for anything else that passed should_trigger_incident
    return "P3"


# -----------------------------------------------------------------------------
# AI review candidate filter (broader than incident trigger)
# -----------------------------------------------------------------------------

def is_candidate_for_ai_review(event: LogEventCreate) -> bool:
    """
    Broader filter for logs that should be sent to the AI triage module.

    This can be wider than should_trigger_incident, e.g.:
    - Include warnings on critical services.
    - Include errors in staging that might indicate upcoming prod issues.
    """
    if event.environment == "production":
        if event.severity in ("warning", "error", "critical", "prod_down"):
            return True

    # Optionally: include critical services even in staging
    if event.service in CRITICAL_SERVICES and event.severity in ("error", "critical"):
        return True

    return False