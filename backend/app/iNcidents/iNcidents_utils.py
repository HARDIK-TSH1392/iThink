from typing import Dict, Set

# -----------------------------------------------------------------------------
# Incident lifecycle status constants
# -----------------------------------------------------------------------------

STATUS_DETECTED = "detected"
STATUS_TRIAGE = "triage"
STATUS_AWAITING_APPROVAL = "awaiting_approval"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_ORCHESTRATING = "orchestrating"
STATUS_IN_CALL = "in_call"
STATUS_POST_CALL = "post_call"
STATUS_RESOLVED = "resolved"
STATUS_CLOSED = "closed"

ALL_STATUSES: Set[str] = {
    STATUS_DETECTED,
    STATUS_TRIAGE,
    STATUS_AWAITING_APPROVAL,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_ORCHESTRATING,
    STATUS_IN_CALL,
    STATUS_POST_CALL,
    STATUS_RESOLVED,
    STATUS_CLOSED,
}

TERMINAL_STATUSES: Set[str] = {STATUS_REJECTED, STATUS_RESOLVED, STATUS_CLOSED}


# -----------------------------------------------------------------------------
# Deterministic lifecycle state machine
# -----------------------------------------------------------------------------
# A lightweight custom state machine (no LangGraph) is sufficient for the
# sprint: it just needs to reject illegal jumps (e.g. resolved -> detected)
# so the pipeline stays explainable.

ALLOWED_TRANSITIONS: Dict[str, Set[str]] = {
    STATUS_DETECTED: {STATUS_TRIAGE, STATUS_CLOSED},
    STATUS_TRIAGE: {STATUS_AWAITING_APPROVAL, STATUS_CLOSED},
    STATUS_AWAITING_APPROVAL: {STATUS_APPROVED, STATUS_REJECTED},
    STATUS_APPROVED: {STATUS_ORCHESTRATING},
    STATUS_ORCHESTRATING: {STATUS_IN_CALL, STATUS_CLOSED},
    STATUS_IN_CALL: {STATUS_POST_CALL, STATUS_CLOSED},
    STATUS_POST_CALL: {STATUS_RESOLVED, STATUS_CLOSED},
    STATUS_REJECTED: set(),
    STATUS_RESOLVED: {STATUS_CLOSED},
    STATUS_CLOSED: set(),
}


def is_valid_transition(current_status: str, new_status: str) -> bool:
    """
    Deterministic guard for the incident lifecycle state machine.
    """
    if new_status not in ALL_STATUSES:
        return False
    return new_status in ALLOWED_TRANSITIONS.get(current_status, set())
