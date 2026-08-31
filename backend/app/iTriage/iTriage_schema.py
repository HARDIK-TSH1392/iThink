from pydantic import BaseModel, Field
from typing import Literal, List


class EvidenceItem(BaseModel):
    """A single discrete, attributable observation used to support a verdict."""

    log_id: int
    observation: str


class TriageVerdict(BaseModel):
    """
    Structured output requested from Gemini. Deliberately excludes:
    - a "root_cause" field (iThink never autonomously declares root cause)
    - a self-reported "confidence" field (confidence is computed deterministically
      by our own code from evidence coverage, not asked of the model)
    - a "recommended_next_action" field (technical remediation is for humans on
      the call to decide, not something inferred from log evidence alone)
    """

    incident_likelihood: Literal["low", "medium", "high"]
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    priority_assessment: Literal["P1", "P2", "P3"]
    priority_rationale: str
    incident_summary: str
    implicated_services: List[str] = Field(default_factory=list)


class TriageResultRead(BaseModel):
    """Schema for reading a persisted triage result (response)."""

    id: int
    log_id: int
    incident_id: int
    deterministic_confidence: Literal["low", "medium", "high"]
    verdict: TriageVerdict

    class Config:
        from_attributes = True
