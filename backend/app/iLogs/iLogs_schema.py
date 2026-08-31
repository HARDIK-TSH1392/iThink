from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Dict, Any, Optional


class LogEventCreate(BaseModel):
    """Schema for creating a log entry (request body)."""

    source_id: str = Field(..., description="Unique ID of the log source")
    source_type: Literal["k8s_deployment", "vm", "lambda", "database", "other"]
    region: str
    service: str
    environment: Literal["production", "staging", "dev"]
    severity: Literal["warning", "error", "critical", "prod_down"]
    timestamp: datetime
    message: str
    attributes: Optional[Dict[str, Any]] = {}
    raw: Optional[Dict[str, Any]] = {}


class LogEventRead(BaseModel):
    """Schema for reading a log entry (response)."""

    id: int
    source_id: str
    source_type: str
    region: str
    service: str
    environment: str
    severity: str
    timestamp: datetime
    message: str
    attributes: Optional[Dict[str, Any]] = {}
    raw: Optional[Dict[str, Any]] = {}
    created_at: datetime

    class Config:
        from_attributes = True