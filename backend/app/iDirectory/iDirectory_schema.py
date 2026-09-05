from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Literal, Optional

EmployeeRole = Literal["member", "lead", "manager"]

# Functional role categories iCall's role-inference classifies participants
# into (see iCall_utils.ROLE_CATEGORIES) -- kept here too since it's the
# same vocabulary an Employee.title is expected to use for a directory
# match to line up with an inferred role.
EmployeeTitle = Literal[
    "backend_engineer",
    "frontend_engineer",
    "ai_engineer",
    "devops",
    "team_lead",
    "manager",
    "business_analyst",
    "qa_engineer",
    "other",
]


class TeamCreate(BaseModel):
    """Schema for creating a team, optionally with the services it owns."""

    name: str = Field(..., min_length=1)
    services: List[str] = Field(default_factory=list)


class TeamRead(BaseModel):
    """Schema for reading a team (response)."""

    id: int
    name: str
    services: List[str] = Field(default_factory=list)
    created_at: datetime

    class Config:
        from_attributes = True


class EmployeeCreate(BaseModel):
    """Schema for creating an employee (request body)."""

    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1)
    slack_user_id: Optional[str] = None
    team_id: int
    role: EmployeeRole
    title: Optional[EmployeeTitle] = None
    region: str
    timezone: str


class EmployeeRead(BaseModel):
    """Schema for reading an employee (response)."""

    id: int
    name: str
    email: str
    slack_user_id: Optional[str] = None
    team_id: int
    role: str
    title: Optional[str] = None
    region: str
    timezone: str
    on_leave: bool
    created_at: datetime

    class Config:
        from_attributes = True


class EmployeeLeaveUpdate(BaseModel):
    """Toggle an employee's leave status."""

    on_leave: bool
