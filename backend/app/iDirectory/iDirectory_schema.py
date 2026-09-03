from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Literal, Optional

EmployeeRole = Literal["member", "lead", "manager"]


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
    region: str
    timezone: str
    on_leave: bool
    created_at: datetime

    class Config:
        from_attributes = True


class EmployeeLeaveUpdate(BaseModel):
    """Toggle an employee's leave status."""

    on_leave: bool
