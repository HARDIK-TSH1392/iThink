from sqlalchemy import String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional

from app.database import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TeamService(Base):
    """
    Which team owns a given service. `service` (not a composite key) is the
    primary key so a service can only ever map to exactly one owning team --
    matches how a multi-service incident (e.g. auth-api + database-primary)
    is expected to resolve to one team per implicated service, never a
    service being ambiguously co-owned by two teams.

    `service` is a plain string, same as iLogs/iNcidents/iTriage's `service`
    field -- there's no separate Service entity anywhere else in the app.
    """

    __tablename__ = "team_services"

    service: Mapped[str] = mapped_column(String, primary_key=True)
    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), index=True
    )


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    slack_user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String, index=True)  # member / lead / manager

    # Functional role (backend_engineer, devops, business_analyst, ...) --
    # deliberately separate from `role` above, which is approval-hierarchy
    # (member/lead/manager), not function. Nullable: existing employees
    # predate this field, and iCall's role-inference treats a null title as
    # "no directory answer, fall back to conversation-inferred role".
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)

    region: Mapped[str] = mapped_column(String, index=True)
    timezone: Mapped[str] = mapped_column(String)

    on_leave: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
