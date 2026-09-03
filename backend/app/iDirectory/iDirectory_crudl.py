from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Sequence

from .iDirectory_model import Team, TeamService, Employee
from .iDirectory_schema import TeamCreate, EmployeeCreate
from .iDirectory_utils import ROLE_MANAGER, ROLE_LEAD


# -----------------------------------------------------------------------------
# Teams
# -----------------------------------------------------------------------------

async def create_team(db: AsyncSession, payload: TeamCreate) -> Team:
    """
    Create a new team, optionally linking the services it owns at creation
    time.
    """
    team = Team(name=payload.name)
    db.add(team)
    await db.flush()

    for service in dict.fromkeys(payload.services):  # de-dupe, preserve order
        db.add(TeamService(team_id=team.id, service=service))

    await db.commit()
    await db.refresh(team)
    return team


async def get_team(db: AsyncSession, team_id: int) -> Optional[Team]:
    return await db.get(Team, team_id)


async def list_teams(db: AsyncSession, limit: int = 100, offset: int = 0) -> Sequence[Team]:
    result = await db.execute(
        select(Team).order_by(Team.id).limit(limit).offset(offset)
    )
    return result.scalars().all()


async def get_team_services(db: AsyncSession, team_id: int) -> List[str]:
    result = await db.execute(
        select(TeamService.service).where(TeamService.team_id == team_id)
    )
    return [row[0] for row in result.all()]


async def add_team_service(db: AsyncSession, team_id: int, service: str) -> bool:
    """
    Link a service to a team. Returns False if the service is already owned
    by a team (including this one) rather than silently overwriting
    ownership -- a service having exactly one owning team is a hard
    invariant, not something to quietly change.

    The upfront check is an early-exit for the common case, not the actual
    safety guarantee -- it's not atomic with the insert, so two concurrent
    callers could both pass it before either commits. `service` being the
    table's primary key is what actually enforces exclusivity; the
    try/except below turns a losing race into a clean False instead of an
    unhandled IntegrityError. Same class of gap as the log_ids/email
    duplicate-key issues found earlier, fixed the same way here up front
    rather than by waiting to reproduce it.
    """
    existing = await db.get(TeamService, service)
    if existing is not None:
        return False

    db.add(TeamService(team_id=team_id, service=service))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        return False
    return True


async def get_owning_team(db: AsyncSession, service: str) -> Optional[Team]:
    mapping = await db.get(TeamService, service)
    if mapping is None:
        return None
    return await db.get(Team, mapping.team_id)


# -----------------------------------------------------------------------------
# Employees
# -----------------------------------------------------------------------------

class DuplicateEmailError(Exception):
    pass


async def get_employee_by_email(db: AsyncSession, email: str) -> Optional[Employee]:
    result = await db.execute(select(Employee).where(Employee.email == email))
    return result.scalar_one_or_none()


async def create_employee(db: AsyncSession, payload: EmployeeCreate) -> Employee:
    """
    Raises DuplicateEmailError if `email` collides with an existing
    employee. The API layer's upfront get_employee_by_email check handles
    the common case; this is the backstop for two concurrent creations
    racing past that check before either commits -- same reasoning as
    add_team_service's try/except above.
    """
    employee = Employee(**payload.model_dump())
    db.add(employee)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise DuplicateEmailError(payload.email)
    await db.refresh(employee)
    return employee


async def get_employee(db: AsyncSession, employee_id: int) -> Optional[Employee]:
    return await db.get(Employee, employee_id)


async def list_employees(
    db: AsyncSession,
    team_id: Optional[int] = None,
    role: Optional[str] = None,
    on_leave: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[Employee]:
    query = select(Employee)

    if team_id is not None:
        query = query.where(Employee.team_id == team_id)
    if role is not None:
        query = query.where(Employee.role == role)
    if on_leave is not None:
        query = query.where(Employee.on_leave == on_leave)

    query = query.order_by(Employee.id).limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


async def set_employee_leave(db: AsyncSession, employee: Employee, on_leave: bool) -> Employee:
    employee.on_leave = on_leave
    await db.commit()
    await db.refresh(employee)
    return employee


# -----------------------------------------------------------------------------
# Deterministic resolution (no AI) -- given a service, who approves and who
# responds. Both return an empty/None result rather than guessing when
# nothing confident is available; callers must treat that as "nobody
# available" and fail loud, never silently proceed as if someone was found.
# -----------------------------------------------------------------------------

async def resolve_approver(db: AsyncSession, service: str) -> Optional[Employee]:
    """
    The owning team's manager, or the team lead if the manager is on leave
    (or there is no manager on record). Returns None if there's no owning
    team, or if both the manager and lead are unavailable/missing.
    """
    team = await get_owning_team(db, service)
    if team is None:
        return None

    members = await list_employees(db, team_id=team.id, limit=500)

    manager = next((m for m in members if m.role == ROLE_MANAGER and not m.on_leave), None)
    if manager is not None:
        return manager

    return next((m for m in members if m.role == ROLE_LEAD and not m.on_leave), None)


async def resolve_responders(db: AsyncSession, service: str) -> List[Employee]:
    """
    Every available (not on leave) member of the service's owning team.
    Returns an empty list if there's no owning team or nobody on it is
    currently available.
    """
    team = await get_owning_team(db, service)
    if team is None:
        return []

    members = await list_employees(db, team_id=team.id, limit=500)
    return [m for m in members if not m.on_leave]
