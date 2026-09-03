from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator, List, Optional

from app.database import async_session
from .iDirectory_schema import (
    TeamCreate,
    TeamRead,
    EmployeeCreate,
    EmployeeRead,
    EmployeeLeaveUpdate,
)
from .iDirectory_crudl import (
    create_team,
    get_team,
    list_teams,
    get_team_services,
    add_team_service,
    create_employee,
    get_employee,
    get_employee_by_email,
    list_employees,
    set_employee_leave,
    resolve_approver,
    resolve_responders,
    DuplicateEmailError,
)


router = APIRouter(prefix="/idirectory", tags=["iDirectory"])


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a DB session per request.
    """
    async with async_session() as session:
        yield session


async def _team_to_read(db: AsyncSession, team) -> TeamRead:
    services = await get_team_services(db, team.id)
    read = TeamRead.model_validate(team)
    read.services = services
    return read


@router.post("/teams", response_model=TeamRead)
async def create_team_endpoint(
    payload: TeamCreate,
    db: AsyncSession = Depends(get_db),
) -> TeamRead:
    """
    Create a new team, optionally linking the services it owns.
    """
    team = await create_team(db, payload)
    return await _team_to_read(db, team)


@router.get("/teams", response_model=List[TeamRead])
async def list_teams_endpoint(db: AsyncSession = Depends(get_db)) -> List[TeamRead]:
    """
    List all teams.
    """
    teams = await list_teams(db)
    return [await _team_to_read(db, team) for team in teams]


@router.get("/teams/{team_id}", response_model=TeamRead)
async def get_team_endpoint(
    team_id: int,
    db: AsyncSession = Depends(get_db),
) -> TeamRead:
    """
    Get a single team by ID.
    """
    team = await get_team(db, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return await _team_to_read(db, team)


@router.post("/teams/{team_id}/services/{service}", response_model=TeamRead)
async def add_team_service_endpoint(
    team_id: int,
    service: str,
    db: AsyncSession = Depends(get_db),
) -> TeamRead:
    """
    Link a service to a team. Fails if the service is already owned by a
    (possibly different) team -- a service has exactly one owning team.
    """
    team = await get_team(db, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    added = await add_team_service(db, team_id, service)
    if not added:
        raise HTTPException(
            status_code=409,
            detail=f"Service '{service}' is already owned by a team",
        )

    return await _team_to_read(db, team)


@router.post("/employees", response_model=EmployeeRead)
async def create_employee_endpoint(
    payload: EmployeeCreate,
    db: AsyncSession = Depends(get_db),
) -> EmployeeRead:
    """
    Create a new employee on an existing team.
    """
    team = await get_team(db, payload.team_id)
    if not team:
        raise HTTPException(
            status_code=422, detail=f"team_id {payload.team_id} does not exist"
        )

    if await get_employee_by_email(db, payload.email):
        raise HTTPException(
            status_code=409, detail=f"An employee with email '{payload.email}' already exists"
        )

    try:
        employee = await create_employee(db, payload)
    except DuplicateEmailError:
        raise HTTPException(
            status_code=409, detail=f"An employee with email '{payload.email}' already exists"
        )
    return EmployeeRead.model_validate(employee)


@router.get("/employees", response_model=List[EmployeeRead])
async def list_employees_endpoint(
    team_id: Optional[int] = Query(None),
    role: Optional[str] = Query(None),
    on_leave: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> List[EmployeeRead]:
    """
    List employees with optional filters.
    """
    employees = await list_employees(db, team_id=team_id, role=role, on_leave=on_leave)
    return [EmployeeRead.model_validate(e) for e in employees]


@router.get("/employees/{employee_id}", response_model=EmployeeRead)
async def get_employee_endpoint(
    employee_id: int,
    db: AsyncSession = Depends(get_db),
) -> EmployeeRead:
    """
    Get a single employee by ID.
    """
    employee = await get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return EmployeeRead.model_validate(employee)


@router.patch("/employees/{employee_id}/leave", response_model=EmployeeRead)
async def set_employee_leave_endpoint(
    employee_id: int,
    payload: EmployeeLeaveUpdate,
    db: AsyncSession = Depends(get_db),
) -> EmployeeRead:
    """
    Toggle an employee's leave status.
    """
    employee = await get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    employee = await set_employee_leave(db, employee, payload.on_leave)
    return EmployeeRead.model_validate(employee)


@router.get("/resolve/approver", response_model=EmployeeRead)
async def resolve_approver_endpoint(
    service: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> EmployeeRead:
    """
    Resolve the approver for a given service: the owning team's manager,
    falling back to the team lead if the manager is on leave. 404s if no
    owning team exists or nobody available was found -- never silently
    picks someone.
    """
    approver = await resolve_approver(db, service)
    if not approver:
        raise HTTPException(
            status_code=404,
            detail=f"No available approver found for service '{service}'",
        )
    return EmployeeRead.model_validate(approver)


@router.get("/resolve/responders", response_model=List[EmployeeRead])
async def resolve_responders_endpoint(
    service: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> List[EmployeeRead]:
    """
    Resolve the responder list for a given service: every available member
    of the owning team. Returns an empty list (not an error) if there's no
    owning team or nobody is currently available -- callers must check for
    an empty result rather than assume responders were found.
    """
    responders = await resolve_responders(db, service)
    return [EmployeeRead.model_validate(r) for r in responders]
