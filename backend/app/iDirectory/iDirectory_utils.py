from typing import Set

# -----------------------------------------------------------------------------
# Employee role constants
# -----------------------------------------------------------------------------

ROLE_MEMBER = "member"
ROLE_LEAD = "lead"
ROLE_MANAGER = "manager"

ALL_ROLES: Set[str] = {ROLE_MEMBER, ROLE_LEAD, ROLE_MANAGER}
