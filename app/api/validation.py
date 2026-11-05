"""Superadmin form-validation helpers: live availability checks for the admin console.

GET /api/admin/validate/org-name  ?name=&exclude_id=  -> {"available": bool}
GET /api/admin/validate/user-email ?email=            -> {"available": bool}

Org-name availability is decided on the SLUG (slugify(name)), matching the
uniqueness rule enforced on create/rename — "Acme Inc" and "acme-inc!" are the
same organization name as far as availability is concerned.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import require_superadmin
from app.db.models import Organization, User
from app.db.session import get_db
from app.utils import parse_uuid, slugify

router = APIRouter(prefix="/api/admin/validate", tags=["validation"])


@router.get("/org-name")
async def validate_org_name(
    name: str,
    exclude_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superadmin),
):
    slug = slugify(name)
    if not slug:
        return {"available": False}
    stmt = select(Organization.id).where(Organization.slug == slug)
    excluded = parse_uuid(exclude_id)
    if excluded is not None:
        stmt = stmt.where(Organization.id != excluded)
    taken = (await db.execute(stmt.limit(1))).scalar_one_or_none()
    return {"available": taken is None}


@router.get("/user-email")
async def validate_user_email(
    email: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_superadmin),
):
    value = email.strip().lower()
    if not value:
        return {"available": False}
    taken = (
        # CITEXT equality is case-insensitive and uses the unique index;
        # lower(email) does not. See app/api/auth.py.
        await db.execute(select(User.id).where(User.email == value).limit(1))
    ).scalar_one_or_none()
    return {"available": taken is None}
