"""Super-admin portal: stats, organizations, departments, users.

Wire contract preserved from RxHivexx (docs/reference/api-auth-admin.md):
serialize_doc-style responses keep `_id` string keys, lists keep the
{data, total, page, limit} envelope, /organizations/all and
/organizations/{org_id}/users keep their bare-array shapes (the latter with
`id` keys), datetimes are ISO-Z, and roles ride the wire as "admin"/"member".

Deliberate contract-preserving fixes (coordinated with the frontend):
- Deactivation (user PUT, bulk, org cascade, org delete) revokes the target
  users' refresh tokens, closing the refresh-forever hole.
- Org reactivation (PUT is_active=true) restores the ORG only — it no longer
  blanket-reactivates users, so individually-deactivated accounts stay off.
- A suspended org stays suspended: creating a user in one is refused, and
  activating a member (singly or in bulk) is refused/skipped. Nothing on the
  login path reads Organization.is_active, so the suspension is carried by the
  users.is_active cascade alone and these are the routes that could undo it.
- reset-password sets must_change_password and revokes sessions; the
  temporary password is still returned per contract. Since batch 73 the flag is
  enforced — the account is refused everywhere but change-password until it
  chooses its own — and the reset also deletes the account's push subscriptions,
  so notification previews stop reaching devices signed in before it. It stamps
  users.sessions_valid_after too (batch 73 review), so an access token issued
  before the reset stays refused after the password is changed, not only until,
  and it locks the target's users row before revoking anything, so a sign-in
  racing the reset cannot keep a session it never saw.
- Create/update user validates the department exists in the target org;
  bulk change_dept skips users whose org doesn't match the department.
- Passwords go through the org password policy; emails are unique
  case-insensitively (citext).
- Search input is escaped and parameterized (no regex/LIKE injection).
"""

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import lock_user_row, wire_role
from app.core.deps import require_superadmin
from app.core.errors import conflict_as_400
from app.core.security import PasswordPolicyError, enforce_password_policy, hash_password
from app.db.models import AuditLog, Department, Organization, PushSubscription, RefreshToken, User, UserRole
from app.db.session import get_db
from app.services import presence
from app.services.audit import log_audit, serialize_audit
from app.utils import generate_password, iso_z, now_utc, parse_uuid, sanitize_text, slugify

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateOrganization(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    logo_url: str | None = None


class UpdateOrganization(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    logo_url: str | None = None
    is_active: bool | None = None


class CreateDepartment(BaseModel):
    org_id: str
    name: str = Field(min_length=2, max_length=100)
    description: str | None = None


class UpdateDepartment(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    description: str | None = None


class CreateUser(BaseModel):
    org_id: str
    dept_id: str
    email: EmailStr
    display_name: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=6)
    role: Literal["admin", "member"] = "member"
    # Mobile is opt-in at creation too, so the default here matches the column's.
    mobile_access: bool = False


class UpdateUser(BaseModel):
    display_name: str | None = Field(default=None, min_length=2, max_length=100)
    dept_id: str | None = None
    role: Literal["admin", "member"] | None = None
    is_active: bool | None = None
    # Granting/revoking native-app sign-in for one account. Superadmin-only by
    # virtue of this router's require_superadmin dependency — the org-admin
    # portal's OrgUpdateUser deliberately has no equivalent field, so an org
    # admin can see the state but cannot change it.
    mobile_access: bool | None = None


class BulkAction(BaseModel):
    user_ids: list[str]
    action: str
    dept_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _like_pattern(value: str) -> str:
    """Escape LIKE metacharacters so search input is matched literally."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _uuid_or_400(value: str, message: str) -> uuid.UUID:
    parsed = parse_uuid(value)
    if parsed is None:
        raise HTTPException(status_code=400, detail=message)
    return parsed


def _db_role(wire: str) -> UserRole:
    return UserRole.org_admin if wire == "admin" else UserRole.member


def _serialize_org(org: Organization) -> dict:
    return {
        "_id": str(org.id),
        "name": org.name,
        "slug": org.slug,
        "logo_url": org.logo_url,
        "created_by": None,  # schema does not persist the creating superadmin
        "created_at": iso_z(org.created_at),
        "is_active": org.is_active,
    }


def _serialize_dept(dept: Department) -> dict:
    return {
        "_id": str(dept.id),
        "org_id": str(dept.org_id),
        "name": dept.name,
        "description": dept.description,
        "created_at": iso_z(dept.created_at),
    }


def _user_row(
    user: User,
    org_names: dict[uuid.UUID, str],
    dept_names: dict[uuid.UUID, str],
    statuses: dict[str, str],
) -> dict:
    return {
        "_id": str(user.id),
        "org_id": str(user.org_id) if user.org_id else None,
        "dept_id": str(user.dept_id) if user.dept_id else None,
        "email": user.email,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "role": wire_role(user.role),
        "status": statuses.get(str(user.id), "offline"),
        "last_seen": iso_z(user.last_seen_at),
        "about": user.about,
        "created_by": str(user.created_by) if user.created_by else None,
        "created_at": iso_z(user.created_at),
        "is_active": user.is_active,
        "mobile_access": user.mobile_access,
        "org_name": org_names.get(user.org_id, "Unknown"),
        "dept_name": dept_names.get(user.dept_id, "Unknown"),
    }


async def _name_maps(
    db: AsyncSession, users: list[User]
) -> tuple[dict[uuid.UUID, str], dict[uuid.UUID, str]]:
    """Batch org/dept display names for a page of users (no N+1)."""
    org_ids = {u.org_id for u in users if u.org_id}
    dept_ids = {u.dept_id for u in users if u.dept_id}
    org_names: dict[uuid.UUID, str] = {}
    dept_names: dict[uuid.UUID, str] = {}
    if org_ids:
        rows = await db.execute(
            select(Organization.id, Organization.name).where(Organization.id.in_(org_ids))
        )
        org_names = dict(rows.all())
    if dept_ids:
        rows = await db.execute(select(Department.id, Department.name).where(Department.id.in_(dept_ids)))
        dept_names = dict(rows.all())
    return org_names, dept_names


async def _serialize_user(db: AsyncSession, user: User) -> dict:
    org_names, dept_names = await _name_maps(db, [user])
    statuses = await presence.get_statuses([user.id])
    return _user_row(user, org_names, dept_names, statuses)


async def _org_counts(
    db: AsyncSession, org_ids: list[uuid.UUID], *, active_users_only: bool
) -> tuple[dict[uuid.UUID, int], dict[uuid.UUID, int]]:
    """Batch (dept_count, user_count) per org for a page of orgs."""
    if not org_ids:
        return {}, {}
    rows = await db.execute(
        select(Department.org_id, func.count())
        .where(Department.org_id.in_(org_ids))
        .group_by(Department.org_id)
    )
    dept_counts = dict(rows.all())
    stmt = select(User.org_id, func.count()).where(User.org_id.in_(org_ids))
    if active_users_only:
        stmt = stmt.where(User.is_active.is_(True))
    rows = await db.execute(stmt.group_by(User.org_id))
    return dept_counts, dict(rows.all())


async def _revoke_refresh_tokens(
    db: AsyncSession, user_ids: list[uuid.UUID], *, client: str | None = None
) -> None:
    """Revoke live refresh sessions for these users.

    `client` narrows the revocation to sessions opened by one client. Revoking a
    mobile grant must sign the user out of the phone without signing them out of
    the browser they may be using right now, so that path passes client="mobile";
    deactivation passes nothing and kills everything.
    """
    if not user_ids:
        return
    stmt = update(RefreshToken).where(RefreshToken.user_id.in_(user_ids), RefreshToken.revoked_at.is_(None))
    if client is not None:
        stmt = stmt.where(RefreshToken.client == client)
    await db.execute(stmt.values(revoked_at=now_utc()).execution_options(synchronize_session=False))


async def _deactivate_org_users(db: AsyncSession, org_id: uuid.UUID) -> int:
    """Cascade org suspension: deactivate every user and revoke their sessions."""
    ids = list((await db.execute(select(User.id).where(User.org_id == org_id))).scalars().all())
    if ids:
        await db.execute(
            update(User)
            .where(User.id.in_(ids))
            .values(is_active=False)
            .execution_options(synchronize_session=False)
        )
        await _revoke_refresh_tokens(db, ids)
    return len(ids)


async def _count_online_users() -> int:
    """Currently-online users, read from the presence index.

    Was a SCAN for presence:* — which walks the entire keyspace, because MATCH
    filters server-side rather than narrowing the iteration, so the cost was the
    size of Redis (ratelimit:, call:, direct: and user: keys included) rather
    than the number of people online. It is now a trim plus a ZCARD against the
    sorted set maintained by app/services/presence.py.

    Still best-effort, as it was: presence degrades to 0 rather than taking the
    four Postgres-backed tiles down with it.
    """
    return await presence.count_online()


async def _lock_orgs(db: AsyncSession, org_ids) -> dict[uuid.UUID, Organization]:
    """SELECT ... FOR UPDATE the given organisations, in a deterministic order.

    Reading Organization.is_active without a lock only settles the sequential
    case. Under READ COMMITTED — which is what this engine runs, no isolation
    level is set — a suspension can commit between another request's read of the
    org and its own write, and the cascade in _deactivate_org_users only touches
    the users that existed when it ran. So a create or activate that checked an
    org it saw as active could still land an active user inside a suspended one.

    Every writer that reads or changes an org's active state takes this lock, so
    suspension and activation serialise against each other. Ordering by id keeps
    two multi-org batches from deadlocking on the same pair in opposite orders;
    single-org callers satisfy it for free.
    """
    ids = sorted(set(org_ids))
    if not ids:
        return {}
    rows = (
        (
            await db.execute(
                select(Organization)
                .where(Organization.id.in_(ids))
                .order_by(Organization.id)
                .with_for_update()
            )
        )
        .scalars()
        .all()
    )
    return {o.id: o for o in rows}


async def _get_org_or_404(db: AsyncSession, org_id: str, *, lock: bool = False) -> Organization:
    """Resolve an organisation. `lock` for callers that go on to write it or to
    decide something from its active state; plain reads skip the lock."""
    oid = _uuid_or_400(org_id, "Invalid organization ID")
    org = (await _lock_orgs(db, [oid])).get(oid) if lock else await db.get(Organization, oid)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org


async def _get_dept_or_404(db: AsyncSession, dept_id: str) -> Department:
    did = _uuid_or_400(dept_id, "Invalid department ID")
    dept = await db.get(Department, did)
    if dept is None:
        raise HTTPException(status_code=404, detail="Department not found")
    return dept


async def _get_user_or_404(db: AsyncSession, user_id: str) -> User:
    uid = _uuid_or_400(user_id, "Invalid user ID")
    user = await db.get(User, uid)
    # Superadmins are not managed through /api/admin/users (parity with the
    # Mongo build, where they lived in a separate collection).
    if user is None or user.role == UserRole.superadmin:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    total_orgs = (await db.execute(select(func.count()).select_from(Organization))).scalar_one()
    total_depts = (await db.execute(select(func.count()).select_from(Department))).scalar_one()
    total_users = (
        await db.execute(select(func.count()).select_from(User).where(User.role != UserRole.superadmin))
    ).scalar_one()
    active_today = await _count_online_users()
    logs = (await db.execute(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(10))).scalars().all()
    return {
        "total_orgs": total_orgs,
        "total_depts": total_depts,
        "total_users": total_users,
        "active_today": active_today,
        "recent_activity": [serialize_audit(log) for log in logs],
    }


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


@router.get("/organizations")
async def list_organizations(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: str = "",
    sort: str = "created_at",
    order: str = "desc",
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    filters = []
    if search:
        filters.append(Organization.name.ilike(_like_pattern(search), escape="\\"))
    sort_col = {
        "name": Organization.name,
        "created_at": Organization.created_at,
        "slug": Organization.slug,
    }.get(sort, Organization.created_at)
    total = (await db.execute(select(func.count()).select_from(Organization).where(*filters))).scalar_one()
    stmt = (
        select(Organization)
        .where(*filters)
        .order_by(sort_col.desc() if order == "desc" else sort_col.asc())
        .offset((page - 1) * limit)
        .limit(limit)
    )
    orgs = (await db.execute(stmt)).scalars().all()
    dept_counts, user_counts = await _org_counts(db, [o.id for o in orgs], active_users_only=False)
    data = [
        {
            **_serialize_org(o),
            "dept_count": dept_counts.get(o.id, 0),
            "user_count": user_counts.get(o.id, 0),
        }
        for o in orgs
    ]
    return {"data": data, "total": total, "page": page, "limit": limit}


@router.post("/organizations")
async def create_organization(
    body: CreateOrganization,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    name = sanitize_text(body.name).strip()
    slug = slugify(name)
    clash = (await db.execute(select(Organization.id).where(Organization.slug == slug))).first()
    if clash:
        raise HTTPException(status_code=400, detail="Organization name already exists")
    org = Organization(name=name, slug=slug, logo_url=body.logo_url, is_active=True, created_at=now_utc())
    db.add(org)
    async with conflict_as_400(db, "Organization name already exists"):
        await db.flush()
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="create_organization",
        target=str(org.id),
        details={"name": name, "slug": slug},
        org_id=org.id,
    )
    await db.commit()
    return _serialize_org(org)


@router.get("/organizations/all")
async def list_all_organizations(
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    orgs = (
        (
            await db.execute(
                select(Organization).where(Organization.is_active.is_(True)).order_by(Organization.name.asc())
            )
        )
        .scalars()
        .all()
    )
    dept_counts, user_counts = await _org_counts(db, [o.id for o in orgs], active_users_only=True)
    return [
        {
            **_serialize_org(o),
            "user_count": user_counts.get(o.id, 0),
            "dept_count": dept_counts.get(o.id, 0),
        }
        for o in orgs
    ]


@router.get("/organizations/{org_id}")
async def get_organization(
    org_id: str,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    org = await _get_org_or_404(db, org_id)
    return _serialize_org(org)


@router.put("/organizations/{org_id}")
async def update_organization(
    org_id: str,
    body: UpdateOrganization,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    # Locked: this is the writer that suspends an org, and it has to serialise
    # against the create/activate paths that read is_active to decide.
    org = await _get_org_or_404(db, org_id, lock=True)
    update_data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "name" in update_data:
        name = sanitize_text(update_data["name"]).strip()
        slug = slugify(name)
        clash = (
            await db.execute(
                select(Organization.id).where(Organization.slug == slug, Organization.id != org.id)
            )
        ).first()
        if clash:
            raise HTTPException(status_code=400, detail="Organization name already exists")
        org.name = name
        org.slug = slug
        update_data["name"] = name
    if "logo_url" in update_data:
        org.logo_url = update_data["logo_url"]
    if "is_active" in update_data:
        org.is_active = update_data["is_active"]
        if update_data["is_active"] is False:
            # Fix: suspending an org also revokes every member session.
            await _deactivate_org_users(db, org.id)
        # Deliberate fix: reactivation restores the org only — users stay as
        # they are, so individually-deactivated accounts are not resurrected.

    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="update_organization",
        target=str(org.id),
        details=update_data,
        org_id=org.id,
    )
    # The rename races the same constraint the pre-check above tests for, and
    # raises from commit() rather than a flush: nothing is flushed explicitly
    # here, so the UPDATE is emitted when the transaction closes.
    async with conflict_as_400(db, "Organization name already exists"):
        await db.commit()
    return _serialize_org(org)


@router.delete("/organizations/{org_id}")
async def delete_organization(
    org_id: str,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    # Locked for the same reason as update_organization: this is a suspension.
    org = await _get_org_or_404(db, org_id, lock=True)
    org.is_active = False
    deactivated = await _deactivate_org_users(db, org.id)
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="delete_organization",
        target=str(org.id),
        details={"name": org.name, "users_deactivated": deactivated},
        org_id=org.id,
    )
    await db.commit()
    return {"message": "Organization deactivated"}


@router.get("/organizations/{org_id}/users")
async def organization_users_by_department(
    org_id: str,
    search: str = "",
    limit: int = Query(200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    """An organization's members grouped by department, for the cross-org picker.

    NARROWED AND CAPPED, which it was not. This returned every active member of
    the organization, and it was the last roster in the API without a ceiling:
    org_admin.list_users and the admin user list both paginate at limit <= 100,
    and api/contacts.list_contacts caps at 500 after measuring the unbounded
    version at 24,999 rows and 5.5 MB on a 25,000-user tenant — 430ms during which
    the worker served nothing else.

    A cap alone would have been worse than the problem. The one caller,
    pages/admin/CrossOrgGroups.jsx, fetched the whole roster and filtered it in the
    browser, so truncating the response would have made every member past the
    ceiling silently unpickable with nothing to indicate why. list_contacts states
    the precondition: "Every picker in the app already narrows with `search`, which
    is what makes a cap safe." This picker did not, so `search` lands here and the
    debounced input lands in the same change.

    Departments are unfiltered by `search` on purpose: the picker renders the tree,
    and a department whose members are all filtered out still has to appear, or the
    admin cannot tell an empty result from a missing group.
    """
    oid = _uuid_or_400(org_id, "Invalid org ID")
    depts = (
        (await db.execute(select(Department).where(Department.org_id == oid).order_by(Department.name.asc())))
        .scalars()
        .all()
    )
    # Five columns, not the whole entity, and the department filter in SQL.
    #
    # This was `select(User)`, which materializes all fifteen mapped columns —
    # including password_hash — for every active member of the organization, in
    # order to serialize the five below. Nothing leaked: the hash is never
    # rendered. It was simply read out of Postgres and held in the worker's memory
    # for every user on a request that renders a directory, which is credential
    # material this endpoint has no reason to touch.
    #
    # And the loop then dropped every user with no department AFTER loading them.
    # Measured on a seeded 2,000-user organization with a tenth of them
    # departmentless: 200 rows fetched and discarded, and 255 KB served. The
    # discard is now a WHERE clause.
    users = (
        await db.execute(
            select(
                User.id,
                User.display_name,
                User.email,
                User.avatar_url,
                User.role,
                User.dept_id,
            )
            .where(
                User.org_id == oid,
                User.is_active.is_(True),
                User.role != UserRole.superadmin,
                User.dept_id.is_not(None),
                *(
                    [
                        or_(
                            User.display_name.ilike(_like_pattern(search), escape="\\"),
                            User.email.ilike(_like_pattern(search), escape="\\"),
                        )
                    ]
                    if search
                    else []
                ),
            )
            .order_by(User.display_name.asc())
            .limit(limit)
        )
    ).all()
    by_dept: dict[uuid.UUID, list[dict]] = {}
    for u in users:
        by_dept.setdefault(u.dept_id, []).append(
            {
                "id": str(u.id),
                "display_name": u.display_name,
                "email": u.email,
                "avatar_url": u.avatar_url,
                "role": wire_role(u.role),
            }
        )
    return [{"id": str(d.id), "name": d.name, "users": by_dept.get(d.id, [])} for d in depts]


# ---------------------------------------------------------------------------
# Departments
# ---------------------------------------------------------------------------


@router.get("/departments")
async def list_departments(
    org_id: str | None = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: str = "",
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    filters = []
    if org_id:
        oid = _uuid_or_400(org_id, "Invalid org_id")
        filters.append(Department.org_id == oid)
    if search:
        filters.append(Department.name.ilike(_like_pattern(search), escape="\\"))
    total = (await db.execute(select(func.count()).select_from(Department).where(*filters))).scalar_one()
    depts = (
        (
            await db.execute(
                select(Department)
                .where(*filters)
                .order_by(Department.created_at.desc())
                .offset((page - 1) * limit)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    member_counts: dict[uuid.UUID, int] = {}
    dept_ids = [d.id for d in depts]
    if dept_ids:
        rows = await db.execute(
            select(User.dept_id, func.count()).where(User.dept_id.in_(dept_ids)).group_by(User.dept_id)
        )
        member_counts = dict(rows.all())
    data = [{**_serialize_dept(d), "member_count": member_counts.get(d.id, 0)} for d in depts]
    return {"data": data, "total": total, "page": page, "limit": limit}


@router.post("/departments")
async def create_department(
    body: CreateDepartment,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    oid = _uuid_or_400(body.org_id, "Invalid org_id")
    org = await db.get(Organization, oid)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    name = sanitize_text(body.name).strip()
    duplicate = (
        await db.execute(select(Department.id).where(Department.org_id == oid, Department.name == name))
    ).first()
    if duplicate:
        raise HTTPException(status_code=400, detail="Department already exists in this organization")
    dept = Department(
        org_id=oid,
        name=name,
        description=sanitize_text(body.description) if body.description else None,
        created_at=now_utc(),
    )
    db.add(dept)
    async with conflict_as_400(db, "Department already exists in this organization"):
        await db.flush()
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="create_department",
        target=str(dept.id),
        details={"name": name},
        org_id=oid,
    )
    await db.commit()
    return _serialize_dept(dept)


@router.get("/departments/{dept_id}")
async def get_department(
    dept_id: str,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    dept = await _get_dept_or_404(db, dept_id)
    return _serialize_dept(dept)


@router.put("/departments/{dept_id}")
async def update_department(
    dept_id: str,
    body: UpdateDepartment,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    dept = await _get_dept_or_404(db, dept_id)
    update_data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "name" in update_data:
        name = sanitize_text(update_data["name"]).strip()
        duplicate = (
            await db.execute(
                select(Department.id).where(
                    Department.org_id == dept.org_id,
                    Department.name == name,
                    Department.id != dept.id,
                )
            )
        ).first()
        if duplicate:
            raise HTTPException(status_code=400, detail="Department name already exists in this organization")
        dept.name = name
        update_data["name"] = name
    if "description" in update_data:
        description = sanitize_text(update_data["description"])
        dept.description = description
        update_data["description"] = description

    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="update_department",
        target=str(dept.id),
        details=update_data,
        org_id=dept.org_id,
    )
    # The rename races the same constraint the pre-check above tests for, and
    # raises from commit() rather than a flush: nothing is flushed explicitly
    # here, so the UPDATE is emitted when the transaction closes.
    async with conflict_as_400(db, "Department already exists in this organization"):
        await db.commit()
    return _serialize_dept(dept)


@router.delete("/departments/{dept_id}")
async def delete_department(
    dept_id: str,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    dept = await _get_dept_or_404(db, dept_id)
    active_users = (
        await db.execute(
            select(func.count()).select_from(User).where(User.dept_id == dept.id, User.is_active.is_(True))
        )
    ).scalar_one()
    if active_users:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete department: {active_users} active users",
        )
    org_id, name = dept.org_id, dept.name
    await db.delete(dept)
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="delete_department",
        target=str(dept.id),
        details={"name": name},
        org_id=org_id,
    )
    await db.commit()
    return {"message": "Department deleted"}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


@router.get("/users")
async def list_users(
    org_id: str | None = None,
    dept_id: str | None = None,
    search: str = "",
    status: str = "",
    # "granted" | "denied" — lets the portal answer "who can actually use the app?"
    # without paging through everyone. Empty means no filter.
    mobile: str = "",
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    sort: str = "created_at",
    order: str = "desc",
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    filters = [User.role != UserRole.superadmin]
    if org_id:
        oid = _uuid_or_400(org_id, "Invalid org_id")
        filters.append(User.org_id == oid)
    if dept_id:
        did = _uuid_or_400(dept_id, "Invalid dept_id")
        filters.append(User.dept_id == did)
    if search:
        pattern = _like_pattern(search)
        filters.append(User.display_name.ilike(pattern, escape="\\") | User.email.ilike(pattern, escape="\\"))
    if status == "active":
        filters.append(User.is_active.is_(True))
    elif status == "inactive":
        filters.append(User.is_active.is_(False))
    if mobile == "granted":
        filters.append(User.mobile_access.is_(True))
    elif mobile == "denied":
        filters.append(User.mobile_access.is_(False))

    sort_col = {
        "display_name": User.display_name,
        "email": User.email,
        "created_at": User.created_at,
        "last_seen": User.last_seen_at,
    }.get(sort, User.created_at)
    total = (await db.execute(select(func.count()).select_from(User).where(*filters))).scalar_one()
    users = (
        (
            await db.execute(
                select(User)
                .where(*filters)
                .order_by(sort_col.desc() if order == "desc" else sort_col.asc())
                .offset((page - 1) * limit)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    org_names, dept_names = await _name_maps(db, list(users))
    statuses = await presence.get_statuses([u.id for u in users])
    data = [_user_row(u, org_names, dept_names, statuses) for u in users]
    return {"data": data, "total": total, "page": page, "limit": limit}


@router.post("/users")
async def create_user(
    body: CreateUser,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    oid = _uuid_or_400(body.org_id, "Invalid org_id")
    did = _uuid_or_400(body.dept_id, "Invalid dept_id")
    # Locked, and held until this request commits: an unlocked read could see an
    # org a concurrent suspension is about to close, and that suspension's
    # cascade would never reach the row inserted below.
    org = (await _lock_orgs(db, [oid])).get(oid)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    # Suspending an org deactivates its members, but nothing on the login path
    # re-reads Organization.is_active — so an account minted here after the
    # suspension would sign in normally against a tenant that is shut off.
    if not org.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot create a user in a suspended organization",
        )
    dept = (
        await db.execute(select(Department).where(Department.id == did, Department.org_id == oid))
    ).scalar_one_or_none()
    if dept is None:
        raise HTTPException(status_code=404, detail="Department not found in this organization")
    try:
        enforce_password_policy(body.password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # citext equality is case-insensitive; the unique index backstops races.
    existing = (await db.execute(select(User.id).where(User.email == str(body.email)))).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use")

    user = User(
        org_id=oid,
        dept_id=did,
        email=str(body.email),
        display_name=sanitize_text(body.display_name).strip(),
        password_hash=await hash_password(body.password),
        role=_db_role(body.role),
        is_active=True,
        mobile_access=body.mobile_access,
        must_change_password=False,
        created_by=actor.id,
        created_at=now_utc(),
    )
    db.add(user)
    async with conflict_as_400(db, "Email already in use"):
        await db.flush()
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="create_user",
        target=str(user.id),
        details={"email": user.email, "role": body.role, "mobile_access": body.mobile_access},
        org_id=oid,
    )
    await db.commit()
    return _user_row(user, {oid: org.name}, {did: dept.name}, {})


@router.post("/users/bulk-action")
async def bulk_user_action(
    body: BulkAction,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    valid_ids = [u for u in (parse_uuid(v) for v in body.user_ids) if u is not None]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid user IDs")

    if body.action in ("deactivate", "activate"):
        stmt = select(User.id).where(User.id.in_(valid_ids), User.role != UserRole.superadmin)
        if body.action == "activate":
            # An org suspension deactivated these accounts deliberately, and the
            # login path never re-checks the org — so a bulk activate is the one
            # call that can quietly bring a whole suspended tenant back online.
            # Members of suspended orgs are skipped rather than failing the batch;
            # `count` in the response already reports what actually changed.
            #
            # Two passes, because the eligibility filter reads org state and this
            # batch can span organisations: the first pass only says which orgs
            # are in play, then those are locked in id order, and only then is
            # eligibility decided — against org rows a concurrent suspension can
            # no longer move underneath us. Doing it in one query would settle
            # eligibility on an unlocked read of exactly the column at issue.
            candidate_org_ids = (
                (
                    await db.execute(
                        select(User.org_id).where(
                            User.id.in_(valid_ids),
                            User.role != UserRole.superadmin,
                            User.org_id.is_not(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
            await _lock_orgs(db, candidate_org_ids)
            stmt = stmt.outerjoin(Organization, User.org_id == Organization.id).where(
                or_(User.org_id.is_(None), Organization.is_active.is_(True))
            )
        affected_ids = list((await db.execute(stmt)).scalars().all())
        if affected_ids:
            await db.execute(
                update(User)
                .where(User.id.in_(affected_ids))
                .values(is_active=(body.action == "activate"))
                .execution_options(synchronize_session=False)
            )
            if body.action == "deactivate":
                # Fix: bulk deactivation revokes the targets' sessions too.
                await _revoke_refresh_tokens(db, affected_ids)
        count = len(affected_ids)
        message = f"{count} users {body.action}d"
    elif body.action in ("grant_mobile", "revoke_mobile"):
        # Approving a department's worth of people one modal at a time is the kind
        # of chore that ends in nobody being approved, so the grant is bulk-able.
        granting = body.action == "grant_mobile"
        affected_ids = list(
            (
                await db.execute(
                    select(User.id).where(User.id.in_(valid_ids), User.role != UserRole.superadmin)
                )
            )
            .scalars()
            .all()
        )
        if affected_ids:
            await db.execute(
                update(User)
                .where(User.id.in_(affected_ids))
                .values(mobile_access=granting)
                .execution_options(synchronize_session=False)
            )
            if not granting:
                await _revoke_refresh_tokens(db, affected_ids, client="mobile")
        count = len(affected_ids)
        message = f"Mobile access {'granted for' if granting else 'revoked for'} {count} users"
    elif body.action == "change_dept":
        if not body.dept_id:
            raise HTTPException(status_code=400, detail="dept_id required for change_dept")
        did = parse_uuid(body.dept_id)
        dept = await db.get(Department, did) if did else None
        if dept is None:
            raise HTTPException(status_code=400, detail="Invalid dept_id")
        users = (
            (await db.execute(select(User).where(User.id.in_(valid_ids), User.role != UserRole.superadmin)))
            .scalars()
            .all()
        )
        affected_ids = []
        for u in users:
            # Fix: only move users whose org owns the target department.
            if u.org_id == dept.org_id:
                u.dept_id = dept.id
                affected_ids.append(u.id)
        count = len(affected_ids)
        message = f"{count} users moved"
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

    # Which organisations this batch actually landed in. Every other log_audit
    # call in this module passes org_id; this one never did, because a bulk
    # action is the only one that can span tenants. Leaving it NULL meant the
    # org-admin activity feed — which filters on audit_logs.org_id — showed
    # nothing for the bulk deactivation that had just emptied the org.
    affected_org_ids: list[uuid.UUID] = []
    if affected_ids:
        affected_org_ids = sorted(
            set(
                (
                    await db.execute(
                        select(User.org_id).where(User.id.in_(affected_ids), User.org_id.is_not(None))
                    )
                )
                .scalars()
                .all()
            )
        )

    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action=f"bulk_{body.action}",
        target="users",
        # The ids actually affected, not every id asked for. activate skips
        # members of suspended orgs and change_dept skips other tenants, so the
        # requested set overstates what happened — and `count` beside it, which
        # is derived from the same list, then disagreed with it.
        details={
            "count": count,
            "user_ids": [str(i) for i in affected_ids],
            "org_ids": [str(o) for o in affected_org_ids],
        },
        # Attributed whenever the batch stayed inside one organisation, which is
        # what the portal's own org filter produces. A genuinely cross-tenant
        # batch stays NULL and carries the full list in details.
        org_id=affected_org_ids[0] if len(affected_org_ids) == 1 else None,
    )
    await db.commit()
    return {"message": message}


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    user = await _get_user_or_404(db, user_id)
    return await _serialize_user(db, user)


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    body: UpdateUser,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    user = await _get_user_or_404(db, user_id)
    update_data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "display_name" in update_data:
        name = sanitize_text(update_data["display_name"]).strip()
        user.display_name = name
        update_data["display_name"] = name
    if "dept_id" in update_data:
        did = _uuid_or_400(update_data["dept_id"], "Invalid dept_id")
        # Fix: the department must exist and belong to the user's org.
        dept = (
            await db.execute(select(Department).where(Department.id == did, Department.org_id == user.org_id))
        ).scalar_one_or_none()
        if dept is None:
            raise HTTPException(status_code=404, detail="Department not found in this organization")
        user.dept_id = did
    if "role" in update_data:
        user.role = _db_role(update_data["role"])
    if "mobile_access" in update_data:
        user.mobile_access = update_data["mobile_access"]
        if update_data["mobile_access"] is False:
            # Revoking the grant ends the phone's session immediately; the web
            # session is untouched (see _revoke_refresh_tokens).
            await _revoke_refresh_tokens(db, [user.id], client="mobile")
    if "is_active" in update_data:
        # The org suspension that deactivated this account is only ever enforced
        # through users.is_active; flipping it back here restores login to a
        # tenant that is supposed to be shut off. Checked before the assignment
        # so the refusal lands before any state changes.
        if update_data["is_active"] is True and user.org_id is not None:
            # Locked: an unlocked read races a concurrent suspension, whose
            # cascade has already passed over this row by the time we write it.
            org = (await _lock_orgs(db, [user.org_id])).get(user.org_id)
            if org is not None and not org.is_active:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot activate a user in a suspended organization",
                )
        user.is_active = update_data["is_active"]
        if update_data["is_active"] is False:
            # Fix: deactivation revokes the user's refresh sessions.
            await _revoke_refresh_tokens(db, [user.id])

    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="update_user",
        target=str(user.id),
        details={k: str(v) for k, v in update_data.items()},
        org_id=user.org_id,
    )
    await db.commit()
    return await _serialize_user(db, user)


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_superadmin),
):
    """Give an account a temporary password and end every session it had.

    Returns the temporary password for the superadmin to pass on. Under the target's row lock and in one
    transaction, the account is flagged must_change_password, its refresh tokens are revoked,
    users.sessions_valid_after is stamped so older access tokens stay refused, and its push subscriptions are
    deleted.
    """
    user = await _get_user_or_404(db, user_id)
    temporary_password = generate_password(12)
    new_hash = await hash_password(temporary_password)
    # The target's row lock, held to commit and taken BEFORE the refresh tokens are
    # revoked (batch 73 review). The revocation below reaches only the rows that
    # exist when it runs, so a sign-in that had verified the old password and was
    # about to insert its session came out of the reset holding a live one. Login
    # takes this same lock between its bcrypt round and its insert, so one of the
    # two now waits for the other: the login is refused for a password that no
    # longer matches, or its session exists by the time it is revoked here. Users
    # row first, then refresh_tokens — the order api/auth.py's lock_user_row
    # documents. After the bcrypt round, so the lock is held only for the writes.
    #
    # Today the users write below happens to take this lock anyway: the ORM
    # autoflushes it ahead of the revocation's UPDATE. That is an accident of
    # statement order and session configuration, not a guarantee — measured, with
    # this line removed and autoflush off around the revocation, a sign-in holding
    # the row kept its session — so it is taken here on purpose.
    if await lock_user_row(db, user.id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = new_hash
    # Fixes: the account must change it on next login, and old sessions die.
    # The flag is what makes the first half true: get_current_user refuses the
    # account everywhere but change-password from the moment this commits (batch 73).
    user.must_change_password = True
    await _revoke_refresh_tokens(db, [user.id])
    # The flag alone did not keep the second half: an access token is a JWT nothing
    # can revoke, so the target's pre-reset token was refused only until the owner
    # chose a new password, and then worked again for the rest of its life (batch 73
    # review). The epoch ends it for good — get_current_user refuses any token
    # issued before this. Stamped AFTER the revocation above so every refresh row it
    # revoked carries revoked_at <= the epoch: that is how change-password tells a
    # session this reset ended (re-issue it) from one a later logout ended (do not).
    user.sessions_valid_after = now_utc()
    # Push is the one channel the flag cannot gate, because it is the server that
    # sends: a browser subscribed before the reset kept receiving message previews
    # for an account its holder may no longer control. Deleted in the same
    # transaction as the reset; the user re-subscribes after changing the password.
    await db.execute(delete(PushSubscription).where(PushSubscription.user_id == user.id))
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="reset_password",
        target=str(user.id),
        details={"email": user.email},
        org_id=user.org_id,
    )
    await db.commit()
    return {"temporary_password": temporary_password}
