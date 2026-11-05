"""Org-admin console: stats, activity, user + department management, org settings.

Wire contract mirrors RxHivexx (docs/reference/api-auth-admin.md): serialized
docs use `_id` string keys, ISO-Z datetimes, and the org-admin action-naming
convention (user_created, dept_updated, ...). Deliberate fixes vs the old
build: DB-backed guard (deactivated/demoted admins lose access immediately),
session revocation on deactivate/reset-password, password policy on create,
per-org department name uniqueness on rename, and global slug uniqueness on
org rename. Every query is org-scoped — objects outside the caller's tenant
404 without revealing existence.

reset-password (batch 73): the must_change_password flag it sets is now enforced
— the account is refused everywhere but change-password until it picks its own —
and the reset deletes the target's push subscriptions and stamps
users.sessions_valid_after, which keeps access tokens issued before the reset
refused for good rather than only until the password changes. It locks the
target's users row before revoking anything, so a sign-in racing the reset cannot
keep a session the reset never saw (batch 73 review). An admin cannot reset
their OWN password here: it never asks for the current one, and it would revoke
the session making the request. Self-service is /api/auth/change-password.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import lock_user_row, wire_role
from app.core.deps import get_current_user
from app.core.errors import conflict_as_400
from app.core.rate_limit import password_limiter
from app.core.security import PasswordPolicyError, enforce_password_policy, hash_password
from app.db.models import (
    AuditLog,
    Conversation,
    Department,
    Organization,
    PushSubscription,
    RefreshToken,
    User,
    UserRole,
)
from app.db.session import get_db
from app.services import presence
from app.services.audit import log_audit, serialize_audit
from app.utils import generate_password, iso_z, now_utc, parse_uuid, sanitize_text, slugify

router = APIRouter(prefix="/api/org-admin", tags=["org-admin"])

_ROLE_MAP = {"admin": UserRole.org_admin, "member": UserRole.member}


async def _require_org_admin(user: User = Depends(get_current_user)) -> User:
    """Org admins ONLY — superadmins are rejected, preserving reference behavior.

    Unlike the old JWT-claim-only check, get_current_user re-reads the DB row
    (and its is_active flag), so demoted or deactivated admins lose access now.
    """
    if user.role != UserRole.org_admin or user.org_id is None:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _serialize_user(user: User, dept_name: str, status: str) -> dict:
    return {
        "_id": str(user.id),
        "org_id": str(user.org_id) if user.org_id else None,
        "dept_id": str(user.dept_id) if user.dept_id else None,
        "email": user.email,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "role": wire_role(user.role),
        "status": status,
        "last_seen": iso_z(user.last_seen_at),
        "about": user.about,
        "created_by": str(user.created_by) if user.created_by else None,
        "created_at": iso_z(user.created_at),
        "is_active": user.is_active,
        # Read-only here. Only a superadmin grants mobile access (OrgUpdateUser has
        # no such field), but an org admin fielding "the app won't let me in" needs
        # to be able to see whether the account was ever approved.
        "mobile_access": user.mobile_access,
        "dept_name": dept_name,
    }


def _serialize_department(dept: Department) -> dict:
    return {
        "_id": str(dept.id),
        "org_id": str(dept.org_id),
        "name": dept.name,
        "description": dept.description,
        "created_at": iso_z(dept.created_at),
    }


def _serialize_org(org: Organization) -> dict:
    return {
        "_id": str(org.id),
        "name": org.name,
        "slug": org.slug,
        "logo_url": org.logo_url,
        "created_by": None,  # not modelled in the Postgres schema; key kept for wire compat
        "created_at": iso_z(org.created_at),
        "is_active": org.is_active,
    }


async def _org_department_or_404(db: AsyncSession, admin: User, dept_id_raw: str) -> Department:
    """Resolve a department in the admin's organisation, or 404.

    Replaces the department-scoping helper. Delegation is gone: an org admin
    manages their whole organisation again, so the only question left is whether
    the department is theirs to touch at all, which means "is it in my org".
    """
    dept_uuid = parse_uuid(dept_id_raw)
    if dept_uuid is None:
        raise HTTPException(status_code=400, detail="Invalid department ID")
    dept = await _org_department(db, admin.org_id, dept_uuid)
    if dept is None:
        raise HTTPException(status_code=404, detail="Department not found in your organization")
    return dept


async def _load_org_user(db: AsyncSession, admin: User, user_id_raw: str) -> User:
    uid = parse_uuid(user_id_raw)
    if uid is None:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    target = (
        await db.execute(select(User).where(User.id == uid, User.org_id == admin.org_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    # Peer admins are off limits, applied at the LOAD helper rather than at each
    # route: every single-user mutation in this module (deactivate, reset
    # password, change department, change role) resolves its target through
    # here, so enforcing once covers them all and a new route inherits it.
    #
    # Without this, "admins cannot create admins" is decorative. An admin who
    # cannot MINT an admin account can simply TAKE one: reset-password returns
    # the new password in its own response body, so one call against a peer
    # hands over a working login for an account the actor does not own. Blocking
    # creation while leaving seizure open would protect nothing.
    #
    # Demotion is still reachable, because update_user resolves a peer through
    # this helper too — so it is blocked as well. That is the deliberate
    # trade: removing another admin is a superadmin action, which is where role
    # grants already live. An admin managing their own account is unaffected;
    # self-service lives in /api/auth, not here.
    if target.role is not UserRole.member and target.id != admin.id:
        raise HTTPException(
            status_code=403,
            detail="Only a superadmin can manage another admin's account",
        )
    return target


async def _org_department(db: AsyncSession, org_id: uuid.UUID, dept_id: uuid.UUID) -> Department | None:
    return (
        await db.execute(select(Department).where(Department.id == dept_id, Department.org_id == org_id))
    ).scalar_one_or_none()


async def _dept_name_map(db: AsyncSession, org_id: uuid.UUID) -> dict[uuid.UUID, str]:
    rows = (await db.execute(select(Department.id, Department.name).where(Department.org_id == org_id))).all()
    return {dept_id: name for dept_id, name in rows}


async def _single_dept_name(db: AsyncSession, org_id: uuid.UUID, dept_id: uuid.UUID | None) -> str:
    if dept_id is None:
        return "Unknown"
    name = (
        await db.execute(select(Department.name).where(Department.id == dept_id, Department.org_id == org_id))
    ).scalar_one_or_none()
    return name or "Unknown"


async def _revoke_refresh_tokens(db: AsyncSession, user_id: uuid.UUID) -> None:
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now_utc())
    )


# ---------------------------------------------------------------------------
# Stats + activity
# ---------------------------------------------------------------------------


@router.get("/stats")
async def org_stats(
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    org_id = admin.org_id
    total_users = (
        await db.execute(select(func.count()).select_from(User).where(User.org_id == org_id))
    ).scalar_one()
    # Was: read every active user id in the tenant out of Postgres, then issue one
    # Redis EXISTS per user — both O(tenant), on every dashboard load. The online
    # index answers it with one trim and one ZCARD.
    #
    # Slight semantic change, in the direction of the tile's own label: this counts
    # everyone from the org with a live socket, where the old version intersected
    # that with users.is_active. The two differ only for an account deactivated
    # while it still holds an open connection, which is a window of seconds and
    # which the old count arguably got wrong — they are, in fact, online.
    active_today = await presence.count_online_in_org(org_id)
    total_departments = (
        await db.execute(select(func.count()).select_from(Department).where(Department.org_id == org_id))
    ).scalar_one()
    total_conversations = (
        await db.execute(
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.org_id == org_id, Conversation.is_active.is_(True))
        )
    ).scalar_one()
    return {
        "total_users": total_users,
        "active_today": active_today,
        "total_departments": total_departments,
        "total_conversations": total_conversations,
    }


@router.get("/activity")
async def org_activity(
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    logs = (
        (
            await db.execute(
                select(AuditLog)
                .where(AuditLog.org_id == admin.org_id)
                .order_by(AuditLog.created_at.desc())
                .limit(10)
            )
        )
        .scalars()
        .all()
    )
    return [serialize_audit(log) for log in logs]


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


@router.get("/users")
async def list_users(
    dept_id: str = "",
    search: str = "",
    status: str = "",
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(User).where(User.org_id == admin.org_id)
    dept_uuid = parse_uuid(dept_id)
    if dept_uuid is not None:  # malformed dept_id is silently ignored (reference behavior)
        stmt = stmt.where(User.dept_id == dept_uuid)
    if search:
        pattern = _like_pattern(search)
        stmt = stmt.where(
            or_(
                User.display_name.ilike(pattern, escape="\\"),
                User.email.ilike(pattern, escape="\\"),
            )
        )
    if status == "active":
        stmt = stmt.where(User.is_active.is_(True))
    elif status == "inactive":
        stmt = stmt.where(User.is_active.is_(False))

    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (
        (await db.execute(stmt.order_by(User.created_at.desc()).offset((page - 1) * limit).limit(limit)))
        .scalars()
        .all()
    )
    dept_names = await _dept_name_map(db, admin.org_id)
    statuses = await presence.get_statuses([u.id for u in rows])
    data = [
        _serialize_user(u, dept_names.get(u.dept_id, "Unknown"), statuses.get(str(u.id), "offline"))
        for u in rows
    ]
    return {"data": data, "total": total, "page": page, "limit": limit}


class OrgCreateUser(BaseModel):
    dept_id: str
    email: EmailStr
    display_name: str = Field(min_length=2, max_length=100)
    password: str
    role: str = "member"


@router.post("/users")
async def create_user(
    body: OrgCreateUser,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    dept = await _org_department_or_404(db, admin, body.dept_id)

    email = body.email.strip().lower()
    taken = (
        # CITEXT equality is case-insensitive and uses the unique index;
        # lower(email) does not. See app/api/auth.py.
        await db.execute(select(User.id).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if taken is not None:
        raise HTTPException(status_code=400, detail="Email already in use")

    # An admin creates members, never other admins.
    #
    # The actor picks the new account's password, so minting an admin is the same
    # thing as handing yourself a second admin login — one that survives your own
    # account being deactivated or demoted. Granting the admin role stays a
    # superadmin action (admin.py). _load_org_user closes the other half of this:
    # you cannot take over an admin that already exists either.
    if body.role != "member":
        raise HTTPException(
            status_code=403,
            detail="Admins can create members only. Ask a superadmin to grant admin access.",
        )
    try:
        enforce_password_policy(body.password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user = User(
        org_id=admin.org_id,
        dept_id=dept.id,
        email=email,
        display_name=sanitize_text(body.display_name).strip(),
        password_hash=await hash_password(body.password),
        role=UserRole.member,
        is_active=True,
        created_by=admin.id,
        created_at=now_utc(),
    )
    db.add(user)
    async with conflict_as_400(db, "Email already in use"):
        await db.flush()
    await log_audit(
        db,
        actor_id=admin.id,
        actor_type="org_admin",
        action="user_created",
        target=email,
        details={"user_id": str(user.id), "role": "member"},
        org_id=admin.org_id,
    )
    await db.commit()
    return _serialize_user(user, dept.name, "offline")


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    target = await _load_org_user(db, admin, user_id)
    dept_name = await _single_dept_name(db, admin.org_id, target.dept_id)
    statuses = await presence.get_statuses([target.id])
    return _serialize_user(target, dept_name, statuses.get(str(target.id), "offline"))


class OrgUpdateUser(BaseModel):
    # Bounded at the User.display_name column width. The create schema above caps
    # this and the update schema did not, so a rename past the column length was
    # a 500 on commit where the identical value on create was a clean 422.
    display_name: str | None = Field(default=None, max_length=200)
    dept_id: str | None = None
    role: str | None = None
    is_active: bool | None = None


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    body: OrgUpdateUser,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    target = await _load_org_user(db, admin, user_id)

    changes: dict = {}
    if body.display_name is not None:
        cleaned = sanitize_text(body.display_name).strip()
        if cleaned:
            target.display_name = cleaned
            changes["display_name"] = cleaned
    if body.dept_id is not None:
        dept = await _org_department_or_404(db, admin, body.dept_id)
        target.dept_id = dept.id
        changes["dept_id"] = str(dept.id)
    if body.role is not None and body.role in _ROLE_MAP:
        # Unknown role values are silently ignored (reference behavior).
        new_role = _ROLE_MAP[body.role]
        # Promotion is creation by another name — the same privilege grant POST
        # /users refuses, routed through a member who already exists. Blocking
        # only the create would have been a one-line detour.
        #
        # The target here is always a member: _load_org_user refuses a peer
        # admin outright. So this is specifically "member -> admin", and the
        # is-it-a-change half of the condition is what keeps a no-op save of an
        # unchanged role from 403ing.
        if new_role is UserRole.org_admin and target.role is not UserRole.org_admin:
            raise HTTPException(
                status_code=403,
                detail="Admins cannot grant admin access. Ask a superadmin to promote this user.",
            )
        target.role = new_role
        changes["role"] = body.role
    if body.is_active is not None:
        target.is_active = body.is_active
        changes["is_active"] = body.is_active
        if not body.is_active:
            await _revoke_refresh_tokens(db, target.id)

    if changes:
        await log_audit(
            db,
            actor_id=admin.id,
            actor_type="org_admin",
            action="user_updated",
            target=target.email,
            details={"user_id": str(target.id), **changes},
            org_id=admin.org_id,
        )
    await db.commit()
    dept_name = await _single_dept_name(db, admin.org_id, target.dept_id)
    statuses = await presence.get_statuses([target.id])
    return _serialize_user(target, dept_name, statuses.get(str(target.id), "offline"))


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
    _rl: None = Depends(password_limiter),
):
    """Give a member of the admin's organization a temporary password and end every session it had.

    The same writes as the superadmin route in admin.py, scoped to the caller's org and rate-limited. An admin
    targeting their own account is refused with 400 before anything is written. Returns the temporary
    password.
    """
    target = await _load_org_user(db, admin, user_id)
    # _load_org_user lets an admin resolve their own row, deliberately, so they can
    # edit their own profile. A reset is the exception (batch 73): it would revoke
    # every session the admin has, including the one making this request, and lock
    # their own account behind a temporary password shown on the screen they are
    # looking at. Self-service lives in /api/auth/change-password, which checks the
    # current password first. Refused before anything is written.
    if target.id == admin.id:
        raise HTTPException(status_code=400, detail="Use Change Password to change your own password.")
    temp = generate_password()
    new_hash = await hash_password(temp)
    # The target's row lock, held to commit and taken before the refresh tokens are
    # revoked, so a sign-in racing this reset either sees the new password and is
    # refused or has its session revoked below (batch 73 review); see the superadmin
    # route in admin.py.
    locked = await lock_user_row(db, target.id)
    # And the scope checks again, on the row as it is under that lock (CodeRabbit,
    # review of 70495a4). _load_org_user judged the row as read before the bcrypt
    # round; a superadmin moving the target to another org, or promoting it to
    # admin, in that window would otherwise have this route write — and return to
    # the caller — a temporary password for an account outside their reach. Same
    # answers as _load_org_user's own.
    if locked is None or locked.org_id != admin.org_id:
        raise HTTPException(status_code=404, detail="User not found")
    if locked.role is not UserRole.member:
        raise HTTPException(status_code=403, detail="Only a superadmin can manage another admin's account")
    target.password_hash = new_hash
    # Enforced by get_current_user from the moment this commits: the account is
    # refused everywhere but change-password until it chooses its own password.
    target.must_change_password = True
    await _revoke_refresh_tokens(db, target.id)
    # And the session epoch, so the target's pre-reset access token stays refused
    # after they change the password rather than only until then (batch 73 review).
    # After the revocation, for the ordering change-password depends on; see the
    # superadmin route in admin.py.
    target.sessions_valid_after = now_utc()
    # Push is the one channel the flag cannot gate, because the server is the one
    # sending. Same transaction as the reset; see the superadmin route in admin.py.
    await db.execute(delete(PushSubscription).where(PushSubscription.user_id == target.id))
    await log_audit(
        db,
        actor_id=admin.id,
        actor_type="org_admin",
        action="password_reset",
        target=target.email,
        details={"email": target.email, "user_id": str(target.id)},
        org_id=admin.org_id,
    )
    await db.commit()
    return {"temporary_password": temp}


# ---------------------------------------------------------------------------
# Departments
# ---------------------------------------------------------------------------


@router.get("/departments")
async def list_departments(
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Department, func.count(User.id))
        .outerjoin(User, and_(User.dept_id == Department.id, User.is_active.is_(True)))
        .where(Department.org_id == admin.org_id)
        .group_by(Department.id)
        .order_by(Department.name.asc())
    )
    rows = (await db.execute(stmt)).all()
    return [{**_serialize_department(dept), "member_count": count} for dept, count in rows]


class OrgCreateDept(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    description: str | None = None


@router.post("/departments")
async def create_department(
    body: OrgCreateDept,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    name = sanitize_text(body.name).strip()
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Department name must be at least 2 characters")
    dup = (
        await db.execute(
            select(Department.id).where(Department.org_id == admin.org_id, Department.name == name).limit(1)
        )
    ).scalar_one_or_none()
    if dup is not None:
        raise HTTPException(status_code=400, detail="Department already exists")

    description = sanitize_text(body.description) if body.description is not None else None
    dept = Department(org_id=admin.org_id, name=name, description=description, created_at=now_utc())
    db.add(dept)
    async with conflict_as_400(db, "Department already exists"):
        await db.flush()
    await log_audit(
        db,
        actor_id=admin.id,
        actor_type="org_admin",
        action="dept_created",
        target=name,
        details={"dept_id": str(dept.id)},
        org_id=admin.org_id,
    )
    await db.commit()
    return _serialize_department(dept)


class OrgUpdateDept(BaseModel):
    # Department.name is String(200) — see OrgUpdateUser.display_name.
    name: str | None = Field(default=None, max_length=200)
    description: str | None = None


@router.put("/departments/{dept_id}")
async def update_department(
    dept_id: str,
    body: OrgUpdateDept,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    dept = await _org_department_or_404(db, admin, dept_id)

    changes: dict = {}
    if body.name is not None:
        cleaned = sanitize_text(body.name).strip()
        if cleaned and cleaned != dept.name:
            dup = (
                await db.execute(
                    select(Department.id)
                    .where(
                        Department.org_id == admin.org_id,
                        Department.name == cleaned,
                        Department.id != dept.id,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if dup is not None:
                raise HTTPException(status_code=400, detail="Department already exists")
            dept.name = cleaned
            changes["name"] = cleaned
    if body.description is not None:
        dept.description = sanitize_text(body.description)
        changes["description"] = dept.description

    if changes:
        await log_audit(
            db,
            actor_id=admin.id,
            actor_type="org_admin",
            action="dept_updated",
            target=dept.name,
            details={"dept_id": str(dept.id), **changes},
            org_id=admin.org_id,
        )
    # Same constraint the pre-check above tests, and it raises from commit()
    # because nothing is flushed explicitly on this path.
    async with conflict_as_400(db, "Department already exists"):
        await db.commit()
    return _serialize_department(dept)


@router.delete("/departments/{dept_id}")
async def delete_department(
    dept_id: str,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    dept = await _org_department_or_404(db, admin, dept_id)

    active_users = (
        await db.execute(
            select(func.count()).select_from(User).where(User.dept_id == dept.id, User.is_active.is_(True))
        )
    ).scalar_one()
    if active_users:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete: {active_users} active users in this department",
        )

    await db.delete(dept)
    await log_audit(
        db,
        actor_id=admin.id,
        actor_type="org_admin",
        action="dept_deleted",
        target=dept.name,
        details={"dept_id": str(dept.id)},
        org_id=admin.org_id,
    )
    await db.commit()
    return {"message": "Department deleted"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@router.get("/settings")
async def get_org_settings(
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, admin.org_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")
    return _serialize_org(org)


class OrgSettingsUpdate(BaseModel):
    # Organization.name is String(200) — see OrgUpdateUser.display_name.
    name: str | None = Field(default=None, max_length=200)
    logo_url: str | None = None


@router.put("/settings")
async def update_org_settings(
    body: OrgSettingsUpdate,
    admin: User = Depends(_require_org_admin),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, admin.org_id)
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    changes: dict = {}
    if body.name is not None:
        cleaned = sanitize_text(body.name).strip()
        new_slug = slugify(cleaned)
        if cleaned and new_slug:  # names that slugify to nothing are ignored, not applied
            if new_slug != org.slug:
                dup = (
                    await db.execute(
                        select(Organization.id)
                        .where(Organization.slug == new_slug, Organization.id != org.id)
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if dup is not None:
                    raise HTTPException(status_code=400, detail="Organization name already exists")
            org.name = cleaned
            org.slug = new_slug
            changes["name"] = cleaned
    if body.logo_url is not None:
        org.logo_url = body.logo_url
        changes["logo_url"] = body.logo_url

    if changes:
        await log_audit(
            db,
            actor_id=admin.id,
            actor_type="org_admin",
            action="org_settings_updated",
            target=org.name,
            details=changes,
            org_id=org.id,
        )
    # Same constraint the pre-check above tests, and it raises from commit()
    # because nothing is flushed explicitly on this path.
    async with conflict_as_400(db, "Organization name already exists"):
        await db.commit()
    return _serialize_org(org)
